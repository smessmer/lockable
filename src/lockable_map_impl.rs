use anyhow::{anyhow, Result};
use futures::stream::{FuturesUnordered, Stream};
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::OwnedMutexGuard;

use super::guard::Guard;
use super::hooks::{Hooks, NoopHooks};
use super::limit::{AsyncLimit, SyncLimit};
use super::map_like::{ArcMutexMapLike, EntryValue};

pub trait FromInto<V> {
    fn fi_from(v: V) -> Self;
    fn fi_into(self) -> V;
}

impl<V> FromInto<V> for V {
    fn fi_from(v: V) -> V {
        v
    }

    fn fi_into(self) -> V {
        self
    }
}

pub struct LockableMapImpl<M, V, H>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    H: Hooks<M::V>,
{
    // We always use std::sync::Mutex for protecting the whole map since its guards
    // never have to be kept across await boundaries, and std::sync::Mutex is faster
    // than tokio::sync::Mutex. But the inner per-key locks use tokio::sync::Mutex
    // because they need to be kept across await boundaries.
    //
    // We never hand the inner Arc around a map entry out of the encapsulation of this class,
    // except through non-cloneable Guard objects encapsulating those Arcs.
    // This allows us to reason about which threads can or cannot increase the refcounts.
    //
    // TODO The following invariant needs to be moved to LockableLruCache
    // - The timestamps in EntryValue will follow the same order as the LRU order of the map,
    //   with an exception for currently locked entries that may be temporarily out of order
    //   while the entry is locked.
    //
    // cache_entries is always Some unless we're currently destructing the object
    cache_entries: Option<std::sync::Mutex<M>>,

    /// Counts the number of currently locked entries.
    num_locked: AtomicUsize,

    hooks: H,

    _v: PhantomData<V>,
}

impl<M, V> LockableMapImpl<M, V, NoopHooks>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
{
    #[inline]
    pub fn new() -> Self {
        Self::new_with_hooks(NoopHooks)
    }
}

impl<M, V> Default for LockableMapImpl<M, V, NoopHooks>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M, V, H> LockableMapImpl<M, V, H>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    H: Hooks<M::V>,
{
    #[inline]
    pub fn new_with_hooks(hooks: H) -> Self {
        Self {
            cache_entries: Some(std::sync::Mutex::new(M::new())),
            num_locked: 0.into(),
            hooks,
            _v: PhantomData,
        }
    }

    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self._cache_entries().len()
    }

    #[inline]
    pub fn num_locked(&self) -> usize {
        self.num_locked.load(Ordering::SeqCst)
    }

    // TODO We can probably remove num_unlocked and the atomic counter for it, because we now use a different mechanism
    // to enforce a max_entries limit.
    #[inline]
    pub fn num_unlocked(&self) -> usize {
        let entries = self._cache_entries();
        self._num_unlocked(&entries)
    }

    fn _num_unlocked(&self, entries: &std::sync::MutexGuard<M>) -> usize {
        // We keep `entries` as a lock here so that no other task can come, call _unlock() and decrease entries.len().
        // There is no other way to decrease entries.len(). Other tasks can come in and increase self.num_locked()
        // while we're in here, but they can never increase it to larger than entries.len().
        assert!(
            entries.len() >= self.num_locked(),
            "LockableMapImpl::num_unlocked: {} < {}",
            entries.len(),
            self.num_locked()
        );
        entries.len() - self.num_locked()
    }

    fn _cache_entries(&self) -> std::sync::MutexGuard<'_, M> {
        self.cache_entries
            .as_ref()
            .expect("Object is currently being destructed")
            .lock()
            .expect("The global mutex protecting the LockableCache is poisoned. This shouldn't happen since there shouldn't be any user code running while this lock is held so no thread should ever panic with it")
    }

    async fn _load_or_insert_mutex_for_key_async<S, E, F, OnEvictFn>(
        this: &S,
        key: &M::K,
        limit: AsyncLimit<M, V, H, S, E, F, OnEvictFn>,
    ) -> Result<Arc<tokio::sync::Mutex<EntryValue<M::V>>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Guard<M, V, H, S>>) -> F,
    {
        // Note: this logic is duplicated in _load_or_insert_mutex_for_key_sync without the .await calls
        let mut cache_entries = match limit {
            AsyncLimit::NoLimit { .. } => {
                // do nothing
                this.borrow()._cache_entries()
            }
            AsyncLimit::SoftLimit {
                max_entries,
                on_evict,
            } => {
                // free up space for the new entry if necessary
                loop {
                    let locked = {
                        let mut cache_entries = this.borrow()._cache_entries();
                        let num_overlimit_entries =
                            cache_entries.len().saturating_sub(max_entries.get() - 1);
                        if num_overlimit_entries == 0 {
                            // There is enough space, no need to free up space
                            break cache_entries;
                        }
                        // There is not enough space, free up some.
                        let locked = Self::_lock_up_to_n_first_unlocked_entries(
                            this,
                            &mut cache_entries,
                            num_overlimit_entries,
                        );

                        // If we couldn't lock any entries to free their space up, then
                        // all cache entries are currently locked. If we just waited
                        // until we lock one, there would be a potential dead lock
                        // if multiple threads hold locks and try to get more locks.
                        // Let's avoid that deadlock and allow the current locking
                        // request, even though it goes above the limit.
                        // This is why we call [AsyncLimit::SoftLimit] a "soft" limit.
                        if locked.len() == 0 {
                            // TODO Test that this works, i.e. that the map still correctly works when it's full and doesn't deadlock (and same for the _load_or_insert_mutex_for_key_sync version)
                            break cache_entries;
                        }

                        // We now have some entries locked that may free up enough space.
                        // Let's evict them. We have to free up the cache_entries lock for that
                        // so that the on_evict user code can call back into Self::_unlock()
                        // for those entries. That means other user code may also run and cause
                        // race conditions. Because of that, once on_evict returns, we'll check
                        // take the lock again in the next loop iteration and check again if we now
                        // have enough space
                        std::mem::drop(cache_entries);
                        locked
                    };
                    on_evict(locked).await?;
                }
            }
        };
        let entry = cache_entries.get_or_insert_none(key);
        Ok(Arc::clone(entry))
    }

    fn _load_or_insert_mutex_for_key_sync<S, E, OnEvictFn>(
        this: &S,
        key: &M::K,
        limit: SyncLimit<M, V, H, S, E, OnEvictFn>,
    ) -> Result<Arc<tokio::sync::Mutex<EntryValue<M::V>>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: Fn(Vec<Guard<M, V, H, S>>) -> Result<(), E>,
    {
        // Note: this logic is duplicated in _load_or_insert_mutex_for_key_sync with some .await calls
        let mut cache_entries = match limit {
            SyncLimit::NoLimit { .. } => {
                // do nothing
                this.borrow()._cache_entries()
            }
            SyncLimit::SoftLimit {
                max_entries,
                on_evict,
            } => {
                // free up space for the new entry if necessary
                loop {
                    let locked = {
                        let mut cache_entries = this.borrow()._cache_entries();
                        let num_overlimit_entries =
                            cache_entries.len().saturating_sub(max_entries.get() - 1);
                        if num_overlimit_entries == 0 {
                            // There is enough space, no need to free up space
                            break cache_entries;
                        }
                        // There is not enough space, free up some.
                        let locked = Self::_lock_up_to_n_first_unlocked_entries(
                            this,
                            &mut cache_entries,
                            num_overlimit_entries,
                        );

                        // If we couldn't lock any entries to free their space up, then
                        // all cache entries are currently locked. If we just waited
                        // until we lock one, there would be a potential dead lock
                        // if multiple threads hold locks and try to get more locks.
                        // Let's avoid that deadlock and allow the current locking
                        // request, even though it goes above the limit.
                        // This is why we call [AsyncLimit::SoftLimit] a "soft" limit.
                        if locked.len() == 0 {
                            break cache_entries;
                        }

                        // We now have some entries locked that may free up enough space.
                        // Let's evict them. We have to free up the cache_entries lock for that
                        // so that the on_evict user code can call back into Self::_unlock()
                        // for those entries. That means other user code may also run and cause
                        // race conditions. Because of that, once on_evict returns, we'll check
                        // take the lock again in the next loop iteration and check again if we now
                        // have enough space
                        std::mem::drop(cache_entries);
                        locked
                    };
                    on_evict(locked)?;
                }
            }
        };
        let entry = cache_entries.get_or_insert_none(key);
        Ok(Arc::clone(entry))
    }

    fn _make_guard<S: Borrow<Self>>(
        this: S,
        key: M::K,
        guard: OwnedMutexGuard<EntryValue<M::V>>,
    ) -> Guard<M, V, H, S> {
        this.borrow().num_locked.fetch_add(1, Ordering::SeqCst);
        Guard::new(this, key, guard)
    }

    #[inline]
    pub fn blocking_lock<S, E, OnEvictFn>(
        this: S,
        key: M::K,
        limit: SyncLimit<M, V, H, S, E, OnEvictFn>,
    ) -> Result<Guard<M, V, H, S>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: Fn(Vec<Guard<M, V, H, S>>) -> Result<(), E>,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_sync(&this, &key, limit)?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the thread until the mutex for this key is acquired.

        // TODO Switch to tokio::sync::Mutex::blocking_lock_owned if it gets implemented, see https://github.com/tokio-rs/tokio/issues/5109
        let guard = match tokio::runtime::Handle::try_current() {
            Ok(runtime) => runtime.block_on(mutex.lock_owned()),
            Err(_) => tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(mutex.lock_owned()),
        };

        Ok(Self::_make_guard(this, key, guard))
    }

    #[inline]
    pub async fn async_lock<S, E, F, OnEvictFn>(
        this: S,
        key: M::K,
        limit: AsyncLimit<M, V, H, S, E, F, OnEvictFn>,
    ) -> Result<Guard<M, V, H, S>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Guard<M, V, H, S>>) -> F,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_async(&this, &key, limit).await?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the task until the mutex for this key is acquired.

        let guard = mutex.lock_owned().await;
        Ok(Self::_make_guard(this, key, guard))
    }

    #[inline]
    pub fn try_lock<S, E, OnEvictFn>(
        this: S,
        key: M::K,
        limit: SyncLimit<M, V, H, S, E, OnEvictFn>,
    ) -> Result<Option<Guard<M, V, H, S>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: Fn(Vec<Guard<M, V, H, S>>) -> Result<(), E>,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_sync(&this, &key, limit)?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        match mutex.try_lock_owned() {
            Ok(guard) => Ok(Some(Self::_make_guard(this, key, guard))),
            Err(_) => Ok(None),
        }
    }

    #[inline]
    pub async fn try_lock_async<S, E, F, OnEvictFn>(
        this: S,
        key: M::K,
        limit: AsyncLimit<M, V, H, S, E, F, OnEvictFn>,
    ) -> Result<Option<Guard<M, V, H, S>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Guard<M, V, H, S>>) -> F,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_async(&this, &key, limit).await?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        match mutex.try_lock_owned() {
            Ok(guard) => Ok(Some(Self::_make_guard(this, key, guard))),
            Err(_) => Ok(None),
        }
    }

    // TODO Test
    pub fn lock_all_unlocked<S: Borrow<Self> + Clone>(
        this: S,
    ) -> impl Iterator<Item = Guard<M, V, H, S>> {
        let cache_entries = this.borrow()._cache_entries();
        cache_entries
            .iter()
            .filter_map(|(key, mutex)| match Arc::clone(mutex).try_lock_owned() {
                Ok(guard) => Some(Self::_make_guard(this.clone(), key.clone(), guard)),
                Err(_) => None,
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    // TODO Test
    pub async fn lock_all<S: Borrow<Self> + Clone>(
        this: S,
    ) -> impl Stream<Item = Guard<M, V, H, S>> {
        let cache_entries = this.borrow()._cache_entries();
        let cache_entries: FuturesUnordered<_> = cache_entries
            .iter()
            .map(|(key, mutex)| {
                let this = this.clone();
                let key = key.clone();
                let mutex = Arc::clone(mutex);
                async move {
                    let guard = mutex.lock_owned().await;
                    Self::_make_guard(this, key, guard)
                }
            })
            .collect();
        cache_entries
    }

    pub(super) fn _unlock(&self, key: &M::K, mut guard: OwnedMutexGuard<EntryValue<M::V>>) {
        let mut cache_entries = self._cache_entries();
        self.hooks.on_unlock(guard.value.as_mut());
        let entry_carries_a_value = guard.value.is_some();
        std::mem::drop(guard);

        let prev_value = self.num_locked.fetch_sub(1, Ordering::SeqCst);
        assert!(
            prev_value > 0,
            "Somehow we returned a guard that was created without incrementing num_locked."
        );

        // Now the guard is dropped and the lock for this key is unlocked.
        // If there are any other Self::blocking_lock/async_lock/try_lock()
        // calls for this key already running and/ waiting for the mutex,
        // they will be unblocked now and their guard will be created.

        // If the guard we dropped carried a value, keep the entry in the map.
        // But it doesn't carry a value, clean up since the entry semantically
        // doesn't exist in the map and was only created to have a place to put
        // the mutex.
        if !entry_carries_a_value {
            Self::_delete_if_unlocked_and_nobody_waiting_for_lock(&mut cache_entries, key);
        }
    }

    fn _delete_if_unlocked_and_nobody_waiting_for_lock<'a>(
        cache_entries: &mut std::sync::MutexGuard<'_, M>,
        key: &M::K,
    ) {
        let mutex: &Arc<tokio::sync::Mutex<EntryValue<M::V>>> = cache_entries
            .get(key)
            .expect("This entry must exist or the guard passed in as a parameter shouldn't exist");
        // If there are any other locks or any other tasks currently waiting in Self::blocking_lock/async_lock/try_lock,
        // then Arc::strong_count() will be larger than one.
        // But since we still have the global mutex on cache_entries, currently no
        // thread can newly call Self::blocking_lock/async_lock/try_lock() and create a
        // new clone of our Arc. Similarly, no other thread can enter Self::_unlock()
        // and reduce the strong_count of the Arc by dropping the guard. This means that if
        // Arc::strong_count() == 1, we know that there is no other thread with access
        // that could modify strong_count. We can clean up without race conditions.
        if Arc::strong_count(mutex) == 1 {
            let remove_result = cache_entries.remove(key);
            assert!(
                remove_result.is_some(),
                "We just got this entry above from the hash map, it cannot have vanished since then"
            );
        } else {
            // Another tasks was currently waiting in a Self::blocking_lock/async_lock/try_lock call
            // and will now get the lock. We shouldn't free the entry.
            // All such waiting tasks will have to eventually call back into _delete_if_unlocked_and_nobody_waiting_for_lock
            // to clean up if the entry they locked was None.
        }
    }

    pub fn into_entries_unordered(mut self) -> impl Iterator<Item = (M::K, M::V)> {
        let entries: M = self
            .cache_entries
            .take()
            .expect("Object is already being destructed")
            .into_inner()
            .expect("Lock poisoned");

        // We now have exclusive access to the LockableMapImpl object. Rust lifetime rules ensure that no other thread or task can have any
        // Guard for an entry since both owned and non-owned guards are bound to the lifetime of the LockableMapImpl (owned guards
        // indirectly through the Arc but if user code calls this function, it means they had to call Arc::try_unwrap or something similar
        // which ensures that there are no other threads with access to it.

        entries
            .into_iter()
            .filter_map(|(key, value)| {
                let value = Arc::try_unwrap(value)
                    .map_err(|_| anyhow!("We're the only one with access, there shouldn't be any other threads or tasks that have a copy of this Arc."))
                    .unwrap();
                let value = value.into_inner();

                // Ignore None entries since they don't actually exist in the map and were only created so we have a place to put the mutex.
                value.value.map(|value| (key, value))
            })
    }

    // Caveat: Locked keys are listed even if they don't carry a value
    #[inline]
    pub fn keys(&self) -> Vec<M::K> {
        let cache_entries = self._cache_entries();
        cache_entries
            .iter()
            .map(|(key, _value)| key)
            .cloned()
            .collect()
    }

    /// If there are more than `num_remaining_unlocked` unlocked keys,
    /// lock all except the `num_remaining_unlocked` last ones (in iteration order).
    /// This can, for example, be used to enforce a capacity limit on
    /// the number of unlocked entries in the cache by locking and then
    /// deleting overlimit entries.
    /// TODO Test
    /// TODO This can probably be removed now since we have a new mechanism for enforcing a limit
    // pub fn lock_all_unlocked_except_n_first<S: Borrow<Self> + Clone>(
    //     this: S,
    //     num_remaining_unlocked: usize,
    // ) -> impl Iterator<Item = Guard<M, V, H, S>> {
    //     let this_borrow = this.borrow();
    //     let cache_entries = this_borrow._cache_entries();
    //     let num_unlocked = this_borrow._num_unlocked(&cache_entries);
    //     let num_overlimit = num_unlocked.saturating_sub(num_remaining_unlocked);
    //     cache_entries
    //         .iter()
    //         .filter_map(
    //             |(key, mutex)| match Arc::clone(mutex).try_lock() {
    //                 Ok(guard) => Some(Self::_make_guard(this.clone(), key.clone(), guard)),
    //                 Err(_) => None,
    //             },
    //         )
    //         .take(num_overlimit)
    //         .collect::<Vec<_>>()
    //         .into_iter()
    // }

    fn _lock_up_to_n_first_unlocked_entries<'a, S: Borrow<Self> + Clone>(
        this: &S,
        // TODO Without explicit 'a possible?
        entries: &mut std::sync::MutexGuard<'a, M>,
        num_entries: usize,
    ) -> Vec<Guard<M, V, H, S>> {
        let mut result = Vec::new();
        let mut to_delete = Vec::new();
        for (key, mutex) in entries.iter() {
            match Arc::clone(mutex).try_lock_owned() {
                Ok(guard) => {
                    if guard.value.is_some() {
                        result.push(Self::_make_guard(this.clone(), key.clone(), guard))
                    } else {
                        // This can happen if the entry was deleted while we waited for the lock.
                        // Because we have a Arc::clone of the mutex here, the _unlock() of that
                        // deletion operation didn't actually delete the entry and we need
                        // to give a chance for deletion now. We can't immediately delete it
                        // because that requires a &mut borrow of entries and we currently have
                        // a & borrow. But we can remember those keys and delete them further down.
                        to_delete.push(key.clone());
                    }
                }
                Err(_) => {
                    // A failed try_lock means we currently have another lock and we can rely on that one
                    // calling _unlock and potentially deleting an item if it is None. So we don't need
                    // to create a guard object here. Because we have a lock on `entries`, there are no
                    // race conditions with that _unlock.
                }
            }
            if result.len() >= num_entries {
                break;
            }
        }
        // Great, we either locked `num_entries` entries, or ran out of entries. Now let's delete the None
        // entries we found and we can return our result. We still have a lock on `entries`, so we know
        // that no new tasks can have entered Self::blocking_lock/async_lock/try_lock and tried to lock any
        // of those keys.
        for key in to_delete {
            Self::_delete_if_unlocked_and_nobody_waiting_for_lock(entries, &key);
        }
        result
    }
}

impl<M, V, H> Debug for LockableMapImpl<M, V, H>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    H: Hooks<M::V>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LockableCache").finish()
    }
}

impl<M, V, H> Drop for LockableMapImpl<M, V, H>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    H: Hooks<M::V>,
{
    fn drop(&mut self) {
        let num_locked = self.num_locked.load(Ordering::SeqCst);
        if 0 != num_locked {
            if std::thread::panicking() {
                // We're already panicking, double panic wouldn't show a good error message anyways. Let's just log instead.
                // A common scenario for this to happen is a failing test case.
                log::error!("Miscalculation in num_locked: {}", num_locked);
                eprintln!("Miscalculation in num_locked: {}", num_locked);
            } else {
                panic!("Miscalculation in num_locked: {}", num_locked);
            }
        }
    }
}
