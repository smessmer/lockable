use futures::stream::{FuturesUnordered, Stream, StreamExt};
use itertools::Itertools;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::OwnedMutexGuard;

use super::guard::Guard;
use super::hooks::{Hooks, NoopHooks};
use super::limit::{AsyncLimit, SyncLimit};
use super::map_like::{ArcMutexMapLike, EntryValue, GetOrInsertNoneResult};

pub trait FromInto<V, H: Hooks<Self>>: Sized {
    fn fi_from(v: V, hooks: &H) -> Self;
    fn fi_into(self) -> V;
}

impl<V> FromInto<V, NoopHooks> for V {
    fn fi_from(v: V, _hooks: &NoopHooks) -> V {
        v
    }

    fn fi_into(self) -> V {
        self
    }
}

pub struct LockableMapImpl<M, V, H>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V, H>,
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
    // entries is always Some unless we're currently destructing the object
    entries: Option<std::sync::Mutex<M>>,

    hooks: H,

    _v: PhantomData<V>,
}

impl<M, V> LockableMapImpl<M, V, NoopHooks>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V, NoopHooks>,
{
    #[inline]
    pub fn new() -> Self {
        Self::new_with_hooks(NoopHooks)
    }
}

impl<M, V> Default for LockableMapImpl<M, V, NoopHooks>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V, NoopHooks>,
{
    fn default() -> Self {
        Self::new()
    }
}

enum LoadOrInsertMutexResult<V> {
    Existing {
        mutex: Arc<tokio::sync::Mutex<V>>,
    },
    Inserted {
        guard: tokio::sync::OwnedMutexGuard<V>,
    },
}

impl<M, V, H> LockableMapImpl<M, V, H>
where
    M: ArcMutexMapLike,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V, H>,
    H: Hooks<M::V>,
{
    #[inline]
    pub fn new_with_hooks(hooks: H) -> Self {
        Self {
            entries: Some(std::sync::Mutex::new(M::new())),
            hooks,
            _v: PhantomData,
        }
    }

    #[inline]
    pub fn hooks(&self) -> &H {
        &self.hooks
    }

    #[cfg(test)]
    #[inline]
    pub fn hooks_mut(&mut self) -> &mut H {
        &mut self.hooks
    }

    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self._entries().len()
    }

    fn _entries(&self) -> std::sync::MutexGuard<'_, M> {
        self.entries
            .as_ref()
            .expect("Object is currently being destructed")
            .lock()
            .expect("The global mutex protecting the LockableCache is poisoned. This shouldn't happen since there shouldn't be any user code running while this lock is held so no thread should ever panic with it")
    }

    async fn _load_or_insert_mutex_for_key_async<S, E, F, OnEvictFn>(
        this: &S,
        key: &M::K,
        limit: AsyncLimit<M, V, H, S, E, F, OnEvictFn>,
    ) -> Result<LoadOrInsertMutexResult<EntryValue<M::V>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<M, V, H, S>>) -> F,
    {
        // Note: this logic is duplicated in _load_or_insert_mutex_for_key_sync without the .await calls
        let mut entries = match limit {
            AsyncLimit::NoLimit { .. } => {
                // do nothing
                this.borrow()._entries()
            }
            AsyncLimit::SoftLimit {
                max_entries,
                mut on_evict,
            } => {
                // free up space for the new entry if necessary
                loop {
                    let locked = {
                        let mut entries = this.borrow()._entries();
                        let num_overlimit_entries =
                            entries.len().saturating_sub(max_entries.get() - 1);
                        if num_overlimit_entries == 0 {
                            // There is enough space, no need to free up space
                            break entries;
                        }
                        // There is not enough space, free up some.
                        let locked = Self::_lock_up_to_n_first_unlocked_entries(
                            this,
                            &mut entries,
                            num_overlimit_entries,
                        );

                        // If we couldn't lock any entries to free their space up, then
                        // all cache entries are currently locked. If we just waited
                        // until we lock one, there would be a potential dead lock
                        // if multiple threads hold locks and try to get more locks.
                        // Let's avoid that deadlock and allow the current locking
                        // request, even though it goes above the limit.
                        // This is why we call [AsyncLimit::SoftLimit] a "soft" limit.
                        if locked.is_empty() {
                            // TODO Test that this works, i.e. that the map still correctly works when it's full and doesn't deadlock (and same for the _load_or_insert_mutex_for_key_sync version)
                            break entries;
                        }

                        // We now have some entries locked that may free up enough space.
                        // Let's evict them. We have to free up the entries lock for that
                        // so that the on_evict user code can call back into Self::_unlock()
                        // for those entries. That means other user code may also run and cause
                        // race conditions. Because of that, once on_evict returns, we'll check
                        // take the lock again in the next loop iteration and check again if we now
                        // have enough space
                        std::mem::drop(entries);
                        locked
                    };
                    on_evict(locked).await?;
                }
            }
        };
        let result = match entries.get_or_insert_none(key) {
            GetOrInsertNoneResult::Existing(mutex) => LoadOrInsertMutexResult::Existing { mutex },
            GetOrInsertNoneResult::Inserted(mutex) => {
                // If we just inserted the new entry, it'll have a `None` value. But our invariant says that only locked items
                // can be `None`, so we need to lock it and our caller needs to make sure they handle this correctly.
                let guard = mutex.try_lock_owned().expect(
                    "We're the only one who has seen this mutex so far. Locking can't fail.",
                );
                LoadOrInsertMutexResult::Inserted { guard }
            }
        };
        Ok(result)
    }

    fn _load_or_insert_mutex_for_key_sync<S, E, OnEvictFn>(
        this: &S,
        key: &M::K,
        limit: SyncLimit<M, V, H, S, E, OnEvictFn>,
    ) -> Result<LoadOrInsertMutexResult<EntryValue<M::V>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<M, V, H, S>>) -> Result<(), E>,
    {
        // Note: this logic is duplicated in _load_or_insert_mutex_for_key_sync with some .await calls
        let mut entries = match limit {
            SyncLimit::NoLimit { .. } => {
                // do nothing
                this.borrow()._entries()
            }
            SyncLimit::SoftLimit {
                max_entries,
                mut on_evict,
            } => {
                // free up space for the new entry if necessary
                loop {
                    let locked = {
                        let mut entries = this.borrow()._entries();
                        let num_overlimit_entries =
                            entries.len().saturating_sub(max_entries.get() - 1);
                        if num_overlimit_entries == 0 {
                            // There is enough space, no need to free up space
                            break entries;
                        }
                        // There is not enough space, free up some.
                        let locked = Self::_lock_up_to_n_first_unlocked_entries(
                            this,
                            &mut entries,
                            num_overlimit_entries,
                        );

                        // If we couldn't lock any entries to free their space up, then
                        // all cache entries are currently locked. If we just waited
                        // until we lock one, there would be a potential dead lock
                        // if multiple threads hold locks and try to get more locks.
                        // Let's avoid that deadlock and allow the current locking
                        // request, even though it goes above the limit.
                        // This is why we call [AsyncLimit::SoftLimit] a "soft" limit.
                        if locked.is_empty() {
                            break entries;
                        }

                        // We now have some entries locked that may free up enough space.
                        // Let's evict them. We have to free up the entries lock for that
                        // so that the on_evict user code can call back into Self::_unlock()
                        // for those entries. That means other user code may also run and cause
                        // race conditions. Because of that, once on_evict returns, we'll check
                        // take the lock again in the next loop iteration and check again if we now
                        // have enough space
                        std::mem::drop(entries);
                        locked
                    };
                    on_evict(locked)?;
                }
            }
        };
        let result = match entries.get_or_insert_none(key) {
            GetOrInsertNoneResult::Existing(mutex) => LoadOrInsertMutexResult::Existing { mutex },
            GetOrInsertNoneResult::Inserted(mutex) => {
                // If we just inserted the new entry, it'll have a `None` value. But our invariant says that only locked items
                // can be `None`, so we need to lock it and our caller needs to make sure they handle this correctly.
                let guard = mutex.try_lock_owned().expect(
                    "We're the only one who has seen this mutex so far. Locking can't fail.",
                );
                LoadOrInsertMutexResult::Inserted { guard }
            }
        };
        Ok(result)
    }

    fn _make_guard<S: Borrow<Self>>(
        this: S,
        key: M::K,
        guard: OwnedMutexGuard<EntryValue<M::V>>,
    ) -> Guard<M, V, H, S> {
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
        OnEvictFn: FnMut(Vec<Guard<M, V, H, S>>) -> Result<(), E>,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_sync(&this, &key, limit)?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the thread until the mutex for this key is acquired.

        let guard = match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => mutex.blocking_lock_owned(),
            LoadOrInsertMutexResult::Inserted { guard } => guard,
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
        OnEvictFn: FnMut(Vec<Guard<M, V, H, S>>) -> F,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_async(&this, &key, limit).await?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the task until the mutex for this key is acquired.

        let guard = match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => mutex.lock_owned().await,
            LoadOrInsertMutexResult::Inserted { guard } => guard,
        };

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
        OnEvictFn: FnMut(Vec<Guard<M, V, H, S>>) -> Result<(), E>,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_sync(&this, &key, limit)?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => match mutex.try_lock_owned() {
                Ok(guard) => Ok(Some(Self::_make_guard(this, key, guard))),
                Err(_) => Ok(None),
            },
            LoadOrInsertMutexResult::Inserted { guard } => {
                Ok(Some(Self::_make_guard(this, key, guard)))
            }
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
        OnEvictFn: FnMut(Vec<Guard<M, V, H, S>>) -> F,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_async(&this, &key, limit).await?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => match mutex.try_lock_owned() {
                Ok(guard) => Ok(Some(Self::_make_guard(this, key, guard))),
                Err(_) => Ok(None),
            },
            LoadOrInsertMutexResult::Inserted { guard } => {
                Ok(Some(Self::_make_guard(this, key, guard)))
            }
        }
    }

    pub fn lock_all_unlocked<S: Borrow<Self> + Clone>(
        this: S,
        take_while_condition: &impl Fn(&Guard<M, V, H, S>) -> bool,
    ) -> Vec<Guard<M, V, H, S>> {
        let entries = this.borrow()._entries();
        let mut previously_unlocked_entries = entries
            .iter()
            .filter_map(|(key, mutex)| match Arc::clone(mutex).try_lock_owned() {
                Ok(guard) => Some(Self::_make_guard(this.clone(), key.clone(), guard)),
                Err(_) => None,
            })
            .take_while_inclusive(take_while_condition)
            // Collecting into a Vec so that we don't have to keep `entries` locked
            // while the returned iterator is alive.
            .collect::<Vec<_>>();

        // We now have all entries fulfilling the `take_while_condition` plus one entry that probably does not
        // (however, it might fulfill the condition if all entries fulfill it).
        // We need to remove that last entry and drop it, but before we can do that, we need to drop
        // `entries` because otherwise we'd have a deadlock when the entry tries to unlock itself.
        // This whole issue is actually the reason why we used `take_while_inclusive` instead of just
        // `take_while` above. `take_while` would drop this entry while the stream is being processed
        // and cause this very deadlock.

        std::mem::drop(entries);
        if let Some(last_entry) = previously_unlocked_entries.last() {
            if !take_while_condition(last_entry) {
                std::mem::drop(previously_unlocked_entries.pop().expect(
                    "In this code branch, we already verified that there is a last entry.",
                ));
            }
        }

        previously_unlocked_entries
    }

    /// Locks all entries in the cache and returns their guards as a stream.
    /// For entries that are locked by other threads or tasks, the stream will wait until they are unlocked.
    /// If that other thread or task having a lock for an entry
    /// - creates the entry => the stream will return them
    /// - removes the entry => the stream will not return them
    /// - entries that were locked by another thread or task but don't have a value will not be returned
    pub async fn lock_all_entries<S: Borrow<Self> + Clone>(
        this: S,
    ) -> impl Stream<Item = Guard<M, V, H, S>> {
        let entries = this.borrow()._entries();
        entries
            .iter()
            .map(|(key, mutex)| {
                let this = this.clone();
                let key = key.clone();
                let mutex = Arc::clone(mutex);
                async move {
                    let guard = mutex.lock_owned().await;
                    let guard = Self::_make_guard(this, key, guard);
                    if guard.value().is_some() {
                        Some(guard)
                    } else {
                        None
                    }
                }
            })
            .collect::<FuturesUnordered<_>>()
            // Filter out entries that were removed or not-preexisting and not created while locked
            .filter_map(futures::future::ready)
    }

    pub(super) fn _unlock(&self, key: &M::K, mut guard: OwnedMutexGuard<EntryValue<M::V>>) {
        // We need to get the `entries` lock before we drop the guard, see comment in [Self::_delete_if_unlocked_and_nobody_waiting_for_lock]
        // about other threads not being able to enter this function.
        let mut entries = self._entries();
        self.hooks.on_unlock(guard.value.as_mut());
        let entry_carries_a_value = guard.value.is_some();
        std::mem::drop(guard);

        // Now the guard is dropped and the lock for this key is unlocked.
        // If there are any other Self::blocking_lock/async_lock/try_lock()
        // calls for this key already running and waiting for the mutex,
        // they will be unblocked now and their guard will be created.

        // If the guard we dropped carried a value, keep the entry in the map.
        // But if it doesn't carry a value, clean up since the entry semantically
        // doesn't exist in the map and was only created to have a place to put
        // the mutex.
        if !entry_carries_a_value {
            Self::_delete_if_unlocked_and_nobody_waiting_for_lock(&mut entries, key);
        }
    }

    fn _delete_if_unlocked_and_nobody_waiting_for_lock(
        entries: &mut std::sync::MutexGuard<'_, M>,
        key: &M::K,
    ) {
        let mutex: &Arc<tokio::sync::Mutex<EntryValue<M::V>>> = entries
            .get(key)
            .expect("This entry must exist or the guard passed in as a parameter shouldn't exist");
        // If there are any other locks or any other tasks currently waiting in Self::blocking_lock/async_lock/try_lock,
        // then Arc::strong_count() will be larger than one.
        // But since we still have the global mutex on entries, currently no
        // thread can newly call Self::blocking_lock/async_lock/try_lock() and create a
        // new clone of our Arc. Similarly, no other thread can enter Self::_unlock()
        // and reduce the strong_count of the Arc by dropping the guard. This means that if
        // Arc::strong_count() == 1, we know that there is no other thread with access
        // that could modify strong_count. We can clean up without race conditions.
        if Arc::strong_count(mutex) == 1 {
            let remove_result = entries.remove(key);
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
            .entries
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
                    .unwrap_or_else(|_| panic!("We're the only one with access, there shouldn't be any other threads or tasks that have a copy of this Arc."));
                let value = value.into_inner();

                // Ignore None entries since they don't actually exist in the map and were only created so we have a place to put the mutex.
                value.value.map(|value| (key, value))
            })
    }

    // Caveat: Locked keys are listed even if they don't carry a value
    #[inline]
    pub fn keys_with_entries_or_locked(&self) -> Vec<M::K> {
        let entries = self._entries();
        entries.iter().map(|(key, _value)| key).cloned().collect()
    }

    fn _lock_up_to_n_first_unlocked_entries<S: Borrow<Self> + Clone>(
        this: &S,
        entries: &mut std::sync::MutexGuard<'_, M>,
        num_entries: usize,
    ) -> Vec<Guard<M, V, H, S>> {
        let mut result = Vec::new();
        let mut to_delete = Vec::new();
        for (key, mutex) in entries.iter() {
            if let Ok(guard) = Arc::clone(mutex).try_lock_owned() {
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
            } else {
                // A failed try_lock means we currently have another lock and we can rely on that one
                // calling _unlock and potentially deleting an item if it is None. So we don't need
                // to create a guard object here. Because we have a lock on `entries`, there are no
                // race conditions with that _unlock.
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
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V, H>,
    H: Hooks<M::V>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LockableCache").finish()
    }
}
