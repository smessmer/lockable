use anyhow::Result;
use futures::stream::{FuturesUnordered, Stream};
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::error::TryLockError;
use super::guard::GuardImpl;
use super::hooks::{Hooks, NoopHooks};
use super::map_like::{ArcMutexMapLike, EntryValue};
use crate::utils::locked_mutex_guard::LockedMutexGuard;

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
    // except through non-cloneable GuardImpl objects encapsulating those Arcs.
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

    #[inline]
    pub fn num_unlocked(&self) -> usize {
        let entries = self._cache_entries();
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

    #[inline]
    fn _cache_entries(&self) -> std::sync::MutexGuard<'_, M> {
        self.cache_entries
            .as_ref()
            .expect("Object is currently being destructed")
            .lock()
            .expect("The global mutex protecting the LockableCache is poisoned. This shouldn't happen since there shouldn't be any user code running while this lock is held so no thread should ever panic with it")
    }

    #[inline]
    fn _load_or_insert_mutex_for_key(
        &self,
        key: &M::K,
    ) -> Arc<tokio::sync::Mutex<EntryValue<M::V>>> {
        let mut cache_entries = self._cache_entries();
        let entry = cache_entries.get_or_insert_none(key);
        Arc::clone(entry)
    }

    #[inline]
    pub fn blocking_lock<S: Borrow<Self>>(this: S, key: M::K) -> GuardImpl<M, V, H, S> {
        let mutex = this.borrow()._load_or_insert_mutex_for_key(&key);
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the thread until the mutex for this key is acquired.

        let guard = LockedMutexGuard::blocking_lock(mutex);
        this.borrow().num_locked.fetch_add(1, Ordering::SeqCst);
        GuardImpl::new(this, key, guard)
    }

    #[inline]
    pub async fn async_lock<S: Borrow<Self>>(this: S, key: M::K) -> GuardImpl<M, V, H, S> {
        let mutex = this.borrow()._load_or_insert_mutex_for_key(&key);
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the task until the mutex for this key is acquired.

        let guard = LockedMutexGuard::async_lock(mutex).await;
        this.borrow().num_locked.fetch_add(1, Ordering::SeqCst);
        GuardImpl::new(this, key, guard)
    }

    #[inline]
    pub fn try_lock<S: Borrow<Self>>(
        this: S,
        key: M::K,
    ) -> Result<GuardImpl<M, V, H, S>, TryLockError> {
        let mutex = this.borrow()._load_or_insert_mutex_for_key(&key);
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        let guard = match LockedMutexGuard::try_lock(mutex) {
            Ok(guard) => Ok(guard),
            Err(_) => Err(TryLockError::WouldBlock),
        }?;
        this.borrow().num_locked.fetch_add(1, Ordering::SeqCst);
        let guard = GuardImpl::new(this, key, guard);
        Ok(guard)
    }

    // TODO Test
    pub fn lock_all_unlocked<S: Borrow<Self> + Clone>(
        this: S,
    ) -> impl Iterator<Item = GuardImpl<M, V, H, S>> {
        let cache_entries = this.borrow()._cache_entries();
        cache_entries
            .iter()
            .filter_map(
                |(key, mutex)| match LockedMutexGuard::try_lock(Arc::clone(mutex)) {
                    Ok(guard) => {
                        this.borrow().num_locked.fetch_add(1, Ordering::SeqCst);
                        Some(GuardImpl::new(this.clone(), key.clone(), guard))
                    }
                    Err(_) => None,
                },
            )
            .collect::<Vec<_>>()
            .into_iter()
    }

    // TODO Test
    pub async fn lock_all<S: Borrow<Self> + Clone>(
        this: S,
    ) -> impl Stream<Item = GuardImpl<M, V, H, S>> {
        let cache_entries = this.borrow()._cache_entries();
        let cache_entries: FuturesUnordered<_> = cache_entries
            .iter()
            .map(|(key, mutex)| {
                let this = this.clone();
                let key = key.clone();
                let mutex = Arc::clone(mutex);
                async move {
                    let guard = LockedMutexGuard::async_lock(mutex).await;
                    this.borrow().num_locked.fetch_add(1, Ordering::SeqCst);
                    GuardImpl::new(this, key, guard)
                }
            })
            .collect();
        cache_entries
    }

    pub(super) fn _unlock(&self, key: &M::K, mut guard: LockedMutexGuard<EntryValue<M::V>>) {
        let mut cache_entries = self._cache_entries();
        let mutex: &Arc<tokio::sync::Mutex<EntryValue<M::V>>> = cache_entries
            .get(key)
            .expect("This entry must exist or the guard passed in as a parameter shouldn't exist");
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
        // But since we still have the global mutex on self.cache_entries, currently no
        // thread can newly call Self::blocking_lock/async_lock/try_lock() and create a
        // new clone of our Arc. Similarly, no other thread can enter Self::_unlock()
        // and reduce the strong_count of the Arc. This means that if
        // Arc::strong_count() == 1, we know that there is no other thread with access
        // that could modify strong_count. We can clean up without race conditions.

        if Arc::strong_count(mutex) == 1 {
            // If the guard we dropped carried a value, keep the entry in the map.
            // But it doesn't carry a value, clean up since the entry semantically
            // doesn't exist in the map and was only created to have a place to put
            // the mutex.
            if !entry_carries_a_value {
                let remove_result = cache_entries.remove(key);
                assert!(
                    remove_result.is_some(),
                    "We just got this entry above from the hash map, it cannot have vanished since then"
                );
            }
        } else {
            // Another thread was currently waiting in a Self::blocking_lock/async_lock/try_lock call
            // and will now get the lock. We shouldn't free the entry.
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
        // GuardImpl for an entry since both owned and non-owned guards are bound to the lifetime of the LockableMapImpl (owned guards
        // indirectly through the Arc but if user code calls this function, it means they had to call Arc::try_unwrap or something similar
        // which ensures that there are no other threads with access to it.

        entries
            .into_iter()
            .filter_map(|(key, value)| {
                let value = Arc::try_unwrap(value)
                    .expect("We're the only one with access, there shouldn't be any other threads or tasks that have a copy of this Arc.");
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
        assert_eq!(
            0,
            self.num_locked.load(Ordering::SeqCst),
            "Miscalculation in num_locked",
        );
    }
}
