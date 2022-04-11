use anyhow::Result;
use futures::{
    future,
    stream::{FuturesUnordered, Stream, StreamExt},
};
use std::borrow::Borrow;
use std::fmt::Debug;
use std::sync::Arc;

use super::error::TryLockError;
use super::guard::GuardImpl;
use super::hooks::{Hooks, NoopHooks};
use super::map_like::{ArcMutexMapLike, EntryValue};
use crate::utils::locked_mutex_guard::LockedMutexGuard;

pub struct LockableMapImpl<M, H>
where
    M: ArcMutexMapLike,
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
    cache_entries: std::sync::Mutex<M>,

    hooks: H,
}

impl<M> LockableMapImpl<M, NoopHooks>
where
    M: ArcMutexMapLike,
{
    #[inline]
    pub fn new() -> Self {
        Self::new_with_hooks(NoopHooks)
    }
}

impl<M, H> LockableMapImpl<M, H>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
{
    #[inline]
    pub fn new_with_hooks(hooks: H) -> Self {
        Self {
            cache_entries: std::sync::Mutex::new(M::new()),
            hooks,
        }
    }

    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self._cache_entries().len()
    }

    #[inline]
    fn _cache_entries(&self) -> std::sync::MutexGuard<'_, M> {
        self.cache_entries
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
    pub fn blocking_lock<S: Borrow<Self>>(this: S, key: M::K) -> GuardImpl<M, H, S> {
        let mutex = this.borrow()._load_or_insert_mutex_for_key(&key);
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks until the mutex for this key is acquired.

        let guard = LockedMutexGuard::blocking_lock(mutex);
        GuardImpl::new(this, key, guard)
    }

    #[inline]
    pub fn try_lock<S: Borrow<Self>>(
        this: S,
        key: M::K,
    ) -> Result<GuardImpl<M, H, S>, TryLockError> {
        let mutex = this.borrow()._load_or_insert_mutex_for_key(&key);
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        let guard = match LockedMutexGuard::try_lock(mutex) {
            Ok(guard) => Ok(guard),
            Err(_) => Err(TryLockError::WouldBlock),
        }?;
        let guard = GuardImpl::new(this, key, guard);
        Ok(guard)
    }

    pub(super) fn _unlock(&self, key: &M::K, mut guard: LockedMutexGuard<EntryValue<M::V>>) {
        let mut cache_entries = self._cache_entries();
        let mutex: &Arc<tokio::sync::Mutex<EntryValue<M::V>>> = cache_entries
            .get(key)
            .expect("This entry must exist or the guard passed in as a parameter shouldn't exist");
        self.hooks.on_unlock(guard.value.as_mut());
        let entry_carries_a_value = guard.value.is_some();
        std::mem::drop(guard);

        // Now the guard is dropped and the lock for this key is unlocked.
        // If there are any other Self::lock() calls for this key already running and
        // waiting for the mutex, they will be unblocked now and their guard
        // will be created.
        // But since we still have the global mutex on self.cache_entries, currently no
        // thread can newly call Self::lock() and create a clone of our Arc. Similarly,
        // no other thread can enter Self::unlock() and reduce the strong_count of the Arc.
        // This means that if Arc::strong_count() == 1, we know that we can clean up
        // without race conditions.

        if Arc::strong_count(mutex) == 1 {
            // The guard we're about to drop is the last guard for this mutex,
            // the only other Arc pointing to it is the one in the hashmap.
            // If it carries a value, keep it, but it doesn't carry a value,
            // clean up to fulfill the invariant
            if !entry_carries_a_value {
                let remove_result = cache_entries.remove(key);
                assert!(
                    remove_result.is_some(),
                    "We just got this entry above from the hash map, it cannot have vanished since then"
                );
            }
        }
    }

    #[inline]
    pub async fn async_lock<S: Borrow<Self>>(this: S, key: M::K) -> GuardImpl<M, H, S> {
        let mutex = this.borrow()._load_or_insert_mutex_for_key(&key);
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks until the mutex for this key is acquired.

        let guard = LockedMutexGuard::async_lock(mutex).await;
        GuardImpl::new(this, key, guard)
    }

    /// TODO Docs
    /// TODO Test
    // pub fn lock_entries_unlocked_for_longer_than(
    //     &self,
    //     duration: Duration,
    // ) -> Vec<Guard<'_, M>> {
    //     let now = Instant::now();
    //     let mut result = vec![];
    //     let cache_entries = self._cache_entries();
    //     let mut current_entry_timestamp = None;
    //     // TODO Check that iter().rev() actually starts with the oldest ones and not with the newest once. Otherwise, remove .rev().
    //     for (key, entry) in cache_entries.iter().rev() {
    //         if Arc::strong_count(&entry) == 1 {
    //             // There is currently nobody who has access to this mutex and could lock it.
    //             // And since we're also blocking the global cache mutex, nobody can get it.
    //             // We must be able to lock this and we can safely prune it.
    //             let guard = LockedMutexGuard::try_lock(Arc::clone(&entry)).expect(
    //                 "We just checked that nobody can lock this. But for some reason it was locked.",
    //             );
    //             assert!(
    //                 guard.last_unlocked >= current_entry_timestamp.unwrap_or(guard.last_unlocked),
    //                 "Cache order broken - entries don't seem to be in LRU order"
    //             );
    //             current_entry_timestamp = Some(guard.last_unlocked);

    //             if now - guard.last_unlocked <= duration {
    //                 // The next entry is too new to be pruned
    //                 // TODO Assert that all remaining entries are too new to be pruned, i.e. continue walk through remaining entries and check order
    //                 return result;
    //             }

    //             result.push(Guard::new(self, key.clone(), guard));
    //         } else {
    //             // Somebody currently has access to this mutex and is likely going to lock it.
    //             // This means the entry shouldn't be pruned, it will soon get a new timestamp.
    //         }
    //     }

    //     // We ran out of entries to check, no entry is too new to be pruned.
    //     result
    // }

    pub fn into_entries_unordered(self) -> impl Stream<Item = (M::K, M::V)> {
        let entries: M = self.cache_entries.into_inner().expect("Lock poisoned");

        // We now have exclusive access to the LruCache object. No other thread or task can call lock() and increase
        // the refcount for one of the Arcs. They still can have (un-cloneable) Guard instances and those will eventually call
        // _unlock() on destruction. We just need to wait until the last thread gives up an Arc and then we can remove it from the mutex.

        let entries: FuturesUnordered<_> = entries
            .into_iter()
            .map(|(key, value)| future::ready((key, value)))
            .collect();
        entries.filter_map(|(key, value)| async {
            while Arc::strong_count(&value) > 1 {
                // TODO Is there a better alternative that doesn't involve busy waiting?
                tokio::task::yield_now().await;
            }
            // Now we're the last task having a reference to this arc.
            let value = Arc::try_unwrap(value)
                .expect("This can't fail since we are the only task having access");
            let value = value.into_inner();

            // Ignore None entries
            value.value.map(|value| (key, value))
        })
    }

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

impl<M, H> Debug for LockableMapImpl<M, H>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LockableCache").finish()
    }
}
