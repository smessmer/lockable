use futures::stream::{FuturesUnordered, Stream, StreamExt};
use itertools::Itertools;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::OwnedMutexGuard;

use super::guard::Guard;
use super::limit::{AsyncLimit, SyncLimit};
use super::map_like::{GetOrInsertNoneResult, MapLike};

pub trait LockableMapConfig {
    type WrappedV<V>;

    fn borrow_value<V>(v: &Self::WrappedV<V>) -> &V;
    fn borrow_value_mut<V>(v: &mut Self::WrappedV<V>) -> &mut V;
    fn wrap_value<V>(&self, v: V) -> Self::WrappedV<V>;
    fn unwrap_value<V>(v: Self::WrappedV<V>) -> V;

    /// This gets executed every time a value is unlocked.
    /// The `v` parameter is the value that is being unlocked.
    /// It is `None` if we locked and then unlocked a key that
    /// actually doesn't have an entry in the map.
    fn on_unlock<V>(&self, v: Option<&mut Self::WrappedV<V>>);
}

#[derive(Debug)]
pub struct EntryValue<V> {
    // While unlocked, an entry is always Some. While locked, it can be temporarily None
    // since we enter None values into the map to lock keys that actually don't exist in the map.
    pub(super) value: Option<V>,
}

pub(super) type Entry<V> = Arc<tokio::sync::Mutex<EntryValue<V>>>;

pub struct LockableMapImpl<M, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    M: MapLike<K, Entry<C::WrappedV<V>>>,
    C: LockableMapConfig + Clone,
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
    // Invariant (true whenever `entries` is unlocked):
    //   If an entry is in the map and the value is `None`, then there must be a lock on the entry.
    //   This is because the only reason to allow `None` in the map is for locking keys that don't have a value.
    //   But once the lock is released, the entry should be removed from the map.
    entries: std::sync::Mutex<M>,

    config: C,

    _k: PhantomData<K>,
    _v: PhantomData<V>,
}

enum LoadOrInsertMutexResult<V> {
    Existing {
        mutex: Arc<tokio::sync::Mutex<V>>,
    },
    Inserted {
        guard: tokio::sync::OwnedMutexGuard<V>,
    },
}

impl<M, K, V, C> LockableMapImpl<M, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    M: MapLike<K, Entry<C::WrappedV<V>>>,
    C: LockableMapConfig + Clone,
{
    #[inline]
    pub fn new(config: C) -> Self {
        Self {
            entries: std::sync::Mutex::new(M::new()),
            config,
            _k: PhantomData,
            _v: PhantomData,
        }
    }

    #[inline]
    pub fn config(&self) -> &C {
        &self.config
    }

    #[cfg(test)]
    #[inline]
    pub fn config_mut(&mut self) -> &mut C {
        &mut self.config
    }

    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self._entries().len()
    }

    fn _entries(&self) -> std::sync::MutexGuard<'_, M> {
        self.entries
            .lock()
            .expect("The global mutex protecting the LockableCache is poisoned. This shouldn't happen since there shouldn't be any user code running while this lock is held so no thread should ever panic with it")
    }

    async fn _load_or_insert_mutex_for_key_async<S, E, F, OnEvictFn>(
        this: &S,
        key: &K,
        limit: AsyncLimit<M, K, V, C, S, E, F, OnEvictFn>,
    ) -> Result<LoadOrInsertMutexResult<EntryValue<C::WrappedV<V>>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<M, K, V, C, S>>) -> F,
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
                        // for those entries. Additionally, we don't want `entries` to stay locked
                        // since user eviction code may write the entry back to an underlying
                        // storage layer and take some time.
                        //
                        // We do want to keep the entry itself locked and pass the guard to user code,
                        // because if user code does do writebacks, it probably needs the entry to be
                        // locked until the writeback is complete.
                        //
                        // However, unlocking entries means other user code may also run and cause
                        // race conditions, e.g. add new entries into the space we just created.
                        // Because of that, once on_evict returns, we'll check take the lock again
                        // in the next loop iteration and check again if we now have enough space.
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
        key: &K,
        limit: SyncLimit<M, K, V, C, S, E, OnEvictFn>,
    ) -> Result<LoadOrInsertMutexResult<EntryValue<C::WrappedV<V>>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<M, K, V, C, S>>) -> Result<(), E>,
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
                        // for those entries. Additionally, we don't want `entries` to stay locked
                        // since user eviction code may write the entry back to an underlying
                        // storage layer and take some time.
                        //
                        // We do want to keep the entry itself locked and pass the guard to user code,
                        // because if user code does do writebacks, it probably needs the entry to be
                        // locked until the writeback is complete.
                        //
                        // However, unlocking entries means other user code may also run and cause
                        // race conditions, e.g. add new entries into the space we just created.
                        // Because of that, once on_evict returns, we'll check take the lock again
                        // in the next loop iteration and check again if we now have enough space.
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
        key: K,
        guard: OwnedMutexGuard<EntryValue<C::WrappedV<V>>>,
    ) -> Guard<M, K, V, C, S> {
        Guard::new(this, key, guard)
    }

    #[inline]
    pub fn blocking_lock<S, E, OnEvictFn>(
        this: S,
        key: K,
        limit: SyncLimit<M, K, V, C, S, E, OnEvictFn>,
    ) -> Result<Guard<M, K, V, C, S>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<M, K, V, C, S>>) -> Result<(), E>,
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
        key: K,
        limit: AsyncLimit<M, K, V, C, S, E, F, OnEvictFn>,
    ) -> Result<Guard<M, K, V, C, S>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<M, K, V, C, S>>) -> F,
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
        key: K,
        limit: SyncLimit<M, K, V, C, S, E, OnEvictFn>,
    ) -> Result<Option<Guard<M, K, V, C, S>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<M, K, V, C, S>>) -> Result<(), E>,
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
        key: K,
        limit: AsyncLimit<M, K, V, C, S, E, F, OnEvictFn>,
    ) -> Result<Option<Guard<M, K, V, C, S>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<M, K, V, C, S>>) -> F,
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
        take_while_condition: &impl Fn(&Guard<M, K, V, C, S>) -> bool,
    ) -> Vec<Guard<M, K, V, C, S>> {
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
        if let Some(last_entry) = previously_unlocked_entries.pop() {
            if take_while_condition(&last_entry) {
                // It actually fulfilled the take_while_condition.
                // This can happen if all entries in the map fulfill the condition and this was the overall last map element.
                // We actually want to return it, so add it back to the return value.
                previously_unlocked_entries.push(last_entry);
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
    ) -> impl Stream<Item = Guard<M, K, V, C, S>> {
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

    pub(super) fn _unlock(&self, key: &K, mut guard: OwnedMutexGuard<EntryValue<C::WrappedV<V>>>) {
        // We need to get the `entries` lock before we drop the guard, see comment in [Self::_delete_if_unlocked_and_nobody_waiting_for_lock]
        // about other threads not being able to enter this function.
        let mut entries = self._entries();
        // TODO Can we move the hook and the entry_carries_a_value calculation to above locking `entries`?
        self.config.on_unlock(guard.value.as_mut());
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
        key: &K,
    ) {
        let mutex: &Arc<tokio::sync::Mutex<EntryValue<C::WrappedV<V>>>> = entries
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
            // TODO Combine the `get` above and `remove` here into a single hashing operation, using the hash map's entry API
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

    pub fn into_entries_unordered(self) -> impl Iterator<Item = (K, C::WrappedV<V>)> {
        let entries: M = self.entries.into_inner().expect("Lock poisoned");

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
    pub fn keys_with_entries_or_locked(&self) -> Vec<K> {
        let entries = self._entries();
        entries.iter().map(|(key, _value)| key).cloned().collect()
    }

    fn _lock_up_to_n_first_unlocked_entries<S: Borrow<Self> + Clone>(
        this: &S,
        entries: &mut std::sync::MutexGuard<'_, M>,
        num_entries: usize,
    ) -> Vec<Guard<M, K, V, C, S>> {
        let mut result = Vec::with_capacity(num_entries);
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

impl<M, K, V, C> Debug for LockableMapImpl<M, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    M: MapLike<K, Entry<C::WrappedV<V>>>,
    C: LockableMapConfig + Clone,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LockableMapImpl").finish()
    }
}
