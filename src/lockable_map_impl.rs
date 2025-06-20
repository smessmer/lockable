use futures::stream::{FuturesUnordered, Stream, StreamExt};
use itertools::Itertools;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use super::guard::Guard;
use super::limit::{AsyncLimit, SyncLimit};
use super::map_like::{GetOrInsertNoneResult, MapLike};
use super::utils::primary_arc::PrimaryArc;
use crate::utils::primary_arc::{ReplicaArc, ReplicaOwnedMutexGuard};

// TODO Does it make sense to make the inner Mutex (i.e. tokio::sync::Mutex) a template parameter with a Mutex trait? It could allow user code to select std::sync::Mutex if they don't need the async_lock functions, or maybe even use their own mutex type if we make the trait public.

pub trait LockableMapConfig {
    type MapImpl<K, V>: MapLike<K, Entry<Self::WrappedV<V>>>
    where
        K: Eq + PartialEq + Hash + Clone;
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

pub(super) type Entry<V> = PrimaryArc<tokio::sync::Mutex<EntryValue<V>>>;

pub struct LockableMapImpl<K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
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
    // Invariants:
    //   1) Every key can only be locked once at the same time, even if the key has no value in the map.
    //      -> This is fulfilled by inserting `None` values into the map for locking keys that don't have a value.
    //   2) An entry can only be `None` if there is a [ReplicaArc]/[ReplicaOwnedMutexGuard] (e.g. [Guard]) for it, or if `entries` is locked (e.g. we're currently running some code that will reinstate the invariant before unlocking `entries`).
    //      This invariant ensures we don't accidentally leave `None` entries behind when we don't need them anymore.
    //      It is fulfilled through the following rules
    //      2A) Any of the [PrimaryArc] entries in the map can only be cloned (i.e. have the refcount increased) while `entries` is locked. Cloning them will create a [ReplicaArc] that cannot be cloned further.
    //          However, while holding a [ReplicaArc], other threads can come in, lock `entries`, and create their own [ReplicaArc] to the same entry.
    //      2B) Code creating a new `None` [PrimaryArc] in the map must always create a [ReplicaArc]/[ReplicaOwnedMutexGuard] for it.
    //      2C) Anybody dropping a [ReplicaArc]/[ReplicaOwnedMutexGuard] must first get a lock on `entries`, then drop the ReplicaArc, then, if it is None,
    //          call [Self::_delete_if_unlocked_and_nobody_waiting_for_lock] or [Self::_delete_if_unlocked_none_and_nobody_waiting_for_lock] to clean up the `None` entry.
    //          The easiest way to do this is to put the [ReplicaArg] into a [Guard] object that will do this correctly in its [Drop].
    //          Exception: If the same lock on `entries` that was present when the [ReplicaArc] was created is still held, i.e. no other thread could have dropped their instance of the [ReplicaArc] inbetween, then it's ok to just drop it without any `None` checks.
    //      (2C) means that it isn't necessarily the [ReplicaArc] created in (2B) that will clean up the `None` entry, since there could still be other [ReplicaArc]s, but the last [ReplicaArc] will clean it up.
    //  TODO Can 2C be enforced with some kind of Guard that isn't about locking but just about cleaning up the `None` entry? The actual [Guard] could then maybe hold this inner guard.
    entries: std::sync::Mutex<C::MapImpl<K, V>>,

    config: C,

    _k: PhantomData<K>,
    _v: PhantomData<V>,
}

enum LoadOrInsertMutexResult<V> {
    Existing {
        mutex: ReplicaArc<tokio::sync::Mutex<EntryValue<V>>>,
    },
    Inserted {
        guard: ReplicaOwnedMutexGuard<EntryValue<V>>,
    },
}

impl<K, V, C> LockableMapImpl<K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    #[inline]
    pub fn new(config: C) -> Self {
        Self {
            entries: std::sync::Mutex::new(C::MapImpl::<K, V>::new()),
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

    fn _entries(&self) -> EntriesGuard<'_, K, V, C> {
        EntriesGuard::new(self.entries
            .lock()
            .expect("The global mutex protecting the LockableCache is poisoned. This shouldn't happen since there shouldn't be any user code running while this lock is held so no thread should ever panic with it"))
    }

    // WARNING: Call site must be very careful to always fulfill invariant 2C with the returned [ReplicaArc].
    async fn _load_or_insert_mutex_for_key_async<S, E, F, OnEvictFn>(
        this: &S,
        key: &K,
        limit: AsyncLimit<K, V, C, S, E, F, OnEvictFn>,
    ) -> Result<LoadOrInsertMutexResult<C::WrappedV<V>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<K, V, C, S>>) -> F,
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
            GetOrInsertNoneResult::Existing(mutex) => LoadOrInsertMutexResult::Existing {
                // The call site needs to make sure it fulfills invariant 2C when dropping this [ReplicaArc].
                mutex: PrimaryArc::clone(mutex),
            },
            GetOrInsertNoneResult::Inserted(mutex) => {
                // If we just inserted the new entry, it'll have a `None` value. To fulfill invariant 2B, we need to put it in a [ReplicaOwnedMutexGuard].
                // The call site needs to make sure it fulfills invariant 2C when dropping that [ReplicaOwnedMutexGuard].
                let Ok(guard) = PrimaryArc::clone(mutex).try_lock_owned() else {
                    panic!(
                        "We're the only one who has seen this mutex so far. Locking can't fail."
                    );
                };
                LoadOrInsertMutexResult::Inserted { guard }
            }
        };
        Ok(result)
    }

    // WARNING: Call site must be very careful to always fulfill invariant 2C with the returned [ReplicaArc].
    fn _load_or_insert_mutex_for_key_sync<S, E, OnEvictFn>(
        this: &S,
        key: &K,
        limit: SyncLimit<K, V, C, S, E, OnEvictFn>,
    ) -> Result<LoadOrInsertMutexResult<C::WrappedV<V>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<K, V, C, S>>) -> Result<(), E>,
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
            GetOrInsertNoneResult::Existing(mutex) => LoadOrInsertMutexResult::Existing {
                // The call site needs to make sure it fulfills invariant 2C when dropping this [ReplicaArc].
                mutex: PrimaryArc::clone(mutex),
            },
            GetOrInsertNoneResult::Inserted(mutex) => {
                // If we just inserted the new entry, it'll have a `None` value. To fulfill invariant 2B, we need to put it in a [ReplicaOwnedMutexGuard].
                // The call site needs to make sure it fulfills invariant 2C when dropping that [ReplicaOwnedMutexGuard].
                let Ok(guard) = PrimaryArc::clone(mutex).try_lock_owned() else {
                    panic!(
                        "We're the only one who has seen this mutex so far. Locking can't fail."
                    );
                };
                LoadOrInsertMutexResult::Inserted { guard }
            }
        };
        Ok(result)
    }

    fn _make_guard<S: Borrow<Self>>(
        this: S,
        key: K,
        guard: ReplicaOwnedMutexGuard<EntryValue<C::WrappedV<V>>>,
    ) -> Guard<K, V, C, S> {
        Guard::new(this, key, guard)
    }

    #[inline]
    pub fn blocking_lock<S, E, OnEvictFn>(
        this: S,
        key: K,
        limit: SyncLimit<K, V, C, S, E, OnEvictFn>,
    ) -> Result<Guard<K, V, C, S>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<K, V, C, S>>) -> Result<(), E>,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_sync(&this, &key, limit)?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the thread until the mutex for this key is acquired.

        let guard = match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => mutex.blocking_lock_owned(),
            LoadOrInsertMutexResult::Inserted { guard } => guard,
        };

        // To fulfill invariant 2C, we immediately put the [ReplicaOwnedMutexGuard] into a [Guard] object.
        Ok(Self::_make_guard(this, key, guard))
    }

    #[inline]
    pub async fn async_lock<S, E, F, OnEvictFn>(
        this: S,
        key: K,
        limit: AsyncLimit<K, V, C, S, E, F, OnEvictFn>,
    ) -> Result<Guard<K, V, C, S>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<K, V, C, S>>) -> F,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_async(&this, &key, limit).await?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following blocks the task until the mutex for this key is acquired.

        let guard = match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => mutex.lock_owned().await,
            LoadOrInsertMutexResult::Inserted { guard } => guard,
        };

        // To fulfill invariant 2C, we immediately put the [ReplicaOwnedMutexGuard] into a [Guard] object.
        Ok(Self::_make_guard(this, key, guard))
    }

    #[inline]
    pub fn try_lock<S, E, OnEvictFn>(
        this: S,
        key: K,
        limit: SyncLimit<K, V, C, S, E, OnEvictFn>,
    ) -> Result<Option<Guard<K, V, C, S>>, E>
    where
        S: Borrow<Self> + Clone,
        OnEvictFn: FnMut(Vec<Guard<K, V, C, S>>) -> Result<(), E>,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_sync(&this, &key, limit)?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => match mutex.try_lock_owned() {
                Ok(guard) => {
                    // To fulfill invariant 2C, we immediately put the [ReplicaOwnedMutexGuard] into a [Guard] object.
                    Ok(Some(Self::_make_guard(this, key, guard)))
                }
                Err(replica_arc) => {
                    // To fulfill invariant 2C, we need to call [Self::_delete_if_unlocked_none_and_nobody_waiting_for_lock] here.
                    let mut entries = this.borrow()._entries();
                    Self::_delete_if_unlocked_none_and_nobody_waiting_for_lock(
                        &mut entries,
                        &key,
                        replica_arc,
                    );
                    Ok(None)
                }
            },
            LoadOrInsertMutexResult::Inserted { guard } => {
                // To fulfill invariant 2C, we immediately put the [ReplicaOwnedMutexGuard] into a [Guard] object.
                Ok(Some(Self::_make_guard(this, key, guard)))
            }
        }
    }

    #[inline]
    pub async fn try_lock_async<S, E, F, OnEvictFn>(
        this: S,
        key: K,
        limit: AsyncLimit<K, V, C, S, E, F, OnEvictFn>,
    ) -> Result<Option<Guard<K, V, C, S>>, E>
    where
        S: Borrow<Self> + Clone,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Guard<K, V, C, S>>) -> F,
    {
        let mutex = Self::_load_or_insert_mutex_for_key_async(&this, &key, limit).await?;
        // Now we have an Arc::clone of the mutex for this key, and the global mutex is already unlocked so other threads can access the cache.
        // The following tries to lock the mutex.

        match mutex {
            LoadOrInsertMutexResult::Existing { mutex } => match mutex.try_lock_owned() {
                Ok(guard) => {
                    // To fulfill invariant 2C, we immediately put the [ReplicaOwnedMutexGuard] into a [Guard] object.
                    Ok(Some(Self::_make_guard(this, key, guard)))
                }
                Err(replica_arc) => {
                    // TODO Deduplicate this code with the try_lock method
                    // To fulfill invariant 2C, we need to call [Self::_delete_if_unlocked_none_and_nobody_waiting_for_lock] here.
                    let mut entries = this.borrow()._entries();
                    Self::_delete_if_unlocked_none_and_nobody_waiting_for_lock(
                        &mut entries,
                        &key,
                        replica_arc,
                    );
                    Ok(None)
                }
            },
            LoadOrInsertMutexResult::Inserted { guard } => {
                // To fulfill invariant 2C, we immediately put the [ReplicaOwnedMutexGuard] into a [Guard] object.
                Ok(Some(Self::_make_guard(this, key, guard)))
            }
        }
    }

    pub fn lock_all_unlocked<S: Borrow<Self> + Clone>(
        this: S,
        take_while_condition: &impl Fn(&Guard<K, V, C, S>) -> bool,
    ) -> Vec<Guard<K, V, C, S>> {
        let entries = this.borrow()._entries();
        let mut previously_unlocked_entries = entries
            .iter()
            .filter_map(
                |(key, mutex)| match PrimaryArc::clone(mutex).try_lock_owned() {
                    Ok(guard) => Some(Self::_make_guard(this.clone(), key.clone(), guard)),
                    Err(_) => {
                        // Just dropping the [ReplicaArc] here without calling [Self::_delete_if_unlocked_none_and_nobody_waiting_for_lock] is fine
                        // despite invariant 2C, because we hold a lock on `entries` and had that lock since the call to [PrimaryArc::clone].
                        None
                    }
                },
            )
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
    ///
    /// *Warning*: This locks in an arbitrary order and is prone to deadlocks if processing any stream entry
    /// waits for other locks. A common pitfall is calling `.collect()` on the stream, which holds locks until
    /// all locks, in an arbitrary order, are locked.
    pub async fn lock_all_entries<S: Borrow<Self> + Clone>(
        this: S,
    ) -> impl Stream<Item = Guard<K, V, C, S>> {
        let entries = this.borrow()._entries();
        let stream = entries
            .iter()
            .map(|(key, mutex)| {
                let this = this.clone();
                let key = key.clone();
                // Concurrency: PrimaryArc::clone must happen before we go async, while we still have the lock on `entries`,
                //              so that invariant 2A is fulfilled (refcount must only be increased while `entries` is locked).
                //              The refcount will only be decreased through the Guard, which means it will also only happen
                //              while `entries` is locked and invariant 2C is fulfilled.
                let mutex = PrimaryArc::clone(mutex);
                async move {
                    let guard = mutex.lock_owned().await;
                    let guard = Self::_make_guard(this, key, guard);
                    if guard.value().is_some() {
                        Some(guard)
                    } else {
                        // Dropping the guard fulfills invariant 2C.
                        None
                    }
                }
            })
            .collect::<FuturesUnordered<_>>()
            // Filter out entries that were removed or not-preexisting and not created while locked
            .filter_map(futures::future::ready);
        // Drop to ensure that the stream doesn't accidentally capture a lock on `entries`,
        // so that other threads can keep locking/unlocking while the stream is being processed.
        std::mem::drop(entries);
        stream
    }

    pub(super) fn _unlock(
        &self,
        key: &K,
        mut guard: ReplicaOwnedMutexGuard<EntryValue<C::WrappedV<V>>>,
    ) {
        self.config.on_unlock(guard.value.as_mut());
        let entry_carries_a_value = guard.value.is_some();

        // We need to get the `entries` lock before we drop the guard, see invariant 2C.
        let mut entries = self._entries();
        std::mem::drop(guard);

        // Now the guard is dropped and the lock for this key is unlocked.
        // If there are any other Self::blocking_lock/async_lock/try_lock()
        // calls for this key already running and waiting for the mutex,
        // they will be unblocked now and their guard will be created.

        // If the guard we dropped carried a value, keep the entry in the map.
        // But if it doesn't carry a value, clean up since the entry semantically
        // doesn't exist in the map and was only created to have a place to put
        // the mutex. This fulfills invariant 2C.
        if !entry_carries_a_value {
            Self::_delete_if_unlocked_and_nobody_waiting_for_lock(&mut entries, key);
        }
    }

    fn _delete_if_unlocked_and_nobody_waiting_for_lock(
        entries: &mut EntriesGuard<'_, K, V, C>,
        key: &K,
    ) {
        let mutex: &Entry<C::WrappedV<V>> = entries
            .get(key)
            .expect("This entry must exist or this function shouldn't have been called");
        // We have a lock on `entries` and invariant 2A ensures that no other threads or tasks can currently
        // increase num_replicas (i.e. `Arc::strong_count`) or create clones of this Arc. This means that if num_replicas == 0,
        // we know that we are the only ones with a handle to this `Arc` and we can clean it up without race conditions.
        if mutex.num_replicas() == 0 {
            // TODO Combine the `get` above and `remove` here into a single hashing operation, using the hash map's entry API
            let remove_result = entries.remove(key);
            assert!(
                remove_result.is_some(),
                "We just got this entry above from the hash map, it cannot have vanished since then"
            );
        } else {
            // Another task or thread currently has a [ReplicaArc] for this entry, it may or may not be locked. We cannot clean up yet.
            // With invariant 2C, we know that thread or task hasn't cleaned up yet but will wait for us to release the `entries`
            // lock and then eventually call [Self::_delete_if_unlocked_and_nobody_waiting_for_lock] again.
            // We can just exit and let them deal with it.
        }
    }

    fn _delete_if_unlocked_none_and_nobody_waiting_for_lock(
        entries: &mut EntriesGuard<'_, K, V, C>,
        key: &K,
        entry: ReplicaArc<tokio::sync::Mutex<EntryValue<C::WrappedV<V>>>>,
    ) {
        // We have a lock on `entries` and invariant 2A ensures that no other threads or tasks can currently
        // increase num_replicas (i.e. `Arc::strong_count`) or create clones of this Arc. This means that if num_replicas == 1,
        // we know that we are the only ones with a handle to this `Arc` and we can clean it up without race conditions.
        if entry.num_replicas() == 1 {
            let locked = entry
                .try_lock()
                .expect("We're the only one who has access to this mutex. Locking can't fail.");
            if locked.value.is_none() {
                // TODO Combine the `get` above and `remove` here into a single hashing operation, using the hash map's entry API
                let remove_result = entries.remove(key);
                assert!(
                    remove_result.is_some(),
                    "We just got this entry above from the hash map, it cannot have vanished since then"
                );
            }
        } else {
            // Another task or thread currently has a [ReplicaArc] for this entry, it may or may not be locked. We cannot clean up yet.
            // With invariant 2C, we know that thread or task hasn't cleaned up yet but will wait for us to release the `entries`
            // lock and then eventually call [Self::_delete_if_unlocked_and_nobody_waiting_for_lock] again.
            // We can just exit and let them deal with it.
        }
    }

    pub fn into_entries_unordered(self) -> impl Iterator<Item = (K, C::WrappedV<V>)> {
        let entries: C::MapImpl<K, V> = self.entries.into_inner().expect("Lock poisoned");

        #[cfg(any(test, feature = "slow_assertions"))]
        EntriesGuard::<K, V, C>::assert_invariant(&entries);

        // We now have exclusive access to the LockableMapImpl object. Rust lifetime rules ensure that no other thread or task can have any
        // Guard for an entry since both owned and non-owned guards are bound to the lifetime of the LockableMapImpl (owned guards
        // indirectly through the Arc but if user code calls this function, it means they had to call Arc::try_unwrap or something similar
        // which ensures that there are no other threads with access to it.

        entries
            .into_iter()
            .map(|(key, value)| {
                let value = PrimaryArc::try_unwrap(value)
                    .unwrap_or_else(|_| panic!("We're the only one with access, there shouldn't be any other threads or tasks that have a copy of this Arc."));
                let value = value.into_inner().value.expect("Invariant 2 violated. There shouldn't be any `None` entries since there aren't any ReplicaArcs.");
                (key, value)
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
        entries: &mut EntriesGuard<'_, K, V, C>,
        num_entries: usize,
    ) -> Vec<Guard<K, V, C, S>> {
        let mut result = Vec::with_capacity(num_entries);
        for (key, mutex) in entries.iter() {
            match PrimaryArc::clone(mutex).try_lock_owned() {
                Ok(guard) => {
                    if guard.value.is_some() {
                        result.push(Self::_make_guard(this.clone(), key.clone(), guard))
                    } else {
                        // We have not created this `None` entry and according to invariant 2, we know that there is
                        // a [ReplicaArc] for it somewhere. Even though it seems to be unlocked, we know it exists.
                        // Because we have a lock on `entries`, invariant 2C tells us that this [ReplicaArc]'s destruction
                        // will wait for us before cleaning up the `None`.
                        // We don't need to worry about cleaning up the `None` ourselves.
                        assert!(mutex.num_replicas() > 1, "Invariant violated");
                    }
                }
                Err(_replica_arc) => {
                    // Invariant 2C:
                    // A failed try_lock means we currently have another lock and we can rely on that one
                    // calling _unlock and potentially deleting an item if it is None. So we don't need
                    // to create a guard object here. Because we have a lock on `entries`, there are no
                    // race conditions with that _unlock.
                    // It's ok to just drop the [ReplicaArc] without calling _delete_if_unlocked_none_and_nobody_waiting_for_lock
                    // because we have a lock on `entries` and had that lock since the call to [PrimaryArc::clone].
                    assert!(mutex.num_replicas() > 1, "Invariant violated");
                }
            }
            if result.len() >= num_entries {
                break;
            }
        }
        result
    }
}

impl<K, V, C> Debug for LockableMapImpl<K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LockableMapImpl").finish()
    }
}

/// Simple wrapper around `MutexGuard<C::MapImpl<K, V>>`.
/// In release mode, this doesn't do anything else.
/// In debug mode or tests, it enforces our invariants.
struct EntriesGuard<'a, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    entries: std::sync::MutexGuard<'a, C::MapImpl<K, V>>,
    _k: PhantomData<K>,
    _v: PhantomData<V>,
    _c: PhantomData<C>,
}

impl<'a, K, V, C> EntriesGuard<'a, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    #[track_caller]
    fn new(entries: std::sync::MutexGuard<'a, C::MapImpl<K, V>>) -> Self {
        #[cfg(any(test, feature = "slow_assertions"))]
        Self::assert_invariant(&entries);

        Self {
            entries,
            _k: PhantomData,
            _v: PhantomData,
            _c: PhantomData,
        }
    }

    #[cfg(any(test, feature = "slow_assertions"))]
    #[track_caller]
    fn assert_invariant(entries: &C::MapImpl<K, V>) {
        for (_key, entry) in entries.iter() {
            if entry.num_replicas() == 0 {
                let Ok(guard) = PrimaryArc::clone(entry).try_lock_owned() else {
                    panic!("We're the only one with access, locking can't fail");
                };
                assert!(
                    guard.value.is_some(),
                    "Invariant 2 violated. Found an entry without ReplicaArcs that is None."
                );
            }
        }
    }
}

impl<K, V, C> Deref for EntriesGuard<'_, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    type Target = C::MapImpl<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl<K, V, C> DerefMut for EntriesGuard<'_, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entries
    }
}

#[cfg(any(test, feature = "slow_assertions"))]
impl<K, V, C> Drop for EntriesGuard<'_, K, V, C>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
{
    fn drop(&mut self) {
        Self::assert_invariant(&self.entries);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lockable_hash_map::LockableHashMapConfig;

    #[test]
    fn test_debug() {
        let map = LockableMapImpl::<i64, String, LockableHashMapConfig>::new(LockableHashMapConfig);
        assert_eq!("LockableMapImpl", format!("{:?}", map));
    }
}
