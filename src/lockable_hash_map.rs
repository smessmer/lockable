use futures::stream::Stream;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::guard::Guard;
use super::hooks::NoopHooks;
use super::limit::{AsyncLimit, SyncLimit};
use super::lockable_map_impl::LockableMapImpl;
use super::lockable_trait::Lockable;
use super::map_like::{ArcMutexMapLike, EntryValue};

type MapImpl<K, V> = HashMap<K, Arc<tokio::sync::Mutex<EntryValue<V>>>>;

impl<K, V> ArcMutexMapLike for MapImpl<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type K = K;
    type V = V;
    type ItemIter<'a> = std::collections::hash_map::Iter<'a, K, Arc<Mutex<EntryValue<V>>>>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        Self::new()
    }

    fn len(&self) -> usize {
        self.iter().len()
    }

    fn get_or_insert_none(&mut self, key: &Self::K) -> &Arc<Mutex<EntryValue<Self::V>>> {
        // TODO Is there a way to only clone the key when the entry doesn't already exist?
        //      Might be possible with the upcoming RawEntry API. If we do that, we may
        //      even be able to remove the `Clone` bound from `K` everywhere in this library.
        self.entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(EntryValue { value: None })))
    }

    fn get(&mut self, key: &Self::K) -> Option<&Arc<Mutex<EntryValue<Self::V>>>> {
        HashMap::get(self, key)
    }

    fn remove(&mut self, key: &Self::K) -> Option<Arc<Mutex<EntryValue<Self::V>>>> {
        self.remove(key)
    }

    fn iter(&self) -> Self::ItemIter<'_> {
        HashMap::iter(self)
    }
}

/// A threadsafe hash map where individual keys can be locked/unlocked, even if there is no entry for this key in the map.
/// It initially considers all keys as "unlocked", but they can be locked and if a second thread tries to acquire a lock
/// for the same key, they will have to wait.
///
/// ```
/// use lockable::{AsyncLimit, LockableHashMap};
///
/// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let entry1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
/// let entry2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
///
/// // This next line would cause a deadlock or panic because `4` is already locked on this thread
/// // let entry3 = lockable_map.async_lock(4).await;
///
/// // After dropping the corresponding guard, we can lock it again
/// std::mem::drop(entry1);
/// let entry3 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// The guards holding a lock for an entry can be used to insert that entry to the hash map, remove
/// it from the hash map, or to modify the value of an existing entry.
///
/// ```
/// use lockable::{AsyncLimit, LockableHashMap};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// async fn insert_entry(
///     lockable_map: &LockableHashMap<i64, String>,
/// ) -> Result<(), lockable::Never> {
///     let mut entry_guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.insert(String::from("Hello World"));
///     Ok(())
/// }
///
/// async fn remove_entry(
///     lockable_map: &LockableHashMap<i64, String>,
/// ) -> Result<(), lockable::Never> {
///     let mut entry_guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.remove();
///     Ok(())
/// }
///
/// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
/// assert_eq!(
///     None,
///     lockable_map
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// insert_entry(&lockable_map).await;
/// assert_eq!(
///     Some(&String::from("Hello World")),
///     lockable_map
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// remove_entry(&lockable_map).await;
/// assert_eq!(
///     None,
///     lockable_map
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
///
/// You can use an arbitrary type to index hash map entries by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone].
///
/// ```
/// use lockable::{AsyncLimit, LockableHashMap};
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// struct CustomLockKey(u32);
///
/// let lockable_map: LockableHashMap<CustomLockKey, String> = LockableHashMap::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = lockable_map
///     .async_lock(CustomLockKey(4), AsyncLimit::no_limit())
///     .await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// Under the hood, a [LockableHashMap] is a [std::collections::HashMap] of [Mutex](tokio::sync::Mutex)es, with some logic making sure that
/// empty entries can also be locked and that there aren't any race conditions when adding or removing entries.
#[derive(Debug)]
pub struct LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    map_impl: LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
}

impl<K, V> Lockable<K, V> for LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type Guard<'a> = Guard<
    MapImpl<K, V>,
    V,
    NoopHooks,
    &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
> where
    K: 'a,
    V: 'a;

    type OwnedGuard = Guard<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>>;

    #[cfg(test)]
    fn _num_entries_or_locked(&self) -> usize {
        self.num_entries_or_locked()
    }
}

impl<K, V> LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    /// Create a new hash map with no entries and no locked keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            map_impl: LockableMapImpl::new(),
        }
    }

    /// Return the number of map entries.
    ///
    /// Corner case: Currently locked keys are counted even if they don't exist in the map.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    ///
    /// // Insert two entries
    /// lockable_map
    ///     .async_lock(4, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 4"));
    /// lockable_map
    ///     .async_lock(5, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 5"));
    /// // Keep a lock on a third entry but don't insert it
    /// let guard = lockable_map.async_lock(6, AsyncLimit::no_limit()).await?;
    ///
    /// // Now we have two entries and one additional locked guard
    /// assert_eq!(3, lockable_map.num_entries_or_locked());
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self.map_impl.num_entries_or_locked()
    }

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
    ///
    /// If the lock with this key is currently locked by a different thread, then the current thread blocks until it becomes available.
    /// Upon returning, the thread is the only thread with the lock held. A RAII guard is returned to allow scoped unlock
    /// of the lock. When the guard goes out of scope, the lock will be unlocked.
    ///
    /// This function can only be used from non-async contexts and will panic if used from async contexts.
    ///
    /// The exact behavior on locking a lock in the thread which already holds the lock is left unspecified.
    /// However, this function will not return on the second call (it might panic or deadlock, for example).
    ///
    /// The `limit` parameter can be used to set a limit on the number of entries in the cache, see the documentation of [SyncLimit]
    /// for an explanation of how exactly it works.
    ///
    /// Panics
    /// -----
    /// - This function might panic when called if the lock is already held by the current thread.
    /// - This function will also panic when called from an `async` context.
    ///   See documentation of [tokio::sync::Mutex] for details.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{LockableHashMap, SyncLimit};
    ///
    /// # (|| {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// let guard1 = lockable_map.blocking_lock(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_map.blocking_lock(5, SyncLimit::no_limit())?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = lockable_map.blocking_lock(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.blocking_lock(4, SyncLimit::no_limit())?;
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn blocking_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: SyncLimit<
            MapImpl<K, V>,
            V,
            NoopHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
            E,
            OnEvictFn,
        >,
    ) -> Result<<Self as Lockable<K, V>>::Guard<'a>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> Result<(), E>,
    {
        LockableMapImpl::blocking_lock(&self.map_impl, key, limit)
    }

    /// Lock a lock by key and return a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableHashMap::blocking_lock], please see documentation for that function for more information.
    /// But different to [LockableHashMap::blocking_lock], [LockableHashMap::blocking_lock_owned] works on an `Arc<LockableHashMap>`
    /// instead of a [LockableHashMap] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableHashMap] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableHashMap::blocking_lock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{LockableHashMap, SyncLimit};
    /// use std::sync::Arc;
    ///
    /// # (|| {
    /// let lockable_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = lockable_map.blocking_lock_owned(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_map.blocking_lock_owned(5, SyncLimit::no_limit())?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = lockable_map.blocking_lock_owned(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.blocking_lock_owned(4, SyncLimit::no_limit())?;
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn blocking_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: SyncLimit<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>, E, OnEvictFn>,
    ) -> Result<<Self as Lockable<K, V>>::OwnedGuard, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> Result<(), E>,
    {
        LockableMapImpl::blocking_lock(Arc::clone(self), key, limit)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
    ///
    /// If the lock could not be acquired because it is already locked, then [Ok](Ok)([None]) is returned. Otherwise, a RAII guard is returned.
    /// The lock will be unlocked when the guard is dropped.
    ///
    /// This function does not block and can be used from both async and non-async contexts.
    ///
    /// The `limit` parameter can be used to set a limit on the number of entries in the cache, see the documentation of [SyncLimit]
    /// for an explanation of how exactly it works.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{LockableHashMap, SyncLimit};
    ///
    /// # (|| {
    /// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
    /// let guard1 = lockable_map.blocking_lock(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_map.blocking_lock(5, SyncLimit::no_limit())?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_map.try_lock(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.try_lock(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn try_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: SyncLimit<
            MapImpl<K, V>,
            V,
            NoopHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
            E,
            OnEvictFn,
        >,
    ) -> Result<Option<<Self as Lockable<K, V>>::Guard<'a>>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> Result<(), E>,
    {
        LockableMapImpl::try_lock(&self.map_impl, key, limit)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableHashMap::try_lock], please see documentation for that function for more information.
    /// But different to [LockableHashMap::try_lock], [LockableHashMap::try_lock_owned] works on an `Arc<LockableHashMap>`
    /// instead of a [LockableHashMap] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableHashMap] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableHashMap::try_lock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{LockableHashMap, SyncLimit};
    /// use std::sync::Arc;
    ///
    /// # (||{
    /// let lockable_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = lockable_map.blocking_lock(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_map.blocking_lock(5, SyncLimit::no_limit())?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_map.try_lock_owned(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.try_lock_owned(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn try_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: SyncLimit<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>, E, OnEvictFn>,
    ) -> Result<Option<<Self as Lockable<K, V>>::OwnedGuard>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> Result<(), E>,
    {
        LockableMapImpl::try_lock(Arc::clone(self), key, limit)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableHashMap::try_lock], please see documentation for that function for more information.
    /// But different to [LockableHashMap::try_lock], [LockableHashMap::try_lock_async] takes an [AsyncLimit] instead of a [SyncLimit]
    /// and therefore allows an `async` callback to be specified for when the cache reaches its limit.
    ///
    /// This function does not block and can be used in async contexts.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// let guard1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_map
    ///     .try_lock_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map
    ///     .try_lock_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub async fn try_lock_async<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: AsyncLimit<
            MapImpl<K, V>,
            V,
            NoopHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
            E,
            F,
            OnEvictFn,
        >,
    ) -> Result<Option<<Self as Lockable<K, V>>::Guard<'a>>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> F,
    {
        LockableMapImpl::try_lock_async(&self.map_impl, key, limit).await
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableHashMap::try_lock_async], please see documentation for that function for more information.
    /// But different to [LockableHashMap::try_lock_async], [LockableHashMap::try_lock_owned_async] works on an `Arc<LockableHashMap>`
    /// instead of a [LockableHashMap] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableHashMap] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableHashMap::try_lock_async].
    ///
    /// This is identical to [LockableHashMap::try_lock_owned], please see documentation for that function for more information.
    /// But different to [LockableHashMap::try_lock_owned], [LockableHashMap::try_lock_owned_async] takes an [AsyncLimit] instead of a [SyncLimit]
    /// and therefore allows an `async` callback to be specified for when the cache reaches its limit.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_map
    ///     .try_lock_owned_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map
    ///     .try_lock_owned_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub async fn try_lock_owned_async<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: AsyncLimit<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>, E, F, OnEvictFn>,
    ) -> Result<Option<<Self as Lockable<K, V>>::OwnedGuard>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> F,
    {
        LockableMapImpl::try_lock_async(Arc::clone(self), key, limit).await
    }

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other tasks from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
    ///
    /// If the lock with this key is currently locked by a different task, then the current tasks `await`s until it becomes available.
    /// Upon returning, the task is the only task with the lock held. A RAII guard is returned to allow scoped unlock
    /// of the lock. When the guard goes out of scope, the lock will be unlocked.
    ///
    /// The `limit` parameter can be used to set a limit on the number of entries in the cache, see the documentation of [AsyncLimit]
    /// for an explanation of how exactly it works.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// let guard1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = lockable_map.async_lock(4).await?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub async fn async_lock<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: AsyncLimit<
            MapImpl<K, V>,
            V,
            NoopHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
            E,
            F,
            OnEvictFn,
        >,
    ) -> Result<<Self as Lockable<K, V>>::Guard<'a>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> F,
    {
        LockableMapImpl::async_lock(&self.map_impl, key, limit).await
    }

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other tasks from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
    ///
    /// This is identical to [LockableHashMap::async_lock], please see documentation for that function for more information.
    /// But different to [LockableHashMap::async_lock], [LockableHashMap::async_lock_owned] works on an `Arc<LockableHashMap>`
    /// instead of a [LockableHashMap] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableHashMap] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableHashMap::async_lock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = lockable_map
    ///     .async_lock_owned(4, AsyncLimit::no_limit())
    ///     .await?;
    /// let guard2 = lockable_map
    ///     .async_lock_owned(5, AsyncLimit::no_limit())
    ///     .await?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = lockable_map.async_lock_owned(4).await?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map
    ///     .async_lock_owned(4, AsyncLimit::no_limit())
    ///     .await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub async fn async_lock_owned<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: AsyncLimit<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>, E, F, OnEvictFn>,
    ) -> Result<<Self as Lockable<K, V>>::OwnedGuard, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> F,
    {
        LockableMapImpl::async_lock(Arc::clone(self), key, limit).await
    }

    /// Consumes the hash map and returns an iterator over all of its entries.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    ///
    /// // Insert two entries
    /// lockable_map
    ///     .async_lock(4, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 4"));
    /// lockable_map
    ///     .async_lock(5, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 5"));
    ///
    /// let entries: Vec<(i64, String)> = lockable_map.into_entries_unordered().collect();
    ///
    /// // `entries` now contains both entries, but in an arbitrary order
    /// assert_eq!(2, entries.len());
    /// assert!(entries.contains(&(4, String::from("Value 4"))));
    /// assert!(entries.contains(&(5, String::from("Value 5"))));
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn into_entries_unordered(self) -> impl Iterator<Item = (K, V)> {
        self.map_impl.into_entries_unordered()
    }

    /// Returns all of the keys that currently have an entry in the map.
    /// Caveat: Currently locked keys are listed even if they don't carry a value.
    ///
    /// This function has a high performance cost because it needs to lock the whole
    /// map to get a consistent snapshot and clone all the keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    ///
    /// // Insert two entries
    /// lockable_map
    ///     .async_lock(4, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 4"));
    /// lockable_map
    ///     .async_lock(5, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 5"));
    /// // Keep a lock on a third entry but don't insert it
    /// let guard = lockable_map.async_lock(6, AsyncLimit::no_limit()).await?;
    ///
    /// let keys: Vec<i64> = lockable_map.keys_with_entries_or_locked();
    ///
    /// // `keys` now contains all three keys
    /// assert_eq!(3, keys.len());
    /// assert!(keys.contains(&4));
    /// assert!(keys.contains(&5));
    /// assert!(keys.contains(&6));
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn keys_with_entries_or_locked(&self) -> Vec<K> {
        self.map_impl.keys_with_entries_or_locked()
    }

    /// Lock all entries of the cache once. The result of this is a [Stream] that will
    /// produce the corresponding lock guards. If items are locked, the [Stream] will
    /// produce them as they become unlocked and can be locked by the stream.
    ///
    /// The returned stream is `async` and therefore may return items much later than
    /// when this function was called, but it only returns an entry if it existed
    /// or was locked at the time this function was called, and still exists when
    /// the stream is returning the entry.
    /// For any entry currently locked by another thread or task while this function
    /// is called, the following rules apply:
    /// - If that thread/task creates the entry => the stream will return it
    /// - If that thread/task removes the entry => the stream will not return it
    /// - If the entry was not pre-existing and that thread/task does not create it => the stream will not return it.
    ///
    /// Examples
    /// -----
    /// ```
    /// use futures::stream::StreamExt;
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    ///
    /// // Insert two entries
    /// lockable_map
    ///     .async_lock(4, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 4"));
    /// lockable_map
    ///     .async_lock(5, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 5"));
    ///
    /// // Lock all entries and add them to an `entries` vector
    /// let mut entries: Vec<(i64, String)> = Vec::new();
    /// let mut stream = lockable_map.lock_all_entries().await;
    /// while let Some(guard) = stream.next().await {
    ///     entries.push((*guard.key(), guard.value().unwrap().clone()));
    /// }
    ///
    /// // `entries` now contains both entries, but in an arbitrary order
    /// assert_eq!(2, entries.len());
    /// assert!(entries.contains(&(4, String::from("Value 4"))));
    /// assert!(entries.contains(&(5, String::from("Value 5"))));
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    pub async fn lock_all_entries(
        &self,
    ) -> impl Stream<Item = <Self as Lockable<K, V>>::Guard<'_>> {
        LockableMapImpl::lock_all_entries(&self.map_impl).await
    }

    /// Lock all entries of the cache once. The result of this is a [Stream] that will
    /// produce the corresponding lock guards. If items are locked, the [Stream] will
    /// produce them as they become unlocked and can be locked by the stream.
    ///
    /// This is identical to [LockableHashMap::lock_all_entries], but but it works on
    /// an `Arc<LockableHashMap>` instead of a [LockableHashMap] and returns a
    /// [Lockable::OwnedGuard] that binds its lifetime to the [LockableHashMap] in that
    /// [Arc]. Such a [Lockable::OwnedGuard] can be more easily moved around or cloned.
    ///
    /// Examples
    /// -----
    /// ```
    /// use futures::stream::StreamExt;
    /// use lockable::{AsyncLimit, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableHashMap::<i64, String>::new());
    ///
    /// // Insert two entries
    /// lockable_map
    ///     .async_lock(4, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 4"));
    /// lockable_map
    ///     .async_lock(5, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 5"));
    ///
    /// // Lock all entries and add them to an `entries` vector
    /// let mut entries: Vec<(i64, String)> = Vec::new();
    /// let mut stream = lockable_map.lock_all_entries_owned().await;
    /// while let Some(guard) = stream.next().await {
    ///     entries.push((*guard.key(), guard.value().unwrap().clone()));
    /// }
    ///
    /// // `entries` now contains both entries, but in an arbitrary order
    /// assert_eq!(2, entries.len());
    /// assert!(entries.contains(&(4, String::from("Value 4"))));
    /// assert!(entries.contains(&(5, String::from("Value 5"))));
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    pub async fn lock_all_entries_owned(
        self: &Arc<Self>,
    ) -> impl Stream<Item = <Self as Lockable<K, V>>::OwnedGuard> {
        LockableMapImpl::lock_all_entries(Arc::clone(self)).await
    }
}

impl<K, V> Default for LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// We implement Borrow<LockableMapImpl> for Arc<LockableHashMap> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableHashMap to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, V, NoopHooks>> for Arc<LockableHashMap<K, V>>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, V, NoopHooks> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instantiate_lockable_tests;

    instantiate_lockable_tests!(lockable_has_limit_support, LockableHashMap<isize, String>);
}
