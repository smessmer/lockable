use futures::stream::Stream;
use lru::LruCache;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::iter::Rev;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

use super::guard::Guard;
use super::hooks::Hooks;
use super::limit::{AsyncLimit, SyncLimit};
use super::lockable_map_impl::{FromInto, LockableMapImpl};
use super::lockable_trait::Lockable;
use super::map_like::{ArcMutexMapLike, EntryValue};

type MapImpl<K, V> = LruCache<K, Arc<tokio::sync::Mutex<EntryValue<CacheEntry<V>>>>>;

// The ArcMutexMapLike implementation here allows LockableMapImpl to
// work with LruCache as an underlying map
impl<K, V> ArcMutexMapLike for MapImpl<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type K = K;
    type V = CacheEntry<V>;
    type ItemIter<'a> = Rev<lru::Iter<'a, K, Arc<Mutex<EntryValue<CacheEntry<V>>>>>>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        Self::unbounded()
    }

    fn len(&self) -> usize {
        self.iter().len()
    }

    fn get_or_insert_none(&mut self, key: &Self::K) -> &Arc<Mutex<EntryValue<Self::V>>> {
        // TODO Is there a way to only clone the key when the entry doesn't already exist?
        self.get_or_insert(key.clone(), || {
            Arc::new(Mutex::new(EntryValue { value: None }))
        })
    }

    fn get(&mut self, key: &Self::K) -> Option<&Arc<Mutex<EntryValue<Self::V>>>> {
        LruCache::get(self, key)
    }

    fn remove(&mut self, key: &Self::K) -> Option<Arc<Mutex<EntryValue<Self::V>>>> {
        self.pop(key)
    }

    fn iter(&self) -> Self::ItemIter<'_> {
        LruCache::iter(self).rev()
    }
}

// The LRUCache actually stores <K, CacheEntry<V>> instead of <K, V> so that we can
// remember a last_unlocked timestamp for each entry
#[derive(Debug)]
pub struct CacheEntry<V> {
    value: V,
    last_unlocked: Instant,
}

// Borrow and BorrowMut are used to allow Guard to offer an API to read/write V while
// the cache actually stores values as CacheEntry<V>
impl<V> Borrow<V> for CacheEntry<V> {
    fn borrow(&self) -> &V {
        &self.value
    }
}
impl<V> BorrowMut<V> for CacheEntry<V> {
    fn borrow_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

// FromInto is used to allow Guard to offer an API to insert V values while
// the cache actually stores values as CacheEntry<V>
impl<V> FromInto<V> for CacheEntry<V> {
    fn fi_from(value: V) -> CacheEntry<V> {
        CacheEntry {
            value,
            // last_unlocked is now since the entry was just freshly inserted
            last_unlocked: Instant::now(),
        }
    }
    fn fi_into(self) -> V {
        self.value
    }
}

// LruCacheHooks ensure that whenever we unlock an entry, its last_unlocked
// timestamp gets updated
pub struct LruCacheHooks;
impl<V> Hooks<CacheEntry<V>> for LruCacheHooks {
    fn on_unlock(&self, v: Option<&mut CacheEntry<V>>) {
        if let Some(v) = v {
            v.last_unlocked = Instant::now();
        }
    }
}

/// A threadsafe LRU cache where individual keys can be locked/unlocked, even if there is no entry for this key in the cache.
/// It initially considers all keys as "unlocked", but they can be locked and if a second thread tries to acquire a lock
/// for the same key, they will have to wait.
///
/// This class is only available if the `lru` crate feature is enabled.
///
/// ```
/// use lockable::{AsyncLimit, LockableLruCache};
///
/// let lockable_cache: LockableLruCache<i64, String> = LockableLruCache::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let entry1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
/// let entry2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
///
/// // This next line would cause a deadlock or panic because `4` is already locked on this thread
/// // let entry3 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
///
/// // After dropping the corresponding guard, we can lock it again
/// std::mem::drop(entry1);
/// let entry3 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// The guards holding a lock for an entry can be used to insert that entry to the cache, remove
/// it from the cache, or to modify the value of an existing entry.
///
/// ```
/// use lockable::{AsyncLimit, LockableLruCache};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// async fn insert_entry(
///     lockable_cache: &LockableLruCache<i64, String>,
/// ) -> Result<(), lockable::Never> {
///     let mut entry_guard = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.insert(String::from("Hello World"));
///     Ok(())
/// }
///
/// async fn remove_entry(
///     lockable_cache: &LockableLruCache<i64, String>,
/// ) -> Result<(), lockable::Never> {
///     let mut entry_guard = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.remove();
///     Ok(())
/// }
///
/// let lockable_cache: LockableLruCache<i64, String> = LockableLruCache::new();
/// assert_eq!(
///     None,
///     lockable_cache
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// insert_entry(&lockable_cache).await;
/// assert_eq!(
///     Some(&String::from("Hello World")),
///     lockable_cache
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// remove_entry(&lockable_cache).await;
/// assert_eq!(
///     None,
///     lockable_cache
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
///
/// You can use an arbitrary type to index cache entries by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone].
///
/// ```
/// use lockable::{AsyncLimit, LockableLruCache};
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// struct CustomLockKey(u32);
///
/// let lockable_cache: LockableLruCache<CustomLockKey, String> = LockableLruCache::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = lockable_cache
///     .async_lock(CustomLockKey(4), AsyncLimit::no_limit())
///     .await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// Under the hood, a [LockableLruCache] is a [lru::LruCache] of [Mutex](tokio::sync::Mutex)es, with some logic making sure that
/// empty entries can also be locked and that there aren't any race conditions when adding or removing entries.
#[derive(Debug)]
pub struct LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    map_impl: LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
}

impl<K, V> Lockable<K, V> for LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type Guard<'a> = Guard<
    MapImpl<K, V>,
    V,
    LruCacheHooks,
    &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
> where
    K: 'a,
    V: 'a;

    type OwnedGuard = Guard<MapImpl<K, V>, V, LruCacheHooks, Arc<LockableLruCache<K, V>>>;
}

impl<K, V> LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    /// Create a new hash map with no entries and no locked keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    ///
    /// let lockable_map: LockableLruCache<i64, String> = LockableLruCache::new();
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            map_impl: LockableMapImpl::new_with_hooks(LruCacheHooks),
        }
    }

    /// Return the number of cache entries.
    ///
    /// Corner case: Currently locked keys are counted even if they don't have any data in the cache.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableLruCache::<i64, String>::new();
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

    /// Lock a key and return a guard with any potential cache entry for that key.
    /// Any changes to that entry will be persisted in the cache.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a cache entry by itself. Cache entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
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
    /// use lockable::{LockableLruCache, SyncLimit};
    ///
    /// # (||{
    /// let lockable_cache = LockableLruCache::<i64, String>::new();
    /// let guard1 = lockable_cache.blocking_lock(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_cache.blocking_lock(5, SyncLimit::no_limit())?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = lockable_cache.blocking_lock(4, SyncLimit::no_limit())?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache.blocking_lock(4, SyncLimit::no_limit())?;
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn blocking_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: SyncLimit<
            MapImpl<K, V>,
            V,
            LruCacheHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
            E,
            OnEvictFn,
        >,
    ) -> Result<<Self as Lockable<K, V>>::Guard<'a>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> Result<(), E>,
    {
        LockableMapImpl::blocking_lock(&self.map_impl, key, limit)
    }

    /// Lock a lock by key and return a guard with any potential cache entry for that key.
    ///
    /// This is identical to [LockableLruCache::blocking_lock], please see documentation for that function for more information.
    /// But different to [LockableLruCache::blocking_lock], [LockableLruCache::blocking_lock_owned] works on an `Arc<LockableLruCache>`
    /// instead of a [LockableLruCache] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableLruCache::blocking_lock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{LockableLruCache, SyncLimit};
    /// use std::sync::Arc;
    ///
    /// # (||{
    /// let lockable_cache = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = lockable_cache.blocking_lock_owned(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_cache.blocking_lock_owned(5, SyncLimit::no_limit())?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = lockable_cache.blocking_lock_owned(4, SyncLimit::no_limit())?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache.blocking_lock_owned(4, SyncLimit::no_limit())?;
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn blocking_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: SyncLimit<
            MapImpl<K, V>,
            V,
            LruCacheHooks,
            Arc<LockableLruCache<K, V>>,
            E,
            OnEvictFn,
        >,
    ) -> Result<<Self as Lockable<K, V>>::OwnedGuard, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> Result<(), E>,
    {
        LockableMapImpl::blocking_lock(Arc::clone(self), key, limit)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential cache entry for that key.
    /// Any changes to that entry will be persisted in the cache.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a cache entry by itself. Cache entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
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
    /// use lockable::{LockableLruCache, SyncLimit};
    ///
    /// # (||{
    /// let lockable_cache: LockableLruCache<i64, String> = LockableLruCache::new();
    /// let guard1 = lockable_cache.blocking_lock(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_cache.blocking_lock(5, SyncLimit::no_limit())?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_cache.try_lock(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache.try_lock(4, SyncLimit::no_limit())?;
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
            LruCacheHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
            E,
            OnEvictFn,
        >,
    ) -> Result<Option<<Self as Lockable<K, V>>::Guard<'a>>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> Result<(), E>,
    {
        LockableMapImpl::try_lock(&self.map_impl, key, limit)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential cache entry for that key.
    ///
    /// This is identical to [LockableLruCache::blocking_lock], please see documentation for that function for more information.
    /// But different to [LockableLruCache::blocking_lock], [LockableLruCache::blocking_lock_owned] works on an `Arc<LockableLruCache>`
    /// instead of a [LockableLruCache] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableLruCache::blocking_lock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{LockableLruCache, SyncLimit};
    /// use std::sync::Arc;
    ///
    /// # (||{
    /// let lockable_cache = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = lockable_cache.blocking_lock(4, SyncLimit::no_limit())?;
    /// let guard2 = lockable_cache.blocking_lock(5, SyncLimit::no_limit())?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_cache.try_lock_owned(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache.try_lock_owned(4, SyncLimit::no_limit())?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn try_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: SyncLimit<
            MapImpl<K, V>,
            V,
            LruCacheHooks,
            Arc<LockableLruCache<K, V>>,
            E,
            OnEvictFn,
        >,
    ) -> Result<Option<<Self as Lockable<K, V>>::OwnedGuard>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> Result<(), E>,
    {
        LockableMapImpl::try_lock(Arc::clone(self), key, limit)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableLruCache::try_lock], please see documentation for that function for more information.
    /// But different to [LockableLruCache::try_lock], [LockableLruCache::try_lock_async] takes an [AsyncLimit] instead of a [SyncLimit]
    /// and therefore allows an `async` callback to be specified for when the cache reaches its limit.
    ///
    /// This function does not block and can be used in async contexts.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_cache = LockableLruCache::<i64, String>::new();
    /// let guard1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_cache
    ///     .try_lock_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache
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
            LruCacheHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
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
    /// This is identical to [LockableLruCache::try_lock_async], please see documentation for that function for more information.
    /// But different to [LockableLruCache::try_lock_async], [LockableLruCache::try_lock_owned_async] works on an `Arc<LockableLruCache>`
    /// instead of a [LockableLruCache] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableLruCache::try_lock_async].
    ///
    /// This is identical to [LockableLruCache::try_lock_owned], please see documentation for that function for more information.
    /// But different to [LockableLruCache::try_lock_owned], [LockableLruCache::try_lock_owned_async] takes an [AsyncLimit] instead of a [SyncLimit]
    /// and therefore allows an `async` callback to be specified for when the cache reaches its limit.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_cache = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_cache
    ///     .try_lock_owned_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache
    ///     .try_lock_owned_async(4, AsyncLimit::no_limit())
    ///     .await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub async fn try_lock_owned_async<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: AsyncLimit<
            MapImpl<K, V>,
            V,
            LruCacheHooks,
            Arc<LockableLruCache<K, V>>,
            E,
            F,
            OnEvictFn,
        >,
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
    /// use lockable::{AsyncLimit, LockableLruCache};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableLruCache::<i64, String>::new();
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
            LruCacheHooks,
            &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
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
    /// This is identical to [LockableLruCache::async_lock], please see documentation for that function for more information.
    /// But different to [LockableLruCache::async_lock], [LockableLruCache::async_lock_owned] works on an `Arc<LockableLruCache>`
    /// instead of a [LockableLruCache] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockableLruCache::async_lock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableLruCache::<i64, String>::new());
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
        limit: AsyncLimit<
            MapImpl<K, V>,
            V,
            LruCacheHooks,
            Arc<LockableLruCache<K, V>>,
            E,
            F,
            OnEvictFn,
        >,
    ) -> Result<<Self as Lockable<K, V>>::OwnedGuard, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::OwnedGuard>) -> F,
    {
        LockableMapImpl::async_lock(Arc::clone(self), key, limit).await
    }

    /// Consumes the cache and returns an iterator over all of its entries.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableLruCache::<i64, String>::new();
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
        self.map_impl
            .into_entries_unordered()
            .map(|(k, v)| (k, v.value))
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
    /// use lockable::{AsyncLimit, LockableLruCache};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableLruCache::<i64, String>::new();
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
    /// use lockable::{AsyncLimit, LockableLruCache};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableLruCache::<i64, String>::new();
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
    /// This is identical to [LockableLruCache::lock_all_entries], but it works on
    /// an `Arc<LockableLruCache>` instead of a [LockableLruCache] and returns a
    /// [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that
    /// [Arc]. Such a [Lockable::OwnedGuard] can be more easily moved around or cloned.
    ///
    /// Examples
    /// -----
    /// ```
    /// use futures::stream::StreamExt;
    /// use lockable::{AsyncLimit, LockableLruCache};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableLruCache::<i64, String>::new());
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

    /// Lock all entries that are currently unlocked and that were unlocked for at least
    /// the given `duration`. This follows the LRU nature of the cache.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    /// use tokio::time::{self, Duration};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableLruCache::<i64, String>::new();
    /// lockable_map
    ///     .async_lock(1, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 1"));
    /// lockable_map
    ///     .async_lock(2, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 2"));
    ///
    /// time::sleep(Duration::from_secs(1)).await;
    ///
    /// // Lock and unlock entry 1
    /// lockable_map.async_lock(1, AsyncLimit::no_limit()).await?;
    ///
    /// // Only entry 2 was unlocked more than half a second ago
    ///
    /// let unlocked_for_at_least_half_a_sec: Vec<(i64, String)> = lockable_map
    ///     .lock_entries_unlocked_for_at_least(Duration::from_millis(500))
    ///     .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
    ///     .collect();
    /// assert_eq!(
    ///     vec![(2, String::from("Value 2"))],
    ///     unlocked_for_at_least_half_a_sec
    /// );
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    ///
    /// TODO Test whether the returned iterator keeps a lock on the whole map and if yes,
    ///      try to fix that or at least document it.
    pub fn lock_entries_unlocked_for_at_least(
        &self,
        duration: Duration,
    ) -> impl Iterator<Item = <Self as Lockable<K, V>>::Guard<'_>> {
        Self::_lock_entries_unlocked_for_at_least(&self.map_impl, duration)
    }

    /// Lock all entries that are currently unlocked and that were unlocked for at least
    /// the given `duration`. This follows the LRU nature of the cache.
    ///
    /// This is identical to [LockableLruCache::lock_entries_unlocked_for_at_least], but it works on
    /// an `Arc<LockableLruCache>` instead of a [LockableLruCache] and returns a
    /// [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that
    /// [Arc]. Such a [Lockable::OwnedGuard] can be more easily moved around or cloned.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableLruCache};
    /// use std::sync::Arc;
    /// use tokio::time::{self, Duration};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableLruCache::<i64, String>::new());
    /// lockable_map
    ///     .async_lock(1, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 1"));
    /// lockable_map
    ///     .async_lock(2, AsyncLimit::no_limit())
    ///     .await?
    ///     .insert(String::from("Value 2"));
    ///
    /// time::sleep(Duration::from_secs(1)).await;
    ///
    /// // Lock and unlock entry 1
    /// lockable_map.async_lock(1, AsyncLimit::no_limit()).await?;
    ///
    /// // Only entry 2 was unlocked more than half a second ago
    ///
    /// let unlocked_for_at_least_half_a_sec: Vec<(i64, String)> = lockable_map
    ///     .lock_entries_unlocked_for_at_least_owned(Duration::from_millis(500))
    ///     .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
    ///     .collect();
    /// assert_eq!(
    ///     vec![(2, String::from("Value 2"))],
    ///     unlocked_for_at_least_half_a_sec
    /// );
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    ///
    /// TODO Test
    pub fn lock_entries_unlocked_for_at_least_owned(
        self: &Arc<Self>,
        duration: Duration,
    ) -> impl Iterator<Item = <Self as Lockable<K, V>>::OwnedGuard> {
        Self::_lock_entries_unlocked_for_at_least(Arc::clone(self), duration)
    }

    fn _lock_entries_unlocked_for_at_least<
        S: Borrow<LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>> + Clone,
    >(
        this: S,
        duration: Duration,
    ) -> impl Iterator<Item = Guard<MapImpl<K, V>, V, LruCacheHooks, S>> {
        // TODO Since entries should be LRU ordered, we don't need to iterate over all of them, just until one is new enough.
        let now = Instant::now();
        let cutoff = now - duration;
        LockableMapImpl::lock_all_unlocked(this).filter(move |entry| {
            if let Some(entry) = entry.value_raw() {
                entry.last_unlocked <= cutoff
            } else {
                false
            }
        })
    }
}

impl<K, V> Default for LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// We implement Borrow<LockableMapImpl> for Arc<LockableLruCache<K, V>> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableLruCache to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>> for Arc<LockableLruCache<K, V>>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instantiate_lockable_tests;

    instantiate_lockable_tests!(LockableLruCache);
}
