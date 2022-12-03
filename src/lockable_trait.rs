use std::future::Future;
use std::sync::Arc;

/// A common trait for both [LockableHashMap](crate::LockableHashMap) and [LockableLruCache](crate::LockableLruCache) that offers some common
/// functionalities.
pub trait Lockable<K, V> {
    /// A non-owning guard holding a lock for an entry in a [LockableHashMap](crate::LockableHashMap) or a [LockableLruCache](crate::LockableLruCache).
    /// This guard is created via [Lockable::blocking_lock](crate::Lockable::blocking_lock), [Lockable::async_lock](crate::Lockable::async_lock)
    /// or [Lockable::try_lock](crate::Lockable::try_lock), and its lifetime is bound to the lifetime of the
    /// [LockableHashMap](crate::LockableHashMap)/[LockableLruCache](crate::LockableLruCache).
    ///
    /// See the documentation of [Guard](crate::Guard) for methods.
    ///
    /// Examples:
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let map = LockableHashMap::<usize, String>::new();
    /// let guard: <LockableHashMap<usize, String> as Lockable<usize, String>>::Guard<'_> =
    ///     map.async_lock(1, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    type Guard<'a>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    /// A owning guard holding a lock for an entry in a [LockableHashMap](crate::LockableHashMap) or a [LockableLruCache](crate::LockableLruCache).
    /// This guard is created via [Lockable::blocking_lock_owned](crate::Lockable::blocking_lock_owned), [Lockable::async_lock_owned](crate::Lockable::async_lock_owned)
    /// or [Lockable::try_lock_owned](crate::Lockable::try_lock_owned), or the corresponding [LockableLruCache](crate::LockableLruCache) methods,
    /// and its lifetime is bound to the lifetime of the [LockableHashMap](crate::LockableHashMap)/[LockableLruCache](crate::LockableLruCache) within its [Arc](std::sync::Arc).
    ///
    /// See the documentation of [Guard](crate::Guard) for methods.
    ///
    /// Examples:
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let map = Arc::new(LockableHashMap::<usize, String>::new());
    /// let guard: <LockableHashMap<usize, String> as Lockable<usize, String>>::OwnedGuard =
    ///     map.async_lock_owned(1, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    type OwnedGuard;

    /// TODO Documentation
    type SyncLimit<'a, OnEvictFn, E>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>;

    /// TODO Documentation
    type SyncLimitOwned<OnEvictFn, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

    /// TODO Documentation
    type AsyncLimit<'a, OnEvictFn, E, F>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F;

    /// TODO Documentation
    type AsyncLimitOwned<OnEvictFn, E, F>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F;

    /// Return the number of map entries.
    ///
    /// Corner case: Currently locked keys are counted even if they don't exist in the map.
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    fn num_entries_or_locked(&self) -> usize;

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
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, SyncLimit};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{Lockable, LockableLruCache, SyncLimit};
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
    fn blocking_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::SyncLimit<'a, OnEvictFn, E>,
    ) -> Result<Self::Guard<'a>, E>
    where
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> Result<(), E>;

    /// Lock a lock by key and return a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::blocking_lock], please see documentation for that function for more information.
    /// But different to [Lockable::blocking_lock], [Lockable::blocking_lock_owned] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [Lockable::blocking_lock].
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, SyncLimit};
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
    ///
    /// Examples (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{Lockable, LockableLruCache, SyncLimit};
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
    fn blocking_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::SyncLimitOwned<OnEvictFn, E>,
    ) -> Result<Self::OwnedGuard, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

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
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, SyncLimit};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{Lockable, LockableLruCache, SyncLimit};
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
    fn try_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::SyncLimit<'a, OnEvictFn, E>,
    ) -> Result<Option<Self::Guard<'a>>, E>
    where
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::try_lock], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock], [Lockable::try_lock_owned] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [Lockable::try_lock].
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, SyncLimit};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{Lockable, LockableLruCache, SyncLimit};
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
    fn try_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::SyncLimitOwned<OnEvictFn, E>,
    ) -> Result<Option<Self::OwnedGuard>, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::try_lock], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock], [Lockable::try_lock_async] takes an [AsyncLimit] instead of a [SyncLimit]
    /// and therefore allows an `async` callback to be specified for when the cache reaches its limit.
    ///
    /// This function does not block and can be used in async contexts.
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    async fn try_lock_async<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::AsyncLimit<'a, OnEvictFn, E, F>,
    ) -> Result<Option<Self::Guard<'a>>, E>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::try_lock_async], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock_async], [Lockable::try_lock_owned_async] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [Lockable::try_lock_async].
    ///
    /// This is identical to [Lockable::try_lock_owned], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock_owned], [Lockable::try_lock_owned_async] takes an [AsyncLimit] instead of a [SyncLimit]
    /// and therefore allows an `async` callback to be specified for when the cache reaches its limit.
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    async fn try_lock_owned_async<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::AsyncLimitOwned<OnEvictFn, E, F>,
    ) -> Result<Option<Self::OwnedGuard>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F;

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
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    async fn async_lock<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::AsyncLimit<'a, OnEvictFn, E, F>,
    ) -> Result<Self::Guard<'a>, E>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F;

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other tasks from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
    ///
    /// This is identical to [Lockable::async_lock], please see documentation for that function for more information.
    /// But different to [Lockable::async_lock], [Lockable::async_lock_owned] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [Lockable::async_lock].
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    async fn async_lock_owned<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::AsyncLimitOwned<OnEvictFn, E, F>,
    ) -> Result<Self::OwnedGuard, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F;

    /// Consumes the hash map and returns an iterator over all of its entries.
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    fn into_entries_unordered(self) -> impl Iterator<Item = (K, V)>;

    /// Returns all of the keys that currently have an entry in the map.
    /// Caveat: Currently locked keys are listed even if they don't carry a value.
    ///
    /// This function has a high performance cost because it needs to lock the whole
    /// map to get a consistent snapshot and clone all the keys.
    ///
    /// Example (LockableHashMap)
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
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
    ///
    /// Example (LockableLruCache)
    /// -----
    /// ```
    #[cfg_attr(not(feature = "lru"), doc = "```\n```ignore")]
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    fn keys_with_entries_or_locked(&self) -> Vec<K>;
}
