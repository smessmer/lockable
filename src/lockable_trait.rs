use async_trait::async_trait;
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;

// TODO impl Default for Lockable

TODO Lru Examples will only compile if feature LRU is enabled. We should otherwise probably just not show the example.
TODO Add documentation for the trait async functions to say they only work with Send/Sync types because of async-trait restrictions and that there are non-trait functions available that also work with non-Send/non-Sync types.

/// TODO Documentation
#[async_trait]
pub trait Lockable<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    /// A non-owning guard holding a lock for an entry in a [Lockable].
    /// This guard is created via [Lockable::blocking_lock], [Lockable::async_lock]
    /// or [Lockable::try_lock] and its lifetime is bound to the lifetime
    /// of the [Lockable].
    ///
    /// See the documentation of [Guard] for methods.
    type Guard<'a>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    /// A owning guard holding a lock for an entry in a [LockableLruCache].
    /// This guard is created via [Lockable::blocking_lock_owned], [Lockable::async_lock_owned]
    /// or [Lockable::try_lock_owned] and its lifetime is bound to the lifetime of the [Lockable]
    /// within its [Arc].
    ///
    /// See the documentation of [Guard] for methods.
    type OwnedGuard;

    /// TODO Documentation
    type SyncLimit<'a, OnEvictFn, E>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> Result<(), E>;

    /// TODO Documentation
    type SyncLimitOwned<OnEvictFn, E>
    where
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> Result<(), E>;

    /// TODO Documentation
    type AsyncLimit<'a, OnEvictFn, E, F>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> F;

    /// TODO Documentation
    type AsyncLimitOwned<OnEvictFn, E, F>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> F;

    /// Create a new map with no entries and no locked keys.
    fn new() -> Self;

    /// Return the number of entries.
    ///
    /// Corner case: Currently locked keys are counted even if they don't have a value.
    fn num_entries_or_locked(&self) -> usize;

    /// Lock a key and return a guard with any potential entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// an entry by itself. Map entries can be inserted and removed using [Guard::insert](crate::Guard::insert) and
    /// [Guard::remove](crate::Guard::remove) on the returned entry guard.
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
    /// The `limit` parameter can be used to set a limit on the number of entries in the cache, see the documentation of [SyncLimit](crate::SyncLimit)
    /// for an explanation of how exactly it works.
    ///
    /// Panics
    /// -----
    /// - This function might panic when called if the lock is already held by the current thread.
    /// - This function will also panic when called from an `async` context.
    ///   See documentation of [tokio::sync::Mutex] for details.
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, SyncLimit, LockableLruCache};
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    fn blocking_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::SyncLimit<'a, OnEvictFn, E>,
    ) -> Result<Self::Guard<'_>, E>
    where
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> Result<(), E>;

    /// Lock a lock by key and return a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::blocking_lock], please see documentation for that function for more information.
    /// But different to [Lockable::blocking_lock], [Lockable::blocking_lock_owned] works on an `Arc<Lockable>`
    /// instead of a [Lockable] directly and returns a [Self::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [Lockable::blocking_lock].
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, SyncLimit, LockableLruCache};
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    fn blocking_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::SyncLimitOwned<OnEvictFn, E>,
    ) -> Result<Self::OwnedGuard, E>
    where
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> Result<(), E>;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert](crate::Guard::insert) and
    /// [Guard::remove](crate::Guard::remove) on the returned entry guard.
    ///
    /// If the lock could not be acquired because it is already locked, then [Ok](Ok)([None]) is returned. Otherwise, a RAII guard is returned.
    /// The lock will be unlocked when the guard is dropped.
    ///
    /// This function does not block and can be used from both async and non-async contexts.
    ///
    /// The `limit` parameter can be used to set a limit on the number of entries in the cache, see the documentation of [SyncLimit](crate::SyncLimit)
    /// for an explanation of how exactly it works.
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, SyncLimit, LockableLruCache};
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    fn try_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::SyncLimit<'a, OnEvictFn, E>,
    ) -> Result<Option<Self::Guard<'a>>, E>
    where
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> Result<(), E>;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::try_lock], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock], [Lockable::try_lock_owned] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [Lockable::try_lock].
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, SyncLimit, LockableLruCache};
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
    /// # Ok::<(), anyhow::Error>(())})().unwrap();
    /// ```
    fn try_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::SyncLimitOwned<OnEvictFn, E>,
    ) -> Result<Option<Self::OwnedGuard>, E>
    where
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> Result<(), E>;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [Lockable::try_lock], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock], [Lockable::try_lock_async] takes an [AsyncLimit](crate::AsyncLimit)
    /// instead of a [SyncLimit](crate::SyncLimit) and therefore allows an `async` callback to be specified for when
    /// the number of entries reaches a limit.
    ///
    /// This function does not block and can be used in async contexts.
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, AsyncLimit};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// let guard1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_map.try_lock_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.try_lock_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](cache::LockableLruCache))
    /// ```
    /// use lockable::{Lockable, LockableLruCache, AsyncLimit};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_cache = LockableLruCache::<i64, String>::new();
    /// let guard1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_cache.try_lock_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache.try_lock_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    /// TODO Test this, we're only testing try_lock so far, not try_lock_async
    async fn try_lock_async<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::AsyncLimit<'a, OnEvictFn, E, F>,
    ) -> Result<Option<Self::Guard<'a>>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync;

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for
    /// that key.
    ///
    /// This is identical to [Lockable::try_lock_async], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock_async], [Lockable::try_lock_owned_async] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [HashMapOwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [HashMapOwnedGuard] can be more easily moved around or cloned than the [HashMapGuard] returned by
    /// [Lockable::try_lock_async].
    ///
    /// This is identical to [Lockable::try_lock_owned], please see documentation for that function for more information.
    /// But different to [Lockable::try_lock_owned], [Lockable::try_lock_owned_async] takes an [AsyncLimit](crate::AsyncLimit)
    /// instead of a [SyncLimit](crate::SyncLimit) and therefore allows an `async` callback to be specified for when the cache
    /// reaches its limit.
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, AsyncLimit};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_map.try_lock_owned_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_map.try_lock_owned_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableLruCache, AsyncLimit};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_cache = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = lockable_cache.try_lock_owned_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = lockable_cache.try_lock_owned_async(4, AsyncLimit::no_limit()).await?;
    /// assert!(guard3.is_some());
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    /// TODO Test, we're only testing try_lock_owned so far, not try_lock_owned_async
    async fn try_lock_owned_async<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::AsyncLimitOwned<OnEvictFn, E, F>,
    ) -> Result<Option<Self::OwnedGuard>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync;

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other tasks from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert](crate::Guard::insert) and
    /// [Guard::remove](crate::Guard::remove) on the returned entry guard.
    ///
    /// If the lock with this key is currently locked by a different task, then the current tasks `await`s until it becomes available.
    /// Upon returning, the task is the only task with the lock held. A RAII guard is returned to allow scoped unlock
    /// of the lock. When the guard goes out of scope, the lock will be unlocked.
    ///
    /// The `limit` parameter can be used to set a limit on the number of entries in the cache, see the documentation of
    /// [AsyncLimit](crate::AsyncLimit) for an explanation of how exactly it works.
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, AsyncLimit};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let locakble_map = LockableHashMap::<i64, String>::new();
    /// let guard1 = locakble_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = locakble_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = locakble_map.async_lock(4).await?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = locakble_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableLruCache, AsyncLimit};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let locakble_map = LockableLruCache::<i64, String>::new();
    /// let guard1 = locakble_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = locakble_map.async_lock(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = locakble_map.async_lock(4).await?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = locakble_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    async fn async_lock<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::AsyncLimit<'a, OnEvictFn, E, F>,
    ) -> Result<Self::Guard<'a>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync;

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other tasks from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [Guard::insert] and [Guard::remove] on the returned entry guard.
    ///
    /// This is identical to [Lockable::async_lock], please see documentation for that function for more information.
    /// But different to [Lockable::async_lock], [Lockable::async_lock_owned] works on an `Arc<Lockable>`
    /// instead of a [Lockable] and returns a [Self::OwnedGuard] that binds its lifetime to the [Lockable] in that [Arc].
    /// Such a [Self::OwnedGuard] can be more easily moved around or cloned than the [Self::Guard] returned by [Lockable::async_lock].
    ///
    /// Example ([LockableHashMap](crate::LockableHashMap))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableHashMap, AsyncLimit};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let locakble_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = locakble_map.async_lock_owned(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = locakble_map.async_lock_owned(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = locakble_map.async_lock_owned(4).await?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = locakble_map.async_lock_owned(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    ///
    /// Example ([LockableLruCache](crate::LockableLruCache))
    /// -----
    /// ```
    /// use lockable::{Lockable, LockableLruCache, AsyncLimit};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let locakble_map = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = locakble_map.async_lock_owned(4, AsyncLimit::no_limit()).await?;
    /// let guard2 = locakble_map.async_lock_owned(5, AsyncLimit::no_limit()).await?;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = locakble_map.async_lock_owned(4).await?;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = locakble_map.async_lock_owned(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), anyhow::Error>(())}).unwrap();
    /// ```
    async fn async_lock_owned<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::AsyncLimitOwned<OnEvictFn, E, F>,
    ) -> Result<Self::OwnedGuard, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync;
}
