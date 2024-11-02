use std::future::Future;
use std::hash::Hash;

use super::limit::{AsyncLimit, SyncLimit};
use super::lockable_hash_map::LockableHashMap;
use super::lockable_trait::Lockable;
use super::utils::never::InfallibleUnwrap;

/// A pool of locks where individual locks can be locked/unlocked by key.
/// It initially considers all keys as "unlocked", but they can be locked
/// and if a second thread tries to acquire a lock for the same key, they will have to wait.
///
/// ```
/// use lockable::LockPool;
///
/// let pool = LockPool::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard1 = pool.async_lock(4).await;
/// let guard2 = pool.async_lock(5).await;
///
/// // This next line would cause a deadlock or panic because `4` is already locked on this thread
/// // let guard3 = pool.async_lock(4).await;
///
/// // After dropping the corresponding guard, we can lock it again
/// std::mem::drop(guard1);
/// let guard3 = pool.async_lock(4).await;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// You can use an arbitrary type to index locks by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone] + [Debug].
///
/// ```
/// use lockable::LockPool;
///
/// #[derive(PartialEq, Eq, Hash, Clone, Debug)]
/// struct CustomLockKey(u32);
///
/// let pool = LockPool::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = pool.async_lock(CustomLockKey(4)).await;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// Under the hood, a [LockPool] is a [LockableHashMap] with `()` as a value type, i.e. `LockPool<K>` is just a wrapper
/// around `LockableHashMap<K, ()>` with a simpler API. If you need more complex functionalities, please look at
/// [LockableHashMap].
pub struct LockPool<K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    map: LockableHashMap<K, ()>,
}

impl<K> Lockable<K, ()> for LockPool<K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type Guard<'a> = <LockableHashMap<K, ()> as Lockable<K, ()>>::Guard<'a>
    where
        K: 'a;

    type OwnedGuard = <LockableHashMap<K, ()> as Lockable<K, ()>>::OwnedGuard;

    type SyncLimit<'a, OnEvictFn, E> = <LockableHashMap<K, ()> as Lockable<K, ()>>::SyncLimit<'a, OnEvictFn, E>
    where
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>,
        K: 'a;

    type SyncLimitOwned<OnEvictFn, E> = <LockableHashMap<K, ()> as Lockable<K, ()>>::SyncLimitOwned<OnEvictFn, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

    type AsyncLimit<'a, OnEvictFn, E, F> = <LockableHashMap<K, ()> as Lockable<K, ()>>::AsyncLimit<'a, OnEvictFn, E, F>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F,
        K: 'a;

    type AsyncLimitOwned<OnEvictFn, E, F> = <LockableHashMap<K, ()> as Lockable<K, ()>>::AsyncLimitOwned<OnEvictFn, E, F>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F;
}

impl<K> LockPool<K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    /// Create a new lock pool with no locked keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::LockPool;
    ///
    /// let pool = LockPool::new();
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let guard = pool.async_lock(4).await;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            map: LockableHashMap::new(),
        }
    }

    /// Return the number of locked keys in the pool.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::LockPool;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let pool = LockPool::new();
    ///
    /// // Lock two entries
    /// let guard1 = pool
    ///     .async_lock(4)
    ///     .await;
    /// let guard2 = pool
    ///     .async_lock(5)
    ///     .await;
    ///
    /// // Now we have two locked entries
    /// assert_eq!(2, pool.num_locked());
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn num_locked(&self) -> usize {
        self.map.num_entries_or_locked()
    }

    /// Lock a key and return a guard for it.
    ///
    /// Locking a key prevents any other threads from locking the same key.
    /// If the lock with this key is currently locked by a different thread, then the current thread blocks until it becomes available.
    /// Upon returning, the thread is the only thread with the lock held. A RAII guard is returned to allow scoped unlock
    /// of the lock. When the guard goes out of scope, the lock will be unlocked.
    ///
    /// This function can only be used from non-async contexts and will panic if used from async contexts.
    ///
    /// The exact behavior on locking a lock in the thread which already holds the lock is left unspecified.
    /// However, this function will not return on the second call (it might panic or deadlock, for example).
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
    /// use lockable::LockPool;
    ///
    /// # (|| {
    /// let pool = LockPool::new();
    /// let guard1 = pool.blocking_lock(4);
    /// let guard2 = pool.blocking_lock(5);
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = pool.blocking_lock(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = pool.blocking_lock(4);
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn blocking_lock(&self, key: K) -> <Self as Lockable<K, ()>>::Guard<'_> {
        self.map
            .blocking_lock(key, SyncLimit::no_limit())
            .infallible_unwrap()
    }

    // TOOD Add this
    // /// Lock a key and return a guard for it.
    // ///
    // /// This is identical to [LockPool::blocking_lock], please see documentation for that function for more information.
    // /// But different to [LockPool::blocking_lock], [LockPool::blocking_lock_owned] works on an `Arc<LockPool>`
    // /// instead of a [LockPool] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockPool] in that [Arc].
    // /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockPool::blocking_lock].
    // ///
    // /// Examples
    // /// -----
    // /// ```
    // /// use lockable::LockPool;
    // /// use std::sync::Arc;
    // ///
    // /// # (|| {
    // /// let pool = Arc::new(LockPool::new());
    // /// let guard1 = pool.blocking_lock_owned(4);
    // /// let guard2 = pool.blocking_lock_owned(5);
    // ///
    // /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    // /// // let guard3 = pool.blocking_lock_owned(4);
    // ///
    // /// // After dropping the corresponding guard, we can lock it again
    // /// std::mem::drop(guard1);
    // /// let guard3 = pool.blocking_lock_owned(4);
    // /// # Ok::<(), lockable::Never>(())})().unwrap();
    // /// ```
    // #[inline]
    // pub fn blocking_lock_owned(self: &Arc<Self>, key: K) -> <Self as Lockable<K, ()>>::OwnedGuard {
    //     self.map
    //         .blocking_lock_owned(key, SyncLimit::no_limit())
    //         .infallible_unwrap()
    // }

    /// Attempts to acquire the lock with the given key.
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
    /// Examples
    /// -----
    /// ```
    /// use lockable::LockPool;
    ///
    /// # (|| {
    /// let pool = LockPool::new();
    /// let guard1 = pool.blocking_lock(4);
    /// let guard2 = pool.blocking_lock(5);
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = pool.try_lock(4);
    /// assert!(guard3.is_none());
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = pool.try_lock(4);
    /// assert!(guard3.is_some());
    /// # Ok::<(), lockable::Never>(())})().unwrap();
    /// ```
    #[inline]
    pub fn try_lock(&self, key: K) -> Option<<Self as Lockable<K, ()>>::Guard<'_>> {
        self.map
            .try_lock(key, SyncLimit::no_limit())
            .infallible_unwrap()
    }

    // TODO Add this
    // /// Attempts to acquire the lock with the given key.
    // ///
    // /// This is identical to [LockPool::try_lock], please see documentation for that function for more information.
    // /// But different to [LockPool::try_lock], [LockPool::try_lock_owned] works on an `Arc<LockPool>`
    // /// instead of a [LockPool] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockPool] in that [Arc].
    // /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockPool::try_lock].
    // ///
    // /// Examples
    // /// -----
    // /// ```
    // /// use lockable::LockPool;
    // /// use std::sync::Arc;
    // ///
    // /// # (||{
    // /// let pool = Arc::new(LockPool::new());
    // /// let guard1 = pool.blocking_lock(4);
    // /// let guard2 = pool.blocking_lock(5);
    // ///
    // /// // This next line cannot acquire the lock because `4` is already locked on this thread
    // /// let guard3 = pool.try_lock_owned(4);
    // /// assert!(guard3.is_none());
    // ///
    // /// // After dropping the corresponding guard, we can lock it again
    // /// std::mem::drop(guard1);
    // /// let guard3 = pool.try_lock_owned(4);
    // /// assert!(guard3.is_some());
    // /// # Ok::<(), lockable::Never>(())})().unwrap();
    // /// ```
    // #[inline]
    // pub fn try_lock_owned(
    //     self: &Arc<Self>,
    //     key: K,
    // ) -> Option<<Self as Lockable<K, ()>>::OwnedGuard> {
    //     self.map
    //         .try_lock_owned(key, SyncLimit::no_limit())
    //         .infallible_unwrap()
    // }

    /// Lock a key and return a guard for it.
    ///
    /// Locking a key prevents any other tasks from locking the same key.
    ///
    /// If the lock with this key is currently locked by a different task, then the current tasks `await`s until it becomes available.
    /// Upon returning, the task is the only task with the lock held. A RAII guard is returned to allow scoped unlock
    /// of the lock. When the guard goes out of scope, the lock will be unlocked.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::LockPool;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let pool = LockPool::new();
    /// let guard1 = pool.async_lock(4).await;
    /// let guard2 = pool.async_lock(5).await;
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = pool.async_lock(4).await;
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = pool.async_lock(4).await;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub async fn async_lock(&self, key: K) -> <Self as Lockable<K, ()>>::Guard<'_> {
        self.map
            .async_lock(key, AsyncLimit::no_limit())
            .await
            .infallible_unwrap()
    }

    // TODO Add this
    // /// Lock a key and return a guard for it.
    // ///
    // /// This is identical to [LockPool::async_lock], please see documentation for that function for more information.
    // /// But different to [LockPool::async_lock], [LockPool::async_lock_owned] works on an `Arc<LockPool>`
    // /// instead of a [LockPool] and returns a [Lockable::OwnedGuard] that binds its lifetime to the [LockPool] in that [Arc].
    // /// Such a [Lockable::OwnedGuard] can be more easily moved around or cloned than the [Lockable::Guard] returned by [LockPool::async_lock].
    // ///
    // /// Examples
    // /// -----
    // /// ```
    // /// use lockable::LockPool;
    // /// use std::sync::Arc;
    // ///
    // /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    // /// let pool = Arc::new(LockPool::new());
    // /// let guard1 = pool
    // ///     .async_lock_owned(4)
    // ///     .await;
    // /// let guard2 = pool
    // ///     .async_lock_owned(5)
    // ///     .await;
    // ///
    // /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    // /// // let guard3 = pool.async_lock_owned(4).await;
    // ///
    // /// // After dropping the corresponding guard, we can lock it again
    // /// std::mem::drop(guard1);
    // /// let guard3 = pool
    // ///     .async_lock_owned(4)
    // ///     .await;
    // /// # Ok::<(), lockable::Never>(())}).unwrap();
    // /// ```
    // #[inline]
    // pub async fn async_lock_owned(
    //     self: &Arc<Self>,
    //     key: K,
    // ) -> <Self as Lockable<K, ()>>::OwnedGuard {
    //     self.map
    //         .async_lock_owned(key, AsyncLimit::no_limit())
    //         .await
    //         .infallible_unwrap()
    // }

    /// Returns all of the keys that are currently locked.
    ///
    /// This function has a high performance cost because it needs to lock the whole
    /// map to get a consistent snapshot and clone all the keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::LockPool;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let pool = LockPool::new();
    ///
    /// // Lock two keys
    /// let guard1 = pool
    ///     .async_lock(4)
    ///     .await;
    /// let guard2 = pool
    ///     .async_lock(5)
    ///     .await;
    ///
    /// let keys: Vec<i64> = pool.locked_keys();
    ///
    /// // `keys` now contains both keys
    /// assert_eq!(2, keys.len());
    /// assert!(keys.contains(&4));
    /// assert!(keys.contains(&5));
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn locked_keys(&self) -> Vec<K> {
        self.map.keys_with_entries_or_locked()
    }
}

impl<K> Default for LockPool<K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// TODO Tests for `LockPool`
