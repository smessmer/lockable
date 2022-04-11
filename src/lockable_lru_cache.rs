use lru::LruCache;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::error::TryLockError;
use super::guard::GuardImpl;
use super::hooks::NoopHooks;
use super::lockable_map_impl::LockableMapImpl;
use super::map_like::{ArcMutexMapLike, EntryValue};

// TODO LockableLruCache likely wants to use Hooks and last_update_instant

// The ArcMutexMapLike implementation here allows LockableMapImpl to
// work with LruCache as an underlying map
impl<K, V> ArcMutexMapLike for LruCache<K, Arc<Mutex<EntryValue<V>>>>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    type K = K;
    type V = V;

    fn new() -> Self {
        Self::unbounded()
    }

    fn len(&self) -> usize {
        self.into_iter().len()
    }

    fn get_or_insert_none(&mut self, key: &Self::K) -> &Arc<Mutex<EntryValue<Self::V>>> {
        // TODO Is there a way to only clone the key when the entry doesn't already exist?
        self.get_or_insert(key.clone(), || {
            Arc::new(Mutex::new(EntryValue { value: None }))
        })
        .expect("Cache capacity is zero. This can't happen since we created an unbounded cache")
    }

    fn get(&mut self, key: &Self::K) -> Option<&Arc<Mutex<EntryValue<Self::V>>>> {
        LruCache::get(self, key)
    }

    fn remove(&mut self, key: &Self::K) -> Option<Arc<Mutex<EntryValue<Self::V>>>> {
        self.pop(key)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (&Self::K, &Arc<Mutex<EntryValue<Self::V>>>)> + '_> {
        Box::new(LruCache::iter(self))
    }
}

type MapImpl<K, V> = LruCache<K, Arc<tokio::sync::Mutex<EntryValue<V>>>>;

/// A threadsafe LRU cache where individual keys can be locked/unlocked, even if there is no entry for this key in the cache.
/// It initially considers all keys as "unlocked", but they can be locked
/// and if a second thread tries to acquire a lock for the same key, they will have to wait.
///
/// ```
/// use lockable::LockableLruCache;
///
/// let cache: LockableLruCache<i64, String> = LockableLruCache::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let entry1 = cache.async_lock(4).await;
/// let entry2 = cache.async_lock(5).await;
///
/// // This next line would cause a deadlock or panic because `4` is already locked on this thread
/// // let entry3 = cache.async_lock(4).await;
///
/// // After dropping the corresponding guard, we can lock it again
/// std::mem::drop(entry1);
/// let entry3 = cache.async_lock(4).await;
/// # });
/// ```
///
/// The guards holding a lock for an entry can be used to insert that entry to the cache, remove it from the cache, or to modify
/// the value of an existing entry.
///
/// ```
/// use lockable::LockableLruCache;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// async fn insert_entry(cache: &LockableLruCache<i64, String>) {
///     let mut entry = cache.async_lock(4).await;
///     entry.insert(String::from("Hello World"));
/// }
///
/// async fn remove_entry(cache: &LockableLruCache<i64, String>) {
///     let mut entry = cache.async_lock(4).await;
///     entry.remove();
/// }
///
/// let cache: LockableLruCache<i64, String> = LockableLruCache::new();
/// assert_eq!(None, cache.async_lock(4).await.value());
/// insert_entry(&cache).await;
/// assert_eq!(Some(&String::from("Hello World")), cache.async_lock(4).await.value());
/// remove_entry(&cache).await;
/// assert_eq!(None, cache.async_lock(4).await.value());
/// # });
/// ```
///
///
/// You can use an arbitrary type to index cache entries by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone] + [Debug].
///
/// ```
/// use lockable::LockableLruCache;
///
/// #[derive(PartialEq, Eq, Hash, Clone, Debug)]
/// struct CustomLockKey(u32);
///
/// let cache: LockableLruCache<CustomLockKey, String> = LockableLruCache::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = cache.async_lock(CustomLockKey(4)).await;
/// # });
/// ```
///
/// Under the hood, a [LockableLruCache] is a [lru::LruCache] of [Mutex](tokio::sync::Mutex)es, with some logic making sure there aren't any race conditions when adding or removing entries.
#[derive(Debug)]
pub struct LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    map_impl: LockableMapImpl<MapImpl<K, V>, NoopHooks>,
}

impl<K, V> LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    /// Create a new hash map with no entries and no locked keys.
    #[inline]
    pub fn new() -> Self {
        Self {
            map_impl: LockableMapImpl::new(),
        }
    }

    /// Return the number of cache entries.
    ///
    /// Corner case: Currently locked keys are counted even if they don't have any data in the cache.
    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self.map_impl.num_entries_or_locked()
    }

    /// Lock a key and return a guard with any potential cache entry for that key.
    /// Any changes to that entry will be persisted in the cache.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a cache entry by itself. Cache entries can be inserted and removed using [LruGuard::insert] and [LruGuard::remove] on the returned entry guard.
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
    /// Panics
    /// -----
    /// - This function might panic when called if the lock is already held by the current thread.
    /// - This function will also panic when called from an `async` context.
    ///   See documentation of [tokio::sync::Mutex] for details.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::LockableLruCache;
    ///
    /// let cache = LockableLruCache::<i64, String>::new();
    /// let guard1 = cache.blocking_lock(4);
    /// let guard2 = cache.blocking_lock(5);
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = cache.blocking_lock(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = cache.blocking_lock(4);
    /// ```
    #[inline]
    pub fn blocking_lock(&self, key: K) -> LruGuard<'_, K, V> {
        LockableMapImpl::blocking_lock(&self.map_impl, key)
    }

    /// Lock a lock by key and return a guard with any potential cache entry for that key.
    ///
    /// This is identical to [LockableLruCache::blocking_lock], but it works on an `Arc<LockableLruCache>` instead of a [LockableLruCache] and
    /// returns a [LruOwnedGuard] that binds its lifetime to the [LockableLruCache] in that [Arc]. Such a [LruOwnedGuard] can be more
    /// easily moved around or cloned.
    ///
    /// This function can be used from non-async contexts but will panic if used from async contexts.
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
    /// use lockable::LockableLruCache;
    /// use std::sync::Arc;
    ///
    /// let cache = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = cache.blocking_lock_owned(4);
    /// let guard2 = cache.blocking_lock_owned(5);
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = cache.blocking_lock_owned(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = cache.blocking_lock_owned(4);
    /// ```
    #[inline]
    pub fn blocking_lock_owned(self: &Arc<Self>, key: K) -> LruOwnedGuard<K, V> {
        LockableMapImpl::blocking_lock(Arc::clone(self), key)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential cache entry for that key.
    /// Any changes to that entry will be persisted in the cache.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a cache entry by itself. Cache entries can be inserted and removed using [LruGuard::insert] and [LruGuard::remove] on the returned entry guard.
    ///
    /// If the lock could not be acquired at this time, then [Err] is returned. Otherwise, a RAII guard is returned.
    /// The lock will be unlocked when the guard is dropped.
    ///
    /// This function does not block and can be used from both async and non-async contexts.
    ///
    /// Errors
    /// -----
    /// - If the lock could not be acquired because it is already locked, then this call will return [TryLockError::WouldBlock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{TryLockError, LockableLruCache};
    ///
    /// let cache: LockableLruCache<i64, String> = LockableLruCache::new();
    /// let guard1 = cache.blocking_lock(4);
    /// let guard2 = cache.blocking_lock(5);
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = cache.try_lock(4);
    /// assert!(matches!(guard3.unwrap_err(), TryLockError::WouldBlock));
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = cache.try_lock(4);
    /// assert!(guard3.is_ok());
    /// ```
    #[inline]
    pub fn try_lock(&self, key: K) -> Result<LruGuard<'_, K, V>, TryLockError> {
        LockableMapImpl::try_lock(&self.map_impl, key)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential cache entry for that key.
    ///
    /// This is identical to [LockableLruCache::try_lock], but it works on an `Arc<LockableLruCache>` instead of a [LockableLruCache] and
    /// returns an [LruOwnedGuard] that binds its lifetime to the [LockableLruCache] in that [Arc]. Such a [LruOwnedGuard] can be more
    /// easily moved around or cloned.
    ///
    /// This function does not block and can be used in both async and non-async contexts.
    ///
    /// Errors
    /// -----
    /// - If the lock could not be acquired because it is already locked, then this call will return [TryLockError::WouldBlock].
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{TryLockError, LockableLruCache};
    /// use std::sync::Arc;
    ///
    /// let pool = Arc::new(LockableLruCache::<i64, String>::new());
    /// let guard1 = pool.blocking_lock(4);
    /// let guard2 = pool.blocking_lock(5);
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = pool.try_lock_owned(4);
    /// assert!(matches!(guard3.unwrap_err(), TryLockError::WouldBlock));
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = pool.try_lock(4);
    /// assert!(guard3.is_ok());
    /// ```
    #[inline]
    pub fn try_lock_owned(self: &Arc<Self>, key: K) -> Result<LruOwnedGuard<K, V>, TryLockError> {
        LockableMapImpl::try_lock(Arc::clone(self), key)
    }

    /// TODO Docs
    #[inline]
    pub async fn async_lock(&self, key: K) -> LruGuard<'_, K, V> {
        LockableMapImpl::async_lock(&self.map_impl, key).await
    }

    /// TODO Docs
    #[inline]
    pub async fn async_lock_owned(self: &Arc<Self>, key: K) -> LruOwnedGuard<K, V> {
        LockableMapImpl::async_lock(Arc::clone(self), key).await
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn into_entries_unordered(self) -> impl Iterator<Item = (K, V)> {
        self.map_impl.into_entries_unordered()
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn keys(&self) -> Vec<K> {
        self.map_impl.keys()
    }
}

/// A non-owning guard holding a lock for an entry in a [LockableLruCache].
/// This guard is created via [LockableLruCache::blocking_lock], [LockableLruCache::async_lock]
/// or [LockableLruCache::try_lock] and its lifetime is bound to the lifetime
/// of the [LockableLruCache].
/// 
/// See the documentation of [GuardImpl] for methods.
pub type LruGuard<'a, K, V> =
    GuardImpl<MapImpl<K, V>, NoopHooks, &'a LockableMapImpl<MapImpl<K, V>, NoopHooks>>;

/// A owning guard holding a lock for an entry in a [LockableLruCache].
/// This guard is created via [LockableLruCache::blocking_lock_owned], [LockableLruCache::async_lock_owned]
/// or [LockableLruCache::try_lock_owned] and its lifetime is bound to the lifetime of the [LockableLruCache]
/// within its [Arc].
/// 
/// See the documentation of [GuardImpl] for methods.
pub type LruOwnedGuard<K, V> = GuardImpl<MapImpl<K, V>, NoopHooks, Arc<LockableLruCache<K, V>>>;

// We implement Borrow<LockableMapImpl> for Arc<LockableLruCache> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableLruCache to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, NoopHooks>> for Arc<LockableLruCache<K, V>>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, NoopHooks> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::super::error::TryLockError;
    use super::LockableLruCache;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    // TODO Add a test adding multiple entries and making sure all locking functions can read them
    // TODO Add tests checking that the async_lock, lock_owned, lock methods all block each other. For lock and lock_owned that can probably go into common tests.rs

    // Launch a thread that
    // 1. locks the given key
    // 2. once it has the lock, increments a counter
    // 3. then waits until a barrier is released before it releases the lock
    fn launch_thread_blocking_lock(
        pool: &Arc<LockableLruCache<isize, String>>,
        key: isize,
        counter: &Arc<AtomicU32>,
        barrier: Option<&Arc<Mutex<()>>>,
    ) -> JoinHandle<()> {
        let pool = Arc::clone(pool);
        let counter = Arc::clone(counter);
        let barrier = barrier.map(Arc::clone);
        thread::spawn(move || {
            let _guard = pool.blocking_lock(key);
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = barrier {
                let _barrier = barrier.lock().unwrap();
            }
        })
    }

    fn launch_thread_blocking_lock_owned(
        pool: &Arc<LockableLruCache<isize, String>>,
        key: isize,
        counter: &Arc<AtomicU32>,
        barrier: Option<&Arc<Mutex<()>>>,
    ) -> JoinHandle<()> {
        let pool = Arc::clone(pool);
        let counter = Arc::clone(counter);
        let barrier = barrier.map(Arc::clone);
        thread::spawn(move || {
            let _guard = pool.blocking_lock_owned(key);
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = barrier {
                let _barrier = barrier.lock().unwrap();
            }
        })
    }

    fn launch_thread_try_lock(
        pool: &Arc<LockableLruCache<isize, String>>,
        key: isize,
        counter: &Arc<AtomicU32>,
        barrier: Option<&Arc<Mutex<()>>>,
    ) -> JoinHandle<()> {
        let pool = Arc::clone(pool);
        let counter = Arc::clone(counter);
        let barrier = barrier.map(Arc::clone);
        thread::spawn(move || {
            let _guard = loop {
                match pool.try_lock(key) {
                    Err(_) =>
                    /* Continue loop */
                    {
                        ()
                    }
                    Ok(guard) => break guard,
                }
            };
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = barrier {
                let _barrier = barrier.lock().unwrap();
            }
        })
    }

    fn launch_thread_try_lock_owned(
        pool: &Arc<LockableLruCache<isize, String>>,
        key: isize,
        counter: &Arc<AtomicU32>,
        barrier: Option<&Arc<Mutex<()>>>,
    ) -> JoinHandle<()> {
        let pool = Arc::clone(pool);
        let counter = Arc::clone(counter);
        let barrier = barrier.map(Arc::clone);
        thread::spawn(move || {
            let _guard = loop {
                match pool.try_lock_owned(key) {
                    Err(_) =>
                    /* Continue loop */
                    {
                        ()
                    }
                    Ok(guard) => break guard,
                }
            };
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = barrier {
                let _barrier = barrier.lock().unwrap();
            }
        })
    }

    fn launch_thread_async_lock(
        pool: &Arc<LockableLruCache<isize, String>>,
        key: isize,
        counter: &Arc<AtomicU32>,
        barrier: Option<&Arc<Mutex<()>>>,
    ) -> JoinHandle<()> {
        let pool = Arc::clone(pool);
        let counter = Arc::clone(counter);
        let barrier = barrier.map(Arc::clone);
        thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let _guard = runtime.block_on(pool.async_lock(key));
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = barrier {
                let _barrier = barrier.lock().unwrap();
            }
        })
    }

    fn launch_thread_async_lock_owned(
        pool: &Arc<LockableLruCache<isize, String>>,
        key: isize,
        counter: &Arc<AtomicU32>,
        barrier: Option<&Arc<Mutex<()>>>,
    ) -> JoinHandle<()> {
        let pool = Arc::clone(pool);
        let counter = Arc::clone(counter);
        let barrier = barrier.map(Arc::clone);
        thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let _guard = runtime.block_on(pool.async_lock_owned(key));
            counter.fetch_add(1, Ordering::SeqCst);
            if let Some(barrier) = barrier {
                let _barrier = barrier.lock().unwrap();
            }
        })
    }

    #[tokio::test]
    #[should_panic(
        expected = "Cannot start a runtime from within a runtime. This happens because a function (like `block_on`) attempted to block the current thread while the thread is being used to drive asynchronous tasks."
    )]
    async fn blocking_lock_from_async_context_with_sync_api() {
        let p = LockableLruCache::<isize, String>::new();
        let _ = p.blocking_lock(3);
    }

    #[tokio::test]
    #[should_panic(
        expected = "Cannot start a runtime from within a runtime. This happens because a function (like `block_on`) attempted to block the current thread while the thread is being used to drive asynchronous tasks."
    )]
    async fn blocking_lock_owned_from_async_context_with_sync_api() {
        let p = Arc::new(LockableLruCache::<isize, String>::new());
        let _ = p.blocking_lock_owned(3);
    }

    mod simple {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.async_lock(4).await;
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.async_lock_owned(4).await;
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.blocking_lock(4);
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.blocking_lock_owned(4);
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.try_lock(4).unwrap();
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.try_lock_owned(4).unwrap();
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }
    }

    mod try_lock {
        use super::*;

        #[test]
        fn try_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.blocking_lock(5);

            let error = pool.try_lock(5).unwrap_err();
            assert!(matches!(error, TryLockError::WouldBlock));

            // Check that we can stil lock other locks while the child is waiting
            {
                let _g = pool.try_lock(4).unwrap();
            }

            // Now free the lock so the we can get it again
            std::mem::drop(guard);

            // And check that we can get it again
            {
                let _g = pool.try_lock(5).unwrap();
            }

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.blocking_lock_owned(5);

            let error = pool.try_lock_owned(5).unwrap_err();
            assert!(matches!(error, TryLockError::WouldBlock));

            // Check that we can stil lock other locks while the child is waiting
            {
                let _g = pool.try_lock_owned(4).unwrap();
            }

            // Now free the lock so the we can get it again
            std::mem::drop(guard);

            // And check that we can get it again
            {
                let _g = pool.try_lock_owned(5).unwrap();
            }

            assert_eq!(0, pool.num_entries_or_locked());
        }
    }

    mod adding_cache_entries {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let mut guard = pool.async_lock(4).await;
            guard.insert(String::from("Cache Entry Value"));
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(1, pool.num_entries_or_locked());
            assert_eq!(
                pool.async_lock(4).await.value(),
                Some(&String::from("Cache Entry Value"))
            );
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let mut guard = pool.async_lock_owned(4).await;
            guard.insert(String::from("Cache Entry Value"));
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(1, pool.num_entries_or_locked());
            assert_eq!(
                pool.async_lock_owned(4).await.value(),
                Some(&String::from("Cache Entry Value"))
            );
        }

        #[test]
        fn blocking_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let mut guard = pool.blocking_lock(4);
            guard.insert(String::from("Cache Entry Value"));
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(1, pool.num_entries_or_locked());
            assert_eq!(
                pool.blocking_lock(4).value(),
                Some(&String::from("Cache Entry Value"))
            );
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let mut guard = pool.blocking_lock_owned(4);
            guard.insert(String::from("Cache Entry Value"));
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(1, pool.num_entries_or_locked());
            assert_eq!(
                pool.blocking_lock_owned(4).value(),
                Some(&String::from("Cache Entry Value"))
            );
        }

        #[test]
        fn try_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let mut guard = pool.try_lock(4).unwrap();
            guard.insert(String::from("Cache Entry Value"));
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(1, pool.num_entries_or_locked());
            assert_eq!(
                pool.try_lock(4).unwrap().value(),
                Some(&String::from("Cache Entry Value"))
            );
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let mut guard = pool.try_lock_owned(4).unwrap();
            guard.insert(String::from("Cache Entry Value"));
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(1, pool.num_entries_or_locked());
            assert_eq!(
                pool.try_lock_owned(4).unwrap().value(),
                Some(&String::from("Cache Entry Value"))
            );
        }
    }

    mod removing_cache_entries {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            pool.async_lock(4)
                .await
                .insert(String::from("Cache Entry Value"));

            assert_eq!(1, pool.num_entries_or_locked());
            let mut guard = pool.async_lock(4).await;
            guard.remove();
            std::mem::drop(guard);

            assert_eq!(0, pool.num_entries_or_locked());
            assert_eq!(pool.async_lock(4).await.value(), None);
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            pool.async_lock_owned(4)
                .await
                .insert(String::from("Cache Entry Value"));

            assert_eq!(1, pool.num_entries_or_locked());
            let mut guard = pool.async_lock_owned(4).await;
            guard.remove();
            std::mem::drop(guard);

            assert_eq!(0, pool.num_entries_or_locked());
            assert_eq!(pool.async_lock_owned(4).await.value(), None);
        }

        #[test]
        fn blocking_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            pool.blocking_lock(4)
                .insert(String::from("Cache Entry Value"));

            assert_eq!(1, pool.num_entries_or_locked());
            let mut guard = pool.blocking_lock(4);
            guard.remove();
            std::mem::drop(guard);

            assert_eq!(0, pool.num_entries_or_locked());
            assert_eq!(pool.blocking_lock(4).value(), None);
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            pool.blocking_lock_owned(4)
                .insert(String::from("Cache Entry Value"));

            assert_eq!(1, pool.num_entries_or_locked());
            let mut guard = pool.blocking_lock_owned(4);
            guard.remove();
            std::mem::drop(guard);

            assert_eq!(0, pool.num_entries_or_locked());
            assert_eq!(pool.blocking_lock_owned(4).value(), None);
        }

        #[test]
        fn try_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            pool.try_lock(4)
                .unwrap()
                .insert(String::from("Cache Entry Value"));

            assert_eq!(1, pool.num_entries_or_locked());
            let mut guard = pool.try_lock(4).unwrap();
            guard.remove();
            std::mem::drop(guard);

            assert_eq!(0, pool.num_entries_or_locked());
            assert_eq!(pool.try_lock(4).unwrap().value(), None);
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            pool.try_lock_owned(4)
                .unwrap()
                .insert(String::from("Cache Entry Value"));

            assert_eq!(1, pool.num_entries_or_locked());
            let mut guard = pool.try_lock_owned(4).unwrap();
            guard.remove();
            std::mem::drop(guard);

            assert_eq!(0, pool.num_entries_or_locked());
            assert_eq!(pool.try_lock_owned(4).unwrap().value(), None);
        }
    }

    mod multi {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard1 = pool.async_lock(1).await;
            assert!(guard1.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            let guard2 = pool.async_lock(2).await;
            assert!(guard2.value().is_none());
            assert_eq!(2, pool.num_entries_or_locked());
            let guard3 = pool.async_lock(3).await;
            assert!(guard3.value().is_none());
            assert_eq!(3, pool.num_entries_or_locked());

            std::mem::drop(guard2);
            assert_eq!(2, pool.num_entries_or_locked());
            std::mem::drop(guard1);
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard3);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard1 = pool.async_lock_owned(1).await;
            assert!(guard1.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            let guard2 = pool.async_lock_owned(2).await;
            assert!(guard2.value().is_none());
            assert_eq!(2, pool.num_entries_or_locked());
            let guard3 = pool.async_lock_owned(3).await;
            assert!(guard3.value().is_none());
            assert_eq!(3, pool.num_entries_or_locked());

            std::mem::drop(guard2);
            assert_eq!(2, pool.num_entries_or_locked());
            std::mem::drop(guard1);
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard3);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard1 = pool.blocking_lock(1);
            assert!(guard1.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            let guard2 = pool.blocking_lock(2);
            assert!(guard2.value().is_none());
            assert_eq!(2, pool.num_entries_or_locked());
            let guard3 = pool.blocking_lock(3);
            assert!(guard3.value().is_none());
            assert_eq!(3, pool.num_entries_or_locked());

            std::mem::drop(guard2);
            assert_eq!(2, pool.num_entries_or_locked());
            std::mem::drop(guard1);
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard3);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard1 = pool.blocking_lock_owned(1);
            assert!(guard1.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            let guard2 = pool.blocking_lock_owned(2);
            assert!(guard2.value().is_none());
            assert_eq!(2, pool.num_entries_or_locked());
            let guard3 = pool.blocking_lock_owned(3);
            assert!(guard3.value().is_none());
            assert_eq!(3, pool.num_entries_or_locked());

            std::mem::drop(guard2);
            assert_eq!(2, pool.num_entries_or_locked());
            std::mem::drop(guard1);
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard3);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock() {
            let pool = LockableLruCache::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard1 = pool.try_lock(1).unwrap();
            assert!(guard1.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            let guard2 = pool.try_lock(2).unwrap();
            assert!(guard2.value().is_none());
            assert_eq!(2, pool.num_entries_or_locked());
            let guard3 = pool.try_lock(3).unwrap();
            assert!(guard3.value().is_none());
            assert_eq!(3, pool.num_entries_or_locked());

            std::mem::drop(guard2);
            assert_eq!(2, pool.num_entries_or_locked());
            std::mem::drop(guard1);
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard3);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard1 = pool.try_lock_owned(1).unwrap();
            assert!(guard1.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            let guard2 = pool.try_lock_owned(2).unwrap();
            assert!(guard2.value().is_none());
            assert_eq!(2, pool.num_entries_or_locked());
            let guard3 = pool.try_lock_owned(3).unwrap();
            assert!(guard3.value().is_none());
            assert_eq!(3, pool.num_entries_or_locked());

            std::mem::drop(guard2);
            assert_eq!(2, pool.num_entries_or_locked());
            std::mem::drop(guard1);
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard3);
            assert_eq!(0, pool.num_entries_or_locked());
        }
    }

    mod concurrent {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.async_lock(5).await;

            let counter = Arc::new(AtomicU32::new(0));

            let child = launch_thread_async_lock(&pool, 5, &counter, None);

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the child is waiting
            {
                let _g = pool.async_lock(4).await;
            }

            // Now free the lock so the child can get it
            std::mem::drop(guard);

            // And check that the child got it
            child.join().unwrap();
            assert_eq!(1, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.async_lock_owned(5).await;

            let counter = Arc::new(AtomicU32::new(0));

            let child = launch_thread_async_lock_owned(&pool, 5, &counter, None);

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the child is waiting
            {
                let _g = pool.async_lock_owned(4).await;
            }

            // Now free the lock so the child can get it
            std::mem::drop(guard);

            // And check that the child got it
            child.join().unwrap();
            assert_eq!(1, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.blocking_lock(5);

            let counter = Arc::new(AtomicU32::new(0));

            let child = launch_thread_blocking_lock(&pool, 5, &counter, None);

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the child is waiting
            {
                let _g = pool.blocking_lock(4);
            }

            // Now free the lock so the child can get it
            std::mem::drop(guard);

            // And check that the child got it
            child.join().unwrap();
            assert_eq!(1, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.blocking_lock_owned(5);

            let counter = Arc::new(AtomicU32::new(0));

            let child = launch_thread_blocking_lock_owned(&pool, 5, &counter, None);

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the child is waiting
            {
                let _g = pool.blocking_lock_owned(4);
            }

            // Now free the lock so the child can get it
            std::mem::drop(guard);

            // And check that the child got it
            child.join().unwrap();
            assert_eq!(1, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.try_lock(5).unwrap();

            let counter = Arc::new(AtomicU32::new(0));

            let child = launch_thread_try_lock(&pool, 5, &counter, None);

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the child is waiting
            {
                let _g = pool.try_lock(4).unwrap();
            }

            // Now free the lock so the child can get it
            std::mem::drop(guard);

            // And check that the child got it
            child.join().unwrap();
            assert_eq!(1, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.try_lock_owned(5).unwrap();

            let counter = Arc::new(AtomicU32::new(0));

            let child = launch_thread_try_lock_owned(&pool, 5, &counter, None);

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the child is waiting
            {
                let _g = pool.try_lock_owned(4).unwrap();
            }

            // Now free the lock so the child can get it
            std::mem::drop(guard);

            // And check that the child got it
            child.join().unwrap();
            assert_eq!(1, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }
    }

    mod multi_concurrent {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.async_lock(5).await;

            let counter = Arc::new(AtomicU32::new(0));
            let barrier = Arc::new(Mutex::new(()));
            let barrier_guard = barrier.lock().unwrap();

            let child1 = launch_thread_async_lock(&pool, 5, &counter, Some(&barrier));
            let child2 = launch_thread_async_lock(&pool, 5, &counter, Some(&barrier));

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can stil lock other locks while the children are waiting
            {
                let _g = pool.async_lock(4).await;
            }

            // Now free the lock so a child can get it
            std::mem::drop(guard);

            // Check that a child got it
            thread::sleep(Duration::from_millis(100));
            assert_eq!(1, counter.load(Ordering::SeqCst));

            // Allow the child to free the lock
            std::mem::drop(barrier_guard);

            // Check that the other child got it
            child1.join().unwrap();
            child2.join().unwrap();
            assert_eq!(2, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.async_lock_owned(5).await;

            let counter = Arc::new(AtomicU32::new(0));
            let barrier = Arc::new(Mutex::new(()));
            let barrier_guard = barrier.lock().unwrap();

            let child1 = launch_thread_async_lock_owned(&pool, 5, &counter, Some(&barrier));
            let child2 = launch_thread_async_lock_owned(&pool, 5, &counter, Some(&barrier));

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can stil lock other locks while the children are waiting
            {
                let _g = pool.async_lock_owned(4).await;
            }

            // Now free the lock so a child can get it
            std::mem::drop(guard);

            // Check that a child got it
            thread::sleep(Duration::from_millis(100));
            assert_eq!(1, counter.load(Ordering::SeqCst));

            // Allow the child to free the lock
            std::mem::drop(barrier_guard);

            // Check that the other child got it
            child1.join().unwrap();
            child2.join().unwrap();
            assert_eq!(2, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.blocking_lock(5);

            let counter = Arc::new(AtomicU32::new(0));
            let barrier = Arc::new(Mutex::new(()));
            let barrier_guard = barrier.lock().unwrap();

            let child1 = launch_thread_blocking_lock(&pool, 5, &counter, Some(&barrier));
            let child2 = launch_thread_blocking_lock(&pool, 5, &counter, Some(&barrier));

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can stil lock other locks while the children are waiting
            {
                let _g = pool.blocking_lock(4);
            }

            // Now free the lock so a child can get it
            std::mem::drop(guard);

            // Check that a child got it
            thread::sleep(Duration::from_millis(100));
            assert_eq!(1, counter.load(Ordering::SeqCst));

            // Allow the child to free the lock
            std::mem::drop(barrier_guard);

            // Check that the other child got it
            child1.join().unwrap();
            child2.join().unwrap();
            assert_eq!(2, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.blocking_lock_owned(5);

            let counter = Arc::new(AtomicU32::new(0));
            let barrier = Arc::new(Mutex::new(()));
            let barrier_guard = barrier.lock().unwrap();

            let child1 = launch_thread_blocking_lock_owned(&pool, 5, &counter, Some(&barrier));
            let child2 = launch_thread_blocking_lock_owned(&pool, 5, &counter, Some(&barrier));

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can stil lock other locks while the children are waiting
            {
                let _g = pool.blocking_lock_owned(4);
            }

            // Now free the lock so a child can get it
            std::mem::drop(guard);

            // Check that a child got it
            thread::sleep(Duration::from_millis(100));
            assert_eq!(1, counter.load(Ordering::SeqCst));

            // Allow the child to free the lock
            std::mem::drop(barrier_guard);

            // Check that the other child got it
            child1.join().unwrap();
            child2.join().unwrap();
            assert_eq!(2, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.try_lock(5).unwrap();

            let counter = Arc::new(AtomicU32::new(0));
            let barrier = Arc::new(Mutex::new(()));
            let barrier_guard = barrier.lock().unwrap();

            let child1 = launch_thread_try_lock(&pool, 5, &counter, Some(&barrier));
            let child2 = launch_thread_try_lock(&pool, 5, &counter, Some(&barrier));

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can still lock other locks while the children are waiting
            {
                let _g = pool.try_lock(4).unwrap();
            }

            // Now free the lock so a child can get it
            std::mem::drop(guard);

            // Check that a child got it
            thread::sleep(Duration::from_millis(100));
            assert_eq!(1, counter.load(Ordering::SeqCst));

            // Allow the child to free the lock
            std::mem::drop(barrier_guard);

            // Check that the other child got it
            child1.join().unwrap();
            child2.join().unwrap();
            assert_eq!(2, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.try_lock_owned(5).unwrap();

            let counter = Arc::new(AtomicU32::new(0));
            let barrier = Arc::new(Mutex::new(()));
            let barrier_guard = barrier.lock().unwrap();

            let child1 = launch_thread_try_lock_owned(&pool, 5, &counter, Some(&barrier));
            let child2 = launch_thread_try_lock_owned(&pool, 5, &counter, Some(&barrier));

            // Check that even if we wait, the child thread won't get the lock
            thread::sleep(Duration::from_millis(100));
            assert_eq!(0, counter.load(Ordering::SeqCst));

            // Check that we can stil lock other locks while the children are waiting
            {
                let _g = pool.try_lock_owned(4).unwrap();
            }

            // Now free the lock so a child can get it
            std::mem::drop(guard);

            // Check that a child got it
            thread::sleep(Duration::from_millis(100));
            assert_eq!(1, counter.load(Ordering::SeqCst));

            // Allow the child to free the lock
            std::mem::drop(barrier_guard);

            // Check that the other child got it
            child1.join().unwrap();
            child2.join().unwrap();
            assert_eq!(2, counter.load(Ordering::SeqCst));

            assert_eq!(0, pool.num_entries_or_locked());
        }
    }

    #[test]
    fn blocking_lock_owned_guards_can_be_passed_around() {
        let make_guard = || {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            pool.blocking_lock_owned(5)
        };
        let _guard = make_guard();
    }

    #[tokio::test]
    async fn async_lock_owned_guards_can_be_passed_around() {
        let make_guard = || async {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            pool.async_lock_owned(5).await
        };
        let _guard = make_guard().await;
    }

    #[test]
    fn test_try_lock_owned_guards_can_be_passed_around() {
        let make_guard = || {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            pool.try_lock_owned(5)
        };
        let guard = make_guard();
        assert!(guard.is_ok());
    }

    #[tokio::test]
    async fn async_lock_guards_can_be_held_across_await_points() {
        let task = async {
            let pool = LockableLruCache::<isize, String>::new();
            let guard = pool.async_lock(3).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
            std::mem::drop(guard);
        };

        // We also need to move the task to a different thread because
        // only then the compiler checks whether the task is Send.
        thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(task);
        });
    }

    #[tokio::test]
    async fn async_lock_owned_guards_can_be_held_across_await_points() {
        let task = async {
            let pool = Arc::new(LockableLruCache::<isize, String>::new());
            let guard = pool.async_lock_owned(3).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
            std::mem::drop(guard);
        };

        // We also need to move the task to a different thread because
        // only then the compiler checks whether the task is Send.
        thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(task);
        });
    }
}
