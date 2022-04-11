use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::error::TryLockError;
use super::guard::GuardImpl;
use super::hooks::NoopHooks;
use super::lockable_map_impl::LockableMapImpl;
use super::map_like::{ArcMutexMapLike, EntryValue};

impl<K, V> ArcMutexMapLike for HashMap<K, Arc<Mutex<EntryValue<V>>>>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    type K = K;
    type V = V;

    fn new() -> Self {
        Self::new()
    }

    fn len(&self) -> usize {
        self.into_iter().len()
    }

    fn get_or_insert_none(&mut self, key: &Self::K) -> &Arc<Mutex<EntryValue<Self::V>>> {
        // TODO Is there a way to only clone the key when the entry doesn't already exist?
        self.entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(EntryValue { value: None })))
    }

    fn get(&mut self, key: &Self::K) -> Option<&Arc<Mutex<EntryValue<Self::V>>>> {
        HashMap::get(self, key)
    }

    fn remove(&mut self, key: &Self::K) -> Option<Arc<Mutex<EntryValue<Self::V>>>> {
        self.remove(key)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (&Self::K, &Arc<Mutex<EntryValue<Self::V>>>)> + '_> {
        Box::new(HashMap::iter(self))
    }
}

type MapImpl<K, V> = HashMap<K, Arc<tokio::sync::Mutex<EntryValue<V>>>>;

/// A threadsafe hash map where individual keys can be locked/unlocked, even if there is no entry for this key in the map.
/// It initially considers all keys as "unlocked", but they can be locked
/// and if a second thread tries to acquire a lock for the same key, they will have to wait.
///
/// ```
/// use lockable::LockableHashMap;
///
/// let hash_map: LockableHashMap<i64, String> = LockableHashMap::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let entry1 = hash_map.async_lock(4).await;
/// let entry2 = hash_map.async_lock(5).await;
///
/// // This next line would cause a deadlock or panic because `4` is already locked on this thread
/// // let entry3 = hash_map.async_lock(4).await;
///
/// // After dropping the corresponding guard, we can lock it again
/// std::mem::drop(entry1);
/// let entry3 = hash_map.async_lock(4).await;
/// # });
/// ```
///
/// The guards holding a lock for an entry can be used to insert that entry to the hash map, remove it from the hash map, or to modify
/// the value of an existing entry.
///
/// ```
/// use lockable::LockableHashMap;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// async fn insert_entry(hash_map: &LockableHashMap<i64, String>) {
///     let mut entry = hash_map.async_lock(4).await;
///     entry.insert(String::from("Hello World"));
/// }
///
/// async fn remove_entry(hash_map: &LockableHashMap<i64, String>) {
///     let mut entry = hash_map.async_lock(4).await;
///     entry.remove();
/// }
///
/// let hash_map: LockableHashMap<i64, String> = LockableHashMap::new();
/// assert_eq!(None, hash_map.async_lock(4).await.value());
/// insert_entry(&hash_map).await;
/// assert_eq!(Some(&String::from("Hello World")), hash_map.async_lock(4).await.value());
/// remove_entry(&hash_map).await;
/// assert_eq!(None, hash_map.async_lock(4).await.value());
/// # });
/// ```
///
///
/// You can use an arbitrary type to index hash map entries by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone] + [Debug].
///
/// ```
/// use lockable::LockableHashMap;
///
/// #[derive(PartialEq, Eq, Hash, Clone, Debug)]
/// struct CustomLockKey(u32);
///
/// let hash_map: LockableHashMap<CustomLockKey, String> = LockableHashMap::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = hash_map.async_lock(CustomLockKey(4)).await;
/// # });
/// ```
///
/// Under the hood, a [LockableHashMap] is a [std::collections::HashMap] of [Mutex](tokio::sync::Mutex)es, with some logic making sure there aren't any race conditions when adding or removing entries.
#[derive(Debug)]
pub struct LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    map_impl: LockableMapImpl<MapImpl<K, V>, NoopHooks>,
}

impl<K, V> LockableHashMap<K, V>
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

    /// Return the number of map entries.
    ///
    /// Corner case: Currently locked keys are counted even if they don't exist in the map.
    #[inline]
    pub fn num_entries_or_locked(&self) -> usize {
        self.map_impl.num_entries_or_locked()
    }

    /// Lock a key and return a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [HashMapGuard::insert] and [HashMapGuard::remove] on the returned entry guard.
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
    /// use lockable::LockableHashMap;
    ///
    /// let hash_map = LockableHashMap::<i64, String>::new();
    /// let guard1 = hash_map.blocking_lock(4);
    /// let guard2 = hash_map.blocking_lock(5);
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = hash_map.blocking_lock(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = hash_map.blocking_lock(4);
    /// ```
    #[inline]
    pub fn blocking_lock(&self, key: K) -> HashMapGuard<'_, K, V> {
        LockableMapImpl::blocking_lock(&self.map_impl, key)
    }

    /// Lock a lock by key and return a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableHashMap::blocking_lock], but it works on an `Arc<LockableHashMap>` instead of a [LockableHashMap] and
    /// returns a [HashMapOwnedGuard] that binds its lifetime to the [LockableHashMap] in that [Arc]. Such a [HashMapOwnedGuard] can be more
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
    /// use lockable::LockableHashMap;
    /// use std::sync::Arc;
    ///
    /// let hash_map = Arc::new(LockableHashMap::<i64, String>::new());
    /// let guard1 = hash_map.blocking_lock_owned(4);
    /// let guard2 = hash_map.blocking_lock_owned(5);
    ///
    /// // This next line would cause a deadlock or panic because `4` is already locked on this thread
    /// // let guard3 = hash_map.blocking_lock_owned(4);
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = hash_map.blocking_lock_owned(4);
    /// ```
    #[inline]
    pub fn blocking_lock_owned(self: &Arc<Self>, key: K) -> HashMapOwnedGuard<K, V> {
        LockableMapImpl::blocking_lock(Arc::clone(self), key)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    /// Any changes to that entry will be persisted in the map.
    /// Locking a key prevents any other threads from locking the same key, but the action of locking a key doesn't insert
    /// a map entry by itself. Map entries can be inserted and removed using [HashMapGuard::insert] and [HashMapGuard::remove] on the returned entry guard.
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
    /// use lockable::{TryLockError, LockableHashMap};
    ///
    /// let hash_map: LockableHashMap<i64, String> = LockableHashMap::new();
    /// let guard1 = hash_map.blocking_lock(4);
    /// let guard2 = hash_map.blocking_lock(5);
    ///
    /// // This next line cannot acquire the lock because `4` is already locked on this thread
    /// let guard3 = hash_map.try_lock(4);
    /// assert!(matches!(guard3.unwrap_err(), TryLockError::WouldBlock));
    ///
    /// // After dropping the corresponding guard, we can lock it again
    /// std::mem::drop(guard1);
    /// let guard3 = hash_map.try_lock(4);
    /// assert!(guard3.is_ok());
    /// ```
    #[inline]
    pub fn try_lock(&self, key: K) -> Result<HashMapGuard<'_, K, V>, TryLockError> {
        LockableMapImpl::try_lock(&self.map_impl, key)
    }

    /// Attempts to acquire the lock with the given key and if successful, returns a guard with any potential map entry for that key.
    ///
    /// This is identical to [LockableHashMap::try_lock], but it works on an `Arc<LockableHashMap>` instead of a [LockableHashMap] and
    /// returns an [HashMapOwnedGuard] that binds its lifetime to the [LockableHashMap] in that [Arc]. Such a [HashMapOwnedGuard] can be more
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
    /// use lockable::{TryLockError, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// let pool = Arc::new(LockableHashMap::<i64, String>::new());
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
    pub fn try_lock_owned(
        self: &Arc<Self>,
        key: K,
    ) -> Result<HashMapOwnedGuard<K, V>, TryLockError> {
        LockableMapImpl::try_lock(Arc::clone(self), key)
    }

    /// TODO Docs
    #[inline]
    pub async fn async_lock(&self, key: K) -> HashMapGuard<'_, K, V> {
        LockableMapImpl::async_lock(&self.map_impl, key).await
    }

    /// TODO Docs
    #[inline]
    pub async fn async_lock_owned(self: &Arc<Self>, key: K) -> HashMapOwnedGuard<K, V> {
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

/// A non-owning guard holding a lock for an entry in a [LockableHashMap].
/// This guard is created via [LockableHashMap::blocking_lock], [LockableHashMap::async_lock]
/// or [LockableHashMap::try_lock] and its lifetime is bound to the lifetime
/// of the [LockableHashMap].
/// 
/// See the documentation of [GuardImpl] for methods.
pub type HashMapGuard<'a, K, V> =
    GuardImpl<MapImpl<K, V>, NoopHooks, &'a LockableMapImpl<MapImpl<K, V>, NoopHooks>>;

/// A owning guard holding a lock for an entry in a [LockableHashMap].
/// This guard is created via [LockableHashMap::blocking_lock_owned], [LockableHashMap::async_lock_owned]
/// or [LockableHashMap::try_lock_owned] and its lifetime is bound to the lifetime of the [LockableHashMap]
/// within its [Arc].
/// 
/// See the documentation of [GuardImpl] for methods.
pub type HashMapOwnedGuard<K, V> = GuardImpl<MapImpl<K, V>, NoopHooks, Arc<LockableHashMap<K, V>>>;

// We implement Borrow<LockableMapImpl> for Arc<LockableHashMap> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableHashMap to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, NoopHooks>> for Arc<LockableHashMap<K, V>>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, NoopHooks> {
        &self.map_impl
    }
}

// TODO Deduplicate tests with the tests for LockableLruCache
#[cfg(test)]
mod tests {
    use super::super::error::TryLockError;
    use super::LockableHashMap;
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
        pool: &Arc<LockableHashMap<isize, String>>,
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
        pool: &Arc<LockableHashMap<isize, String>>,
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
        pool: &Arc<LockableHashMap<isize, String>>,
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
        pool: &Arc<LockableHashMap<isize, String>>,
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
        pool: &Arc<LockableHashMap<isize, String>>,
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
        pool: &Arc<LockableHashMap<isize, String>>,
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
        let p = LockableHashMap::<isize, String>::new();
        let _ = p.blocking_lock(3);
    }

    #[tokio::test]
    #[should_panic(
        expected = "Cannot start a runtime from within a runtime. This happens because a function (like `block_on`) attempted to block the current thread while the thread is being used to drive asynchronous tasks."
    )]
    async fn blocking_lock_owned_from_async_context_with_sync_api() {
        let p = Arc::new(LockableHashMap::<isize, String>::new());
        let _ = p.blocking_lock_owned(3);
    }

    mod simple {
        use super::*;

        #[tokio::test]
        async fn async_lock() {
            let pool = LockableHashMap::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.async_lock(4).await;
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[tokio::test]
        async fn async_lock_owned() {
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.async_lock_owned(4).await;
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock() {
            let pool = LockableHashMap::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.blocking_lock(4);
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn blocking_lock_owned() {
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.blocking_lock_owned(4);
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock() {
            let pool = LockableHashMap::<isize, String>::new();
            assert_eq!(0, pool.num_entries_or_locked());
            let guard = pool.try_lock(4).unwrap();
            assert!(guard.value().is_none());
            assert_eq!(1, pool.num_entries_or_locked());
            std::mem::drop(guard);
            assert_eq!(0, pool.num_entries_or_locked());
        }

        #[test]
        fn try_lock_owned() {
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
            pool.blocking_lock_owned(5)
        };
        let _guard = make_guard();
    }

    #[tokio::test]
    async fn async_lock_owned_guards_can_be_passed_around() {
        let make_guard = || async {
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
            pool.async_lock_owned(5).await
        };
        let _guard = make_guard().await;
    }

    #[test]
    fn test_try_lock_owned_guards_can_be_passed_around() {
        let make_guard = || {
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
            pool.try_lock_owned(5)
        };
        let guard = make_guard();
        assert!(guard.is_ok());
    }

    #[tokio::test]
    async fn async_lock_guards_can_be_held_across_await_points() {
        let task = async {
            let pool = LockableHashMap::<isize, String>::new();
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
            let pool = Arc::new(LockableHashMap::<isize, String>::new());
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
