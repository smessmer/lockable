use lru::LruCache;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

use super::error::TryLockError;
use super::guard::GuardImpl;
use super::hooks::Hooks;
use super::lockable_map_impl::LockableMapImpl;
use super::map_like::{ArcMutexMapLike, EntryValue};

type MapImpl<K, V> = LruCache<K, Arc<tokio::sync::Mutex<EntryValue<CacheEntry<V>>>>>;

// The ArcMutexMapLike implementation here allows LockableMapImpl to
// work with LruCache as an underlying map
impl<K, V> ArcMutexMapLike for MapImpl<K, V>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    type K = K;
    type V = CacheEntry<V>;

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

// The LRUCache actually stores <K, CacheEntry<V>> instead of <K, V> so that we can
// remember a last_unlocked timestamp for each entry
#[derive(Debug)]
pub struct CacheEntry<V> {
    value: V,
    last_unlocked: Instant,
}

// Borrow and BorrowMut are used to allow GuardImpl to offer an API to read/write V while
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

// From is used to allow GuardImpl to offer an API to insert V values while
// the cache actually stores values as CacheEntry<V>
impl<V> From<V> for CacheEntry<V> {
    fn from(value: V) -> CacheEntry<V> {
        CacheEntry {
            value,
            // last_unlocked is now since the entry was just freshly inserted
            last_unlocked: Instant::now(),
        }
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
    map_impl: LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
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
            map_impl: LockableMapImpl::new_with_hooks(LruCacheHooks),
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
        self.map_impl
            .into_entries_unordered()
            .map(|(k, v)| (k, v.value))
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn keys(&self) -> Vec<K> {
        self.map_impl.keys()
    }

    /// TODO Docs
    /// TODO Test
    pub fn remove_entries_unlocked_for_longer_than(
        &self,
        duration: Duration,
    ) {
        let now = Instant::now();
        for mut entry in LockableMapImpl::lock_all_unlocked(&self.map_impl) {
            if let Some(value) = entry.value_raw() {
                if value.last_unlocked + duration <= now {
                    entry.remove();
                }
            }
        }
    }
}

impl<K, V> Default for LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A non-owning guard holding a lock for an entry in a [LockableLruCache].
/// This guard is created via [LockableLruCache::blocking_lock], [LockableLruCache::async_lock]
/// or [LockableLruCache::try_lock] and its lifetime is bound to the lifetime
/// of the [LockableLruCache].
///
/// See the documentation of [GuardImpl] for methods.
pub type LruGuard<'a, K, V> = GuardImpl<
    MapImpl<K, V>,
    V,
    LruCacheHooks,
    &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
>;

/// A owning guard holding a lock for an entry in a [LockableLruCache].
/// This guard is created via [LockableLruCache::blocking_lock_owned], [LockableLruCache::async_lock_owned]
/// or [LockableLruCache::try_lock_owned] and its lifetime is bound to the lifetime of the [LockableLruCache]
/// within its [Arc].
///
/// See the documentation of [GuardImpl] for methods.
pub type LruOwnedGuard<K, V> =
    GuardImpl<MapImpl<K, V>, V, LruCacheHooks, Arc<LockableLruCache<K, V>>>;

// We implement Borrow<LockableMapImpl> for Arc<LockableLruCache> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableLruCache to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>> for Arc<LockableLruCache<K, V>>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::LockableLruCache;
    use crate::instantiate_lockable_tests;

    instantiate_lockable_tests!(LockableLruCache);
}
