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

type MapImpl<K, V> = HashMap<K, Arc<tokio::sync::Mutex<EntryValue<V>>>>;

impl<K, V> ArcMutexMapLike for MapImpl<K, V>
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
        self.iter().len()
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
    map_impl: LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
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
    /// Caveat: Locked keys are listed even if they don't carry a value
    #[inline]
    pub fn keys(&self) -> Vec<K> {
        self.map_impl.keys()
    }
}

impl<K, V> Default for LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A non-owning guard holding a lock for an entry in a [LockableHashMap].
/// This guard is created via [LockableHashMap::blocking_lock], [LockableHashMap::async_lock]
/// or [LockableHashMap::try_lock] and its lifetime is bound to the lifetime
/// of the [LockableHashMap].
///
/// See the documentation of [GuardImpl] for methods.
pub type HashMapGuard<'a, K, V> =
    GuardImpl<MapImpl<K, V>, V, NoopHooks, &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>>;

/// A owning guard holding a lock for an entry in a [LockableHashMap].
/// This guard is created via [LockableHashMap::blocking_lock_owned], [LockableHashMap::async_lock_owned]
/// or [LockableHashMap::try_lock_owned] and its lifetime is bound to the lifetime of the [LockableHashMap]
/// within its [Arc].
///
/// See the documentation of [GuardImpl] for methods.
pub type HashMapOwnedGuard<K, V> =
    GuardImpl<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>>;

// We implement Borrow<LockableMapImpl> for Arc<LockableHashMap> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableHashMap to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, V, NoopHooks>> for Arc<LockableHashMap<K, V>>
where
    K: Eq + PartialEq + Hash + Clone + Debug + 'static,
    V: Debug + 'static,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, V, NoopHooks> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::LockableHashMap;
    use crate::instantiate_lockable_tests;

    instantiate_lockable_tests!(LockableHashMap);
}
