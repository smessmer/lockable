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
use super::utils::time::{RealTime, TimeProvider};

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
// Invariant:
// -  The `last_unlocked` timestamps of CacheEntry instances in the map will follow the
//    same order as the LRU order of the map, with an exception for currently locked
//    entries that may be temporarily out of order while the entry is locked.
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
impl<V, Time> FromInto<V, LruCacheHooks<Time>> for CacheEntry<V>
where
    Time: TimeProvider + Clone,
{
    fn fi_from(value: V, hooks: &LruCacheHooks<Time>) -> CacheEntry<V> {
        CacheEntry {
            value,
            // last_unlocked is now since the entry was just freshly inserted
            last_unlocked: hooks.time_provider.now(),
        }
    }
    fn fi_into(self) -> V {
        self.value
    }
}

// LruCacheHooks ensure that whenever we unlock an entry, its last_unlocked
// timestamp gets updated
#[derive(Clone)]
pub struct LruCacheHooks<Time: TimeProvider + Clone> {
    time_provider: Time,
}
impl<V, Time: TimeProvider + Clone> Hooks<CacheEntry<V>> for LruCacheHooks<Time> {
    fn on_unlock(&self, v: Option<&mut CacheEntry<V>>) {
        if let Some(v) = v {
            v.last_unlocked = self.time_provider.now();
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
/// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
/// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
/// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
pub struct LockableLruCache<K, V, Time = RealTime>
where
    K: Eq + PartialEq + Hash + Clone,
    Time: TimeProvider + Default + Clone,
{
    map_impl: LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>>,
}

impl<K, V, Time> Lockable<K, V> for LockableLruCache<K, V, Time>
where
    K: Eq + PartialEq + Hash + Clone,
    Time: TimeProvider + Default + Clone,
{
    type Guard<'a> = Guard<
        MapImpl<K, V>,
        V,
        LruCacheHooks<Time>,
        &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>>,
    > where
        K: 'a,
        V: 'a,
        Time: 'a;

    type OwnedGuard =
        Guard<MapImpl<K, V>, V, LruCacheHooks<Time>, Arc<LockableLruCache<K, V, Time>>>;

    type SyncLimit<'a, OnEvictFn, E> = SyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks<Time>,
        &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>>,
        E,
        OnEvictFn,
    > where
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>,
        K: 'a,
        V: 'a,
        Time: 'a;

    type SyncLimitOwned<OnEvictFn, E> = SyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks<Time>,
        Arc<LockableLruCache<K, V, Time>>,
        E,
        OnEvictFn,
    > where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

    type AsyncLimit<'a, OnEvictFn, E, F> = AsyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks<Time>,
        &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>>,
        E,
        F,
        OnEvictFn,
    > where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F,
        K: 'a,
        V: 'a,
        Time: 'a;

    type AsyncLimitOwned<OnEvictFn, E, F> = AsyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks<Time>,
        Arc<LockableLruCache<K, V, Time>>,
        E,
        F,
        OnEvictFn,
    > where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F;

    #[inline]
    fn num_entries_or_locked(&self) -> usize {
        self.map_impl.num_entries_or_locked()
    }

    #[inline]
    fn blocking_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::SyncLimit<'a, OnEvictFn, E>,
    ) -> Result<Self::Guard<'a>, E>
    where
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>,
    {
        LockableMapImpl::blocking_lock(&self.map_impl, key, limit)
    }

    #[inline]
    fn blocking_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::SyncLimitOwned<OnEvictFn, E>,
    ) -> Result<Self::OwnedGuard, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>,
    {
        LockableMapImpl::blocking_lock(Arc::clone(self), key, limit)
    }

    #[inline]
    fn try_lock<'a, E, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::SyncLimit<'a, OnEvictFn, E>,
    ) -> Result<Option<Self::Guard<'a>>, E>
    where
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>,
    {
        LockableMapImpl::try_lock(&self.map_impl, key, limit)
    }

    #[inline]
    fn try_lock_owned<E, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::SyncLimitOwned<OnEvictFn, E>,
    ) -> Result<Option<Self::OwnedGuard>, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>,
    {
        LockableMapImpl::try_lock(Arc::clone(self), key, limit)
    }

    #[inline]
    async fn try_lock_async<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::AsyncLimit<'a, OnEvictFn, E, F>,
    ) -> Result<Option<Self::Guard<'a>>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<<Self as Lockable<K, V>>::Guard<'a>>) -> F,
    {
        LockableMapImpl::try_lock_async(&self.map_impl, key, limit).await
    }

    #[inline]
    async fn try_lock_owned_async<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::AsyncLimitOwned<OnEvictFn, E, F>,
    ) -> Result<Option<Self::OwnedGuard>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F,
    {
        LockableMapImpl::try_lock_async(Arc::clone(self), key, limit).await
    }

    #[inline]
    async fn async_lock<'a, E, F, OnEvictFn>(
        &'a self,
        key: K,
        limit: Self::AsyncLimit<'a, OnEvictFn, E, F>,
    ) -> Result<Self::Guard<'a>, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F,
    {
        LockableMapImpl::async_lock(&self.map_impl, key, limit).await
    }

    #[inline]
    async fn async_lock_owned<E, F, OnEvictFn>(
        self: &Arc<Self>,
        key: K,
        limit: Self::AsyncLimitOwned<OnEvictFn, E, F>,
    ) -> Result<Self::OwnedGuard, E>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F,
    {
        LockableMapImpl::async_lock(Arc::clone(self), key, limit).await
    }

    #[inline]
    fn into_entries_unordered(self) -> impl Iterator<Item = (K, V)> {
        self.map_impl
            .into_entries_unordered()
            .map(|(k, v)| (k, v.value))
    }

    #[inline]
    fn keys_with_entries_or_locked(&self) -> Vec<K> {
        self.map_impl.keys_with_entries_or_locked()
    }

    #[inline]
    async fn lock_all_entries(&self) -> impl Stream<Item = <Self as Lockable<K, V>>::Guard<'_>> {
        LockableMapImpl::lock_all_entries(&self.map_impl).await
    }

    #[inline]
    async fn lock_all_entries_owned(
        self: &Arc<Self>,
    ) -> impl Stream<Item = <Self as Lockable<K, V>>::OwnedGuard> {
        LockableMapImpl::lock_all_entries(Arc::clone(self)).await
    }
}

impl<K, V, Time> LockableLruCache<K, V, Time>
where
    K: Eq + PartialEq + Hash + Clone,
    Time: TimeProvider + Default + Clone,
{
    /// Create a new hash map with no entries and no locked keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
    ///
    /// let lockable_map: LockableLruCache<i64, String> = LockableLruCache::new();
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn new() -> Self {
        let time_provider = Time::default();
        Self {
            map_impl: LockableMapImpl::new_with_hooks(LruCacheHooks { time_provider }),
        }
    }

    /// Lock all entries that are currently unlocked and that were unlocked for at least
    /// the given `duration`. This follows the LRU nature of the cache.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    pub fn lock_entries_unlocked_for_at_least(
        &self,
        duration: Duration,
    ) -> impl Iterator<Item = <Self as Lockable<K, V>>::Guard<'_>> {
        Self::_lock_entries_unlocked_for_at_least(
            &self.map_impl,
            self.map_impl.hooks().time_provider.now(),
            duration,
        )
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
    /// use lockable::{AsyncLimit, Lockable, LockableLruCache};
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
    pub fn lock_entries_unlocked_for_at_least_owned(
        self: &Arc<Self>,
        duration: Duration,
    ) -> impl Iterator<Item = <Self as Lockable<K, V>>::OwnedGuard> {
        Self::_lock_entries_unlocked_for_at_least(
            Arc::clone(self),
            self.map_impl.hooks().time_provider.now(),
            duration,
        )
    }

    fn _lock_entries_unlocked_for_at_least<
        S: Borrow<LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>>> + Clone,
    >(
        this: S,
        now: Instant,
        duration: Duration,
    ) -> impl Iterator<Item = Guard<MapImpl<K, V>, V, LruCacheHooks<Time>, S>> {
        let cutoff = now - duration;
        LockableMapImpl::lock_all_unlocked(this, &move |entry| {
            let entry = entry.value_raw().expect("There must be a value, otherwise it cannot exist in the map as an 'unlocked' entry");
            entry.last_unlocked <= cutoff
        }).into_iter()
    }

    #[cfg(test)]
    fn time_provider(&self) -> &Time {
        &self.map_impl.hooks().time_provider
    }
}

impl<K, V, Time> Default for LockableLruCache<K, V, Time>
where
    K: Eq + PartialEq + Hash + Clone,
    Time: TimeProvider + Default + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// We implement Borrow<LockableMapImpl> for Arc<LockableLruCache<K, V>> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableLruCache to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V, Time> Borrow<LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>>>
    for Arc<LockableLruCache<K, V, Time>>
where
    K: Eq + PartialEq + Hash + Clone,
    Time: TimeProvider + Default + Clone,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks<Time>> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instantiate_lockable_tests;
    use crate::utils::time::MockTime;

    instantiate_lockable_tests!(LockableLruCache);

    macro_rules! instantiate_lock_entries_unlocked_for_at_least_tests {
        ($create_map:expr, $lock_entries_unlocked_for_at_least:ident) => {
            #[test]
            fn zero_entries() {
                let map = $create_map;
                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                assert_eq!(Vec::<(i64, String)>::new(), old_enough);
            }

            #[tokio::test]
            async fn one_entry_not_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                assert_eq!(Vec::<(i64, String)>::new(), old_enough);
            }

            #[tokio::test]
            async fn one_entry_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));

                map.time_provider().advance_time(Duration::from_secs(1));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn one_entry_locked() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));

                map.time_provider().advance_time(Duration::from_secs(1));

                let _guard = map.async_lock(1, AsyncLimit::no_limit()).await.unwrap();

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(vec![], old_enough);
            }

            #[tokio::test]
            async fn two_entries_zero_old_enough_zero_locked() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                assert_eq!(Vec::<(i64, String)>::new(), old_enough);
            }

            #[tokio::test]
            async fn two_entries_one_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.time_provider().advance_time(Duration::from_secs(1));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn two_entries_one_old_enough_and_locked() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.time_provider().advance_time(Duration::from_secs(1));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));

                let _guard = map.async_lock(1, AsyncLimit::no_limit()).await.unwrap();

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(vec![], old_enough);
            }

            #[tokio::test]
            async fn two_entries_both_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1")), (2, String::from("Value 2"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn two_entries_both_old_enough_one_locked_1() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let _guard = map.async_lock(1, AsyncLimit::no_limit()).await.unwrap();

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(2, String::from("Value 2"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn two_entries_both_old_enough_one_locked_2() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let _guard = map.async_lock(2, AsyncLimit::no_limit()).await.unwrap();

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn two_entries_both_old_enough_both_locked() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let _guard1 = map.async_lock(1, AsyncLimit::no_limit()).await.unwrap();
                let _guard2 = map.async_lock(2, AsyncLimit::no_limit()).await.unwrap();

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(vec![], old_enough);
            }

            #[tokio::test]
            async fn three_entries_zero_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                assert_eq!(Vec::<(i64, String)>::new(), old_enough);
            }

            #[tokio::test]
            async fn three_entries_one_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.time_provider().advance_time(Duration::from_secs(1));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn three_entries_two_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.time_provider().advance_time(Duration::from_secs(1));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1")), (2, String::from("Value 2"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn three_entries_three_old_enough() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![
                        (1, String::from("Value 1")),
                        (2, String::from("Value 2")),
                        (3, String::from("Value 3")),
                    ],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn locking_an_entry_makes_it_not_old_enough_anymore_1() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let guard = map.async_lock(1, AsyncLimit::no_limit()).await.unwrap();
                std::mem::drop(guard);

                map.time_provider().advance_time(Duration::from_millis(100));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(2, String::from("Value 2")), (3, String::from("Value 3"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn locking_an_entry_makes_it_not_old_enough_anymore_2() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let guard = map.async_lock(2, AsyncLimit::no_limit()).await.unwrap();
                std::mem::drop(guard);

                map.time_provider().advance_time(Duration::from_millis(100));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1")), (3, String::from("Value 3"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn locking_an_entry_makes_it_not_old_enough_anymore_3() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));
                map.async_lock(3, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 3"));
                map.time_provider().advance_time(Duration::from_secs(1));

                let guard = map.async_lock(3, AsyncLimit::no_limit()).await.unwrap();
                std::mem::drop(guard);

                map.time_provider().advance_time(Duration::from_millis(100));

                let old_enough: Vec<(i64, String)> = map
                    .$lock_entries_unlocked_for_at_least(Duration::from_millis(500))
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1")), (2, String::from("Value 2"))],
                    old_enough,
                );
            }

            #[tokio::test]
            async fn can_lock_other_elements_while_iterator_is_running() {
                let map = $create_map;
                map.async_lock(1, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 1"));
                map.time_provider().advance_time(Duration::from_secs(1));
                map.async_lock(2, AsyncLimit::no_limit())
                    .await
                    .unwrap()
                    .insert(String::from("Value 2"));

                map.time_provider().advance_time(Duration::from_millis(100));

                let old_enough_iterator =
                    map.$lock_entries_unlocked_for_at_least(Duration::from_millis(500));

                // Locking another entry should work and not interfere with the iterator
                let guard = map.async_lock(2, AsyncLimit::no_limit()).await.unwrap();
                std::mem::drop(guard);

                let old_enough: Vec<(i64, String)> = old_enough_iterator
                    .map(|guard| (*guard.key(), guard.value().cloned().unwrap()))
                    .collect();
                crate::tests::assert_vec_eq_unordered(
                    vec![(1, String::from("Value 1"))],
                    old_enough,
                );
            }
        };
    }

    mod lock_entries_unlocked_for_at_least {
        use super::*;

        instantiate_lock_entries_unlocked_for_at_least_tests!(
            LockableLruCache::<i64, String, MockTime>::new(),
            lock_entries_unlocked_for_at_least
        );
    }

    mod lock_entries_unlocked_for_at_least_owned {
        use super::*;

        instantiate_lock_entries_unlocked_for_at_least_tests!(
            Arc::new(LockableLruCache::<i64, String, MockTime>::new()),
            lock_entries_unlocked_for_at_least_owned
        );
    }
}
