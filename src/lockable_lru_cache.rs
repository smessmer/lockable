use anyhow::Result;
use async_trait::async_trait;
use futures::stream::Stream;
use lru::LruCache;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
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
/// use lockable::{Lockable, AsyncLimit, LockableLruCache};
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
/// # Ok::<(), anyhow::Error>(())}).unwrap();
/// ```
///
/// The guards holding a lock for an entry can be used to insert that entry to the cache, remove
/// it from the cache, or to modify the value of an existing entry.
///
/// ```
/// use anyhow::Result;
/// use lockable::{Lockable, AsyncLimit, LockableLruCache};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// async fn insert_entry(lockable_cache: &LockableLruCache<i64, String>) -> Result<()> {
///     let mut entry_guard = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.insert(String::from("Hello World"));
///     Ok(())
/// }
///
/// async fn remove_entry(lockable_cache: &LockableLruCache<i64, String>) -> Result<()> {
///     let mut entry_guard = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.remove();
///     Ok(())
/// }
///
/// let lockable_cache: LockableLruCache<i64, String> = LockableLruCache::new();
/// assert_eq!(None, lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?.value());
/// insert_entry(&lockable_cache).await;
/// assert_eq!(Some(&String::from("Hello World")), lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?.value());
/// remove_entry(&lockable_cache).await;
/// assert_eq!(None, lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?.value());
/// # Ok::<(), anyhow::Error>(())}).unwrap();
/// ```
///
///
/// You can use an arbitrary type to index cache entries by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone].
///
/// ```
/// use lockable::{Lockable, AsyncLimit, LockableLruCache};
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// struct CustomLockKey(u32);
///
/// let lockable_cache: LockableLruCache<CustomLockKey, String> = LockableLruCache::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = lockable_cache.async_lock(CustomLockKey(4), AsyncLimit::no_limit()).await?;
/// # Ok::<(), anyhow::Error>(())}).unwrap();
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

#[async_trait]
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

    type SyncLimit<'a, OnEvictFn, E> = SyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks,
        &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
        E,
        OnEvictFn,
    > where
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> Result<(), E>,
        K: 'a,
        V: 'a;

    type SyncLimitOwned<OnEvictFn, E> = SyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks,
        Arc<LockableLruCache<K, V>>,
        E,
        OnEvictFn,
    > where
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> Result<(), E>;

    type AsyncLimit<'a, OnEvictFn, E, F> = AsyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks,
        &'a LockableMapImpl<MapImpl<K, V>, V, LruCacheHooks>,
        E,
        F,
        OnEvictFn,
    > where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> F,
        K: 'a,
        V: 'a;

    type AsyncLimitOwned<OnEvictFn, E, F> = AsyncLimit<
        MapImpl<K, V>,
        V,
        LruCacheHooks,
        Arc<LockableLruCache<K, V>>,
        E,
        F,
        OnEvictFn,
    > where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> F;

    #[inline]
    fn new() -> Self {
        Self {
            map_impl: LockableMapImpl::new_with_hooks(LruCacheHooks),
        }
    }

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
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> Result<(), E>,
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
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> Result<(), E>,
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
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> Result<(), E>,
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
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> Result<(), E>,
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
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync,
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
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync,
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
        OnEvictFn: Fn(Vec<Self::Guard<'a>>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync,
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
        OnEvictFn: Fn(Vec<Self::OwnedGuard>) -> F,
        F: Send,
        K: Send + Sync,
        V: Send + Sync,
        OnEvictFn: Send + Sync,
    {
        LockableMapImpl::async_lock(Arc::clone(self), key, limit).await
    }
}

impl<K, V> LockableLruCache<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    /// Consumes the cache and returns an iterator over all of its entries.
    /// TODO Test
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
    #[inline]
    pub fn keys_with_entries_or_locked(&self) -> Vec<K> {
        self.map_impl.keys_with_entries_or_locked()
    }

    /// Lock all entries that are currently unlocked and that were unlocked for at least
    /// the given `duration`. This follows the LRU nature of the cache.
    /// TODO Test whether the returned iterator keeps a lock on the whole map and if yes,
    ///      try to fix that or at least document it.
    /// TODO Test
    pub fn lock_entries_unlocked_for_at_least_owned(
        self: &Arc<Self>,
        duration: Duration,
    ) -> impl Iterator<Item = <Self as Lockable<K, V>>::OwnedGuard> {
        // TODO Since entries should be LRU ordered, we don't need to iterate over all of them, just until one is new enough.
        let now = Instant::now();
        LockableMapImpl::lock_all_unlocked(Arc::clone(self)).filter(move |entry| {
            if let Some(entry) = entry.value_raw() {
                entry.last_unlocked + duration <= now
            } else {
                false
            }
        })
    }

    /// Lock all entries of the cache once. The result of this is a [Stream] that will
    /// produce the corresponding lock guards. If items are locked, the [Stream] will
    /// produce them as they become unlocked and can be locked by the stream.
    ///
    /// The returned stream is `async` and therefore may return items much later than
    /// when this function was called, but it only returns the entries that existed at
    /// the time this function was called. Any items that were added since the call to
    /// this function will not be returned by the stream, and any items that were
    /// deleted since the function call will still be returned by the stream.
    ///
    /// TODO Test that this doesn't lock the whole map while the stream hasn't gotten
    /// all locks yet and still allows locking/unlocking locks.
    ///
    /// TODO Test
    pub async fn lock_all_entries(
        &self,
    ) -> impl Stream<Item = <Self as Lockable<K, V>>::Guard<'_>> {
        LockableMapImpl::lock_all(&self.map_impl).await
    }

    /// Lock all entries of the cache once. The result of this is a [Stream] that will
    /// produce the corresponding lock guards. If items are locked, the [Stream] will
    /// produce them as they become unlocked and can be locked by the stream.
    ///
    /// This is identical to [LockableLruCache::lock_all_entries], but but it works on
    /// an `Arc<LockableLruCache>` instead of a [LockableLruCache] and returns a
    /// [Lockable::OwnedGuard] that binds its lifetime to the [LockableLruCache] in that
    /// [Arc]. Such a [Lockable::OwnedGuard] can be more easily moved around or cloned.
    ///
    /// The returned stream is `async` and therefore may return items much later than
    /// when this function was called, but it only returns the entries that existed at
    /// the time this function was called. Any items that were added since the call to
    /// this function will not be returned by the stream, and any items that were
    /// deleted since the function call will still be returned by the stream.
    ///
    /// TODO Test that this doesn't lock the whole map while the stream hasn't gotten
    /// all locks yet and still allows locking/unlocking locks.
    ///
    /// TODO Test
    pub async fn lock_all_entries_owned(
        self: &Arc<Self>,
    ) -> impl Stream<Item = <Self as Lockable<K, V>>::OwnedGuard> {
        LockableMapImpl::lock_all(Arc::clone(self)).await
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
