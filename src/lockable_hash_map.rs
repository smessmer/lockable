use futures::stream::Stream;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::guard::Guard;
use super::hooks::NoopHooks;
use super::limit::{AsyncLimit, SyncLimit};
use super::lockable_map_impl::LockableMapImpl;
use super::lockable_trait::Lockable;
use super::map_like::{ArcMutexMapLike, EntryValue};

type MapImpl<K, V> = HashMap<K, Arc<tokio::sync::Mutex<EntryValue<V>>>>;

impl<K, V> ArcMutexMapLike for MapImpl<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type K = K;
    type V = V;
    type ItemIter<'a> = std::collections::hash_map::Iter<'a, K, Arc<Mutex<EntryValue<V>>>>
    where
        K: 'a,
        V: 'a;

    fn new() -> Self {
        Self::new()
    }

    fn len(&self) -> usize {
        self.iter().len()
    }

    fn get_or_insert_none(&mut self, key: &Self::K) -> &Arc<Mutex<EntryValue<Self::V>>> {
        // TODO Is there a way to only clone the key when the entry doesn't already exist?
        //      Might be possible with the upcoming RawEntry API. If we do that, we may
        //      even be able to remove the `Clone` bound from `K` everywhere in this library.
        self.entry(key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(EntryValue { value: None })))
    }

    fn get(&mut self, key: &Self::K) -> Option<&Arc<Mutex<EntryValue<Self::V>>>> {
        HashMap::get(self, key)
    }

    fn remove(&mut self, key: &Self::K) -> Option<Arc<Mutex<EntryValue<Self::V>>>> {
        self.remove(key)
    }

    fn iter(&self) -> Self::ItemIter<'_> {
        HashMap::iter(self)
    }
}

/// A threadsafe hash map where individual keys can be locked/unlocked, even if there is no entry for this key in the map.
/// It initially considers all keys as "unlocked", but they can be locked and if a second thread tries to acquire a lock
/// for the same key, they will have to wait.
///
/// ```
/// use lockable::{AsyncLimit, Lockable, LockableHashMap};
///
/// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let entry1 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
/// let entry2 = lockable_map.async_lock(5, AsyncLimit::no_limit()).await?;
///
/// // This next line would cause a deadlock or panic because `4` is already locked on this thread
/// // let entry3 = lockable_map.async_lock(4).await;
///
/// // After dropping the corresponding guard, we can lock it again
/// std::mem::drop(entry1);
/// let entry3 = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// The guards holding a lock for an entry can be used to insert that entry to the hash map, remove
/// it from the hash map, or to modify the value of an existing entry.
///
/// ```
/// use lockable::{AsyncLimit, Lockable, LockableHashMap};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// async fn insert_entry(
///     lockable_map: &LockableHashMap<i64, String>,
/// ) -> Result<(), lockable::Never> {
///     let mut entry_guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.insert(String::from("Hello World"));
///     Ok(())
/// }
///
/// async fn remove_entry(
///     lockable_map: &LockableHashMap<i64, String>,
/// ) -> Result<(), lockable::Never> {
///     let mut entry_guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
///     entry_guard.remove();
///     Ok(())
/// }
///
/// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
/// assert_eq!(
///     None,
///     lockable_map
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// insert_entry(&lockable_map).await;
/// assert_eq!(
///     Some(&String::from("Hello World")),
///     lockable_map
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// remove_entry(&lockable_map).await;
/// assert_eq!(
///     None,
///     lockable_map
///         .async_lock(4, AsyncLimit::no_limit())
///         .await?
///         .value()
/// );
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
///
/// You can use an arbitrary type to index hash map entries by, as long as that type implements [PartialEq] + [Eq] + [Hash] + [Clone].
///
/// ```
/// use lockable::{AsyncLimit, Lockable, LockableHashMap};
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// struct CustomLockKey(u32);
///
/// let lockable_map: LockableHashMap<CustomLockKey, String> = LockableHashMap::new();
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let guard = lockable_map
///     .async_lock(CustomLockKey(4), AsyncLimit::no_limit())
///     .await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// Under the hood, a [LockableHashMap] is a [std::collections::HashMap] of [Mutex](tokio::sync::Mutex)es, with some logic making sure that
/// empty entries can also be locked and that there aren't any race conditions when adding or removing entries.
#[derive(Debug)]
pub struct LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    map_impl: LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
}

impl<K, V> Lockable<K, V> for LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type Guard<'a> = Guard<
        MapImpl<K, V>,
        V,
        NoopHooks,
        &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
    > where
        K: 'a,
        V: 'a;

    type OwnedGuard = Guard<MapImpl<K, V>, V, NoopHooks, Arc<LockableHashMap<K, V>>>;

    type SyncLimit<'a, OnEvictFn, E> = SyncLimit<
        MapImpl<K, V>,
        V,
        NoopHooks,
        &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
        E,
        OnEvictFn,
    > where
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>,
        K: 'a,
        V: 'a;

    type SyncLimitOwned<OnEvictFn, E> = SyncLimit<
        MapImpl<K, V>,
        V,
        NoopHooks,
        Arc<LockableHashMap<K, V>>,
        E,
        OnEvictFn,
    > where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

    type AsyncLimit<'a, OnEvictFn, E, F> = AsyncLimit<
        MapImpl<K, V>,
        V,
        NoopHooks,
        &'a LockableMapImpl<MapImpl<K, V>, V, NoopHooks>,
        E,
        F,
        OnEvictFn,
    > where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F,
        K: 'a,
        V: 'a;

    type AsyncLimitOwned<OnEvictFn, E, F> = AsyncLimit<
        MapImpl<K, V>,
        V,
        NoopHooks,
        Arc<LockableHashMap<K, V>>,
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
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F,
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
        self.map_impl.into_entries_unordered()
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

impl<K, V> LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    /// Create a new hash map with no entries and no locked keys.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
    ///
    /// let lockable_map: LockableHashMap<i64, String> = LockableHashMap::new();
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            map_impl: LockableMapImpl::new(),
        }
    }
}

impl<K, V> Default for LockableHashMap<K, V>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// We implement Borrow<LockableMapImpl> for Arc<LockableHashMap> because that's the way, our LockableMapImpl can "see through" an instance
// of LockableHashMap to get to its "self" parameter in calls like LockableMapImpl::blocking_lock_owned.
// Since LockableMapImpl is a type private to this crate, this Borrow doesn't escape crate boundaries.
impl<K, V> Borrow<LockableMapImpl<MapImpl<K, V>, V, NoopHooks>> for Arc<LockableHashMap<K, V>>
where
    K: Eq + PartialEq + Hash + Clone,
{
    fn borrow(&self) -> &LockableMapImpl<MapImpl<K, V>, V, NoopHooks> {
        &self.map_impl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instantiate_lockable_tests;

    instantiate_lockable_tests!(LockableHashMap);
}
