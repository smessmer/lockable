use std::borrow::Borrow;
use std::future::Future;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use crate::guard::Guard;
use crate::lockable_map_impl::{LockableMapConfig, LockableMapImpl};
use crate::utils::never::Never;

/// An instance of this enum defines a limit on the number of entries in a [LockableLruCache](crate::LockableLruCache) or a [LockableHashMap](crate::LockableHashMap).
/// It can be used to cause old entries to be evicted if a limit on the number of entries is exceeded in a call to the following functions:
///
/// | [LockableLruCache](crate::LockableLruCache)                            | [LockableHashMap](crate::LockableHashMap)                            |
/// |------------------------------------------------------------------------|----------------------------------------------------------------------|
/// | [async_lock](crate::LockableLruCache::async_lock)                      | [async_lock](crate::LockableHashMap::async_lock)                     |
/// | [async_lock_owned](crate::LockableLruCache::async_lock_owned)          | [async_lock_owned](crate::LockableHashMap::async_lock_owned)         |
/// | [try_lock_async](crate::LockableLruCache::try_lock_async)              | [try_lock_async](crate::LockableHashMap::try_lock_async)             |
/// | [try_lock_owned_async](crate::LockableLruCache::try_lock_owned_async)  | [try_lock_owned_async](crate::LockableHashMap::try_lock_owned_async) |
///
/// The purpose of this class is the same as the purpose of [SyncLimit], but it has an `async` callback
/// to evict entries instead of a synchronous callback.
///
/// # Example (without limit)
/// ```
/// use lockable::{AsyncLimit, LockableHashMap};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let lockable_map = LockableHashMap::<i64, String>::new();
/// let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
///
/// # Example (with limit)
#[cfg_attr(feature = "lru", doc = "```")]
#[cfg_attr(not(feature = "lru"), doc = "```ignore")]
/// use lockable::{LockableLruCache, AsyncLimit};
/// use std::cell::RefCell;
/// use std::rc::Rc;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
///
/// let lockable_map = LockableLruCache::<i64, String>::new();
/// // Insert two entries
/// lockable_map.async_lock(4, AsyncLimit::no_limit()).await?.insert("Value 4".to_string());
/// lockable_map.async_lock(5, AsyncLimit::no_limit()).await?.insert("Value 5".to_string());
///
/// // Lock a third entry but set a limit of 2 entries
/// // Collect any evicted entries in the `evicted` vector
/// let evicted = Rc::new(RefCell::new(vec![]));
/// let guard = lockable_map.async_lock(4, AsyncLimit::SoftLimit {
///   max_entries: 2.try_into().unwrap(),
///   on_evict: |entries| {
///     let evicted = Rc::clone(&evicted);
///     async move {
///       for mut entry in entries {
///         evicted.borrow_mut().push(*entry.key());
///         // We have to remove the entry from the map, the map doesn't do it for us.
///         // If we don't remove it, we could end up in an infinite loop because the
///         // map is still above the limit.
///         entry.remove();
///       }
///       Ok::<(), lockable::Never>(())
///     }
///   }
/// }).await?;
///
/// // We evicted the entry with key 4 because it was the least recently used
/// assert_eq!(evicted.borrow().len(), 1);
/// assert!(evicted.borrow().contains(&4));
/// # Ok::<(), lockable::Never>(())}).unwrap();
/// ```
pub enum AsyncLimit<K, V, C, P, E, F, OnEvictFn>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
    F: Future<Output = Result<(), E>>,
    OnEvictFn: FnMut(Vec<Guard<K, V, C, P>>) -> F,
{
    /// This enum variant specifies that there is no limit on the number of entries.
    /// If the locking operation causes a new entry to be created, it will be created
    /// without evicting anything.
    ///
    /// Use [AsyncLimit::no_limit] to create an instance.
    NoLimit {
        #[doc(hidden)]
        _k: PhantomData<K>,
        #[doc(hidden)]
        _v: PhantomData<V>,
        #[doc(hidden)]
        _c: PhantomData<C>,
        #[doc(hidden)]
        _p: PhantomData<P>,
    },
    /// Setting a [AsyncLimit::SoftLimit] for a locking call means that there is a limit on the number of entries.
    /// Entries that either have a value or that don't have a value but are currently locked count towards that limit,
    /// see [LockableHashMap::num_entries_or_locked](crate::LockableHashMap::num_entries_or_locked)
    /// or [LockableLruCache::num_entries_or_locked](crate::LockableLruCache::num_entries_or_locked).
    ///
    /// If the locking call would cause the limit to be exceeded, the given `on_evict` callback will be called with
    /// some other entries. Those entries are already locked for you and `on_evict` is expected to delete those entries.
    /// It is possible that `on_evict` is called multiple times if the limit is still exceeded after the call.
    /// The `on_evict` callback is responsible for deleting those entries, [LockableHashMap](crate::LockableHashMap)
    /// and [LockableLruCache](crate::LockableLruCache) will not delete any entries for you. If `on_evict` doesn't delete
    /// any entries, you will end up in an infinite loop because the total number of entries never gets below the limit.
    ///
    /// There is one exception, and this is why this is called a "soft" limit. If a call to a locking function has
    /// a [AsyncLimit::SoftLimit] set but there are no entries in the cache that are currently unlocked and that could
    /// be passed to an `on_evict` callback, i.e. if the limit is exceeded but at the same time all entries are currently
    /// locked, then exceeding the limit will be allowed, `on_evict` will not be called, and the locking function
    /// will successfully lock return. This is to protect against a deadlock that would otherwise be hard to protect
    /// against where multiple threads or tasks lock different keys and want to lock more keys, but the limit would block
    /// them and no thread/task wants to give up their held locks. Note that this only protects against a deadlock
    /// caused by the limit. If those threads or tasks are trying to lock each others locks, you will still run into
    /// a deadlock.
    ///
    /// If this is used in a [LockableLruCache](crate::LockableLruCache), then `on_evict` will be called with the
    /// least recently used entries, to allow for LRU style pruning.
    SoftLimit {
        /// The maximal allowed number of entries in the cache. If this number gets exceeded by a locking call with
        /// this [AsyncLimit] set, the `on_evict` callback will be called.
        max_entries: NonZeroUsize,
        /// This callback will be called if `max_entries` is exceeded. It will be passed a list of guards for entries
        /// and it can be used to clean up or flush data from those entries. Afterwards, these entries will be removed
        /// from the [LockableHashMap](crate::LockableHashMap) or [LockableLruCache](crate::LockableLruCache).
        ///
        /// It is `async` and can do asynchronous operations in its implementation.
        ///
        /// It is possible for some of the guards to have `None` values when the entry was removed concurrently by
        /// another thread while we are trying to evict it.
        on_evict: OnEvictFn,
        // TODO Once Rust has async drop, user code should do this in V::drop and we should remove `on_evict`. But we need to be careful that the entry stays locked while drop is running, so that user code doing a cache writeback can rely that no other thread tries to read it while it's written back.
    },
}

impl<K, V, C, P>
    AsyncLimit<
        K,
        V,
        C,
        P,
        Never,
        std::future::Ready<Result<(), Never>>,
        fn(Vec<Guard<K, V, C, P>>) -> std::future::Ready<Result<(), Never>>,
    >
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
{
    /// See [AsyncLimit::NoLimit]. This helper function can be used
    /// to create an instance of [AsyncLimit::NoLimit] without having
    /// to specify all the [PhantomData] members.
    pub fn no_limit() -> Self {
        Self::NoLimit {
            _k: PhantomData,
            _v: PhantomData,
            _c: PhantomData,
            _p: PhantomData,
        }
    }
}

/// An instance of this enum defines a limit on the number of entries in a [LockableLruCache](crate::LockableLruCache) or a [LockableHashMap](crate::LockableHashMap).
/// It can be used to cause old entries to be evicted if a limit on the number of entries is exceeded in a call to the following functions:
///
/// | [LockableLruCache](crate::LockableLruCache)                            | [LockableHashMap](crate::LockableHashMap)                          |
/// |------------------------------------------------------------------------|--------------------------------------------------------------------|
/// | [blocking_lock](crate::LockableLruCache::blocking_lock)                | [blocking_lock](crate::LockableHashMap::blocking_lock)             |
/// | [blocking_lock_owned](crate::LockableLruCache::blocking_lock_owned)    | [blocking_lock_owned](crate::LockableHashMap::blocking_lock_owned) |
/// | [try_lock](crate::LockableLruCache::try_lock)                          | [try_lock](crate::LockableHashMap::try_lock)                       |
/// | [try_lock_owned](crate::LockableLruCache::try_lock_owned)              | [try_lock_owned](crate::LockableHashMap::try_lock_owned)           |
///
/// The purpose of this class is the same as the purpose of [AsyncLimit], but it has a synchronous callback
/// to evict entries instead of an `async` callback.
///
/// # Example (without limit)
/// ```
/// use lockable::{LockableHashMap, SyncLimit};
///
/// # (|| {
/// let lockable_map = LockableHashMap::<i64, String>::new();
/// let guard = lockable_map.blocking_lock(4, SyncLimit::no_limit())?;
/// # Ok::<(), lockable::Never>(())})().unwrap();
/// ```
///
/// # Example (with limit)
#[cfg_attr(feature = "lru", doc = "```")]
#[cfg_attr(not(feature = "lru"), doc = "```ignore")]
/// use lockable::{LockableLruCache, SyncLimit};
/// use std::cell::RefCell;
/// use std::rc::Rc;
///
/// # (|| {
/// let lockable_map = LockableLruCache::<i64, String>::new();
///
/// // Insert two entries
/// lockable_map.blocking_lock(4, SyncLimit::no_limit())?.insert("Value 4".to_string());
/// lockable_map.blocking_lock(5, SyncLimit::no_limit())?.insert("Value 5".to_string());
///
/// // Lock a third entry but set a limit of 2 entries
/// // Collect any evicted entries in the `evicted` vector
/// let mut evicted = vec![];
/// let guard = lockable_map.blocking_lock(4, SyncLimit::SoftLimit {
///   max_entries: 2.try_into().unwrap(),
///   on_evict: |entries| {
///     for mut entry in entries {
///       evicted.push(*entry.key());
///       // We have to remove the entry from the map, the map doesn't do it for us.
///       // If we don't remove it, we could end up in an infinite loop because the
///       // map is still above the limit.
///       entry.remove();
///     }
///     Ok::<(), lockable::Never>(())
///   }
/// })?;
///
/// // We evicted the entry with key 4 because it was the least recently used
/// assert_eq!(evicted.len(), 1);
/// assert!(evicted.contains(&4));
/// # Ok::<(), lockable::Never>(())})().unwrap();
/// ```
pub enum SyncLimit<K, V, C, P, E, OnEvictFn>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
    OnEvictFn: FnMut(Vec<Guard<K, V, C, P>>) -> Result<(), E>,
{
    /// This enum variant specifies that there is no limit on the number of entries.
    /// If the locking operation causes a new entry to be created, it will be created
    /// without evicting anything.
    ///
    /// Use [SyncLimit::no_limit] to create an instance.
    NoLimit {
        #[doc(hidden)]
        _k: PhantomData<K>,
        #[doc(hidden)]
        _v: PhantomData<V>,
        #[doc(hidden)]
        _c: PhantomData<C>,
        #[doc(hidden)]
        _p: PhantomData<P>,
    },
    /// Setting a [SyncLimit::SoftLimit] for a locking call means that there is a limit on the number of entries.
    /// Entries that either have a value or that don't have a value but are currently locked count towards that limit,
    /// see [LockableHashMap::num_entries_or_locked](crate::LockableHashMap::num_entries_or_locked) or
    /// [LockableLruCache::num_entries_or_locked](crate::LockableLruCache::num_entries_or_locked).
    ///
    /// If the locking call would cause the limit to be exceeded, the given `on_evict` callback will be called with
    /// some other entries. Those entries are already locked for you and `on_evict` is expected to delete those entries.
    /// It is possible that `on_evict` is called multiple times if the limit is still exceeded after the call.
    /// The `on_evict` callback is responsible for deleting those entries, [LockableHashMap](crate::LockableHashMap)
    /// and [LockableLruCache](crate::LockableLruCache) will not delete any entries for you. If `on_evict` doesn't
    /// delete any entries, you will end up in an infinite loop because the total number of entries never gets below the
    /// limit.
    ///
    /// There is one exception, and this is why this is called a "soft" limit. If a call to a locking function has
    /// a [SyncLimit::SoftLimit] set but there are no entries in the cache that are currently unlocked and that could
    /// be passed to an `on_evict` callback, i.e. if the limit is exceeded but at the same time all entries are currently
    /// locked, then exceeding the limit will be allowed, `on_evict` will not be called, and the locking function
    /// will successfully lock return. This is to protect against a deadlock that would otherwise be hard to protect
    /// against where multiple threads or tasks lock different keys and want to lock more keys, but the limit would block
    /// them and no thread/task wants to give up their held locks. Note that this only protects against a deadlock
    /// caused by the limit. If those threads or tasks are trying to lock each others locks, you will still run into
    /// a deadlock.
    ///
    /// If this is used in a [LockableLruCache](crate::LockableLruCache), then `on_evict` will be called with the
    /// least recently used entries, to allow for LRU style pruning.
    SoftLimit {
        /// The maximal allowed number of entries in the cache. If this number gets exceeded by a locking call with
        /// this [SyncLimit] set, the `on_evict` callback will be called.
        max_entries: NonZeroUsize,
        /// This callback will be called if `max_entries` is exceeded. It will be passed a list of guards for entries
        /// and it will be expected to delete those entries from the [LockableHashMap](crate::LockableHashMap) or
        /// [LockableLruCache](crate::LockableLruCache) using [Guard::remove]. This callback can also do any operations
        /// you need to clean up or flush data from those entries before you delete them. It is not `async`. If you need
        /// an `async` callback, take a look at the functions taking [AsyncLimit] instead of [SyncLimit].
        on_evict: OnEvictFn,
    },
}

impl<K, V, C, P> SyncLimit<K, V, C, P, Never, fn(Vec<Guard<K, V, C, P>>) -> Result<(), Never>>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
{
    /// See [SyncLimit::NoLimit]. This helper function can be used
    /// to create an instance of [SyncLimit::NoLimit] without having
    /// to specify all the [PhantomData] members.
    pub fn no_limit() -> Self {
        Self::NoLimit {
            _k: PhantomData,
            _v: PhantomData,
            _c: PhantomData,
            _p: PhantomData,
        }
    }
}
