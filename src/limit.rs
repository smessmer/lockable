use std::borrow::{Borrow, BorrowMut};
use std::future::Future;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use crate::guard::Guard;
use crate::hooks::Hooks;
use crate::lockable_map_impl::{FromInto, LockableMapImpl};
use crate::map_like::ArcMutexMapLike;
use crate::never::Never;

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
pub enum AsyncLimit<M, V, H, P, E, F, OnEvictFn>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
    F: Future<Output = Result<(), E>>,
    OnEvictFn: Fn(Vec<Guard<M, V, H, P>>) -> F,
{
    /// This enum variant specifies that there is no limit on the number of entries.
    /// If the locking operation causes a new entry to be created, it will be created
    /// without evicting anything.
    ///
    /// Use [AsyncLimit::no_limit] to create an instance.
    NoLimit {
        #[allow(missing_docs)]
        _m: PhantomData<M>,
        #[allow(missing_docs)]
        _v: PhantomData<V>,
        #[allow(missing_docs)]
        _h: PhantomData<H>,
        #[allow(missing_docs)]
        _p: PhantomData<P>,
        #[allow(missing_docs)]
        _o: PhantomData<OnEvictFn>,
    },
    /// TODO Docs
    SoftLimit {
        /// TODO Docs
        max_entries: NonZeroUsize,
        /// TODO Docs
        on_evict: OnEvictFn,
    },
}

impl<M, V, H, P>
    AsyncLimit<
        M,
        V,
        H,
        P,
        Never,
        std::future::Ready<Result<(), Never>>,
        fn(Vec<Guard<M, V, H, P>>) -> std::future::Ready<Result<(), Never>>,
    >
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    /// See [AsyncLimit::NoLimit]. This helper function can be used
    /// to create an instance of [AsyncLimit::NoLimit] without having
    /// to specify all the [PhantomData] members.
    pub fn no_limit() -> Self {
        Self::NoLimit {
            _m: PhantomData,
            _v: PhantomData,
            _h: PhantomData,
            _p: PhantomData,
            _o: PhantomData,
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
pub enum SyncLimit<M, V, H, P, E, OnEvictFn>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
    OnEvictFn: Fn(Vec<Guard<M, V, H, P>>) -> Result<(), E>,
{
    /// This enum variant specifies that there is no limit on the number of entries.
    /// If the locking operation causes a new entry to be created, it will be created
    /// without evicting anything.
    ///
    /// Use [SyncLimit::no_limit] to create an instance.
    NoLimit {
        #[allow(missing_docs)]
        _m: PhantomData<M>,
        #[allow(missing_docs)]
        _v: PhantomData<V>,
        #[allow(missing_docs)]
        _h: PhantomData<H>,
        #[allow(missing_docs)]
        _p: PhantomData<P>,
        #[allow(missing_docs)]
        _o: PhantomData<OnEvictFn>,
    },
    /// TODO Docs
    SoftLimit {
        /// TODO Docs
        max_entries: NonZeroUsize,
        /// TODO Docs
        on_evict: OnEvictFn,
    },
}

impl<M, V, H, P> SyncLimit<M, V, H, P, Never, fn(Vec<Guard<M, V, H, P>>) -> Result<(), Never>>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    /// See [SyncLimit::NoLimit]. This helper function can be used
    /// to create an instance of [SyncLimit::NoLimit] without having
    /// to specify all the [PhantomData] members.
    pub fn no_limit() -> Self {
        Self::NoLimit {
            _m: PhantomData,
            _v: PhantomData,
            _h: PhantomData,
            _p: PhantomData,
            _o: PhantomData,
        }
    }
}
