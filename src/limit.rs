use anyhow::Result;
use std::borrow::{Borrow, BorrowMut};
use std::future::Future;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use crate::guard::GuardImpl;
use crate::hooks::Hooks;
use crate::lockable_map_impl::{FromInto, LockableMapImpl};
use crate::map_like::ArcMutexMapLike;

// TODO This approach of enforcing a limit on the number of total entries during a lock() call isn't great.
// It means that even if all entries are currently locked and in use and no entry is really "in the cache",
// it would still block adding new entries. Better would be to put a limit on the number of *unlocked* entries,
// i.e. entries that are actually in the cache. But since entries are becoming unlocked in the drop implementation
// of a guard, and evicting often needs to be async, it would require us to have async drop code for the guard.
// Rust doesn't really support that yet.

/// TODO Docs
pub enum AsyncLimit<M, V, H, P, F, OnEvictFn>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
    F: Future<Output = Result<()>>,
    OnEvictFn: Fn(Vec<GuardImpl<M, V, H, P>>) -> F,
{
    /// TODO Docs
    Unbounded {
        /// TODO Docs
        _m: PhantomData<M>,
        /// TODO Docs
        _v: PhantomData<V>,
        /// TODO Docs
        _h: PhantomData<H>,
        /// TODO Docs
        _p: PhantomData<P>,
        /// TODO Docs
        _f: PhantomData<F>,
        /// TODO Docs
        _o: PhantomData<OnEvictFn>,
    },
    /// TODO Docs
    Bounded {
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
        std::future::Ready<Result<()>>,
        fn(Vec<GuardImpl<M, V, H, P>>) -> std::future::Ready<Result<()>>,
    >
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    /// TODO Docs
    pub fn unbounded() -> Self {
        Self::Unbounded {
            _m: PhantomData,
            _v: PhantomData,
            _h: PhantomData,
            _p: PhantomData,
            _f: PhantomData,
            _o: PhantomData,
        }
    }
}

/// TODO Docs
pub enum SyncLimit<M, V, H, P, OnEvictFn>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
    OnEvictFn: Fn(Vec<GuardImpl<M, V, H, P>>) -> Result<()>,
{
    /// TODO Docs
    Unbounded {
        /// TODO Docs
        _m: PhantomData<M>,
        /// TODO Docs
        _v: PhantomData<V>,
        /// TODO Docs
        _h: PhantomData<H>,
        /// TODO Docs
        _p: PhantomData<P>,
        /// TODO Docs
        _o: PhantomData<OnEvictFn>,
    },
    /// TODO Docs
    Bounded {
        /// TODO Docs
        max_entries: NonZeroUsize,
        /// TODO Docs
        on_evict: OnEvictFn,
    },
}

impl<M, V, H, P> SyncLimit<M, V, H, P, fn(Vec<GuardImpl<M, V, H, P>>) -> Result<()>>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    /// TODO Docs
    pub fn unbounded() -> Self {
        Self::Unbounded {
            _m: PhantomData,
            _v: PhantomData,
            _h: PhantomData,
            _p: PhantomData,
            _o: PhantomData,
        }
    }
}
