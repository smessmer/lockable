use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use super::hooks::Hooks;
use super::lockable_map_impl::LockableMapImpl;
use super::map_like::{ArcMutexMapLike, EntryValue};
use crate::utils::locked_mutex_guard::LockedMutexGuard;

/// A RAII implementation of a scoped lock for locks from a [LockableHashMap](super::LockableHashMap) or [LockableLruCache](super::LockableLruCache). When this instance is dropped (falls out of scope), the lock will be unlocked.
#[must_use = "if unused the Mutex will immediately unlock"]
pub struct GuardImpl<M, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    P: Borrow<LockableMapImpl<M, H>>,
{
    pool: P,
    key: M::K,
    // Invariant: Is always Some(LockedMutexGuard) unless in the middle of destruction
    guard: Option<LockedMutexGuard<EntryValue<M::V>>>,
    _hooks: PhantomData<H>,
}

impl<'a, M, H, P> GuardImpl<M, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    P: Borrow<LockableMapImpl<M, H>>,
{
    pub(super) fn new(pool: P, key: M::K, guard: LockedMutexGuard<EntryValue<M::V>>) -> Self {
        Self {
            pool,
            key,
            guard: Some(guard),
            _hooks: PhantomData,
        }
    }

    #[inline]
    fn _guard(&self) -> &LockedMutexGuard<EntryValue<M::V>> {
        self.guard
            .as_ref()
            .expect("The self.guard field must always be set unless this was already destructed")
    }

    #[inline]
    fn _guard_mut(&mut self) -> &mut LockedMutexGuard<EntryValue<M::V>> {
        self.guard
            .as_mut()
            .expect("The self.guard field must always be set unless this was already destructed")
    }

    /// TODO Test
    /// TODO Docs
    #[inline]
    pub fn key(&self) -> &M::K {
        &self.key
    }

    /// TODO Test
    /// TODO Docs
    #[inline]
    pub fn value(&self) -> Option<&M::V> {
        // We're returning Option<&M::V> instead of &Option<M::V> so that
        // user code can't change the Option from None to Some or the other
        // way round. They should use Self::insert() and Self::remove() for that.
        self._guard().value.as_ref()
    }

    /// TODO Test
    /// TODO Docs
    #[inline]
    pub fn value_mut(&mut self) -> Option<&mut M::V> {
        // We're returning Option<&M::V> instead of &Option<M::V> so that
        // user code can't change the Option from None to Some or the other
        // way round. They should use Self::insert() and Self::remove() for that.
        self._guard_mut().value.as_mut()
    }

    /// TODO Test
    /// TODO Docs
    #[inline]
    pub fn remove(&mut self) -> Option<M::V> {
        // Setting this to None will cause Lockable::_unlock() to remove it
        self._guard_mut().value.take()
    }

    /// TODO Test
    /// TODO Docs
    #[inline]
    pub fn insert(&mut self, value: M::V) -> Option<M::V> {
        self._guard_mut().value.replace(value)
    }
}

impl<M, H, P> Drop for GuardImpl<M, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    P: Borrow<LockableMapImpl<M, H>>,
{
    fn drop(&mut self) {
        let guard = self
            .guard
            .take()
            .expect("The self.guard field must always be set unless this was already destructed");
        self.pool.borrow()._unlock(&self.key, guard);
    }
}

impl<M, H, P> Debug for GuardImpl<M, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    P: Borrow<LockableMapImpl<M, H>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GuardImpl({:?})", self.key)
    }
}
