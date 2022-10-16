use std::borrow::{Borrow, BorrowMut};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use tokio::sync::OwnedMutexGuard;

use super::error::TryInsertError;
use super::hooks::Hooks;
use super::lockable_map_impl::{FromInto, LockableMapImpl};
use super::map_like::{ArcMutexMapLike, EntryValue};
// use crate::utils::locked_mutex_guard::LockedMutexGuard;

/// A RAII implementation of a scoped lock for locks from a [LockableHashMap](super::LockableHashMap) or [LockableLruCache](super::LockableLruCache). When this instance is dropped (falls out of scope), the lock will be unlocked.
#[must_use = "if unused the Mutex will immediately unlock"]
pub struct GuardImpl<M, V, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    pool: P,
    key: M::K,
    // Invariant: Is always Some(LockedMutexGuard) unless in the middle of destruction
    guard: Option<OwnedMutexGuard<EntryValue<M::V>>>,
    _hooks: PhantomData<H>,
    _v: PhantomData<V>,
}

impl<'a, M, V, H, P> GuardImpl<M, V, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    pub(super) fn new(pool: P, key: M::K, guard: OwnedMutexGuard<EntryValue<M::V>>) -> Self {
        Self {
            pool,
            key,
            guard: Some(guard),
            _hooks: PhantomData,
            _v: PhantomData,
        }
    }

    #[inline]
    fn _guard(&self) -> &OwnedMutexGuard<EntryValue<M::V>> {
        self.guard
            .as_ref()
            .expect("The self.guard field must always be set unless this was already destructed")
    }

    #[inline]
    fn _guard_mut(&mut self) -> &mut OwnedMutexGuard<EntryValue<M::V>> {
        self.guard
            .as_mut()
            .expect("The self.guard field must always be set unless this was already destructed")
    }

    /// TODO Docs
    #[inline]
    pub fn key(&self) -> &M::K {
        &self.key
    }

    #[inline]
    pub(super) fn value_raw(&self) -> Option<&M::V> {
        self._guard().value.as_ref()
    }

    /// TODO Docs
    #[inline]
    pub fn value(&self) -> Option<&V> {
        // We're returning Option<&V> instead of &Option<V> so that
        // user code can't change the Option from None to Some or the other
        // way round. They should use Self::insert() and Self::remove() for that.
        self.value_raw().map(|v| v.borrow())
    }

    /// TODO Docs
    #[inline]
    pub fn value_mut(&mut self) -> Option<&mut V> {
        // We're returning Option<&M::V> instead of &Option<M::V> so that
        // user code can't change the Option from None to Some or the other
        // way round. They should use Self::insert() and Self::remove() for that.
        self._guard_mut().value.as_mut().map(|v| v.borrow_mut())
    }

    /// TODO Docs
    /// TODO Test return value
    #[inline]
    pub fn remove(&mut self) -> Option<V> {
        // Setting this to None will cause Lockable::_unlock() to remove it
        let removed_value = self._guard_mut().value.take();
        removed_value.map(|v| v.fi_into())
    }

    /// TODO Docs
    /// TODO Test return value
    #[inline]
    pub fn insert(&mut self, value: V) -> Option<V> {
        let old_value = self._guard_mut().value.replace(M::V::fi_from(value));
        old_value.map(|v| v.fi_into())
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn try_insert(&mut self, value: V) -> Result<&mut V, TryInsertError<V>> {
        let guard = self._guard_mut();
        if guard.value.is_none() {
            guard.value = Some(M::V::fi_from(value));
            Ok(&mut *guard
                .value
                .as_mut()
                .expect("We just created this item")
                .borrow_mut())
        } else {
            Err(TryInsertError::AlreadyExists { value })
        }
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn value_or_insert_with(&mut self, value_fn: impl FnOnce() -> V) -> &mut V {
        let guard = self._guard_mut();
        if guard.value.is_none() {
            guard.value = Some(M::V::fi_from(value_fn()));
        }
        &mut *guard
            .value
            .as_mut()
            .expect("We just created this item if it didn't already exist")
            .borrow_mut()
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn value_or_insert(&mut self, value: V) -> &mut V {
        self.value_or_insert_with(move || value)
    }

    /// TODO Docs
    /// TODO Test
    #[inline]
    pub fn pool(&self) -> &P {
        &self.pool
    }
}

impl<M, V, H, P> Drop for GuardImpl<M, V, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    fn drop(&mut self) {
        let guard = self
            .guard
            .take()
            .expect("The self.guard field must always be set unless this was already destructed");
        self.pool.borrow()._unlock(&self.key, guard);
    }
}

impl<M, V, H, P> Debug for GuardImpl<M, V, H, P>
where
    M: ArcMutexMapLike,
    H: Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GuardImpl({:?})", self.key)
    }
}
