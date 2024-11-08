use derive_more::{Display, Error};
use std::borrow::Borrow;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::marker::PhantomData;

use crate::lockable_map_impl::{EntryValue, LockableMapConfig};
use crate::utils::primary_arc::ReplicaOwnedMutexGuard;

use super::lockable_map_impl::LockableMapImpl;

/// A RAII implementation of a scoped lock for locks from a [LockableHashMap](super::LockableHashMap) or [LockableLruCache](super::LockableLruCache). When this instance is dropped (falls out of scope), the lock will be unlocked.
#[must_use = "if unused the Mutex will immediately unlock"]
pub struct Guard<K, V, C, P>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
{
    map: P,
    key: K,
    // Invariant: Is always Some(OwnedMutexGuard) unless in the middle of destruction
    guard: Option<ReplicaOwnedMutexGuard<EntryValue<C::WrappedV<V>>>>,
    _c: PhantomData<C>,
    _v: PhantomData<V>,
}

impl<K, V, C, P> Guard<K, V, C, P>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
{
    pub(super) fn new(
        map: P,
        key: K,
        guard: ReplicaOwnedMutexGuard<EntryValue<C::WrappedV<V>>>,
    ) -> Self {
        Self {
            map,
            key,
            guard: Some(guard),
            _c: PhantomData,
            _v: PhantomData,
        }
    }

    #[inline]
    fn _guard(&self) -> &ReplicaOwnedMutexGuard<EntryValue<C::WrappedV<V>>> {
        self.guard
            .as_ref()
            .expect("The self.guard field must always be set unless this was already destructed")
    }

    #[inline]
    fn _guard_mut(&mut self) -> &mut ReplicaOwnedMutexGuard<EntryValue<C::WrappedV<V>>> {
        self.guard
            .as_mut()
            .expect("The self.guard field must always be set unless this was already destructed")
    }

    /// Returns the key of the entry that was locked with this guard.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    /// assert_eq!(4, *guard.key());
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }

    #[inline]
    pub(super) fn value_raw(&self) -> Option<&C::WrappedV<V>> {
        self._guard().value.as_ref()
    }

    /// Returns the value of the entry that was locked with this guard.
    ///
    /// If the locked entry didn't exist, then this returns None, but the guard still represents a lock on this key
    /// and no other thread or task can lock the same key.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Entry doesn't exist yet
    ///     assert_eq!(None, guard.value());
    ///
    ///     // Insert the entry
    ///     guard.insert(String::from("Hello World"));
    /// }
    /// {
    ///     let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Now this entry exists
    ///     assert_eq!(Some(&String::from("Hello World")), guard.value());
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn value(&self) -> Option<&V> {
        // We're returning Option<&V> instead of &Option<V> so that
        // user code can't change the Option from None to Some or the other
        // way round. They should use Self::insert() and Self::remove() for that.
        self.value_raw().map(C::borrow_value)
    }

    /// Returns the value of the entry that was locked with this guard.
    ///
    /// If the locked entry didn't exist, then this returns None, but the guard still represents a lock on this key
    /// and no other thread or task can lock the same key.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Entry doesn't exist yet
    ///     assert_eq!(None, guard.value_mut());
    ///
    ///     // Insert the entry
    ///     guard.insert(String::from("Hello World"));
    /// }
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Modify the value
    ///     *guard.value_mut().unwrap() = String::from("New Value");
    /// }
    /// {
    ///     let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Now it has the new value
    ///     assert_eq!(Some(&String::from("New Value")), guard.value());
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn value_mut(&mut self) -> Option<&mut V> {
        // We're returning Option<&M::V> instead of &Option<M::V> so that
        // user code can't change the Option from None to Some or the other
        // way round. They should use Self::insert() and Self::remove() for that.
        self._guard_mut().value.as_mut().map(C::borrow_value_mut)
    }

    /// Removes the entry this guard has locked from the map.
    ///
    /// If the entry existed, its value is returned. If the entry didn't exist, [None] is returned.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Insert the entry
    ///     guard.insert(String::from("Hello World"));
    /// }
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // The value exists
    ///     assert_eq!(Some(&String::from("Hello World")), guard.value());
    ///
    ///     // Remove the value
    ///     guard.remove();
    /// }
    /// {
    ///     let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Now the value doesn't exist anymore
    ///     assert_eq!(None, guard.value());
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn remove(&mut self) -> Option<V> {
        // Setting this to None will cause Lockable::_unlock() to remove it
        let removed_value = self._guard_mut().value.take();
        removed_value.map(C::unwrap_value)
    }

    /// Inserts a value for the entry this guard has locked to the map.
    ///
    /// If the entry existed already, its old value is returned. If the entry didn't exist yet, [None] is returned.
    /// In both cases, the map will contain the new value after the call.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Insert the entry
    ///     let prev_entry = guard.insert(String::from("Hello World"));
    ///
    ///     // The value didn't exist previously
    ///     assert_eq!(None, prev_entry);
    /// }
    /// {
    ///     let guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Now the value exists
    ///     assert_eq!(Some(&String::from("Hello World")), guard.value());
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn insert(&mut self, value: V) -> Option<V> {
        let new_value = self.map.borrow().config().wrap_value(value);
        let old_value = self._guard_mut().value.replace(new_value);
        old_value.map(C::unwrap_value)
    }

    /// Inserts a value for the entry this guard has locked to the map if it didn't exist yet.
    /// If it already existed, this call returns [TryInsertError::AlreadyExists] instead.
    ///
    /// This function also returns a mutable reference to the new entry, which can be used to further modify it.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Insert the entry
    ///     let insert_result = guard.try_insert(String::from("Hello World"));
    ///     assert!(insert_result.is_ok());
    /// }
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // We cannot insert it again because it already exists
    ///     let insert_result = guard.try_insert(String::from("Hello World"));
    ///     assert!(insert_result.is_err());
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn try_insert(&mut self, value: V) -> Result<&mut V, TryInsertError<V>> {
        let config = self.map.borrow().config().clone();
        let guard = self._guard_mut();
        if guard.value.is_none() {
            let new_value = config.wrap_value(value);
            guard.value = Some(new_value);
            Ok(C::borrow_value_mut(
                &mut *guard.value.as_mut().expect("We just created this item"),
            ))
        } else {
            Err(TryInsertError::AlreadyExists { value })
        }
    }

    /// Returns a mutable reference to the value of the entry this guard has locked.
    ///
    /// If the entry doesn't exist, then `value_fn` is invoked to create it, the value
    /// is added to the map, and then a mutable reference to it is returned.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Entry doesn't exist yet, `value_or_insert_with` will create it
    ///     let value = guard.value_or_insert_with(|| String::from("Old Value"));
    ///     assert_eq!(&String::from("Old Value"), value);
    /// }
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Since the entry already exists, `value_or_insert_with` will not create it
    ///     // but return the existing value instead.
    ///     let value = guard.value_or_insert_with(|| String::from("New Value"));
    ///     assert_eq!(&String::from("Old Value"), value);
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn value_or_insert_with(&mut self, value_fn: impl FnOnce() -> V) -> &mut V {
        let config = self.map.borrow().config().clone();
        let guard = self._guard_mut();
        if guard.value.is_none() {
            let new_value = config.wrap_value(value_fn());
            guard.value = Some(new_value);
        }
        C::borrow_value_mut(
            &mut *guard
                .value
                .as_mut()
                .expect("We just created this item if it didn't already exist"),
        )
    }

    /// Returns a mutable reference to the value of the entry this guard has locked.
    ///
    /// If the entry doesn't exist, then `value` is inserted into the map for this entry,
    /// and then a mutable reference to it is returned.
    ///
    /// Examples
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let lockable_map = LockableHashMap::<i64, String>::new();
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Entry doesn't exist yet, `value_or_insert_with` will create it
    ///     let value = guard.value_or_insert(String::from("Old Value"));
    ///     assert_eq!(&String::from("Old Value"), value);
    /// }
    /// {
    ///     let mut guard = lockable_map.async_lock(4, AsyncLimit::no_limit()).await?;
    ///
    ///     // Since the entry already exists, `value_or_insert_with` will not create it
    ///     // but return the existing value instead.
    ///     let value = guard.value_or_insert(String::from("New Value"));
    ///     assert_eq!(&String::from("Old Value"), value);
    /// }
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    #[inline]
    pub fn value_or_insert(&mut self, value: V) -> &mut V {
        self.value_or_insert_with(move || value)
    }
}

impl<K, V, C, P> Drop for Guard<K, V, C, P>
where
    K: Eq + PartialEq + Hash + Clone,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
{
    fn drop(&mut self) {
        let guard = self
            .guard
            .take()
            .expect("The self.guard field must always be set unless this was already destructed");
        self.map.borrow()._unlock(&self.key, guard);
    }
}

impl<K, V, C, P> Debug for Guard<K, V, C, P>
where
    K: Eq + PartialEq + Hash + Clone + Debug,
    C: LockableMapConfig + Clone,
    P: Borrow<LockableMapImpl<K, V, C>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Guard({:?})", self.key)
    }
}

/// This error is thrown by [Guard::try_insert] if the entry already exists
#[derive(Error, Debug, Display, PartialEq, Eq, Hash, Clone, Copy)]
pub enum TryInsertError<V> {
    /// The entry couldn't be inserted because it already exists
    #[display("The entry couldn't be inserted because it already exists")]
    AlreadyExists {
        /// The value that was attempted to be inserted
        value: V,
    },
}

#[cfg(test)]
mod tests {
    use crate::{LockableHashMap, SyncLimit};

    #[test]
    fn test_debug() {
        let map = LockableHashMap::<i64, String>::new();
        let guard = map.blocking_lock(4, SyncLimit::no_limit()).unwrap();
        assert_eq!("Guard(4)", format!("{:?}", guard));
    }
}
