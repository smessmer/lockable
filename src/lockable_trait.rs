use std::future::Future;

/// A common trait for both [LockableHashMap](crate::LockableHashMap) and [LockableLruCache](crate::LockableLruCache) that offers some common
/// functionalities.
pub trait Lockable<K, V> {
    /// A non-owning guard holding a lock for an entry in a [LockableHashMap](crate::LockableHashMap) or a [LockableLruCache](crate::LockableLruCache).
    /// This guard is created via [LockableHashMap::blocking_lock](crate::LockableHashMap::blocking_lock), [LockableHashMap::async_lock](crate::LockableHashMap::async_lock)
    /// or [LockableHashMap::try_lock](crate::LockableHashMap::try_lock), or the corresponding [LockableLruCache](crate::LockableLruCache) methods,
    /// and its lifetime is bound to the lifetime of the [LockableHashMap](crate::LockableHashMap)/[LockableLruCache](crate::LockableLruCache).
    ///
    /// See the documentation of [Guard](crate::Guard) for methods.
    ///
    /// Examples:
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let map = LockableHashMap::<usize, String>::new();
    /// let guard: <LockableHashMap<usize, String> as Lockable<usize, String>>::Guard<'_> =
    ///     map.async_lock(1, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    type Guard<'a>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    /// A owning guard holding a lock for an entry in a [LockableHashMap](crate::LockableHashMap) or a [LockableLruCache](crate::LockableLruCache).
    /// This guard is created via [LockableHashMap::blocking_lock_owned](crate::LockableHashMap::blocking_lock_owned), [LockableHashMap::async_lock_owned](crate::LockableHashMap::async_lock_owned)
    /// or [LockableHashMap::try_lock_owned](crate::LockableHashMap::try_lock_owned), or the corresponding [LockableLruCache](crate::LockableLruCache) methods,
    /// and its lifetime is bound to the lifetime of the [LockableHashMap](crate::LockableHashMap)/[LockableLruCache](crate::LockableLruCache) within its [Arc](std::sync::Arc).
    ///
    /// See the documentation of [Guard](crate::Guard) for methods.
    ///
    /// Examples:
    /// -----
    /// ```
    /// use lockable::{AsyncLimit, Lockable, LockableHashMap};
    /// use std::sync::Arc;
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async {
    /// let map = Arc::new(LockableHashMap::<usize, String>::new());
    /// let guard: <LockableHashMap<usize, String> as Lockable<usize, String>>::OwnedGuard =
    ///     map.async_lock_owned(1, AsyncLimit::no_limit()).await?;
    /// # Ok::<(), lockable::Never>(())}).unwrap();
    /// ```
    type OwnedGuard;

    /// TODO Documentation
    type SyncLimit<'a, OnEvictFn, E>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> Result<(), E>;

    /// TODO Documentation
    type SyncLimitOwned<OnEvictFn, E>
    where
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> Result<(), E>;

    /// TODO Documentation
    type AsyncLimit<'a, OnEvictFn, E, F>
    where
        Self: 'a,
        K: 'a,
        V: 'a,
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::Guard<'a>>) -> F;

    /// TODO Documentation
    type AsyncLimitOwned<OnEvictFn, E, F>
    where
        F: Future<Output = Result<(), E>>,
        OnEvictFn: FnMut(Vec<Self::OwnedGuard>) -> F;
}
