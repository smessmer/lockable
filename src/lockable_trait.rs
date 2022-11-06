/// A common trait for both [LockableHashMap](crate::LockableHashMap) and [LockableLruCache](crate::LockableLruCache) that offers some common
/// functionalities.
pub trait Lockable<K, V> {
    /// A non-owning guard holding a lock for an entry in a [LockableHashMap](crate::LockableHashMap) or a [LockableLruCache](crate::LockableLruCache).
    /// This guard is created via [LockableHashMap::blocking_lock](crate::LockableHashMap::blocking_lock), [LockableHashMap::async_lock](crate::LockableHashMap::async_lock)
    /// or [LockableHashMap::try_lock](crate::LockableHashMap::try_lock), or the corresponding [LockableLruCache](crate::LockableLruCache) methods,
    /// and its lifetime is bound to the lifetime of the [LockableHashMap](crate::LockableHashMap)/[LockableLruCache](crate::LockableLruCache).
    ///
    /// See the documentation of [Guard](crate::Guard) for methods.
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
    type OwnedGuard;
}
