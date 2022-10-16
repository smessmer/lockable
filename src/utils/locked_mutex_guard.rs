use owning_ref_lockable::OwningHandle;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::fmt;
use tokio::sync::{Mutex, MutexGuard, TryLockError};

/// A LockedMutexGuard carries an Arc<Mutex<T>> together with a MutexGuard locking that Data.
/// 
/// The implementation is based on a [tokio::sync::Mutex] and locking behavior matches the behavior
/// of that mutex.
pub struct LockedMutexGuard<'a, T: 'a> {
    mutex_and_guard: OwningHandle<Arc<Mutex<T>>, MutexGuard<'a, T>>,
}

impl<'a, T: 'a> LockedMutexGuard<'a, T> {
    /// Lock the given mutex and return a [LockedMutexGuard] pointing to the data behind the mutex.
    /// 
    /// See [tokio::sync::Mutex::blocking_lock] for behavioral details.
    pub fn blocking_lock(mutex: Arc<Mutex<T>>) -> Self {
        let mutex_and_guard = OwningHandle::new_with_fn(mutex, |mutex: *const Mutex<T>| {
            let mutex: &Mutex<T> = unsafe { &*mutex };
            mutex.blocking_lock()
        });
        Self { mutex_and_guard }
    }

    /// Lock the given mutex and return a [LockedMutexGuard] pointing to the data behind the mutex.
    /// 
    /// See [tokio::sync::Mutex::lock] for behavioral details.
    pub async fn async_lock(mutex: Arc<Mutex<T>>) -> LockedMutexGuard<'a, T> {
        let mutex_and_guard = OwningHandle::new_with_async_fn(mutex, |mutex: *const Mutex<T>| {
            let mutex: &Mutex<T> = unsafe { &*mutex };
            mutex.lock()
        })
        .await;
        Self { mutex_and_guard }
    }

    /// Try to lock the given mutex and return an [Ok] of [LockedMutexGuard] if successful, and an [Err] of [TryLockError] if unsuccessful because it is already locked.
    /// 
    /// See [tokio::sync::Mutex::try_lock] for behavioral details.
    pub fn try_lock(mutex: Arc<Mutex<T>>) -> Result<Self, TryLockError> {
        let mutex_and_guard = OwningHandle::try_new(mutex, |mutex: *const Mutex<T>| {
            let mutex: &Mutex<T> = unsafe { &*mutex };
            let guard = mutex.try_lock()?;
            Ok(guard)
        })?;
        Ok(Self { mutex_and_guard })
    }
}

impl<'a, T> Deref for LockedMutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.mutex_and_guard
    }
}

impl<'a, T> DerefMut for LockedMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.mutex_and_guard
    }
}

impl <'a, T> fmt::Debug for LockedMutexGuard<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LockedMutexGuard")
            .field(self.deref())
            .finish()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_i32_blocking_lock() {
        let mutex = Arc::new(Mutex::new(5i32));
        let guard = LockedMutexGuard::blocking_lock(mutex);
        assert_eq!(5i32, *guard);
    }

    #[test]
    fn test_string_blocking_lock() {
        let mutex = Arc::new(Mutex::new(String::from("Hello World")));
        let guard = LockedMutexGuard::blocking_lock(mutex);
        assert_eq!(String::from("Hello World"), *guard);
    }

    #[test]
    fn test_str_blocking_lock() {
        let mutex = Arc::new(Mutex::new("Hello World"));
        let guard = LockedMutexGuard::blocking_lock(mutex);
        assert_eq!("Hello World", *guard);
    }

    #[test]
    fn test_ref_blocking_lock() {
        let string = String::from("Hello World");
        let mutex = Arc::new(Mutex::new(&string));
        let guard = LockedMutexGuard::blocking_lock(mutex);
        assert_eq!("Hello World", *guard);
    }

    #[test]
    fn test_mut_ref_blocking_lock() {
        let mut string = String::from("Hello");
        {
            let mutex = Arc::new(Mutex::new(&mut string));
            let mut guard = LockedMutexGuard::blocking_lock(mutex);
            assert_eq!("Hello", *guard);
            guard.push_str(" World");
        }
        assert_eq!(String::from("Hello World"), string);
    }

    #[tokio::test]
    async fn test_i32_async_lock() {
        let mutex = Arc::new(Mutex::new(5i32));
        let guard = LockedMutexGuard::async_lock(mutex).await;
        assert_eq!(5i32, *guard);
    }

    #[tokio::test]
    async fn test_string_async_lock() {
        let mutex = Arc::new(Mutex::new(String::from("Hello World")));
        let guard = LockedMutexGuard::async_lock(mutex).await;
        assert_eq!(String::from("Hello World"), *guard);
    }

    #[tokio::test]
    async fn test_str_async_lock() {
        let mutex = Arc::new(Mutex::new("Hello World"));
        let guard = LockedMutexGuard::async_lock(mutex).await;
        assert_eq!("Hello World", *guard);
    }

    #[tokio::test]
    async fn test_ref_async_lock() {
        let string = String::from("Hello World");
        let mutex = Arc::new(Mutex::new(&string));
        let guard = LockedMutexGuard::async_lock(mutex).await;
        assert_eq!("Hello World", *guard);
    }

    #[tokio::test]
    async fn test_async_lock_can_be_held_across_an_await_point() {
        let mutex = Arc::new(Mutex::new(String::from("Hello World")));
        let guard = LockedMutexGuard::async_lock(mutex).await;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert_eq!(String::from("Hello World"), *guard);
    }

    #[tokio::test]
    async fn test_mut_ref_async_lock() {
        let mut string = String::from("Hello");
        {
            let mutex = Arc::new(Mutex::new(&mut string));
            let mut guard = LockedMutexGuard::async_lock(mutex).await;
            assert_eq!("Hello", *guard);
            guard.push_str(" World");
        }
        assert_eq!(String::from("Hello World"), string);
    }

    #[tokio::test]
    async fn test_i32_try_lock() {
        let mutex = Arc::new(Mutex::new(5i32));
        let guard = LockedMutexGuard::try_lock(mutex).unwrap();
        assert_eq!(5i32, *guard);
    }

    #[test]
    fn test_string_try_lock() {
        let mutex = Arc::new(Mutex::new(String::from("Hello World")));
        let guard = LockedMutexGuard::try_lock(mutex).unwrap();
        assert_eq!(String::from("Hello World"), *guard);
    }

    #[test]
    fn test_str_try_lock() {
        let mutex = Arc::new(Mutex::new("Hello World"));
        let guard = LockedMutexGuard::try_lock(mutex).unwrap();
        assert_eq!("Hello World", *guard);
    }

    #[test]
    fn test_ref_try_lock() {
        let string = String::from("Hello World");
        let mutex = Arc::new(Mutex::new(&string));
        let guard = LockedMutexGuard::try_lock(mutex).unwrap();
        assert_eq!("Hello World", *guard);
    }

    #[test]
    fn test_mut_ref_try_lock() {
        let mut string = String::from("Hello");
        {
            let mutex = Arc::new(Mutex::new(&mut string));
            let mut guard = LockedMutexGuard::try_lock(mutex).unwrap();
            assert_eq!("Hello", *guard);
            guard.push_str(" World");
        }
        assert_eq!(String::from("Hello World"), string);
    }

    #[test]
    fn test_try_lock_when_already_locked() {
        let mutex = Arc::new(Mutex::new(5i32));
        let _guard = LockedMutexGuard::try_lock(Arc::clone(&mutex)).unwrap();
        LockedMutexGuard::try_lock(mutex).unwrap_err();
    }
}
