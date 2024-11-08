use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard, TryLockError};

/// A [PrimaryArc] is an [Arc] that can only be cloned from the [PrimaryArc] instance. Each clone will be a [ReplicaArc] that cannot be cloned further.
pub struct PrimaryArc<T> {
    inner: Arc<T>,
}

impl<T> PrimaryArc<T> {
    /// Create a new [PrimaryArc] from a value.
    #[inline]
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(value),
        }
    }

    /// Clone the [PrimaryArc] into a [ReplicaArc].
    #[inline]
    pub fn clone(&self) -> ReplicaArc<T> {
        ReplicaArc {
            inner: Arc::clone(&self.inner),
        }
    }

    #[inline]
    pub fn num_replicas(&self) -> usize {
        Arc::strong_count(&self.inner) - 1
    }

    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Arc<T>> {
        Arc::try_unwrap(this.inner)
    }
}

/// A [ReplicaArc] is an [Arc] that cannot be cloned further.
pub struct ReplicaArc<T> {
    inner: Arc<T>,
}

impl<T> Deref for ReplicaArc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> ReplicaArc<Mutex<T>> {
    /// Lock the mutex and return a [OwnedMutexGuard].
    #[inline]
    pub async fn lock_owned(self) -> ReplicaOwnedMutexGuard<T> {
        ReplicaOwnedMutexGuard {
            inner: self.inner.lock_owned().await,
        }
    }

    #[inline]
    pub fn try_lock_owned(self) -> Result<ReplicaOwnedMutexGuard<T>, TryLockError> {
        self.inner
            .try_lock_owned()
            .map(|inner| ReplicaOwnedMutexGuard { inner })
    }

    #[inline]
    pub fn blocking_lock_owned(self) -> ReplicaOwnedMutexGuard<T> {
        ReplicaOwnedMutexGuard {
            inner: self.inner.blocking_lock_owned(),
        }
    }
}

/// A [OwnedMutexGuard] that can only be created from a `PrimaryArc<Mutex>`.
/// It holds a clone of the original [PrimaryArc] but cannot be cloned further.
pub struct ReplicaOwnedMutexGuard<T> {
    inner: OwnedMutexGuard<T>,
}

impl<T> Deref for ReplicaOwnedMutexGuard<T> {
    type Target = OwnedMutexGuard<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for ReplicaOwnedMutexGuard<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
