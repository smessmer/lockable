use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard};

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

impl<T> ReplicaArc<T> {
    #[inline]
    pub fn num_replicas(&self) -> usize {
        Arc::strong_count(&self.inner) - 1
    }
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
    pub fn blocking_lock_owned(self) -> ReplicaOwnedMutexGuard<T> {
        ReplicaOwnedMutexGuard {
            inner: self.inner.blocking_lock_owned(),
        }
    }

    #[inline]
    pub fn try_lock_owned(self) -> Result<ReplicaOwnedMutexGuard<T>, Self> {
        // TODO This `Arc::clone` is violating our invariant. Even though we immediately drop it or ourselves, it's still a clone.
        //      It's currently fine because all call sites make sure to call [LockableMapImpl::_delete_if_unlocked_none_and_nobody_waiting_for_lock],
        //      so the `None` entry isn't left behind, but it still violates the invariant.

        // This function is in `PrimaryArc` not `ReplicaArc` because it needs to call `Arc::clone`.
        let locked = Arc::clone(&self.inner).try_lock_owned();
        match locked {
            Ok(inner) => Ok(ReplicaOwnedMutexGuard { inner }),
            Err(_) => Err(self),
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
