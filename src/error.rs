use std::fmt::Debug;
use thiserror::Error;

/// Errors that can be thrown by [LockableLruCache::try_lock](super::LockableLruCache::try_lock) and [LockableHashMap::try_lock](super::LockableHashMap::try_lock).
#[derive(Error, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum TryLockError {
    /// The lock could not be acquired at this time because the operation would otherwise block
    #[error(
        "The lock could not be acquired at this time because the operation would otherwise block"
    )]
    WouldBlock,
}
