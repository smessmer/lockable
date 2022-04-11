//! TODO Crate level documentation

#![deny(missing_docs)]

mod error;
mod guard;
mod hooks;
mod lockable_map_impl;
mod map_like;
mod utils;

#[cfg(test)]
mod tests;

mod lockable_hash_map;
#[cfg(feature = "lru")]
mod lockable_lru_cache;

pub use error::TryLockError;
pub use guard::GuardImpl;
pub use lockable_hash_map::{HashMapGuard, HashMapOwnedGuard, LockableHashMap};
#[cfg(feature = "lru")]
pub use lockable_lru_cache::{LockableLruCache, LruGuard, LruOwnedGuard};
