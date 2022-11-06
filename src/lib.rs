//! TODO Crate level documentation

#![deny(missing_docs)]

mod guard;
mod hooks;
mod limit;
mod lockable_map_impl;
mod lockable_trait;
mod map_like;
mod never;

#[cfg(test)]
mod tests;

mod lockable_hash_map;
#[cfg(feature = "lru")]
mod lockable_lru_cache;

pub use guard::{Guard, TryInsertError};
pub use limit::{AsyncLimit, SyncLimit};
pub use lockable_hash_map::LockableHashMap;
#[cfg(feature = "lru")]
pub use lockable_lru_cache::LockableLruCache;
pub use lockable_trait::Lockable;
pub use never::{InfallibleUnwrap, Never};
