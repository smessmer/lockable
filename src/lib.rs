//! TODO Crate level documentation

// TODO rustfmt on all code examples
// TODO Figure out which functions actually should or shouldn't be #[inline]

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod guard;
mod hooks;
mod limit;
mod lockable_map_impl;
mod lockable_trait;
mod map_like;
mod never;
mod utils;

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
