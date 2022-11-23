//! The [lockable](https://crates.io/crates/lockable) library offers thread-safe
//! HashMap (see [LockableHashMap]) and LruCache (see [LockableLruCache]) types
//! where individual keys can be locked/unlocked, even if there is no entry for
//! this key in the map. This is great for synchronizing access to an underlying
//! key-value store or for building cache data structures on top of such
//! key-value stores.
//!
//! ```
//! use lockable::{AsyncLimit, LockableLruCache};
//!
//! let lockable_cache: LockableLruCache<i64, String> = LockableLruCache::new();
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let entry1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
//! let entry2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
//!
//! // This next line would cause a deadlock or panic because `4` is already locked on this thread
//! // let entry3 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
//!
//! // After dropping the corresponding guard, we can lock it again
//! std::mem::drop(entry1);
//! let entry3 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
//! # Ok::<(), lockable::Never>(())}).unwrap();
//! ```

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
mod time;
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
