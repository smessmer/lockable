//! The [lockable](https://crates.io/crates/lockable) library offers thread-safe
//! HashMap (see [struct@LockableHashMap]) and LruCache (see [struct@LockableLruCache])
//! types where individual keys can be locked/unlocked, even if there is no entry
//! for this key in the map.
//!
//! This can be very useful for synchronizing access to an underlying key-value
//! store or for building cache data structures on top of such a key-value store.
//!
//! ## LRU cache example
//! This example builds a simple LRU cache and locks some entries.
//!
#![cfg_attr(feature = "lru", doc = "```")]
#![cfg_attr(not(feature = "lru"), doc = "```ignore")]
//! use lockable::{AsyncLimit, LockableLruCache};
//!
//! let lockable_cache = LockableLruCache::<i64, String>::new();
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//!
//! // Insert an entry
//! lockable_cache.async_lock(4, AsyncLimit::no_limit())
//!     .await?
//!     .insert(String::from("Value"));
//!
//! // Hold a lock on a different entry
//! let guard = lockable_cache.async_lock(5, AsyncLimit::no_limit())
//!     .await?;
//!
//! // This next line would wait until the lock gets released,
//! // which in this case would cause a deadlock because we're
//! // on the same thread
//! // let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit())
//! //    .await?;
//!
//! // After dropping the corresponding guard, we can lock it again
//! std::mem::drop(guard);
//! let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit())
//!     .await?;
//! # Ok::<(), lockable::Never>(())}).unwrap();
//! ```
//!
//! ## Lockpool example
//! This example builds a simple lock pool using the [struct@LockableHashMap] data
//! structure. A lock pool is a pool of keyable locks. In this example, the entries
//! don't have a value assigned to them and the lock pool is only used to synchronize
//! access to some keyed resource.
//! ```
//! use lockable::{AsyncLimit, LockableHashMap};
//!
//! let lockable_cache = LockableHashMap::<i64, ()>::new();
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let guard1 = lockable_cache.async_lock(4, AsyncLimit::no_limit())
//!     .await?;
//! let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit())
//!     .await?;
//!
//! // This next line would wait until the lock gets released,
//! // which in this case would cause a deadlock because we're
//! // on the same thread.
//! // let guard3 = lockable_cache.async_lock(4, AsyncLimit::no_limit())
//! //    .await?;
//!
//! // After dropping the corresponding guard, we can lock it again
//! std::mem::drop(guard1);
//! let guard3 = lockable_cache.async_lock(4, AsyncLimit::no_limit())
//!     .await?;
//! # Ok::<(), lockable::Never>(())}).unwrap();
//! ```
//!
//! ## Crate Features
//! - `lru`: Enables the [struct@LockableLruCache] type which adds a dependency
//!    on the [lru](https://crates.io/crates/lru) crate.

// TODO Figure out which functions actually should or shouldn't be #[inline]
// TODO Add benchmarks, maybe take the one from the `lockpool` crate.

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
