//! The [lockable](https://crates.io/crates/lockable) library offers thread-safe
//! HashMap (see [LockableHashMap](crate::lockable_hash_map::LockableHashMap)),
//! LruCache (see [LockableLruCache](crate::lockable_lru_cache::LockableLruCache))
//! and LockPool (see [LockPool](crate::lockpool::LockPool)) types. In all of these
//! dat atypes, individual keys can be locked/unlocked, even if there is no entry
//! for this key in the map or cache.
//!
//! This can be very useful for synchronizing access to an underlying key-value
//! store or for building cache data structures on top of such a key-value store.
//!
//! ## LRU cache example
//! This example builds a simple LRU cache and locks some entries.
//! ```
#![cfg_attr(
    not(feature = "lru"),
    doc = "```
```ignore"
)]
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
//! This example builds a simple lock pool using the [LockPool](crate::lockpool::LockPool)
//! data structure. A lock pool is a pool of keyable locks. This can be used if
//! you don't need a cache but just some way to synchronize access to an underlying
//! resource.
//! ```
//! use lockable::LockPool;
//!
//! let lockpool = LockPool::new();
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let guard1 = lockpool.async_lock(4).await;
//! let guard2 = lockpool.async_lock(5).await;
//!
//! // This next line would wait until the lock gets released,
//! // which in this case would cause a deadlock because we're
//! // on the same thread.
//! // let guard3 = lockpool.async_lock(4).await;
//!
//! // After dropping the corresponding guard, we can lock it again
//! std::mem::drop(guard1);
//! let guard3 = lockpool.async_lock(4).await;
//! # Ok::<(), lockable::Never>(())}).unwrap();
//! ```
//!
//! ## HashMap example
//! If you need a lockable key-value store but don't need the LRU ordering,
//! you can use [LockableHashMap](crate::lockable_hash_map::LockableHashMap).
//! ```
//! use lockable::{AsyncLimit, LockableHashMap};
//!
//! let lockable_map = LockableHashMap::<i64, String>::new();
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//!
//! // Insert an entry
//! lockable_map.async_lock(4, AsyncLimit::no_limit())
//!     .await?
//!     .insert(String::from("Value"));
//!
//! // Hold a lock on a different entry
//! let guard = lockable_map.async_lock(5, AsyncLimit::no_limit())
//!     .await?;
//!
//! // This next line would wait until the lock gets released,
//! // which in this case would cause a deadlock because we're
//! // on the same thread
//! // let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit())
//! //    .await?;
//!
//! // After dropping the corresponding guard, we can lock it again
//! std::mem::drop(guard);
//! let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit())
//!     .await?;
//! # Ok::<(), lockable::Never>(())}).unwrap();
//! ```
//!
//! ## Crate Features
//! - `lru`: Enables the [LockableLruCache](crate::lockable_lru_cache::LockableLruCache)
//!    type which adds a dependency on the [lru](https://crates.io/crates/lru) crate.

// TODO Figure out which functions actually should or shouldn't be #[inline]

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// We need to add explicit links because our `gen_readme.sh` script requires them.
#![allow(rustdoc::redundant_explicit_links)]

mod guard;
mod hooks;
mod limit;
mod lockable_map_impl;
mod lockable_trait;
mod map_like;
mod utils;

#[cfg(test)]
mod tests;

mod lockable_hash_map;
#[cfg(feature = "lru")]
mod lockable_lru_cache;
mod lockpool;

pub use guard::{Guard, TryInsertError};
pub use limit::{AsyncLimit, SyncLimit};
pub use lockable_hash_map::LockableHashMap;
#[cfg(feature = "lru")]
pub use lockable_lru_cache::LockableLruCache;
pub use lockable_trait::Lockable;
pub use lockpool::LockPool;
pub use utils::never::{InfallibleUnwrap, Never};
