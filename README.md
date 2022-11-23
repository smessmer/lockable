[![Build Status](https://github.com/smessmer/lockable/actions/workflows/ci.yml/badge.svg)](https://github.com/smessmer/lockable/actions/workflows/ci.yml)
[![Latest Version](https://img.shields.io/crates/v/lockable.svg)](https://crates.io/crates/lockable)
[![docs.rs](https://docs.rs/lockable/badge.svg)](https://docs.rs/lockable)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/smessmer/lockable/blob/master/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-APACHE-blue.svg)](https://github.com/smessmer/lockable/blob/master/LICENSE-APACHE)
[![codecov](https://codecov.io/gh/smessmer/lockable/branch/master/graph/badge.svg?token=FRSBH7YYA9)](https://codecov.io/gh/smessmer/lockable)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

# lockable

<!-- cargo-rdme start -->

The [lockable](https://crates.io/crates/lockable) library offers thread-safe
HashMap (see [LockableHashMap](https://docs.rs/lockable/latest/lockable/lockable_hash_map/struct.LockableHashMap.html)),
LruCache (see [LockableLruCache](https://docs.rs/lockable/latest/lockable/lockable_lru_cache/struct.LockableLruCache.html))
and LockPool (see [LockPool](https://docs.rs/lockable/latest/lockable/lockpool/struct.LockPool.html)) types. In all of these
dat atypes, individual keys can be locked/unlocked, even if there is no entry
for this key in the map or cache.

This can be very useful for synchronizing access to an underlying key-value
store or for building cache data structures on top of such a key-value store.

### LRU cache example
This example builds a simple LRU cache and locks some entries.
```rust
use lockable::{AsyncLimit, LockableLruCache};

let lockable_cache = LockableLruCache::<i64, String>::new();

// Insert an entry
lockable_cache.async_lock(4, AsyncLimit::no_limit())
    .await?
    .insert(String::from("Value"));

// Hold a lock on a different entry
let guard = lockable_cache.async_lock(5, AsyncLimit::no_limit())
    .await?;

// This next line would wait until the lock gets released,
// which in this case would cause a deadlock because we're
// on the same thread
// let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit())
//    .await?;

// After dropping the corresponding guard, we can lock it again
std::mem::drop(guard);
let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit())
    .await?;
```

### Lockpool example
This example builds a simple lock pool using the [LockPool](https://docs.rs/lockable/latest/lockable/lockpool/struct.LockPool.html)
data structure. A lock pool is a pool of keyable locks. This can be used if
you don't need a cache but just some way to synchronize access to an underlying
resource.
```rust
use lockable::LockPool;

let lockpool = LockPool::new();
let guard1 = lockpool.async_lock(4).await;
let guard2 = lockpool.async_lock(5).await;

// This next line would wait until the lock gets released,
// which in this case would cause a deadlock because we're
// on the same thread.
// let guard3 = lockpool.async_lock(4).await;

// After dropping the corresponding guard, we can lock it again
std::mem::drop(guard1);
let guard3 = lockpool.async_lock(4).await;
```

### HashMap example
If you need a lockable key-value store but don't need the LRU ordering,
you can use [LockableHashMap](https://docs.rs/lockable/latest/lockable/lockable_hash_map/struct.LockableHashMap.html).
```rust
use lockable::{AsyncLimit, LockableHashMap};

let lockable_map = LockableHashMap::<i64, String>::new();

// Insert an entry
lockable_map.async_lock(4, AsyncLimit::no_limit())
    .await?
    .insert(String::from("Value"));

// Hold a lock on a different entry
let guard = lockable_map.async_lock(5, AsyncLimit::no_limit())
    .await?;

// This next line would wait until the lock gets released,
// which in this case would cause a deadlock because we're
// on the same thread
// let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit())
//    .await?;

// After dropping the corresponding guard, we can lock it again
std::mem::drop(guard);
let guard2 = lockable_map.async_lock(5, AsyncLimit::no_limit())
    .await?;
```

### Crate Features
- `lru`: Enables the [LockableLruCache](https://docs.rs/lockable/latest/lockable/lockable_lru_cache/struct.LockableLruCache.html)
   type which adds a dependency on the [lru](https://crates.io/crates/lru) crate.

<!-- cargo-rdme end -->

License: MIT OR Apache-2.0
