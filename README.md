[![Build Status](https://github.com/smessmer/lockable/actions/workflows/ci.yml/badge.svg)](https://github.com/smessmer/lockable/actions/workflows/ci.yml)
[![Latest Version](https://img.shields.io/crates/v/lockable.svg)](https://crates.io/crates/lockable)
[![docs.rs](https://docs.rs/lockable/badge.svg)](https://docs.rs/lockable)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/smessmer/lockable/blob/master/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-APACHE-blue.svg)](https://github.com/smessmer/lockable/blob/master/LICENSE-APACHE)
[![codecov](https://codecov.io/gh/smessmer/lockable/branch/master/graph/badge.svg?token=FRSBH7YYA9)](https://codecov.io/gh/smessmer/lockable)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

# lockable

The [lockable](https://crates.io/crates/lockable) library offers thread-safe
HashMap (see [LockableHashMap](crate::LockableHashMap)) and LruCache
(see [LockableLruCache](crate::LockableLruCache)) types where individual keys
can be locked/unlocked, even if there is no entry for this key in the map.

This can be very useful for synchronizing access to an underlying key-value
store or for building cache data structures on top of such a key-value store.

### LRU cache example
This example builds a simple LRU cache and locks some entries.

```rust
use lockable::{AsyncLimit, LockableLruCache};

let lockable_cache: LockableLruCache<i64, String> = LockableLruCache::new();

// Insert an entry
lockable_cache.async_lock(4, AsyncLimit::no_limit())
    .await?
    .insert(String::from("Value"));

// Hold a lock on a different entry
let guard = lockable_cache.async_lock(5, AsyncLimit::no_limit())
    .await?;

// This next line would cause a deadlock or panic because `5` is already locked on this thread
// let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;

// After dropping the corresponding guard, we can lock it again
std::mem::drop(guard);
let guard2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;
```

### Lockpool example
This example builds a simple lock pool using the [LockableHashMap](crate::LockableHashMap)
data structure. A lock pool is a pool of keyable locks. In this example, the entries
don't have a value assigned to them and the lock pool is only used to synchronize
access to some keyed resource.
```rust
use lockable::{AsyncLimit, LockableHashMap};

let lockable_cache: LockableHashMap<i64, ()> = LockableHashMap::new();
let entry1 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
let entry2 = lockable_cache.async_lock(5, AsyncLimit::no_limit()).await?;

// This next line would cause a deadlock or panic because `4` is already locked on this thread
// let entry3 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;

// After dropping the corresponding guard, we can lock it again
std::mem::drop(entry1);
let entry3 = lockable_cache.async_lock(4, AsyncLimit::no_limit()).await?;
```

License: MIT OR Apache-2.0
