//! Reproductions of the bugs found in a correctness audit of this crate.
//!
//! Every test asserts the behavior a user would expect from the documentation. That means
//! every test in this file FAILS for as long as the bug it demonstrates exists. They are
//! marked `#[ignore]` so that a plain `cargo test` stays green. Run them with
//!
//! ```text
//! cargo test --test audit_repro -- --ignored --test-threads=1
//! ```
//!
//! This is an integration test on purpose: it only uses the public API, and the library is
//! compiled the way downstream users get it, i.e. without the internal invariant assertions.
//! With `--features slow_assertions`, some of these tests fail earlier, in the invariant check
//! ("Invariant 2 violated. Found an entry without ReplicaArcs that is None.").
//!
//! The reproductions that need a mocked clock are in `src/lockable_lru_cache.rs`, in the test
//! module `audit_repro`.

#![cfg(feature = "lru")]

use futures::stream::StreamExt;
use lockable::{
    AsyncLimit, InfallibleUnwrap, LockPool, LockableHashMap, LockableLruCache, SyncLimit,
};
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

#[derive(Debug)]
enum Outcome<T> {
    Completed(T),
    Panicked(String),
    /// The closure neither returned nor panicked within the timeout.
    Hung,
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        String::from("<non-string panic payload>")
    }
}

/// Runs `f` on a separate thread so that a deadlock inside `f` cannot hang the test binary.
/// A deadlocked thread is left behind and gets killed when the test binary exits.
fn run_with_watchdog<T: Send + 'static>(
    timeout: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> Outcome<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = catch_unwind(AssertUnwindSafe(f));
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(value)) => Outcome::Completed(value),
        Ok(Err(payload)) => Outcome::Panicked(panic_message(payload)),
        Err(_) => Outcome::Hung,
    }
}

/// Polls a future once, without registering a waker that does anything.
fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> std::task::Poll<F::Output> {
    let waker = futures::task::noop_waker();
    future.poll(&mut std::task::Context::from_waker(&waker))
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Leaves a stale entry for `key` behind: an entry that is neither locked nor has a value.
/// See `stale_entries::cancelled_async_lock` for why this happens.
async fn create_stale_entry(cache: &LockableLruCache<i64, String>, key: i64) {
    let guard = cache
        .async_lock(key, AsyncLimit::no_limit())
        .await
        .infallible_unwrap();
    let mut waiter = Box::pin(cache.async_lock(key, AsyncLimit::no_limit()));
    assert!(poll_once(waiter.as_mut()).is_pending());
    drop(guard);
    drop(waiter);
}

/// `LockableLruCache::lock_entries_unlocked_for_at_least` panics when it finds an entry that
/// is unlocked and has no value. Such an entry legitimately exists for a moment whenever
/// somebody else is just about to lock that key. The panic happens while the map-wide mutex is
/// held, and unwinding then tries to take that same mutex again: the thread hangs forever and
/// every other user of the cache blocks behind it.
///
/// With tokio older than 1.54 there is one more way into that state, and it needs neither a
/// second thread nor a stream: a task that runs out of its cooperative budget inside
/// `async_lock` yields before it queued up for the key. tokio 1.54 acquires the lock first.
mod scan_freezes_the_cache {
    use super::*;

    /// Everything here is alive and legal: a (not yet consumed) `lock_all_entries` stream
    /// exists, an entry gets removed, and then `lock_entries_unlocked_for_at_least` is called.
    #[test]
    #[ignore = "known bug"]
    fn with_a_pending_lock_all_entries_stream() {
        let outcome = run_with_watchdog(Duration::from_secs(10), || {
            current_thread_runtime().block_on(async {
                let cache = LockableLruCache::<i64, String>::new();
                cache
                    .async_lock(1, AsyncLimit::no_limit())
                    .await
                    .infallible_unwrap()
                    .insert(String::from("value"));

                let mut stream = cache.lock_all_entries().await;

                cache
                    .async_lock(1, AsyncLimit::no_limit())
                    .await
                    .infallible_unwrap()
                    .remove();

                let old_enough: Vec<i64> = cache
                    .lock_entries_unlocked_for_at_least(Duration::ZERO)
                    .map(|guard| *guard.key())
                    .collect();
                assert_eq!(Vec::<i64>::new(), old_enough);

                // The stream is still fine and correctly reports no entries
                assert!(stream.next().await.is_none());
            })
        });
        match outcome {
            Outcome::Completed(()) => {}
            Outcome::Panicked(msg) => panic!("library panicked: {msg}"),
            Outcome::Hung => panic!(
                "DEADLOCK: the thread calling lock_entries_unlocked_for_at_least never returned"
            ),
        }
    }

    /// Plain multi-threading, no async code involved: two threads lock and unlock a key that
    /// has no value, a third thread calls `lock_entries_unlocked_for_at_least`.
    /// This is a race, so the test can pass by accident. It usually hits within a few
    /// seconds, the test gives it twenty (and runs for all of them once the bug is fixed).
    #[test]
    #[ignore = "known bug"]
    fn race_with_blocking_lock() {
        let cache = Arc::new(LockableLruCache::<i64, String>::new());
        let stop = Arc::new(AtomicBool::new(false));
        let num_locked = Arc::new(AtomicUsize::new(0));
        let num_scans = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let cache = Arc::clone(&cache);
            let stop = Arc::clone(&stop);
            let num_locked = Arc::clone(&num_locked);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _guard = cache
                        .blocking_lock(1, SyncLimit::no_limit())
                        .infallible_unwrap();
                    num_locked.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        {
            let cache = Arc::clone(&cache);
            let stop = Arc::clone(&stop);
            let num_scans = Arc::clone(&num_scans);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _old_enough: Vec<_> = cache
                        .lock_entries_unlocked_for_at_least(Duration::ZERO)
                        .collect();
                    num_scans.fetch_add(1, Ordering::Relaxed);
                }
            });
        }

        // Watch whether the locking threads keep making progress
        let start = std::time::Instant::now();
        let mut last_seen = 0;
        let mut stalled_since: Option<std::time::Instant> = None;
        let mut failure = None;
        while start.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(100));
            let current = num_locked.load(Ordering::Relaxed);
            if current == last_seen {
                let since = *stalled_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() > Duration::from_secs(3) {
                    failure = Some(format!(
                        "no progress for 3s after {current} lock/unlock cycles and {} scans",
                        num_scans.load(Ordering::Relaxed),
                    ));
                    break;
                }
            } else {
                stalled_since = None;
                last_seen = current;
            }
        }
        stop.store(true, Ordering::Relaxed);
        if let Some(failure) = failure {
            panic!("DEADLOCK: {failure}");
        }
    }
}

/// A pending `async_lock` future and a `lock_all_entries` stream both hold a reference to the
/// entry they are going to lock. If they get dropped, nothing cleans up after them. When the
/// entry has no value at that point, it stays in the map: neither locked nor with a value.
mod stale_entries {
    use super::*;

    /// A task waiting in `async_lock` gets cancelled right after the previous holder
    /// released the key. (Polled by hand to make it deterministic.)
    #[tokio::test]
    #[ignore = "known bug"]
    async fn cancelled_async_lock() {
        let pool = LockPool::<i64>::new();
        let guard = pool.async_lock(1).await;

        // A second task starts waiting for the same key ...
        let mut waiter = Box::pin(pool.async_lock(1));
        assert!(poll_once(waiter.as_mut()).is_pending());

        // ... the first one releases the key ...
        drop(guard);
        // ... and the waiter gets cancelled before it is polled again.
        drop(waiter);

        assert_eq!(
            (Vec::<i64>::new(), 0),
            (pool.locked_keys(), pool.num_locked())
        );
    }

    /// Same with real tasks: the waiting task is cancelled through `select!`
    /// (think: request cancelled, client disconnected, shutdown signal).
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "known bug"]
    async fn cancelled_through_select() {
        let pool = Arc::new(LockPool::<i64>::new());
        let guard = pool.async_lock(1).await;

        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn({
            let pool = Arc::clone(&pool);
            async move {
                tokio::select! {
                    biased;
                    _ = cancel_rx => {}
                    _guard = pool.async_lock(1) => {}
                }
            }
        });
        // Let the task run until it waits for the lock
        tokio::task::yield_now().await;

        // Release the lock and cancel the waiting task before it gets to run again
        drop(guard);
        cancel_tx.send(()).unwrap();
        task.await.unwrap();

        assert_eq!(
            (Vec::<i64>::new(), 0),
            (pool.locked_keys(), pool.num_locked())
        );
    }

    /// Same, but the waiting task is aborted.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "known bug"]
    async fn cancelled_through_abort() {
        let pool = Arc::new(LockPool::<i64>::new());
        let guard = pool.async_lock(1).await;

        let task = tokio::spawn({
            let pool = Arc::clone(&pool);
            async move {
                let _guard = pool.async_lock(1).await;
            }
        });
        tokio::task::yield_now().await;

        drop(guard);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        assert_eq!(
            (Vec::<i64>::new(), 0),
            (pool.locked_keys(), pool.num_locked())
        );
    }

    /// The holder and the waiter are cancelled together, here by a timeout around both.
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "known bug"]
    async fn holder_and_waiter_cancelled_together() {
        let pool = LockPool::<i64>::new();
        let work = async {
            tokio::join!(
                async {
                    let _guard = pool.async_lock(1).await;
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                },
                async {
                    let _guard = pool.async_lock(1).await;
                },
            )
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(50), work)
                .await
                .is_err()
        );

        assert_eq!(
            (Vec::<i64>::new(), 0),
            (pool.locked_keys(), pool.num_locked())
        );
    }

    /// A `lock_all_entries` stream that is dropped before it was fully consumed, after an
    /// entry it has not returned yet got removed. (Consuming a part of the stream first makes
    /// no difference.)
    #[tokio::test]
    #[ignore = "known bug"]
    async fn dropped_lock_all_entries_stream() {
        let map = LockableHashMap::<i64, String>::new();
        map.async_lock(1, AsyncLimit::no_limit())
            .await
            .infallible_unwrap()
            .insert(String::from("value"));

        let stream = map.lock_all_entries().await;

        // Meanwhile, the entry gets removed through a regular guard.
        map.async_lock(1, AsyncLimit::no_limit())
            .await
            .infallible_unwrap()
            .remove();

        drop(stream);

        assert_eq!(
            (Vec::<i64>::new(), 0),
            (
                map.keys_with_entries_or_locked(),
                map.num_entries_or_locked()
            )
        );
    }

    /// No removal needed: it is enough that a key is locked, without a value, at the time
    /// the stream is created.
    #[tokio::test]
    #[ignore = "known bug"]
    async fn dropped_lock_all_entries_stream_and_a_key_without_value() {
        let map = LockableHashMap::<i64, String>::new();
        let guard = map
            .async_lock(1, AsyncLimit::no_limit())
            .await
            .infallible_unwrap();
        let stream = map.lock_all_entries().await;
        drop(guard);
        drop(stream);

        assert_eq!(
            (Vec::<i64>::new(), 0),
            (
                map.keys_with_entries_or_locked(),
                map.num_entries_or_locked()
            )
        );
    }

    /// Consequence 1: `into_entries_unordered` panics when it reaches the stale entry.
    #[tokio::test]
    #[ignore = "known bug"]
    async fn then_into_entries_unordered_panics() {
        let map = LockableHashMap::<i64, String>::new();
        map.async_lock(1, AsyncLimit::no_limit())
            .await
            .infallible_unwrap()
            .insert(String::from("value"));
        let stream = map.lock_all_entries().await;
        map.async_lock(1, AsyncLimit::no_limit())
            .await
            .infallible_unwrap()
            .remove();
        drop(stream);

        let entries: Vec<(i64, String)> = map.into_entries_unordered().collect();
        assert_eq!(Vec::<(i64, String)>::new(), entries);
    }

    /// Consequence 2: the next locking call that has to evict panics while it holds the
    /// map-wide mutex. That poisons the mutex, and from then on every operation panics.
    #[test]
    #[ignore = "known bug"]
    fn then_eviction_panics_and_poisons_the_cache() {
        let outcome = run_with_watchdog(Duration::from_secs(10), || {
            let cache = LockableLruCache::<i64, String>::new();
            current_thread_runtime().block_on(create_stale_entry(&cache, 1));

            // Lock a different key with a limit
            let first = catch_unwind(AssertUnwindSafe(|| {
                let _guard = cache
                    .try_lock(
                        2,
                        SyncLimit::SoftLimit {
                            max_entries: 1.try_into().unwrap(),
                            on_evict: |entries| {
                                for mut entry in entries {
                                    entry.remove();
                                }
                                Ok::<(), !>(())
                            },
                        },
                    )
                    .infallible_unwrap();
            }))
            .map_err(panic_message);

            // And now try to use the cache for something entirely unrelated
            let second = catch_unwind(AssertUnwindSafe(|| cache.num_entries_or_locked()))
                .map_err(panic_message);

            (first, second)
        });
        match outcome {
            Outcome::Completed((first, second)) => {
                assert_eq!(Ok(()), first, "locking with a limit panicked");
                assert!(second.is_ok(), "follow-up call panicked: {second:?}");
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    /// Consequence 2b: if the eviction scan has already locked another entry when it reaches
    /// the stale one, the panic does not even propagate. The thread hangs with the map-wide
    /// mutex held.
    #[test]
    #[ignore = "known bug"]
    fn then_eviction_deadlocks() {
        let outcome = run_with_watchdog(Duration::from_secs(10), || {
            let cache = LockableLruCache::<i64, String>::new();
            cache
                .blocking_lock(10, SyncLimit::no_limit())
                .infallible_unwrap()
                .insert(String::from("value"));
            current_thread_runtime().block_on(create_stale_entry(&cache, 1));

            // Lock a different key with a limit that makes two entries go
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = cache
                    .try_lock(
                        2,
                        SyncLimit::SoftLimit {
                            max_entries: 1.try_into().unwrap(),
                            on_evict: |entries| {
                                for mut entry in entries {
                                    entry.remove();
                                }
                                Ok::<(), !>(())
                            },
                        },
                    )
                    .infallible_unwrap();
            }))
            .map_err(panic_message)
        });
        match outcome {
            Outcome::Completed(Ok(())) => {}
            Outcome::Completed(Err(msg)) => panic!("library panicked: {msg}"),
            Outcome::Panicked(msg) => panic!("test setup panicked: {msg}"),
            Outcome::Hung => panic!("DEADLOCK: the thread calling try_lock never returned"),
        }
    }

    /// Consequence 3: `lock_entries_unlocked_for_at_least` panics and deadlocks on it
    /// (see `scan_freezes_the_cache`), here even with a duration of one hour.
    #[test]
    #[ignore = "known bug"]
    fn then_lock_entries_unlocked_for_at_least_deadlocks() {
        let outcome = run_with_watchdog(Duration::from_secs(10), || {
            let cache = LockableLruCache::<i64, String>::new();
            current_thread_runtime().block_on(create_stale_entry(&cache, 1));
            cache
                .lock_entries_unlocked_for_at_least(Duration::from_secs(3600))
                .count()
        });
        match outcome {
            Outcome::Completed(num_old_enough) => assert_eq!(0, num_old_enough),
            Outcome::Panicked(msg) => panic!("library panicked: {msg}"),
            Outcome::Hung => panic!(
                "DEADLOCK: the thread calling lock_entries_unlocked_for_at_least never returned"
            ),
        }
    }
}

/// `SoftLimit` evicts in situations where the locking call cannot exceed the limit,
/// and it evicts far too much while a `lock_all_entries` stream is alive.
mod soft_limit {
    use super::*;

    fn cache_with_entries(keys: impl IntoIterator<Item = i64>) -> LockableLruCache<i64, String> {
        let cache = LockableLruCache::<i64, String>::new();
        for key in keys {
            cache
                .blocking_lock(key, SyncLimit::no_limit())
                .infallible_unwrap()
                .insert(format!("value {key}"));
        }
        cache
    }

    /// Two keys fit into a cache that is limited to two entries, so accessing them
    /// alternately should never evict anything.
    #[test]
    #[ignore = "known bug"]
    fn cache_hit_on_a_full_cache_must_not_evict() {
        let cache = cache_with_entries([1, 2]);

        let mut evicted = vec![];
        let mut cache_misses = 0;
        for i in 0..10 {
            let key = 1 + (i % 2);
            let mut guard = cache
                .blocking_lock(
                    key,
                    SyncLimit::SoftLimit {
                        max_entries: 2.try_into().unwrap(),
                        on_evict: |entries| {
                            for mut entry in entries {
                                evicted.push(*entry.key());
                                entry.remove();
                            }
                            Ok::<(), !>(())
                        },
                    },
                )
                .infallible_unwrap();
            if guard.value().is_none() {
                // cache miss, we'd have to reload the value from the underlying storage
                cache_misses += 1;
                guard.insert(format!("value {key}"));
            }
        }

        assert_eq!(Vec::<i64>::new(), evicted, "entries were evicted");
        assert_eq!(0, cache_misses);
    }

    /// The entry that gets evicted can be the very entry the call is locking.
    #[test]
    #[ignore = "known bug"]
    fn cache_hit_must_not_evict_the_entry_it_is_locking() {
        let cache = cache_with_entries([1, 2]);

        let mut evicted = vec![];
        let guard = cache
            .blocking_lock(
                1,
                SyncLimit::SoftLimit {
                    max_entries: 2.try_into().unwrap(),
                    on_evict: |entries| {
                        for mut entry in entries {
                            evicted.push(*entry.key());
                            entry.remove();
                        }
                        Ok::<(), !>(())
                    },
                },
            )
            .infallible_unwrap();

        assert_eq!(
            (Vec::<i64>::new(), Some(&String::from("value 1"))),
            (evicted, guard.value())
        );
    }

    /// A `try_lock` that does not get the lock adds nothing to the cache.
    #[test]
    #[ignore = "known bug"]
    fn failed_try_lock_must_not_evict() {
        let cache = cache_with_entries([1, 2]);
        let _guard = cache
            .blocking_lock(2, SyncLimit::no_limit())
            .infallible_unwrap();

        let mut evicted = vec![];
        let result = cache
            .try_lock(
                2,
                SyncLimit::SoftLimit {
                    max_entries: 2.try_into().unwrap(),
                    on_evict: |entries| {
                        for mut entry in entries {
                            evicted.push(*entry.key());
                            entry.remove();
                        }
                        Ok::<(), !>(())
                    },
                },
            )
            .infallible_unwrap();

        assert!(result.is_none());
        assert_eq!(Vec::<i64>::new(), evicted);
    }

    /// Five entries, a limit of five, one new key: one entry has to go. But a
    /// `lock_all_entries` stream is alive (think: a background task iterating over the cache).
    /// The stream holds a reference to every entry, so evicted entries stay in the map without
    /// a value, the entry count never drops, and the call keeps evicting until nothing is left.
    #[tokio::test]
    #[ignore = "known bug"]
    async fn one_insert_must_not_evict_everything_while_a_lock_all_entries_stream_is_alive() {
        let cache = cache_with_entries(1..=5);
        let _stream = cache.lock_all_entries().await;

        let evicted = RefCell::new(vec![]);
        let guard = cache
            .async_lock(
                6,
                AsyncLimit::SoftLimit {
                    max_entries: 5.try_into().unwrap(),
                    on_evict: async |entries| {
                        for mut entry in entries {
                            evicted.borrow_mut().push(*entry.key());
                            entry.remove();
                        }
                        Ok::<(), !>(())
                    },
                },
            )
            .await
            .infallible_unwrap();
        drop(guard);

        assert_eq!(vec![1], *evicted.borrow());
    }
}

mod misc {
    use super::*;

    /// Nothing can have been unlocked for that long, so this should just return nothing.
    #[test]
    #[ignore = "known bug"]
    fn lock_entries_unlocked_for_at_least_with_a_huge_duration() {
        let cache = LockableLruCache::<i64, String>::new();
        cache
            .blocking_lock(1, SyncLimit::no_limit())
            .infallible_unwrap()
            .insert(String::from("value"));

        let old_enough: Vec<i64> = cache
            .lock_entries_unlocked_for_at_least(Duration::MAX)
            .map(|guard| *guard.key())
            .collect();
        assert_eq!(Vec::<i64>::new(), old_enough);
    }

    /// `LockPool` hands out the full `Guard` API of the underlying `LockableHashMap<K, ()>`.
    /// Inserting the `()` value makes the key look locked until somebody removes it again.
    /// (If the fix is a guard type for `LockPool` that has no `insert`, this test stops
    /// compiling and can simply be deleted.)
    #[test]
    #[ignore = "known bug"]
    fn lockpool_guard_insert_makes_the_key_look_locked() {
        let pool = LockPool::<i64>::new();
        {
            let mut guard = pool.blocking_lock(1);
            guard.insert(());
        }
        assert_eq!(
            (Vec::<i64>::new(), 0),
            (pool.locked_keys(), pool.num_locked())
        );
    }

    #[derive(PartialEq, Eq, Hash, Debug)]
    struct KeyWithPanickingClone(i64);
    impl Clone for KeyWithPanickingClone {
        fn clone(&self) -> Self {
            if self.0 < 0 {
                panic!("clone panicked");
            }
            Self(self.0)
        }
    }

    /// User code (`K::clone`, also `K::hash` and `K::eq`) runs while the map-wide mutex is
    /// held. One panic in there poisons the mutex and the map is unusable from then on.
    #[test]
    #[ignore = "known bug"]
    fn panic_in_key_clone_must_not_break_the_map() {
        let map = LockableHashMap::<KeyWithPanickingClone, String>::new();
        map.blocking_lock(KeyWithPanickingClone(1), SyncLimit::no_limit())
            .infallible_unwrap()
            .insert(String::from("value"));

        let failed = catch_unwind(AssertUnwindSafe(|| {
            let _ = map.try_lock(KeyWithPanickingClone(-1), SyncLimit::no_limit());
        }));
        assert!(failed.is_err());

        let afterwards =
            catch_unwind(AssertUnwindSafe(|| map.num_entries_or_locked())).map_err(panic_message);
        assert_eq!(Ok(1), afterwards);
    }

    /// A minimal executor that polls its task right away on the thread that wakes it.
    struct InlineTask {
        future: std::sync::Mutex<Option<std::pin::Pin<Box<dyn Future<Output = ()> + Send>>>>,
        is_polling: AtomicBool,
    }
    impl std::task::Wake for InlineTask {
        fn wake(self: Arc<Self>) {
            if self.is_polling.swap(true, Ordering::SeqCst) {
                return;
            }
            let mut slot = self.future.lock().unwrap();
            if let Some(future) = slot.as_mut() {
                let waker = std::task::Waker::from(Arc::clone(&self));
                if future
                    .as_mut()
                    .poll(&mut std::task::Context::from_waker(&waker))
                    .is_ready()
                {
                    *slot = None;
                }
            }
            drop(slot);
            self.is_polling.store(false, Ordering::SeqCst);
        }
    }

    /// Unlocking wakes the next waiter while the map-wide mutex is held. tokio only enqueues
    /// the task in `wake`, but an executor that polls inline deadlocks in `Guard::drop`.
    #[test]
    #[ignore = "known bug"]
    fn unlocking_with_a_waiter_whose_waker_polls_inline() {
        let map = Arc::new(LockableHashMap::<i64, String>::new());
        let holder = map
            .blocking_lock_owned(1, SyncLimit::no_limit())
            .infallible_unwrap();

        let task = Arc::new(InlineTask {
            future: std::sync::Mutex::new(Some(Box::pin({
                let map = Arc::clone(&map);
                async move {
                    let guard = map
                        .async_lock_owned(1, AsyncLimit::no_limit())
                        .await
                        .infallible_unwrap();
                    drop(guard);
                }
            }))),
            is_polling: AtomicBool::new(false),
        });
        // First poll: the task queues up for the lock
        std::task::Wake::wake(Arc::clone(&task));

        let outcome = run_with_watchdog(Duration::from_secs(5), move || drop(holder));
        assert!(
            matches!(outcome, Outcome::Completed(())),
            "dropping the guard: {outcome:?}"
        );
    }
}
