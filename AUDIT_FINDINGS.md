# Audit findings and fix plan

This file is the working document for fixing the issues found in the correctness audit of this
crate. It lives on the branch `claude/wonderful-brahmagupta-ujwpg7` only (it is not meant to be
merged) and is updated after every fix.

## How we work

Instructions from the maintainer, verbatim:

* In each PR, please explain the exact issue in the PR description.
* Use red-green-refactor. First build a regression test, run it and see it fail to prove it tests
  the right thing (ensure it fails for the right reasons), build a fix, see the same test green.
  If you need to change the test to make it green, it was wrong initially and we first need to
  show the new test to be red before making it green with the fix.

Also from the maintainer:

* Fix the issues one by one, the big ones first.
* All issues are written down here, including the smaller ones.

Working notes (how the above is carried out):

* One PR per issue, each on its own branch that starts from the current `master`. The audit
  branch stays as it is and only receives updates to this file.
* The regression test of a PR goes into the crate's own test suite, not ignored. The
  reproductions on the audit branch (see "Reproductions" below) are the starting point for it.
  When a fix lands, the matching reproduction should pass too.
* Order: 1, 2, 3 (its fix also covers 7), 4, 5, 6, 8, 9, 10, then the smaller issues.
* After each PR: update the status table below and push this file.

## Status

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | High | `lock_entries_unlocked_for_at_least` panics, then deadlocks the whole cache | open |
| 2 | High | Cancelled lock futures and dropped `lock_all_entries` streams skip cleanup | open |
| 3 | Medium | With tokio 1.54+, a `lock_all_entries` stream locks entries it has not returned yet | open |
| 4 | Medium | `lock_entries_unlocked_for_at_least` resets the age of the entry it stops at | open |
| 5 | Medium | With a live `lock_all_entries` stream, one insert at the limit evicts everything | open |
| 6 | Medium | `SoftLimit` on a `LockableHashMap` evicts the entry inserted last | open |
| 7 | Medium (large maps) | Consuming a `lock_all_entries` stream in a tokio task takes quadratic time | open |
| 8 | Low to medium | `SoftLimit` evicts on cache hits and on failed `try_lock` | open |
| 9 | Low | LRU order and `last_unlocked` order disagree | open |
| 10 | Low | User code runs under the map-wide mutex | open |
| S1 to S17 | Low | Smaller issues, see below | open |

## What was audited

* Code: `master` at `ced5f1a` (crate version 0.3.0, MSRV Rust 1.100), with the dependency
  versions of its `Cargo.lock` (tokio 1.54.1, lru 0.19.1, itertools 0.15.0, futures 0.3.34).
  The first audit pass was done on `7730e99` (0.2.0) and everything was re-checked after the
  update.
* All line numbers in this file refer to `master` at `ced5f1a`.
* Verdict: no `unsafe` in the crate, Miri reports no undefined behaviour with the current
  dependencies, and in every stress test and model check no key was ever locked twice and no
  value was lost or resurrected. The issues below are about liveness (freezes, poisoned maps),
  bookkeeping (phantom entries), and the behaviour of pruning and of the entry limit.

## Reproductions

On the audit branch, every issue from 1 to 10 has at least one test that asserts the expected
behaviour and therefore fails while the bug exists. All of them are `#[ignore]`d.

* `tests/audit_repro.rs`: 26 tests that use only the public API.
  `cargo test --test audit_repro -- --ignored --test-threads=1`
* `src/lockable_lru_cache.rs`, test module `audit_repro`: 3 tests that need the mocked clock.
  `cargo test --lib audit_repro -- --ignored`

With `--features slow_assertions` the stale-entry tests fail earlier, in the invariant check.

---

## 1. `lock_entries_unlocked_for_at_least` panics, then deadlocks the whole cache (high)

**Where:** `src/lockable_map_impl.rs:440-481` (`lock_all_unlocked`),
`src/lockable_lru_cache.rs:1035-1038` (the predicate), `src/lockable_map_impl.rs:527-551`
(`_unlock`).

**What happens:**

1. `lock_all_unlocked` try-locks the entries one after the other in iteration order while it
   holds the map-wide `entries` mutex. It is lazy: it stops after the first unlocked entry for
   which the predicate returns false. Every entry it manages to lock is wrapped in a `Guard`
   (`:449`) without a look at whether the entry has a value.
2. The predicate of `LockableLruCache::_lock_entries_unlocked_for_at_least` does
   `entry.value_raw().expect("There must be a value, otherwise it cannot exist in the map as an
   'unlocked' entry")` (`lockable_lru_cache.rs:1036`) and panics for an entry without value.
3. The panic happens while `entries` is still locked (it is dropped at `:470`). Unwinding drops
   the `Guard`, `Guard::drop` calls `_unlock`, and `_unlock` calls `self._entries()` (`:536`),
   which locks the same `std::sync::Mutex` again on the same thread. On Linux that blocks
   forever. The thread never returns and keeps the mutex, so every other operation on the cache
   blocks forever too: `num_entries_or_locked`, operations on unrelated keys, dropping guards
   that were taken earlier, and async callers (they block inside `poll`, so a timeout around
   them does not help).

**Why the state is legal:** an entry that is unlocked and has no value is allowed by invariant 2
(`lockable_map_impl.rs:61`). It exists while nobody holds the key's lock, the lock has not been
handed to a queued waiter, and somebody still holds a `ReplicaArc` for the key. The sibling scan
`_lock_up_to_n_first_unlocked_entries` handles exactly this case (`:643-652`). The comment on
`EntryValue` (`:37-38`, "While unlocked, an entry is always Some") states the opposite and is
the false premise behind the `expect`.

**How to get there:**

* (a) A `lock_all_entries` stream whose future for the entry has not been polled at all (for
  example a stream that was created but not polled yet, or the not-yet-visited part of a
  partially consumed stream), and that entry loses its value or never had one. Deterministic.
* (b) A plain thread race between `blocking_lock` or `try_lock` on a key without value and the
  scan: the locker has released `entries` and not reached the key's mutex yet (`:325`-`:330`),
  or a failed `try_lock` has not re-taken `entries` yet (`:377`-`:384`). Hits within about a
  second in a tight loop.
* (c) A stale entry left behind by issue 2.
* (d) Only with tokio older than 1.54 (the manifest allows `^1.45`, `Cargo.lock` has 1.54.1): a
  task that runs out of its cooperative budget inside `async_lock` yields before it queued up
  for the key.

**Precondition:** the scan must reach the entry. It walks from least to most recently used and
stops at the first unlocked entry with a value that is not old enough. The entry without value
sits at the most-recently-used end when it comes into being, so the bug only fires if every
unlocked entry with a value in front of it is old enough: with `Duration::ZERO` that is every
entry, with a real duration it needs a small or mostly idle cache.

**Scope:** only `LockableLruCache` (`lock_all_unlocked` has one caller), both
`lock_entries_unlocked_for_at_least` and `lock_entries_unlocked_for_at_least_owned`.

**Fix direction:** in `lock_all_unlocked`, look at the raw value before a `Guard` exists and
skip entries without value, dropping the raw mutex guard while `entries` is still held (allowed
by the exception in invariant 2C, and what `_lock_up_to_n_first_unlocked_entries` does). Do not
add an assertion about the reference count there: a panic at that point would drop the guards
collected so far under `entries` and deadlock again. Evaluating the predicate on the raw value
before the `Guard` exists also removes issue 4, but that gets its own PR.

**Reproductions:** `scan_freezes_the_cache::with_a_pending_lock_all_entries_stream`,
`scan_freezes_the_cache::race_with_blocking_lock`,
`stale_entries::then_lock_entries_unlocked_for_at_least_deadlocks`.

---

## 2. Cancelled lock futures and dropped `lock_all_entries` streams skip cleanup (high)

**Where:** `src/lockable_map_impl.rs:354` (`async_lock`), `:506-508` (the per-entry futures of
`lock_all_entries`), `:330` (`blocking_lock` when it panics inside a runtime).

**What happens:** `async_lock` holds a `ReplicaArc` across `mutex.lock_owned().await`. If the
future is dropped there, the `ReplicaArc` is dropped without taking `entries` and without the
cleanup that invariant 2C (`:67-70`) requires. The same is true for the per-entry futures of a
`lock_all_entries` stream that is dropped before it is exhausted, and for `blocking_lock` when
tokio panics because it was called inside a runtime for a key that has an entry.

**Phantom entries:** if the previous holder released a key without value while such a future
still existed, `_delete_if_unlocked_and_nobody_waiting_for_lock` found another reference (the
check `num_replicas() == 0` at `:563` failed) and left the entry in the map, relying on the
waiter to clean up. After the cancellation nobody does: the entry stays in the map, unlocked and
without value. For a stream this needs the guard that left the entry without value to be dropped
before the stream is dropped.

**Triggers:** dropping a polled `async_lock` future after the holder released; `tokio::select!`
with a cancel branch; `JoinHandle::abort`; a `tokio::time::timeout` around a `join!` of holder
and waiter (the holder's branch must be dropped first); leaving a `lock_all_entries` loop early.
A plain `tokio::time::timeout` around a single waiter rarely triggers it, because `Timeout`
polls the lock future before it gives up.

**Consequences:**

* `num_entries_or_locked`, `keys_with_entries_or_locked`, `LockPool::num_locked` and
  `LockPool::locked_keys` report a key that is neither locked nor has a value. It stays until
  the same key is locked and unlocked again, or a fully consumed `lock_all_entries` visits it.
  For keys that are never used again the entries accumulate.
* `into_entries_unordered` panics when its iterator reaches the entry (`:622`, "Invariant 2
  violated...").
* A later locking call with a `SoftLimit` whose eviction scan reaches the entry fails
  `assert!(mutex.num_replicas() > 1, "Invariant violated")` (`:651`) while `entries` is locked.
  If the scan has not collected a guard yet, the mutex gets poisoned and every later operation
  panics ("The global mutex protecting the LockableCache is poisoned"; `into_entries_unordered`
  then panics with "Lock poisoned"). If it already collected a guard, unwinding drops that guard
  under the lock and the thread deadlocks like in issue 1. If the panicking thread holds another
  guard of the same map, that guard's destructor panics during unwinding and the process aborts.
* `lock_entries_unlocked_for_at_least` panics and deadlocks on it (issue 1), also with a long
  duration once the entry has drifted to the least-recently-used end.
* With `--features slow_assertions` the next access of any kind panics in the invariant check
  ("Invariant 2 violated. Found an entry without ReplicaArcs that is None.") and poisons the map.
* No phantom entry is needed for the worst case: a waiter that was handed the lock is cancelled
  exactly between the eviction scan's failed try-lock and its
  `assert!(mutex.num_replicas() > 1, "Invariant violated")` at `:662`. Every entry can have a
  value. A targeted test hits this within 50 ms and poisons the map.
* Model level only (see S12): the same unsynchronised drop can make the `expect` at `:587-589`
  fire on weakly ordered CPUs.

**Fix direction:** the cleanup guard that the TODO at `:72` describes. Wrap the pending tokio
`lock_owned()` future (`:354`, `:508`, and the blocking path at `:330`) in a guard whose `Drop`
locks `entries` first, then drops the inner future (which returns the permit and drops the
`Arc`), then removes the entry if it is unreferenced and has no value. The order matters:
taking `entries` only after the inner future was dropped fixes the phantom entries but not the
race at `:662` and not S12. As hardening, a failed try-lock at `:587-589`, `:662` and `:721`
could be treated as "somebody else still has it" instead of panicking while `entries` is held.

**Reproductions:** module `stale_entries` (11 tests), among them
`cancellation_racing_with_an_eviction_scan` for the race at `:662`.

---

## 3. With tokio 1.54+, a `lock_all_entries` stream locks entries it has not returned yet (medium)

**Where:** `src/lockable_map_impl.rs:493-525`, one `mutex.lock_owned().await` future per entry
(`:508`) in a `FuturesUnordered` (`:518`).

**What happens:** tokio 1.54.0 ("sync: preserve FIFO lock order when coop budget is exhausted")
takes the semaphore permit first and checks the task's cooperative budget afterwards, so a
`lock_owned()` future polled by a task without budget returns `Pending` while it already owns
the lock. `FuturesUnordered` keeps polling its other children because tokio defers the wake-up.
Once the consuming task has used up its budget (128 operations per poll, shared with everything
else the task does), every remaining per-entry future takes its lock and stays pending. The
stream then holds the locks of all entries it has not returned yet, until the consumer reaches
them or drops the stream.

**Observed:** with 300 entries and a consumer that does not yield, 171 other keys are locked
when item 129 is handed out (0 with tokio 1.45.1, on a non-tokio executor, or when the consumer
really yields for every item). If the budget was already spent before the first poll, all
entries are locked at once, also in a 10-entry map. A consumer that leaves its loop and keeps
the stream alive keeps the remaining entries locked. Other tasks wait until the stream reaches
their key (worst wait rose from 6 ms to about 600 ms in a flush-loop simulation). A loop body
that waits for another key of the same map hangs deterministically; the docs warn against that
pattern, but with older tokio it only hung under contention.

**Where it applies:** a consumer polled on a tokio scheduler thread: a spawned task, anything on
the current-thread runtime (also default `#[tokio::test]`), a `LocalSet`. In the future passed
to `block_on` of a multi-thread runtime only two entries per budget exhaustion get locked. No
effect outside tokio or under `tokio::task::coop::unconstrained`.

**Relation to issue 1:** the same tokio change removed trigger (d) of issue 1, so raising or
lowering the tokio requirement is not a fix. Downstream users get whatever tokio they resolve
within `^1.45`.

**Fix direction:** do not park a `lock_owned()` future per entry up front. In the per-entry
future, call `try_lock_owned()` first and fall back to `lock_owned().await` only when the entry
is locked. That spends no budget for unlocked entries; if cooperative yielding is wanted, add
one budget check per returned item at the stream level. A four-line version of this passed all
765 unit tests in a scratch copy and also removes issue 7. Alternative: run the per-entry
futures under `tokio::task::coop::unconstrained` (needs tokio's `rt` feature).

**Reproductions:** `lock_all_entries_stream::must_not_lock_entries_it_has_not_returned_yet`.

---

## 4. `lock_entries_unlocked_for_at_least` resets the age of the entry it stops at (medium)

**Where:** `src/lockable_map_impl.rs:457` (`take_while_inclusive`), `:471-478`;
`src/lockable_lru_cache.rs:120-124` (`on_unlock`).

**What happens:** `lock_all_unlocked` uses `take_while_inclusive`, so the first entry that does
not fulfil the predicate is locked in a `Guard` too. It is popped and dropped at `:471-478`.
Dropping that `Guard` goes through `_unlock`, which calls `config.on_unlock` (`:532`), and the
LRU config's `on_unlock` sets `last_unlocked = now`. Every call therefore makes the first
not-old-enough unlocked entry "just unlocked" again.

**Effect:** as long as nobody else locks that entry (a lock moves it to the most-recently-used
end), a caller that polls more often than `duration` never gets it and never gets anything
behind it. Three idle entries, pruned every 100 ms with a 500 ms threshold: nothing is returned
in a simulated minute. There is no effect if every call starts at least `duration` after the
previous call finished. A fixed-rate caller whose period equals `duration` is still affected,
because the reset timestamp is taken after the call's own `now`. Any second caller that calls
more often (a metrics probe, another policy) starves a caller that polls slowly enough. The
entry is also really locked for a moment, so a concurrent `try_lock` on that key can fail
although no user holds it (S17).

**Fix direction:** release the merely inspected entry without running `on_unlock`, or evaluate
the predicate on the raw value before a `Guard` exists and drop the raw mutex guard while
`entries` is still held. The second variant removes the need for `take_while_inclusive` and the
transient lock window that is visible to other threads. Both were tried in scratch copies; all
765 unit tests stayed green with the first.

**Reproductions:** mocked-clock tests `looking_at_an_entry_must_not_reset_its_age` and
`pruning_more_often_than_the_duration_must_still_prune`.

---

## 5. With a live `lock_all_entries` stream, one insert at the limit evicts everything (medium)

**Where:** `src/lockable_map_impl.rs:149-196` (async eviction loop), `:239-285` (sync twin),
`:563`.

**What happens:** the eviction loop recomputes `entries.len()` on every iteration (`:152-153`,
`:242-243`). While a `lock_all_entries` stream is alive it holds a `ReplicaArc` for every entry
it has not returned yet. When `on_evict` removes a victim's value and drops the guard, the entry
stays in the map without value (the check `num_replicas() == 0` at `:563` fails), so `len()`
does not shrink and the loop evicts the next entry, until enough entries that the stream does
not reference have been evicted, or no unlocked entry with a value is left. If the stream
references every entry, that means everything. Both map types are affected. A queued waiter for
an evicted key has the same effect for that one key.

**Observed:** 5 entries, `max_entries` = 5, one new key, stream alive: all 5 entries are evicted
instead of 1. Related: unlocked keys without value that a stream references are counted and
listed by `num_entries_or_locked` and `keys_with_entries_or_locked` for the lifetime of the
stream.

**Fix direction:** do not count unlocked entries without value toward the limit (keep a counter
of such placeholders, or let the loop subtract what it has just evicted).

**Reproductions:**
`soft_limit::one_insert_must_not_evict_everything_while_a_lock_all_entries_stream_is_alive`.

---

## 6. `SoftLimit` on a `LockableHashMap` evicts the entry inserted last (medium)

**Where:** `src/lockable_map_impl.rs:634-670` (`_lock_up_to_n_first_unlocked_entries` walks
`entries.iter()` from the start, `:640`); `src/lockable_hash_map.rs:58-60`.

**What happens:** the victims are taken from the front of `HashMap::iter()`, which is bucket
order. Once the map has been full for roughly two times `max_entries` further inserts, the
entry inserted last is always the first one there. From then on every miss evicts what the
previous miss inserted, and the other `max_entries - 1` entries stay forever. Which keys are
kept is decided by hash position and arrival order, not by use.

**Observed:** limit 16, 2000 inserts of different keys: 1954 of 1984 evictions removed the entry
that had just been inserted, and 15 of the 16 entries left at the end came from the first 1000
inserts. As a cache this gives about half the hit rate of random eviction on a stationary Zipf
workload (0.30 against 0.56; `LockableLruCache` 0.61) and about zero after the key population
changes. Together with issue 8, "insert a value, drop the guard, lock the same key again with
the limit" returns a guard without value every time. `LockableLruCache` and `LockPool` are not
affected.

**Contract:** the docs promise only "some other entries" for the hash map, so no stated order is
broken, but they give the purpose of a limit as evicting old entries (`src/limit.rs:10`, `:163`),
and old entries are exactly what is never evicted.

**Fix direction:** give the hash map its own eviction order, for example insertion order (a FIFO
queue of keys) or a rotating or random starting point for the scan (`skip(n)` on std's
`HashMap` is O(n) under the map-wide mutex, so a queue is the better choice). If eviction
quality on `LockableHashMap` is out of scope, say so in the `SoftLimit` docs and point to
`LockableLruCache`.

**Reproductions:** `soft_limit::hash_map_must_evict_old_entries_and_not_the_newest_one`.

---

## 7. Consuming a `lock_all_entries` stream in a tokio task takes quadratic time (medium for large maps)

**Where:** `src/lockable_map_impl.rs:507-508`.

**What happens:** inside a tokio task each lock acquisition costs one unit of the cooperative
budget. After 128 items the budget is used up, every remaining per-entry future returns
`Pending`, and `FuturesUnordered` polls all of them before the task yields. That repeats every
128 items: about n²/256 + n/2 polls for n entries instead of n. With tokio 1.54+ every remaining
entry is also locked for that whole time (issue 3).

**Observed (release build):** 20,000 entries about 0.2 s instead of 10 ms; 64,000 entries 2.1 to
2.5 s instead of 35 to 45 ms; 256,000 entries 35 s instead of 0.1 s. Not affected: the future
passed to `block_on` of a multi-thread runtime, non-tokio executors, loop bodies that really
yield at least once per 128 budget units. Negligible below a few thousand entries.

**Fix direction:** the fix for issue 3 (`try_lock_owned()` first) removes it.

**Reproductions:**
`lock_all_entries_stream::consuming_it_in_a_tokio_task_must_not_take_quadratic_time` (compares
two timings, fails from a factor of 5; observed 15 to 40).

---

## 8. `SoftLimit` evicts on cache hits and on failed `try_lock` (low to medium)

**Where:** `src/lockable_map_impl.rs:152-153`, `:242-243`; doc examples `src/limit.rs:46-66`,
`:199-218`.

**What happens:** `num_overlimit_entries = entries.len().saturating_sub(max_entries.get() - 1)`
is computed before looking whether the key already exists, so locking an existing key in a map
that holds exactly `max_entries` entries evicts one entry. In `LockableLruCache` the victim is
the least recently used unlocked entry, which can be the very key being locked: the caller then
gets a guard without value for a key that was cached. A `try_lock` with a limit that fails
because the key is locked elsewhere has already evicted an entry. A pure cache hit also inherits
`on_evict`'s failure modes and latency.

**Effect:** a cache limited to N entries behaves like one with N-1 slots. With `max_entries` = 1
an uncontended lookup never hits. Two keys alternating in a cache limited to two evict on every
access. Concurrent hits on a full cache each pick their own victim.

**Docs:** the two "Example (with limit)" doc examples say "Lock a third entry" (`:46`, `:199`)
but lock the existing key 4 (`:49`, `:202`) and assert that key 4 is evicted (`:64-66`,
`:216-218`). They encode this behaviour and have to change with the fix (lock a new key).

**Fix direction:** add a non-promoting `contains_key` to `MapLike` and make room only when the
key is absent. If the key does not exist, `try_lock` always succeeds, so this covers the failed
`try_lock` case too. For the over-limit case the key should also be exempt from the victim scan.
Callers who compensated with `max_entries + 1` will hold one entry more after the fix.

**Reproductions:** `soft_limit::cache_hit_on_a_full_cache_must_not_evict`,
`soft_limit::cache_hit_must_not_evict_the_entry_it_is_locking`,
`soft_limit::failed_try_lock_must_not_evict`.

---

## 9. LRU order and `last_unlocked` order disagree (low)

**Where:** `src/lockable_lru_cache.rs:47` (`get_or_insert` sets the position), `:120-124`
(`on_unlock` sets the timestamp), invariant comment at `:129-132`.

**What happens:** an entry's position in the LRU order is set when it is looked up for locking
(also by a request that is still waiting and by a `try_lock` that fails), its `last_unlocked`
timestamp when it is unlocked. Nothing re-orders an entry with a value at unlock. That breaks
the invariant documented at `:129-132`, which the early exit of the scan in
`lock_entries_unlocked_for_at_least` relies on: an entry that was locked earlier but unlocked
later hides older entries behind it. Guards handed out by `lock_entries_unlocked_for_at_least`
and `lock_all_entries` update the timestamp on drop but not the position, with the same effect.

**Effect:** on its own the delay is bounded (the hidden entries are returned once the entry in
front of them has itself been unlocked for `duration`). Together with issue 4 they are not
returned for as long as the entry in front stays unused and the caller keeps polling more often
than `duration`. For eviction, the victim flips at the moment of release: an entry that was in
use until just now is treated as the oldest. Whether that is wrong is a design question; the
docs promise no more than "least recently used".

**Fix direction:** promote the entry in `_unlock` while `entries` is held (`LruCache::promote`)
and take the timestamp under `entries` as well: today `on_unlock` runs at `:532`, before the
mutex is taken at `:536`, so two racing unlocks can still be ordered against their timestamps.
Alternatively drop the early exit and scan all unlocked entries (O(n)). Promoting on unlock
changes the eviction order, so it needs a decision on what "recently used" means for an entry
that is held for a long time.

**Reproductions:** mocked-clock test
`old_entry_must_not_be_hidden_behind_an_entry_that_was_locked_for_longer`.

---

## 10. User code runs under the map-wide mutex (low)

**Where:** the `expect` message at `src/lockable_map_impl.rs:124` says no user code runs while
the lock is held. It does in these places:

* `K::clone`: `src/lockable_hash_map.rs:40`, `src/lockable_lru_cache.rs:47`,
  `src/lockable_map_impl.rs:449`, `:501`, `:631`, `:644`.
* `K::hash` and `K::eq`: every map lookup.
* `K::drop`: `entries.remove` at `:565` and `:592`.
* Wakers of waiting tasks: `drop(guard)` at `:537` releases the per-key lock while `entries` is
  held, and tokio calls the next waiter's waker there.
* The destructor of the `on_evict` closure: the `SoftLimit` arm binds the closure (`:144-147`,
  sync twin `:234-237`), and the arm's value is the still locked `EntriesGuard`, so the closure
  and everything it captured by move is dropped at the end of the arm (`:197`, `:286`) while the
  mutex is held. This happens on every locking call with a `SoftLimit` that gets past the
  eviction loop, also far below the limit when `on_evict` is never called.

**Consequences:**

* A panic in a key function poisons the map for good: every later call panics.
* If such a panic happens in one of the two scans after at least one guard was collected
  (`:449`, `:644`), unwinding drops that guard under `entries` and the thread deadlocks: the map
  is frozen, not poisoned.
* If the panicking thread holds another guard of the same map, that guard's destructor panics
  on the poisoned mutex during unwinding and the process aborts.
* An executor whose waker polls inline deadlocks in `Guard::drop` as soon as the polled task
  touches the same map again. tokio's and futures' wakers only enqueue, so they are not affected.
* A `move` closure for `on_evict` that owns a guard of the same map, or a value whose `Drop`
  calls into the map, deadlocks every `SoftLimit` locking call. If the `Drop` of a captured
  value panics, the `EntriesGuard` on its way out of the match arm is leaked: the mutex stays
  locked forever without being poisoned. Closures that only borrow are not affected.

**Fix direction:** clone keys before taking `entries` where possible. Keep the closure alive
beyond the `EntriesGuard`: match on `&mut limit` and `drop(entries); drop(limit);` explicitly
before returning. Evaluate predicates and clone keys in the scans in a way that cannot drop a
`Guard` under `entries` on unwind. With `slow_assertions`, skip `assert_invariant` in
`EntriesGuard::drop` while `std::thread::panicking()` (see S15).

**Reproductions:** `misc::panic_in_key_clone_must_not_break_the_map`,
`misc::unlocking_with_a_waiter_whose_waker_polls_inline`,
`misc::on_evict_closure_that_owns_a_guard_of_the_same_cache` and its async twin.

---

## Smaller issues

**S1. `lock_entries_unlocked_for_at_least(Duration::MAX)` panics.** `now - duration` is computed
unchecked (`src/lockable_lru_cache.rs:1034`): "overflow when subtracting duration from instant"
instead of returning nothing. Both the borrowed and the owned variant. Fix: `checked_sub`, and
return nothing if it fails. Reproduction:
`misc::lock_entries_unlocked_for_at_least_with_a_huge_duration`.

**S2. `LockPool` hands out the full `Guard` API of `LockableHashMap<K, ()>`.** After
`guard.insert(())` (also `try_insert`, `value_or_insert`, `value_or_insert_with`),
`num_locked` and `locked_keys` report the key as locked until some guard calls `remove()`, and
the entry stays in the map. The `LockPool::try_lock` docs (`src/lockpool.rs:216-220`) even
describe inserting entries. Fix: a guard type for `LockPool` without the value API, or docs.
Reproduction: `misc::lockpool_guard_insert_makes_the_key_look_locked`.

**S3. The maps are `Sync` only if `K` and `V` are `Sync`.** The cause is the
`PhantomData<K>`/`PhantomData<V>` fields at `src/lockable_map_impl.rs:77-78`. A map of
`Send`-only values cannot be shared between threads although every access goes through a mutex;
guards, `Arc<map>` and the async futures are affected the same way. Removing the two fields only
adds auto-trait impls and the unit tests pass without them.

**S4. Documentation that does not match the code.**

* `blocking_lock` "can only be used from non-async contexts and will panic if used from async
  contexts" (`src/lockable_hash_map.rs:307`, `:318`; `src/lockable_lru_cache.rs:385`, `:396`;
  `src/lockpool.rs:146`, `:154`): it only panics if the key already has an entry (a new entry is
  locked with `try_lock_owned`), so the misuse goes unnoticed while keys are free. Without
  tokio's `rt` feature it blocks instead of panicking.
* `AsyncLimit::SoftLimit::on_evict` field doc: "Afterwards, these entries will be removed"
  (`src/limit.rs:121-123`). The map never removes them (the variant doc at `:101-103` says so),
  and an `on_evict` written against the field doc loops forever.
* Same field doc: "It is possible for some of the guards to have `None` values" (`:127-128`).
  That cannot happen: only entries with a value are selected.
* Owned guards "can be more easily moved around or cloned" (`src/lockable_hash_map.rs:356`,
  `:439`, `:525`, `:620`, `:798`; `src/lockable_lru_cache.rs:434`, `:517`, `:603`, `:698`, `:878`,
  `:979`): `Guard` is not `Clone`.
* `LockPool::try_lock`: "then Ok(None) is returned" (`src/lockpool.rs:222`); it returns an
  `Option`.
* `LockPool` keys must implement "... + Debug" (`src/lockpool.rs:29`); they need not.
* `LockableLruCache::try_lock_owned` says it is identical to `blocking_lock`
  (`src/lockable_lru_cache.rs:512-517`), a copy-and-paste leftover.
* Comments in the code: `src/lockable_map_impl.rs:37-38` ("While unlocked, an entry is always
  Some") is false, see issue 1; `:228` says the logic is duplicated in
  `_load_or_insert_mutex_for_key_sync`, which is the function it sits in; `:560-562` and
  `:583-585` argue that the reference count is reliable while `entries` is locked, which only
  covers increments (see issue 2); `:612-615` says no guard can exist in
  `into_entries_unordered`, see S5.

**S5. `std::mem::forget` on a guard, then `into_entries_unordered`.** The iterator panics when
it reaches that entry, with a message that says this cannot happen
(`src/lockable_map_impl.rs:621`). Informational: the expected result is not achievable with one
`Arc` per entry. A clearer panic message or a doc note would do.

**S6. API ergonomics.**

* The futures of the `*_owned` async functions borrow the `Arc` (`self: &Arc<Self>`), so
  `tokio::spawn(map.async_lock_owned(..))` does not compile; wrapping in `async move` works.
* `Guard<K, V, C, P>` exposes type parameters whose bounds live in private modules, so user code
  cannot be generic over guards, and two impls for guard types named through `Lockable` conflict.
* The `Lockable` trait has only associated types and no methods.
* `LockPool` has an `OwnedGuard` type but no `*_owned` functions (they are commented out).
* `LockPool` does not implement `Debug`; the maps are `Debug` only if `K` and `V` are, although
  the output prints neither.

**S7. `RUSTDOCFLAGS=-Dwarnings cargo doc --no-default-features` fails.** 30 unresolved intra-doc
links to `LockableLruCache` items (in `src/limit.rs`, `src/lockable_trait.rs`, `src/lib.rs`,
`src/guard.rs`) and one empty code block (`src/lib.rs:16`). The CI job only builds docs with
default features; docs.rs uses all features, so published docs are fine.

**S8. `cargo bench -- <criterion option>` fails.** For example `--save-baseline x`:
"Unrecognized option", because the library target's default bench harness is not disabled (no
`[lib] bench = false` in `Cargo.toml`). `cargo bench --bench bench -- ...` works.

**S9. ChangeLog.** 0.3.0 removes the public type `lockable::Never`; the ChangeLog lists it under
"Internal code modernizations, use never type". Code written like the 0.2.0 doc examples no
longer compiles.

**S10. README sync job.** `.cargo-rdme.toml` pins the nightly by date, but CI installs the
newest `cargo-rdme` (`.github/workflows/ci.yml:116`). The first `cargo-rdme` release that
expects a newer rustdoc JSON format breaks the job without any change in the repository.
The remedy the config comment names (remove the override) makes `gen_readme.sh` run
`rustup toolchain install ""`. Fix: pin `cargo-rdme` too, and let the script skip the install
when the key is absent.

**S11. Cost of the eviction scan.** A locking call with a `SoftLimit` on a full map steps over
every locked entry while it holds the map-wide mutex (`src/lockable_map_impl.rs:634-670`): 3.5
to 4 ms per call with 100,000 locked entries, 36 to 63 ms with 1,000,000. Because of issue 8,
cache hits and failed `try_lock`s pay it too. Fix direction: keep locked entries out of the
scan's path, for example by tracking idle entries in their own order (which also helps issue 9).

**S12. Weak memory ordering (model level only).** `num_replicas()` is `Arc::strong_count`, a
relaxed load. After a cancelled waiter's unsynchronised drop (issue 2), a reader can see the new
count and a stale permit count, and then the `expect` at `src/lockable_map_impl.rs:587-589`, the
assert at `:662` or the slow assertion at `:721` fires with `entries` held. Reproduced only
under Miri's weak-memory emulation; impossible on x86 and never observed natively. It goes away
with the fix for issue 2 if `entries` is taken before the pending future is dropped.

**S13. Gaps and defects in the crate's own tests.**

* No unit test uses `SoftLimit`; the only executions are the two doc examples in
  `src/limit.rs` (TODOs at `src/tests.rs:3-6` and `src/lockable_map_impl.rs:173`).
* `LockPool` has no unit tests (TODO at `src/lockpool.rs:415`), `lock_all_entries_owned` has
  doctests only.
* No test drops a pending lock future or an unfinished stream, none crosses tokio's budget of
  128 operations, and `lock_entries_unlocked_for_at_least` is called once per test.
* `map_with_two_entries_all_are_locked` waits for `child1` twice and never for `child2`
  (`src/tests.rs:2233-2234`, `:2269-2270`).
* The tests are timing based (one-second waits as checks, `src/tests.rs:22-39`, `:464-470`).
  Under Miri some of them time out; no undefined behaviour is reported.
* `benches/bench.rs` covers neither `lock_all_entries` nor `SoftLimit`.

**S14. Behaviour depends on the tokio version within `^1.45`.** Below 1.54: trigger (d) of
issue 1. From 1.54: issue 3. The CI minimal-versions jobs run tokio 1.45.0, the regular jobs
1.54.1. Both issues are fixable in the crate, after which the version no longer matters.

**S15. `slow_assertions` can abort the process.** `EntriesGuard::drop` runs `assert_invariant`
(`src/lockable_map_impl.rs:754-763`) also while a panic is unwinding. If the map is in a
half-finished state at that point (for example a panic in `K::hash` during a removal), the
second panic aborts the process. Fix: skip the check while `std::thread::panicking()`.

**S16. Two clocks under tokio's paused test clock.** `RealTime` is
`tokio::time::Instant::now()`. With tokio's `test-util` feature and a paused runtime, threads of
that runtime read the mock clock and plain threads read the real clock, so `last_unlocked`
stamps and the `now` of a scan can come from different clocks. No effect without `test-util`.
Worth a sentence in the docs.

**S17. Internal probe locks are visible to `try_lock`.** The scan in
`lock_entries_unlocked_for_at_least` keeps the entry it stops at locked for a moment after the
map-wide mutex is released (issue 4), so a concurrent `try_lock` on that key can return `None`
although no user holds it. Goes away with the second fix variant of issue 4.

---

## Fixed since the first audit pass (by the updates on master)

* `lru` was pinned to 0.14.0 by `lru = "^0.14"`, with two RustSec advisories. One was reachable
  on the old lockfile: with `--features slow_assertions`, a key whose `Drop` panics and a
  two-thread race, `EntriesGuard::drop` read a freed list node (use-after-free, reproduced under
  Miri). `Cargo.toml` now requires `^0.19`; the same tests are clean under Miri.
* `cargo audit` on the current `Cargo.lock` reports nothing.
* The CI job "Sync README.md" passes again (pinned nightly for `cargo-rdme`).
* The ARM CI jobs test the MSRV again.
* The redundant `homepage` key in `Cargo.toml` is gone.
* Trigger (d) of issue 1 does not occur with the tokio version in `Cargo.lock`.

## Checked and found fine

* Mutual exclusion per key and value integrity under stress, cancellation and eviction, in an
  8-million-step single-threaded model check, and under Miri for the single-threaded LRU tests.
* Cancelling a waiter while the holder still holds the lock; cancelling inside `on_evict`;
  panicking `on_evict` or `value_or_insert_with` closures; `on_evict` calling back into the map.
* The async futures are `Send` for `K, V: Send + Sync`; the API works from `tokio::spawn`.
* README links to docs.rs resolve through rustdoc's redirect pages.
* clippy, rustfmt, docs (default and all features), all feature sets, minimal versions, and all
  CI checks on `master` pass.

## Not covered

* Only Linux x86_64 was run. Nothing was executed on Windows, macOS or ARM hardware.
* No exhaustive exploration of thread interleavings (loom or similar). Multi-thread evidence
  comes from stress tests and from seeded two-thread tests under Miri.
* Performance beyond the two scaling issues above.
