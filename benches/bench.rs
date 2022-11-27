use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lockable::LockPool;
use std::sync::{Arc, Mutex};
use std::thread;

pub fn hashmap(c: &mut Criterion) {
    let mut g = c.benchmark_group("hashmap (insert + remove)");
    g.bench_function("std::HashMap", |b| {
        let mut hashmap = std::collections::HashMap::<i64, ()>::new();
        b.iter(|| {
            hashmap.insert(4, ());
            hashmap.remove(&4);
        })
    });
    g.finish();
}

pub fn single_thread_lock_unlock(c: &mut Criterion) {
    let mut g = c.benchmark_group("single thread lock unlock");
    g.bench_function("std::Mutex", |b| {
        let mutex = Mutex::new(());
        b.iter(|| {
            let _g = mutex.lock().unwrap();
        })
    });
    g.bench_function("tokio::Mutex (blocking_lock)", |b| {
        let mutex = tokio::sync::Mutex::new(());
        b.iter(|| {
            let _g = mutex.blocking_lock();
        })
    });
    g.bench_function("tokio::Mutex (async_lock)", |b| {
        let mutex = tokio::sync::Mutex::new(());
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let _g = mutex.lock().await;
            })
    });
    g.bench_function("LockPool (blocking_lock)", |b| {
        let pool = LockPool::new();
        b.iter(|| {
            let _g = pool.blocking_lock(black_box(3));
        })
    });
    // TODO add _owned methods once they're implemented
    g.bench_function("LockPool (try_lock)", |b| {
        let pool = LockPool::new();
        b.iter(|| {
            let _g = pool.try_lock(black_box(3)).unwrap();
        })
    });
    g.bench_function("LockPool (async_lock)", |b| {
        let pool = LockPool::new();
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let _g = pool.async_lock(black_box(3)).await;
            })
    });
    g.finish();
}

pub fn multi_thread_lock_unlock(c: &mut Criterion) {
    const NUM_THREADS: usize = 500;
    const NUM_LOCKS_PER_THREAD: usize = 1000;

    let mut g = c.benchmark_group("multi thread lock unlock");
    g.bench_function("std::Mutex", |b| {
        let mutex = Arc::new(Mutex::new(()));
        b.iter(move || {
            spawn_threads(NUM_THREADS, |_| {
                for _ in 0..NUM_LOCKS_PER_THREAD {
                    let _g = mutex.lock().unwrap();
                }
            });
        })
    });
    g.bench_function("tokio::Mutex (blocking_lock)", |b| {
        let mutex = Arc::new(tokio::sync::Mutex::new(()));
        b.iter(move || {
            spawn_threads(NUM_THREADS, |_| {
                for _ in 0..NUM_LOCKS_PER_THREAD {
                    let _g = mutex.blocking_lock();
                }
            });
        })
    });
    g.bench_function("LockPool (blocking_lock)", |b| {
        let pool = LockPool::new();
        b.iter(move || {
            spawn_threads(NUM_THREADS, |_| {
                for _ in 0..NUM_LOCKS_PER_THREAD {
                    let _g = pool.blocking_lock(black_box(3));
                }
            });
        })
    });
}

fn spawn_threads(num: usize, func: impl Fn(usize) + Send + Sync) {
    thread::scope(|s| {
        for _ in 0..num {
            s.spawn(|| func(num));
        }
    });
}

criterion_group!(
    benches,
    hashmap,
    single_thread_lock_unlock,
    multi_thread_lock_unlock,
);
criterion_main!(benches);
