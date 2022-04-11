//! Some generic tests applicable to all Lockable hash map types.

// TODO Add a test adding multiple entries and making sure all locking functions can read them
// TODO Add tests checking that the async_lock, lock_owned, lock methods all block each other. For lock and lock_owned that can probably go into common tests.rs

#[macro_export]
macro_rules! instantiate_lockable_tests {
    ($lockable_type: ident) => {
        use crate::error::TryLockError;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::{Arc, Mutex};
        use std::thread::{self, JoinHandle};
        use std::time::Duration;

        // Launch a thread that
        // 1. locks the given key
        // 2. once it has the lock, increments a counter
        // 3. then waits until a barrier is released before it releases the lock
        fn launch_thread_blocking_lock(
            pool: &Arc<$lockable_type<isize, String>>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()> {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let _guard = pool.blocking_lock(key);
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        fn launch_thread_blocking_lock_owned(
            pool: &Arc<$lockable_type<isize, String>>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()> {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let _guard = pool.blocking_lock_owned(key);
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        fn launch_thread_try_lock(
            pool: &Arc<$lockable_type<isize, String>>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()> {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let _guard = loop {
                    match pool.try_lock(key) {
                        Err(_) =>
                        /* Continue loop */
                        {
                            ()
                        }
                        Ok(guard) => break guard,
                    }
                };
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        fn launch_thread_try_lock_owned(
            pool: &Arc<$lockable_type<isize, String>>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()> {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let _guard = loop {
                    match pool.try_lock_owned(key) {
                        Err(_) =>
                        /* Continue loop */
                        {
                            ()
                        }
                        Ok(guard) => break guard,
                    }
                };
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        fn launch_thread_async_lock(
            pool: &Arc<$lockable_type<isize, String>>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()> {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                let _guard = runtime.block_on(pool.async_lock(key));
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        fn launch_thread_async_lock_owned(
            pool: &Arc<$lockable_type<isize, String>>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()> {
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                let _guard = runtime.block_on(pool.async_lock_owned(key));
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        #[tokio::test]
        #[should_panic(
            expected = "Cannot start a runtime from within a runtime. This happens because a function (like `block_on`) attempted to block the current thread while the thread is being used to drive asynchronous tasks."
        )]
        async fn blocking_lock_from_async_context_with_sync_api() {
            let p = $lockable_type::<isize, String>::new();
            let _ = p.blocking_lock(3);
        }

        #[tokio::test]
        #[should_panic(
            expected = "Cannot start a runtime from within a runtime. This happens because a function (like `block_on`) attempted to block the current thread while the thread is being used to drive asynchronous tasks."
        )]
        async fn blocking_lock_owned_from_async_context_with_sync_api() {
            let p = Arc::new($lockable_type::<isize, String>::new());
            let _ = p.blocking_lock_owned(3);
        }

        mod simple {
            use super::*;

            #[tokio::test]
            async fn async_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let guard = pool.async_lock(4).await;
                assert!(guard.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[tokio::test]
            async fn async_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let guard = pool.async_lock_owned(4).await;
                assert!(guard.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let guard = pool.blocking_lock(4);
                assert!(guard.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let guard = pool.blocking_lock_owned(4);
                assert!(guard.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let guard = pool.try_lock(4).unwrap();
                assert!(guard.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let guard = pool.try_lock_owned(4).unwrap();
                assert!(guard.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.num_entries_or_locked());
            }
        }

        mod try_lock {
            use super::*;

            #[test]
            fn try_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock(5);

                let error = pool.try_lock(5).unwrap_err();
                assert!(matches!(error, TryLockError::WouldBlock));

                // Check that we can stil lock other locks while the child is waiting
                {
                    let _g = pool.try_lock(4).unwrap();
                }

                // Now free the lock so the we can get it again
                std::mem::drop(guard);

                // And check that we can get it again
                {
                    let _g = pool.try_lock(5).unwrap();
                }

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock_owned(5);

                let error = pool.try_lock_owned(5).unwrap_err();
                assert!(matches!(error, TryLockError::WouldBlock));

                // Check that we can stil lock other locks while the child is waiting
                {
                    let _g = pool.try_lock_owned(4).unwrap();
                }

                // Now free the lock so the we can get it again
                std::mem::drop(guard);

                // And check that we can get it again
                {
                    let _g = pool.try_lock_owned(5).unwrap();
                }

                assert_eq!(0, pool.num_entries_or_locked());
            }
        }

        mod adding_cache_entries {
            use super::*;

            #[tokio::test]
            async fn async_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let mut guard = pool.async_lock(4).await;
                guard.insert(String::from("Cache Entry Value"));
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(1, pool.num_entries_or_locked());
                assert_eq!(
                    pool.async_lock(4).await.value(),
                    Some(&String::from("Cache Entry Value"))
                );
            }

            #[tokio::test]
            async fn async_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let mut guard = pool.async_lock_owned(4).await;
                guard.insert(String::from("Cache Entry Value"));
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(1, pool.num_entries_or_locked());
                assert_eq!(
                    pool.async_lock_owned(4).await.value(),
                    Some(&String::from("Cache Entry Value"))
                );
            }

            #[test]
            fn blocking_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let mut guard = pool.blocking_lock(4);
                guard.insert(String::from("Cache Entry Value"));
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(1, pool.num_entries_or_locked());
                assert_eq!(
                    pool.blocking_lock(4).value(),
                    Some(&String::from("Cache Entry Value"))
                );
            }

            #[test]
            fn blocking_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let mut guard = pool.blocking_lock_owned(4);
                guard.insert(String::from("Cache Entry Value"));
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(1, pool.num_entries_or_locked());
                assert_eq!(
                    pool.blocking_lock_owned(4).value(),
                    Some(&String::from("Cache Entry Value"))
                );
            }

            #[test]
            fn try_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let mut guard = pool.try_lock(4).unwrap();
                guard.insert(String::from("Cache Entry Value"));
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(1, pool.num_entries_or_locked());
                assert_eq!(
                    pool.try_lock(4).unwrap().value(),
                    Some(&String::from("Cache Entry Value"))
                );
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let mut guard = pool.try_lock_owned(4).unwrap();
                guard.insert(String::from("Cache Entry Value"));
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(1, pool.num_entries_or_locked());
                assert_eq!(
                    pool.try_lock_owned(4).unwrap().value(),
                    Some(&String::from("Cache Entry Value"))
                );
            }
        }

        mod removing_cache_entries {
            use super::*;

            #[tokio::test]
            async fn async_lock() {
                let pool = $lockable_type::<isize, String>::new();
                pool.async_lock(4)
                    .await
                    .insert(String::from("Cache Entry Value"));

                assert_eq!(1, pool.num_entries_or_locked());
                let mut guard = pool.async_lock(4).await;
                guard.remove();
                std::mem::drop(guard);

                assert_eq!(0, pool.num_entries_or_locked());
                assert_eq!(pool.async_lock(4).await.value(), None);
            }

            #[tokio::test]
            async fn async_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.async_lock_owned(4)
                    .await
                    .insert(String::from("Cache Entry Value"));

                assert_eq!(1, pool.num_entries_or_locked());
                let mut guard = pool.async_lock_owned(4).await;
                guard.remove();
                std::mem::drop(guard);

                assert_eq!(0, pool.num_entries_or_locked());
                assert_eq!(pool.async_lock_owned(4).await.value(), None);
            }

            #[test]
            fn blocking_lock() {
                let pool = $lockable_type::<isize, String>::new();
                pool.blocking_lock(4)
                    .insert(String::from("Cache Entry Value"));

                assert_eq!(1, pool.num_entries_or_locked());
                let mut guard = pool.blocking_lock(4);
                guard.remove();
                std::mem::drop(guard);

                assert_eq!(0, pool.num_entries_or_locked());
                assert_eq!(pool.blocking_lock(4).value(), None);
            }

            #[test]
            fn blocking_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.blocking_lock_owned(4)
                    .insert(String::from("Cache Entry Value"));

                assert_eq!(1, pool.num_entries_or_locked());
                let mut guard = pool.blocking_lock_owned(4);
                guard.remove();
                std::mem::drop(guard);

                assert_eq!(0, pool.num_entries_or_locked());
                assert_eq!(pool.blocking_lock_owned(4).value(), None);
            }

            #[test]
            fn try_lock() {
                let pool = $lockable_type::<isize, String>::new();
                pool.try_lock(4)
                    .unwrap()
                    .insert(String::from("Cache Entry Value"));

                assert_eq!(1, pool.num_entries_or_locked());
                let mut guard = pool.try_lock(4).unwrap();
                guard.remove();
                std::mem::drop(guard);

                assert_eq!(0, pool.num_entries_or_locked());
                assert_eq!(pool.try_lock(4).unwrap().value(), None);
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.try_lock_owned(4)
                    .unwrap()
                    .insert(String::from("Cache Entry Value"));

                assert_eq!(1, pool.num_entries_or_locked());
                let mut guard = pool.try_lock_owned(4).unwrap();
                guard.remove();
                std::mem::drop(guard);

                assert_eq!(0, pool.num_entries_or_locked());
                assert_eq!(pool.try_lock_owned(4).unwrap().value(), None);
            }
        }

        mod multi {
            use super::*;

            #[tokio::test]
            async fn async_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let guard1 = pool.async_lock(1).await;
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                let guard2 = pool.async_lock(2).await;
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.num_entries_or_locked());
                let guard3 = pool.async_lock(3).await;
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[tokio::test]
            async fn async_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let guard1 = pool.async_lock_owned(1).await;
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                let guard2 = pool.async_lock_owned(2).await;
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.num_entries_or_locked());
                let guard3 = pool.async_lock_owned(3).await;
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let guard1 = pool.blocking_lock(1);
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                let guard2 = pool.blocking_lock(2);
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.num_entries_or_locked());
                let guard3 = pool.blocking_lock(3);
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let guard1 = pool.blocking_lock_owned(1);
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                let guard2 = pool.blocking_lock_owned(2);
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.num_entries_or_locked());
                let guard3 = pool.blocking_lock_owned(3);
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock() {
                let pool = $lockable_type::<isize, String>::new();
                assert_eq!(0, pool.num_entries_or_locked());
                let guard1 = pool.try_lock(1).unwrap();
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                let guard2 = pool.try_lock(2).unwrap();
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.num_entries_or_locked());
                let guard3 = pool.try_lock(3).unwrap();
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                assert_eq!(0, pool.num_entries_or_locked());
                let guard1 = pool.try_lock_owned(1).unwrap();
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.num_entries_or_locked());
                let guard2 = pool.try_lock_owned(2).unwrap();
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.num_entries_or_locked());
                let guard3 = pool.try_lock_owned(3).unwrap();
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.num_entries_or_locked());
            }
        }

        mod concurrent {
            use super::*;

            #[tokio::test]
            async fn async_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock(5).await;

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_async_lock(&pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = pool.async_lock(4).await;
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[tokio::test]
            async fn async_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock_owned(5).await;

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_async_lock_owned(&pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = pool.async_lock_owned(4).await;
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock(5);

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_blocking_lock(&pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = pool.blocking_lock(4);
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock_owned(5);

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_blocking_lock_owned(&pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = pool.blocking_lock_owned(4);
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.try_lock(5).unwrap();

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_try_lock(&pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = pool.try_lock(4).unwrap();
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.try_lock_owned(5).unwrap();

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_try_lock_owned(&pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = pool.try_lock_owned(4).unwrap();
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }
        }

        mod multi_concurrent {
            use super::*;

            #[tokio::test]
            async fn async_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock(5).await;

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_async_lock(&pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_async_lock(&pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = pool.async_lock(4).await;
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(100));
                assert_eq!(1, counter.load(Ordering::SeqCst));

                // Allow the child to free the lock
                std::mem::drop(barrier_guard);

                // Check that the other child got it
                child1.join().unwrap();
                child2.join().unwrap();
                assert_eq!(2, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[tokio::test]
            async fn async_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock_owned(5).await;

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_async_lock_owned(&pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_async_lock_owned(&pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = pool.async_lock_owned(4).await;
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(100));
                assert_eq!(1, counter.load(Ordering::SeqCst));

                // Allow the child to free the lock
                std::mem::drop(barrier_guard);

                // Check that the other child got it
                child1.join().unwrap();
                child2.join().unwrap();
                assert_eq!(2, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock(5);

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_blocking_lock(&pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_blocking_lock(&pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = pool.blocking_lock(4);
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(100));
                assert_eq!(1, counter.load(Ordering::SeqCst));

                // Allow the child to free the lock
                std::mem::drop(barrier_guard);

                // Check that the other child got it
                child1.join().unwrap();
                child2.join().unwrap();
                assert_eq!(2, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn blocking_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock_owned(5);

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_blocking_lock_owned(&pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_blocking_lock_owned(&pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = pool.blocking_lock_owned(4);
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(100));
                assert_eq!(1, counter.load(Ordering::SeqCst));

                // Allow the child to free the lock
                std::mem::drop(barrier_guard);

                // Check that the other child got it
                child1.join().unwrap();
                child2.join().unwrap();
                assert_eq!(2, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.try_lock(5).unwrap();

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_try_lock(&pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_try_lock(&pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the children are waiting
                {
                    let _g = pool.try_lock(4).unwrap();
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(100));
                assert_eq!(1, counter.load(Ordering::SeqCst));

                // Allow the child to free the lock
                std::mem::drop(barrier_guard);

                // Check that the other child got it
                child1.join().unwrap();
                child2.join().unwrap();
                assert_eq!(2, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.try_lock_owned(5).unwrap();

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_try_lock_owned(&pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_try_lock_owned(&pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = pool.try_lock_owned(4).unwrap();
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(100));
                assert_eq!(1, counter.load(Ordering::SeqCst));

                // Allow the child to free the lock
                std::mem::drop(barrier_guard);

                // Check that the other child got it
                child1.join().unwrap();
                child2.join().unwrap();
                assert_eq!(2, counter.load(Ordering::SeqCst));

                assert_eq!(0, pool.num_entries_or_locked());
            }
        }

        #[test]
        fn blocking_lock_owned_guards_can_be_passed_around() {
            let make_guard = || {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.blocking_lock_owned(5)
            };
            let _guard = make_guard();
        }

        #[tokio::test]
        async fn async_lock_owned_guards_can_be_passed_around() {
            let make_guard = || async {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.async_lock_owned(5).await
            };
            let _guard = make_guard().await;
        }

        #[test]
        fn test_try_lock_owned_guards_can_be_passed_around() {
            let make_guard = || {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.try_lock_owned(5)
            };
            let guard = make_guard();
            assert!(guard.is_ok());
        }

        #[tokio::test]
        async fn async_lock_guards_can_be_held_across_await_points() {
            let task = async {
                let pool = $lockable_type::<isize, String>::new();
                let guard = pool.async_lock(3).await;
                tokio::time::sleep(Duration::from_millis(10)).await;
                std::mem::drop(guard);
            };

            // We also need to move the task to a different thread because
            // only then the compiler checks whether the task is Send.
            thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(task);
            });
        }

        #[tokio::test]
        async fn async_lock_owned_guards_can_be_held_across_await_points() {
            let task = async {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock_owned(3).await;
                tokio::time::sleep(Duration::from_millis(10)).await;
                std::mem::drop(guard);
            };

            // We also need to move the task to a different thread because
            // only then the compiler checks whether the task is Send.
            thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(task);
            });
        }
    }
}
