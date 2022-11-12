//! Some generic tests applicable to all Lockable hash map types.

// TODO Add a test adding multiple entries and making sure all locking functions can read them
// TODO Add tests checking that the async_lock, lock_owned, lock methods all block each other. For lock and lock_owned that can probably go into common tests.rs
// TODO Test `limit` parameter of all locking functions

use crate::guard::TryInsertError;
use crate::lockable_map_impl::LockableMapImpl;
use crate::map_like::ArcMutexMapLike;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;

/// Asserts that both vectors have the same entries but they may have a different order
pub(crate) fn assert_vec_eq_unordered<T: Ord + Eq + Debug>(mut lhs: Vec<T>, mut rhs: Vec<T>) {
    lhs.sort();
    rhs.sort();
    assert_eq!(lhs, rhs);
}

pub(crate) trait Guard<K, V> {
    fn key(&self) -> &K;
    fn value(&self) -> Option<&V>;
    fn value_mut(&mut self) -> Option<&mut V>;
    fn value_or_insert(&mut self, value: V) -> &mut V;
    fn value_or_insert_with(&mut self, value_fn: impl FnOnce() -> V) -> &mut V;
    fn insert(&mut self, value: V) -> Option<V>;
    fn try_insert(&mut self, value: V) -> Result<&mut V, TryInsertError<V>>;
    fn remove(&mut self) -> Option<V>;
}
impl<M, V, H, P> Guard<M::K, V> for crate::guard::Guard<M, V, H, P>
where
    M: ArcMutexMapLike,
    H: crate::hooks::Hooks<M::V>,
    M::V: Borrow<V> + BorrowMut<V> + crate::lockable_map_impl::FromInto<V>,
    P: Borrow<LockableMapImpl<M, V, H>>,
{
    fn key(&self) -> &M::K {
        crate::guard::Guard::key(self)
    }
    fn value(&self) -> Option<&V> {
        crate::guard::Guard::value(self)
    }
    fn value_mut(&mut self) -> Option<&mut V> {
        crate::guard::Guard::value_mut(self)
    }
    fn value_or_insert_with(&mut self, value_fn: impl FnOnce() -> V) -> &mut V {
        crate::guard::Guard::value_or_insert_with(self, value_fn)
    }
    fn value_or_insert(&mut self, value: V) -> &mut V {
        crate::guard::Guard::value_or_insert(self, value)
    }
    fn insert(&mut self, value: V) -> Option<V> {
        crate::guard::Guard::insert(self, value)
    }
    fn try_insert(&mut self, value: V) -> Result<&mut V, TryInsertError<V>> {
        crate::guard::Guard::try_insert(self, value)
    }
    fn remove(&mut self) -> Option<V> {
        crate::guard::Guard::remove(self)
    }
}

#[macro_export]
macro_rules! instantiate_lockable_tests {
    (@gen_tests, $lockable_type: ident, $test_sync_fn: ident, $test_async_fn: ident) => {
        #[tokio::test]
        async fn async_lock() {
            $test_async_fn(AsyncLock).await;
        }

        #[tokio::test]
        async fn async_lock_owned() {
            $test_async_fn(AsyncLockOwned).await;
        }

        #[tokio::test]
        async fn try_lock_async() {
            $test_async_fn(TryLockAsync).await;
        }

        #[tokio::test]
        async fn try_lock_owned_async() {
            $test_async_fn(TryLockOwnedAsync).await;
        }

        #[test]
        fn blocking_lock() {
            $test_sync_fn(BlockingLock);
        }

        #[test]
        fn blocking_lock_owned() {
            $test_sync_fn(BlockingLockOwned);
        }

        #[test]
        fn try_lock() {
            $test_sync_fn(TryLock);
        }

        #[test]
        fn try_lock_owned() {
            $test_sync_fn(TryLockOwned);
        }
    };
    ($lockable_type: ident) => {
        use async_trait::async_trait;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::{Arc, Mutex};
        use std::thread::{self, JoinHandle};
        use std::time::Duration;
        use futures::stream::StreamExt;
        use crate::{Lockable, InfallibleUnwrap, TryInsertError, tests::Guard};

        /// A trait that allows our test cases to abstract over different sync locking methods
        /// (i.e. blocking_lock, blocking_lock_owned, try_lock, try_lock_owned)
        trait SyncLocking<S, K, V> : Clone + Send
        where
            S: Borrow<$lockable_type<K, V>>,
            K: Eq + PartialEq + Hash + Clone,
        {
            type Guard<'a> : $crate::tests::Guard<K, V> where S: 'a;

            fn new(&self) -> S;
            fn lock<'a>(&self, map: &'a S, key: K) -> Self::Guard<'a>;
            fn lock_waiting_is_ok<'a>(&self, map: &'a S, key: K) -> Self::Guard<'a> {
                self.lock(map, key)
            }
            fn extract(&self, s: S) -> $lockable_type<K, V>;
        }
        #[derive(Clone, Copy)]
        struct BlockingLock;
        impl <K, V> SyncLocking<$lockable_type::<K, V>, K, V> for BlockingLock
        where
            K: Eq + PartialEq + Hash + Clone,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::Guard<'a> where $lockable_type<K, V>: 'a;

            fn new(&self) -> $lockable_type<K, V> {
                $lockable_type::<K, V>::new()
            }
            fn lock<'a>(&self, map: &'a $lockable_type::<K, V>, key: K) -> Self::Guard<'a> {
                map.blocking_lock(key, SyncLimit::no_limit()).infallible_unwrap()
            }
            fn extract(&self, s: $lockable_type<K, V>) -> $lockable_type<K, V> {
                s
            }
        }
        #[derive(Clone, Copy)]
        struct BlockingLockOwned;
        impl <K, V> SyncLocking<Arc<$lockable_type::<K, V>>, K, V> for BlockingLockOwned
        where
            K: Eq + PartialEq + Hash + Clone + Debug + 'static,
            V: Debug + 'static,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::OwnedGuard;

            fn new(&self) -> Arc<$lockable_type::<K, V>> {
                Arc::new($lockable_type::<K, V>::new())
            }
            fn lock<'a>(&self, map: &'a Arc<$lockable_type<K, V>>, key: K) -> Self::Guard<'a> {
                map.blocking_lock_owned(key, SyncLimit::no_limit()).infallible_unwrap()
            }
            fn extract(&self, s: Arc<$lockable_type<K, V>>) -> $lockable_type<K, V> {
                Arc::try_unwrap(s).unwrap()
            }
        }
        #[derive(Clone, Copy)]
        struct TryLock;
        impl <K, V> SyncLocking<$lockable_type::<K, V>, K, V> for TryLock
        where
            K: Eq + PartialEq + Hash + Clone,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::Guard<'a> where $lockable_type<K, V>: 'a;

            fn new(&self) -> $lockable_type<K, V> {
                $lockable_type::<K, V>::new()
            }
            fn lock<'a>(&self, map: &'a $lockable_type<K, V>, key: K) -> Self::Guard<'a> {
                map.try_lock(key, SyncLimit::no_limit()).infallible_unwrap().expect("Entry already locked")
            }
            fn lock_waiting_is_ok<'a>(&self, map: &'a $lockable_type<K, V>, key: K) -> Self::Guard<'a> {
                let start = std::time::Instant::now();
                loop {
                    if let Some(guard) = map.try_lock(key.clone(), SyncLimit::no_limit()).infallible_unwrap() {
                        break guard;
                    }
                    if std::time::Instant::now() - start > Duration::from_secs(10) {
                        panic!("Timeout trying to get lock in TryLock::lock_waiting_is_ok");
                    }
                }
            }
            fn extract(&self, s: $lockable_type<K, V>) -> $lockable_type<K, V> {
                s
            }
        }
        #[derive(Clone, Copy)]
        struct TryLockOwned;
        impl <K, V> SyncLocking<Arc<$lockable_type::<K, V>>, K, V> for TryLockOwned
        where
            K: Eq + PartialEq + Hash + Clone + Debug + 'static,
            V: Debug + 'static,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::OwnedGuard;

            fn new(&self) -> Arc<$lockable_type::<K, V>> {
                Arc::new($lockable_type::<K, V>::new())
            }
            fn lock<'a>(&self, map: &'a Arc<$lockable_type<K, V>>, key: K) -> Self::Guard<'a> {
                let start = std::time::Instant::now();
                loop {
                    if let Some(guard) = map.try_lock_owned(key.clone(), SyncLimit::no_limit()).infallible_unwrap() {
                        break guard;
                    }
                    if std::time::Instant::now() - start > Duration::from_secs(10) {
                        panic!("Timeout trying to get lock in TryLockOwned::lock_waiting_is_ok");
                    }
                }
            }
            fn extract(&self, s: Arc<$lockable_type<K, V>>) -> $lockable_type<K, V> {
                Arc::try_unwrap(s).unwrap()
            }
        }
        /// A trait that allows our test cases to abstract over different async locking methods
        /// (i.e. async_lock, async_lock_owned, try_lock_async, try_lock_owned_async)
        #[async_trait]
        trait AsyncLocking<S, K, V> : Clone + Send
        where
            S: Borrow<$lockable_type::<K, V>> + Sync,
            K: Eq + PartialEq + Hash + Clone + Send + 'static,
        {
            type Guard<'a> : $crate::tests::Guard<K, V> where S: 'a;

            fn new(&self) -> S;
            async fn lock<'a>(&self, map: &'a S, key: K) -> Self::Guard<'a>;
            async fn lock_waiting_is_ok<'a>(&self, map: &'a S, key: K) -> Self::Guard<'a> {
                self.lock(map, key).await
            }
            fn extract(&self, s: S) -> $lockable_type<K, V>;
        }
        #[derive(Clone, Copy)]
        struct TryLockAsync;
        #[async_trait]
        impl <K, V> AsyncLocking<$lockable_type::<K, V>, K, V> for TryLockAsync
        where
            K: Eq + PartialEq + Hash + Clone + Send + Sync + 'static,
            V: Send + Sync,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::Guard<'a> where $lockable_type<K, V>: 'a;

            fn new(&self) -> $lockable_type<K, V> {
                $lockable_type::<K, V>::new()
            }
            async fn lock<'a>(&self, map: &'a $lockable_type::<K, V>, key: K) -> Self::Guard<'a> {
                map.try_lock_async(key, AsyncLimit::no_limit()).await.infallible_unwrap().expect("Entry already locked")
            }
            async fn lock_waiting_is_ok<'a>(&self, map: &'a $lockable_type::<K, V>, key: K) -> Self::Guard<'a> {
                let start = std::time::Instant::now();
                loop {
                    if let Some(guard) = map.try_lock_async(key.clone(), AsyncLimit::no_limit()).await.infallible_unwrap() {
                        break guard;
                    }
                    if std::time::Instant::now() - start > Duration::from_secs(10) {
                        panic!("Timeout trying to get lock in TryLockAsync::lock_waiting_is_ok");
                    }
                }
            }
            fn extract(&self, s: $lockable_type<K, V>) -> $lockable_type<K, V> {
                s
            }
        }
        #[derive(Clone, Copy)]
        struct TryLockOwnedAsync;
        #[async_trait]
        impl <K, V> AsyncLocking<Arc<$lockable_type::<K, V>>, K, V> for TryLockOwnedAsync
        where
            K: Eq + PartialEq + Hash + Clone + Send + Sync + Debug + 'static,
            V: Send + Sync + Debug + 'static,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::OwnedGuard;

            fn new(&self) -> Arc<$lockable_type::<K, V>> {
                Arc::new($lockable_type::<K, V>::new())
            }
            async fn lock<'a>(&self, map: &'a Arc<$lockable_type<K, V>>, key: K) -> Self::Guard<'a> {
                let start = std::time::Instant::now();
                loop {
                    if let Some(guard) = map.try_lock_owned_async(key.clone(), AsyncLimit::no_limit()).await.infallible_unwrap() {
                        break guard;
                    }
                    if std::time::Instant::now() - start > Duration::from_secs(10) {
                        panic!("Timeout trying to get lock in TryLockAsync::lock_waiting_is_ok");
                    }
                }
            }
            fn extract(&self, s: Arc<$lockable_type<K, V>>) -> $lockable_type<K, V> {
                Arc::try_unwrap(s).unwrap()
            }
        }
        #[derive(Clone, Copy)]
        struct AsyncLock;
        #[async_trait]
        impl <K, V> AsyncLocking<$lockable_type::<K, V>, K, V> for AsyncLock
        where
            K: Eq + PartialEq + Hash + Clone + Send + Sync + 'static,
            V: Send + Sync,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::Guard<'a> where $lockable_type<K, V>: 'a;

            fn new(&self) -> $lockable_type<K, V> {
                $lockable_type::<K, V>::new()
            }
            async fn lock<'a>(&self, map: &'a $lockable_type::<K, V>, key: K) -> Self::Guard<'a> {
                map.async_lock(key, AsyncLimit::no_limit()).await.infallible_unwrap()
            }
            fn extract(&self, s: $lockable_type<K, V>) -> $lockable_type<K, V> {
                s
            }
        }
        #[derive(Clone, Copy)]
        struct AsyncLockOwned;
        #[async_trait]
        impl <K, V> AsyncLocking<Arc<$lockable_type::<K, V>>, K, V> for AsyncLockOwned
        where
            K: Eq + PartialEq + Hash + Clone + Send + Sync + Debug + 'static,
            V: Send + Sync + Debug + 'static,
        {
            type Guard<'a> = <$lockable_type::<K, V> as Lockable<K, V>>::OwnedGuard;

            fn new(&self) -> Arc<$lockable_type::<K, V>> {
                Arc::new($lockable_type::<K, V>::new())
            }
            async fn lock<'a>(&self, map: &'a Arc<$lockable_type<K, V>>, key: K) -> Self::Guard<'a> {
                map.async_lock_owned(key, AsyncLimit::no_limit()).await.infallible_unwrap()
            }
            fn extract(&self, s: Arc<$lockable_type<K, V>>) -> $lockable_type<K, V> {
                Arc::try_unwrap(s).unwrap()
            }
        }

        // Launch a thread that
        // 1. locks the given key
        // 2. once it has the lock, increments a counter
        // 3. then waits until a barrier is released before it releases the lock
        fn launch_thread_sync_lock<S, L>(
            locking: &L,
            pool: &Arc<S>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()>
        where
            S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            L: SyncLocking<S, isize, String> + Sync + 'static,
        {
            let locking = (*locking).clone();
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let _guard = locking.lock_waiting_is_ok(&pool, key);
                counter.fetch_add(1, Ordering::SeqCst);
                if let Some(barrier) = barrier {
                    let _barrier = barrier.lock().unwrap();
                }
            })
        }

        fn launch_thread_async_lock<S, L>(
            locking: &L,
            pool: &Arc<S>,
            key: isize,
            counter: &Arc<AtomicU32>,
            barrier: Option<&Arc<Mutex<()>>>,
        ) -> JoinHandle<()>
        where
            S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            L: AsyncLocking<S, isize, String> + Sync + 'static,
        {
            let locking = (*locking).clone();
            let pool = Arc::clone(pool);
            let counter = Arc::clone(counter);
            let barrier = barrier.map(Arc::clone);
            thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                let _guard = runtime.block_on(locking.lock_waiting_is_ok(&pool, key));
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
            let _ = p.blocking_lock(3, SyncLimit::no_limit());
        }

        #[tokio::test]
        #[should_panic(
            expected = "Cannot start a runtime from within a runtime. This happens because a function (like `block_on`) attempted to block the current thread while the thread is being used to drive asynchronous tasks."
        )]
        async fn blocking_lock_owned_from_async_context_with_sync_api() {
            let p = Arc::new($lockable_type::<isize, String>::new());
            let _ = p.blocking_lock_owned(3, SyncLimit::no_limit());
        }

        mod simple {
            use super::*;

            fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
            where
                S: Borrow<$lockable_type<isize, String>>,
            {
                let pool = locking.new();
                assert_eq!(0, pool.borrow().num_entries_or_locked());
                let guard = locking.lock(&pool, 4);
                assert!(guard.value().is_none());
                assert_eq!(1, pool.borrow().num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.borrow().num_entries_or_locked());
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
            where
                S: Borrow<$lockable_type<isize, String>> + Sync,
            {
                let pool = locking.new();
                assert_eq!(0, pool.borrow().num_entries_or_locked());
                let guard = locking.lock(&pool, 4).await;
                assert!(guard.value().is_none());
                assert_eq!(1, pool.borrow().num_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(0, pool.borrow().num_entries_or_locked());
            }

            $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
        }

        mod try_lock {
            use super::*;

            #[test]
            fn try_lock() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock(5, SyncLimit::no_limit()).unwrap();

                let result = pool.try_lock(5, SyncLimit::no_limit()).unwrap();
                assert!(result.is_none());

                // Check that we can stil lock other locks while the child is waiting
                {
                    let _g = pool.try_lock(4, SyncLimit::no_limit()).unwrap().unwrap();
                }

                // Now free the lock so the we can get it again
                std::mem::drop(guard);

                // And check that we can get it again
                {
                    let _g = pool.try_lock(5, SyncLimit::no_limit()).unwrap().unwrap();
                }

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[test]
            fn try_lock_owned() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.blocking_lock_owned(5, SyncLimit::no_limit()).unwrap();

                let result = pool.try_lock_owned(5, SyncLimit::no_limit()).unwrap();
                assert!(result.is_none());

                // Check that we can stil lock other locks while the child is waiting
                {
                    let _g = pool.try_lock_owned(4, SyncLimit::no_limit()).unwrap().unwrap();
                }

                // Now free the lock so the we can get it again
                std::mem::drop(guard);

                // And check that we can get it again
                {
                    let _g = pool.try_lock_owned(5, SyncLimit::no_limit()).unwrap().unwrap();
                }

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[tokio::test]
            async fn try_lock_async() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock(5, AsyncLimit::no_limit()).await.unwrap();

                let result = pool.try_lock_async(5, AsyncLimit::no_limit()).await.unwrap();
                assert!(result.is_none());

                // Check that we can stil lock other locks while the child is waiting
                {
                    let _g = pool.try_lock_async(4, AsyncLimit::no_limit()).await.unwrap().unwrap();
                }

                // Now free the lock so the we can get it again
                std::mem::drop(guard);

                // And check that we can get it again
                {
                    let _g = pool.try_lock_async(5, AsyncLimit::no_limit()).await.unwrap().unwrap();
                }

                assert_eq!(0, pool.num_entries_or_locked());
            }

            #[tokio::test]
            async fn try_lock_owned_async() {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock_owned(5, AsyncLimit::no_limit()).await.unwrap();

                let result = pool.try_lock_owned_async(5, AsyncLimit::no_limit()).await.unwrap();
                assert!(result.is_none());

                // Check that we can stil lock other locks while the child is waiting
                {
                    let _g = pool.try_lock_owned_async(4, AsyncLimit::no_limit()).await.unwrap().unwrap();
                }

                // Now free the lock so the we can get it again
                std::mem::drop(guard);

                // And check that we can get it again
                {
                    let _g = pool.try_lock_owned_async(5, AsyncLimit::no_limit()).await.unwrap().unwrap();
                }

                assert_eq!(0, pool.num_entries_or_locked());
            }
        }

        mod guard {
            use super::*;

            mod insert {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        let prev_value = locking.lock(&pool, 4).insert(String::from("Previous value"));
                        assert_eq!(None, prev_value);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4);
                        let prev_value = guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(Some(String::from("Previous value")), prev_value);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        let prev_value = locking.lock(&pool, 4).await.insert(String::from("Previous value"));
                        assert_eq!(None, prev_value);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4).await;
                        let prev_value = guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(Some(String::from("Previous value")), prev_value);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4);
                        let prev_value = guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(None, prev_value);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4).await;
                        let prev_value = guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(None, prev_value);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod try_insert {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4).insert(String::from("Previous Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4);
                        let result = guard.try_insert(String::from("Cache Entry Value"));
                        assert_eq!(Err(TryInsertError::AlreadyExists{value: String::from("Cache Entry Value")}), result);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Previous Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4).await.insert(String::from("Previous Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4).await;
                        let result = guard.try_insert(String::from("Cache Entry Value"));
                        assert_eq!(Err(TryInsertError::AlreadyExists{value: String::from("Cache Entry Value")}), result);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Previous Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4);
                        let new_entry = guard.try_insert(String::from("Cache Entry Value"));
                        assert_eq!(String::from("Cache Entry Value"), *new_entry.unwrap());
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4).await;
                        let new_entry = guard.try_insert(String::from("Cache Entry Value"));
                        assert_eq!(String::from("Cache Entry Value"), *new_entry.unwrap());
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry_and_modify_it_through_the_returned_reference {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4);
                        let new_entry = guard.try_insert(String::from("Cache Entry Value"));
                        *new_entry.unwrap() = String::from("New Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("New Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 4).await;
                        let new_entry = guard.try_insert(String::from("Cache Entry Value"));
                        *new_entry.unwrap() = String::from("New Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());

                        std::mem::drop(guard);
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("New Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod remove {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4)
                            .insert(String::from("Cache Entry Value"));

                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        assert_eq!(Some("Cache Entry Value".into()), guard.remove());
                        std::mem::drop(guard);

                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(locking.lock(&pool, 4).value(), None);
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4)
                            .await
                            .insert(String::from("Cache Entry Value"));

                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        assert_eq!(Some("Cache Entry Value".into()), guard.remove());
                        std::mem::drop(guard);

                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(locking.lock(&pool, 4).await.value(), None);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry_in_nonempty_map {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4)
                            .insert(String::from("Cache Entry Value"));

                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 5);
                        assert_eq!(None, guard.remove());
                        std::mem::drop(guard);

                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(locking.lock(&pool, 4).value(), Some(&String::from("Cache Entry Value")));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4)
                            .await
                            .insert(String::from("Cache Entry Value"));

                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 5).await;
                        assert_eq!(None, guard.remove());
                        std::mem::drop(guard);

                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(locking.lock(&pool, 4).await.value(), Some(&String::from("Cache Entry Value")));
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry_in_empty_map {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 5);
                        assert_eq!(None, guard.remove());
                        std::mem::drop(guard);

                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(locking.lock(&pool, 4).value(), None);
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());

                        let mut guard = locking.lock(&pool, 5).await;
                        assert_eq!(None, guard.remove());
                        std::mem::drop(guard);

                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(locking.lock(&pool, 4).await.value(), None);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod value {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        std::mem::drop(guard);

                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        std::mem::drop(guard);

                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();

                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            None,
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();

                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            None,
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod value_mut {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).value_mut().unwrap());
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).await.value_mut().unwrap());
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod existing_entry_modify_returned_reference {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        *locking.lock(&pool, 4).value_mut().unwrap() = String::from("New Cache Entry Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("New Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        *locking.lock(&pool, 4).await.value_mut().unwrap() = String::from("New Cache Entry Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("New Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(None, locking.lock(&pool, 4).value_mut());
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(None, locking.lock(&pool, 4).await.value_mut());
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod value_or_insert {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).value_or_insert(String::from("New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).await.value_or_insert(String::from("Unused New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod existing_entry_modify_returned_reference {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        *locking.lock(&pool, 4).value_or_insert(String::from("Unused New Value")) = String::from("New Cache Entry Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("New Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        *locking.lock(&pool, 4).await.value_or_insert(String::from("Unused New Value")) = String::from("New Cache Entry Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("New Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(String::from("New Value"), *locking.lock(&pool, 4).value_or_insert(String::from("New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("New Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(String::from("New Value"), *locking.lock(&pool, 4).await.value_or_insert(String::from("New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("New Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry_modify_returned_reference {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        *locking.lock(&pool, 4).value_or_insert(String::from("New Value")) = String::from("Even Newer Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Even Newer Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        *locking.lock(&pool, 4).await.value_or_insert(String::from("New Value")) = String::from("Even Newer Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Even Newer Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod value_or_insert_with {
                use super::*;

                mod existing_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).value_or_insert_with(|| String::from("New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).await.value_or_insert_with(|| String::from("Unused New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod existing_entry_modify_returned_reference {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        *locking.lock(&pool, 4).value_or_insert_with(|| String::from("Unused New Value")) = String::from("New Cache Entry Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("New Cache Entry Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        *locking.lock(&pool, 4).await.value_or_insert_with(|| String::from("Unused New Value")) = String::from("New Cache Entry Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("New Cache Entry Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod existing_entry_then_callback_isnt_called {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4);
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).value_or_insert_with(|| panic!("Callback called")));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        let mut guard = locking.lock(&pool, 4).await;
                        guard.insert(String::from("Cache Entry Value"));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        std::mem::drop(guard);
                        assert_eq!(String::from("Cache Entry Value"), *locking.lock(&pool, 4).await.value_or_insert_with(|| panic!("Callback called")));
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(String::from("New Value"), *locking.lock(&pool, 4).value_or_insert_with(|| String::from("New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("New Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        assert_eq!(String::from("New Value"), *locking.lock(&pool, 4).await.value_or_insert_with(|| String::from("New Value")));
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("New Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod nonexisting_entry_modify_returned_reference {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        *locking.lock(&pool, 4).value_or_insert_with(|| String::from("New Value")) = String::from("Even Newer Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).value(),
                            Some(&String::from("Even Newer Value"))
                        );
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        assert_eq!(0, pool.borrow().num_entries_or_locked());
                        *locking.lock(&pool, 4).await.value_or_insert_with(|| String::from("New Value")) = String::from("Even Newer Value");
                        assert_eq!(1, pool.borrow().num_entries_or_locked());
                        assert_eq!(
                            locking.lock(&pool, 4).await.value(),
                            Some(&String::from("Even Newer Value"))
                        );
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod key {
                use super::*;

                fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>>,
                {
                    let pool = locking.new();
                    assert_eq!(0, pool.borrow().num_entries_or_locked());
                    let mut guard = locking.lock(&pool, 4);
                    assert_eq!(4, *guard.key());  // key of nonexisting entry
                    guard.insert(String::from("Cache Entry Value"));
                    std::mem::drop(guard);
                    assert_eq!(4, *locking.lock(&pool, 4).key());  // key of existing entry
                }

                async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>> + Sync,
                {
                    let pool = locking.new();
                    assert_eq!(0, pool.borrow().num_entries_or_locked());
                    let mut guard = locking.lock(&pool, 4).await;
                    assert_eq!(4, *guard.key());  // key of nonexisting entry
                    guard.insert(String::from("Cache Entry Value"));
                    std::mem::drop(guard);
                    assert_eq!(4, *locking.lock(&pool, 4).await.key()); // key of existing entry
                }

                $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
            }

            mod multiple_operations_on_same_guard {
                use super::*;

                fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>>,
                {
                    let pool = locking.new();
                    assert_eq!(0, pool.borrow().num_entries_or_locked());
                    let mut guard = locking.lock(&pool, 4);
                    assert_eq!(None, guard.value());

                    assert_eq!(None, guard.insert(String::from("Cache Entry Value")));
                    assert_eq!(&String::from("Cache Entry Value"), guard.value().unwrap());

                    *guard.value_mut().unwrap() = String::from("Another value");
                    assert_eq!(&String::from("Another value"), guard.value().unwrap());

                    assert_eq!(Some(String::from("Another value")), guard.remove());
                    assert_eq!(None, guard.value());

                    assert_eq!(None, guard.insert(String::from("Last Value")));

                    std::mem::drop(guard);
                    assert_eq!(String::from("Last Value"), *locking.lock(&pool, 4).value().unwrap());
                }

                async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>> + Sync,
                {
                    let pool = locking.new();
                    assert_eq!(0, pool.borrow().num_entries_or_locked());
                    let mut guard = locking.lock(&pool, 4).await;
                    assert_eq!(None, guard.value());

                    assert_eq!(None, guard.insert(String::from("Cache Entry Value")));
                    assert_eq!(&String::from("Cache Entry Value"), guard.value().unwrap());

                    *guard.value_mut().unwrap() = String::from("Another value");
                    assert_eq!(&String::from("Another value"), guard.value().unwrap());

                    assert_eq!(Some(String::from("Another value")), guard.remove());
                    assert_eq!(None, guard.value());

                    assert_eq!(None, guard.insert(String::from("Last Value")));

                    std::mem::drop(guard);
                    assert_eq!(String::from("Last Value"), *locking.lock(&pool, 4).await.value().unwrap());
                }

                $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
            }
        }

        mod keys_with_entries_or_locked {
            use super::*;

            fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
            where
                S: Borrow<$lockable_type<isize, String>>,
            {
                let pool = locking.new();
                assert_eq!(Vec::<isize>::new(), pool.borrow().keys_with_entries_or_locked());

                // Locking lists key, unlocking unlists key
                let guard = locking.lock(&pool, 4);
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(Vec::<isize>::new(), pool.borrow().keys_with_entries_or_locked());

                // If entry is inserted, it remains listed after unlocking
                let mut guard = locking.lock(&pool, 4);
                guard.insert(String::from("Value"));
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());

                // If entry is removed, it is not listed anymore after unlocking
                let mut guard = locking.lock(&pool, 4);
                guard.remove();
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(Vec::<isize>::new(), pool.borrow().keys_with_entries_or_locked());

                // Add multiple keys
                locking.lock(&pool, 4).insert(String::from("Content"));
                locking.lock(&pool, 5).insert(String::from("Content"));
                locking.lock(&pool, 6).insert(String::from("Content"));
                let mut keys = pool.borrow().keys_with_entries_or_locked();
                keys.sort();
                assert_eq!(vec![4, 5, 6], keys);
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
            where
                S: Borrow<$lockable_type<isize, String>> + Sync,
            {
                let pool = locking.new();
                assert_eq!(Vec::<isize>::new(), pool.borrow().keys_with_entries_or_locked());

                // Locking lists key, unlocking unlists key
                let guard = locking.lock(&pool, 4).await;
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(Vec::<isize>::new(), pool.borrow().keys_with_entries_or_locked());

                // If entry is inserted, it remains listed after unlocking
                let mut guard = locking.lock(&pool, 4).await;
                guard.insert(String::from("Value"));
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());

                // If entry is removed, it is not listed anymore after unlocking
                let mut guard = locking.lock(&pool, 4).await;
                guard.remove();
                assert_eq!(vec![4], pool.borrow().keys_with_entries_or_locked());
                std::mem::drop(guard);
                assert_eq!(Vec::<isize>::new(), pool.borrow().keys_with_entries_or_locked());

                // Add multiple keys
                locking.lock(&pool, 4).await.insert(String::from("Content"));
                locking.lock(&pool, 5).await.insert(String::from("Content"));
                locking.lock(&pool, 6).await.insert(String::from("Content"));
                let mut keys = pool.borrow().keys_with_entries_or_locked();
                keys.sort();
                assert_eq!(vec![4, 5, 6], keys);
            }

            $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
        }

        mod into_entries_unordered {
            use super::*;

            mod map_with_0_entries {
                use super::*;

                fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>>,
                {
                    let pool = locking.new();
                    let iter = locking.extract(pool).into_entries_unordered().collect::<Vec<(isize, String)>>();
                    assert_eq!(Vec::<(isize, String)>::new(), iter);
                }

                async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>> + Sync,
                {
                    let pool = locking.new();
                    let iter = locking.extract(pool).into_entries_unordered().collect::<Vec<(isize, String)>>();
                    assert_eq!(Vec::<(isize, String)>::new(), iter);
                }

                $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
            }

            mod map_with_1_entry {
                use super::*;

                fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>>,
                {
                    let pool = locking.new();
                    locking.lock(&pool, 4).insert(String::from("Value"));
                    let iter = locking.extract(pool).into_entries_unordered().collect::<Vec<(isize, String)>>();
                    $crate::tests::assert_vec_eq_unordered(vec![(4, String::from("Value"))], iter);
                }

                async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>> + Sync,
                {
                    let pool = locking.new();
                    locking.lock(&pool, 4).await.insert(String::from("Value"));
                    let iter = locking.extract(pool).into_entries_unordered().collect::<Vec<(isize, String)>>();
                    $crate::tests::assert_vec_eq_unordered(vec![(4, String::from("Value"))], iter);
                }

                $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
            }

            mod map_with_multiple_entries {
                use super::*;

                fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>>,
                {
                    let pool = locking.new();
                    locking.lock(&pool, 4).insert(String::from("Value 4"));
                    locking.lock(&pool, 5).insert(String::from("Value 5"));
                    locking.lock(&pool, 3).insert(String::from("Value 3"));
                    let iter = locking.extract(pool).into_entries_unordered().collect::<Vec<(isize, String)>>();
                    $crate::tests::assert_vec_eq_unordered(vec![
                        (3, String::from("Value 3")),
                        (4, String::from("Value 4")),
                        (5, String::from("Value 5")),
                    ], iter);
                }

                async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                where
                    S: Borrow<$lockable_type<isize, String>> + Sync,
                {
                    let pool = locking.new();
                    locking.lock(&pool, 4).await.insert(String::from("Value 4"));
                    locking.lock(&pool, 5).await.insert(String::from("Value 5"));
                    locking.lock(&pool, 3).await.insert(String::from("Value 3"));
                    let iter = locking.extract(pool).into_entries_unordered().collect::<Vec<(isize, String)>>();
                    $crate::tests::assert_vec_eq_unordered(vec![
                        (3, String::from("Value 3")),
                        (4, String::from("Value 4")),
                        (5, String::from("Value 5")),
                    ], iter);
                }

                $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
            }
        }

        mod lock_all_entries {
            use super::*;

            mod when_all_are_unlocked {
                use super::*;

                mod map_with_0_entries {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        let guards: Vec<(isize, Option<String>)> =
                            futures::executor::block_on(
                                futures::executor::block_on(pool.borrow().lock_all_entries())
                                    .map(|guard| (*guard.key(), guard.value().cloned()))
                                    .collect()
                            );
                        crate::tests::assert_vec_eq_unordered(Vec::<(isize, Option<String>)>::new(), guards);
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        let guards: Vec<(isize, Option<String>)> =
                            pool.borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()))
                                .collect()
                                .await;
                        crate::tests::assert_vec_eq_unordered(Vec::<(isize, Option<String>)>::new(), guards);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod map_with_1_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4).insert(String::from("Value"));

                        let guards: Vec<(isize, Option<String>)> =
                            futures::executor::block_on(
                                futures::executor::block_on(pool.borrow().lock_all_entries())
                                    .map(|guard| (*guard.key(), guard.value().cloned()))
                                    .collect()
                            );
                        crate::tests::assert_vec_eq_unordered(vec![(4, Some(String::from("Value")))], guards);
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4).await.insert(String::from("Value"));

                        let guards: Vec<(isize, Option<String>)> =
                            futures::executor::block_on(
                                futures::executor::block_on(pool.borrow().lock_all_entries())
                                    .map(|guard| (*guard.key(), guard.value().cloned()))
                                    .collect()
                            );
                        crate::tests::assert_vec_eq_unordered(vec![(4, Some(String::from("Value")))], guards);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod map_with_multiple_entries {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>>,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4).insert(String::from("Value 4"));
                        locking.lock(&pool, 5).insert(String::from("Value 5"));
                        locking.lock(&pool, 3).insert(String::from("Value 3"));

                        let guards: Vec<(isize, Option<String>)> =
                            futures::executor::block_on(
                                futures::executor::block_on(pool.borrow().lock_all_entries())
                                    .map(|guard| (*guard.key(), guard.value().cloned()))
                                    .collect()
                            );
                        crate::tests::assert_vec_eq_unordered(vec![
                            (3, Some(String::from("Value 3"))),
                            (4, Some(String::from("Value 4"))),
                            (5, Some(String::from("Value 5"))),
                        ], guards);
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Sync,
                    {
                        let pool = locking.new();
                        locking.lock(&pool, 4).await.insert(String::from("Value 4"));
                        locking.lock(&pool, 5).await.insert(String::from("Value 5"));
                        locking.lock(&pool, 3).await.insert(String::from("Value 3"));

                        let guards: Vec<(isize, Option<String>)> =
                        futures::executor::block_on(
                            futures::executor::block_on(pool.borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()))
                                .collect()
                        );
                        crate::tests::assert_vec_eq_unordered(vec![
                            (3, Some(String::from("Value 3"))),
                            (4, Some(String::from("Value 4"))),
                            (5, Some(String::from("Value 5"))),
                        ], guards);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }
            // TODO Test that currently locked entries will be produced in the stream and will lock the stream until they become unlocked
            // TODO Test that currently locked entries will be produced in the stream
            //     - if they were pre-existing before they are locked
            //     - if they are created while locked
            //     - but not if they are not pre-existing and not created
            //     - and also not if they are pre-existing but deleted while locked
            // TODO Test that lock_all_entries doesn't lock the whole map while the stream hasn't gotten all locks yet and still allows locking/unlocking locks.
            // TODO Duplicate the lock_all_entries test cases for lock_all_entries_owned (or use some macro to do it)
        }

        mod multi {
            use super::*;

            fn test_sync<S>(locking: impl SyncLocking<S, isize, String>)
            where
                S: Borrow<$lockable_type<isize, String>>,
            {
                let pool = locking.new();
                assert_eq!(0, pool.borrow().num_entries_or_locked());
                let guard1 = locking.lock(&pool, 1);
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.borrow().num_entries_or_locked());
                let guard2 = locking.lock(&pool, 2);
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.borrow().num_entries_or_locked());
                let guard3 = locking.lock(&pool, 3);
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.borrow().num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.borrow().num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.borrow().num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.borrow().num_entries_or_locked());
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String>)
            where
                S: Borrow<$lockable_type<isize, String>> + Sync,
            {
                let pool = locking.new();
                assert_eq!(0, pool.borrow().num_entries_or_locked());
                let guard1 = locking.lock(&pool, 1).await;
                assert!(guard1.value().is_none());
                assert_eq!(1, pool.borrow().num_entries_or_locked());
                let guard2 = locking.lock(&pool, 2).await;
                assert!(guard2.value().is_none());
                assert_eq!(2, pool.borrow().num_entries_or_locked());
                let guard3 = locking.lock(&pool, 3).await;
                assert!(guard3.value().is_none());
                assert_eq!(3, pool.borrow().num_entries_or_locked());

                std::mem::drop(guard2);
                assert_eq!(2, pool.borrow().num_entries_or_locked());
                std::mem::drop(guard1);
                assert_eq!(1, pool.borrow().num_entries_or_locked());
                std::mem::drop(guard3);
                assert_eq!(0, pool.borrow().num_entries_or_locked());
            }

            $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
        }

        mod concurrent {
            use super::*;

            fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            {
                let pool = Arc::new(locking.new());
                let guard = locking.lock(&*pool, 5);

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_sync_lock(&locking, &pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = locking.lock(&*pool, 4);
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, (*pool).borrow().num_entries_or_locked());
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            {
                let pool = Arc::new(locking.new());
                let guard = locking.lock(&*pool, 5).await;

                let counter = Arc::new(AtomicU32::new(0));

                let child = launch_thread_async_lock(&locking, &pool, 5, &counter, None);

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = locking.lock(&*pool, 4).await;
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.join().unwrap();
                assert_eq!(1, counter.load(Ordering::SeqCst));

                assert_eq!(0, (*pool).borrow().num_entries_or_locked());
            }

            $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
        }

        mod multi_concurrent {
            use super::*;

            fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            {
                let pool = Arc::new(locking.new());
                let guard = locking.lock(&pool, 5);

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_sync_lock(&locking, &pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_sync_lock(&locking, &pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = locking.lock(&pool, 4);
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

                assert_eq!(0, (*pool).borrow().num_entries_or_locked());
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            {
                let pool = Arc::new(locking.new());
                let guard = locking.lock(&pool, 5).await;

                let counter = Arc::new(AtomicU32::new(0));
                let barrier = Arc::new(Mutex::new(()));
                let barrier_guard = barrier.lock().unwrap();

                let child1 = launch_thread_async_lock(&locking, &pool, 5, &counter, Some(&barrier));
                let child2 = launch_thread_async_lock(&locking, &pool, 5, &counter, Some(&barrier));

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(100));
                assert_eq!(0, counter.load(Ordering::SeqCst));

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = locking.lock(&pool, 4).await;
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

                assert_eq!(0, (*pool).borrow().num_entries_or_locked());
            }

            $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
        }

        #[test]
        fn blocking_lock_owned_guards_can_be_passed_around() {
            let make_guard = || {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.blocking_lock_owned(5, SyncLimit::no_limit()).unwrap()
            };
            let _guard = make_guard();
        }

        #[tokio::test]
        async fn async_lock_owned_guards_can_be_passed_around() {
            let make_guard = || async {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.async_lock_owned(5, AsyncLimit::no_limit()).await.unwrap()
            };
            let _guard = make_guard().await;
        }

        #[test]
        fn test_try_lock_owned_guards_can_be_passed_around() {
            let make_guard = || {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                pool.try_lock_owned(5, SyncLimit::no_limit()).unwrap()
            };
            let guard = make_guard();
            assert!(guard.is_some());
        }

        fn assert_is_send(_v: impl Send) {}

        #[tokio::test]
        async fn async_lock_guards_can_be_held_across_await_points() {
            let task = async {
                let pool = $lockable_type::<isize, String>::new();
                let guard = pool.async_lock(3, AsyncLimit::no_limit()).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
                std::mem::drop(guard);
            };

            assert_is_send(task);
        }

        #[tokio::test]
        async fn async_lock_owned_guards_can_be_held_across_await_points() {
            let task = async {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.async_lock_owned(3, AsyncLimit::no_limit()).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
                std::mem::drop(guard);
            };

            assert_is_send(task);
        }

        #[tokio::test]
        async fn try_lock_async_guards_can_be_held_across_await_points() {
            let task = async {
                let pool = $lockable_type::<isize, String>::new();
                let guard = pool.try_lock_async(3, AsyncLimit::no_limit()).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
                std::mem::drop(guard);
            };

            assert_is_send(task);
        }

        #[tokio::test]
        async fn try_lock_owned_async_guards_can_be_held_across_await_points() {
            let task = async {
                let pool = Arc::new($lockable_type::<isize, String>::new());
                let guard = pool.try_lock_owned_async(3, AsyncLimit::no_limit()).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
                std::mem::drop(guard);
            };

            assert_is_send(task);
        }
    }
}
