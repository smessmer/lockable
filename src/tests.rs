//! Some generic tests applicable to all Lockable hash map types.

// TODO Add a test adding multiple entries and making sure all locking functions can read them
// TODO Add tests checking that the async_lock, lock_owned, lock methods all block each other. For lock and lock_owned that can probably go into common tests.rs
// TODO Test `limit` parameter of all locking functions

use crate::guard::TryInsertError;
use crate::lockable_map_impl::LockableMapImpl;
use crate::map_like::ArcMutexMapLike;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Asserts that both vectors have the same entries but they may have a different order
pub(crate) fn assert_vec_eq_unordered<T: Ord + Eq + Debug>(mut lhs: Vec<T>, mut rhs: Vec<T>) {
    lhs.sort();
    rhs.sort();
    assert_eq!(lhs, rhs);
}

pub(crate) fn wait_for(mut func: impl FnMut() -> bool, timeout: Duration) {
    let start = Instant::now();
    while !func() {
        if start.elapsed() > timeout {
            panic!("Timeout waiting for condition");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub(crate) async fn wait_for_async(mut func: impl FnMut() -> bool, timeout: Duration) {
    let start = Instant::now();
    while !func() {
        if start.elapsed() > timeout {
            panic!("Timeout waiting for condition");
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
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
        use std::ops::Deref;
        use std::sync::Arc;
        use tokio::sync::{Mutex, OwnedMutexGuard};
        use std::thread::{self, JoinHandle};
        use std::time::{Duration, Instant};
        use futures::{stream::StreamExt};
        use $crate::{Lockable, InfallibleUnwrap, TryInsertError, tests::Guard, utils::MyStreamExt};

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
                let start = Instant::now();
                loop {
                    if let Some(guard) = map.try_lock(key.clone(), SyncLimit::no_limit()).infallible_unwrap() {
                        break guard;
                    }
                    if Instant::now() - start > Duration::from_secs(10) {
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
                let start = Instant::now();
                loop {
                    if let Some(guard) = map.try_lock_owned(key.clone(), SyncLimit::no_limit()).infallible_unwrap() {
                        break guard;
                    }
                    if Instant::now() - start > Duration::from_secs(10) {
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
                let start = Instant::now();
                loop {
                    if let Some(guard) = map.try_lock_async(key.clone(), AsyncLimit::no_limit()).await.infallible_unwrap() {
                        break guard;
                    }
                    if Instant::now() - start > Duration::from_secs(10) {
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
                let start = Instant::now();
                loop {
                    if let Some(guard) = map.try_lock_owned_async(key.clone(), AsyncLimit::no_limit()).await.infallible_unwrap() {
                        break guard;
                    }
                    if Instant::now() - start > Duration::from_secs(10) {
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

        struct LockingThread {
            join_handle: Option<JoinHandle<()>>,
            // A mutex that is locked only while the LockingThread hasn't acquired a lock yet
            acquire_barrier: Arc<Mutex<()>>,
            // A mutex that can be released to signal the LockingThread to release its lock
            release_barrier_guard: Option<OwnedMutexGuard<()>>,
        }

        impl LockingThread {
            // Launch a thread that
            // 1. locks the given key
            // 2. once it has the lock, increments a counter
            // 3. then waits until a barrier is released before it releases the lock
            pub fn launch_thread_sync_lock<S, L>(
                locking: &L,
                pool: &Arc<S>,
                key: isize,
            ) -> Self
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                L: SyncLocking<S, isize, String> + Sync + 'static,
            {
                Self::launch_thread_sync_lock_with_callback(locking, pool, key, |_| {})
            }

            pub fn launch_thread_sync_lock_with_callback<S, L>(
                locking: &L,
                pool: &Arc<S>,
                key: isize,
                callback: impl FnOnce(&mut L::Guard<'_>) + Send + 'static,
            ) -> Self
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                L: SyncLocking<S, isize, String> + Sync + 'static,
            {
                let locking = (*locking).clone();
                let pool = Arc::clone(pool);
                let acquire_barrier = Arc::new(Mutex::new(()));
                let acquire_barrier_guard = Arc::clone(&acquire_barrier).blocking_lock_owned();
                let release_barrier = Arc::new(Mutex::new(()));
                let release_barrier_guard = Some(Arc::clone(&release_barrier).blocking_lock_owned());
                let join_handle = Some(thread::spawn(move || {
                    let mut guard = locking.lock_waiting_is_ok(&pool, key);
                    callback(&mut guard);
                    drop(acquire_barrier_guard);
                    let _release_barrier = release_barrier.blocking_lock();
                }));
                Self {
                    join_handle,
                    acquire_barrier,
                    release_barrier_guard,
                }
            }

            pub async fn launch_thread_async_lock<S, L>(
                locking: &L,
                pool: &Arc<S>,
                key: isize,
            ) -> Self
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                L: AsyncLocking<S, isize, String> + Sync + 'static,
            {
                Self::launch_thread_async_lock_with_callback(locking, pool, key, |_| {}).await
            }

            pub async fn launch_thread_async_lock_with_callback<S, L>(
                locking: &L,
                pool: &Arc<S>,
                key: isize,
                callback: impl FnOnce(&mut L::Guard<'_>) + Send + 'static,
            ) -> Self
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                L: AsyncLocking<S, isize, String> + Sync + 'static,
            {
                let locking = (*locking).clone();
                let pool = Arc::clone(pool);
                let acquire_barrier = Arc::new(Mutex::new(()));
                let acquire_barrier_guard = Arc::clone(&acquire_barrier).lock_owned().await;
                let release_barrier = Arc::new(Mutex::new(()));
                let release_barrier_guard = Some(Arc::clone(&release_barrier).lock_owned().await);
                let join_handle = Some(thread::spawn(move || {
                    let runtime = tokio::runtime::Runtime::new().unwrap();
                    let mut guard = runtime.block_on(locking.lock_waiting_is_ok(&pool, key));
                    callback(&mut guard);
                    drop(acquire_barrier_guard);
                    let _release_barrier = release_barrier.blocking_lock();
                }));
                Self {
                    join_handle,
                    acquire_barrier,
                    release_barrier_guard,
                }
            }

            pub fn entered_lock_section(&self) -> bool {
                self.acquire_barrier.try_lock().is_ok()
            }

            pub fn wait_for_lock(&self) {
                $crate::tests::wait_for(|| self.entered_lock_section(), Duration::from_secs(1));
            }

            pub async fn wait_for_lock_async(&self) {
                $crate::tests::wait_for_async(|| self.entered_lock_section(), Duration::from_secs(1)).await;
            }

            pub fn release(&mut self) {
                if let Some(release_barrier_guard) = self.release_barrier_guard.take() {
                    drop(release_barrier_guard);
                }
            }

            pub fn release_and_wait(mut self) {
                self.release();
                self.join();
            }

            pub fn join(&mut self) {
                if let Some(join_handle) = self.join_handle.take() {
                    join_handle.join().unwrap();
                }
            }
        }

        impl Drop for LockingThread {
            fn drop(&mut self) {
                self.release();
                self.join();
            }
        }

        #[tokio::test]
        #[should_panic(
            expected = "Cannot block the current thread from within a runtime. This happens because a function attempted to block the current thread while the thread is being used to drive asynchronous tasks."
        )]
        async fn blocking_lock_from_async_context_with_sync_api() {
            let p = $lockable_type::<isize, String>::new();
            let _ = p.blocking_lock(3, SyncLimit::no_limit());
        }

        #[tokio::test]
        #[should_panic(
            expected = "Cannot block the current thread from within a runtime. This happens because a function attempted to block the current thread while the thread is being used to drive asynchronous tasks."
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
                            pool.borrow()
                                .lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()))
                                .collect()
                                .await;
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
                            pool.borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()))
                                .collect()
                                .await;
                        crate::tests::assert_vec_eq_unordered(vec![
                            (3, Some(String::from("Value 3"))),
                            (4, Some(String::from("Value 4"))),
                            (5, Some(String::from("Value 5"))),
                        ], guards);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod when_some_are_locked {
                use super::*;

                mod map_with_1_entry {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).insert(String::from("Value"));

                        let child = LockingThread::launch_thread_sync_lock(&locking, &pool, 4);
                        child.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that the stream doesn't produce any value while the entry is locked
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock the entry
                        child.release_and_wait();

                        // Check the stream now produces the entry
                        assert_eq!(Some((4, Some(String::from("Value")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).await.insert(String::from("Value"));

                        let child = LockingThread::launch_thread_async_lock(&locking, &pool, 4).await;
                        child.wait_for_lock_async().await;

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that the stream doesn't produce any value while the entry is locked
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock the entry
                        child.release_and_wait();

                        // Check the stream now produces the entry
                        assert_eq!(Some((4, Some(String::from("Value")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod map_with_two_entries_all_are_locked {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).insert(String::from("Value 4"));
                        locking.lock(&pool, 5).insert(String::from("Value 5"));

                        let child1 = LockingThread::launch_thread_sync_lock(&locking, &pool, 4);
                        let child2 = LockingThread::launch_thread_sync_lock(&locking, &pool, 5);

                        child1.wait_for_lock();
                        child1.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that the stream doesn't produce any value while the entry is locked
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock one entry
                        child1.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock the other entry
                        child2.release_and_wait();
                        assert_eq!(Some((5, Some(String::from("Value 5")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).await.insert(String::from("Value 4"));
                        locking.lock(&pool, 5).await.insert(String::from("Value 5"));

                        let child1 = LockingThread::launch_thread_async_lock(&locking, &pool, 4).await;
                        let child2 = LockingThread::launch_thread_async_lock(&locking, &pool, 5).await;

                        child1.wait_for_lock_async().await;
                        child1.wait_for_lock_async().await;

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that the stream doesn't produce any value while the entry is locked
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock one entry
                        child1.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock the other entry
                        child2.release_and_wait();
                        assert_eq!(Some((5, Some(String::from("Value 5")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod map_with_two_entries_some_are_locked {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 3).insert(String::from("Value 3"));
                        locking.lock(&pool, 4).insert(String::from("Value 4"));
                        locking.lock(&pool, 5).insert(String::from("Value 5"));
                        locking.lock(&pool, 6).insert(String::from("Value 6"));

                        let child1 = LockingThread::launch_thread_sync_lock(&locking, &pool, 4);
                        let child2 = LockingThread::launch_thread_sync_lock(&locking, &pool, 5);

                        child1.wait_for_lock();
                        child2.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that the stream produces the unlocked values while some other entries are locked
                        let values = vec![
                            futures::executor::block_on(guards_stream.next()),
                            futures::executor::block_on(guards_stream.next()),
                        ];
                        $crate::tests::assert_vec_eq_unordered(
                            vec![
                                Some((3, Some(String::from("Value 3")))),
                                Some((6, Some(String::from("Value 6")))),
                            ],
                            values,
                        );

                        // Check that even if we wait, it doesn't get any other values
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock one entry
                        child1.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock the other entry
                        child2.release_and_wait();
                        assert_eq!(Some((5, Some(String::from("Value 5")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 3).await.insert(String::from("Value 3"));
                        locking.lock(&pool, 4).await.insert(String::from("Value 4"));
                        locking.lock(&pool, 5).await.insert(String::from("Value 5"));
                        locking.lock(&pool, 6).await.insert(String::from("Value 6"));

                        let child1 = LockingThread::launch_thread_async_lock(&locking, &pool, 4).await;
                        let child2 = LockingThread::launch_thread_async_lock(&locking, &pool, 5).await;

                        child1.wait_for_lock();
                        child2.wait_for_lock();

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that the stream produces the unlocked values while some other entries are locked
                        let values = vec![
                            guards_stream.next().await,
                            guards_stream.next().await,
                        ];
                        $crate::tests::assert_vec_eq_unordered(
                            vec![
                                Some((3, Some(String::from("Value 3")))),
                                Some((6, Some(String::from("Value 6")))),
                            ],
                            values,
                        );

                        // Check that even if we wait, it doesn't get any other values
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock one entry
                        child1.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // Unlock the other entry
                        child2.release_and_wait();
                        assert_eq!(Some((5, Some(String::from("Value 5")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }

            mod locked_entry_types {
                use super::*;

                mod given_preexisting_when_not_deleted_while_locked_then_is_returned {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).insert(String::from("Value 4"));

                        let child = LockingThread::launch_thread_sync_lock(&locking, &pool, 4);

                        child.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that even if we wait, we don't get the value
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // But we get it after releasing the lock
                        child.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).await.insert(String::from("Value 4"));

                        let child = LockingThread::launch_thread_async_lock(&locking, &pool, 4).await;

                        child.wait_for_lock();

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that even if we wait, we don't get the value
                        tokio::time::sleep(Duration::from_millis(1000)).await;
                        assert_eq!(None, guards_stream.next_if_ready());

                        // But we get it after releasing the lock
                        child.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod given_preexisting_when_deleted_while_locked_then_is_not_returned {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).insert(String::from("Value 4"));

                        let child = LockingThread::launch_thread_sync_lock_with_callback(&locking, &pool, 4, |guard| {
                            guard.remove();
                        });

                        child.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        child.release_and_wait();

                        // Assert there are no entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());
                        locking.lock(&pool, 4).await.insert(String::from("Value 4"));

                        let child = LockingThread::launch_thread_async_lock_with_callback(&locking, &pool, 4, |guard| {
                            guard.remove();
                        }).await;

                        child.wait_for_lock();

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        child.release_and_wait();

                        // Assert there are no entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod given_not_preexisting_when_not_created_while_locked_then_is_not_returned {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());

                        let child = LockingThread::launch_thread_sync_lock(&locking, &pool, 4);

                        child.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        child.release_and_wait();

                        // Assert there are no entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());

                        let child = LockingThread::launch_thread_async_lock(&locking, &pool, 4).await;

                        child.wait_for_lock();

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        child.release_and_wait();

                        // Assert there are no entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }

                mod given_not_preexisting_when_created_while_locked_then_is_returned {
                    use super::*;

                    fn test_sync<S>(locking: impl SyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());

                        let child = LockingThread::launch_thread_sync_lock_with_callback(&locking, &pool, 4, |guard| {
                            guard.insert(String::from("Value 4"));
                        });

                        child.wait_for_lock();

                        let mut guards_stream =
                            futures::executor::block_on(pool.deref().borrow().lock_all_entries())
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that even if we wait, we don't get the value
                        thread::sleep(Duration::from_millis(1000));
                        assert_eq!(None, guards_stream.next_if_ready());

                        // But we get it after releasing the lock
                        child.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, futures::executor::block_on(guards_stream.next()));
                    }

                    async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
                    where
                        S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
                    {
                        let pool = Arc::new(locking.new());

                        let child = LockingThread::launch_thread_async_lock_with_callback(&locking, &pool, 4, |guard| {
                            guard.insert(String::from("Value 4"));
                        }).await;

                        child.wait_for_lock();

                        let mut guards_stream =
                            pool.deref().borrow().lock_all_entries()
                                .await
                                .map(|guard| (*guard.key(), guard.value().cloned()));

                        // Check that even if we wait, we don't get the value
                        tokio::time::sleep(Duration::from_millis(1000)).await;
                        assert_eq!(None, guards_stream.next_if_ready());

                        // But we get it after releasing the lock
                        child.release_and_wait();
                        assert_eq!(Some((4, Some(String::from("Value 4")))), guards_stream.next_if_ready());

                        // Assert there are no other entries
                        assert_eq!(None, guards_stream.next().await);
                    }

                    $crate::instantiate_lockable_tests!(@gen_tests, $lockable_type, test_sync, test_async);
                }
            }
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

                let child = LockingThread::launch_thread_sync_lock(&locking, &pool, 5);

                // Check that even if we wait, the child thread won't get the lock
                std::thread::sleep(Duration::from_millis(1000));
                assert!(!child.entered_lock_section());

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = locking.lock(&*pool, 4);
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.wait_for_lock();
                child.release_and_wait();

                assert_eq!(0, (*pool).borrow().num_entries_or_locked());
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            {
                let pool = Arc::new(locking.new());
                let guard = locking.lock(&*pool, 5).await;

                let child = LockingThread::launch_thread_async_lock(&locking, &pool, 5).await;

                // Check that even if we wait, the child thread won't get the lock
                tokio::time::sleep(Duration::from_millis(1000)).await;
                assert!(!child.entered_lock_section());

                // Check that we can still lock other locks while the child is waiting
                {
                    let _g = locking.lock(&*pool, 4).await;
                }

                // Now free the lock so the child can get it
                std::mem::drop(guard);

                // And check that the child got it
                child.wait_for_lock_async().await;
                child.release_and_wait();

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

                let mut child1 = LockingThread::launch_thread_sync_lock(&locking, &pool, 5);
                let mut child2 = LockingThread::launch_thread_sync_lock(&locking, &pool, 5);

                // Check that even if we wait, the child threads won't get the lock)
                thread::sleep(Duration::from_millis(1000));
                assert!(!child1.entered_lock_section());
                assert!(!child2.entered_lock_section());

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = locking.lock(&pool, 4);
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                $crate::tests::wait_for(|| {
                    child1.entered_lock_section() ^ child2.entered_lock_section()
                }, Duration::from_secs(1));

                // Allow the child to free the lock
                child1.release();
                child2.release();

                // Check that the other child got it
                child1.wait_for_lock();
                child2.wait_for_lock();

                child1.release_and_wait();
                child2.release_and_wait();

                assert_eq!(0, (*pool).borrow().num_entries_or_locked());
            }

            async fn test_async<S>(locking: impl AsyncLocking<S, isize, String> + Sync + 'static)
            where
                S: Borrow<$lockable_type<isize, String>> + Send + Sync + 'static,
            {
                let pool = Arc::new(locking.new());
                let guard = locking.lock(&pool, 5).await;

                let mut child1 = LockingThread::launch_thread_async_lock(&locking, &pool, 5).await;
                let mut child2 = LockingThread::launch_thread_async_lock(&locking, &pool, 5).await;

                // Check that even if we wait, the child thread won't get the lock
                thread::sleep(Duration::from_millis(1000));
                assert!(!child1.entered_lock_section());
                assert!(!child2.entered_lock_section());

                // Check that we can stil lock other locks while the children are waiting
                {
                    let _g = locking.lock(&pool, 4).await;
                }

                // Now free the lock so a child can get it
                std::mem::drop(guard);

                // Check that a child got it
                thread::sleep(Duration::from_millis(1000));
                $crate::tests::wait_for_async(|| {
                    child1.entered_lock_section() ^ child2.entered_lock_section()
                }, Duration::from_secs(1)).await;

                // Allow the child to free the lock
                child1.release();
                child2.release();

                // Check that the other child got it
                child1.wait_for_lock();
                child2.wait_for_lock();

                child1.release_and_wait();
                child2.release_and_wait();

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
