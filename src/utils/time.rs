use tokio::time::Instant;

pub trait TimeProvider {
    fn now(&self) -> Instant;
}

#[derive(Debug, Clone, Copy)]
pub struct RealTime;
impl TimeProvider for RealTime {
    fn now(&self) -> Instant {
        Instant::now()
    }
}
impl Default for RealTime {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod mock_time {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::Duration;

    #[derive(Debug, Clone)]
    pub struct MockTime {
        now: Arc<Mutex<Instant>>,
    }
    impl MockTime {
        pub fn advance_time(&self, delta: Duration) {
            *self.now.lock().unwrap() += delta;
        }
    }
    impl Default for MockTime {
        fn default() -> Self {
            Self {
                now: Arc::new(Mutex::new(Instant::now())),
            }
        }
    }
    impl TimeProvider for MockTime {
        fn now(&self) -> Instant {
            *self.now.lock().unwrap()
        }
    }
}
#[cfg(test)]
pub use mock_time::MockTime;
