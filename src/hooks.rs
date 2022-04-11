/// This trait can be implemented to define hooks into [LockableMap] to
/// execute certain callbacks when certain operations happen.
pub trait Hooks<V> {
    /// This gets executed every time a value is unlocked.
    /// The `v` parameter is the value that is being unlocked.
    /// It is `None` if we locked and then unlocked a key that
    /// actually doesn't have an entry in the map.
    fn on_unlock(&self, v: Option<&mut V>);
}

pub struct NoopHooks;
impl<V> Hooks<V> for NoopHooks {
    fn on_unlock(&self, _v: Option<&mut V>) {
        // noop
    }
}
