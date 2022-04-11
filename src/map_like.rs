use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct EntryValue<V> {
    // While unlocked, an entry is always Some. While locked, it can be temporarily None
    // since we enter None values into the map to lock keys that actually don't exist in the map.
    pub(super) value: Option<V>,
}

/// [ArcMutexMapLike] needs to be implemented for each kind of map we want to support, e.g.
/// for [LruCache] and [HashMap]. This is the basis for that map becoming usable in a [LockableMapImpl]
/// instance.
pub trait ArcMutexMapLike: IntoIterator<Item = (Self::K, Arc<Mutex<EntryValue<Self::V>>>)> {
    // TODO Can we remove the 'static bound from K and V?
    type K: Eq + PartialEq + Hash + Clone + Debug + 'static;
    type V: Debug + 'static;

    fn new() -> Self;

    fn len(&self) -> usize;

    fn get_or_insert_none(&mut self, key: &Self::K) -> &Arc<Mutex<EntryValue<Self::V>>>;

    fn get(&mut self, key: &Self::K) -> Option<&Arc<Mutex<EntryValue<Self::V>>>>;

    fn remove(&mut self, key: &Self::K) -> Option<Arc<Mutex<EntryValue<Self::V>>>>;

    // TODO No box dyn
    fn iter(&self) -> Box<dyn Iterator<Item = (&Self::K, &Arc<Mutex<EntryValue<Self::V>>>)> + '_>;
}
