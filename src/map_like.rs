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

type Entry<V> = Arc<Mutex<EntryValue<V>>>;

pub enum GetOrInsertNoneResult<V> {
    Existing(V),
    Inserted(V),
}

/// [ArcMutexMapLike] needs to be implemented for each kind of map we want to support, e.g.
/// for [LruCache] and [HashMap]. This is the basis for that map becoming usable in a [LockableMapImpl]
/// instance.
pub trait ArcMutexMapLike<K, V>: IntoIterator<Item = (K, Entry<V>)>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type ItemIter<'a>: Iterator<Item = (&'a K, &'a Entry<V>)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    fn new() -> Self;

    fn len(&self) -> usize;

    fn get_or_insert_none(&mut self, key: &K) -> GetOrInsertNoneResult<Entry<V>>;

    fn get(&mut self, key: &K) -> Option<&Entry<V>>;

    fn remove(&mut self, key: &K) -> Option<Entry<V>>;

    fn iter(&self) -> Self::ItemIter<'_>;
}
