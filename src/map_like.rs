use std::hash::Hash;

pub enum GetOrInsertNoneResult<'a, V> {
    Existing(&'a V),
    Inserted(&'a V),
}

/// [MapLike] needs to be implemented for each kind of map we want to support, e.g.
/// for [LruCache] and [HashMap]. This is the basis for that map becoming usable in a [LockableMapImpl]
/// instance.
pub trait MapLike<K, V>: IntoIterator<Item = (K, V)>
where
    K: Eq + PartialEq + Hash + Clone,
{
    type ItemIter<'a>: Iterator<Item = (&'a K, &'a V)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    fn new() -> Self;

    fn len(&self) -> usize;

    fn get_or_insert_none<'s>(&'s mut self, key: &K) -> GetOrInsertNoneResult<'s, V>;

    fn get(&mut self, key: &K) -> Option<&V>;

    fn remove(&mut self, key: &K) -> Option<V>;

    fn iter(&self) -> Self::ItemIter<'_>;
}
