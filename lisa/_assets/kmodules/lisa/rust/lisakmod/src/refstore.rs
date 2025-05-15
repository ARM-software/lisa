/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{
    boxed::Box,
    collections::{BTreeMap, btree_map::Entry},
};

pub type Key = usize;

#[derive(Debug, Clone)]
pub struct RefStore<T> {
    map: BTreeMap<Key, Box<T>>,
}

impl<T> Default for RefStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

// FIXME: remove module if it is used nowhere
impl<T> RefStore<T> {
    pub fn new() -> RefStore<T> {
        RefStore {
            map: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, x: T) -> (Key, &mut T) {
        let x = Box::new(x);
        let key = {
            let ptr: &T = x.as_ref();
            let ptr = ptr as *const T;
            ptr as usize
        };
        match self.map.entry(key) {
            Entry::Vacant(entry) => (key, entry.insert(x)),
            Entry::Occupied(entry) => unreachable!(),
        }
    }

    pub fn remove_key(&mut self, key: Key) -> Option<T> {
        self.map.remove(&key).map(|x| *x)
    }

    pub fn remove_ref(&mut self, x: &T) -> Option<T> {
        let key = {
            let ptr = x as *const T;
            ptr as usize
        };
        self.remove_key(key)
    }

    pub fn clear(&mut self) {
        self.map.clear()
    }
}
