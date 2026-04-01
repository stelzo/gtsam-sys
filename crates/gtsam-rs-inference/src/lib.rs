#![forbid(unsafe_code)]

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Key(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Symbol {
    pub chr: char,
    pub index: u32,
}

impl Symbol {
    pub fn new(chr: char, index: u32) -> Self {
        Self { chr, index }
    }

    pub fn key(self) -> Key {
        let c = self.chr as u64;
        Key((c << 56) | self.index as u64)
    }

    pub fn from_key(key: Key) -> Self {
        let chr = char::from_u32((key.0 >> 56) as u32).unwrap_or('?');
        let index = (key.0 & 0xFFFF_FFFF) as u32;
        Self { chr, index }
    }
}

pub trait Factor {
    fn keys(&self) -> &[Key];
}

pub struct FactorGraph<F> {
    factors: Vec<F>,
}

impl<F> Default for FactorGraph<F> {
    fn default() -> Self {
        Self {
            factors: Vec::new(),
        }
    }
}

impl<F> FactorGraph<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor(&mut self, factor: F) {
        self.factors.push(factor);
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.factors.iter()
    }
}

#[derive(Default)]
pub struct Values<T> {
    map: BTreeMap<Key, T>,
}

impl<T> Values<T> {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, key: Key, value: T) -> Option<T> {
        self.map.insert(key, value)
    }

    pub fn get(&self, key: &Key) -> Option<&T> {
        self.map.get(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{Key, Symbol};

    #[test]
    fn symbol_roundtrip_key() {
        let s = Symbol::new('x', 42);
        let k = s.key();
        assert_eq!(k, Key(((b'x' as u64) << 56) | 42));
        assert_eq!(Symbol::from_key(k), s);
    }
}
