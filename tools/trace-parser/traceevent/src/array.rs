// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2024, ARM Limited and contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use core::{
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Array<'a, T> {
    // Unlike Str, this lacks an Owned variant since it's typically not so
    // useful and would either require using heap allocation (slow) or bring in
    // an extra dependency like tinyvec.
    Borrowed(&'a [T]),
    Arc(Arc<[T]>),
}

impl<'a, T> Array<'a, T> {
    #[inline]
    pub fn into_static(self) -> Array<'static, T>
    where
        T: Clone,
    {
        match self {
            Array::Borrowed(slice) => Array::Arc(slice.into()),
            Array::Arc(arr) => Array::Arc(arr),
        }
    }
}

impl<'a, T> Deref for Array<'a, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Array::Borrowed(s) => s,
            Array::Arc(s) => s,
        }
    }
}

impl<'a, T> AsRef<[T]> for Array<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<'a, T> From<&'a [T]> for Array<'a, T> {
    #[inline]
    fn from(arr: &'a [T]) -> Array<'a, T> {
        Array::Borrowed(arr)
    }
}

impl<'r, 'a: 'r, T> From<&'r Array<'a, T>> for &'r [T] {
    #[inline]
    fn from(arr: &'r Array<'a, T>) -> &'r [T] {
        arr
    }
}

impl<'a, T: Clone> From<&Array<'a, T>> for Vec<T> {
    #[inline]
    fn from(arr: &Array<'a, T>) -> Vec<T> {
        arr.to_vec()
    }
}

impl<'a, T: PartialEq> PartialEq<Self> for Array<'a, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<'a, T: Eq> Eq for Array<'a, T> {}

impl<'a, T: PartialOrd> PartialOrd for Array<'a, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.deref().partial_cmp(other.deref())
    }
}

impl<'a, T: Ord> Ord for Array<'a, T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other.deref())
    }
}

impl<'a, T: Hash> Hash for Array<'a, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(self.deref(), state)
    }
}

impl<'a, T> Borrow<[T]> for Array<'a, T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<'a, T> fmt::Display for Array<'a, T>
where
    T: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "[")?;
        for (i, x) in self.iter().enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            fmt::Display::fmt(x, f)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}
