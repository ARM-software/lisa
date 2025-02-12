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

//! Custom string type that fits all use cases inside this crate.
//!
//! The main reason for a custom type are the various ownership models supported and using
//! [smartstring::alias::String] internally where possible.

use core::{
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};
use std::sync::Arc;

use smartstring::alias::String;

use crate::{
    memo::Memo,
    scratch::{OwnedScratchBox, OwnedScratchBox_as_dyn, ScratchAlloc},
};

/// String type with various ownership model available.
#[derive(Debug, Clone)]
pub struct Str<'a> {
    pub(crate) inner: InnerStr<'a>,
}

/// Alias for a [Memo]-ized string.
type StrProcedure<'a> = Memo<
    String,
    OwnedScratchBox<'a, dyn StringProducer>,
    fn(&OwnedScratchBox<dyn StringProducer>) -> String,
>;

#[derive(Debug, Clone)]
pub(crate) enum InnerStr<'a> {
    Borrowed(&'a str),
    Owned(String),
    Arc(Arc<str>),
    Procedural(StrProcedure<'a>),
}

impl Clone for OwnedScratchBox<'_, dyn StringProducer> {
    #[inline]
    fn clone(&self) -> Self {
        self.clone_box(self.alloc)
    }
}

/// Lazily produce a string by writing it to a [fmt::Write] object.
pub trait StringProducer: Send + Sync {
    /// Write the string to `out`
    fn write(&self, out: &mut dyn fmt::Write);
    /// Clone the producer.
    fn clone_box<'a>(&self, alloc: &'a ScratchAlloc) -> OwnedScratchBox<'a, dyn StringProducer>
    where
        Self: 'a;
}

// impl StringProducer for &dyn StringProducer {
// #[inline]
// fn write(&self, out: &mut dyn fmt::Write) {
// (*self).write(out)
// }

// #[inline]
// fn clone_box<'a>(&self, alloc: &'a ScratchAlloc) -> OwnedScratchBox<'a, dyn StringProducer>
// where
// Self: 'a,
// {
// (*self).clone_box(alloc)
// }
// }

impl<F> StringProducer for F
where
    F: Fn(&mut dyn fmt::Write) + Send + Sync + Clone,
{
    #[inline]
    fn write(&self, out: &mut dyn fmt::Write) {
        self(out)
    }

    fn clone_box<'a>(&self, alloc: &'a ScratchAlloc) -> OwnedScratchBox<'a, dyn StringProducer>
    where
        Self: 'a,
    {
        let sbox = OwnedScratchBox::new_in(self.clone(), alloc);
        OwnedScratchBox_as_dyn!(sbox, StringProducer)
    }
}

fn write_to_string(f: &OwnedScratchBox<dyn StringProducer>) -> String {
    let mut new = String::new();
    f.write(&mut new);
    new
}

impl<'a> Str<'a> {
    #[inline]
    pub fn new_procedural<T: StringProducer>(f: OwnedScratchBox<'a, T>) -> Self {
        let f = OwnedScratchBox_as_dyn!(f, StringProducer);
        Str {
            inner: InnerStr::Procedural(Memo::new(f, write_to_string)),
        }
    }
    #[inline]
    pub fn new_borrowed(s: &'a str) -> Self {
        Str {
            inner: InnerStr::Borrowed(s),
        }
    }

    #[inline]
    pub fn new_owned(s: String) -> Self {
        Str {
            inner: InnerStr::Owned(s),
        }
    }

    #[inline]
    pub fn new_arc(s: Arc<str>) -> Self {
        Str {
            inner: InnerStr::Arc(s),
        }
    }

    /// Create a [`Str<'static>`] and optimize the result for cheap cloning.
    #[inline]
    pub fn into_static(self) -> Str<'static> {
        let inner = match self.inner {
            InnerStr::Arc(s) => InnerStr::Arc(s),
            _ => {
                let s: &str = self.deref();
                // smartstring will keep strings smaller than 23 bytes directly in the value rather
                // than allocating on the heap. It's cheap to clone and will not create unnecessary
                // atomic writes memory traffic.
                if s.len() <= 23 {
                    InnerStr::Owned(s.into())
                } else {
                    InnerStr::Arc(Arc::from(s))
                }
            }
        };
        Str { inner }
    }
}

impl Deref for Str<'_> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match &self.inner {
            InnerStr::Borrowed(s) => s,
            InnerStr::Owned(s) => s,
            InnerStr::Arc(s) => s,
            InnerStr::Procedural(memo) => memo.deref(),
        }
    }
}

impl DerefMut for Str<'_> {
    #[inline]
    fn deref_mut<'b>(&'b mut self) -> &'b mut Self::Target {
        macro_rules! own {
            ($s:expr) => {{
                let s: &str = $s.deref();
                *self = Str::new_owned(s.into());
                match self.inner {
                    InnerStr::Owned(ref mut s) => s,
                    _ => unreachable!(),
                }
            }};
        }

        macro_rules! cast_lifetime {
            ($s:expr) => {{
                // SAFETY: This works around a borrow checker limitation, where
                // simply returning s.deref_mut() would borrow "self" for the
                // lifetime of the scope, preventing from modifying "self" in
                // the other unrelated match arms.
                unsafe { core::mem::transmute::<&mut str, &'b mut str>($s) }
            }};
        }

        match &mut self.inner {
            InnerStr::Owned(s) => {
                cast_lifetime!(s.deref_mut())
            }
            InnerStr::Borrowed(s) => own!(s),
            InnerStr::Arc(s) => own!(s),
            InnerStr::Procedural(memo) => cast_lifetime!(memo.deref_mut()),
        }
    }
}

impl AsRef<str> for Str<'_> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.deref()
    }
}

impl AsMut<str> for Str<'_> {
    #[inline]
    fn as_mut(&mut self) -> &mut str {
        self.deref_mut()
    }
}

impl<'a> From<&'a str> for Str<'a> {
    #[inline]
    fn from(s: &'a str) -> Str<'a> {
        Str {
            inner: InnerStr::Borrowed(s),
        }
    }
}

impl<'r, 'a: 'r> From<&'r Str<'a>> for &'r str {
    #[inline]
    fn from(s: &'r Str<'a>) -> &'r str {
        s
    }
}

impl<'a> From<&Str<'a>> for String {
    #[inline]
    fn from(s: &Str<'a>) -> String {
        match &s.inner {
            InnerStr::Owned(s) => s.clone(),
            InnerStr::Borrowed(s) => (*s).into(),
            InnerStr::Arc(s) => s.deref().into(),
            InnerStr::Procedural(memo) => memo.deref().clone(),
        }
    }
}

impl PartialEq<Self> for Str<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for Str<'_> {}

impl PartialOrd for Str<'_> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Str<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other.deref())
    }
}

impl Hash for Str<'_> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(self.deref(), state)
    }
}

impl Borrow<str> for Str<'_> {
    #[inline]
    fn borrow(&self) -> &str {
        self
    }
}

impl fmt::Display for Str<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self.deref(), f)
    }
}
