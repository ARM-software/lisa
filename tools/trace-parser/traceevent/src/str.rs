use core::{
    borrow::Borrow,
    cmp::Ordering,
    convert::{AsMut, AsRef},
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

#[derive(Debug, Clone)]
pub struct Str<'a> {
    pub(crate) inner: InnerStr<'a>,
}

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

impl<'a> Clone for OwnedScratchBox<'a, dyn StringProducer> {
    #[inline]
    fn clone(&self) -> Self {
        self.clone_box(self.alloc)
    }
}

pub trait StringProducer: Send + Sync {
    fn write(&self, out: &mut dyn fmt::Write);
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

    #[inline]
    pub fn into_static(self) -> Str<'static> {
        Str {
            inner: InnerStr::Owned(match self.inner {
                InnerStr::Owned(s) => s,
                InnerStr::Borrowed(s) => (*s).into(),
                InnerStr::Arc(s) => (&*s).into(),
                InnerStr::Procedural(p) => p.into_owned(),
            }),
        }
    }
}

impl<'a> Deref for Str<'a> {
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

impl<'a> DerefMut for Str<'a> {
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

impl<'a> AsRef<str> for Str<'a> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.deref()
    }
}

impl<'a> AsMut<str> for Str<'a> {
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

impl<'a> PartialEq<Self> for Str<'a> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<'a> Eq for Str<'a> {}

impl<'a> PartialOrd for Str<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for Str<'a> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other.deref())
    }
}

impl<'a> Hash for Str<'a> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(self.deref(), state)
    }
}

impl<'a> Borrow<str> for Str<'a> {
    #[inline]
    fn borrow(&self) -> &str {
        self
    }
}

impl<'a> fmt::Display for Str<'a> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self.deref(), f)
    }
}
