use core::{fmt::Debug, ops::Deref};

#[derive(Clone, Debug)]
pub(crate) struct NestedPointer<Outer> {
    outer: Outer,
}
impl<Outer> NestedPointer<Outer> {
    #[inline]
    pub fn new(outer: Outer) -> Self {
        NestedPointer { outer }
    }

    // #[inline]
    // pub fn into_outer(self) -> Outer {
    // self.outer
    // }
}

impl<Outer> Deref for NestedPointer<Outer>
where
    Outer: Deref,
    Outer::Target: Deref,
{
    type Target = <Outer::Target as Deref>::Target;

    fn deref(&self) -> &Self::Target {
        &self.outer
    }
}

impl<T, Outer, Inner> AsRef<T> for NestedPointer<Outer>
where
    // We can't use AsRef here because otherwise the compiler might be left with
    // multiple choices of N. With Deref, only one choice is possible since
    // Target is an associated type.
    Outer: Deref<Target = Inner>,
    Inner: Deref<Target = T> + ?Sized,
    T: ?Sized,
{
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}
