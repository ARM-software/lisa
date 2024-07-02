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

//! Fast arena memory allocator

use core::{
    fmt::{Debug, Formatter},
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut, RangeBounds},
    ptr::NonNull,
};
use std::{io, sync::Arc};

use bumpalo::{boxed::Box as BumpaloBox, collections::Vec as BumpaloVec, Bump};
use thread_local::ThreadLocal;

/// Allocator used for quick processing that will not need persistent memory allocation.
pub struct ScratchAlloc {
    pub(crate) bump: ThreadLocal<Bump>,
}

impl ScratchAlloc {
    #[inline]
    pub fn new() -> Self {
        ScratchAlloc {
            bump: ThreadLocal::new(),
        }
    }

    #[inline]
    fn bump(&self) -> &Bump {
        self.bump.get_or(Bump::new)
    }

    #[inline]
    pub fn reset(&mut self) {
        for bump in self.bump.iter_mut() {
            bump.reset()
        }
    }
}

impl AsRef<ScratchAlloc> for ScratchAlloc {
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Default for ScratchAlloc {
    fn default() -> Self {
        Self::new()
    }
}

/// Equivalent of [Box] with extra flexibility in ownership, at the expense of being able to move
/// out of the box.
#[derive(Clone)]
pub enum ScratchBox<'a, T: 'a + ?Sized, A = &'a ScratchAlloc>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    /// Owned box, allocated inside a [ScratchAlloc]
    Owned(OwnedScratchBox<'a, T, A>),
    /// Owned but shared and cheap to clone.
    Arc(Arc<T>),
}

impl<'a, T: Debug + ?Sized, A> Debug for ScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.as_ref().fmt(f)
    }
}

impl<'a, T, A> ScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    /// Move `value` in the `alloc` [ScratchAlloc].
    ///
    /// This results into an [ScratchBox::Owned] variant.
    #[inline]
    pub fn new_in(value: T, alloc: A) -> ScratchBox<'a, T, A> {
        ScratchBox::Owned(OwnedScratchBox::new_in(value, alloc))
    }

    #[inline]
    pub fn into_static(self) -> ScratchBox<'static, T, A>
    where
        T: Clone,
    {
        match self {
            ScratchBox::Owned(owned) => ScratchBox::Arc(Arc::new(owned.into_inner())),
            ScratchBox::Arc(rc) => ScratchBox::Arc(rc),
        }
    }

    /// Get the inner value, which may imply a [Clone::clone] operation for some [ScratchBox]
    /// variants.
    #[inline]
    pub fn into_inner(self) -> T
    where
        T: Clone,
    {
        match self {
            ScratchBox::Owned(owned) => owned.into_inner(),
            ScratchBox::Arc(rc) => rc.deref().clone(),
        }
    }
}

impl<'a, T: ?Sized, A> Deref for ScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            ScratchBox::Owned(owned) => owned,
            ScratchBox::Arc(rc) => rc,
        }
    }
}

impl<'a, T: ?Sized, A> AsRef<T> for ScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<'a, T: ?Sized, A> From<OwnedScratchBox<'a, T, A>> for ScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    #[inline]
    fn from(x: OwnedScratchBox<'a, T, A>) -> Self {
        ScratchBox::Owned(x)
    }
}

impl<'a, T: PartialEq + ?Sized, A> PartialEq<Self> for ScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<'a, T: Eq + ?Sized, A> Eq for ScratchBox<'a, T, A> where A: 'a + AsRef<ScratchAlloc> {}

/// Owned value allocated inside a [ScratchAlloc]
pub struct OwnedScratchBox<'a, T: 'a + ?Sized, A = &'a ScratchAlloc> {
    // BumpaloBox<'_, T> is unfortunately invariant in T even though it should be covariant, like std::boxed::Box<T>:
    // https://github.com/fitzgen/bumpalo/issues/170
    // To work around that, we store the NonNull<> pointer that is designed for
    // exactly that use case, and we convert back to BumpaloBox when it's
    // required.
    pub(crate) inner: NonNull<T>,
    pub(crate) alloc: A,
    pub(crate) __phantom: PhantomData<&'a ()>,
}

impl<'a, T, A> OwnedScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    #[inline]
    pub fn new_in(value: T, alloc: A) -> OwnedScratchBox<'a, T, A> {
        OwnedScratchBox {
            inner: Self::alloc_nonnull(value, alloc.as_ref()),
            alloc,
            __phantom: PhantomData,
        }
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> OwnedScratchBox<'a, [T], A>
    where
        T: Default + Clone,
    {
        let mut vec = BumpaloVec::with_capacity_in(capacity, alloc.as_ref().bump());
        // Fill the vec with placeholder value
        vec.resize(capacity, Default::default());

        // Convert to a BumpaloBox first to ensure that casting back the NonNull
        // to a BumpaloBox is not UB.
        let bbox: BumpaloBox<[T]> = vec.into();

        OwnedScratchBox {
            inner: NonNull::from(BumpaloBox::leak(bbox)),
            alloc,
            __phantom: PhantomData,
        }
    }

    pub fn from_slice(data: &[T], alloc: A) -> OwnedScratchBox<'a, [T], A>
    where
        T: Default + Clone,
    {
        let mut new = Self::with_capacity_in(data.len(), alloc);
        new.deref_mut().clone_from_slice(data);
        new
    }

    #[inline]
    pub fn into_inner(self) -> T {
        BumpaloBox::into_inner(self.into_bumpalobox())
    }

    #[inline]
    fn alloc_nonnull(value: T, alloc: &ScratchAlloc) -> NonNull<T> {
        let bbox = BumpaloBox::new_in(value, alloc.bump());
        NonNull::from(BumpaloBox::leak(bbox))
    }
}

/// Create an [OwnedScratchBox] containing a `dyn Trait` value.
macro_rules! OwnedScratchBox_as_dyn {
    ($expr:expr, $trait:path) => {{
        fn make<T: $trait, A: ::core::clone::Clone>(
            expr: $crate::scratch::OwnedScratchBox<'_, T, A>,
        ) -> $crate::scratch::OwnedScratchBox<'_, dyn $trait, A> {
            let new = $crate::scratch::OwnedScratchBox {
                alloc: ::core::clone::Clone::clone(&expr.alloc),
                inner: ::core::ptr::NonNull::new(
                    // Use leak() so that we consume the box without freeing the
                    // value.
                    expr.inner.as_ptr() as *mut dyn $trait,
                )
                .unwrap(),
                __phantom: ::core::marker::PhantomData,
            };
            // Ensure the destructor of the underlying value is never called,
            // otherwise that would lead to a double free when the new box is
            // dropped.
            ::core::mem::forget(expr);
            new
        }
        make($expr)
    }};
}
pub(crate) use OwnedScratchBox_as_dyn;

/// Marker trait for types that don't have any Drop implementation, so not using
/// [core::mem::forget] on them will not lead to any nasty effect (memory leak, locks unreleased
/// etc.).
pub trait NoopDrop {}
macro_rules! nodrop_impl {
    ($($typ:ty),*) => {
        $(
            impl NoopDrop for $typ {}
        )*
    };
}
nodrop_impl!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, f32, f64, bool, str);
impl<T: NoopDrop> NoopDrop for [T] {}
impl<T: NoopDrop, const N: usize> NoopDrop for [T; N] {}
impl<T> NoopDrop for &T {}
impl<T> NoopDrop for &mut T {}

impl<'a, T: ?Sized, A> OwnedScratchBox<'a, T, A> {
    #[inline]
    fn into_bumpalobox(self) -> BumpaloBox<'a, T> {
        let this = ManuallyDrop::new(self);
        // SAFETY: We own the pointer, it is not aliased anywhere. Also, the
        // destructor will not run since we used ManuallyDrop
        unsafe { BumpaloBox::from_raw(this.inner.as_ptr()) }
    }

    #[inline]
    pub fn leak(self) -> &'a mut T
    where
        T: NoopDrop,
    {
        BumpaloBox::leak(self.into_bumpalobox())
    }
}

// SAFETY: If the allocator and the value are Send, then the boxed value is also
// Send.
unsafe impl<'a, T: Send + ?Sized, A> Send for OwnedScratchBox<'a, T, A>
where
    T: Send,
    A: Send,
{
}

// SAFETY: If the allocator and the value are Sync, then the boxed value is also
// Sync.
unsafe impl<'a, T, A> Sync for OwnedScratchBox<'a, T, A>
where
    T: Sync + ?Sized,
    A: Sync,
{
}

impl<'a, T: ?Sized, A> Drop for OwnedScratchBox<'a, T, A> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: We own the pointer, it is not aliased anywhere. Also, it was
        // created by leaking a BumpaloBox in the first place.
        let ptr = self.inner.as_ptr();
        unsafe {
            BumpaloBox::from_raw(ptr);
        }
    }
}

impl<'a, T: ?Sized, A> Deref for OwnedScratchBox<'a, T, A> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: the pointer is not aliased anywhere
        unsafe { self.inner.as_ref() }
    }
}

impl<'a, T: ?Sized, A> DerefMut for OwnedScratchBox<'a, T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: the pointer is not aliased anywhere
        unsafe { self.inner.as_mut() }
    }
}

impl<'a, T: ?Sized, A> AsRef<T> for OwnedScratchBox<'a, T, A> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<'a, T: ?Sized, A> AsMut<T> for OwnedScratchBox<'a, T, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<'a, T: Clone, A: Clone> Clone for OwnedScratchBox<'a, T, A>
where
    A: 'a + AsRef<ScratchAlloc>,
{
    #[inline]
    fn clone(&self) -> Self {
        let value = Clone::clone(self.deref());
        OwnedScratchBox {
            inner: Self::alloc_nonnull(value, self.alloc.as_ref()),
            alloc: Clone::clone(&self.alloc),
            __phantom: PhantomData,
        }
    }
}

impl<'a, T: Debug + ?Sized, A> Debug for OwnedScratchBox<'a, T, A> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.deref().fmt(f)
    }
}
impl<'a, T: PartialEq + ?Sized, A> PartialEq<Self> for OwnedScratchBox<'a, T, A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<'a, T: Eq + ?Sized, A> Eq for OwnedScratchBox<'a, T, A> {}

/// [ScratchVec] is to [Vec] what [ScratchBox] is to [Box]
pub struct ScratchVec<'a, T: 'a>(BumpaloVec<'a, T>);

impl<'a, T> ScratchVec<'a, T> {
    #[inline]
    pub fn new_in(alloc: &'a ScratchAlloc) -> Self {
        ScratchVec(BumpaloVec::new_in(alloc.bump()))
    }

    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: &'a ScratchAlloc) -> Self {
        ScratchVec(BumpaloVec::with_capacity_in(capacity, alloc.bump()))
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.0.push(value)
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
    }

    #[inline]
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.0.extend(iter)
    }

    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len)
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        self.0.resize(new_len, value)
    }
    #[inline]
    pub fn leak(self) -> &'a mut [T]
    where
        T: NoopDrop,
    {
        self.0.into_bump_slice_mut()
    }
}

impl<'a, T> ScratchVec<'a, T>
where
    T: Clone,
{
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.0.extend_from_slice(other)
    }
}

impl<'a, T> ScratchVec<'a, T>
where
    T: Copy,
{
    #[inline]
    pub fn copy_from_slice(&mut self, other: &[T]) {
        self.0.copy_from_slice(other)
    }

    #[inline]
    pub fn copy_within<R>(&mut self, src: R, dst: usize)
    where
        R: RangeBounds<usize>,
        T: Copy,
    {
        self.0.copy_within(src, dst)
    }
}

impl<'a, T> Deref for ScratchVec<'a, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, T> DerefMut for ScratchVec<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<'a, T> AsRef<[T]> for ScratchVec<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<'a, T> AsMut<[T]> for ScratchVec<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

impl<'a, T: Clone> Clone for ScratchVec<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        ScratchVec(self.0.clone())
    }
}

impl<'a, T: Debug> Debug for ScratchVec<'a, T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        self.0.fmt(f)
    }
}

impl<'a> io::Write for ScratchVec<'a, u8> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a, T> IntoIterator for ScratchVec<'a, T> {
    type Item = T;
    type IntoIter = IntoIterVec<'a, T>;

    // Required method
    fn into_iter(self) -> Self::IntoIter {
        IntoIterVec(self.0.into_iter())
    }
}

pub struct IntoIterVec<'a, T>(<BumpaloVec<'a, T> as IntoIterator>::IntoIter);

impl<'a, T> Iterator for IntoIterVec<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }
}

impl<'a, A> Extend<A> for ScratchVec<'a, A> {
    #[inline]
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = A>,
    {
        for x in iter {
            self.push(x)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_test() {
        // As long as this compiles, we know that OwnedScratchBox<'a, T> will be
        // covariant in 'a
        #[allow(dead_code)]
        fn subtype<'alloc_child, 'alloc_parent: 'alloc_child, 'child, 'parent: 'child>(
            x: OwnedScratchBox<'alloc_parent, &'parent i32>,
        ) -> OwnedScratchBox<'alloc_child, &'child i32> {
            x
        }
    }
}
