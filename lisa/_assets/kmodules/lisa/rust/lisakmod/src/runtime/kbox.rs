/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    convert::{AsMut, AsRef},
    fmt,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    mem::forget,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr,
    ptr::NonNull,
};

use crate::runtime::alloc::{kfree, kmalloc};
use lisakmod_macros::inlinec::{FfiType, FromFfi, IntoFfi, MutPtr, Opaque};

pub struct KBox<T: ?Sized> {
    ptr: NonNull<T>,
}

impl<T: ?Sized> KBox<T> {
    #[inline]
    pub fn new(x: T) -> Self
    where
        T: Sized,
    {
        let mut new = Self::new_uninit();
        let maybe: &mut MaybeUninit<T> = &mut new;
        maybe.write(x);
        // SAFETY: The memory is now initialize.
        unsafe { new.assume_init() }
    }

    #[inline]
    pub fn new_uninit() -> KBox<MaybeUninit<T>>
    where
        T: Sized,
    {
        let layout = Layout::new::<T>();
        Self::from_layout(layout)
    }

    #[inline]
    fn from_layout(layout: Layout) -> KBox<MaybeUninit<T>>
    where
        T: Sized,
    {
        let ptr = kmalloc(layout);
        let ptr = ptr::NonNull::new(ptr as *mut MaybeUninit<T>).expect("Allocation failed");
        // SAFETY: MaybeUninit<T> does not mandate any bit pattern for its data, so it's safe to
        // construct it without any initialization (that's the whole point).
        KBox { ptr }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        KBox { ptr }
    }

    #[inline]
    fn into_ptr(mut self) -> *mut T {
        let ptr = self.as_mut_ptr();
        // SAFETY: Do not free memory here.
        forget(self);
        ptr
    }

    #[inline]
    pub fn into_pin(self) -> Pin<Self> {
        // It's not possible to move or replace the insides of a `Pin<KBox<T>>`
        // when `T: !Unpin`, so it's safe to pin it directly without any
        // additional requirements.
        //
        // We also satisfy the restriction stated in core::pin doc:
        // > Pin<Ptr> requires that implementations of Deref and DerefMut on Ptr return a pointer
        // > to the pinned data directly and do not move out of the self parameter during their
        // > implementation of DerefMut::deref_mut. It is unsound for unsafe code to wrap pointer
        // > types with such “malicious” implementations of Deref; see Pin<Ptr>::new_unchecked for
        // > details.
        // https://doc.rust-lang.org/std/pin/index.html#interaction-between-deref-and-pinptr
        //
        // https://doc.rust-lang.org/std/pin/struct.Pin.html#method.new_unchecked
        unsafe { Pin::new_unchecked(self) }
    }

    #[inline]
    pub fn pin(x: T) -> Pin<Self>
    where
        T: Sized,
    {
        KBox::into_pin(KBox::new(x))
    }
}

impl<T> KBox<MaybeUninit<T>> {
    #[inline]
    pub unsafe fn assume_init(self) -> KBox<T> {
        let ptr: *const MaybeUninit<T> = self.into_ptr();
        // SAFETY: It is sound to cast *const MaybeUninit<T> to *const T since MaybeUninit<T> is
        // #[repr(transparent)]
        KBox {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
        }
    }
}

impl<T: ?Sized> Drop for KBox<T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: KBox::ptr always points at an initialized and owned location.
        unsafe {
            self.ptr.drop_in_place();
            // Allocation was done with [kmalloc] so we can free with [kfree]
            kfree(self.ptr.as_ptr());
        }
    }
}

impl<T> From<T> for KBox<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

// SAFETY: KBox<T> owns the T, so it inherits T: Send
unsafe impl<T: ?Sized + Send> Send for KBox<T> {}
// SAFETY: KBox<T> owns the T, so it inherits T: Sync
unsafe impl<T: ?Sized + Sync> Sync for KBox<T> {}

impl<T: ?Sized> Deref for KBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: self.ptr is guaranteed to be valid and initialized
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for KBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: self.ptr is guaranteed to be valid and initialized
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: ?Sized> AsRef<T> for KBox<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> AsMut<T> for KBox<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T: ?Sized> Borrow<T> for KBox<T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> BorrowMut<T> for KBox<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T: Clone> Clone for KBox<T> {
    #[inline]
    fn clone(&self) -> Self {
        let x: T = <T as Clone>::clone(self.deref());
        Self::new(x)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for KBox<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: Default> Default for KBox<T> {
    #[inline]
    fn default() -> Self {
        KBox::new(<T as Default>::default())
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for KBox<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<T: ?Sized + PartialEq> PartialEq for KBox<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: ?Sized + Eq> Eq for KBox<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for KBox<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Ord> Ord for KBox<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Hash> Hash for KBox<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized> fmt::Pointer for KBox<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T: ?Sized> Unpin for KBox<T> {}

impl<T> FfiType for KBox<T>
where
    T: ?Sized,
    NonNull<T>: FfiType,
{
    const C_DECL: &'static str = <NonNull<T> as FfiType>::C_DECL;
    type FfiType = <NonNull<T> as FfiType>::FfiType;
}

impl<T> FromFfi for KBox<T>
where
    T: ?Sized,
    MutPtr<T>: FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr: NonNull<T> = unsafe { FromFfi::from_ffi(x) };
        // SAFETY: The calling code promises that ptr is valid and initialized.
        unsafe { Self::from_ptr(ptr) }
    }
}

impl<T> IntoFfi for KBox<T>
where
    T: ?Sized,
    MutPtr<T>: IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        let ptr: *mut T = self.into_ptr();
        ptr.into_ffi()
    }
}

pub trait OpaqueExt: Opaque {
    unsafe fn new_kbox<E, F>(init: F) -> Result<KBox<Self>, E>
    where
        F: FnOnce(*mut Self) -> Result<(), E>;
}

impl<T> OpaqueExt for T
where
    T: Opaque,
{
    unsafe fn new_kbox<E, F>(init: F) -> Result<KBox<Self>, E>
    where
        F: FnOnce(*mut Self) -> Result<(), E>,
    {
        let mut b: KBox<MaybeUninit<T>> = KBox::new_uninit();
        let ptr: *mut T = (*b).as_mut_ptr();
        // SAFETY: If init() succeeded, the value is now initialized.
        init(ptr).map(|_| unsafe { b.assume_init() })
    }
}
