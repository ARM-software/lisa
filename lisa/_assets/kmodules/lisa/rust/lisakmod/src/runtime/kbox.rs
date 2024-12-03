/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    alloc::{Layout, LayoutError},
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    convert::{AsMut, AsRef},
    fmt,
    hash::{Hash, Hasher},
    marker::Unsize,
    mem::MaybeUninit,
    ops::{CoerceUnsized, Deref, DerefMut},
    pin::Pin,
    ptr,
    ptr::NonNull,
};

use lisakmod_macros::inlinec::{FfiType, FromFfi, IntoFfi, MutPtr, Opaque};

use crate::{
    misc::destructure,
    runtime::alloc::{GFPFlags, KernelAlloc, KmallocAllocator},
};

#[derive(Debug)]
pub enum AllocError {
    KmallocFailed,
    LayoutError(LayoutError),
}

impl From<LayoutError> for AllocError {
    #[inline]
    fn from(err: LayoutError) -> AllocError {
        AllocError::LayoutError(err)
    }
}

pub struct KBox<T: ?Sized, KA: KernelAlloc> {
    ptr: NonNull<T>,
    alloc: KA,
}

pub type KernelKBox<T> = KBox<T, KmallocAllocator<{ GFPFlags::Kernel }>>;
pub type AtomicKBox<T> = KBox<T, KmallocAllocator<{ GFPFlags::Atomic }>>;

impl<T, KA> KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    #[inline]
    pub fn allocator(&self) -> &KA {
        &self.alloc
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
    fn into_ptr_alloc(self) -> (*mut T, KA) {
        let (alloc, ptr) = destructure!(self, alloc, ptr);

        // // SAFETY: Since we are returning the raw pointer, we must absolutely avoid self from being
        // // dropped. MaybeUninit does not drop the wrapped value so it is fine.
        // let this = MaybeUninit::new(self);
        // // SAFETY: we move the fields out of the value, which is ok since we will not make use of
        // // the value anymore and it will not be dropped.
        // let alloc = unsafe { core::ptr::read(&(*MaybeUninit::as_ptr(&this)).alloc) };
        (ptr.as_ptr(), alloc)
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

    /// # Safety
    ///
    /// The passed pointer must have been allocated by the KA allocator and passing it moves the
    /// data, so no-one else is allowed to keep a mutable reference to it, unless [`T`] is
    /// [`core::cell::UnsafeCell<...>`]
    #[inline]
    pub unsafe fn from_ptr_in(ptr: NonNull<T>, alloc: KA) -> Self {
        KBox { ptr, alloc }
    }
}

impl<T, KA> KBox<T, KA>
where
    KA: KernelAlloc,
{
    #[inline]
    pub fn new_in(x: T, alloc: KA) -> Self {
        Self::try_new_in(x, alloc).expect("Allocation failed")
    }

    #[inline]
    pub fn new_uninit_in(alloc: KA) -> KBox<MaybeUninit<T>, KA> {
        Self::try_new_uninit_in(alloc).expect("Allocation failed")
    }

    #[inline]
    pub fn pin_in(x: T, alloc: KA) -> Pin<Self> {
        KBox::into_pin(KBox::new_in(x, alloc))
    }

    #[inline]
    pub fn try_new_in(x: T, alloc: KA) -> Result<Self, AllocError> {
        let mut new = Self::try_new_uninit_in(alloc)?;
        let maybe: &mut MaybeUninit<T> = &mut new;
        maybe.write(x);
        // SAFETY: The memory is now initialize.
        Ok(unsafe { new.assume_init() })
    }

    #[inline]
    pub fn try_new_uninit_in(alloc: KA) -> Result<KBox<MaybeUninit<T>, KA>, AllocError> {
        let new: KBox<[MaybeUninit<T>; 1], _> = Self::try_new_uninit_array_in::<1>(alloc)?;
        let new = unsafe { new.transmute_inner::<MaybeUninit<T>>() };
        Ok(new)
    }

    #[inline]
    pub fn try_new_uninit_array_in<const N: usize>(
        alloc: KA,
    ) -> Result<KBox<[MaybeUninit<T>; N], KA>, AllocError> {
        let mut new = Self::try_new_uninit_slice_in(N, alloc)?;
        let arr: &mut [MaybeUninit<T>; N] = (&mut *new).try_into().unwrap();
        let ptr = NonNull::new(arr as *mut _).unwrap();
        let (alloc,) = destructure!(new, alloc);
        Ok(KBox { ptr, alloc })
    }

    #[inline]
    pub fn try_new_uninit_slice_in(
        len: usize,
        alloc: KA,
    ) -> Result<KBox<[MaybeUninit<T>], KA>, AllocError> {
        let layout = Layout::array::<T>(len)?;
        let ptr = alloc.alloc(layout) as *mut MaybeUninit<T>;
        let ptr: *mut [MaybeUninit<T>] = core::ptr::slice_from_raw_parts_mut(ptr, len);
        let ptr = ptr::NonNull::new(ptr);
        match ptr {
            // SAFETY: MaybeUninit<T> does not mandate any bit pattern for its data, so it's safe
            // to construct it without any initialization (that's the whole point).
            Some(ptr) => Ok(unsafe { KBox::from_ptr_in(ptr, alloc) }),
            None => Err(AllocError::KmallocFailed),
        }
    }

    #[inline]
    pub fn into_inner(this: Self) -> T {
        let (ptr, _) = this.into_ptr_alloc();
        // SAFETY: The pointer is still valid since:
        // 1. We only store valid pointers
        // 2. We used core::mem::forget() Drop does not run and the memory stays allocated.
        unsafe { ptr.read() }
    }

    /// # Safety
    /// It must be sound to transmute a [T] into a [U]
    unsafe fn transmute_inner<U>(self) -> KBox<U, KA>
    where
        // Ensure we don't transmute between a fat and a thin pointer.
        *mut T: Sized,
        *mut U: Sized,
    {
        let (ptr, alloc) = self.into_ptr_alloc();
        // SAFETY: The caller guarantees that we can reinterpret the bytes of T as bytes of U.
        let ptr: *mut U = unsafe { core::mem::transmute::<*mut T, *mut U>(ptr) };
        let ptr = NonNull::new(ptr).unwrap();
        KBox { ptr, alloc }
    }
}

impl<T, KA> KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc + Default,
{
    /// # Safety
    ///
    /// The passed pointer must have been allocated by [KA] and passing it moves the data, so
    /// no-one else is allowed to keep a mutable reference to it, unless [`T`] is
    /// [`core::cell::UnsafeCell<...>`]
    #[inline]
    pub unsafe fn from_ptr(ptr: NonNull<T>) -> Self {
        unsafe { Self::from_ptr_in(ptr, Default::default()) }
    }
}

impl<T, KA> KBox<T, KA>
where
    KA: KernelAlloc + Default,
{
    #[inline]
    pub fn new(x: T) -> Self {
        Self::new_in(x, Default::default())
    }

    #[inline]
    pub fn new_uninit() -> KBox<MaybeUninit<T>, KA> {
        Self::new_uninit_in(Default::default())
    }

    #[inline]
    pub fn pin(x: T) -> Pin<Self> {
        Self::pin_in(x, Default::default())
    }

    #[inline]
    pub fn try_new(x: T) -> Result<Self, AllocError> {
        Self::try_new_in(x, Default::default())
    }

    #[inline]
    pub fn try_new_uninit() -> Result<KBox<MaybeUninit<T>, KA>, AllocError> {
        Self::try_new_uninit_in(Default::default())
    }

    #[inline]
    pub fn try_new_uninit_array<const N: usize>()
    -> Result<KBox<[MaybeUninit<T>; N], KA>, AllocError> {
        Self::try_new_uninit_array_in(Default::default())
    }
}

impl<T, KA> KBox<MaybeUninit<T>, KA>
where
    KA: KernelAlloc,
{
    /// # Safety
    /// This function inherits all the safety expectations of [MaybeUninit::assume_init]
    #[inline]
    pub unsafe fn assume_init(self) -> KBox<T, KA> {
        unsafe { self.transmute_inner::<T>() }
    }
}

impl<T, KA, const N: usize> KBox<[MaybeUninit<T>; N], KA>
where
    KA: KernelAlloc,
{
    /// # Safety
    /// This function inherits all the safety expectations of [MaybeUninit::assume_init]
    #[inline]
    pub unsafe fn assume_init(self) -> KBox<[T; N], KA> {
        unsafe { self.transmute_inner::<[T; N]>() }
    }
}

impl<T, KA> KBox<[MaybeUninit<T>], KA>
where
    KA: KernelAlloc,
{
    /// # Safety
    /// This function inherits all the safety expectations of [MaybeUninit::assume_init]
    #[inline]
    pub unsafe fn assume_init(self) -> KBox<[T], KA> {
        let (ptr, alloc): (*mut [MaybeUninit<T>], _) = self.into_ptr_alloc();
        // SAFETY: It is sound to cast *const [MaybeUninit<T>] to *const [T] since MaybeUninit<T>
        // is #[repr(transparent)]
        unsafe { KBox::from_ptr_in(NonNull::new_unchecked(ptr as *mut [T]), alloc) }
    }
}

impl<T, KA> Drop for KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    #[inline]
    fn drop(&mut self) {
        let ptr = self.ptr;
        // SAFETY: KBox::ptr always points at an initialized and owned location.
        unsafe {
            ptr.drop_in_place();
            // SAFETY: Allocation was done with the same allocator we are freeing it with.
            self.alloc.dealloc(ptr.as_ptr() as *mut u8);
        }
    }
}

impl<T, KA> From<T> for KBox<T, KA>
where
    KA: KernelAlloc + Default,
{
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

// SAFETY: KBox<T> owns the T, so it inherits T: Send
unsafe impl<T: ?Sized + Send, KA: KernelAlloc + Send> Send for KBox<T, KA> {}
// SAFETY: KBox<T> owns the T, so it inherits T: Sync
unsafe impl<T: ?Sized + Sync, KA: KernelAlloc + Sync> Sync for KBox<T, KA> {}

impl<T: ?Sized + Unsize<U>, U: ?Sized, KA: KernelAlloc> CoerceUnsized<KBox<U, KA>> for KBox<T, KA> {}

impl<T, KA> Deref for KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: self.ptr is guaranteed to be valid and initialized
        unsafe { self.ptr.as_ref() }
    }
}

impl<T, KA> DerefMut for KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: self.ptr is guaranteed to be valid and initialized
        unsafe { self.ptr.as_mut() }
    }
}

impl<T, KA> AsRef<T> for KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<T, KA> AsMut<T> for KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T, KA> Borrow<T> for KBox<T, KA>
where
    T: ?Sized,
    KA: KernelAlloc,
{
    #[inline]
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T, KA> BorrowMut<T> for KBox<T, KA>
where
    KA: KernelAlloc,
{
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T: Clone, KA: KernelAlloc + Clone> Clone for KBox<T, KA> {
    #[inline]
    fn clone(&self) -> Self {
        let x: T = <T as Clone>::clone(self.deref());
        Self::new_in(x, self.alloc.clone())
    }
}

impl<T: ?Sized + fmt::Debug, KA: KernelAlloc> fmt::Debug for KBox<T, KA> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: Default, KA: KernelAlloc + Default> Default for KBox<T, KA> {
    #[inline]
    fn default() -> Self {
        KBox::new_in(Default::default(), Default::default())
    }
}

impl<T: ?Sized + fmt::Display, KA: KernelAlloc> fmt::Display for KBox<T, KA> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<T: ?Sized + PartialEq, KA: KernelAlloc> PartialEq for KBox<T, KA> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: ?Sized + Eq, KA: KernelAlloc> Eq for KBox<T, KA> {}

impl<T: ?Sized + PartialOrd, KA: KernelAlloc> PartialOrd for KBox<T, KA> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Ord, KA: KernelAlloc> Ord for KBox<T, KA> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Hash, KA: KernelAlloc> Hash for KBox<T, KA> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized, KA: KernelAlloc> fmt::Pointer for KBox<T, KA> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T: ?Sized, KA: KernelAlloc> Unpin for KBox<T, KA> {}

impl<T, KA> FfiType for KBox<T, KA>
where
    KA: KernelAlloc,
    T: ?Sized,
    NonNull<T>: FfiType,
{
    const C_TYPE: &'static str = <NonNull<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <NonNull<T> as FfiType>::C_HEADER;
    type FfiType = <NonNull<T> as FfiType>::FfiType;
}

impl<T, KA> FromFfi for KBox<T, KA>
where
    KA: KernelAlloc + Default,
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

impl<T, KA> IntoFfi for KBox<T, KA>
where
    KA: KernelAlloc,
    T: ?Sized,
    MutPtr<T>: IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        let (ptr, _alloc): (*mut T, _) = self.into_ptr_alloc();
        ptr.into_ffi()
    }
}

pub trait OpaqueExt: Opaque {
    /// # Safety
    /// The `init` function will be passed a [core::mem::MaybeUninit] value, which it needs to
    /// fully initialize before it successfully returns. If an [Result::Err] is returned, then not
    /// fully initializing the object is allowed.
    unsafe fn new_kbox_in<E, F, KA>(init: F, alloc: KA) -> Result<KBox<Self, KA>, E>
    where
        KA: KernelAlloc,
        Self: Sized,
        F: FnOnce(&mut MaybeUninit<Self>) -> Result<(), E>;

    /// # Safety
    /// The `init` function will be passed a [core::mem::MaybeUninit] value, which it needs to
    /// fully initialize before it successfully returns. If an [Result::Err] is returned, then not
    /// fully initializing the object is allowed.
    #[inline]
    unsafe fn new_kbox<E, F, KA>(init: F) -> Result<KBox<Self, KA>, E>
    where
        KA: KernelAlloc + Default,
        Self: Sized,
        F: FnOnce(&mut MaybeUninit<Self>) -> Result<(), E>,
    {
        let alloc: KA = Default::default();
        unsafe { OpaqueExt::new_kbox_in(init, alloc) }
    }
}

impl<T> OpaqueExt for T
where
    T: Opaque,
{
    unsafe fn new_kbox_in<E, F, KA>(init: F, alloc: KA) -> Result<KBox<Self, KA>, E>
    where
        KA: KernelAlloc,
        Self: Sized,
        F: FnOnce(&mut MaybeUninit<Self>) -> Result<(), E>,
    {
        let mut b: KBox<MaybeUninit<T>, KA> = KBox::new_uninit_in(alloc);
        // SAFETY: If init() succeeded, the value is now initialized.
        init(&mut b).map(|_| unsafe { b.assume_init() })
    }
}
