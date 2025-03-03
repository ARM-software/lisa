/* SPDX-License-Identifier: Apache-2.0 */

use alloc::{boxed::Box, sync::Arc};
use core::{
    alloc::Layout,
    cell::UnsafeCell,
    convert::Infallible,
    ffi::{c_char, c_int, c_void, CStr},
    mem::MaybeUninit,
    ops::Deref,
    pin::Pin,
    ptr::{null, null_mut, NonNull},
};

pub use lisakmod_macros_proc::{cconstant, cexport, cfunc, cstatic};

pub trait FfiType {
    // TODO: if and when Rust gains const trait methods, we can just define a const function to
    // build a type rather than providing a C macro body and C preprocessor machinery to build full
    // type names.
    const C_TYPE: &'static str;
    const C_HEADER: Option<&'static str>;
    type FfiType;
}

pub trait FromFfi: FfiType {
    /// # Safety
    /// Implementations must take as many precautions as possible not to trigger any undefined
    /// behavior when converting from the C representation to Rust. For example, a Rust reference
    /// can never be NULL, so this should be checked by the implementation and panic if necessary.
    /// Obviously, it's impossible to provide compiler-checked guarantees about C code that would
    /// provide the Rust level of safety, so this is on a best-effort basis.
    unsafe fn from_ffi(x: Self::FfiType) -> Self;
}

pub trait IntoFfi: FfiType {
    fn into_ffi(self) -> Self::FfiType;
}

pub trait IntoPtr<Ptr> {
    fn into_ptr(self) -> Ptr;
}

pub trait FromPtr<Ptr> {
    fn from_ptr(ptr: Ptr) -> Self;
}

macro_rules! impl_ptr {
    ($ptr:ty, $ptr2:ty, $ref:ty) => {
        impl<T: ?Sized> IntoPtr<$ptr> for $ptr {
            #[inline]
            fn into_ptr(self) -> $ptr {
                self
            }
        }

        impl<'a, T: ?Sized> IntoPtr<*const $ref> for *const $ptr {
            #[inline]
            fn into_ptr(self) -> *const $ref {
                self as _
            }
        }

        impl<'a, T: ?Sized> IntoPtr<*mut $ref> for *mut $ptr {
            #[inline]
            fn into_ptr(self) -> *mut $ref {
                self as _
            }
        }

        impl<T: ?Sized> FromPtr<$ptr> for $ptr {
            #[inline]
            fn from_ptr(ptr: $ptr) -> Self {
                ptr
            }
        }

        impl<'a, T: ?Sized> FromPtr<*const $ref> for *const $ptr
        {
            #[inline]
            fn from_ptr(ptr: *const $ref) -> Self {
                ptr as _
            }
        }

        impl<'a, T: ?Sized> FromPtr<*mut $ref> for *mut $ptr {
            #[inline]
            fn from_ptr(ptr: *mut $ref) -> Self {
                ptr as _
            }
        }

        impl_ptr!(@nested, ConstPtr<$ptr2>, ConstPtr<$ref>);
        impl_ptr!(@nested, MutPtr<$ptr2>, MutPtr<$ref>);

    };
    (@nested, $nested_ptr:ty, $nested_ref:ty) => {
        impl<'a, T> FfiType for $nested_ref
        where
            // We only implement for Sized types, so that unsized types can encode their metadata in custom
            // ways, like [T]
            T: Sized,
            $nested_ptr: FfiType,
        {
            const C_TYPE: &'static str = <$nested_ptr as FfiType>::C_TYPE;
            const C_HEADER: Option<&'static str> = <$nested_ptr as FfiType>::C_HEADER;
            type FfiType = <$nested_ptr as FfiType>::FfiType;
        }

    }
}

/// [*const T] newtype.
///
/// [FfiType], [FromFfi] and [IntoFfi] implementations on [ConstPtr<T>] are expected to be provided
/// by the user. These implementations will in turn be used by blanket implementations to provide
/// those traits for:
/// * [*const T]
/// * [&'a T]
#[fundamental]
pub struct ConstPtr<T: ?Sized>(*const T);
impl_ptr!(*const T, ConstPtr<T>, &'a T);

impl<T: ?Sized> From<ConstPtr<T>> for *const T {
    #[inline]
    fn from(x: ConstPtr<T>) -> *const T {
        x.0
    }
}

impl<T> FromFfi for ConstPtr<T>
where
    // Only implement for Sized so that dynamically sized types (DST) like [T] can have their own
    // representation that can encode that size
    T: Sized,
    ConstPtr<T>: FfiType<FfiType: IntoPtr<*const T>>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        ConstPtr(x.into_ptr())
    }
}

impl<T> IntoFfi for ConstPtr<T>
where
    // Only implement for Sized so that dynamically sized types (DST) like [T] can have their own
    // representation that can encode that size
    T: Sized,
    ConstPtr<T>: FfiType<FfiType: FromPtr<*const T>>,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        <Self::FfiType as FromPtr<_>>::from_ptr(self.0)
    }
}

/// [*mut T] newtype.
///
/// [FfiType], [FromFfi] and [IntoFfi] implementations on [MutPtr<T>] are expected to be provided
/// by the user. These implementations will in turn be used by blanket implementations to provide
/// those traits for:
/// * [*mut T]
/// * [&'a mut T]
/// * [NonNull<T>]
/// * [Option<NonNull<T>>]
#[fundamental]
pub struct MutPtr<T: ?Sized>(*mut T);
impl_ptr!(*mut T, MutPtr<T>, &'a mut T);

impl<T: ?Sized> From<MutPtr<T>> for *mut T {
    #[inline]
    fn from(x: MutPtr<T>) -> *mut T {
        x.0
    }
}

impl<T> FromFfi for MutPtr<T>
where
    // Only implement for Sized so that dynamically sized types (DST) like [T] can have their own
    // representation that can encode that size
    T: Sized,
    MutPtr<T>: FfiType<FfiType: IntoPtr<*mut T>>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        MutPtr(x.into_ptr())
    }
}

impl<T> IntoFfi for MutPtr<T>
where
    // Only implement for Sized so that dynamically sized types (DST) like [T] can have their own
    // representation that can encode that size
    T: Sized,
    MutPtr<T>: FfiType<FfiType: FromPtr<*mut T>>,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        <Self::FfiType as FromPtr<_>>::from_ptr(self.0)
    }
}

// Since it is not possible for the user to write implementations for *const T and *mut T directly,
// they provide impl for ConstPtr<T> and MutPtr<T> newtypes and we just have a blanket
// implementation for the real pointer type.

impl<T> FfiType for *const T
where
    T: ?Sized,
    ConstPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <ConstPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <ConstPtr<T> as FfiType>::C_HEADER;
    type FfiType = <ConstPtr<T> as FfiType>::FfiType;
}

impl<T> FromFfi for *const T
where
    T: ?Sized,
    ConstPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr: ConstPtr<T> = FromFfi::from_ffi(x);
        ptr.0
    }
}

impl<T> IntoFfi for *const T
where
    T: ?Sized,
    ConstPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        ConstPtr(self).into_ffi()
    }
}

impl<T> FfiType for ConstPtr<*const T>
where
    T: ?Sized,
    ConstPtr<ConstPtr<T>>: FfiType,
{
    const C_TYPE: &'static str = <ConstPtr<ConstPtr<T>> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <ConstPtr<ConstPtr<T>> as FfiType>::C_HEADER;
    type FfiType = <ConstPtr<ConstPtr<T>> as FfiType>::FfiType;
}

impl<T> FfiType for ConstPtr<*mut T>
where
    T: ?Sized,
    ConstPtr<MutPtr<T>>: FfiType,
{
    const C_TYPE: &'static str = <ConstPtr<MutPtr<T>> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <ConstPtr<MutPtr<T>> as FfiType>::C_HEADER;
    type FfiType = <ConstPtr<MutPtr<T>> as FfiType>::FfiType;
}

impl<T> FfiType for *mut T
where
    T: ?Sized,
    MutPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <MutPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<T> as FfiType>::C_HEADER;
    type FfiType = <MutPtr<T> as FfiType>::FfiType;
}

impl<T> FromFfi for *mut T
where
    T: ?Sized,
    MutPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr: MutPtr<T> = FromFfi::from_ffi(x);
        ptr.0
    }
}

impl<T> IntoFfi for *mut T
where
    T: ?Sized,
    MutPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        MutPtr(self).into_ffi()
    }
}

impl<T> FfiType for MutPtr<*const T>
where
    T: ?Sized,
    MutPtr<ConstPtr<T>>: FfiType,
{
    const C_TYPE: &'static str = <MutPtr<ConstPtr<T>> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<ConstPtr<T>> as FfiType>::C_HEADER;
    type FfiType = <MutPtr<ConstPtr<T>> as FfiType>::FfiType;
}

impl<T> FfiType for MutPtr<*mut T>
where
    T: ?Sized,
    MutPtr<MutPtr<T>>: FfiType,
{
    const C_TYPE: &'static str = <MutPtr<MutPtr<T>> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<MutPtr<T>> as FfiType>::C_HEADER;
    type FfiType = <MutPtr<MutPtr<T>> as FfiType>::FfiType;
}

trait PtrToMaybeSized<T: ?Sized> {
    #[inline]
    fn is_aligned(&self) -> Option<bool> {
        let ptr = self.as_ptr();
        // SAFETY: Unstable API, so the safety requirements might change, but overall the
        // expectation is that it won't bring more UB than using a pointer without checking whether
        // it was aligned or not.
        let layout = unsafe { Layout::for_value_raw(ptr) };
        let addr: usize = ptr as *const u8 as usize;
        Some((addr % layout.align()) == 0)
    }
    fn as_ptr(&self) -> *const T;
}

impl<T: ?Sized> PtrToMaybeSized<T> for *const T {
    #[inline]
    fn as_ptr(&self) -> *const T {
        *self
    }
}

impl<T: ?Sized> PtrToMaybeSized<T> for *mut T {
    #[inline]
    fn as_ptr(&self) -> *const T {
        *self
    }
}

impl<T> FfiType for Option<&T>
where
    T: ?Sized,
    *const T: PtrToMaybeSized<T>,
    ConstPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <ConstPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <ConstPtr<T> as FfiType>::C_HEADER;
    type FfiType = <ConstPtr<T> as FfiType>::FfiType;
}

impl<T> FfiType for &T
where
    T: ?Sized,
    *const T: PtrToMaybeSized<T>,
    ConstPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <ConstPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <ConstPtr<T> as FfiType>::C_HEADER;
    type FfiType = <ConstPtr<T> as FfiType>::FfiType;
}

impl<T> FromFfi for Option<&T>
where
    T: ?Sized,
    ConstPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr = <ConstPtr<T> as FromFfi>::from_ffi(x).0;
        assert!(PtrToMaybeSized::is_aligned(&ptr).unwrap_or(true));
        ptr.as_ref()
    }
}

impl<'a, T> FromFfi for &'a T
where
    T: ?Sized,
    ConstPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        <Option<&'a T> as FromFfi>::from_ffi(x).expect("Unexpected NULL pointer")
    }
}

impl<T> IntoFfi for Option<&T>
where
    // We need Sized here so that we can use core::ptr::null()
    T: Sized,
    ConstPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        match self {
            Some(x) => x.into_ffi(),
            None => null::<T>().into_ffi(),
        }
    }
}

impl<T> IntoFfi for &T
where
    T: ?Sized,
    ConstPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        (self as *const T).into_ffi()
    }
}

impl<T> FfiType for Option<&mut T>
where
    T: ?Sized,
    MutPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <MutPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<T> as FfiType>::C_HEADER;
    type FfiType = <MutPtr<T> as FfiType>::FfiType;
}

impl<'a, T> FfiType for &'a mut T
where
    T: ?Sized,
    MutPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <Option<&'a mut T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <Option<&'a mut T> as FfiType>::C_HEADER;
    type FfiType = <Option<&'a mut T> as FfiType>::FfiType;
}

impl<T> FromFfi for Option<&mut T>
where
    T: ?Sized,
    *mut T: PtrToMaybeSized<T>,
    MutPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr = <MutPtr<T> as FromFfi>::from_ffi(x).0;
        assert!(PtrToMaybeSized::is_aligned(&ptr).unwrap_or(true));
        ptr.as_mut()
    }
}

impl<'a, T> FromFfi for &'a mut T
where
    T: ?Sized,
    *mut T: PtrToMaybeSized<T>,
    MutPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        <Option<&'a mut T> as FromFfi>::from_ffi(x).expect("Unexpected NULL pointer")
    }
}

impl<T> IntoFfi for Option<&mut T>
where
    // We need Sized here so that we can use core::ptr::null_mut()
    T: Sized,
    MutPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        match self {
            Some(x) => x.into_ffi(),
            None => null_mut::<T>().into_ffi(),
        }
    }
}

impl<T> IntoFfi for &mut T
where
    T: ?Sized,
    MutPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        (self as *mut T).into_ffi()
    }
}

impl<T> FfiType for ConstPtr<UnsafeCell<T>>
where
    T: ?Sized,
    ConstPtr<T>: FfiType,
    MutPtr<T>: FfiType,
{
    // Expose as a mutable pointer for C code, since the whole point of UnsafeCell<T> is to allow
    // mutation of T from a &UnsafeCell<T>.
    const C_TYPE: &'static str = <MutPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<T> as FfiType>::C_HEADER;
    // Expose the pointer as *const for the FFI functions so that the signatures are compatible
    // with the other blanket implementations. This will effectively transmute the *const
    // UnsafeCell<T> into *mut T in the IntoFfi implementation at the FFI boundary.
    type FfiType = *const UnsafeCell<T>;
}

impl<T> FfiType for MutPtr<UnsafeCell<T>>
where
    T: ?Sized,
    MutPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <MutPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<T> as FfiType>::C_HEADER;
    type FfiType = *mut UnsafeCell<T>;
}

macro_rules! impl_transparent_wrapper {
    ($wrapper:ty) => {
        impl<T> FfiType for $wrapper
        where
            T: FfiType,
        {
            const C_TYPE: &'static str = <T as FfiType>::C_TYPE;
            const C_HEADER: Option<&'static str> = <T as FfiType>::C_HEADER;
            type FfiType = <T as FfiType>::FfiType;
        }

        impl<T> FfiType for ConstPtr<$wrapper>
        where
            ConstPtr<T>: FfiType,
        {
            const C_TYPE: &'static str = <ConstPtr<T> as FfiType>::C_TYPE;
            const C_HEADER: Option<&'static str> = <ConstPtr<T> as FfiType>::C_HEADER;
            type FfiType = *const $wrapper;
        }

        impl<T> FfiType for MutPtr<$wrapper>
        where
            MutPtr<T>: FfiType,
        {
            const C_TYPE: &'static str = <MutPtr<T> as FfiType>::C_TYPE;
            const C_HEADER: Option<&'static str> = <MutPtr<T> as FfiType>::C_HEADER;
            type FfiType = *mut $wrapper;
        }
    };
}

impl_transparent_wrapper!(Pin<T>);

impl<T> FromFfi for Pin<T>
where
    T: FromFfi + Deref,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let x: T = FromFfi::from_ffi(x);
        // SAFETY: We get the value transferred from C to Rust, so C should not preserve ownership
        // of the data.
        unsafe { Pin::new_unchecked(x) }
    }
}

impl<T> IntoFfi for Pin<T>
where
    T: IntoFfi + Deref,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        IntoFfi::into_ffi(
            // SAFETY: It is the responsibility of the C code writer to ensure the Pin invariants
            // are upheld once the data live in C land.
            unsafe { Pin::into_inner_unchecked(self) },
        )
    }
}

#[macro_export]
macro_rules! __impl_primitive_ptr {
    ($pointee:ty, $c_pointee:literal, $c_header:expr) => {
        $crate::inlinec::__impl_primitive_ptr!(
            @impl, $pointee, $pointee, $c_header, $c_pointee
        );

        // TODO: These implementations are necessary since we cannot currently have a recursive
        // implementation for ConstPtr<ConstPtr<T>>, because we cannot express the resulting C_TYPE
        // (lack of const function in traits). We therefore unroll 2 level of pointers, since we
        // don't really need more in practice.
        $crate::inlinec::__impl_primitive_ptr!(
            @impl,
            $crate::inlinec::ConstPtr<$pointee>,
            <$crate::inlinec::ConstPtr<$pointee> as $crate::inlinec::FfiType>::FfiType,
            $c_header,
            "const __typeof__(", $c_pointee, ") *"
        );
        $crate::inlinec::__impl_primitive_ptr!(
            @impl,
            $crate::inlinec::MutPtr<$pointee>,
            <$crate::inlinec::MutPtr<$pointee> as $crate::inlinec::FfiType>::FfiType,
            $c_header,
            "__typeof__(", $c_pointee, ") *"
        );
    };
    (@impl, $pointee:ty, $ffi_pointee:ty, $c_header:expr, $($c_pointee:literal),*) => {
        impl $crate::inlinec::FfiType for $crate::inlinec::ConstPtr<$pointee> {
            const C_TYPE: &'static str = $crate::misc::concatcp!(
                "const __typeof__(", $($c_pointee),*, ") *"
            );
            const C_HEADER: Option<&'static str> = $c_header;
            type FfiType = *const $ffi_pointee;
        }

        impl $crate::inlinec::FfiType for $crate::inlinec::MutPtr<$pointee> {
            const C_TYPE: &'static str = $crate::misc::concatcp!(
                "__typeof__(", $($c_pointee),*, ") *"
            );
            const C_HEADER: Option<&'static str> = $c_header;
            type FfiType = *mut $ffi_pointee;
        }
    }
}
pub use crate::__impl_primitive_ptr;

macro_rules! impl_primitive {
    ($ty:ty, $c_name:literal, $c_header:expr) => {
        impl FfiType for $ty {
            const C_TYPE: &'static str = $c_name;
            const C_HEADER: Option<&'static str> = $c_header;
            type FfiType = $ty;
        }

        impl FromFfi for $ty {
            #[inline]
            unsafe fn from_ffi(x: Self::FfiType) -> Self {
                x
            }
        }

        impl IntoFfi for $ty {
            #[inline]
            fn into_ffi(self) -> Self::FfiType {
                self
            }
        }

        __impl_primitive_ptr!($ty, $c_name, $c_header);
    };
}

impl_primitive!(u8, "uint8_t", Some("linux/types.h"));
impl_primitive!(u16, "uint16_t", Some("linux/types.h"));
impl_primitive!(u32, "uint32_t", Some("linux/types.h"));
impl_primitive!(u64, "uint64_t", Some("linux/types.h"));
impl_primitive!(usize, "size_t", Some("linux/types.h"));

impl_primitive!(i8, "int8_t", Some("linux/types.h"));
impl_primitive!(i16, "int16_t", Some("linux/types.h"));
impl_primitive!(i32, "int32_t", Some("linux/types.h"));
impl_primitive!(i64, "int64_t", Some("linux/types.h"));
impl_primitive!(isize, "ssize_t", Some("linux/types.h"));

impl_primitive!(bool, "_Bool", None);

// This is used for function returning void exclusively. We never pass a void parameter to a
// function.
impl FfiType for () {
    const C_TYPE: &'static str = "void";
    const C_HEADER: Option<&'static str> = None;
    type FfiType = ();
}

impl FromFfi for () {
    #[inline]
    unsafe fn from_ffi(_: Self::FfiType) -> Self {}
}

impl IntoFfi for () {
    #[inline]
    fn into_ffi(self) -> Self::FfiType {}
}

// This is used for C void pointers exclusively. The usage is distinct from a function returning
// void, which is covered by the unit type ().
impl FfiType for c_void {
    const C_TYPE: &'static str = "void";
    const C_HEADER: Option<&'static str> = None;
    type FfiType = c_void;
}
// Only implement FromFfi/IntoFfi for pointers to c_void, never for c_void itself
__impl_primitive_ptr!(c_void, "void", None);

pub trait NullPtr {
    fn null_mut() -> *mut Self;
    #[inline]
    fn null() -> *const Self {
        Self::null_mut()
    }
}

impl<T: Sized> NullPtr for T {
    #[inline]
    fn null_mut() -> *mut Self {
        null_mut()
    }
}

impl<T> NullPtr for [T] {
    #[inline]
    fn null_mut() -> *mut Self {
        core::ptr::slice_from_raw_parts_mut(null_mut(), 0)
    }
}

impl<T> FfiType for Option<NonNull<T>>
where
    T: ?Sized,
    MutPtr<T>: FfiType,
{
    const C_TYPE: &'static str = <MutPtr<T> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <MutPtr<T> as FfiType>::C_HEADER;
    type FfiType = <MutPtr<T> as FfiType>::FfiType;
}

impl<T> IntoFfi for Option<NonNull<T>>
where
    T: ?Sized + NullPtr,
    MutPtr<T>: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        MutPtr(match self {
            None => <T as NullPtr>::null_mut(),
            Some(p) => p.as_ptr(),
        })
        .into_ffi()
    }
}

impl<T> FromFfi for Option<NonNull<T>>
where
    T: ?Sized,
    MutPtr<T>: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr: MutPtr<T> = FromFfi::from_ffi(x);
        let ptr = ptr.0;
        if ptr.is_null() {
            None
        } else {
            Some(NonNull::new(ptr).unwrap())
        }
    }
}

impl<T> FfiType for NonNull<T>
where
    T: ?Sized,
    Option<NonNull<T>>: FfiType,
{
    const C_TYPE: &'static str = <Option<NonNull<T>> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <Option<NonNull<T>> as FfiType>::C_HEADER;
    type FfiType = <Option<NonNull<T>> as FfiType>::FfiType;
}

impl<T> IntoFfi for NonNull<T>
where
    T: ?Sized,
    Option<NonNull<T>>: IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        Some(self).into_ffi()
    }
}

impl<T> FromFfi for NonNull<T>
where
    T: ?Sized,
    Option<NonNull<T>>: FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let x = <Option<NonNull<T>> as FromFfi>::from_ffi(x);
        x.expect("NULL pointer was passed to NonNull<T>")
    }
}

impl FfiType for Option<&CStr> {
    const C_TYPE: &'static str = <&'static c_char as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <&'static c_char as FfiType>::C_HEADER;
    type FfiType = <&'static c_char as FfiType>::FfiType;
}

impl IntoFfi for Option<&CStr> {
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        match self {
            None => null(),
            Some(s) => s.as_ptr(),
        }
    }
}

impl FromFfi for Option<&CStr> {
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        if x.is_null() {
            None
        } else {
            Some(CStr::from_ptr(x))
        }
    }
}

impl<'a> FfiType for &'a CStr {
    const C_TYPE: &'static str = <Option<&'a CStr> as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <Option<&'a CStr> as FfiType>::C_HEADER;
    type FfiType = <Option<&'a CStr> as FfiType>::FfiType;
}

impl IntoFfi for &CStr {
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        Some(self).into_ffi()
    }
}

impl FromFfi for &CStr {
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        Option::<&CStr>::from_ffi(x)
            .expect("NULL pointer was returned as a &CStr, use Option<&CStr> to allow that.")
    }
}

impl<'a> FfiType for &'a str {
    const C_TYPE: &'static str = <&'a CStr as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <&'a CStr as FfiType>::C_HEADER;
    type FfiType = <&'a CStr as FfiType>::FfiType;
}

impl FromFfi for &str {
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        <&CStr>::from_ffi(x)
            .to_str()
            .expect("Invalid UTF-8 content in C string")
    }
}

// No IntoFfi instance for &str as we cannot provide NULL-terminated string out of an &str. Use
// &CStr for that.

impl<'a> FfiType for Option<&'a str> {
    const C_TYPE: &'static str = <&'a CStr as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <&'a CStr as FfiType>::C_HEADER;
    type FfiType = <&'a CStr as FfiType>::FfiType;
}

impl FromFfi for Option<&str> {
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        Some(
            <Option<&CStr>>::from_ffi(x)?
                .to_str()
                .expect("Invalid UTF-8 content in C string"),
        )
    }
}

trait Pointer {
    type Pointee;
}

impl<T> Pointer for *const T {
    type Pointee = T;
}

impl<T> Pointer for *mut T {
    type Pointee = T;
}

#[repr(C)]
#[allow(private_bounds)]
pub struct FfiSlice<Ptr>
where
    // SAFETY: Guarantees *const T has the same layout as a C pointer (i.e. it is a thin pointer,
    // not a fat pointer)
    Ptr: Pointer<Pointee: Sized>,
{
    // This layout guarantees minimum padding if natural alignment is followed since sizeof(*const
    // T) >= sizeof(usize). It also matches the layout of a Rust slice, so convertions to and from
    // it can be fully optimized out by the compiler.
    data: Ptr,
    len: usize,
}

macro_rules! impl_slice {
    ($ty:ty, $c_name_const:literal, $c_name_mut:literal, $c_header:expr) => {
        impl FfiType for ConstPtr<[$ty]> {
            const C_TYPE: &'static str = $c_name_const;
            const C_HEADER: Option<&'static str> = Some($c_header);
            type FfiType = FfiSlice<*const $ty>;
        }

        impl FfiType for MutPtr<[$ty]> {
            const C_TYPE: &'static str = $c_name_mut;
            const C_HEADER: Option<&'static str> = Some($c_header);
            type FfiType = FfiSlice<*mut $ty>;
        }
    };
}

impl_slice!(
    u8,
    "struct slice_const_u8",
    "struct slice_u8",
    "rust/lisakmod-macros/cffi.h"
);

impl<T> IntoFfi for ConstPtr<[T]>
where
    ConstPtr<[T]>: FfiType<FfiType = FfiSlice<*const T>>,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        let this = self.0;
        Self::FfiType {
            data: this as *const _,
            len: this.len(),
        }
    }
}

impl<T> FromFfi for ConstPtr<[T]>
where
    ConstPtr<[T]>: FfiType<FfiType = FfiSlice<*const T>>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        ConstPtr(core::ptr::slice_from_raw_parts(x.data, x.len))
    }
}

impl<T> IntoFfi for MutPtr<[T]>
where
    MutPtr<[T]>: FfiType<FfiType = FfiSlice<*mut T>>,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        let this = self.0;
        Self::FfiType {
            data: this as *mut _,
            len: this.len(),
        }
    }
}

impl<T> FromFfi for MutPtr<[T]>
where
    MutPtr<[T]>: FfiType<FfiType = FfiSlice<*mut T>>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        MutPtr(core::ptr::slice_from_raw_parts_mut(x.data, x.len))
    }
}

impl FfiType for Result<(), c_int> {
    const C_TYPE: &'static str = "int";
    const C_HEADER: Option<&'static str> = None;
    type FfiType = c_int;
}

impl IntoFfi for Result<(), c_int> {
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        match self {
            Ok(()) => 0,
            Err(x) => x,
        }
    }
}

impl FromFfi for Result<(), c_int> {
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        match x {
            0 => Ok(()),
            x => Err(x),
        }
    }
}

impl FfiType for Result<(), Infallible> {
    const C_TYPE: &'static str = "void";
    const C_HEADER: Option<&'static str> = None;
    type FfiType = ();
}

impl FromFfi for Result<(), Infallible> {
    #[inline]
    unsafe fn from_ffi(_: Self::FfiType) -> Self {
        Ok(())
    }
}

// Combine both alignment and size so that the resulting type can be used in a repr(C) parent
// struct. Otherwise, we would end up having to either:
// * Use a ZST for alignment, which cannot be repr(C) as of 25/11/2024
// * Use repr(transparent) instead of repr(C), but then we have more than one field with non-zero
//   size or non-1 alignment.
pub trait GetAlignedData<const SIZE: usize, const ALIGN: usize> {
    type AlignedData;
}

macro_rules! make_getaligned {
    ($($align:tt),*) => {
        $(
            const _:() = {
                #[repr(C)]
                #[repr(align($align))]
                pub struct AlignedData<const SIZE: usize>([u8; SIZE]);

                impl<const SIZE: usize> GetAlignedData<SIZE, $align> for () {
                    type AlignedData = AlignedData<SIZE>;
                }
            };
        )*
    }
}

make_getaligned!(
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288
);

pub trait Opaque {
    #[inline]
    unsafe fn try_new<F, E>(init: F) -> Result<Self, E>
    where
        Self: Sized,
        F: FnOnce(*mut Self) -> Result<(), E>,
    {
        let mut this = Self::new_uninit();
        init(this.as_mut_ptr()).map(|_| unsafe { this.assume_init() })
    }

    #[inline]
    fn new_uninit() -> MaybeUninit<Self>
    where
        Self: Sized,
    {
        MaybeUninit::uninit()
    }

    #[inline]
    unsafe fn new_stack<E, F>(init: F) -> Result<Self, E>
    where
        Self: Sized,
        F: FnOnce(*mut Self) -> Result<(), E>,
    {
        let mut new = Self::new_uninit();
        let ptr: *mut Self = new.as_mut_ptr();
        // SAFETY: If the function succeeded, the contract means we can assume self is initialized.
        init(ptr).map(|_| unsafe { new.assume_init() })
    }

    #[inline]
    unsafe fn new_arc<E, F>(init: F) -> Result<Arc<Self>, E>
    where
        Self: Sized,
        F: FnOnce(*mut Self) -> Result<(), E>,
    {
        let mut arc = Arc::new_uninit();
        let ptr: *mut Self = Arc::get_mut(&mut arc).unwrap().as_mut_ptr();
        // SAFETY: If the function succeeded, the contract means we can assume self is initialized.
        init(ptr).map(|_| unsafe { arc.assume_init() })
    }

    #[inline]
    unsafe fn new_box<E, F>(init: F) -> Result<Box<Self>, E>
    where
        Self: Sized,
        F: FnOnce(*mut Self) -> Result<(), E>,
    {
        let mut b = Box::new_uninit();
        let maybe: &mut MaybeUninit<Self> = &mut b;
        let ptr: *mut Self = maybe.as_mut_ptr();
        // SAFETY: If the function succeeded, the contract means we can assume self is initialized.
        init(ptr).map(|_| unsafe { b.assume_init() })
    }
}

#[macro_export]
macro_rules! __internal_opaque_type {
    ($vis:vis struct $name:ident, $c_name:literal, $c_header:expr $(, $($opt_name:ident {$($opt:tt)*}),* $(,)?)?) => {
        // Model opaque types as recommended in the Rustonomicon:
        // https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
        // On top of that recipe, we add:
        // * A way to get the correct alignment using C compile-time reflection.
        // * repr(transparent), so that the FFI-safe warning is satisfied in all use cases. This
        //   way, we guarantee that the struct has only one non-ZST member, and its ABI is that of
        //   this member. The member in question is an array of u8, which is FFI-safe.
        #[repr(transparent)]
        $vis struct $name {
            // Since we cannot make Opaque types aligned with a simple attribute
            // (#[repr(align(my_macro!()))] is rejected since my_macro!() is not an integer
            // literal), we add a zero-sized member that allows specifying the alignment as a
            // generic const parameter.
            _data: <
                () as $crate::inlinec::GetAlignedData<
                    {
                        match $crate::inlinec::cconstant!(
                            ("#include \"", $c_header, "\""),
                            ("sizeof (", $c_name, ")"),
                        ) {
                            Some(x) => x,
                            None => 1,
                        }
                    },
                    {
                        match $crate::inlinec::cconstant!(
                            ("#include \"", $c_header, "\""),
                            ("_Alignof (", $c_name, ")"),
                        ) {
                            Some(x) => x,
                            None => 1,
                        }
                    }
                >
            >::AlignedData,
            _marker: ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)>,
        }

        $($(
            $crate::inlinec::opaque_type!(@opt $opt_name, $name, $c_header, $($opt)*);
        )*)?

        // Double check that the we did not fumble the Rust struct layout somehow.
        const _:() = {
            const fn member_layout<A, B, F: FnOnce(&A) -> &B>(f: F) -> (usize, usize) {
                ::core::mem::forget(f);
                (
                    ::core::mem::size_of::<B>(),
                    ::core::mem::align_of::<B>(),
                )
            }
            let (size, align): (usize, usize) = member_layout(|x: &$name| &x._data);
            // Check alignment first as a wrong alignment will likely impact the overal size as
            // well due to different padding.
            assert!(
                ::core::mem::align_of::<$name>() == align,
                "Rust opaque type alignment differs from C type."
            );
            assert!(
                ::core::mem::size_of::<$name>() == size,
                "Rust opaque type size differs from C type."
            );
        };

        $crate::inlinec::__impl_primitive_ptr!($name, $c_name, Some($c_header));

        use $crate::inlinec::Opaque as _;
        impl $crate::inlinec::Opaque for $name {}
        impl $crate::inlinec::FfiType for $name {
            type FfiType = $name;
            const C_TYPE: &'static str = $c_name;
            const C_HEADER: Option<&'static str> = Some($c_header);
        }

        impl $crate::inlinec::FromFfi for $name {
            #[inline]
            unsafe fn from_ffi(x: Self::FfiType) -> Self {
                x
            }
        }
        impl $crate::inlinec::IntoFfi for $name {
            #[inline]
            fn into_ffi(self) -> Self::FfiType {
                self
            }
        }
    };
    (@opt attr_by_ref, $name:ident, $c_header:expr, fn $attr:ident(&self) -> &$attr_ty:ty) => {
        impl $name {
            #[inline]
            fn $attr(&self) -> &$attr_ty {
                #[$crate::inlinec::cfunc]
                fn get(this: &$name) -> &$attr_ty {
                    $crate::misc::concatcp!(
                        "#include \"", $c_header, "\"\n"
                    );

                    $crate::misc::concatcp!(
                        "return &this->", ::core::stringify!($attr), ";"
                    )
                }
                get(self)
            }
        }
    };
    (@opt attr_by_mut, $name:ident, $c_header:expr, fn $attr:ident(&mut self) -> &mut $attr_ty:ty) => {
        impl $name {
            #[inline]
            fn $attr(&mut self) -> &mut $attr_ty {
                #[$crate::inlinec::cfunc]
                fn get(this: &mut $name) -> &mut $attr_ty {
                    $crate::misc::concatcp!(
                        "#include \"", $c_header, "\"\n"
                    );

                    $crate::misc::concatcp!(
                        "return &this->", ::core::stringify!($attr), ";"
                    )
                }
                get(self)
            }
        }
    };
    (@opt attr_by_value, $name:ident, $c_header:expr, fn $attr:ident(&self) -> $attr_ty:ty) => {
        impl $name {
            #[inline]
            fn $attr(&self) -> $attr_ty {
                #[$crate::inlinec::cfunc]
                fn get(this: &$name) -> $attr_ty {
                    $crate::misc::concatcp!(
                        "#include \"", $c_header, "\"\n"
                    );

                    $crate::misc::concatcp!(
                        "return this->", ::core::stringify!($attr), ";"
                    )
                }
                get(self)
            }
        }
    };
}
// Since the macro is tagged with #[macro_export], it will be exposed in the crate namespace
// directly for public use. We then re-export it from here under its pretty name, so that it is
// effectively part of the pub API of the current module (and technically as part of the root
// namespace under its private name).
pub use crate::__internal_opaque_type as opaque_type;

#[macro_export]
macro_rules! __internal_c_static_assert {
    ($headers:literal, $expr:tt) => {{
        #[$crate::inlinec::cfunc]
        fn constant_assert() {
            $headers;
            $crate::misc::concatcp!(
                "_Static_assert(",
                "(",
                $expr,
                "),",
                stringify!("C static assert failed"),
                ");"
            )
        }
    }};
}
pub use crate::__internal_c_static_assert as c_static_assert;
