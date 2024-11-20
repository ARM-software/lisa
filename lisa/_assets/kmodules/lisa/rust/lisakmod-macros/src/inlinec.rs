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

pub use lisakmod_macros_proc::{c_constant, cexport, cfunc};

pub trait FfiType {
    // TODO: if and when Rust gains const trait methods, we can just define a const function to
    // build a type rather than providing a C macro body and C preprocessor machinery to build full
    // type names.
    const C_DECL: &'static str;
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

/// [*const T] newtype.
///
/// [FfiType], [FromFfi] and [IntoFfi] implementations on [ConstPtr<T>] are expected to be provided
/// by the user. These implementations will in turn be used by blanket implementations to provide
/// those traits for:
/// * [*const T]
/// * [&'a T]
#[fundamental]
pub struct ConstPtr<T: ?Sized>(*const T);

impl<T: ?Sized> From<ConstPtr<T>> for *const T {
    #[inline]
    fn from(x: ConstPtr<T>) -> *const T {
        x.0
    }
}

impl<T: ?Sized> FromFfi for ConstPtr<T>
where
    ConstPtr<T>: FfiType<FfiType = *const T>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        ConstPtr(x)
    }
}

impl<T: ?Sized> IntoFfi for ConstPtr<T>
where
    ConstPtr<T>: FfiType<FfiType = *const T>,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        self.0
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

impl<T: ?Sized> From<MutPtr<T>> for *mut T {
    #[inline]
    fn from(x: MutPtr<T>) -> *mut T {
        x.0
    }
}

impl<T> FromFfi for MutPtr<T>
where
    T: ?Sized,
    MutPtr<T>: FfiType<FfiType = *mut T>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        MutPtr(x)
    }
}

impl<T> IntoFfi for MutPtr<T>
where
    T: ?Sized,
    MutPtr<T>: FfiType<FfiType = *mut T>,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        self.0
    }
}

// The user-provided "root" implementation is for &T and &mut T. Everything is provided from there.
// This is because &T and &mut T are fundamental (as in #[fundamental]), meaning that even if the
// "reference of" generic type is not provided by the crate (since it's a primitive), the user is
// free to implement traits for it despite the regular orphan rule. *T, *mut T, NonNull<T> etc do
// not enjoy such treatment, making it impossible to provide implementation for foreign traits.
impl<T> FfiType for *const T
where
    T: ?Sized,
    ConstPtr<T>: FfiType,
{
    const C_DECL: &'static str = <ConstPtr<T> as FfiType>::C_DECL;
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

// These implementations allow arbitrary nesting of pointers, as they provide the definition for a
// pointer to pointer. Unfortunately, we cannot provide a valid generic C_DECL for that, so we
// currently cannot express this.

// impl<T> FfiType for ConstPtr<ConstPtr<T>>
// where
// T: ?Sized,
// ConstPtr<T>: FfiType,
// {
// const C_DECL: &'static str = <ConstPtr<T> as FfiType>::C_DECL;
// type FfiType = *const <ConstPtr<T> as FfiType>::FfiType;
// }

// impl<T> FfiType for ConstPtr<MutPtr<T>>
// where
// T: ?Sized,
// MutPtr<T>: FfiType,
// {
// const C_DECL: &'static str = <MutPtr<T> as FfiType>::C_DECL;
// type FfiType = *const <MutPtr<T> as FfiType>::FfiType;
// }

impl<T> FfiType for ConstPtr<*const T>
where
    T: ?Sized,
    ConstPtr<ConstPtr<T>>: FfiType,
{
    const C_DECL: &'static str = <ConstPtr<ConstPtr<T>> as FfiType>::C_DECL;
    type FfiType = <ConstPtr<ConstPtr<T>> as FfiType>::FfiType;
}

impl<T> FfiType for ConstPtr<*mut T>
where
    T: ?Sized,
    ConstPtr<MutPtr<T>>: FfiType,
{
    const C_DECL: &'static str = <ConstPtr<MutPtr<T>> as FfiType>::C_DECL;
    type FfiType = <ConstPtr<MutPtr<T>> as FfiType>::FfiType;
}

impl<T> FfiType for *mut T
where
    T: ?Sized,
    MutPtr<T>: FfiType,
{
    const C_DECL: &'static str = <MutPtr<T> as FfiType>::C_DECL;
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

impl<T> FfiType for MutPtr<*const T>
where
    T: ?Sized,
    MutPtr<ConstPtr<T>>: FfiType,
{
    const C_DECL: &'static str = <MutPtr<ConstPtr<T>> as FfiType>::C_DECL;
    type FfiType = <MutPtr<ConstPtr<T>> as FfiType>::FfiType;
}

impl<T> FfiType for MutPtr<*mut T>
where
    T: ?Sized,
    MutPtr<MutPtr<T>>: FfiType,
{
    const C_DECL: &'static str = <MutPtr<MutPtr<T>> as FfiType>::C_DECL;
    type FfiType = <MutPtr<MutPtr<T>> as FfiType>::FfiType;
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

impl<T> FfiType for &T
where
    T: ?Sized,
    *const T: FfiType + PtrToMaybeSized<T>,
{
    const C_DECL: &'static str = <*const T as FfiType>::C_DECL;
    type FfiType = <*const T as FfiType>::FfiType;
}

impl<T> FromFfi for &T
where
    T: ?Sized,
    *const T: FfiType + FromFfi,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr = <*const T as FromFfi>::from_ffi(x);
        assert!(PtrToMaybeSized::is_aligned(&ptr).unwrap_or(true));
        ptr.as_ref().expect("Unexpected NULL pointer")
    }
}

impl<T> IntoFfi for &T
where
    T: ?Sized,
    *const T: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        (self as *const T).into_ffi()
    }
}

impl<T> FfiType for &mut T
where
    T: ?Sized,
    *mut T: FfiType,
{
    const C_DECL: &'static str = <*mut T as FfiType>::C_DECL;
    type FfiType = <*mut T as FfiType>::FfiType;
}

impl<T> FromFfi for &mut T
where
    T: ?Sized,
    *mut T: FfiType + FromFfi + PtrToMaybeSized<T>,
{
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let ptr = <*mut T as FromFfi>::from_ffi(x);
        assert!(PtrToMaybeSized::is_aligned(&ptr).unwrap_or(true));
        ptr.as_mut().expect("Unexpected NULL pointer")
    }
}

impl<T> IntoFfi for &mut T
where
    T: ?Sized,
    *mut T: FfiType + IntoFfi,
{
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        (self as *mut T).into_ffi()
    }
}

impl<T> FfiType for ConstPtr<UnsafeCell<T>>
where
    T: ?Sized,
    *const T: FfiType,
    *mut T: FfiType,
{
    // Expose as a mutable pointer for C code, since the whole point of UnsafeCell<T> is to allow
    // mutation of T from a &UnsafeCell<T>.
    const C_DECL: &'static str = <*mut T as FfiType>::C_DECL;
    // Expose the pointer as *const for the FFI functions so that the signatures are compatible
    // with the other blanket implementations. This will effectively transmute the *const
    // UnsafeCell<T> into *mut T in the IntoFfi implementation at the FFI boundary.
    type FfiType = *const UnsafeCell<T>;
}

impl<T> FfiType for MutPtr<UnsafeCell<T>>
where
    T: ?Sized,
    *mut T: FfiType,
{
    const C_DECL: &'static str = <*mut T as FfiType>::C_DECL;
    type FfiType = *mut UnsafeCell<T>;
}

impl<T> FfiType for Pin<T>
where
    T: FfiType,
{
    const C_DECL: &'static str = <T as FfiType>::C_DECL;
    type FfiType = <T as FfiType>::FfiType;
}

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

impl<T> FfiType for ConstPtr<Pin<T>>
where
    *const T: FfiType,
{
    const C_DECL: &'static str = <*const T as FfiType>::C_DECL;
    type FfiType = *const Pin<T>;
}

impl<T> FfiType for MutPtr<Pin<T>>
where
    *mut T: FfiType,
{
    const C_DECL: &'static str = <*mut T as FfiType>::C_DECL;
    type FfiType = *mut Pin<T>;
}

#[macro_export]
macro_rules! __internal_ptr_cffi {
    ($pointee:ty, $c_pointee:literal) => {
        $crate::inlinec::__internal_ptr_cffi!(
            @impl, $pointee, $pointee, $c_pointee,
            "", ""
        );

        // TODO: These implementations are necessary since we cannot currently have a recursive
        // implementation for ConstPtr<ConstPtr<T>>, because we cannot express the resulting C_DECL
        // (lack of const function in traits). We therefore unroll 2 level of pointers, since we
        // don't really need more in practice.
        $crate::inlinec::__internal_ptr_cffi!(
            @impl,
            $crate::inlinec::ConstPtr<$pointee>,
            <$crate::inlinec::ConstPtr<$pointee> as $crate::inlinec::FfiType>::FfiType,
            $c_pointee,
            "CONST_TY_DECL(PTR_TY_DECL(", "))"
        );
        $crate::inlinec::__internal_ptr_cffi!(
            @impl,
            $crate::inlinec::MutPtr<$pointee>,
            <$crate::inlinec::MutPtr<$pointee> as $crate::inlinec::FfiType>::FfiType,
            $c_pointee,
            "PTR_TY_DECL(", ")"
        );
    };
    (@impl, $pointee:ty, $ffi_pointee:ty, $c_pointee:literal, $decl_pre:literal, $decl_post:literal) => {
        impl $crate::inlinec::FfiType for $crate::inlinec::ConstPtr<$pointee> {
            const C_DECL: &'static str = $crate::misc::concatcp!(
                "BUILTIN_TY_DECL(", $c_pointee, ", ", $decl_pre, "CONST_TY_DECL(PTR_TY_DECL(DECLARATOR))", $decl_post, ")"
            );
            type FfiType = *const $ffi_pointee;
        }

        impl $crate::inlinec::FfiType for $crate::inlinec::MutPtr<$pointee> {
            const C_DECL: &'static str = $crate::misc::concatcp!(
                "BUILTIN_TY_DECL(", $c_pointee, ", ", $decl_pre, "PTR_TY_DECL(DECLARATOR)", $decl_post, ")"
            );
            type FfiType = *mut $ffi_pointee;
        }
    }
}
pub use crate::__internal_ptr_cffi;

macro_rules! transparent_cffi {
    ($ty:ty, $c_name:literal) => {
        impl FfiType for $ty {
            const C_DECL: &'static str =
                $crate::misc::concatcp!("BUILTIN_TY_DECL(", $c_name, ", DECLARATOR)");
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

        __internal_ptr_cffi!($ty, $c_name);
    };
}

transparent_cffi!(u8, "uint8_t");
transparent_cffi!(u16, "uint16_t");
transparent_cffi!(u32, "uint32_t");
transparent_cffi!(u64, "uint64_t");
transparent_cffi!(usize, "size_t");

transparent_cffi!(i8, "int8_t");
transparent_cffi!(i16, "int16_t");
transparent_cffi!(i32, "int32_t");
transparent_cffi!(i64, "int64_t");
transparent_cffi!(isize, "ssize_t");

transparent_cffi!(bool, "_Bool");

// This is used for function returning void exclusively. We never pass a void parameter to a
// function.
impl FfiType for () {
    const C_DECL: &'static str = "BUILTIN_TY_DECL(void, DECLARATOR)";
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
    const C_DECL: &'static str = "BUILTIN_TY_DECL(void, DECLARATOR)";
    type FfiType = c_void;
}
// Only implement FromFfi/IntoFfi for pointers to c_void, never for c_void itself
__internal_ptr_cffi!(c_void, "void");

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
    const C_DECL: &'static str = <MutPtr<T> as FfiType>::C_DECL;
    type FfiType = <MutPtr<T> as FfiType>::FfiType;
}

impl<T> IntoFfi for Option<NonNull<T>>
where
    T: NullPtr,
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
    const C_DECL: &'static str = <Option<NonNull<T>> as FfiType>::C_DECL;
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
    const C_DECL: &'static str = <&'static c_char as FfiType>::C_DECL;
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
    const C_DECL: &'static str = <Option<&'a CStr> as FfiType>::C_DECL;
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
    const C_DECL: &'static str = <&'a CStr as FfiType>::C_DECL;
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
    const C_DECL: &'static str = <&'a CStr as FfiType>::C_DECL;
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

impl FfiType for ConstPtr<[u8]> {
    const C_DECL: &'static str = "BUILTIN_TY_DECL(struct slice_const_u8, DECLARATOR)";
    type FfiType = FfiSlice<*const u8>;
}

impl FfiType for MutPtr<[u8]> {
    const C_DECL: &'static str = "BUILTIN_TY_DECL(struct slice_u8, DECLARATOR)";
    type FfiType = FfiSlice<*mut u8>;
}

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
    const C_DECL: &'static str = "BUILTIN_TY_DECL(int, DECLARATOR)";
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
    const C_DECL: &'static str = "BUILTIN_TY_DECL(void, DECLARATOR)";
    type FfiType = ();
}

impl FromFfi for Result<(), Infallible> {
    #[inline]
    unsafe fn from_ffi(_: Self::FfiType) -> Self {
        Ok(())
    }
}

pub struct Align<const N: usize>(<Self as GetAligned>::Aligned)
where
    Self: GetAligned;

pub trait GetAligned {
    type Aligned;
}

macro_rules! make_getaligned {
    ($($align:tt),*) => {
        $(
            const _:() = {
                #[repr(align($align))]
                pub struct Aligned;
                impl GetAligned for Align<$align> {
                    type Aligned = Aligned;
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
    ($vis:vis struct $name:ident, $c_name:literal, $c_header:literal) => {
        // Model opaque types as recommended in the Rustonomicon:
        // https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
        #[repr(C)]
        $vis struct $name {
            _data: [
                u8;
                $crate::inlinec::c_constant!(
                    ("#include <", $c_header, ">"),
                    ("sizeof (", $c_name, ")"),
                    "1"
                )
            ],
            // Since we cannot make Opaque types aligned with a simple attribute
            // (#[repr(align(my_macro!()))] is rejected since my_macro!() is not an integer
            // literal), we add a zero-sized member that allows specifying the alignment as a
            // generic const parameter.
            _align: $crate::inlinec::Align<{
                $crate::inlinec::c_constant!(
                    ("#include <", $c_header, ">"),
                    ("_Alignof (", $c_name, ")"),
                    "1"
                )
            }>,
            _marker:
                ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)>,
        }

        // Double check that the we did not fumble the Rust struct layout somehow.
        const _:() = {
            const fn member_layout<A, B, F: FnOnce(&A) -> &B>(f: F) -> (usize, usize) {
                ::core::mem::forget(f);
                (
                    ::core::mem::size_of::<B>(),
                    ::core::mem::align_of::<B>(),
                )
            }
            let (size, _): (usize, usize) = member_layout(|x: &$name| &x._data);
            let (_, align): (usize, usize) = member_layout(|x: &$name| &x._align);
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

        $crate::inlinec::__internal_ptr_cffi!($name, $c_name);

        use $crate::inlinec::Opaque as _;
        impl $crate::inlinec::Opaque for $name {}

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
