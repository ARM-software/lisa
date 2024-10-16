/* SPDX-License-Identifier: Apache-2.0 */

use core::{
    ffi::{c_char, CStr},
    ptr::{null, null_mut, NonNull},
};

pub trait FfiType {
    // TODO: if and when Rust gains const trait methods, we can just define a const function to
    // build a type rather than providing a C macro body and C preprocessor machinery to build full
    // type names.
    const C_DECL: &'static str;
    type FfiType;
}

pub trait FromFfi: FfiType {
    unsafe fn from_ffi(x: Self::FfiType) -> Self;
}

pub trait IntoFfi: FfiType {
    fn into_ffi(self) -> Self::FfiType;
}

macro_rules! transparent_cff {
    ($ty:ty, $c_decl:expr) => {
        impl FfiType for $ty {
            const C_DECL: &'static str = $c_decl;
            type FfiType = $ty;
        }

        impl FromFfi for $ty {
            unsafe fn from_ffi(x: Self::FfiType) -> Self {
                x
            }
        }

        impl IntoFfi for $ty {
            fn into_ffi(self) -> Self::FfiType {
                self
            }
        }
    };
}

transparent_cff!(u8, "BUILTIN_TY_DECL(uint8_t, DECLARATOR)");
transparent_cff!(u16, "BUILTIN_TY_DECL(uint16_t, DECLARATOR)");
transparent_cff!(u32, "BUILTIN_TY_DECL(uint32_t, DECLARATOR)");
transparent_cff!(u64, "BUILTIN_TY_DECL(uint64_t, DECLARATOR)");

transparent_cff!(i8, "BUILTIN_TY_DECL(signed char, DECLARATOR)");
transparent_cff!(i16, "BUILTIN_TY_DECL(int16_t, DECLARATOR)");
transparent_cff!(i32, "BUILTIN_TY_DECL(int32_t, DECLARATOR)");
transparent_cff!(i64, "BUILTIN_TY_DECL(int64_t, DECLARATOR)");

transparent_cff!(usize, "BUILTIN_TY_DECL(size_t, DECLARATOR)");
transparent_cff!(isize, "BUILTIN_TY_DECL(ssize_t, DECLARATOR)");

transparent_cff!(bool, "BUILTIN_TY_DECL(_Bool, DECLARATOR)");
transparent_cff!((), "BUILTIN_TY_DECL(void, DECLARATOR)");

transparent_cff!(
    *const u8,
    "BUILTIN_TY_DECL(const uint8_t, PTR_TY_DECL(DECLARATOR))"
);

transparent_cff!(*mut u8, "BUILTIN_TY_DECL(uint8_t, PTR_TY_DECL(DECLARATOR))");
transparent_cff!(
    *const c_char,
    "BUILTIN_TY_DECL(const char, PTR_TY_DECL(DECLARATOR))"
);

impl FfiType for Option<NonNull<u8>> {
    const C_DECL: &'static str = "BUILTIN_TY_DECL(const uint8_t, PTR_TY_DECL(DECLARATOR))";
    type FfiType = *mut u8;
}

impl IntoFfi for Option<NonNull<u8>> {
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        match self {
            None => null_mut(),
            Some(p) => p.as_ptr(),
        }
    }
}

impl FromFfi for Option<NonNull<u8>> {
    #[inline]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        if x.is_null() {
            None
        } else {
            Some(NonNull::new(x).unwrap())
        }
    }
}

impl FfiType for NonNull<u8> {
    const C_DECL: &'static str = <Option<NonNull<u8>> as FfiType>::C_DECL;
    type FfiType = <Option<NonNull<u8>> as FfiType>::FfiType;
}

impl IntoFfi for NonNull<u8> {
    fn into_ffi(self) -> Self::FfiType {
        Some(self).into_ffi()
    }
}

impl FromFfi for NonNull<u8> {
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        let x = <Option<NonNull<u8>> as FromFfi>::from_ffi(x);
        x.expect("NULL pointer was passed to NonNull<T>")
    }
}

impl FfiType for Option<&CStr> {
    const C_DECL: &'static str = <*const c_char as FfiType>::C_DECL;
    type FfiType = <*const c_char as FfiType>::FfiType;
}

impl IntoFfi for Option<&CStr> {
    fn into_ffi(self) -> Self::FfiType {
        match self {
            None => null(),
            Some(s) => s.as_ptr(),
        }
    }
}

impl FromFfi for Option<&CStr> {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        if x.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(x) })
        }
    }
}

impl<'a> FfiType for &'a CStr {
    const C_DECL: &'static str = <Option<&'a CStr> as FfiType>::C_DECL;
    type FfiType = <Option<&'a CStr> as FfiType>::FfiType;
}

impl IntoFfi for &CStr {
    fn into_ffi(self) -> Self::FfiType {
        Some(self).into_ffi()
    }
}

impl FromFfi for &CStr {
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
    unsafe fn from_ffi(x: Self::FfiType) -> Self {
        Some(
            <Option<&CStr>>::from_ffi(x)?
                .to_str()
                .expect("Invalid UTF-8 content in C string"),
        )
    }
}
