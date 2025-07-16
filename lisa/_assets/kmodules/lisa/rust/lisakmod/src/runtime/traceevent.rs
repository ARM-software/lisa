/* SPDX-License-Identifier: GPL-2.0 */

use core::ffi::CStr;

use lisakmod_macros::inlinec::{FfiType, IntoFfi};

pub type AsPtrArg<T> = *mut <T as FfiType>::FfiType;

pub trait FieldTy
where
    Self: FfiType + IntoFfi,
    // Since the tracepoint parameters have to fit in a u64 due to some kernel constraints, we pass
    // everything by reference.
    AsPtrArg<Self>: FfiType + IntoFfi,
{
    const NAME: &'static str;
}

#[derive(Clone, Copy, Debug)]
pub struct TracepointString {
    pub s: &'static CStr,
}
impl TracepointString {
    pub fn __private_new(s: &'static CStr) -> TracepointString {
        TracepointString { s }
    }
}

#[allow(unused_macros)]
macro_rules! new_tracepoint_string {
    ($s:expr) => {{
        const S: &'static str = $s;
        const SLICE: &'static [u8] = S.as_bytes();

        const BUF_LEN: usize = SLICE.len();
        // +1 for the NULL terminator
        static BUF: [u8; (BUF_LEN + 1)] = {
            let mut arr = [0u8; BUF_LEN + 1];
            let mut idx = 0;
            while idx < BUF_LEN {
                arr[idx] = SLICE[idx];
                idx += 1;
            }
            assert!(arr[S.len()] == 0);
            arr
        };

        // TODO: the tracepoint_string() C macro seems to be appropriate, but unfortunately does
        // not work in a module (despite the doc mentioning it should), so instead we abuse the
        // __trace_printk_fmt section directly.

        // Note that as of Linux v6.15 the ftrace infrastructure will copy the string the first
        // time the module is loaded and then modify the BUF_ADDR pointer itself to point at where
        // the string has been copied to. When the module is unloaded, the copy stays alive and
        // will be re-used the next time the module is loaded, by modifying BUF_ADDR again. It is
        // therefore critical to always load the string address from BUF_ADDR itself, and not let
        // the compiler optimize-away that load.
        //
        // Use a "static mut" so that the __trace_printk_fmt section is configured for read/write
        // by the linker, otherwise the kernel will try to modify it and crash as a result.
        #[unsafe(link_section = "__trace_printk_fmt")]
        static mut BUF_ADDR: $crate::mem::UnsafeSync<*const u8> =
            $crate::mem::UnsafeSync(BUF.as_ptr());

        let s2 = $crate::runtime::traceevent::TracepointString::__private_new(unsafe {
            ::core::ffi::CStr::from_ptr(::core::ptr::read_volatile(&raw mut BUF_ADDR).0 as *const _)
        });
        s2
    }};
}

#[allow(unused_imports)]
pub(crate) use new_tracepoint_string;

impl FfiType for TracepointString {
    const C_TYPE: &'static str = <&'static CStr as FfiType>::C_TYPE;
    const C_HEADER: Option<&'static str> = <&'static CStr as FfiType>::C_HEADER;
    type FfiType = <&'static CStr as FfiType>::FfiType;
}

impl IntoFfi for TracepointString {
    #[inline]
    fn into_ffi(self) -> Self::FfiType {
        self.s.into_ffi()
    }
}

macro_rules! impl_field {
    ($ty:ty, $c_name:literal) => {
        impl FieldTy for $ty {
            const NAME: &'static str = $c_name;
        }
    };
}

// New implementations of FieldTy should get a matching implementations in the process_rust.py
// script.
impl_field!(u8, "u8");
impl_field!(i8, "s8");
impl_field!(u16, "u16");
impl_field!(i16, "s16");
impl_field!(u32, "u32");
impl_field!(i32, "s32");
impl_field!(u64, "u64");
impl_field!(i64, "s64");
impl_field!(&CStr, "c-string");
impl_field!(&str, "rust-string");
impl_field!(TracepointString, "c-static-string");

macro_rules! new_event {
    ($name:ident, fields: {$($field_name:ident: $field_ty:ty),* $(,)?}) => {{
        ::lisakmod_macros::misc::json_metadata!({
            "type": "define-ftrace-event",
            "name": (::core::stringify!($name)),
            "fields": [
                $(
                    {
                        "name": (::core::stringify!($field_name)),
                        "logical-type": (<$field_ty as $crate::runtime::traceevent::FieldTy>::NAME),
                        "c-field-type": (<$field_ty as ::lisakmod_macros::inlinec::FfiType>::C_TYPE),
                        // All values are passed by pointer rather than being passed directly, as
                        // tracepoints parameters must not be larger than 64 bits. Since some
                        // parameter are (e.g. &str), we simply pass everything by reference.
                        "c-arg-type": (<$crate::runtime::traceevent::AsPtrArg::<$field_ty> as ::lisakmod_macros::inlinec::FfiType>::C_TYPE),
                        "c-arg-header": (
                            match <$crate::runtime::traceevent::AsPtrArg::<$field_ty> as ::lisakmod_macros::inlinec::FfiType>::C_HEADER {
                                Some(header) => header,
                                None => ""
                            }
                        )
                    }
                ),*
            ]
        });

        {
            #[::lisakmod_macros::inlinec::cfunc]
            fn emit($($field_name: $crate::runtime::traceevent::AsPtrArg::<$field_ty>),*) {
                r#"
                #include "ftrace_events.h"
                "#;

                // Call the trace_foo() function created by the TRACE_EVENT(foo, ...) kernel macro
                ::core::concat!(
                    "trace_", ::core::stringify!($name), "(",
                        $crate::misc::join!(
                            ", ",
                            $(::core::stringify!($field_name)),*
                        ),
                    ");"
                );
            }

            // Wrap the function in a closure so that we can later on switch to a dynamic event
            // creation facility if it becomes available. This is achieved by:
            // 1. Returning a closure rather than the fn item, so client code is used to dealing
            //    with a closure.
            // 2. Make the closure !Copy, so that client code does not accidentally relies on that.
            struct NonCopy;
            let noncopy = NonCopy;
            Ok::<_, $crate::error::Error>(
                move |$($field_name: $field_ty),*| {
                    let _ = &noncopy;
                    emit($(&mut ::lisakmod_macros::inlinec::IntoFfi::into_ffi($field_name)),*)
                }
            )
        }
    }};
}
#[allow(unused_imports)]
pub(crate) use new_event;
