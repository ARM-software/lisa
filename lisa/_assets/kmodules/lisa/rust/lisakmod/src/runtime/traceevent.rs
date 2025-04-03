/* SPDX-License-Identifier: GPL-2.0 */

use core::ffi::CStr;

use lisakmod_macros::inlinec::IntoFfi;

pub trait FieldTy: IntoFfi {
    const NAME: &'static str;
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
impl_field!(&CStr, "string");

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
                        "c-arg-type": (<$field_ty as ::lisakmod_macros::inlinec::FfiType>::C_TYPE)
                    }
                ),*
            ]
        });

        {
            #[::lisakmod_macros::inlinec::cfunc]
            fn emit($($field_name: $field_ty),*) {
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
            let x = NonCopy;
            Ok::<_, $crate::error::Error>(
                move |$($field_name: $field_ty),*| {
                    let _ = &x;
                    emit($($field_name),*)
                }
            )
        }
    }};
}
#[allow(unused_imports)]
pub(crate) use new_event;
