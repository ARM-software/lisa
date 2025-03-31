/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{ffi::CString, string::String, vec::Vec};
use core::{
    cell::UnsafeCell,
    ffi::{CStr, c_int, c_uchar},
};

use crate::{
    error::{Error, error},
    inlinec::{cfunc, incomplete_opaque_type, opaque_type},
};

pub trait FieldTy {
    const NAME: &'static str;

    fn field_desc(name: String) -> FieldDesc {
        FieldDesc {
            c_ty: Self::NAME.into(),
            name,
        }
    }

    fn to_u64(self) -> u64;
}

macro_rules! impl_field {
    ($ty:ty, $c_name:literal, $to_u64:expr) => {
        impl FieldTy for $ty {
            const NAME: &'static str = $c_name;

            #[inline]
            fn to_u64(self) -> u64 {
                $to_u64(self)
            }
        }
    };
}

impl_field!(u8, "u8", |x| x as u64);
impl_field!(i8, "s8", |x| x as u64);
impl_field!(u16, "u16", |x| x as u64);
impl_field!(i16, "s16", |x| x as u64);
impl_field!(u32, "u32", |x| x as u64);
impl_field!(i32, "s32", |x| x as u64);
impl_field!(u64, "u64", |x| x);
impl_field!(i64, "s64", |x| x as u64);
impl_field!(&CStr, "char[]", |x: &CStr| x.as_ptr() as usize as u64);

pub struct FieldDesc {
    name: String,
    c_ty: String,
}

struct Fields {}

opaque_type!(
    pub struct CEventFile,
    "struct trace_event_file",
    "linux/trace_events.h",
);

incomplete_opaque_type!(
    pub struct CSynthEvent,
    "struct synth_event",
    "linux/trace_events.h"
);

opaque_type!(
    struct CSynthFieldDesc,
    "struct synth_field_desc",
    "linux/trace_events.h",

    attr_accessors {name: *const c_uchar},
    attr_accessors {type as typ: *const c_uchar},
);

pub struct EventDesc {
    // SAFETY: We need to hold onto those as they have been passed to the synthetic event kernel
    // API.
    name: CString,
    c_field_descs: Vec<(CString, CString)>,
    pub __c_event_file: *const UnsafeCell<CEventFile>,
    pub __c_synth_event: *const UnsafeCell<CSynthEvent>,
}
unsafe impl Send for EventDesc {}

impl EventDesc {
    pub fn new(name: &str, field_descs: Vec<FieldDesc>) -> Result<EventDesc, Error> {
        #[cfunc]
        unsafe fn make_event<'a, 'b>(
            name: &'a CStr,
            fields: *mut CSynthFieldDesc,
            len: usize,
        ) -> Result<(), c_int> {
            r#"
            #include <linux/trace_events.h>
            #include <linux/module.h>
            "#;

            r#"
            return synth_event_create(name, fields, len, THIS_MODULE);
            "#
        }

        fn to_c_string(s: &str) -> CString {
            CString::new(s)
                .expect("Cannot convert Rust string to C string if it contains nul bytes")
        }

        let c_field_descs: Vec<_> = field_descs
            .iter()
            .map(|desc| (to_c_string(&desc.name), to_c_string(&desc.c_ty)))
            .collect();

        let name = to_c_string(name);
        {
            let mut array: Vec<CSynthFieldDesc> = c_field_descs
                .iter()
                .map(|(name, ty)| {
                    unsafe {
                        CSynthFieldDesc::new_stack(move |this| {
                            this.name_raw_mut().write(name.as_ptr() as *const c_uchar);
                            this.typ_raw_mut().write(ty.as_ptr() as *const c_uchar);
                            Ok::<(), core::convert::Infallible>(())
                        })
                    }
                    .unwrap()
                })
                .collect();

            // SAFETY: make_event() will fail if an event already exists with that name, so once we
            // have successfully created the event, we can rely on any by-name lookup in various
            // kernel APIs to always give back data related to what we allocated here.

            // SAFETY: we will hold onto all the CString we created here until we call
            // delete_event(), at which point the kernel will not need them anymore.
            unsafe { make_event(name.as_c_str(), array.as_mut_ptr(), array.len()) }.map_err(
                |code| error!("Synthetic ftrace event {name:?} registration failed: {code}"),
            )?
        }

        #[cfunc]
        unsafe fn synth_event_find(name: &CStr) -> *const UnsafeCell<CSynthEvent> {
            r#"
            #include <linux/trace_events.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_SYMBOL(synth_event_find)
                return synth_event_find(name);
            #else
                return NULL;
            #endif
            "#
        }

        let __c_synth_event = unsafe { synth_event_find(name.as_c_str()) };

        #[cfunc]
        unsafe fn trace_get_event_file<'a, 'b>(
            name: &'a CStr,
        ) -> Option<&'b UnsafeCell<CEventFile>> {
            r#"
            #include <linux/trace_events.h>
            "#;

            r#"
            return trace_get_event_file(NULL, "synthetic", name);
            "#
        }
        let __c_event_file = unsafe { trace_get_event_file(name.as_c_str()) }
            .ok_or_else(|| error!("Could not get trace_event_file for synthetic event {name:?}"))?;

        Ok(EventDesc {
            name,
            c_field_descs,
            __c_event_file,
            __c_synth_event,
        })
    }

    /// # Safety
    ///
    /// The passed `vals` must be consistent with the event format
    #[inline]
    pub unsafe fn __trace(&self, vals: &mut [u64]) {
        #[cfunc]
        unsafe fn trace_synth(
            event_file: *const UnsafeCell<CEventFile>,
            synth_event: *const UnsafeCell<CSynthEvent>,
            vals: *mut u64,
            n_vals: usize,
        ) -> Result<(), c_int> {
            r#"
            #include <linux/trace_events.h>
            #include "introspection.h"
            "#;

            r#"
            // As of 6.13, there is no way to emit the event in all instances where it was enabled, only in
            // the top-level buffer. This may get solved if the patches mentioned there are merged:
            // https://bugzilla.kernel.org/show_bug.cgi?id=219876
            #if HAS_SYMBOL(synth_event_trace2)
                static unsigned int var_ref_idx[] = {
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                };
                BUG_ON(!synth_event);
                BUG_ON(n_vals > ARRAY_SIZE(var_ref_idx));
                synth_event_trace2(synth_event, vals, var_ref_idx);
                return 0;
            #elif HAS_SYMBOL(synth_event_trace_array)
                BUG_ON(!event_file);
                return synth_event_trace_array(event_file, vals, n_vals);
            #elif !defined(CONFIG_SYNTH_EVENTS)
            #   error "CONFIG_SYNTH_EVENTS=y is necessary"
            #endif
            "#
        }
        unsafe {
            trace_synth(
                self.__c_event_file,
                self.__c_synth_event,
                vals.as_mut_ptr(),
                vals.len(),
            )
        }
        .expect("Could not emit synthetic ftrace event");
    }
}

impl Drop for EventDesc {
    fn drop(&mut self) {
        #[cfunc]
        unsafe fn trace_put_event_file<'a>(event_file: *const UnsafeCell<CEventFile>) {
            r#"
            #include <linux/trace_events.h>
            #include "utils.h"
            "#;

            r#"
            return trace_put_event_file(CONST_CAST(struct trace_event_file *, event_file));
            "#
        }

        #[cfunc]
        unsafe fn delete_event(name: &CStr) -> Result<(), c_int> {
            r#"
            #include <linux/trace_events.h>
            "#;

            r#"
            return synth_event_delete(name);
            "#
        }

        // SAFETY: CEventFile is valid until the event gets deleted, which hasn't happened yet.
        unsafe { trace_put_event_file(self.__c_event_file) }

        // This may fail if the kernel in use exhibits this problem:
        // https://bugzilla.kernel.org/show_bug.cgi?id=219875
        unsafe { delete_event(self.name.as_c_str()) }.expect("Could not delete synthetic event");
    }
}

macro_rules! new_event {
    ($name:ident, fields: {$($field_name:ident: $field_ty:ty),* $(,)?}) => {
        {
            $crate::runtime::traceevent::EventDesc::new(
                stringify!($name),
                ::alloc::vec![
                    $(
                        <$field_ty as $crate::runtime::traceevent::FieldTy>::field_desc(stringify!($field_name).into())
                    ),*
                ],
            ).map(|event_desc| move |$($field_name: $field_ty),*| {
                let mut vals = [
                    $(
                        <$field_ty as $crate::runtime::traceevent::FieldTy>::to_u64($field_name)
                    ),*
                ];
                // SAFETY: The event fields type and order is guaranteed to be consistent between
                // the defintion and the use of the array, as they are both created here in the
                // new_event!() macro in the same order.
                unsafe {
                    event_desc.__trace(&mut vals)
                }
            })
        }
    };
}
pub(crate) use new_event;
