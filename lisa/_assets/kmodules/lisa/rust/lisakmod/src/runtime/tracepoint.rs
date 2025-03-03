/* SPDX-License-Identifier: GPL-2.0 */

use core::{cell::UnsafeCell, ffi::c_void};

use crate::inlinec::{cfunc, opaque_type};

opaque_type!(
    struct CTracepoint,
    "struct tracepoint",
    "linux/tracepoint.h",
);

pub struct Tracepoint<'tp> {
    // struct tracepoint API allows mutation but protects it with an internal mutex
    // (tracepoint_mutex static variable in tracepoint.c)
    c_tp: &'tp UnsafeCell<CTracepoint>,
}

impl Tracepoint<'static> {
    /// # Safety
    ///
    /// The tracepoint being looked up must be defined in the base kernel image, not in a module.
    /// Otherwise the lifetime of the resulting tracepoint is unknown.
    pub unsafe fn lookup(name: &'static str) -> Option<Self> {
        /// # Safety
        ///
        /// The lifetime of the tracepoint depends on where it was defined. If it was defined in the
        /// base kernel Image, then it's (probably) 'static, but if it was defined in a module, then
        /// it's as long as the owning module lives.
        ///
        /// ftrace does seem to provide a way to know who owns
        /// a given tracepoint though (to be checked).
        #[cfunc]
        unsafe fn find_tp<'tp>(name: &'static [u8]) -> Option<&'tp UnsafeCell<CTracepoint>> {
            r#"
            #include <linux/tracepoint.h>
            #include <linux/string.h>

            struct __find_tracepoint_params {
                const char *name;
                struct tracepoint *found;
            };

            static void __do_find_tracepoint(struct tracepoint *tp, void *__finder) {
                struct __find_tracepoint_params *finder = __finder;
                if (!strcmp(tp->name, finder->name))
                    finder->found = tp;
            }
            "#;

            r#"
            struct __find_tracepoint_params res = {.name = name.data, .found = NULL};
            for_each_kernel_tracepoint(__do_find_tracepoint, &res);
            return res.found;
            "#
        }

        // SAFETY: The user guarantees that lookup() will only be used for tracepoints defined in
        // the base kernel image, so with a 'static lifetime
        let c_tp = unsafe { find_tp(name.as_bytes())? };
        Some(Tracepoint { c_tp })
    }

    fn register_probe(&self, probe: *mut c_void) -> Result<(), ()> {
        #[cfunc]
        fn register(tp: *mut CTracepoint, probe: *mut c_void) {
            r#"
            #include <linux/tracepoint.h>
            #include <linux/string.h>
            "#;

            r#"
            tracepoint_probe_register(tp, probe, NULL); \
            "#
        }
        register(self.c_tp.get(), probe);
        Ok(())
    }
}
