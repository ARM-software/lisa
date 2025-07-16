/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, ffi::CString, vec::Vec};
use core::{
    any::Any,
    cell::UnsafeCell,
    ffi::{CStr, c_int, c_void},
    fmt,
    marker::PhantomData,
    pin::Pin,
};

use lisakmod_macros::inlinec::{cfunc, opaque_type};

use crate::{
    error::{Error, error},
    runtime::{
        printk::pr_debug,
        sync::{Lock as _, Mutex, new_static_lockdep_class},
    },
};

opaque_type!(
    struct CTracepoint,
    "struct tracepoint",
    "linux/tracepoint.h",
);

pub struct Tracepoint<'tp, Args> {
    // struct tracepoint API allows mutation but protects it with an internal mutex
    // (tracepoint_mutex static variable in tracepoint.c)
    c_tp: &'tp UnsafeCell<CTracepoint>,
    _phantom: PhantomData<Args>,
}
// SAFETY: The Tracepoint is safe to pass around as it only stores shared references.
unsafe impl<'tp, Args> Send for Tracepoint<'tp, Args> {}

// SAFETY: the tracepoint kernel API is protected with "tracepoints_mutex" mutex (see
// tracepoint.c), so it safe to call it from multiple threads.
unsafe impl<'tp, Args> Sync for Tracepoint<'tp, Args> {}

impl<Args> Tracepoint<'static, Args> {
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
        unsafe fn find_tp<'tp>(name: &CStr) -> Option<&'tp UnsafeCell<CTracepoint>> {
            r#"
            #include <linux/tracepoint.h>
            #include <linux/string.h>

            struct __find_tracepoint_params {
                const char *name;
                size_t len;
                struct tracepoint *found;
            };

            static void __do_find_tracepoint(struct tracepoint *tp, void *__finder) {
                struct __find_tracepoint_params *finder = __finder;
                if (!strcmp(tp->name, finder->name))
                    finder->found = tp;
            }
            "#;

            r#"
            struct __find_tracepoint_params res = {.name = name, .found = NULL};
            for_each_kernel_tracepoint(__do_find_tracepoint, &res);
            return res.found;
            "#
        }

        let name: CString =
            CString::new(name).expect("tracepoint name cannot be converted to a C-style string");

        // SAFETY: The user guarantees that lookup() will only be used for tracepoints defined in
        // the base kernel image, so with a 'static lifetime
        let c_tp = unsafe { find_tp(name.as_c_str())? };

        Some(Tracepoint {
            c_tp,
            _phantom: PhantomData,
        })
    }
}

#[derive(Debug)]
pub enum TracepointError {
    Kernel(c_int),
}

impl<'tp, Args> Tracepoint<'tp, Args> {
    // We cannot use the facilities of opaque_type!() as they require a &Self, which we cannot
    // safely materialize for CTracepoint as there may be concurrent users of the struct.
    pub fn name<'a>(&'a self) -> &'a str {
        #[cfunc]
        fn name<'a>(tp: *const CTracepoint) -> Option<&'a CStr> {
            r#"
            #include <linux/tracepoint.h>
            "#;

            r#"
            return tp->name;
            "#
        }
        name::<'a>(self.c_tp.get())
            .map(|s| s.to_str().expect("Invalid UTF-8 in C string"))
            .unwrap_or("<unnamed>")
    }

    pub fn register_probe<'probe>(
        &'probe self,
        probe: &'probe Probe<'probe, Args>,
    ) -> Result<RegisteredProbe<'probe, Args>, Error>
    where
        'tp: 'probe,
    {
        /// # Safety
        ///
        /// All pointers must be valid until tracepoint_probe_unregister() kernel function is called
        /// with the same parameters.
        #[cfunc]
        unsafe fn register(
            tp: *mut CTracepoint,
            probe: *const c_void,
            data: *const c_void,
        ) -> Result<(), c_int> {
            r#"
            #include <linux/tracepoint.h>
            "#;

            r#"
            return tracepoint_probe_register(tp, CONST_CAST(void *, probe), CONST_CAST(void *, data));
            "#
        }

        // SAFETY: If the Probe is alive, then its closure is alive too. Probe will be kept alive
        // until the last RegisteredProbe is dropped, at which point we known that
        // tracepoint_probe_unregister() has been called.
        unsafe { register(self.c_tp.get(), probe.probe, probe.closure) }
            .map_err(|code| error!("Tracepoint probe registration failed: {code}"))?;
        Ok(RegisteredProbe { probe, tp: self })
    }
}

impl<'tp, Args> fmt::Debug for Tracepoint<'tp, Args> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("Tracepoint")
            .field("name", &self.name())
            .finish()
    }
}

pub struct RegisteredProbe<'probe, Args> {
    probe: &'probe Probe<'probe, Args>,
    tp: &'probe Tracepoint<'probe, Args>,
}

impl<'probe, Args> Drop for RegisteredProbe<'probe, Args> {
    fn drop(&mut self) {
        /// # Safety
        ///
        /// This must be called with arguments matching tracepoint_probe_register(), once per such
        /// call.
        #[cfunc]
        unsafe fn unregister(
            tp: *mut CTracepoint,
            probe: *const c_void,
            data: *const c_void,
        ) -> Result<(), c_int> {
            r#"
            #include <linux/tracepoint.h>
            "#;

            r#"
            return tracepoint_probe_unregister(tp, CONST_CAST(void *, probe), CONST_CAST(void *, data));
            "#
        }

        let probe = self.probe;
        // SAFETY: Since we only create RegisteredProbe values when the registration succeeded, we
        // ensure that this will be the only matching call to tracepoint_probe_unregister()
        unsafe { unregister(self.tp.c_tp.get(), probe.probe, probe.closure as *mut _) }
            .expect("Failed to unregister tracepoint probe");

        pr_debug!(
            "Called tracepoint_probe_unregister() for a probe attached to {:?}",
            self.tp
        );
    }
}

pub struct ProbeDropper {
    droppers: Mutex<Vec<Option<Pin<Box<dyn Any + Send + Sync>>>>>,
}

impl fmt::Debug for ProbeDropper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("ProbeDropper").finish()
    }
}

impl Default for ProbeDropper {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbeDropper {
    pub fn new() -> ProbeDropper {
        new_static_lockdep_class!(PROBE_DROPPER_LOCKDEP_CLASS);
        ProbeDropper {
            droppers: Mutex::new(Vec::new(), PROBE_DROPPER_LOCKDEP_CLASS.clone()),
        }
    }
}

impl Drop for ProbeDropper {
    fn drop(&mut self) {
        #[cfunc]
        fn tracepoint_synchronize_unregister() {
            r#"
            #include <linux/tracepoint.h>
            "#;

            r#"
            tracepoint_synchronize_unregister();
            "#
        }

        tracepoint_synchronize_unregister();

        pr_debug!("Called tracepoint_synchronize_unregister(), de-allocating closures now");

        // SAFETY: If we are dropped, that means we are not borrowed anymore, and since:
        // * RegisteredProbe borrows Probe us while the probe can fire, and
        // * Probe borrows ProbeDropper while it is alive.
        // then it means the only consumer of the closure left are the tracepoint themselves, which
        // we ensured do not use it anymore since:
        // * <RegisteredProbe as Drop::drop() called tracepoint_probe_unregister()
        // * We just called tracepoint_synchronize_unregister()
        //
        // It is therefore now safe to drop the closure.
        for drop_ in &mut *self.droppers.lock() {
            drop(drop_.take());
        }
    }
}

pub struct Probe<'probe, Args> {
    closure: *const c_void,
    probe: *const c_void,
    _phantom: PhantomData<(&'probe (), Args)>,
}

impl<'probe, Args> Probe<'probe, Args> {
    /// # Safety
    ///
    /// The `probe` must be compatible with the [`Closure`] type in use.
    #[inline]
    pub unsafe fn __private_new<Closure>(
        closure: Pin<Box<Closure>>,
        probe: *const c_void,
        // SAFETY: we borrow ProbeDropper for &'probe, which ensures the ProbeDropper will not be
        // dropped before Probe<'probe, _>
        dropper: &'probe ProbeDropper,
    ) -> Probe<'probe, Args>
    where
        Closure: 'static + Send + Sync,
    {
        let ptr: &Closure = &closure;
        let ptr: *const Closure = ptr;

        dropper.droppers.lock().push(Some(closure));
        // SAFETY: the probe and closure pointer we store here are guaranteed to be valid for the
        // lifetime of the Probe object, as we borrow the ProbeDropper for that duration.
        Probe {
            closure: ptr as *const c_void,
            probe,
            _phantom: PhantomData,
        }
    }
}

// SAFETY: this is ok as the closure we store is also Send
unsafe impl<'probe, Args> Send for Probe<'probe, Args> {}
// SAFETY: this is ok as the closure we store is also Sync
unsafe impl<'probe, Args> Sync for Probe<'probe, Args> {}

macro_rules! new_probe {
    ($dropper:expr, ( $($arg_name:ident: $arg_ty:ty),* ) $body:block) => {
        {
            // SAFETY: We need to ensure Send and Sync for the closure, as Probe relies on that to
            // soundly implement Send and Sync
            type Closure = impl Fn($($arg_ty),*) + ::core::marker::Send + ::core::marker::Sync + 'static;

            let closure: ::core::pin::Pin<::alloc::boxed::Box<Closure>> = ::alloc::boxed::Box::pin(
                move |$($arg_name: $arg_ty),*| {
                    $body
                }
            );

            #[::lisakmod_macros::inlinec::cexport]
            fn probe(closure: *const c_void, $($arg_name: $arg_ty),*) {
                let closure = closure as *const Closure;
                // SAFETY: Since we call tracepoint_probe_unregister() in <RegisteredProbe as
                // Drop>::drop(), and RegisteredProbe keeps the Probe and its closure alive, then
                // the probe should never run after the closure is dropped.
                let closure = unsafe { closure.as_ref().expect("Unexpected NULL pointer") };
                closure($($arg_name),*)
            }

            #[allow(unused_unsafe)]
            unsafe {
                $crate::runtime::tracepoint::Probe::<( $($arg_ty),* )>::__private_new(
                    closure,
                    probe as *const c_void,
                    $dropper,
                )
            }
        }
    }
}

#[allow(unused_imports)]
pub(crate) use new_probe;
