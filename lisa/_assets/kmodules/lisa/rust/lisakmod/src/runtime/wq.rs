/* SPDX-License-Identifier: GPL-2.0 */

use alloc::boxed::Box;
use core::{
    cell::UnsafeCell,
    convert::Infallible,
    ffi::{CStr, c_int, c_void},
    pin::Pin,
    ptr::NonNull,
};

use lisakmod_macros::inlinec::{cfunc, incomplete_opaque_type, opaque_type};
use pin_project::pin_project;

use crate::{
    mem::{FromContained, impl_from_contained},
    runtime::sync::{LockdepClass, Mutex, PinnedLock},
};

incomplete_opaque_type!(
    struct CWq,
    "struct workqueue_struct",
    "linux/workqueue.h"
);

#[derive(Debug)]
pub struct Wq {
    c_wq: NonNull<UnsafeCell<CWq>>,
}
unsafe impl Send for Wq {}
unsafe impl Sync for Wq {}

impl Default for Wq {
    fn default() -> Self {
        Self::new()
    }
}

impl Wq {
    #[inline]
    pub fn new() -> Wq {
        #[cfunc]
        fn allo_workqueue(name: &CStr) -> Option<NonNull<UnsafeCell<CWq>>> {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            // Expose the workqueue in sysfs if a user needs to tune the CPU placement.
            return alloc_workqueue(name, WQ_SYSFS | WQ_FREEZABLE, 0);
            "#;
        }

        let c_wq = allo_workqueue(c"lisa").expect("Unable to allocate struct workqueue_struct");
        Wq { c_wq }
    }

    #[inline]
    fn c_wq(&self) -> &UnsafeCell<CWq> {
        unsafe { self.c_wq.as_ref() }
    }
}

impl Drop for Wq {
    fn drop(&mut self) {
        #[cfunc]
        fn destroy_workqueue(wq: NonNull<UnsafeCell<CWq>>) {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            destroy_workqueue(wq);
            "#;
        }
        destroy_workqueue(self.c_wq);
    }
}

opaque_type!(
    pub struct CDelayedWork,
    "struct delayed_work",
    "linux/workqueue.h"
);

#[pin_project]
pub struct DelayedWork<'wq> {
    #[pin]
    c_dwork: CDelayedWork,
    wq: &'wq Wq,
    // Flag set to true when the work should not be re-enqueued anymore.
    // This flag is unnecessary if disable_delayed_work_sync() function is available but for older
    // kernel < 6.10 where we only have cancel_delayed_work_sync(), we need to handle that manually
    disable: bool,
}
// SAFETY: DelayedWork is Send but not Sync as struct delayed_work is not Sync. Some APIs like
// queue_delayed_work() seem to be callable from any thread, but this is only partly true: the
// workqueue is designed for concurrent access, but the delayed_work itself is not protected
// against concurrent updates. The assumption of the kernel API is that only one thread will
// manipulate the delayed_work at a time.
unsafe impl<'wq> Send for DelayedWork<'wq> {}

impl_from_contained!(('wq) DelayedWork<'wq>, c_dwork: CDelayedWork);

impl<'wq> DelayedWork<'wq> {
    #[inline]
    fn enqueue(self: Pin<&mut Self>, delay_us: u64) {
        #[cfunc]
        fn enqueue(wq: &UnsafeCell<CWq>, dwork: Pin<&mut CDelayedWork>, delay_us: u64) -> bool {
            r#"
            #include <linux/workqueue.h>
            #include <linux/jiffies.h>
            "#;

            r#"
            return queue_delayed_work(wq, dwork, usecs_to_jiffies(delay_us));
            "#;
        }
        let this = self.project();
        if !*this.disable {
            // If the work was already enqueued, queue_delayed_work() will return false and not do
            // anything. queue_delayed_work() can also return false when the work was disabled.
            let _ = enqueue(this.wq.c_wq(), this.c_dwork, delay_us);
        }
    }
}

#[pin_project]
pub struct WorkItemInner<'wq, F> {
    // SAFETY: WorkItemInner _must_ be pinned as the address of __dwork will be passed around C
    // APIs.
    #[pin]
    pub __dwork: DelayedWork<'wq>,
    pub __f: F,
}

impl_from_contained!(('wq, F) WorkItemInner<'wq, F>, __dwork: DelayedWork<'wq>);

impl<'wq, F> WorkItemInner<'wq, F>
where
    // SAFETY: We only require F to be Send instead of Send + Sync as the workqueue API guarantees
    // that the worker will never execute more than once at a time provided the following
    // conditions are met:
    // > Workqueue guarantees that a work item cannot be re-entrant if the following conditions hold
    // > after a work item gets queued:
    // >  1. The work function hasn’t been changed.
    // >  2. No one queues the work item to another workqueue.
    // >  3. The work item hasn’t been reinitiated.
    // > In other words, if the above conditions hold, the work item is guaranteed to be executed by
    // > at most one worker system-wide at any given time.
    //
    // Condition 1. is trivially satisfied as we never update the DelayedWork.
    // Condition 2. is trivially satisfied as we never enqueue the DelayedWork to more than one
    // workqueue.
    // Condition 3. is trivially satisfied as we never re-initialize a work item.
    //
    // So we can assume that the workqueue API will take care of synchronization and all we need is
    // F to be Send.
    F: Send,
{
    #[inline]
    fn new<W>(wq: &'wq Wq, wrapper: W) -> Pin<Box<PinnedWorkItemInner<'wq, F>>>
    where
        W: ClosureWrapper<Closure = F>,
    {
        #[cfunc]
        unsafe fn init_dwork(
            wq: &UnsafeCell<CWq>,
            dwork: Pin<&mut CDelayedWork>,
            worker: *const c_void,
        ) -> Result<(), c_int> {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            INIT_DELAYED_WORK(dwork, worker);
            return 0;
            "#;
        }
        // SAFETY: CDelayedWork does not have any specific validity invariant since it's
        // essentially an opaque type. We don't want to pass it to the C API before it is moved to
        // its final memory location in a Box.
        let c_dwork = unsafe { CDelayedWork::new_stack(|_| Ok::<(), Infallible>(())) }.unwrap();
        let __dwork = DelayedWork {
            c_dwork,
            wq,
            disable: false,
        };
        let new = Box::pin(PinnedLock::new(Mutex::new(
            WorkItemInner {
                __dwork,
                __f: wrapper.closure(),
            },
            LockdepClass::new(),
        )));

        unsafe {
            init_dwork(
                wq.c_wq(),
                new.as_ref()
                    .lock()
                    .as_mut()
                    .project()
                    .__dwork
                    .project()
                    .c_dwork,
                W::worker() as *const c_void,
            )
        }
        .expect("Could not initialize workqueue's delayed work");

        new
    }

    #[inline]
    unsafe fn from_dwork(c_dwork: Pin<&CDelayedWork>) -> Pin<&PinnedWorkItemInner<'wq, F>> {
        unsafe {
            let dwork = DelayedWork::<'wq>::from_contained(c_dwork.get_ref())
                .as_ref()
                .unwrap();
            let inner = WorkItemInner::<'wq, F>::from_contained(dwork)
                .as_ref()
                .unwrap();
            let inner = Mutex::<WorkItemInner<'wq, F>>::from_contained(inner)
                .as_ref()
                .unwrap();
            let inner = PinnedLock::<Mutex<WorkItemInner<'wq, F>>>::from_contained(inner)
                .as_ref()
                .unwrap();
            Pin::new_unchecked(inner)
        }
    }
}

type PinnedWorkItemInner<'wq, F> = PinnedLock<Mutex<WorkItemInner<'wq, F>>>;

pub struct WorkItem<'wq, F>
where
    F: Send,
{
    // SAFETY: The WorkItemInner need to _always_ be accessed after locking the lock, i.e. even if
    // we have an &mut, we cannot use that knowledge to bypass taking the lock. This is because the
    // actual worker materializes a shared reference and takes the lock, so we need to play ball
    // with it as well. If the design was using some sort of guard type borrowing WorkItemInner
    // returned by WorkItem::enqueue(), we would not be able to get an &mut until all guards were
    // dropped, but we don't so we need to be careful
    //
    // We need a Mutex as the queue_delayed_work() function protects the workqueue but not the the
    // dwork itself. If multiple threads try to enqueue the same dwork at the same time, or if a
    // thread tries to enqueue it at the same time as it enqueues itself, there would be a race
    // condition.
    inner: Pin<Box<PinnedWorkItemInner<'wq, F>>>,
}

impl<'wq, F> Drop for WorkItem<'wq, F>
where
    F: Send,
{
    #[inline]
    fn drop(&mut self) {
        // SAFETY: We need to ensure the worker will not fire anymore and has finished running
        // before we return, so we don't accidentally free the closure while it is in use.
        //
        // This looks like it belongs to <DelayedWork as Drop::drop() but that is misleading. The
        // Mutex we have in PinnedWorkItemInner dropped before the contained value, and that is a
        // valid thing to allow for Rust as this is not supposed to happen while there are some
        // shared references to that Mutex by the time it is dropped. However, the worker can
        // materialize such shared references out of thin air until we call disable_delayed_work_sync(). If we
        // did not call disable_delayed_work_sync() before dropping the Mutex, we have a race condition where the
        // Mutex is dropped while the worker still has a reference over it, which leads to UB.
        // This can be solved in a few ways:
        // 1. Prevent the Mutex from actually being dropped until it's not used anymore using
        //    dynamic tracking e.g. an Arc. This has a runtime cost.
        // 2. Make WorkItem::enqueue() return a guard value that borrows from WorkItem, so that
        //    WorkItem can only be dropped after that guard is dropped. That guard can then call
        //    disable_delayed_work_sync(). The guard design may make enqueuing in a loop harder, so instead a
        //    guard for the whole object should be created, and the enqueue() method should be on
        //    the guard itself.
        // 3. What we chose: make WorkItem be its own guard, so that when it is dropped, it calls
        //    disable_delayed_work_sync(), and that will be called before dropping the inner Mutex as per the
        //    Rust drop order for nested struct (outer first, then inner).
        //    This requires care as manipulating an &mut WorkItem may allow us to get an &mut on
        //    the Mutex, which would technically allow us to get an &mut on the protected content
        //    without actually locking the Mutex. This must never happen though, as this would be
        //    undefined behavior since a shared reference to the content may still be materialized
        //    in the worker at the same time.
        #[cfunc]
        unsafe fn disable_delayed_work_sync(dwork: *mut CDelayedWork) {
            r#"
            #include <linux/workqueue.h>
            #include "introspection.h"
            "#;

            r#"
            #if HAS_SYMBOL(disable_delayed_work_sync)
                disable_delayed_work_sync(dwork);
            #else
                cancel_delayed_work_sync(dwork);
            #endif
            "#
        }

        // We get the pointer, then unlock, then disable the work. This avoids a deadlock where the
        // worker would attempt to lock the WorkItemInner after we locked it, and we would wait for
        // the worker to finish before unlocking.
        let c_dwork = self.with_dwork(|dwork| {
            let dwork = dwork.project();
            *dwork.disable = true;
            let c_dwork: &CDelayedWork = dwork.c_dwork.as_ref().get_ref();
            c_dwork as *const CDelayedWork as *mut _
        });
        unsafe {
            disable_delayed_work_sync(c_dwork);
        }
    }
}

impl<'wq, F> WorkItem<'wq, F>
where
    F: Send,
{
    #[inline]
    pub fn __private_new<W>(wq: &'wq Wq, wrapper: W) -> WorkItem<'wq, F>
    where
        W: ClosureWrapper<Closure = F>,
    {
        WorkItem {
            inner: WorkItemInner::new(wq, wrapper),
        }
    }

    #[inline]
    fn with_dwork<T, _F>(&self, f: _F) -> T
    where
        _F: FnOnce(Pin<&mut DelayedWork<'wq>>) -> T,
    {
        f(self.inner.as_ref().lock().as_mut().project().__dwork)
    }

    #[inline]
    pub fn enqueue(&self, delay_us: u64) {
        self.with_dwork(|dwork| dwork.enqueue(delay_us))
    }
}

pub trait AbstractWorkItem {
    fn enqueue(&mut self, delay_us: u64);
}

impl<'wq> AbstractWorkItem for Pin<&mut DelayedWork<'wq>> {
    #[inline]
    fn enqueue(&mut self, delay_us: u64) {
        self.as_mut().enqueue(delay_us)
    }
}

incomplete_opaque_type!(
    pub struct __CWorkStruct,
    "struct work_struct",
    "linux/workqueue.h"
);

impl __CWorkStruct {
    /// # Safety
    ///
    /// This function assumes that the __CWorkStruct is nested inside a PinnedWorkItemInner.
    pub unsafe fn __to_work_item<'wq, F>(self: Pin<&Self>) -> Pin<&PinnedWorkItemInner<'wq, F>>
    where
        F: Send,
    {
        #[cfunc]
        fn to_dwork(work: Pin<&__CWorkStruct>) -> Pin<&CDelayedWork> {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            return to_delayed_work(CONST_CAST(struct work_struct *, work));
            "#
        }
        let c_dwork = to_dwork(self);
        unsafe { WorkItemInner::<'wq, F>::from_dwork(c_dwork) }
    }
}

/// # Safety
///
/// This trait must not be implemented by the user directly. Use the new_work_item!() macro
/// instead.
pub unsafe trait Worker {
    fn worker() -> unsafe extern "C" fn(*mut __CWorkStruct);
}

pub trait ClosureWrapper: Worker {
    type Closure: FnMut(&mut dyn AbstractWorkItem) + Send;
    fn closure(self) -> Self::Closure;
}

macro_rules! new_work_item {
    ($wq:expr, $f:expr) => {{
        // SAFETY: We need to ensure Send for the closure, as WorkItem relies on that
        // to soundly implement Send
        pub type Closure = impl ::core::ops::FnMut(&mut dyn $crate::runtime::wq::AbstractWorkItem)
            + ::core::marker::Send
            + 'static;
        let closure: Closure = $f;

        // A layer is necessary as we cannot implement Worker directly for Closure due to this
        // issue:
        // https://github.com/rust-lang/rust/issues/139583
        struct Wrapper {
            closure: Closure,
        }

        struct MyClosure {
            closure: Closure,
        }
        impl ::core::ops::FnOnce<(&mut dyn $crate::runtime::wq::AbstractWorkItem,)> for MyClosure {
            type Output = ();
            extern "rust-call" fn call_once(
                mut self,
                args: (&mut dyn $crate::runtime::wq::AbstractWorkItem,),
            ) -> Self::Output {
                (self.closure)(args.0)
            }
        }
        impl ::core::ops::FnMut<(&mut dyn $crate::runtime::wq::AbstractWorkItem,)> for MyClosure {
            extern "rust-call" fn call_mut(
                &mut self,
                args: (&mut dyn $crate::runtime::wq::AbstractWorkItem,),
            ) -> Self::Output {
                (self.closure)(args.0)
            }
        }

        impl $crate::runtime::wq::ClosureWrapper for Wrapper {
            type Closure = MyClosure;
            #[inline]
            // TODO: On Rust 1.86, using the Closure type alias directly triggers an error stating
            // the type is unconstrained by appears in closure() return type. As a workaround, we
            // encapsulate the closure into MyClosure, which hides the type alias and everything
            // works on older rustc versions.
            fn closure(self) -> MyClosure {
                MyClosure {
                    closure: self.closure,
                }
            }
        }

        unsafe impl $crate::runtime::wq::Worker for Wrapper {
            fn worker() -> unsafe extern "C" fn(*mut $crate::runtime::wq::__CWorkStruct) {
                #[::lisakmod_macros::inlinec::cexport]
                pub unsafe fn worker(c_work_struct: *mut $crate::runtime::wq::__CWorkStruct) {
                    // The prototype of the exported function must be exactly as the C API expects,
                    // otherwise we get a CFI violation. However, we know we are actually passed a
                    // Pin<&__CWorkStruct>
                    let c_work_struct = unsafe {
                        ::core::pin::Pin::new_unchecked(
                            c_work_struct.as_ref().expect("Unexpected NULL pointer"),
                        )
                    };
                    let inner = unsafe { c_work_struct.__to_work_item::<Closure>() };
                    let mut inner = inner.lock();
                    let mut inner = inner.as_mut().project();
                    let abstr: &mut dyn $crate::runtime::wq::AbstractWorkItem = &mut inner.__dwork;
                    (inner.__f)(abstr);
                }
                worker
            }
        }

        $crate::runtime::wq::WorkItem::__private_new($wq, Wrapper { closure })
    }};
}
#[allow(unused_imports)]
pub(crate) use new_work_item;
