/* SPDX-License-Identifier: GPL-2.0 */

use alloc::boxed::Box;
use core::{
    cell::UnsafeCell,
    convert::{AsMut, AsRef, Infallible},
    ffi::{CStr, c_int, c_void},
    pin::Pin,
    ptr::NonNull,
};

use lisakmod_macros::inlinec::{cfunc, incomplete_opaque_type, opaque_type};
use pin_project::{pin_project, pinned_drop};

use crate::{
    mem::{FromContained, impl_from_contained},
    runtime::sync::{Lock as _, LockdepClass, Mutex, PinnedLock},
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
            return alloc_workqueue(name, WQ_FREEZABLE, 0);
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

#[pin_project(PinnedDrop)]
pub struct DelayedWork<'wq> {
    #[pin]
    c_dwork: CDelayedWork,
    wq: &'wq Wq,
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
    fn enqueue(self: Pin<&mut Self>, delay: u64) {
        #[cfunc]
        fn enqueue(
            wq: &UnsafeCell<CWq>,
            dwork: Pin<&mut CDelayedWork>,
            delay: u64,
        ) -> Result<(), c_int> {
            r#"
            #include <linux/workqueue.h>
            #include <linux/jiffies.h>
            "#;

            r#"
            queue_delayed_work(wq, dwork, usecs_to_jiffies(delay));
            return 0;
            "#;
        }
        let this = self.project();
        enqueue(this.wq.c_wq(), this.c_dwork, delay)
            .expect("Could not initialize workqueue's delayed work");
    }

    #[inline]
    fn cancel_sync(self: Pin<&mut Self>) {
        #[cfunc]
        fn cancel_delayed_work_sync(dwork: Pin<&mut CDelayedWork>) {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            cancel_delayed_work_sync(dwork);
            "#
        }
        cancel_delayed_work_sync(self.project().c_dwork)
    }
}

#[pinned_drop]
impl<'wq> PinnedDrop for DelayedWork<'wq> {
    #[inline]
    fn drop(self: Pin<&mut Self>) {
        // SAFETY: We need to ensure the worker will not fire anymore and has finished running
        // before we return, so we don't accidentally free the closure while it is in use.
        self.cancel_sync();
    }
}

#[pin_project]
pub struct WorkItemInner<'wq, F: 'wq + Send + Sync> {
    // SAFETY: WorkItemInner _must_ be pinned as the address of __dwork will be passed around C
    // APIs.
    #[pin]
    pub __dwork: DelayedWork<'wq>,
    pub __f: F,
}

impl_from_contained!(('wq, F: Send + Sync) WorkItemInner<'wq, F>, __dwork: DelayedWork<'wq>);

impl<'wq, F> WorkItemInner<'wq, F>
where
    F: Send + Sync,
{
    #[inline]
    fn new(wq: &'wq Wq, f: F) -> Pin<Box<PinnedWorkItemInner<'wq, F>>>
    where
        F: Worker,
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
        let __dwork = DelayedWork { c_dwork, wq };
        let new = Box::pin(PinnedLock::new(Mutex::new(
            WorkItemInner { __dwork, __f: f },
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
                F::worker() as *const c_void,
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
    F: Send + Sync,
{
    // We need a Mutex as the queue_delayed_work() function protects the workqueue but not the the
    // dwork itself. If multiple threads try to enqueue the same dwork at the same time, or if a
    // thread tries to enqueue it at the same time as it enqueues itself, there would be a race
    // condition.
    inner: Pin<Box<PinnedWorkItemInner<'wq, F>>>,
}

impl<'wq, F> WorkItem<'wq, F>
where
    F: Fn(&mut dyn AbstractWorkItem) + Send + Sync + Worker,
{
    #[inline]
    pub fn __private_new(wq: &'wq Wq, f: F) -> WorkItem<'wq, F> {
        WorkItem {
            inner: WorkItemInner::new(wq, f),
        }
    }

    #[inline]
    pub fn enqueue(&self, delay: u64) {
        self.inner
            .as_ref()
            .lock()
            .as_mut()
            .project()
            .__dwork
            .enqueue(delay)
    }

    #[inline]
    fn cancel_sync(self: Pin<&mut Self>) {
        self.inner
            .as_ref()
            .lock()
            .as_mut()
            .project()
            .__dwork
            .cancel_sync()
    }
}

pub trait AbstractWorkItem {
    fn enqueue(&mut self, delay: u64);
}

impl<'wq> AbstractWorkItem for Pin<&mut DelayedWork<'wq>> {
    #[inline]
    fn enqueue(&mut self, delay: u64) {
        self.as_mut().enqueue(delay)
    }
}

incomplete_opaque_type!(
    pub struct __CWorkStruct,
    "struct work_struct",
    "linux/workqueue.h"
);

impl __CWorkStruct {
    pub unsafe fn __to_work_item<'wq, F>(self: Pin<&Self>) -> Pin<&PinnedWorkItemInner<'wq, F>>
    where
        F: Send + Sync,
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

pub unsafe trait Worker {
    fn worker() -> unsafe extern "C" fn(*mut __CWorkStruct);
}

macro_rules! new_work_item {
    ($wq:expr, $f:expr) => {{
        // SAFETY: We need to ensure Send and Sync for the closure, as WorkItem relies on that
        // to soundly implement Send and Sync
        pub type Closure = impl Fn(&mut dyn $crate::runtime::wq::AbstractWorkItem)
            + ::core::marker::Send
            + ::core::marker::Sync
            + 'static;
        let closure: Closure = $f;

        unsafe impl $crate::runtime::wq::Worker for Closure {
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

        $crate::runtime::wq::WorkItem::__private_new($wq, closure)
    }};
}
pub(crate) use new_work_item;
