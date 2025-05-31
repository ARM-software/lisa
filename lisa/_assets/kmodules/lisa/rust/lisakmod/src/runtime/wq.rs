/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, collections::BTreeMap, string::String};
use core::{
    cell::UnsafeCell,
    convert::Infallible,
    ffi::c_void,
    fmt,
    pin::Pin,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering},
};

use lisakmod_macros::inlinec::{cexport, cfunc, incomplete_opaque_type, opaque_type};
use pin_project::{pin_project, pinned_drop};

use crate::{
    error::{Error, error},
    mem::{FromContained, destructure, impl_from_contained},
    runtime::sync::{Lock as _, LockdepClass, Mutex, PinnedLock, new_static_lockdep_class},
};

incomplete_opaque_type!(
    pub struct CWq,
    "struct workqueue_struct",
    "linux/workqueue.h"
);

type Key = usize;

struct OwnedWorkItem {
    ptr: Option<*const ()>,
    drop_from_worker: Box<dyn Fn(*const ())>,
    drop_normal: Box<dyn Fn(*const ())>,
}
unsafe impl Send for OwnedWorkItem {}

impl OwnedWorkItem {
    fn drop_from_worker(mut self) {
        (self.drop_from_worker)(
            self.ptr
                .take()
                .expect("OwnedWorkItem has already been dropped"),
        )
    }

    fn drop_normal(mut self) {
        (self.drop_normal)(
            self.ptr
                .take()
                .expect("OwnedWorkItem has already been dropped"),
        )
    }
}

impl Drop for OwnedWorkItem {
    fn drop(&mut self) {
        assert!(
            self.ptr.is_none(),
            "OwnedWorkItem must be dropped with a method"
        )
    }
}

pub struct Wq {
    c_wq: NonNull<UnsafeCell<CWq>>,
    work_nr: AtomicUsize,
    name: String,
    owned_work: Mutex<BTreeMap<Key, OwnedWorkItem>>,
}
unsafe impl Send for Wq {}
unsafe impl Sync for Wq {}

impl fmt::Debug for Wq {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wq").finish_non_exhaustive()
    }
}

impl Wq {
    #[inline]
    pub fn new(name: &str) -> Result<Wq, Error> {
        #[cfunc]
        fn allo_workqueue(name: &str) -> Option<NonNull<UnsafeCell<CWq>>> {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            // Expose the workqueue in sysfs if a user needs to tune the CPU placement.
            return alloc_workqueue("%.*s", WQ_SYSFS | WQ_FREEZABLE, 0, (int)name.len, name.data);
            "#;
        }

        new_static_lockdep_class!(WQ_OWNED_WORK_LOCKDEP_CLASS);

        match allo_workqueue(name) {
            None => Err(error!("Unable to allocate struct workqueue_struct")),
            Some(c_wq) => Ok(Wq {
                c_wq,
                work_nr: AtomicUsize::new(0),
                name: name.into(),
                owned_work: Mutex::new(BTreeMap::new(), WQ_OWNED_WORK_LOCKDEP_CLASS.clone()),
            }),
        }
    }

    #[inline]
    fn c_wq(&self) -> &UnsafeCell<CWq> {
        unsafe { self.c_wq.as_ref() }
    }

    pub fn __attach<'a, 'wq, 'f, Init>(
        self: Pin<&'a Self>,
        work_item: WorkItem<'wq, 'f>,
        init: Init,
    ) -> Key
    where
        Self: 'wq,
        Init: FnOnce(&mut dyn AbstractWorkItem),
    {
        let key: *const DelayedWork =
            work_item.with_dwork(|dwork| dwork.as_ref().get_ref() as *const _);
        let key = key as usize;

        let work_item = Box::into_raw(Box::new(work_item));
        // SAFETY: We clear the owned_work in Drop, thereby ensuring a WorkItem<'wq> never ends
        // up surviving its associated Wq
        {
            let mut owned_work_guard = self.owned_work.lock();
            owned_work_guard.insert(
                key,
                OwnedWorkItem {
                    ptr: Some(work_item as *const ()),
                    drop_from_worker: Box::new(|ptr| {
                        let ptr = ptr as *mut WorkItem<'wq, 'f>;
                        let work = unsafe { Box::from_raw(ptr) };
                        // OwnedWorkItem::drop_normal() would block until the worker is not running
                        // anymore, leading to a deadlock if called from the worker function.
                        work.drop_unsync();
                    }),
                    drop_normal: Box::new(|ptr| {
                        let ptr = ptr as *mut WorkItem<'wq, 'f>;
                        let _ = unsafe { Box::from_raw(ptr) };
                    }),
                },
            );
            // SAFETY:
            // * We hold the owned_work lock here, so nothing else can remove the work item we just
            //   inserted.
            // * There is no other reference to the work_item (we stored a pointer in the BTreeMap,
            //   not a reference)
            // * We hold the work_item lock as well, so it cannot run and drop itself while we are
            //   running the init function.
            let work_item = unsafe { work_item.as_mut().unwrap() };
            work_item.with_dwork(|mut dwork| {
                let abstr: &mut dyn AbstractWorkItem = &mut dwork.as_mut();
                init(abstr);
            });
            drop(owned_work_guard);
        }
        key
    }

    pub fn __detach(&self, key: Key) {
        if let Some(owned) = self.owned_work.lock().remove(&key) {
            owned.drop_from_worker()
        }
    }

    pub fn clear_owned_work(&self) {
        let map = core::mem::take(&mut *self.owned_work.lock());
        for owned in map.into_values() {
            owned.drop_normal()
        }
    }
}

impl Drop for Wq {
    fn drop(&mut self) {
        self.clear_owned_work();

        let work_nr = self.work_nr.load(Ordering::Relaxed);
        if work_nr != 0 {
            panic!(
                "{work_nr} work items have not been dropped befre the destruction of the {name} workqueue",
                name = self.name
            );
        }

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
    pub wq: &'wq Wq,
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

    #[inline]
    fn is_pending(&self) -> bool {
        #[cfunc]
        fn is_pending(dwork: &CDelayedWork) -> bool {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            return delayed_work_pending(dwork);
            "#;
        }
        is_pending(&self.c_dwork)
    }
}

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
type WorkF<'f> = Box<dyn FnMut(&mut dyn AbstractWorkItem) -> Action + Send + 'f>;

#[pin_project(PinnedDrop)]
pub struct WorkItemInner<'wq, 'f> {
    // SAFETY: WorkItemInner _must_ be pinned as the address of __dwork will be passed around C
    // APIs.
    #[pin]
    pub __dwork: DelayedWork<'wq>,
    pub __f: WorkF<'f>,
}

impl_from_contained!(('wq, 'f) WorkItemInner<'wq, 'f>, __dwork: DelayedWork<'wq>);

impl<'wq, 'f> WorkItemInner<'wq, 'f> {
    #[inline]
    fn new(
        wq: &'wq Wq,
        f: WorkF<'f>,
        lockdep_class: LockdepClass,
        init_dwork: fn(wq: &UnsafeCell<CWq>, dwork: Pin<&mut CDelayedWork>, worker: *const c_void),
    ) -> Pin<Box<PinnedWorkItemInner<'wq, 'f>>> {
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
            WorkItemInner { __dwork, __f: f },
            lockdep_class,
        )));
        wq.work_nr.fetch_add(1, Ordering::Relaxed);

        #[cexport]
        pub unsafe fn worker(c_work_struct: *mut __CWorkStruct) {
            // The prototype of the exported function must be exactly as the C API expects,
            // otherwise we get a CFI violation. However, we know we are actually passed a
            // Pin<&__CWorkStruct>
            let c_work_struct = unsafe {
                ::core::pin::Pin::new_unchecked(
                    c_work_struct.as_ref().expect("Unexpected NULL pointer"),
                )
            };
            let (action, wq, key) = {
                let inner = unsafe { c_work_struct.__to_work_item() };
                let mut inner = inner.lock();
                let mut inner = inner.as_mut().project();
                let dwork: &DelayedWork = &inner.__dwork;
                let key = dwork as *const _ as usize;
                let abstr: &mut dyn AbstractWorkItem = &mut inner.__dwork;
                ((inner.__f)(abstr), inner.__dwork.wq, key)
            };
            match action {
                Action::DropWorkItem => {
                    wq.__detach(key);
                }
                Action::Noop => {}
            }
        }

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
                worker as *const c_void,
            )
        }

        new
    }

    #[inline]
    unsafe fn from_dwork(c_dwork: Pin<&CDelayedWork>) -> Pin<&PinnedWorkItemInner<'wq, 'f>> {
        unsafe {
            let dwork = DelayedWork::<'wq>::from_contained(c_dwork.get_ref())
                .as_ref()
                .unwrap();
            let inner = WorkItemInner::<'wq, 'f>::from_contained(dwork)
                .as_ref()
                .unwrap();
            let inner = Mutex::<WorkItemInner<'wq, 'f>>::from_contained(inner)
                .as_ref()
                .unwrap();
            let inner = PinnedLock::<Mutex<WorkItemInner<'wq, 'f>>>::from_contained(inner)
                .as_ref()
                .unwrap();
            Pin::new_unchecked(inner)
        }
    }
}

#[pinned_drop]
impl<'wq, 'f> PinnedDrop for WorkItemInner<'wq, 'f> {
    fn drop(self: Pin<&mut Self>) {
        self.__dwork.wq.work_nr.fetch_sub(1, Ordering::Relaxed);
    }
}

type PinnedWorkItemInner<'wq, 'f> = PinnedLock<Mutex<WorkItemInner<'wq, 'f>>>;

pub struct WorkItem<'wq, 'f> {
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
    inner: Pin<Box<PinnedWorkItemInner<'wq, 'f>>>,
}

impl<'wq, 'f> Drop for WorkItem<'wq, 'f> {
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

impl<'wq, 'f> WorkItem<'wq, 'f> {
    #[inline]
    pub fn __private_new<F>(
        wq: &'wq Wq,
        f: F,
        lockdep_class: LockdepClass,
        init_dwork: fn(wq: &UnsafeCell<CWq>, dwork: Pin<&mut CDelayedWork>, worker: *const c_void),
    ) -> WorkItem<'wq, 'f>
    where
        F: 'f + FnMut(&mut dyn AbstractWorkItem) -> Action + Send,
    {
        WorkItem {
            inner: WorkItemInner::new(wq, Box::new(f), lockdep_class, init_dwork),
        }
    }

    #[inline]
    fn with_dwork<T, F>(&self, f: F) -> T
    where
        F: FnOnce(Pin<&mut DelayedWork<'wq>>) -> T,
    {
        f(self.inner.as_ref().lock().as_mut().project().__dwork)
    }

    #[inline]
    pub fn enqueue(&self, delay_us: u64) {
        self.with_dwork(|dwork| dwork.enqueue(delay_us))
    }

    fn drop_unsync(self) {
        // We need to make sure we drop all the fields. To make sure we didn't forget any, we
        // pattern match on the struct.
        let WorkItem { inner } = &self;
        // Simply drop the fields, without running any logic to synchronize with the workqueue.
        let (inner,) = destructure!(self, inner);
    }
}

pub trait AbstractWorkItem {
    fn enqueue(&mut self, delay_us: u64);
    fn is_pending(&self) -> bool;
}

impl<'wq> AbstractWorkItem for Pin<&mut DelayedWork<'wq>> {
    #[inline]
    fn enqueue(&mut self, delay_us: u64) {
        self.as_mut().enqueue(delay_us)
    }

    #[inline]
    fn is_pending(&self) -> bool {
        self.as_ref().is_pending()
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
    pub unsafe fn __to_work_item<'wq, 'f>(self: Pin<&Self>) -> Pin<&PinnedWorkItemInner<'wq, 'f>> {
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
        unsafe { WorkItemInner::<'wq, 'f>::from_dwork(c_dwork) }
    }
}

pub enum Action {
    Noop,
    DropWorkItem,
}

macro_rules! __new_work_item {
    ($wq:expr, $f:expr) => {{
        // We create this function here and pass it down rather than having a single copy of the
        // function used for all work items because INIT_DELAYED_WORK() also statically creates a
        // lockdep class for the item. If we use the same INIT_DELAYED_WORK() call site for all
        // workers, they will collectively be treated as a single function from lockdep
        // perspective, creating dependencies between locks that do not exist in practice.
        #[::lisakmod_macros::inlinec::cfunc]
        fn init_dwork(
            wq: &::core::cell::UnsafeCell<$crate::runtime::wq::CWq>,
            dwork: ::core::pin::Pin<&mut $crate::runtime::wq::CDelayedWork>,
            worker: *const ::core::ffi::c_void,
        ) {
            r#"
            #include <linux/workqueue.h>
            "#;

            r#"
            INIT_DELAYED_WORK(dwork, worker);
            "#;
        }
        $crate::runtime::sync::new_static_lockdep_class!(WORK_ITEM_INNER_LOCKDEP_CLASS);
        $crate::runtime::wq::WorkItem::__private_new(
            $wq,
            $f,
            WORK_ITEM_INNER_LOCKDEP_CLASS.clone(),
            init_dwork,
        )
    }};
}

#[allow(unused_imports)]
pub(crate) use __new_work_item;

macro_rules! new_work_item {
    ($wq:expr, $f:expr) => {{
        fn coerce_hrtb<F: ::core::ops::FnMut(&mut dyn $crate::runtime::wq::AbstractWorkItem)>(
            f: F,
        ) -> F {
            f
        }
        let mut f_ = coerce_hrtb($f);
        let f = move |work: &mut dyn $crate::runtime::wq::AbstractWorkItem| {
            f_(work);
            $crate::runtime::wq::Action::Noop
        };
        $crate::runtime::wq::__new_work_item!($wq, f)
    }};
}

#[allow(unused_imports)]
pub(crate) use new_work_item;

macro_rules! new_attached_work_item {
    ($wq:expr, $f:expr, $init:expr) => {{
        fn check<F>(f: F) -> F
        where
            F: ::core::ops::FnMut(&mut dyn $crate::runtime::wq::AbstractWorkItem),
        {
            f
        }

        let mut f_ = check($f);
        let f = move |work: &mut dyn $crate::runtime::wq::AbstractWorkItem| {
            f_(work);
            if work.is_pending() {
                $crate::runtime::wq::Action::Noop
            } else {
                // If the worker did not re-enqueue itself, we can drop it as nothing else will
                // enqueue it from now-on.
                $crate::runtime::wq::Action::DropWorkItem
            }
        };

        $crate::runtime::sync::new_static_lockdep_class!(ATTACHED_WORK_ITEM_INNER_LOCKDEP_CLASS);
        let wq: ::core::pin::Pin<&$crate::runtime::wq::Wq> = $wq;
        // SAFETY: Ensure the 'f lifetime parameter of WorkItem is 'static, so that we cannot
        // accidentally pass a closure that would become invalid before the workqueue tries to drop
        // it.
        let work: $crate::runtime::wq::WorkItem<'_, 'static> =
            $crate::runtime::wq::__new_work_item!(wq.get_ref(), f);
        wq.__attach(work, $init);
    }};
}
#[allow(unused_imports)]
pub(crate) use new_attached_work_item;
