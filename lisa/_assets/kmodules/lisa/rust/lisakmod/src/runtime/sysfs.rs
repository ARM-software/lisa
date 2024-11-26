/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, string::String, sync::Arc};
use core::{
    cell::UnsafeCell,
    convert::Infallible,
    ffi::{c_int, c_uint, c_void},
    marker::PhantomData,
    mem::MaybeUninit,
    pin::Pin,
};

use crate::{
    inlinec::{c_eval, cexport, cfunc, opaque_type},
    {container_of, mut_container_of},
};

opaque_type!(
    struct CKObj,
    "struct kobject",
    "linux/kobject.h",
    attr_by_value {fn state_in_sysfs(&self) -> bool},
    attr_by_value {fn ktype(&self) -> &CKObjType},
);
opaque_type!(struct CKObjType, "struct kobj_type", "linux/kobject.h");

// SAFETY: CKObjType is a plain-old-data struct, there is nothing in there that can't be shared
// between threads.
unsafe impl Send for CKObjType {}
unsafe impl Sync for CKObjType {}

pub struct KObjType {
    c_kobj_type: CKObjType,
}

impl Default for KObjType {
    fn default() -> Self {
        Self::new()
    }
}

// This type alias is used to ensure that all the places that need to create an allocated
// KObjectInner and the ones that re-materalize one from a raw pointer agree on what exact type is
// used.
type AllocatedKObjectInner = Pin<Box<KObjectInner>>;

impl KObjType {
    pub fn new() -> KObjType {
        #[cexport]
        fn release(this: &mut CKObj) {
            // SAFETY: This assumes the CKObj is always only located inside a KObjectInner. It also
            // assumes nothing else has a &mut KObjectInner over it, otherwise we would end up
            // aliasing a mutable reference which is UB.
            let inner: *mut KObjectInner = unsafe { mut_container_of!(KObjectInner, c_kobj, this) };

            // SAFETY: We built this KObjectInner in a Box<> and used Box::into_raw(). Here we know
            // that refcount == 0, so no-one else has a mutable reference to KObjectInner anymore,
            // so we can rematerialize the Box and free the lot.
            let b: AllocatedKObjectInner = Box::into_pin(unsafe { Box::from_raw(inner) });
            let kobj_type: *const KObjType = b.kobj_type();
            drop(b);

            // SAFETY: We destroyed KObjectInner, so now we can decrement the Arc refcount.
            unsafe {
                Arc::decrement_strong_count(kobj_type);
            }
        }

        #[cfunc]
        fn init_kobj_type(
            kobj_type: *mut CKObjType,
            release: *const c_void,
        ) -> Result<(), Infallible> {
            r#"
            #include <linux/string.h>
            #include <linux/kobject.h>
            "#;

            r#"
            memset(kobj_type, 0, sizeof(*kobj_type));
            kobj_type->release = release;
            "#
        }

        let release_f: unsafe extern "C" fn(*mut CKObj) = release;
        let init = |this| init_kobj_type(this, release_f as *const c_void);
        let c_kobj_type = unsafe { CKObjType::new_stack(init) }.unwrap();
        KObjType { c_kobj_type }
    }
}

// SAFETY: repr(transparent) is relied on to convert a *const CKObj into a *const KObjectInner.
#[repr(transparent)]
struct KObjectInner {
    c_kobj: UnsafeCell<CKObj>,
    // CKObj contains a pointer to CKObjType, so we reflect that dependency here. This is probably
    // not very useful.
    _phantom: PhantomData<KObjType>,
}

// SAFETY: KObjectInner is expected to be shared between multiple threads, as is the underlying
// struct kobject. Not all of struct kobject is thread-safe, but the API we expose should.
unsafe impl Send for KObjectInner {}
unsafe impl Sync for KObjectInner {}

impl KObjectInner {
    pub fn new(kobj_type: Arc<KObjType>) -> AllocatedKObjectInner {
        #[cfunc]
        unsafe fn kobj_init(kobj: *mut CKObj, kobj_type: &CKObjType) -> Result<(), Infallible> {
            r#"
            #include <linux/string.h>
            #include <linux/kobject.h>
            #include <utils.h>
            "#;

            r#"
            // memset() is necessary here, otherwise kobj->name stays uninitialized and that leads
            // to crashes when calling kobject_add() since it tries to free the previous name.
            memset(kobj, 0, sizeof(*kobj));

            // Some old kernels (e.g. 5.15) don't have a const qualifier for kobj_type, so we cast
            // it away with the warnings disabled to avoid hitting -Werror
            kobject_init(kobj, CONST_CAST(struct kobj_type*, kobj_type));
            "#
        }
        // Increase refcount since we are going to keep a pointer to KObjType. We decrement it
        // later when we don't need it anymore.
        core::mem::forget(kobj_type.clone());

        // Initialize the CKObj in-place in the heap. This is best since:
        // * In case kobject_init() expects the struct kobject to be pinned and never move again.
        // * The API documentation states that in the future, initialization on the stack might
        //   become forbidden. So make sure we don't do that.
        let mut new: Box<MaybeUninit<Self>> = Box::new_uninit();
        let new_ptr: *mut Self = new.as_mut_ptr();
        // SAFETY: new_ptr is valid
        unsafe {
            kobj_init((*new_ptr).c_kobj.get_mut(), &kobj_type.c_kobj_type).unwrap();
            (&raw mut (*new_ptr)._phantom).write(PhantomData);
        }

        // SAFETY: We initialized succesfully all the members.
        Box::into_pin(unsafe { new.assume_init() })
    }

    #[inline]
    fn with_c_kobj<T, F>(&self, f: F) -> T
    where
        F: for<'a> FnOnce(&'a CKObj) -> T,
    {
        // SAFETY: The reference passed to f() cannot be leaked by f() since it has a rank-2 type
        // (HRTB). The closure has to work for _any_ lifetime, so it cannot make any assumption
        // about that lifetime, and in particular, it cannot claim it returns a value with that
        // lifetime since the lifetime in question is unknown at the point the closure is defined
        // (it is only known inside here).
        let c_kobj = unsafe { &*self.c_kobj.get() };
        f(c_kobj)
    }

    fn from_c_kobj(c_kobj: *mut CKObj) -> *mut KObjectInner {
        const {
            // This is relied on in order to be able transmute a CKObj pointer to a KObjectInner
            // pointer so we can manipulate foreign kobjects, such as the sysfs root folder for the
            // module in /sys/module/lisa/
            //
            // If that assertion was not true anymore because we needed to store metadata in
            // KObjectInner, we would need to add a level of indirection.
            assert!(core::mem::size_of::<KObjectInner>() == core::mem::size_of::<CKObj>());
        };

        // SAFETY: it is safe to do that since we check that KObjectInner is exactly the same size
        // as CKObj and has repr(transparent)
        c_kobj as *mut KObjectInner
    }

    fn refcount(&self) -> u64 {
        #[cfunc]
        fn kobj_refcount(kobj: *const CKObj) -> c_uint {
            r#"
            #include <linux/kobject.h>
            "#;

            r#"
            return kref_read(&kobj->kref);
            "#
        }
        kobj_refcount(self.c_kobj.get()).into()
    }

    #[inline]
    fn is_in_sysfs(&self) -> bool {
        self.with_c_kobj(|c_kobj| c_kobj.state_in_sysfs())
    }

    unsafe fn update_refcount(&self, increase: bool) {
        #[cfunc]
        fn kobj_update_refcount(kobj: *mut CKObj, increase: bool) -> bool {
            r#"
            #include <linux/kobject.h>
            "#;

            r#"
            if (increase) {
                return kobject_get(kobj) != NULL;
            } else {
                kobject_put(kobj);
                return 1;
            }
            "#
        }
        let ok = kobj_update_refcount(self.c_kobj.get(), increase);
        if !ok {
            panic!("Could not update KObjectInner refcount as it is undergoing destruction.")
        }
    }

    #[inline]
    fn incref(&self) {
        // SAFETY: increasing the refcount is always safe, since worst case is we leak memory.
        unsafe { self.update_refcount(true) }
    }

    #[inline]
    unsafe fn decref(&self) {
        unsafe { self.update_refcount(false) }
    }

    fn kobj_type(&self) -> &KObjType {
        let c_kobj_type = self.with_c_kobj(|c_kobj| c_kobj.ktype() as *const CKObjType);

        // SAFETY: All the CKObjType we manipulate are part of a KObjType, so it is safe to cast
        // back.
        unsafe {
            container_of!(KObjType, c_kobj_type, c_kobj_type)
                .as_ref()
                .unwrap()
        }
    }
}

impl Drop for KObjectInner {
    fn drop(&mut self) {
        let refcount = self.refcount();
        if refcount != 0 {
            panic!("Tried to drop a KObjType with non zero refcount: {refcount}")
        }
    }
}

mod private {
    pub trait Sealed {}
}
pub trait KObjectState: private::Sealed {}

struct SysfsSpec {
    name: String,
    parent: Option<KObject<Finalized>>,
}

#[derive(Default)]
pub struct Init {
    sysfs: Option<SysfsSpec>,
}
impl private::Sealed for Init {}
impl KObjectState for Init {}

#[derive(Default)]
pub struct Finalized {}
impl private::Sealed for Finalized {}
impl KObjectState for Finalized {}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {}

pub struct KObject<State: KObjectState> {
    // Use a pointer, so Rust does not make any assumption on the validity of the pointee. This
    // simplifies the drop flow.
    //
    // Morally, this is a Pin<*const KObjectInner> since the data behind it is pinned (it's coming
    // from an AllocatedKObjectInner and re-materalized as such when de-allocating). However,
    // Pin<Ptr> requires Ptr: Deref for anything interesting so we can't actually use it here.
    inner: *const KObjectInner,
    state: State,
}

// SAFETY: Nothing in KObject cares about what thread it is on, so it can be sent around without
// issues.
unsafe impl<State: KObjectState> Send for KObject<State> {}
// SAFETY: In the Finalized state, all APIs are as thread-safe as KObjectInner is. In the Init
// state, we are not thread safe as there is some builder state maintained in the KObject
unsafe impl Sync for KObject<Finalized> {}

impl KObject<Init> {
    #[inline]
    pub fn new(kobj_type: Arc<KObjType>) -> Self {
        let inner = KObjectInner::new(kobj_type);
        // SAFETY: Nothing in the public KObject API allows moving out the KObjectInner pointed at,
        // since that KObjectInner could be shared by any number of KObject instances, just like
        // ArcInner is shared among multiple Arc. We therefore can guarantee that KObjectInner will
        // stay exactly where it is until it is garbage-collected using kobject_put() somewhere in
        // our code or kernel code.
        let inner = unsafe { Pin::into_inner_unchecked(inner) };
        KObject {
            inner: Box::into_raw(inner),
            state: Default::default(),
        }
    }

    pub fn add<'a, 'parent, 'name>(
        &'a mut self,
        parent: Option<&'parent KObject<Finalized>>,
        name: &'name str,
    ) where
        'parent: 'a,
    {
        // FIXME: should we return an error or panic ?
        if let Some(parent) = parent {
            assert!(
                parent.inner().is_in_sysfs(),
                "The parent of KObject \"{name}\" was not added to sysfs"
            );
        }
        self.state.sysfs = Some(SysfsSpec {
            name: name.into(),
            parent: parent.cloned(),
        });
    }

    pub fn finalize(self) -> Result<KObject<Finalized>, Error> {
        #[cfunc]
        unsafe fn kobj_add(kobj: *mut CKObj, parent: *mut CKObj, name: &[u8]) -> Result<(), c_int> {
            r#"
            #include <linux/kobject.h>
            #include <linux/init.h>
            "#;

            r#"
            return kobject_add(kobj, parent, "%.*s", (int)name.len, name.data);
            "#
        }
        if let Some(spec) = &self.state.sysfs {
            let parent = match &spec.parent {
                None => core::ptr::null_mut(),
                Some(parent) => parent.c_kobj(),
            };
            unsafe { kobj_add(self.c_kobj(), parent, spec.name.as_bytes()) }
                .expect("Call to kobject_add() failed");
        }
        Ok(KObject::from_inner(self.inner()))
    }
}

impl KObject<Finalized> {
    fn from_inner(inner: &KObjectInner) -> Self {
        inner.incref();
        KObject {
            inner,
            state: Default::default(),
        }
    }
    #[inline]
    pub fn sysfs_module_root() -> Self {
        let c_kobj = c_eval!("linux/kobject.h", "&THIS_MODULE->mkobj.kobj", *mut CKObj);
        let inner = KObjectInner::from_c_kobj(c_kobj);
        // SAFETY: We assume that the pointer we got is valid as it is comming from a well-known
        // kernel API.
        let inner = unsafe { inner.as_ref().unwrap() };
        Self::from_inner(inner)
    }
}

impl<State: KObjectState> KObject<State> {
    fn inner(&self) -> &KObjectInner {
        // SAFETY: Since we hold a kref reference, we know we are pointing at valid memory.
        unsafe { self.inner.as_ref().unwrap() }
    }

    fn c_kobj(&self) -> *mut CKObj {
        self.inner().c_kobj.get()
    }
}

impl<State: KObjectState> Drop for KObject<State> {
    fn drop(&mut self) {
        // SAFETY: We increased the refcount when created, so we decrease it when dropped. The
        // release implementation of KObjType will take care of freeing the memory if needed.
        // Also, since KObject::inner is a pointer, it is ok if what it points to becomes garbage
        // during decref(), it would only be an issue if we had a reference instead of a pointer.
        unsafe {
            self.inner().decref();
        }
    }
}

// Only implement clone for the Finalized state so that we do not accidentally call kobject_add()
// twice on the same underlying "struct kobject".
impl Clone for KObject<Finalized> {
    fn clone(&self) -> Self {
        // SAFETY: self.inner is always valid as long as a KObject points to it.
        let inner = unsafe { self.inner.as_ref().unwrap() };
        Self::from_inner(inner)
    }
}
