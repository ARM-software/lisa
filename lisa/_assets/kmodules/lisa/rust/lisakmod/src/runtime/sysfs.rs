/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, sync::Arc};
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

opaque_type!(struct CKObj, "struct kobject", "linux/kobject.h");
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

#[repr(transparent)]
struct KObjectInner {
    c_kobj: UnsafeCell<CKObj>,
    // CKObj contains a pointer to CKObjType, so we reflect that dependency here.
    _phantom: PhantomData<KObjType>,
}

// SAFETY: KObjectInner is expected to be shared between multiple threads, as is the underlying
// struct kobject. Not all of struct kobject is thread-safe, but the API we expose should.
unsafe impl Send for KObjectInner {}
unsafe impl Sync for KObjectInner {}

const _: () = {
    // This is relied on in order to be able transmute a CKObj pointer to a KObjectInner pointer so
    // we can manipulate foreign kobjects, such as the sysfs root folder for the module in
    // /sys/module/lisa/
    //
    // If that assertion is not true anymore because we need to store metadata in KObjectInner, we
    // would need to add a level of indirection.
    assert!(core::mem::size_of::<KObjectInner>() == core::mem::size_of::<CKObj>());
};

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

    fn from_c_kobj(c_kobj: *mut CKObj) -> *mut KObjectInner {
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
        #[cfunc]
        fn kobj_kobj_type(kobj: *const CKObj) -> *const CKObjType {
            r#"
            #include <linux/kobject.h>
            "#;

            r#"
            return kobj->ktype;
            "#
        }
        let c_kobj_type = kobj_kobj_type(self.c_kobj.get());
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

pub struct KObject {
    // Use a pointer, so Rust does not make any assumption on the validity of the pointee. This
    // simplifies the drop flow.
    //
    // Morally, this is a Pin<*const KObjectInner> since the data behind it is pinned (it's coming
    // from an AllocatedKObjectInner and re-materalized as such when de-allocating). However,
    // Pin<Ptr> requires Ptr: Deref for anything interesting so we can't actually use it here.
    inner: *const KObjectInner,
}

// SAFETY: KObject is just a reference to a KObjectInner, so it is safe to pass it around threads.
unsafe impl Send for KObject {}

impl KObject {
    #[inline]
    pub fn new(kobj_type: Arc<KObjType>) -> KObject {
        let inner = KObjectInner::new(kobj_type);
        // SAFETY: Nothing in the public KObject API allows moving out the KObjectInner pointed at,
        // since that KObjectInner could be shared by any number of KObject instances, just like
        // ArcInner is shared among multiple Arc. We therefore can guarantee that KObjectInner will
        // stay exactly where it is until it is garbage-collected using kobject_put() somewhere in
        // our code or kernel code.
        let inner = unsafe { Pin::into_inner_unchecked(inner) };
        KObject {
            inner: Box::into_raw(inner),
        }
    }

    #[inline]
    pub fn module_root() -> KObject {
        let c_kobj = c_eval!("linux/kobject.h", "&THIS_MODULE->mkobj.kobj", *mut CKObj);
        let inner = KObjectInner::from_c_kobj(c_kobj);
        // SAFETY: We assume that the pointer we got is valid as it is comming from a well-known
        // kernel API.
        let inner = unsafe { inner.as_ref().unwrap() };
        KObject::from_inner(inner)
    }

    fn from_inner(inner: &KObjectInner) -> Self {
        inner.incref();
        KObject { inner }
    }

    fn inner(&self) -> &KObjectInner {
        // SAFETY: Since we hold a kref reference, we know we are pointing at valid memory.
        unsafe { self.inner.as_ref().unwrap() }
    }

    fn c_kobj(&self) -> *mut CKObj {
        self.inner().c_kobj.get()
    }

    // FIXME: We need to prevent the following sequence:
    // a=new()
    // b=new()
    // a.add(parent=Some(a), name=bar)
    // b.add(parent=None, name=foo)
    //
    // This is because we can apparently only add children under a parent that has been added
    // already, we cannot build the hierarchy "virtually" before attaching it via a top-level add.
    pub fn add<'a, 'parent, 'name>(&'a self, parent: Option<&'parent Self>, name: &'name str)
    where
        'parent: 'a,
    {
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
        let parent: *mut CKObj = match parent {
            Some(parent) => parent.c_kobj(),
            None => core::ptr::null_mut(),
        };
        unsafe { kobj_add(self.c_kobj(), parent, name.as_bytes()) }
            .expect("Failed to call kobject_add()");
    }
}

impl Drop for KObject {
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

impl Clone for KObject {
    fn clone(&self) -> Self {
        // SAFETY: self.inner is always valid as long as a KObject points to it.
        let inner = unsafe { self.inner.as_ref().unwrap() };
        Self::from_inner(inner)
    }
}
