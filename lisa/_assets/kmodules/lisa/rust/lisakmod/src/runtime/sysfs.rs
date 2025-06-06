/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, ffi::CString, string::String, sync::Arc, vec::Vec};
use core::{
    any::Any,
    cell::UnsafeCell,
    cmp::{max, min},
    convert::Infallible,
    ffi::{CStr, c_longlong, c_uint, c_void},
    marker::PhantomData,
    mem::MaybeUninit,
    pin::Pin,
};

use lisakmod_macros::inlinec::{NegativeError, c_realchar, ceval, cexport, cfunc, opaque_type};

use crate::{
    error::{Error, error},
    mem::{FromContained, container_of, impl_from_contained, mut_container_of},
    runtime::{
        fs::{CFile, FsMode},
        sync::{Lock as _, Mutex, new_static_lockdep_class},
    },
};

opaque_type!(
    pub struct CKObj,
    "struct kobject",
    "linux/kobject.h",
    attr_accessors {ktype: &'a CKObjType},
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
            #include "utils.h"
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
        // We cannot use opaque_type!() attribute accessor facilities as state_in_sysfs is a
        // bitfield, so taking its address is not allowed
        #[cfunc]
        fn state_in_sysfs(c_kobj: &CKObj) -> bool {
            r#"
            #include <linux/kobject.h>
            "#;

            r#"
            return c_kobj->state_in_sysfs;
            "#
        }
        self.with_c_kobj(state_in_sysfs)
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
    parent: Option<KObject<Published>>,
}

#[derive(Default)]
pub struct Init {
    sysfs: Option<SysfsSpec>,
}
impl private::Sealed for Init {}
impl KObjectState for Init {}

#[derive(Default)]
pub struct Published {}
impl private::Sealed for Published {}
impl KObjectState for Published {}

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
// SAFETY: In the Published state, all APIs are as thread-safe as KObjectInner is. In the Init
// state, we are not thread safe as there is some builder state maintained in the KObject
unsafe impl Sync for KObject<Published> {}

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
        parent: Option<&'parent KObject<Published>>,
        name: &'name str,
    ) -> Result<(), Error>
    where
        'parent: 'a,
    {
        let parent = match parent {
            Some(parent) => {
                if parent.inner().is_in_sysfs() {
                    Ok(Some(parent))
                } else {
                    Err(error!(
                        "The parent of KObject \"{name}\" was not added to sysfs"
                    ))
                }
            }
            None => Ok(None),
        }?;
        self.state.sysfs = Some(SysfsSpec {
            name: name.into(),
            parent: parent.cloned(),
        });
        Ok(())
    }

    pub fn publish(self) -> Result<KObject<Published>, Error> {
        #[cfunc]
        unsafe fn kobj_add(
            kobj: *mut CKObj,
            parent: *mut CKObj,
            name: &[u8],
        ) -> Result<c_uint, NegativeError<c_uint>> {
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
                .map_err(|err| error!("Call to kobject_add() failed: {err}"))?;
        }
        Ok(KObject::from_inner(self.inner()))
    }
}

impl KObject<Published> {
    fn from_inner(inner: &KObjectInner) -> Self {
        inner.incref();
        KObject {
            inner,
            state: Default::default(),
        }
    }

    /// # Safety
    ///
    /// The passed *mut CKObj must be a pointer valid for reads and writes
    pub unsafe fn from_c_kobj(c_kobj: *mut CKObj) -> Self {
        let inner = KObjectInner::from_c_kobj(c_kobj);
        let inner = unsafe { inner.as_ref().expect("Unexpected NULL pointer") };
        Self::from_inner(inner)
    }

    #[inline]
    pub fn sysfs_module_root() -> Self {
        let c_kobj = ceval!("linux/kobject.h", "&THIS_MODULE->mkobj.kobj", *mut CKObj);
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

    pub fn refcount(&self) -> u64 {
        self.inner().refcount()
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

// Only implement clone for the Published state so that we do not accidentally call kobject_add()
// twice on the same underlying "struct kobject".
impl Clone for KObject<Published> {
    fn clone(&self) -> Self {
        // SAFETY: self.inner is always valid as long as a KObject points to it.
        let inner = unsafe { self.inner.as_ref().unwrap() };
        Self::from_inner(inner)
    }
}

pub struct Folder {
    kobj: Arc<KObject<Published>>,
}

impl Folder {
    pub fn new(parent: &mut Folder, name: &str) -> Result<Folder, Error> {
        let kobj_type = Arc::new(KObjType::new());
        let mut kobj = KObject::new(kobj_type.clone());
        kobj.add(Some(&parent.kobj), name)?;
        let kobj = Arc::new(kobj.publish()?);
        Ok(Folder { kobj })
    }

    pub fn sysfs_module_root() -> Folder {
        Folder {
            kobj: Arc::new(KObject::<Published>::sysfs_module_root()),
        }
    }
}

opaque_type!(
    struct _CBinAttribute,
    "struct bin_attribute",
    "linux/sysfs.h",
);

#[repr(transparent)]
struct CBinAttribute(UnsafeCell<_CBinAttribute>);
unsafe impl Send for CBinAttribute {}
unsafe impl Sync for CBinAttribute {}

// SAFETY: FileInner must be pinned as CBinAttribute will contain a pointer to it in its
// "private" member.
pub struct FileInner {
    c_bin_attr: CBinAttribute,
    // SAFETY: name needs to be dropped _after_ c_bin_attr, as c_bin_attr contains a reference to
    // it. It therefore needs to be specified afterwards in the struct definition order.
    name: CString,
    parent_kobj: Arc<KObject<Published>>,

    // Use a trait object so that we can implement FileInner::from_attr(). This way, we can get an
    // arbitrary pointer and all the necessary type information is dynamically represented in the
    // data.
    ops: Box<dyn BinOps>,
}

impl_from_contained!(()FileInner, c_bin_attr: CBinAttribute);

impl FileInner {
    unsafe fn from_attr<'a>(attr: *const UnsafeCell<_CBinAttribute>) -> Pin<&'a FileInner> {
        let attr = attr as *const CBinAttribute;
        let inner: *const FileInner = unsafe { FromContained::from_contained(attr) };
        let inner: &FileInner = unsafe { inner.as_ref().unwrap() };
        unsafe { Pin::new_unchecked(inner) }
    }
}

pub trait BinOps: Any + Send + Sync {
    fn read(&self, offset: usize, out: &mut [u8]) -> Result<usize, NegativeError<usize>>;
    fn write(&self, offset: usize, in_: &[u8]) -> Result<usize, NegativeError<usize>>;
}

pub struct BinRWContent {
    content: Mutex<Vec<u8>>,
}

impl Default for BinRWContent {
    fn default() -> Self {
        Self::new()
    }
}

impl BinRWContent {
    pub fn new() -> BinRWContent {
        new_static_lockdep_class!(BIN_ATTR_CONTENT_LOCKDEP_CLASS);
        BinRWContent {
            content: Mutex::new(Vec::new(), BIN_ATTR_CONTENT_LOCKDEP_CLASS.clone()),
        }
    }

    pub fn with_content<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&[u8]) -> T,
    {
        f(&self.content.lock())
    }
}

impl BinOps for BinRWContent {
    fn read(&self, offset: usize, out: &mut [u8]) -> Result<usize, NegativeError<usize>> {
        match self.content.lock().get(offset..) {
            None => Ok(0),
            Some(in_) => {
                let written = min(in_.len(), out.len());
                out[..written].copy_from_slice(&in_[..written]);
                Ok(written)
            }
        }
    }

    fn write(&self, offset: usize, in_: &[u8]) -> Result<usize, NegativeError<usize>> {
        let mut content = self.content.lock();
        // Since we don't know have an opening mode, we don't know if the user wants to
        // write the data from scratch or simply edit the current content at the given
        // offset. As a result, we clear any existing data when writing at offset 0, but
        // writing at other offsets simply appends/updates.
        //
        // Unfortunately, "echo -n "" > foo" will not result in emptying the file, as
        // userspace tools typically do not issue write() syscalls with count == 0.
        // Instead, they just open it with O_WRONLY|O_CREAT|O_TRUNC, expecting the file to
        // be truncated. Unfortunately, sysfs just ignores O_TRUNC so nothing particular
        // happens and the file is not emptied.
        let content_size = if offset == 0 {
            content.clear();
            in_.len()
        } else {
            max(content.len(), offset.saturating_add(in_.len()))
        };
        content.resize(content_size, 0);

        // Release memory if we are done with it, to avoid carrying a huge allocation forever just
        // because of a single large write() happened in the lifetime of the file.
        if content.capacity() > content.len().saturating_mul(4) {
            content.shrink_to_fit()
        }

        match content.get_mut(offset..) {
            None => Err(NegativeError::EFBIG()),
            Some(out) => {
                let written = min(in_.len(), out.len());
                out[..written].copy_from_slice(&in_[..written]);
                Ok(written)
            }
        }
    }
}

pub struct BinROContent {
    inner: BinRWContent,
    read: Box<dyn Fn() -> Vec<u8> + Send + Sync>,
    eof_read: Box<dyn Fn() + Send + Sync>,
}

impl BinROContent {
    pub fn new<ReadF, EofReadF>(read: ReadF, eof_read: EofReadF) -> BinROContent
    where
        ReadF: 'static + Fn() -> Vec<u8> + Send + Sync,
        EofReadF: 'static + Fn() + Send + Sync,
    {
        BinROContent {
            inner: BinRWContent::new(),
            read: Box::new(read),
            eof_read: Box::new(eof_read),
        }
    }
}

impl BinOps for BinROContent {
    fn read(&self, offset: usize, out: &mut [u8]) -> Result<usize, NegativeError<usize>> {
        let inner = &self.inner;
        if offset == 0 {
            *inner.content.lock() = (self.read)();
        }
        let count = inner.read(offset, out)?;
        if count == 0 {
            (self.eof_read)();
        }
        Ok(count)
    }

    fn write(&self, _offset: usize, _in: &[u8]) -> Result<usize, NegativeError<usize>> {
        Err(NegativeError::EINVAL())
    }
}

pub struct BinFile<Ops> {
    // SAFETY: We need to pin FileInner as CBinAttribute contains a pointer to it.
    inner: Pin<Box<FileInner>>,
    _phantom: PhantomData<Ops>,
}
unsafe impl<Ops: Send> Send for BinFile<Ops> {}

impl<Ops> BinFile<Ops>
where
    Ops: 'static + BinOps,
{
    #[inline]
    pub fn name(&self) -> &str {
        // It was originally a String, so it can be converted to &str for sure.
        self.inner.name.to_str().unwrap()
    }

    pub fn new(
        parent: &mut Folder,
        name: &str,
        mode: FsMode,
        max_size: usize,
        ops: Ops,
    ) -> Result<BinFile<Ops>, Error> {
        #[cexport]
        fn read(
            _file: &mut CFile,
            _c_kobj: &mut CKObj,
            attr: *const UnsafeCell<_CBinAttribute>,
            // We need to use an FFI type that will turn into "char *" rather than "unsigned char*"
            // or "signed char *", otherwise CFI will get upset as that function will be used for
            // indirect calls by the sysfs infrastructure.
            out: *mut c_realchar,
            offset: c_longlong,
            count: usize,
        ) -> Result<usize, NegativeError<usize>> {
            let inner = unsafe { FileInner::from_attr(attr) };
            let out = unsafe { core::slice::from_raw_parts_mut(out as *mut u8, count) };
            if offset < 0 {
                let offset: isize = offset.try_into().unwrap();
                Err(NegativeError::new(offset))
            } else {
                let offset: usize = offset.try_into().unwrap();
                inner.ops.read(offset, out)
            }
        }

        #[cexport]
        fn write(
            _file: &mut CFile,
            _c_kobj: &mut CKObj,
            attr: *const UnsafeCell<_CBinAttribute>,
            in_: *mut c_realchar,
            offset: c_longlong,
            count: usize,
        ) -> Result<usize, NegativeError<usize>> {
            let inner = unsafe { FileInner::from_attr(attr) };
            let in_ = unsafe { core::slice::from_raw_parts_mut(in_ as *mut u8, count) };
            if offset < 0 {
                let offset: isize = offset.try_into().unwrap();
                Err(NegativeError::new(offset))
            } else {
                let offset: usize = offset.try_into().unwrap();
                inner.ops.write(offset, in_)
            }
        }

        #[cfunc]
        unsafe fn init_bin_attribute(
            attr: *mut _CBinAttribute,
            name: &CStr,
            mode: FsMode,
            read: *mut c_void,
            write: *mut c_void,
        ) {
            r#"
            #include <linux/sysfs.h>
            "#;

            r#"
            sysfs_bin_attr_init(attr);
            attr->attr.name = name;
            attr->attr.mode = mode;
            attr->read = read;
            attr->write = write;
            "#
        }

        let name = CString::new(name)
            .map_err(|err| error!("Could not convert file name to CString: {err}"))?;

        let inner = Box::into_pin(Box::new(FileInner {
            // Allocate the CBinAttribute in a pinned box, and only then initialize it properly.
            c_bin_attr: CBinAttribute(UnsafeCell::new(
                unsafe { _CBinAttribute::new_stack(|_| Ok::<_, Infallible>(())) }.unwrap(),
            )),
            name,
            parent_kobj: Arc::clone(&parent.kobj),
            ops: Box::new(ops),
        }));
        unsafe {
            init_bin_attribute(
                inner.c_bin_attr.0.get(),
                inner.name.as_c_str(),
                mode,
                read as *mut c_void,
                write as *mut c_void,
            );
        }

        #[cfunc]
        unsafe fn create(
            kobj: *mut CKObj,
            attr: *mut _CBinAttribute,
            inner: *mut c_void,
            size: usize,
        ) -> Result<c_uint, NegativeError<c_uint>> {
            r#"
            #include <linux/sysfs.h>
            "#;

            r#"
            attr->size = size;
            attr->private = inner;
            return sysfs_create_bin_file(kobj, attr);
            "#
        }
        let ptr: *const FileInner = &*inner;
        unsafe {
            create(
                inner.parent_kobj.c_kobj(),
                inner.c_bin_attr.0.get(),
                ptr as *mut c_void,
                max_size,
            )
            .map_err(|err| error!("sysfs_create_bin_file() failed: {err}"))?;
        }

        Ok(BinFile {
            inner,
            _phantom: PhantomData,
        })
    }

    pub fn ops(&self) -> &Ops {
        let ops = &*self.inner.ops;
        let ops = ops as &dyn Any;
        ops.downcast_ref().unwrap()
    }
}

impl<Ops> Drop for BinFile<Ops> {
    fn drop(&mut self) {
        #[cfunc]
        unsafe fn remove(kobj: *mut CKObj, attr: *mut _CBinAttribute) {
            r#"
            #include <linux/sysfs.h>
            "#;

            r#"
            return sysfs_remove_bin_file(kobj, attr);
            "#
        }

        unsafe {
            remove(
                self.inner.parent_kobj.c_kobj(),
                self.inner.c_bin_attr.0.get(),
            )
        }
    }
}
