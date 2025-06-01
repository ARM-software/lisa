/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, ffi::CString, sync::Arc};
use core::{
    cell::{Cell, UnsafeCell},
    ffi::{CStr, c_uint, c_void},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    pin::Pin,
};

use lisakmod_macros::inlinec::{cfunc, opaque_type};

use crate::{
    mem::{NotSend, impl_from_contained},
    runtime::kbox::KernelKBox,
};

opaque_type!(pub struct CLockClassKey, "struct lock_class_key", "linux/lockdep.h");

// Inner type that we can make Sync, as UnsafeCell is !Sync
struct InnerLockdepClass {
    c_key: UnsafeCell<CLockClassKey>,
    name: CString,
}

#[derive(Clone)]
enum AllocatedLockdepClass {
    // SAFETY: InnerLockdepClass needs to be pinned, as CLockClassKey is part of some linked lists.
    Dyn(Pin<Arc<InnerLockdepClass>>),
    Static {
        key: &'static CLockClassKey,
        name: &'static CStr,
    },
}

unsafe impl Send for AllocatedLockdepClass {}
unsafe impl Sync for AllocatedLockdepClass {}

#[derive(Clone)]
pub struct LockdepClass {
    inner: AllocatedLockdepClass,
}

impl LockdepClass {
    pub fn new(name: &str) -> LockdepClass {
        #[cfunc]
        unsafe fn lockdep_register_key(key: *mut CLockClassKey) {
            r#"
            #include <linux/lockdep.h>
            "#;

            r#"
            lockdep_register_key(key);
            "#
        }

        let c_key = UnsafeCell::new(
            unsafe { CLockClassKey::new_stack(|_| Ok::<_, ()>(())) }
                .expect("Could not allocate lock_class_key"),
        );

        let name = CString::new(name).expect("Cannot convert lockdep class name to C string");
        let this = LockdepClass {
            inner: AllocatedLockdepClass::Dyn(Arc::pin(InnerLockdepClass { c_key, name })),
        };
        unsafe {
            lockdep_register_key(this.get_key());
        }
        this
    }

    pub const fn __internal_from_ref(
        key: &'static CLockClassKey,
        name: &'static CStr,
    ) -> LockdepClass {
        LockdepClass {
            inner: AllocatedLockdepClass::Static { key, name },
        }
    }

    fn get_key(&self) -> *mut CLockClassKey {
        match &self.inner {
            AllocatedLockdepClass::Dyn(inner) => UnsafeCell::get(&inner.c_key),
            AllocatedLockdepClass::Static { key, .. } => *key as *const _ as *mut _,
        }
    }

    fn c_name(&self) -> &CStr {
        match &self.inner {
            AllocatedLockdepClass::Dyn(inner) => inner.name.as_c_str(),
            AllocatedLockdepClass::Static { name, .. } => name,
        }
    }
}

impl Drop for LockdepClass {
    fn drop(&mut self) {
        #[cfunc]
        unsafe fn lockdep_unregister_key(key: *mut CLockClassKey) {
            r#"
            #include <linux/lockdep.h>
            "#;

            r#"
            lockdep_unregister_key(key);
            "#
        }

        match self.inner {
            AllocatedLockdepClass::Dyn(_) => unsafe {
                lockdep_unregister_key(self.get_key());
            },
            AllocatedLockdepClass::Static { .. } => {}
        }
    }
}

macro_rules! new_static_lockdep_class {
    ($vis:vis $name:ident) => {
        $vis static $name: $crate::runtime::sync::LockdepClass = $crate::runtime::sync::LockdepClass::__internal_from_ref(
            {
                #[::lisakmod_macros::inlinec::cstatic]
                static STATIC_LOCKDEP_CLASS_KEY: $crate::runtime::sync::CLockClassKey = (
                    "#include <linux/mutex.h>",
                    "struct lock_class_key STATIC_VARIABLE;"
                );
                unsafe {
                    &STATIC_LOCKDEP_CLASS_KEY
                }
            },
            match ::core::ffi::CStr::from_bytes_with_nul(
                ::core::concat!(::core::stringify!($name),"\0").as_bytes()
            ) {
                Ok(s) => s,
                Err(err) => unreachable!(),
            }
        );
    };
}
#[allow(unused_imports)]
pub(crate) use new_static_lockdep_class;

pub trait LockdepSubclass: Default {
    fn to_u32(&self) -> u32;
}

impl LockdepSubclass for u32 {
    #[inline]
    fn to_u32(&self) -> u32 {
        *self
    }
}

impl LockdepSubclass for () {
    #[inline]
    fn to_u32(&self) -> u32 {
        0
    }
}

type DefaultSubclass = ();

pub trait LockGuard<'guard>
where
    Self: Deref<Target = Self::T>,
{
    type T;
    type Subclass;
}

pub trait Lock
where
    Self: Sync,
{
    type Subclass: LockdepSubclass;
    type T;
    type Guard<'a>: LockGuard<'a, Subclass = Self::Subclass, T = Self::T>
    where
        Self: 'a;
    #[inline]
    fn lock(&self) -> Self::Guard<'_> {
        self.lock_nested(<Self::Subclass as Default>::default())
    }
    fn lock_nested(&self, subclass: Self::Subclass) -> Self::Guard<'_>;

    #[inline]
    fn with_lock<U, F: FnOnce(&Self::T) -> U>(&self, f: F) -> U {
        f(self.lock().deref())
    }
}

#[allow(private_bounds)]
pub trait LockMut: for<'a> _LockMut<'a> {}

// TODO: We need to share the 'a lifetime between the Guard<'a> type and Self: 'a. Unfortunately,
// there seems to be no direct way of doing that since a for<> lifetime only affects what comes
// immediately after it. As a workaround, we can just hide that in a trait, and then use
// for<'a>_LockMut<'a> in the user-exposed trait.
trait _LockMut<'a>
where
    <Self as Lock>::Guard<'a>: DerefMut,
    Self: 'a + Lock,
{
}

pub struct PinnedGuard<'a, L>
where
    L: Lock + 'a,
{
    guard: <L as Lock>::Guard<'a>,
}

impl<'a, L> PinnedGuard<'a, L>
where
    L: Lock + 'a,
{
    #[inline]
    pub fn as_ref(&self) -> Pin<&<L as Lock>::T> {
        // SAFETY: The guard we have can only come from a pinned lock, and that lock is hidden
        // internally so that the only way to lock it is via a pinned reference. Since we do not
        // allow unlocking that lock via a regular reference, it is not possible to circumvent the
        // pinning guarantee and get a plain &mut T with Pin::get_ref(&Lock<T>).
        unsafe { Pin::new_unchecked(self.guard.deref()) }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Pin<&mut <L as Lock>::T>
    where
        <L as Lock>::Guard<'a>: DerefMut,
    {
        // SAFETY: See PinnedGuard::as_ref()
        unsafe { Pin::new_unchecked(self.guard.deref_mut()) }
    }
}

/// # Safety
///
/// This trait can only be implemented for types that do not allow creating multiple locks that can
/// all be used to get an ``&mut T``. For example, ``&Mutex<T>`` could not implement this trait, as
/// it is possible to just copy it, and such reference can then be used to gain access to a ``&mut
/// T``.
pub unsafe trait PinnableLock: Lock {}
unsafe impl<T: Send, Subclass: LockdepSubclass> PinnableLock for Mutex<T, Subclass> {}
unsafe impl<T: Send, Subclass: LockdepSubclass> PinnableLock for SpinLock<T, Subclass> {}

pub struct PinnedLock<L> {
    inner: L,
}

impl_from_contained!((L,) PinnedLock<L>, inner: L);

impl<L> PinnedLock<L>
where
    L: PinnableLock,
{
    #[inline]
    pub fn new(lock: L) -> PinnedLock<L> {
        PinnedLock { inner: lock }
    }

    #[inline]
    pub fn lock<'a>(self: Pin<&'a Self>) -> PinnedGuard<'a, L> {
        self.lock_nested(<<L as Lock>::Subclass as Default>::default())
    }

    #[inline]
    pub fn lock_nested<'a>(
        self: Pin<&'a Self>,
        subclass: <L as Lock>::Subclass,
    ) -> PinnedGuard<'a, L> {
        // SAFETY: As per PinnableLock guarantees, there is nothing else that can leak an &mut T,
        // so we can safely create a PinnedGuard that will allow getting an Pin<&mut T>
        PinnedGuard {
            guard: self.get_ref().inner.lock_nested(subclass),
        }
    }
}

opaque_type!(struct CSpinLock, "spinlock_t", "linux/spinlock.h");

pub struct SpinLock<T, Subclass = DefaultSubclass> {
    data: UnsafeCell<T>,
    // The Rust For Linux binding pins the spinlock binding, so do the same here to avoid any
    // problems.
    c_lock: Pin<KernelKBox<UnsafeCell<CSpinLock>>>,
    // SAFETY: This needs to be freed after the CSpinLock that makes use of it as part of its
    // lockdep map.
    lockdep_class: LockdepClass,
    _phantom: PhantomData<Subclass>,
}

impl<T, Subclass> SpinLock<T, Subclass> {
    #[inline]
    pub fn new(x: T, lockdep_class: LockdepClass) -> Self {
        #[cfunc]
        fn spinlock_alloc(
            lockdep_key: *mut CLockClassKey,
        ) -> Pin<KernelKBox<UnsafeCell<CSpinLock>>> {
            r#"
            #include <linux/spinlock.h>
            #include <linux/slab.h>
            #include <linux/lockdep.h>
            "#;

            r#"
            spinlock_t *lock = kzalloc(sizeof(spinlock_t), GFP_KERNEL);
            if (lock) {
                spin_lock_init(lock);
                lockdep_set_class(lock, lockdep_key);
            }
            return lock;
            "#
        }
        let c_lock = spinlock_alloc(lockdep_class.get_key());
        SpinLock {
            c_lock,
            lockdep_class,
            data: x.into(),
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn is_locked(&self) -> bool {
        #[cfunc]
        fn lock_is_locked(lock: &UnsafeCell<CSpinLock>) -> bool {
            "#include <linux/spinlock.h>";

            r#"
            return spin_is_locked(lock);
            "#
        }
        lock_is_locked(&self.c_lock)
    }
}

impl<T, Subclass> Drop for SpinLock<T, Subclass> {
    #[inline]
    fn drop(&mut self) {
        if self.is_locked() {
            panic!("Attempted to drop a currently-locked spinlock");
        }
    }
}

pub struct SpinLockGuard<'a, T, Subclass> {
    lock: &'a SpinLock<T, Subclass>,
    flags: u64,
    // The kernel API forces us to unlock a lock in the same task that locked it.
    _notsend: NotSend,
}

impl<T, Subclass> SpinLockGuard<'_, T, Subclass> {
    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn get_mut(&self) -> &mut T {
        let ptr = self.lock.data.get();
        // SAFETY: If SpinLockGuard<T> exists, then the protected T data cannot currently be used
        // by any other piece of code since we acquired the lock successfully. Since T is behind an
        // UnsafeCell, it is sound to access the data as long as there is no race.
        //
        // Note that this only stands because the kernel spinlocks are not recursive. Such
        // recursive call would be allowed by the borrow checker, but will end up in a deadlock so
        // we will never _actually_ end up creating two &mut T pointing at the same piece of data.
        unsafe { &mut *ptr }
    }
}

impl<T, Subclass> Deref for SpinLockGuard<'_, T, Subclass> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.get_mut()
    }
}

impl<T, Subclass> DerefMut for SpinLockGuard<'_, T, Subclass> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T, Subclass> Drop for SpinLockGuard<'_, T, Subclass> {
    #[inline]
    fn drop(&mut self) {
        #[cfunc]
        fn spinlock_unlock(lock: &UnsafeCell<CSpinLock>, flags: u64) {
            "#include <linux/spinlock.h>";

            r#"
            spin_unlock_irqrestore(lock, flags);
            "#
        }
        spinlock_unlock(&self.lock.c_lock, self.flags);
    }
}

impl<'guard, T, Subclass> LockGuard<'guard> for SpinLockGuard<'guard, T, Subclass> {
    type T = T;
    type Subclass = Subclass;
}

unsafe impl<T: Send, Subclass> Sync for SpinLock<T, Subclass> {}
unsafe impl<T: Send, Subclass> Send for SpinLock<T, Subclass> {}

impl<T, Subclass> Lock for SpinLock<T, Subclass>
where
    T: Send,
    Subclass: LockdepSubclass,
{
    type Subclass = Subclass;
    type T = T;
    type Guard<'a>
        = SpinLockGuard<'a, Self::T, Self::Subclass>
    where
        Self: 'a;

    #[inline]
    fn lock_nested(&self, subclass: Self::Subclass) -> Self::Guard<'_> {
        #[cfunc]
        fn spinlock_lock(lock: &UnsafeCell<CSpinLock>, subclass: c_uint) -> u64 {
            "#include <linux/spinlock.h>";

            r#"
            unsigned long flags;
            spin_lock_irqsave_nested(lock, flags, subclass);
            return (uint64_t)flags;
            "#
        }
        let flags = spinlock_lock(&self.c_lock, subclass.to_u32());

        SpinLockGuard {
            lock: self,
            flags,
            _notsend: NotSend::new(),
        }
    }
}

impl<'a, T, Subclass> _LockMut<'a> for SpinLock<T, Subclass>
where
    T: 'a + Send,
    Subclass: 'a + LockdepSubclass,
{
}

opaque_type!(pub struct CMutex, "struct mutex", "linux/mutex.h");

enum AllocatedCMutex {
    KBox {
        c_mutex: Pin<KernelKBox<UnsafeCell<CMutex>>>,
        lockdep_class: LockdepClass,
    },
    Static {
        c_mutex: Pin<&'static UnsafeCell<CMutex>>,
    },
}

impl AllocatedCMutex {
    #[inline]
    fn as_pin_ref(&self) -> Pin<&UnsafeCell<CMutex>> {
        match self {
            AllocatedCMutex::KBox { c_mutex, .. } => c_mutex.as_ref(),
            AllocatedCMutex::Static { c_mutex, .. } => *c_mutex,
        }
    }
}

macro_rules! new_static_mutex {
    ($vis:vis $name:ident, $ty:ty, $data:expr) => {
        $vis static $name: $crate::runtime::sync::Mutex<$ty> = $crate::runtime::sync::Mutex::__internal_from_ref(
            $data,
            {
                #[::lisakmod_macros::inlinec::cstatic]
                static STATIC_MUTEX: $crate::runtime::sync::CMutex = (
                    "#include <linux/mutex.h>",
                    "DEFINE_MUTEX(STATIC_VARIABLE);"
                );
                unsafe {
                    ::core::pin::Pin::new_unchecked(
                        // SAFETY: Similarly to UnsafeCell::from_mut(), we convert &T to &UnsafeCell<T>.
                        // UnsafeCell::from_mut() requires a &mut, which is not possible for a
                        // static variable in Rust 2024 edition however. Requiring an &mut provides
                        // the assurance that no other reference is floating around, which is
                        // necessary for the operation to be sound. Here, it is guaranteed as
                        // nothing else has access to our static variable and we do not leak any
                        // &T.
                        //
                        // Also, UnsafeCell<T> and T have the same memory layout, and no &T or &mut
                        // T can escape here as STATIC_MUTEX is only reachable here.
                        &*( &STATIC_MUTEX as *const _ as *const ::core::cell::UnsafeCell<_>)
                    )
                }
            }
        );
    };
}
#[allow(unused_imports)]
pub(crate) use new_static_mutex;

pub struct Mutex<T, Subclass = DefaultSubclass> {
    data: UnsafeCell<T>,
    // struct mutex contains a list_head, so it must be pinned.
    c_mutex: AllocatedCMutex,
    _phantom: PhantomData<Subclass>,
}

impl<T, Subclass> Mutex<T, Subclass> {
    #[inline]
    pub fn new(x: T, lockdep_class: LockdepClass) -> Self {
        #[cfunc]
        fn mutex_alloc(
            lockdep_key: *mut CLockClassKey,
            name: &CStr,
        ) -> Pin<KernelKBox<UnsafeCell<CMutex>>> {
            r#"
            #include <linux/mutex.h>
            #include <linux/slab.h>
            #include <linux/lockdep.h>
            "#;

            r#"
            struct mutex *mutex = kzalloc(sizeof(struct mutex), GFP_KERNEL);
            if (mutex) {
                mutex_init(mutex);
                lockdep_set_class_and_name(mutex, lockdep_key, name);
            }
            return mutex;
            "#
        }
        let c_mutex = mutex_alloc(lockdep_class.get_key(), lockdep_class.c_name());
        Mutex {
            c_mutex: AllocatedCMutex::KBox {
                c_mutex,
                lockdep_class,
            },
            data: x.into(),
            _phantom: PhantomData,
        }
    }

    // FIXME: this can only be done before the lock is locked for the first time, so disable this
    // code for now as there is no immediate need.
    // pub fn set_lockdep_class(&mut self, lockdep_class: LockdepClass) {
    // assert!(
    // !self.is_locked(),
    // "Changing lockdep class while lock is taken will lead to a lock imbalance warning"
    // );
    // #[cfunc]
    // fn set_lockdep_class(
    // mutex: Pin<&UnsafeCell<CMutex>>,
    // lockdep_key: *mut CLockClassKey,
    // name: &CStr,
    // ) {
    // r#"
    // #include <linux/mutex.h>
    // #include <linux/slab.h>
    // #include <linux/lockdep.h>
    // "#;

    // r#"
    // lockdep_set_class_and_name(mutex, lockdep_key, name);
    // "#
    // }
    // set_lockdep_class(
    // self.c_mutex.as_pin_ref(),
    // lockdep_class.get_key(),
    // lockdep_class.c_name(),
    // );
    // }

    // pub fn disable_lockdep(&mut self) {
    // #[cfunc]
    // fn lockdep_set_novalidate_class(mutex: Pin<&UnsafeCell<CMutex>>) {
    // r#"
    // #include <linux/mutex.h>
    // #include <linux/slab.h>
    // #include <linux/lockdep.h>
    // "#;

    // r#"
    // lockdep_set_novalidate_class(mutex);
    // "#
    // }
    // lockdep_set_novalidate_class(self.c_mutex.as_pin_ref());
    // }

    #[inline]
    pub const fn __internal_from_ref(x: T, c_mutex: Pin<&'static UnsafeCell<CMutex>>) -> Self {
        Mutex {
            c_mutex: AllocatedCMutex::Static { c_mutex },
            data: UnsafeCell::new(x),
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn is_locked(&self) -> bool {
        #[cfunc]
        fn mutex_is_locked(lock: &UnsafeCell<CMutex>) -> bool {
            "#include <linux/mutex.h>";

            r#"
            return mutex_is_locked(lock);
            "#
        }
        mutex_is_locked(&self.c_mutex.as_pin_ref())
    }
}

impl_from_contained!((T, Subclass) Mutex<T, Subclass>, data: T);

impl<T, Subclass> Drop for Mutex<T, Subclass> {
    #[inline]
    fn drop(&mut self) {
        if self.is_locked() {
            panic!("Attempted to drop a currently-locked mutex");
        }
    }
}

pub struct MutexGuard<'a, T, Subclass> {
    mutex: &'a Mutex<T, Subclass>,
    // The kernel API forces us to unlock a lock in the same task that locked it.
    _notsend: NotSend,
}

impl<T, Subclass> MutexGuard<'_, T, Subclass> {
    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn get_mut(&self) -> &mut T {
        let ptr = self.mutex.data.get();
        // SAFETY: If MutexGuard<T> exists, then the protected T data cannot currently be used
        // by any other piece of code since we acquired the lock successfully. Since T is behind an
        // UnsafeCell, it is sound to access the data as long as there is no race.
        //
        // Note that this only stands because the kernel mutex are not recursive. Such
        // recursive call would be allowed by the borrow checker, but will end up in a deadlock so
        // we will never _actually_ end up creating two &mut T pointing at the same piece of data.
        unsafe { &mut *ptr }
    }
}

impl<T, Subclass> Deref for MutexGuard<'_, T, Subclass> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.get_mut()
    }
}

impl<T, Subclass> DerefMut for MutexGuard<'_, T, Subclass> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T, Subclass> Drop for MutexGuard<'_, T, Subclass> {
    #[inline]
    fn drop(&mut self) {
        #[cfunc]
        fn mutex_unlock(mutex: &UnsafeCell<CMutex>) {
            "#include <linux/mutex.h>";

            r#"
            mutex_unlock(mutex);
            "#
        }
        mutex_unlock(&self.mutex.c_mutex.as_pin_ref());
    }
}

impl<'guard, T, Subclass> LockGuard<'guard> for MutexGuard<'guard, T, Subclass> {
    type T = T;
    type Subclass = Subclass;
}

unsafe impl<T: Send, Subclass> Send for Mutex<T, Subclass> {}
unsafe impl<T: Send, Subclass> Sync for Mutex<T, Subclass> {}

impl<T: Send, Subclass> Lock for Mutex<T, Subclass>
where
    Subclass: LockdepSubclass,
{
    type Subclass = Subclass;
    type T = T;
    type Guard<'a>
        = MutexGuard<'a, Self::T, Self::Subclass>
    where
        Self: 'a;

    #[inline]
    fn lock_nested(&self, subclass: Self::Subclass) -> Self::Guard<'_> {
        #[cfunc]
        fn mutex_lock_nested(mutex: &UnsafeCell<CMutex>, subclass: c_uint) {
            "#include <linux/mutex.h>";

            r#"
            return mutex_lock_nested(mutex, subclass);
            "#
        }
        mutex_lock_nested(&self.c_mutex.as_pin_ref(), subclass.to_u32());

        MutexGuard {
            mutex: self,
            _notsend: NotSend::new(),
        }
    }
}

impl<'a, T, Subclass> _LockMut<'a> for Mutex<T, Subclass>
where
    T: 'a + Send,
    Subclass: 'a + LockdepSubclass,
{
}

pub struct Rcu<T, Subclass = DefaultSubclass> {
    // This pointer is actually a Box in disguise obtained with Box::into_raw(). We keep it as a
    // raw pointer so that we can use C's rcu_assign_pointer() on it.
    //
    // Use a level of indirection so that Rcu<T> does not need to dish out references to inside
    // itself. This avoids the need for pinning it, making it much more ergonomic. Since an Rcu is
    // typically mostly read and not often modified, the box allocation is not unreasonable.
    data: Cell<*const T>,

    // Mutex used to protect the writers.
    writer_mutex: Mutex<(), Subclass>,
}

impl<T, Subclass> Rcu<T, Subclass>
where
    Subclass: LockdepSubclass,
{
    pub fn new(data: T, lockdep_class: LockdepClass) -> Self {
        Rcu {
            data: Cell::new(Box::into_raw(Box::new(data))),
            writer_mutex: Mutex::new((), lockdep_class),
        }
    }

    pub fn update(&self, data: T) {
        #[cfunc]
        unsafe fn rcu_assign(rcu: *mut *const c_void, new_ptr: *const c_void) {
            "#include <linux/rcupdate.h>";

            r#"
            rcu_assign_pointer((*rcu), new_ptr);
            "#
        }

        #[cfunc]
        fn rcu_synchronize() {
            "#include <linux/rcupdate.h>";

            r#"
            synchronize_rcu();
            "#
        }

        // Allocate the data before we take the writer lock to avoid taking it for longer than
        // necessary.
        let new_data = Box::into_raw(Box::new(data));

        let writer_guard = self.writer_mutex.lock();

        // Get the old pointer before we overwrite it. It is safe to take it without
        // rcu_dereference() as we are holding the lock. This is equivalent to using
        // rcu_dereference_protected().
        let old_data: *const T = self.data.get();

        // SAFETY: It is safe to modify self.data here as:
        // * We locked self.writer_mutex, so no-one else is attempting to modify it.
        // * It is in a Cell<>, so interior mutability is allowed.
        // * The update is atomically done by rcu_assign_pointer(), so no reader will see a
        //   torn value.
        unsafe {
            rcu_assign(
                self.data.as_ptr() as *mut *const c_void,
                new_data as *const c_void,
            );
        }

        // Once we used rcu_assign_pointer(), we can drop the writer lock safely, wait for the
        // grace period and drop the old data.
        drop(writer_guard);

        rcu_synchronize();

        // SAFETY: The Box<T> is not in use anymore by anyone else after the call to
        // synchronize_rcu(), so we can safely re-materialize it to drop it.
        drop(unsafe { Box::from_raw(old_data as *mut T) });
    }
}

pub struct RcuGuard<'a, T, Subclass> {
    data: *const T,
    _phantom: PhantomData<(&'a T, Subclass)>,
    // The kernel API forces us to unlock a lock in the same task that locked it.
    _notsend: NotSend,
}

impl<'guard, T, Subclass> LockGuard<'guard> for RcuGuard<'guard, T, Subclass> {
    type T = T;
    type Subclass = Subclass;
}

impl<T, Subclass> Deref for RcuGuard<'_, T, Subclass> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: we are guaranteed that the pointer we obtained will be valid for the lifetime of
        // the guard.
        unsafe { self.data.as_ref().unwrap() }
    }
}

impl<T, Subclass> Drop for RcuGuard<'_, T, Subclass> {
    fn drop(&mut self) {
        #[cfunc]
        fn rcu_unlock() {
            "#include <linux/rcupdate.h>";

            r#"
            rcu_read_unlock();
            "#
        }
        rcu_unlock();
    }
}

unsafe impl<T: Send, Subclass> Sync for Rcu<T, Subclass> {}
unsafe impl<T: Send, Subclass> Send for Rcu<T, Subclass> {}

impl<T, Subclass> Lock for Rcu<T, Subclass>
where
    T: Send,
    Subclass: LockdepSubclass,
{
    type Subclass = Subclass;
    type T = T;
    type Guard<'a>
        = RcuGuard<'a, Self::T, Self::Subclass>
    where
        Self: 'a;

    #[inline]
    fn lock_nested(&self, _subclass: Subclass) -> Self::Guard<'_> {
        #[cfunc]
        fn rcu_lock(ptr: *const c_void) -> *const c_void {
            "#include <linux/rcupdate.h>";

            r#"
            rcu_read_lock();
            return rcu_dereference(ptr);
            "#
        }
        // SAFETY: we guarantee that for the pointer we pass to RcuGuard<'a, T> will be valid for
        // the lifetime 'a of the guard, sine the RcuGuard::drop() runs rcu_read_unlock() and we
        // wait for synchronize_rcu() before dropping self.data
        let data = {
            let in_ptr = self.data.get() as *const c_void;
            let out_ptr = rcu_lock(in_ptr);
            assert_eq!(in_ptr, out_ptr);
            out_ptr as *const T
        };
        RcuGuard {
            data,
            _phantom: PhantomData,
            _notsend: NotSend::new(),
        }
    }
}
