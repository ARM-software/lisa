/* SPDX-License-Identifier: GPL-2.0 */

use alloc::boxed::Box;
use core::{
    cell::{Cell, UnsafeCell},
    ffi::c_void,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    pin::Pin,
};

use crate::{
    inlinec::{cfunc, opaque_type},
    runtime::kbox::KBox,
};

pub trait LockGuard<T>
where
    Self: Deref<Target = T>,
{
}

pub trait Lock<T>
where
    Self: Sync,
{
    type Guard<'a>: LockGuard<T>
    where
        Self: 'a;
    fn lock(&self) -> Self::Guard<'_>;

    #[inline]
    fn with_lock<U, F: FnOnce(&T) -> U>(&self, f: F) -> U {
        f(self.lock().deref())
    }
}

#[allow(private_bounds)]
pub trait LockMut<T>: for<'a> _LockMut<'a, T> {}

// TODO: We need to share the 'a lifetime between the Guard<'a> type and Self: 'a. Unfortunately,
// there seems to be no direct way of doing that since a for<> lifetime only affects what comes
// immediately after it. As a workaround, we can just hide that in a trait, and then use
// for<'a>_LockMut<'a, T> in the user-exposed trait.
trait _LockMut<'a, T>
where
    <Self as Lock<T>>::Guard<'a>: DerefMut,
    Self: 'a + Lock<T>,
{
}

opaque_type!(struct CSpinLock, "spinlock_t", "linux/spinlock.h");

pub struct SpinLock<T> {
    data: UnsafeCell<T>,
    // The Rust For Linux binding pins the spinlock binding, so do the same here to avoid any
    // problems.
    c_lock: Pin<KBox<UnsafeCell<CSpinLock>>>,
}

impl<T> SpinLock<T> {
    #[inline]
    pub fn new(x: T) -> Self {
        #[cfunc]
        fn spinlock_alloc() -> Pin<KBox<UnsafeCell<CSpinLock>>> {
            r#"
            #include <linux/spinlock.h>
            #include <linux/slab.h>
            "#;

            r#"
            spinlock_t *lock = kzalloc(sizeof(spinlock_t), GFP_KERNEL);
            if (lock) {
                spin_lock_init(lock);
            }
            return lock;
            "#
        }
        let c_lock = spinlock_alloc();
        SpinLock {
            c_lock,
            data: x.into(),
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

impl<T> Drop for SpinLock<T> {
    #[inline]
    fn drop(&mut self) {
        if self.is_locked() {
            panic!("Attempted to drop a currently-locked spinlock");
        }
    }
}

pub struct SpinLockGuard<'a, T> {
    lock: &'a SpinLock<T>,
    flags: u64,
}

impl<T> SpinLockGuard<'_, T> {
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

impl<T> Deref for SpinLockGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.get_mut()
    }
}

impl<T> DerefMut for SpinLockGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
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

impl<T> LockGuard<T> for SpinLockGuard<'_, T> {}

unsafe impl<T: Send> Sync for SpinLock<T> {}
impl<T: Send> Lock<T> for SpinLock<T> {
    type Guard<'a>
        = SpinLockGuard<'a, T>
    where
        Self: 'a;

    #[inline]
    fn lock(&self) -> Self::Guard<'_> {
        #[cfunc]
        fn spinlock_lock(lock: &UnsafeCell<CSpinLock>) -> u64 {
            "#include <linux/spinlock.h>";

            r#"
            unsigned long flags;
            spin_lock_irqsave(lock, flags);
            return (uint64_t)flags;
            "#
        }
        let flags = spinlock_lock(&self.c_lock);

        SpinLockGuard { lock: self, flags }
    }
}

impl<'a, T: 'a + Send> _LockMut<'a, T> for SpinLock<T> {}

opaque_type!(pub struct CMutex, "struct mutex", "linux/mutex.h");

enum AllocatedCMutex {
    KBox(Pin<KBox<UnsafeCell<CMutex>>>),
    // FIXME: introduce #[cstatic] to allow defining a static global with C code, the same way we
    // have #[cfunc]. This would be restricted to types that are FfiType + IntoFfi<FfiType=Self> +
    // FromFfi<FfiType=Self>

    // Mutexes in static variables have to be defined in C code using the DEFINE_MUTEX() macro.
    // Unfortunately, that means we can't get their address directly in Rust, as that would require
    // having an equivalent of #[cfunc] for other values than functions, which would be doable but
    // extra work. Instead, we can just make a C function that declares a static mutex inside it
    // and returns its address.
    Static(fn() -> Pin<&'static UnsafeCell<CMutex>>),
}

impl AllocatedCMutex {
    #[inline]
    fn as_pin_ref(&self) -> Pin<&UnsafeCell<CMutex>> {
        match self {
            AllocatedCMutex::KBox(c_mutex) => c_mutex.as_ref(),
            AllocatedCMutex::Static(f) => f(),
        }
    }
}

macro_rules! new_static_mutex {
    ($vis:vis $name:ident, $ty:ty, $data:expr) => {
        $vis static $name: $crate::runtime::sync::Mutex<$ty> =
            $crate::runtime::sync::Mutex::__internal_from_ref($data, {
                #[$crate::inlinec::cfunc]
                fn get_static_mutex() -> ::core::pin::Pin<
                    &'static ::core::cell::UnsafeCell<$crate::runtime::sync::CMutex>,
                > {
                    "#include <linux/mutex.h>";

                    r#"
                    static DEFINE_MUTEX(mutex);
                    return &mutex;
                    "#
                }
                get_static_mutex
            });
    };
}
pub(crate) use new_static_mutex;

pub struct Mutex<T> {
    data: UnsafeCell<T>,
    // struct mutex contains a list_head, so it must be pinned.
    c_mutex: AllocatedCMutex,
}

impl<T> Mutex<T> {
    #[inline]
    pub fn new(x: T) -> Self {
        #[cfunc]
        fn mutex_alloc() -> Pin<KBox<UnsafeCell<CMutex>>> {
            r#"
            #include <linux/mutex.h>
            #include <linux/slab.h>
            "#;

            r#"
            struct mutex *mutex = kzalloc(sizeof(struct mutex), GFP_KERNEL);
            if (mutex) {
                mutex_init(mutex);
            }
            return mutex;
            "#
        }
        let c_mutex = mutex_alloc();
        Mutex {
            c_mutex: AllocatedCMutex::KBox(c_mutex),
            data: x.into(),
        }
    }

    #[inline]
    pub const fn __internal_from_ref(x: T, f: fn() -> Pin<&'static UnsafeCell<CMutex>>) -> Self {
        Mutex {
            c_mutex: AllocatedCMutex::Static(f),
            data: UnsafeCell::new(x),
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

impl<T> Drop for Mutex<T> {
    #[inline]
    fn drop(&mut self) {
        if self.is_locked() {
            panic!("Attempted to drop a currently-locked mutex");
        }
    }
}

pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> MutexGuard<'_, T> {
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

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.get_mut()
    }
}

impl<T> DerefMut for MutexGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T> Drop for MutexGuard<'_, T> {
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

impl<T> LockGuard<T> for MutexGuard<'_, T> {}

unsafe impl<T: Send> Sync for Mutex<T> {}
impl<T: Send> Lock<T> for Mutex<T> {
    type Guard<'a>
        = MutexGuard<'a, T>
    where
        Self: 'a;

    #[inline]
    fn lock(&self) -> Self::Guard<'_> {
        #[cfunc]
        fn mutex_lock(mutex: &UnsafeCell<CMutex>) {
            "#include <linux/mutex.h>";

            r#"
            return mutex_lock(mutex);
            "#
        }
        mutex_lock(&self.c_mutex.as_pin_ref());

        MutexGuard { mutex: self }
    }
}

impl<'a, T: 'a + Send> _LockMut<'a, T> for Mutex<T> {}

pub struct Rcu<T> {
    // This pointer is actually a Box in disguise obtained with Box::into_raw(). We keep it as a
    // raw pointer so that we can use C's rcu_assign_pointer() on it.
    //
    // Use a level of indirection so that Rcu<T> does not need to dish out references to inside
    // itself. This avoids the need for pinning it, making it much more ergonomic. Since an Rcu is
    // typically mostly read and not often modified, the box allocation is not unreasonable.
    data: Cell<*const T>,

    // Mutex used to protect the writers.
    writer_mutex: Mutex<()>,
}

impl<T> Rcu<T> {
    pub fn new(data: T) -> Self {
        Rcu {
            data: Cell::new(Box::into_raw(Box::new(data))),
            writer_mutex: Mutex::new(()),
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

pub struct RcuGuard<'a, T> {
    data: *const T,
    _phantom: PhantomData<&'a T>,
}

impl<T> LockGuard<T> for RcuGuard<'_, T> {}

impl<T> Deref for RcuGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: we are guaranteed that the pointer we obtained will be valid for the lifetime of
        // the guard.
        unsafe { self.data.as_ref().unwrap() }
    }
}

impl<T> Drop for RcuGuard<'_, T> {
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

unsafe impl<T: Send> Sync for Rcu<T> {}
impl<T: Send> Lock<T> for Rcu<T> {
    type Guard<'a>
        = RcuGuard<'a, T>
    where
        Self: 'a;

    #[inline]
    fn lock(&self) -> Self::Guard<'_> {
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
        }
    }
}
