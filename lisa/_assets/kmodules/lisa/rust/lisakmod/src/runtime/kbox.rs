/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    convert::{AsMut, AsRef},
    fmt,
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    ptr,
    ptr::NonNull,
};

use crate::runtime::alloc::{kfree, kmalloc};

pub struct KBox<T: ?Sized> {
    ptr: NonNull<T>,
}

impl<T: ?Sized> KBox<T> {
    #[inline]
    pub fn new(x: T) -> Self
    where
        T: Sized,
    {
        let layout = Layout::new::<T>();
        let ptr = kmalloc(layout);
        let ptr = ptr::NonNull::new(ptr as *mut T).expect("Allocation failed");
        unsafe {
            ptr::write(ptr.as_ptr(), x);
        }
        KBox { ptr }
    }

    #[inline]
    pub fn from_ptr(ptr: NonNull<T>) -> Self {
        KBox { ptr }
    }

    #[inline]
    pub fn as_ptr(&self) -> NonNull<T> {
        self.ptr
    }
}

impl<T: ?Sized> Drop for KBox<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.ptr.drop_in_place();
            kfree(self.ptr.as_ptr());
        }
    }
}

impl<T> From<T> for KBox<T> {
    #[inline]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

impl<T> From<NonNull<T>> for KBox<T> {
    #[inline]
    fn from(t: NonNull<T>) -> Self {
        Self::from_ptr(t)
    }
}

unsafe impl<T: ?Sized + Send> Send for KBox<T> {}
unsafe impl<T: ?Sized + Sync> Sync for KBox<T> {}

impl<T: ?Sized> Deref for KBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for KBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: ?Sized> AsRef<T> for KBox<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> AsMut<T> for KBox<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T: ?Sized> Borrow<T> for KBox<T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<T: ?Sized> BorrowMut<T> for KBox<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<T: Clone> Clone for KBox<T> {
    #[inline]
    fn clone(&self) -> Self {
        let x: T = <T as Clone>::clone(self.deref());
        Self::new(x)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for KBox<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<T: Default> Default for KBox<T> {
    #[inline]
    fn default() -> Self {
        KBox::new(<T as Default>::default())
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for KBox<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl<T: ?Sized + PartialEq> PartialEq for KBox<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: ?Sized + Eq> Eq for KBox<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for KBox<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Ord> Ord for KBox<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Hash> Hash for KBox<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized> fmt::Pointer for KBox<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T: ?Sized> Unpin for KBox<T> {}
