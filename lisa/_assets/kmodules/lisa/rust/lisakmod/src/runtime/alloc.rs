/* SPDX-License-Identifier: GPL-2.0 */

use alloc::alloc::handle_alloc_error;
use core::{
    alloc::{GlobalAlloc, Layout},
    ffi::c_void,
};

use lisakmod_macros::inlinec::{cconstant, cfunc};

use crate::runtime::printk::pr_err;

#[inline]
fn with_size<F: FnOnce(usize) -> *mut u8>(layout: Layout, f: F) -> *mut u8 {
    let minalign: usize =
        cconstant!("#include <linux/slab.h>", "ARCH_KMALLOC_MINALIGN").unwrap_or(1);

    let size = layout.size();
    let align = layout.align();
    // For sizes which are a power of two, the kmalloc() alignment is also guaranteed to be at
    // least the respective size.
    if (size.is_power_of_two() && align <= size) || (align <= minalign) {
        let ptr = f(layout.size());
        assert_eq!((ptr as usize % align), 0);
        ptr
    } else {
        // Do not panic as this would create UB
        pr_err!(
            "Rust: cannot allocate memory with alignment > {minalign} and a size {size} that is not a power of two"
        );
        core::ptr::null_mut()
    }
}

// TODO: Replace by Option::unwrap_or() if it becomes const
macro_rules! unwrap_or {
    ($expr:expr, $default:expr) => {
        match $expr {
            Some(x) => x,
            None => $default,
        }
    };
}

#[derive(core::marker::ConstParamTy, PartialEq, Eq)]
pub enum GFPFlags {
    Kernel = unwrap_or!(cconstant!("#include <linux/slab.h>", "GFP_KERNEL"), 0),
    Atomic = unwrap_or!(cconstant!("#include <linux/slab.h>", "GFP_ATOMIC"), 1),
}

/// This function is guaranteed to return the pointer given by the kernel's kmalloc() without
/// re-aligning it in any way. This makes it suitable to pass to kfree() without knowing the
/// original layout.
///
/// Note that any layout is admissible here, including if layout.size() == 0 unlike when going
/// through the GlobalAlloc API.
#[inline]
pub fn kmalloc(layout: Layout, flags: GFPFlags) -> *mut u8 {
    #[cfunc]
    fn alloc(size: usize, flags: u64) -> *mut u8 {
        r#"
        #include <linux/slab.h>
        #include <linux/types.h>
        "#;

        "return kmalloc(size, (gfp_t)flags);"
    }
    with_size(layout, |size| alloc(size, flags as u64))
}

/// Counterpart to [kmalloc]
///
/// # Safety
/// This function can only be called with a pointer allocated with the kernel's kmalloc() function.
#[inline]
pub unsafe fn kfree<T: ?Sized>(ptr: *mut T) {
    #[cfunc]
    unsafe fn dealloc(ptr: *mut c_void) {
        "#include <linux/slab.h>";

        "return kfree(ptr);"
    }
    unsafe { dealloc(ptr as *mut c_void) }
}

pub trait KernelAlloc {
    /// # Safety
    /// This method must comply with all [GlobalAlloc::alloc] requirements. Additionally, it must
    /// be able to handle Layout with size=0.
    fn alloc(&self, layout: Layout) -> *mut u8;
    /// # Safety
    /// This method must comply with all [GlobalAlloc::dealloc] requirements.
    unsafe fn dealloc(&self, ptr: *mut u8);
}

impl KernelAlloc for &dyn KernelAlloc {
    #[inline]
    fn alloc(&self, layout: Layout) -> *mut u8 {
        (*self).alloc(layout)
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8) {
        unsafe { (*self).dealloc(ptr) }
    }
}

impl<T> KernelAlloc for &T
where
    T: KernelAlloc,
{
    #[inline]
    fn alloc(&self, layout: Layout) -> *mut u8 {
        (*self).alloc(layout)
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8) {
        unsafe { (*self).dealloc(ptr) }
    }
}

#[derive(Default, Clone)]
pub struct KmallocAllocator<const FLAGS: GFPFlags>;

impl<const FLAGS: GFPFlags> KernelAlloc for KmallocAllocator<FLAGS> {
    #[inline]
    fn alloc(&self, layout: Layout) -> *mut u8 {
        kmalloc(layout, FLAGS)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8) {
        unsafe { kfree(ptr) };
    }
}

unsafe impl<const FLAGS: GFPFlags> GlobalAlloc for KmallocAllocator<FLAGS> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        <Self as KernelAlloc>::alloc(self, layout)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { <Self as KernelAlloc>::dealloc(self, ptr) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        #[cfunc]
        fn alloc_zeroed(size: usize, flags: u64) -> *mut u8 {
            "#include <linux/slab.h>";

            "return kzalloc(size, (gfp_t)flags);"
        }
        with_size(layout, |size| alloc_zeroed(size, FLAGS as u64))
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        #[cfunc]
        unsafe fn realloc(ptr: *mut u8, size: usize, flags: u64) -> *mut u8 {
            "#include <linux/slab.h>";

            r#"
            if (!size) {
                // Do not feed a size=0 to krealloc() as it will free it,
                // leading to a double-free.
                size = 1;
            }
            return krealloc(ptr, size, (gfp_t)flags);
            "#
        }
        let new_layout = Layout::from_size_align(new_size, layout.align());
        let new_layout = match new_layout {
            Ok(layout) => layout,
            Err(_) => handle_alloc_error(layout),
        };
        with_size(new_layout, |size| unsafe {
            realloc(ptr, size, FLAGS as u64)
        })
    }
}

#[cfg(not(test))]
#[global_allocator]
/// cbindgen:ignore
static GLOBAL: KmallocAllocator<{ GFPFlags::Kernel }> = KmallocAllocator;
