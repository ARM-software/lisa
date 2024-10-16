/* SPDX-License-Identifier: GPL-2.0 */

use alloc::alloc::handle_alloc_error;
use core::alloc::{GlobalAlloc, Layout};

use crate::{
    inlinec::{cfunc, get_c_macro},
    runtime::printk::pr_err,
};

struct KernelAllocator;

#[inline]
fn with_size<F: FnOnce(usize) -> *mut u8>(layout: Layout, f: F) -> *mut u8 {
    let minalign = || get_c_macro!("linux/slab.h", ARCH_KMALLOC_MINALIGN, usize);

    let size = layout.size();
    let align = layout.align();
    // For sizes which are a power of two, the kmalloc() alignment is also guaranteed to be at
    // least the respective size.
    if (size.is_power_of_two() && align <= size) || (align <= minalign()) {
        let ptr = f(layout.size());
        assert_eq!((ptr as usize % align), 0);
        ptr
    } else {
        // Do not panic as this would create UB
        pr_err!("Rust: cannot allocate memory with alignment > {minalign} and a size {size} that is not a power of two", minalign=minalign());
        core::ptr::null_mut()
    }
}

/// This function is guaranteed to return the pointer given by the kernel's kmalloc() without
/// re-aligning it in any way. This makes it suitable to pass to kfree() without knowing the
/// original layout.
#[inline]
pub fn kmalloc(layout: Layout) -> *mut u8 {
    #[cfunc]
    fn alloc(size: usize) -> *mut u8 {
        "#include <linux/slab.h>";

        "return kmalloc(size, GFP_KERNEL);"
    }
    with_size(layout, alloc)
}

#[inline]
pub unsafe fn kfree<T: ?Sized>(ptr: *mut T) {
    #[cfunc]
    unsafe fn dealloc(ptr: *mut u8) {
        "#include <linux/slab.h>";

        "return kfree(ptr);"
    }
    unsafe { dealloc(ptr as *mut u8) }
}

unsafe impl GlobalAlloc for KernelAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        kmalloc(layout)
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { kfree(ptr) };
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        #[cfunc]
        fn alloc_zeroed(size: usize) -> *mut u8 {
            "#include <linux/slab.h>";

            "return kzalloc(size, GFP_KERNEL);"
        }
        with_size(layout, alloc_zeroed)
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        #[cfunc]
        unsafe fn realloc(ptr: *mut u8, size: usize) -> *mut u8 {
            "#include <linux/slab.h>";

            r#"
            if (!size) {
                // Do not feed a size=0 to krealloc() as it will free it,
                // leading to a double-free.
                size = 1;
            }
            return krealloc(ptr, size, GFP_KERNEL);
            "#
        }
        let new_layout = Layout::from_size_align(new_size, layout.align());
        let new_layout = match new_layout {
            Ok(layout) => layout,
            Err(_) => handle_alloc_error(layout),
        };
        with_size(new_layout, |size| unsafe { realloc(ptr, size) })
    }
}

#[global_allocator]
/// cbindgen:ignore
static GLOBAL: KernelAllocator = KernelAllocator;
