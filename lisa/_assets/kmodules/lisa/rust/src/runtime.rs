/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    alloc::{GlobalAlloc, Layout},
};

extern "C" {
    // All these low-level functions need to have the __lisa prefix, otherwise we will clash with
    // the ones provided by the Rust toolchain.
    fn __lisa_rust_alloc(size: usize) -> *mut u8;
    fn __lisa_rust_dealloc(ptr: *mut u8);
    fn __lisa_rust_alloc_zeroed(size: usize) -> *mut u8;
    fn __lisa_rust_realloc(ptr: *mut u8, size: usize) -> *mut u8;

    fn __lisa_rust_panic(msg: *const u8, len: usize);
    fn __lisa_rust_pr_info(msg: *const u8, len: usize);
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    match info.message().as_str() {
        Some(s) => unsafe { __lisa_rust_panic(s.as_ptr(), s.len()) },
        None => unsafe { __lisa_rust_panic(core::ptr::null(), 0) },
    };
    loop {}
}

struct KernelAllocator;

fn with_size<F: FnOnce(usize) -> *mut u8>(layout: Layout, f: F) -> *mut u8 {
    let size = layout.size();
    let align = layout.align();
    // For sizes which are a power of two, the kmalloc() alignment is also guaranteed to be at
    // least the respective size.
    if align <= 8 || (size.is_power_of_two() && align <= size) {
        f(layout.size())
    } else {
        pr_info!("Rust: cannot allocate memory with alignment > 8");
        core::ptr::null_mut()
    }
}

unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        with_size(layout, |size| unsafe { __lisa_rust_alloc(size) })
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { __lisa_rust_dealloc(ptr) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        with_size(layout, |size| unsafe { __lisa_rust_alloc_zeroed(size) })
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        with_size(layout, |_| unsafe { __lisa_rust_realloc(ptr, new_size) })
    }
}

#[global_allocator]
/// cbindgen:ignore
static GLOBAL: KernelAllocator = KernelAllocator;

// FIXME: Find a way to issue pr_cont() calls, otherwise each __lisa_rust_pr_info() call will
// create a newline, so each fragment of the write!() format string will be printed on a separated
// line.
pub struct __DmesgWriter {}

impl core::fmt::Write for __DmesgWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        unsafe { __lisa_rust_pr_info(s.as_ptr(), s.len()) }
        Ok(())
    }
}

macro_rules! pr_info {
    ($($arg:tt)*) => {{
        use ::core::fmt::Write as _;
        ::core::write!(
            $crate::runtime::__DmesgWriter {},
            $($arg)*
        ).expect("Could not write to dmesg")
    }}
}
pub(crate) use pr_info;
