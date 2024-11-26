/* SPDX-License-Identifier: GPL-2.0 */
#![no_std]
#![no_builtins]
#![feature(gen_blocks)]
#![feature(const_pin)]

extern crate alloc;

pub mod init;
pub mod inlinec;
pub mod prelude;
pub mod runtime;
pub mod tests;

macro_rules! container_of {
    ($container:ty, $member:ident, $ptr:expr) => {{
        let ptr = $ptr;
        let ptr: *const _ = (&*ptr);
        let offset = core::mem::offset_of!($container, $member);
        // SAFETY: Our contract is that c_kobj must be the member of a KObjectInner, so we can
        // safely compute the pointer to the parent.
        let container: *const $container = (ptr as *const $container).byte_sub(offset);
        container
    }};
}
pub(crate) use container_of;

#[allow(unused_macros)]
macro_rules! mut_container_of {
    ($container:ty, $member:ident, $ptr:expr) => {{ $crate::container_of!($container, $member, $ptr) as *mut $container }};
}
#[allow(unused_imports)]
pub(crate) use mut_container_of;
