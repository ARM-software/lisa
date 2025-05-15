/* SPDX-License-Identifier: GPL-2.0 */

use core::fmt::Write;

use lisakmod_macros::inlinec::cfunc;

use crate::{
    fmt::{KBoxWriter, print_prefix},
    runtime::alloc::{GFPFlags, KmallocAllocator},
};

#[allow(dead_code)]
fn _panic(info: &core::panic::PanicInfo) -> ! {
    #[cfunc]
    fn panic<'a>(prefix: &'a [u8], msg: &'a [u8]) {
        r#"
        if (msg.data && msg.len) {
            panic("%.*sRust panic: %.*s", (int)prefix.len, prefix.data, (int)msg.len, msg.data);
        } else {
            panic("%.*sRust panic with no message or panicking encountered an error", (int)prefix.len, prefix.data);
        }
        "#
    }

    let msg = info.message();
    match msg.as_str() {
        Some(s) => panic(print_prefix().as_bytes(), s.as_bytes()),
        None => {
            KBoxWriter::<KmallocAllocator<{ GFPFlags::Atomic }>, _>::with_writer(
                print_prefix(),
                "[...]",
                128,
                |mut writer| {
                    // Not much we can do with a write error here since we already are panicking.
                    let _ = write!(writer, "{msg}");
                    panic(&b""[..], writer.written())
                },
            );
        }
    };

    #[allow(clippy::empty_loop)]
    loop {}
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    _panic(info)
}
