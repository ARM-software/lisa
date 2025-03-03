/* SPDX-License-Identifier: GPL-2.0 */

use core::fmt::Write;

use crate::{
    inlinec::cfunc,
    misc::KBoxWriter,
    runtime::alloc::{GFPFlags, KmallocAllocator},
};

#[allow(dead_code)]
fn _panic(info: &core::panic::PanicInfo) -> ! {
    #[cfunc]
    fn panic<'a>(msg: &'a [u8]) {
        r#"
        if (msg.data && msg.len) {
            panic("Rust panic: %.*s", (int)msg.len, msg.data);
        } else {
            panic("Rust panic with no message or panicking encountered an error");
        }
        "#
    }

    let msg = info.message();
    match msg.as_str() {
        Some(s) => panic(s.as_bytes()),
        None => {
            KBoxWriter::<KmallocAllocator<{ GFPFlags::Atomic }>, _>::with_writer(
                "[...]",
                128,
                |mut writer| {
                    // Not much we can do with a write error here since we already are panicking.
                    let _ = write!(writer, "{}", msg);
                    panic(writer.written())
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
