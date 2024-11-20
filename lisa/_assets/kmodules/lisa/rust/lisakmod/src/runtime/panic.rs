/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    cmp::min,
    fmt::Write,
    fmt::{self},
    ops::{Deref, DerefMut},
};

use crate::inlinec::cfunc;

struct SliceWriter<'a> {
    slice: &'a mut [u8],
    cursor: usize,
}

impl<'a> SliceWriter<'a> {
    fn new(slice: &'a mut [u8]) -> Self {
        SliceWriter { slice, cursor: 0 }
    }
}

impl Deref for SliceWriter<'_> {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.slice[self.cursor..]
    }
}

impl DerefMut for SliceWriter<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slice[self.cursor..]
    }
}

impl fmt::Write for SliceWriter<'_> {
    #[inline(always)]
    fn write_str(&mut self, s: &str) -> Result<(), fmt::Error> {
        let src = s.as_bytes();
        let dst = &mut *self;

        // Crop to our size, so we don't risk panicking.
        let len = min(src.len(), dst.len());
        dst[0..len].clone_from_slice(&src[0..len]);
        self.cursor += len;
        Ok(())
    }
}

#[allow(dead_code)]
fn _panic(info: &core::panic::PanicInfo) -> ! {
    #[cfunc]
    fn panic<'a>(msg: &'a [u8]) {
        r#"
        if (msg.data && msg.len) {
            panic("Rust panic: %.*s", (int)msg.len, msg.data);
        } else {
            panic("Rust panic with no message");
        }
        "#
    }

    let mut buf = [0; 128];
    let msg = info.message();
    let out: &[u8] = match msg.as_str() {
        Some(s) => s.as_bytes(),
        None => {
            let mut out = SliceWriter::new(buf.as_mut_slice());
            match write!(out, "{}", msg) {
                Ok(()) => &buf,
                Err(_) => "<error while formatting panic message>".as_bytes(),
            }
        }
    };
    panic(out);

    #[allow(clippy::empty_loop)]
    loop {}
}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    _panic(info)
}
