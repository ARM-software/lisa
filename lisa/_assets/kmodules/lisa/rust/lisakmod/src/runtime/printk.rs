/* SPDX-License-Identifier: GPL-2.0 */

use core::fmt::Write as _;

use crate::{
    fmt::{KBoxWriter, PRINT_PREFIX},
    inlinec::cfunc,
    runtime::alloc::{GFPFlags, KmallocAllocator},
};

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum DmesgLevel {
    Emerg = 0,
    Alert = 1,
    Crit = 2,
    Err = 3,
    Warning = 4,
    Notice = 5,
    Info = 6,
    Debug = 7,
    Cont = 8,
}

pub fn __pr_level_impl(level: DmesgLevel, fmt: core::fmt::Arguments<'_>) -> core::fmt::Result {
    fn write_dmesg<T: AsRef<[u8]>>(level: DmesgLevel, prefix: T, msg: T) -> core::fmt::Result {
        #[cfunc]
        fn printk<'a>(level: u8, prefix: &[u8], msg: &[u8]) {
            "#include <linux/printk.h>";

            r#"
            #pragma push_macro("pr_fmt")
            #undef pr_fmt
            #define pr_fmt(fmt) fmt

            #define HANDLE(level, f) case level: f("%.*s%.*s\n", (int)prefix.len, prefix.data, (int)msg.len, msg.data); break;
            switch (level) {
                HANDLE(0, pr_emerg);
                HANDLE(1, pr_alert);
                HANDLE(2, pr_crit);
                HANDLE(3, pr_err);
                HANDLE(4, pr_warn);
                HANDLE(5, pr_notice);
                HANDLE(6, pr_info);
                HANDLE(7, pr_debug);
                HANDLE(8, pr_cont);
            }
            #undef HANDLE
            #pragma pop_macro("pr_fmt")
            "#
        }

        printk(level as u8, prefix.as_ref(), msg.as_ref());
        Ok(())
    }

    match fmt.as_str() {
        // If the format is just a plain string, we can simply display it directly, no need to
        // allocate anything.
        Some(s) => write_dmesg(level, PRINT_PREFIX, s),
        None => {
            KBoxWriter::<KmallocAllocator<{ GFPFlags::Atomic }>, _>::with_writer(
                PRINT_PREFIX,
                "[...]",
                128,
                |mut writer| {
                    let res = writer.write_fmt(fmt);
                    // Make sure we always print what we have, even if we had some errors when rendering the
                    // string to the buffer. Errors could be as mundane as running out of space in the buffer,
                    // but we still want to see what _could_ be rendered.
                    write_dmesg(level, &b""[..], writer.written()).and(res)
                },
            )
        }
    }
}

macro_rules! __pr_level {
    ($level:expr, $($arg:tt)*) => {{
        $crate::runtime::printk::__pr_level_impl(
            $level,
            ::core::format_args!($($arg)*)
        ).expect("Could not write to dmesg")
    }}
}
pub(crate) use __pr_level;

macro_rules! pr_debug {
    ($($arg:tt)*) => {{
        $crate::runtime::printk::__pr_level!(
            $crate::runtime::printk::DmesgLevel::Debug,
            $($arg)*
        )
    }}
}
pub(crate) use pr_debug;

macro_rules! pr_info {
    ($($arg:tt)*) => {{
        $crate::runtime::printk::__pr_level!(
            $crate::runtime::printk::DmesgLevel::Info,
            $($arg)*
        )
    }}
}
pub(crate) use pr_info;

macro_rules! pr_err {
    ($($arg:tt)*) => {{
        $crate::runtime::printk::__pr_level!(
            $crate::runtime::printk::DmesgLevel::Err,
            $($arg)*
        )
    }}
}
pub(crate) use pr_err;
