/* SPDX-License-Identifier: GPL-2.0 */

use crate::inlinec::cfunc;

enum DmesgWriterState {
    Init,
    Cont,
}

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

pub struct __DmesgWriter {
    state: DmesgWriterState,
    level: DmesgLevel,
}

impl __DmesgWriter {
    pub fn new(level: DmesgLevel) -> Self {
        __DmesgWriter {
            level,
            state: DmesgWriterState::Init,
        }
    }
}

impl core::fmt::Write for __DmesgWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let level = match self.state {
            DmesgWriterState::Init => self.level,
            DmesgWriterState::Cont => DmesgLevel::Cont,
        };
        self.state = DmesgWriterState::Cont;

        #[cfunc]
        fn printk<'a>(level: u8, msg: &[u8]) {
            "#include <main.h>";

            r#"
            #define HANDLE(level, f) case level: f("%.*s", (int)msg.len, msg.data); break;
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
            "#
        }

        printk(level as u8, s.as_bytes());
        Ok(())
    }
}

macro_rules! __pr_level {
    ($level:expr, $($arg:tt)*) => {{
        use ::core::fmt::Write as _;
        ::core::write!(
            $crate::runtime::printk::__DmesgWriter::new($level),
            $($arg)*
        ).expect("Could not write to dmesg")
    }}
}
pub(crate) use __pr_level;

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
