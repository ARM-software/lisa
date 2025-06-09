/* SPDX-License-Identifier: GPL-2.0 */

use core::ffi::c_ulong;

use lisakmod_macros::inlinec::cfunc;

pub struct LocalIrqDisabledGuard {
    flags: c_ulong,
}

pub fn local_irq_save() -> LocalIrqDisabledGuard {
    #[cfunc]
    pub fn local_irq_save() -> c_ulong {
        r#"
        #include <linux/irqflags.h>
        "#;
        r#"
        unsigned long flags;
        local_irq_save(flags);
        return flags;
        "#
    }

    LocalIrqDisabledGuard {
        flags: local_irq_save(),
    }
}

impl Drop for LocalIrqDisabledGuard {
    fn drop(&mut self) {
        #[cfunc]
        pub fn local_irq_restore(flags: c_ulong) {
            r#"
            #include <linux/irqflags.h>
            "#;
            r#"
            // In C, "unsigned long *" is incompatible with "unsigned long long*" even when they
            // are both the same size. On 64bits platform, Rust c_ulong is u64, which translates
            // over FFI as uint64_t, which is "unsigned long long".
            //
            // To avoid type check warnings, provide an actual "unsigned long" to
            // local_irq_restore().
            unsigned long _flags = flags;
            local_irq_restore(_flags);
            "#
        }

        local_irq_restore(self.flags)
    }
}
