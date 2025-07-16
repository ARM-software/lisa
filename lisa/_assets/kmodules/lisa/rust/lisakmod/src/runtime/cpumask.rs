/* SPDX-License-Identifier: GPL-2.0 */

use core::ffi::{c_int, c_uint};

use lisakmod_macros::inlinec::cfunc;

pub type CpuId = c_uint;

#[cfunc]
pub fn smp_processor_id() -> CpuId {
    r#"
    #include <linux/cpumask.h>
    "#;
    r#"
    return smp_processor_id();
    "#
}

pub fn active_cpus() -> impl Iterator<Item = CpuId> {
    #[cfunc]
    fn next(cpu: &mut c_int) -> bool {
        r#"
        #include <linux/cpumask.h>
        "#;
        r#"
        *cpu = cpumask_next(*cpu, cpu_active_mask);
        return *cpu < nr_cpu_ids;
        "#
    }

    let mut cpu = -1;
    core::iter::from_fn(move || {
        if next(&mut cpu) {
            let cpu = cpu.try_into().expect("Invalid CPU ID");
            Some(cpu)
        } else {
            None
        }
    })
}
