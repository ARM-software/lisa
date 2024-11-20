/* SPDX-License-Identifier: GPL-2.0 */

use core::ffi::c_int;

use crate::runtime::printk::pr_err;

pub fn module_main() -> impl Iterator<Item = c_int> {
    gen {
        // use crate::runtime::sysfs::{KObjType, KObject};

        // let root = KObject::module_root();
        // let kobj_type = Arc::new(KObjType::new());
        // let kobject = KObject::new(kobj_type.clone());
        // let kobject2 = KObject::new(kobj_type.clone());
        // kobject.add(Some(&root), "foo");
        // kobject2.add(Some(&kobject), "bar");
        yield match crate::tests::init_tests() {
            Err(x) => {
                pr_err!("Lisa module Rust support validation failed: {x}");
                0
            }
            Ok(()) => 0,
        };
    }
}
