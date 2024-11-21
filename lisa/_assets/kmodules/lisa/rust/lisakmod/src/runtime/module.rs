/* SPDX-License-Identifier: GPL-2.0 */

use alloc::boxed::Box;
use core::ffi::c_int;

use crate::init::module_main;
use crate::prelude::*;
use crate::runtime::sync::{Lock, new_static_mutex};

new_static_mutex!(
    MAIN_ITERATOR,
    Option<Box<dyn Iterator<Item = c_int> + Send>>,
    None
);

#[cexport(export_name = "rust_mod_init")]
pub fn rust_mod_init() -> c_int {
    let mut iterator = Box::new(module_main());
    let ret = iterator.next().expect("Main iterator did not yield once");
    if ret == 0 {
        *MAIN_ITERATOR.lock() = Some(iterator);
        0
    } else {
        assert_eq!(
            iterator.next(),
            None,
            "Main iterator yielded more than once"
        );
        ret
    }
}

#[cexport(export_name = "rust_mod_exit")]
pub fn rust_mod_exit() {
    let mut iterator = MAIN_ITERATOR.lock();
    let iterator = iterator
        .as_mut()
        .expect("rust_mod_init() was not called before rust_mod_exit()");
    assert_eq!(
        iterator.next(),
        None,
        "Main iterator yielded more than once"
    );
}
