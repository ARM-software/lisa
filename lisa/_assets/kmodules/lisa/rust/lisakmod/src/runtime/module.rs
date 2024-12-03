/* SPDX-License-Identifier: GPL-2.0 */

use core::ffi::c_int;

use crate::{
    init::module_main,
    lifecycle::LifeCycle,
    prelude::*,
    runtime::{
        printk::pr_err,
        sync::{Lock, new_static_mutex},
    },
};

enum State<T> {
    Initial,
    Initialized(T),
    Deinitialized,
}

new_static_mutex!(
    MAIN_ITERATOR,
    State<LifeCycle<(), (), c_int>>,
    State::Initial
);

#[cexport(export_name = "rust_mod_init")]
pub fn rust_mod_init() -> c_int {
    // Take the lock at the beginning so there is no risk of nesting multiple calls or anything
    // nasty like that.
    let mut guard = MAIN_ITERATOR.lock();
    assert!(
        matches!(&*guard, State::Initial),
        "rust_mod_init() was called multiple times"
    );

    let mut lifecycle = module_main();
    match lifecycle.start(()) {
        Ok(()) => {
            *guard = State::Initialized(lifecycle);
            0
        }
        Err(err) => {
            pr_err!("rust_mod_init() returned an error: {err}");
            // Immediately run the deinit() function, since the kernel will prevent the module from
            // being unloaded if module_init returned non-zero.
            pr_info!("Running rust_mod_exit() Immediately as rust_mod_init() returned an error.");
            *guard = State::Deinitialized;
            if let Err(err) = lifecycle.stop() {
                pr_err!("rust_mod_exit() returned an error: {err}");
            };

            // We cleaned up behind us, so we can return success. If we return an error here, the
            // kernel will not recover from that and it will not be possible to unload the module,
            // which is going to be problematic for the user.
            0
        }
    }
}

#[cexport(export_name = "rust_mod_exit")]
pub fn rust_mod_exit() {
    let mut guard = MAIN_ITERATOR.lock();
    match &mut *guard {
        State::Initial => panic!("rust_mod_init() was not called before rust_mod_exit()"),
        State::Initialized(lifecycle) => {
            if let Err(err) = lifecycle.stop() {
                pr_err!("rust_mod_exit() returned an error: {err}")
            }
            *guard = State::Deinitialized;
        }
        State::Deinitialized => pr_info!("rust_mod_exit() already called, doing nothing"),
    }
}
