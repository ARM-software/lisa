/* SPDX-License-Identifier: GPL-2.0 */

#[allow(unused_imports)]
pub(crate) use ::alloc::boxed::Box;
#[allow(unused_imports)]
pub(crate) use lisakmod_macros::inlinec::{c_eval, cconstant, cexport, cfunc, cstatic};

#[allow(unused_imports)]
pub(crate) use crate::{
    runtime::kbox::{KBox, KernelKBox},
    runtime::printk::{pr_err, pr_info},
};
