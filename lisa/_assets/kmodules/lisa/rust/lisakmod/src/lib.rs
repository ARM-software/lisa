/* SPDX-License-Identifier: GPL-2.0 */
#![no_std]
#![no_builtins]
#![feature(adt_const_params)]
#![feature(maybe_uninit_fill)]
#![feature(coerce_unsized)]
#![feature(unsize)]
#![feature(maybe_uninit_write_slice)]
#![feature(type_alias_impl_trait)]
#![feature(arbitrary_self_types_pointers)]
#![feature(formatting_options)]
#![feature(try_trait_v2)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]

extern crate alloc;

// Allow std in unit tests for convenience, so we can use e.g. println!()
#[cfg(test)]
extern crate std;

pub mod error;
pub mod features;
pub mod fmt;
pub mod graph;
pub mod init;
pub mod lifecycle;
pub mod mem;
pub mod misc;
pub mod parsec;
pub mod prelude;
pub mod query;
pub mod refstore;
pub mod registry;
pub mod runtime;
pub mod typemap;
pub mod version;
