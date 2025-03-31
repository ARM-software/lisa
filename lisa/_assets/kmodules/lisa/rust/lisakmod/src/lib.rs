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

extern crate alloc;

pub mod error;
pub mod features;
pub mod fmt;
pub mod graph;
pub mod init;
pub mod lifecycle;
pub mod misc;
pub mod prelude;
pub mod registry;
pub mod runtime;
pub mod typemap;
