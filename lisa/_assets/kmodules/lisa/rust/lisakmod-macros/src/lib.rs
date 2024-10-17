/* SPDX-License-Identifier: Apache-2.0 */
#![no_std]
#![feature(fundamental)]
#![feature(layout_for_ptr)]

extern crate alloc;

// Without that, proc macros that contains references to items such as ::lisakmod_macros::foo  in
// the expand code will not work, as the current crate is named "crate" but not
// "lisakmod_macros".  With the "extern crate", we introduce our name in our current scope.
//
// https://users.rust-lang.org/t/how-to-express-crate-path-in-procedural-macros/91274/17
extern crate self as lisakmod_macros;

pub mod inlinec;
pub mod misc;
pub mod private;
