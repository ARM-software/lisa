/* SPDX-License-Identifier: GPL-2.0 */

pub use lisakmod_macros::inlinec::*;

macro_rules! c_eval {
    ($header:expr, $expr:literal, $ty:ty) => {{
        // Emit the C function code that will be extracted from the Rust object file and then
        // compiled as C.
        #[::lisakmod_macros::inlinec::cfunc]
        #[allow(non_snake_case)]
        fn snippet() -> $ty {
            concat!("#include<", $header, ">");
            concat!("return (", $expr, ");")
        }
        snippet()
    }};
}
pub(crate) use c_eval;
