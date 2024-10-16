/* SPDX-License-Identifier: GPL-2.0 */

pub use lisakmodprocmacro::cfunc;

macro_rules! get_c_macro {
    ($header:expr, $macro:ident, $ty:ty) => {{
        ::paste::paste! {
            // Emit the C function code that will be extracted from the Rust object file and then
            // compiled as C.
            #[::lisakmodprocmacro::cfunc]
            #[allow(non_snake_case)]
            fn [<__macro_getter_ $macro>]() -> $ty {
                concat!("#include<", $header, ">");
                concat!("return (", stringify!($macro), ");")
            }
            [<__macro_getter_ $macro>]()
        }
    }};
}
pub(crate) use get_c_macro;
