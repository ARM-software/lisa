/* SPDX-License-Identifier: Apache-2.0 */
pub use lisakmod_macros_proc::{concatcp, dump_to_binstore, json_metadata};

#[macro_export]
macro_rules! __internal_export_symbol {
    ($sym:ident) => {
        $crate::misc::json_metadata!({
            "type": "export-symbol",
            "symbol": (::core::stringify!($sym))
        });
    };
}
pub use crate::__internal_export_symbol as export_symbol;
