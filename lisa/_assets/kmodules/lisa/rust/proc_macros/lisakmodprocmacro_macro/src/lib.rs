/* SPDX-License-Identifier: Apache-2.0 */

use proc_macro::TokenStream;
use syn::Error;

mod inlinec;

#[proc_macro_attribute]
pub fn cfunc(attrs: TokenStream, code: TokenStream) -> TokenStream {
    inlinec::cfunc(attrs, code).unwrap_or_else(|err| Error::into_compile_error(err).into())
}
