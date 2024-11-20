/* SPDX-License-Identifier: Apache-2.0 */

use proc_macro::TokenStream;
use syn::Error;

mod inlinec;
mod misc;

fn convert(out: Result<proc_macro2::TokenStream, Error>) -> proc_macro::TokenStream {
    match out {
        Err(err) => Error::into_compile_error(err).into(),
        Ok(tokens) => tokens.into(),
    }
}

#[proc_macro_attribute]
pub fn cfunc(attrs: TokenStream, code: TokenStream) -> TokenStream {
    convert(inlinec::cfunc(attrs.into(), code.into()))
}

#[proc_macro_attribute]
pub fn cexport(attrs: TokenStream, code: TokenStream) -> TokenStream {
    convert(inlinec::cexport(attrs.into(), code.into()))
}

#[proc_macro]
pub fn c_constant(args: TokenStream) -> TokenStream {
    convert(inlinec::c_constant(args.into()))
}

#[proc_macro]
pub fn export_symbol(args: TokenStream) -> TokenStream {
    convert(inlinec::export_symbol(args.into()))
}

#[proc_macro]
pub fn concatcp(args: TokenStream) -> TokenStream {
    convert(misc::concatcp(args.into()))
}
