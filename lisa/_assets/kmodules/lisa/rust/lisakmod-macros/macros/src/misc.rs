/* SPDX-License-Identifier: Apache-2.0 */

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{punctuated::Punctuated, Error, Expr, Ident, Token};

// Use getrandom instead of e.g. uuid crate as it has far fewer dependencies, so faster build
// time.
pub(crate) fn get_random() -> u128 {
    let mut buf: [u8; 128 / 8] = [0; 128 / 8];
    getrandom::getrandom(&mut buf).expect("Could not get random number");
    u128::from_le_bytes(buf)
}

pub(crate) fn concatcp(args: TokenStream) -> Result<TokenStream, Error> {
    let items = if args.is_empty() {
        Vec::new()
    } else {
        syn::parse::Parser::parse2(
            Punctuated::<Expr, Token![,]>::parse_terminated,
            args.clone(),
        )?
        .into_iter()
        .collect()
    };

    let item_names: Vec<Ident> = items
        .iter()
        .map(|_| format_ident!("__concat_item_{}", get_random()))
        .collect();

    let out = quote! {
        {
            #(
                const #item_names: &'static str = #items;
            )*

            unsafe {
                ::core::str::from_utf8_unchecked(
                    &::lisakmod_macros::private::misc::concat::<
                        {
                            #(
                                #item_names.len() +
                            )*
                            0
                        }
                    >(
                        &[
                            #(
                                #item_names
                            ),*
                        ]
                    )
                )
            }
        }
    };
    // eprintln!("OUT {out}");
    Ok(out)
}
