/* SPDX-License-Identifier: Apache-2.0 */

use itertools::Itertools as _;
use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::{format_ident, quote};
use syn::{Error, Expr, ExprLit, Ident, Lit, Token, punctuated::Punctuated, spanned::Spanned};

fn try_expand(tokens: TokenStream) -> Result<TokenStream, TokenStream> {
    let tokens: proc_macro::TokenStream = tokens.into();
    match tokens.expand_expr() {
        // As of 03/2025, TokenStream::expand_expr() will not evaluate
        // const expressions (e.g. trait associated constants), only
        // macros.
        Err(_) => Err(tokens.into()),
        Ok(tokens) => Ok(tokens.into()),
    }
}

// Use getrandom instead of e.g. uuid crate as it has far fewer dependencies, so faster build
// time.
pub(crate) fn get_random() -> u128 {
    let mut buf: [u8; 128 / 8] = [0; 128 / 8];
    getrandom::fill(&mut buf).expect("Could not get random number");
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

    // Concatenate all string literals we find beforehand to make the generate code more compact.
    #[allow(clippy::large_enum_variant)]
    enum Item {
        LitStr(String),
        Other(Expr),
    }
    impl quote::ToTokens for Item {
        fn to_tokens(&self, tokens: &mut TokenStream) {
            match self {
                Item::LitStr(s) => s.to_tokens(tokens),
                Item::Other(expr) => expr.to_tokens(tokens),
            }
        }
    }

    let items: Vec<_> = items
        .into_iter()
        .map(|item| match try_expand(quote! {#item}) {
            Ok(expr) => match syn::parse2::<Expr>(expr) {
                Ok(Expr::Lit(ExprLit {
                    lit: Lit::Str(litstr),
                    ..
                })) => Item::LitStr(litstr.value()),
                _ => Item::Other(item),
            },
            Err(_) => Item::Other(item),
        })
        .chunk_by(|item| matches!(item, Item::LitStr(_)))
        .into_iter()
        .flat_map(|(is_lit, chunk)| {
            if is_lit {
                vec![Item::LitStr(
                    chunk
                        .into_iter()
                        .map(|item| match item {
                            Item::LitStr(s) => s,
                            _ => unreachable!(),
                        })
                        .collect(),
                )]
            } else {
                chunk.into_iter().collect()
            }
        })
        .collect();

    let item_names: Vec<Ident> = items
        .iter()
        .map(|_| format_ident!("__concat_item_{:0>39}", get_random()))
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

pub fn dump_to_binstore(args: TokenStream) -> Result<TokenStream, Error> {
    let _args = if args.is_empty() {
        Vec::new()
    } else {
        syn::parse::Parser::parse2(
            Punctuated::<Expr, Token![,]>::parse_terminated,
            args.clone(),
        )?
        .into_iter()
        .collect()
    };
    match &_args[..] {
        [subsection, content] => __dump_to_binstore(quote! {#subsection}, quote! {#content}),
        _ => Err(Error::new(
            args.span(),
            "The usage is: dump_to_binstore!(<sub section name>, \"content\") ",
        )),
    }
}

pub(crate) fn _dump_to_binstore(
    subsection: &str,
    content: TokenStream,
) -> Result<TokenStream, Error> {
    __dump_to_binstore(quote! {#subsection}, content)
}

pub fn __dump_to_binstore(
    subsection: TokenStream,
    content: TokenStream,
) -> Result<TokenStream, Error> {
    Ok(quote! {
        const _: () = {
            // Allow a TokenStream input so we allow e.g. macro calls etc, rather than just string
            // literals.
            const CONTENT_STR: &str = #content;
            const CONTENT_SLICE: &[u8] = CONTENT_STR.as_bytes();
            const CONTENT_LEN: usize = CONTENT_SLICE.len();

            // Store the C function in a section of the binary, that will be extracted by the
            // module Makefile and the processed (e.g. compiled for C code).
            #[unsafe(link_section = ::core::concat!(".binstore.", #subsection))]
            #[used]
            static CONTENT_ARRAY: [u8; CONTENT_LEN] = {
                let mut arr = [0u8; CONTENT_LEN];
                let mut idx: usize = 0;
                while idx < CONTENT_LEN {
                    arr[idx] = CONTENT_SLICE[idx];
                    idx += 1;
                }
                arr
            };
        };
    })
}

pub fn json_metadata(args: TokenStream) -> Result<TokenStream, Error> {
    fn process(stream: TokenStream) -> Result<TokenStream, Error> {
        let tokens: Result<Vec<_>, Error> = stream
            .into_iter()
            .map(|tt| match tt {
                TokenTree::Group(ref grp) => {
                    match grp.delimiter() {
                        Delimiter::Parenthesis => {
                            // Allow input to contain parenthesized groups that will be expanded,
                            // rather than directly stringified to form the JSON output.
                            let grp = grp.stream();
                            match try_expand(grp) {
                                Err(grp) => {
                                    // Poor man's stringify!(). Ideally, we would evaluate grp to a
                                    // string, and then quote it according to JSON rules.
                                    // Unfortunately, we can't do that easily in a const fn for
                                    // now, so we just add quotes around and hope for the best.
                                    Ok(quote! { "\"", #grp, "\"" })
                                }
                                Ok(grp) => {
                                    // We evaluated to a literal, and now we can stringify!() it so
                                    // that it can be concatenated to the rest. This relies on a
                                    // the Rust native string syntax to be acceptable to a JSON
                                    // parser. Otherwise, we would need to convert the evaluated
                                    // literal to something JSON is happy with, but we cannot do
                                    // that easily.
                                    Ok(quote! {::core::stringify!(#grp)})
                                }
                            }
                        }
                        // Braces are allowed as part of the JSON syntax, we just recurse inside
                        Delimiter::Brace => {
                            let grp = process(grp.stream())?;
                            Ok(quote! { "{", #grp, "}" })
                        }
                        Delimiter::Bracket => {
                            let grp = process(grp.stream())?;
                            Ok(quote! { "[", #grp, "]" })
                        }
                        Delimiter::None => process(grp.stream()),
                    }
                }
                tt => Ok(quote! { ::core::stringify!(#tt) }),
            })
            .collect();
        let tokens = tokens?;
        Ok(quote! { #(#tokens),* })
    }
    let tokens = process(args)?;
    let tokens = if tokens.is_empty() {
        quote! { "{}" }
    } else {
        quote! { ::lisakmod_macros::misc::concatcp!(#tokens, "\n") }
    };
    _dump_to_binstore("json", tokens)
}
