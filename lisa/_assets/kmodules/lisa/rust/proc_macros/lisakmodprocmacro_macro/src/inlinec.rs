/* SPDX-License-Identifier: Apache-2.0 */

use proc_macro::TokenStream as RustcTokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parse, spanned::Spanned, token, Attribute, Error, FnArg, Generics, Ident, ItemFn, Pat,
    ReturnType, Stmt, StmtMacro, Type,
};

struct CFuncInput {
    name: Ident,
    c_code: (Stmt, Stmt, Stmt),
    f_args: Vec<(Ident, Type)>,
    f_attrs: Vec<Attribute>,
    f_ret_ty: Type,
    f_generics: Generics,
    f_unsafety: Option<token::Unsafe>,
}

impl Parse for CFuncInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_fn: ItemFn = input.parse()?;
        let span = item_fn.span();
        let f_attrs = item_fn.attrs;
        let name = item_fn.sig.ident;
        let f_ret_ty = match item_fn.sig.output {
            ReturnType::Type(_, ty) => *ty,
            ReturnType::Default => syn::parse_str::<Type>("()")?,
        };

        let mut snippets = Vec::new();
        for stmt in item_fn.block.stmts {
            let snippet = match &stmt {
                Stmt::Expr(expr, _) => Ok(Stmt::Expr(expr.clone(), None)),
                Stmt::Macro(mac) => {
                    let mac = mac.clone();
                    Ok(Stmt::Macro(StmtMacro {
                        semi_token: None,
                        ..mac
                    }))
                }
                stmt => Err(Error::new(
                    stmt.span(),
                    "An inline C function must contain string expressions.",
                )),
            }?;
            snippets.push(snippet.clone());
        }

        let empty = || Stmt::Expr(syn::parse_str("\"\"").unwrap(), None);

        let c_code = match snippets.len() {
            1 => Ok((
                empty(),
                snippets[0].clone(),
                empty(),
            )),
            2 => Ok((
                snippets[0].clone(),
                snippets[1].clone(),
                empty(),
            )),
            3 => Ok((
                snippets[0].clone(),
                snippets[1].clone(),
                snippets[2].clone(),
            )),
            _ => Err(Error::new(span, "An inline C function must contain either a single string expression with the C code in it or two string expressions (one output before the function and one for the body)."))
        }?;

        let f_args: Vec<_> = item_fn
            .sig
            .inputs
            .into_iter()
            .map(|arg| match arg {
                FnArg::Typed(arg) => {
                    let ident = match *arg.pat {
                        Pat::Ident(ident) => Ok(ident.ident),
                        _ => Err(Error::new(
                            arg.span(),
                            "An inline C function argument must be an identifier.",
                        )),
                    }?;
                    Ok((ident, *arg.ty))
                }
                _ => Err(Error::new(
                    arg.span(),
                    "An inline C function argument must have a type.",
                )),
            })
            .collect::<Result<_, _>>()?;

        let f_generics = item_fn.sig.generics.clone();
        let f_unsafety = item_fn.sig.unsafety;

        Ok(Self {
            name,
            c_code,
            f_args,
            f_attrs,
            f_ret_ty,
            f_generics,
            f_unsafety,
        })
    }
}

pub fn cfunc(_attrs: RustcTokenStream, code: RustcTokenStream) -> Result<RustcTokenStream, Error> {
    let input = syn::parse::<CFuncInput>(code)?;
    let (pre_c_code, c_code, post_c_code) = input.c_code;

    // TODO: Due to this issue, the span reported by syn is currently always going to have
    // span.line == 0, so we workaround that by just emitting a line!() call. This is not as
    // precise though.
    // https://github.com/dtolnay/proc-macro2/issues/402
    // EDIT: the issue is fixed, let's keep the workaround at hand in case it is needed again as
    // this was originally disabled because the rustc API was in flux.
    //
    // let pre_c_code_line = quote! { line!() };
    // let c_code_line = quote! { line!() };
    // let post_c_code_line = quote! { line!() };
    let pre_c_code_line = pre_c_code.span().start().line;
    let c_code_line = c_code.span().start().line;
    let post_c_code_line = post_c_code.span().start().line;

    let name = input.name;
    let f_ret_ty = input.f_ret_ty;
    let f_attrs = input.f_attrs;
    let f_args = input.f_args;
    let f_unsafety = input.f_unsafety;
    let f_generics = input.f_generics;
    let f_where = f_generics.where_clause.clone();

    let section_str = format!(".binstore.c_shims.{}", name);

    // Use getrandom instead of e.g. uuid crate as it has far fewer dependencies, so faster build
    // time.
    fn get_random() -> u128 {
        let mut buf: [u8; 128 / 8] = [0; 128 / 8];
        getrandom::getrandom(&mut buf).expect("Could not get random number");
        u128::from_le_bytes(buf)
    }
    let c_name_str = format!("__lisa_c_shim_{name}_{}", get_random());
    let c_name = format_ident!("{}", c_name_str);
    let c_proto = format!("{c_name_str}_proto");
    let c_ret_ty = format!("{c_name_str}_ret_ty");
    let (c_args, c_args_ty_macros, rust_args, rust_extern_args, rust_extern_call_args) = if f_args
        .is_empty()
    {
        (
            quote! { "void" },
            quote! { "" },
            quote! {},
            quote! {},
            quote! {},
        )
    } else {
        let c_nr_args = f_args.len();
        let (arg_names, arg_tys): (Vec<_>, Vec<_>) = f_args.into_iter().unzip();
        let c_arg_names: Vec<_> = arg_names.iter().map(|name| name.to_string()).collect();
        let c_args_commas: Vec<_> = c_arg_names
            .iter()
            .enumerate()
            .map(|(i, _)| if i == (c_nr_args - 1) { "" } else { ", " })
            .collect();
        let c_arg_ty_names: Vec<_> = arg_names
            .iter()
            .map(|arg| format!("{c_name_str}_arg_ty_{arg}"))
            .collect();

        // Argument types are encoded as a function-like macro. Calling this macro with an
        // identifier declares a variable (or function argument) of that type.  This allows using
        // any type, including more complex ones like arrays and function pointers.
        let c_args_ty_macros = quote! {
            ::lisakmodprocmacro::private::const_format::concatcp!(
                #(
                    "\n",
                    ::lisakmodprocmacro::private::const_format::concatcp!(
                        "#define ", #c_arg_ty_names, "(DECLARATOR)",
                        {
                            // Use a const fn to introduce f_generics and f_where
                            const fn get #f_generics() -> &'static str #f_where {
                                <#arg_tys as ::lisakmodprocmacro::private::inlinec::FfiType>::C_DECL
                            }
                            get()
                        },
                    )
                ),*
            )
        };

        let c_args = quote! {
            ::lisakmodprocmacro::private::const_format::concatcp!(
                #(
                    #c_arg_ty_names,
                    "(", #c_arg_names, ")", #c_args_commas
                ),*
            )
        };
        let rust_args = quote! {
            #(
                #arg_names : #arg_tys
            ),*
        };
        let rust_extern_args = quote! {
            #(
                #arg_names : <#arg_tys as ::lisakmodprocmacro::private::inlinec::FfiType>::FfiType
            ),*
        };
        let rust_extern_call_args = quote! {
            #(
                ::lisakmodprocmacro::private::inlinec::IntoFfi::into_ffi(#arg_names)
            ),*
        };
        (
            c_args,
            c_args_ty_macros,
            rust_args,
            rust_extern_args,
            rust_extern_call_args,
        )
    };

    let out = quote! {
        // Store the C function in a section of the binary, that will be extracted by the
        // module Makefile and compiled separately as C code.
        const _: () = {
            // Keep this out of the concatcp!() call so we can have the generic parameters for
            // f_ret_ty in scope. This is possible with the "generic_const_item" unstable feature.
            const CODE_SLICE: &[u8] = ::lisakmodprocmacro::private::const_format::concatcp!(
                r#"
                #include <linux/types.h>

                #define BUILTIN_TY_DECL(ty, declarator) ty (declarator)
                #define PTR_TY_DECL(declarator) *(declarator)
                #define ARR_TY_DECL(N, declarator) (declarator)[N]
                #define FN_PTR_TY_DECL(args, declarator) (*(declarator))args
                #define FN_TY_DECL(args, declarator) (declarator)args
                "#,

                "#line ", line!(), " \"", file!(), "\"\n",
                #c_args_ty_macros,

                // See comment on how arguments type are handled, as we do the same for the return
                // type.
                "\n#define ", #c_ret_ty, "(DECLARATOR)",
                {
                    // Use a const fn to introduce f_generics and f_where
                    const fn get #f_generics() -> &'static str #f_where {
                        <#f_ret_ty as ::lisakmodprocmacro::private::inlinec::FfiType>::C_DECL
                    }
                    get()
                },
                "\n#define ", #c_proto, " ", #c_ret_ty, "(FN_TY_DECL((", #c_args, "), ", #c_name_str, "))",
                "\n",

                "#line ", #pre_c_code_line, " \"", file!(), "\"\n",
                #pre_c_code,
                "\n",

                // Prototype
                "#line ", #c_code_line, " \"", file!(), "\"\n",
                #c_proto, ";\n",
                // Definition
                #c_proto,
                "{\n#line ", #c_code_line, " \"", file!(), "\"\n",
                #c_code,
                "\n}\n",
                "#line ", #post_c_code_line, " \"", file!(), "\"\n",
                #post_c_code,
                "\n",
            ).as_bytes();
            const CODE_LEN: usize = CODE_SLICE.len();

            #[link_section = #section_str ]
            #[used]
            static CODE: [u8; CODE_LEN] = {
                let mut arr = [0u8; CODE_LEN];
                let mut idx: usize = 0;
                while idx < CODE_LEN {
                    arr[idx] = CODE_SLICE[idx];
                    idx += 1;
                }
                arr
            };
        };

        extern "C" {
            fn #c_name #f_generics(#rust_extern_args) -> <#f_ret_ty as ::lisakmodprocmacro::private::inlinec::FfiType>::FfiType #f_where;
        }

        #[inline]
        #(#f_attrs)*
        #f_unsafety fn #name #f_generics(#rust_args) -> #f_ret_ty #f_where {
            unsafe {
                ::lisakmodprocmacro::private::inlinec::FromFfi::from_ffi(
                    #c_name(#rust_extern_call_args)
                )
            }
        }
    };
    // eprintln!("{}", &out.to_string());
    Ok(out.into())
}
