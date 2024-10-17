/* SPDX-License-Identifier: Apache-2.0 */

use std::io::{self, Write};

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    parse::Parse, parse_quote, punctuated::Punctuated, spanned::Spanned, token, Abi, Attribute,
    Error, Expr, ExprGroup, ExprLit, ExprTuple, FnArg, Generics, Ident, ItemFn, Lit, LitStr, Meta,
    Pat, ReturnType, Stmt, StmtMacro, Token, Type, Visibility,
};

use crate::misc::{concatcp, get_random};

struct CFuncInput {
    name: Ident,
    c_code: (
        Option<TokenStream>,
        Option<TokenStream>,
        Option<TokenStream>,
    ),
    f_args: Vec<(Ident, Type)>,
    f_attrs: Vec<Attribute>,
    f_ret_ty: Type,
    f_generics: Generics,
    f_unsafety: Option<token::Unsafe>,
}

fn get_f_args(item_fn: &ItemFn) -> syn::Result<Vec<(Ident, Type)>> {
    item_fn
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Typed(arg) => {
                let ident = match &*arg.pat {
                    Pat::Ident(ident) => Ok(ident.ident.clone()),
                    _ => Err(Error::new(
                        arg.span(),
                        "An inline C function parameter must be an identifier.",
                    )),
                }?;
                Ok((ident, *arg.ty.clone()))
            }
            FnArg::Receiver(..) => Err(Error::new(
                arg.span(),
                "self is not allowed in inline C function parameter",
            )),
        })
        .collect::<Result<_, _>>()
}

fn get_f_ret_ty(item_fn: &ItemFn) -> syn::Result<Type> {
    Ok(match &item_fn.sig.output {
        ReturnType::Type(_, ty) => *ty.clone(),
        ReturnType::Default => parse_quote!(()),
    })
}

impl Parse for CFuncInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_fn: ItemFn = input.parse()?;
        let span = item_fn.span();
        let f_args = get_f_args(&item_fn)?;
        let f_ret_ty = get_f_ret_ty(&item_fn)?;
        let f_attrs = item_fn.attrs;
        let name = item_fn.sig.ident;

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
            snippets.push(quote! {#snippet});
        }

        let c_code = match snippets.len() {
            1 => Ok((
                None,
                Some(snippets[0].clone()),
                None,
            )),
            2 => Ok((
                Some(snippets[0].clone()),
                Some(snippets[1].clone()),
                None,
            )),
            3 => Ok((
                Some(snippets[0].clone()),
                Some(snippets[1].clone()),
                Some(snippets[2].clone()),
            )),
            _ => Err(Error::new(span, "An inline C function must contain either a single string expression with the C code in it or two string expressions (one output before the function and one for the body)."))
        }?;

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

fn make_c_func(
    rust_name: Option<&Ident>,
    c_name: &Ident,
    f_generics: Option<&Generics>,
    f_args: &[(Ident, Type)],
    f_ret_ty: &Type,
    c_code: (
        Option<TokenStream>,
        Option<TokenStream>,
        Option<TokenStream>,
    ),
) -> Result<TokenStream, Error> {
    let (c_out, c_header_out, rust_out) =
        _make_c_func(rust_name, c_name, f_generics, f_args, f_ret_ty, c_code)?;

    let outs = [c_out, c_header_out];
    let sections = [
        format!(".binstore.c.code.{}", c_name),
        format!(".binstore.c.header.{}", c_name),
    ];

    Ok(quote! {
        // Store the C function in a section of the binary, that will be extracted by the
        // module Makefile and compiled separately as C code.
        #(
            const _: () = {
                const CODE_SLICE: &[u8] = #outs.as_bytes();
                const CODE_LEN: usize = CODE_SLICE.len();

                #[link_section = #sections ]
                #[used]
                static CODE: [u8; CODE_LEN] = ::lisakmod_macros::private::misc::slice_to_array::<{CODE_LEN}>(CODE_SLICE);
            };
        )*

        #rust_out
    })
}

fn _make_c_func(
    rust_name: Option<&Ident>,
    c_name: &Ident,
    f_generics: Option<&Generics>,
    f_args: &[(Ident, Type)],
    f_ret_ty: &Type,
    c_code: (
        Option<TokenStream>,
        Option<TokenStream>,
        Option<TokenStream>,
    ),
) -> Result<(TokenStream, TokenStream, TokenStream), Error> {
    let c_name_str: String = c_name.to_string();
    let f_where = f_generics.map(|x| x.where_clause.clone());
    let c_proto = format!("{c_name_str}_proto");
    let c_ret_ty = format!("{c_name_str}_ret_ty");

    let (pre_c_code, c_code, post_c_code) = c_code;
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
    let pre_c_code_line = match &pre_c_code {
        Some(tokens) => tokens.span().start().line,
        None => 0,
    }
    .to_string();
    let c_code_line = match &c_code {
        Some(tokens) => tokens.span().start().line.to_string(),
        None => pre_c_code_line.clone(),
    };
    let post_c_code_line = match &post_c_code {
        Some(tokens) => tokens.span().start().line.to_string(),
        None => c_code_line.clone(),
    };

    let c_nr_args = f_args.len();
    let (arg_names, arg_tys): (Vec<_>, Vec<_>) = f_args.iter().cloned().unzip();
    let c_args_name: Vec<_> = arg_names.iter().map(|name| name.to_string()).collect();
    let c_args_commas: Vec<_> = c_args_name
        .iter()
        .enumerate()
        .map(|(i, _)| if i == (c_nr_args - 1) { "" } else { ", " })
        .collect();
    let (c_args_ty_macro, c_args_ty_typedef): (Vec<_>, Vec<_>) = arg_names
        .iter()
        .map(|arg| {
            (
                format!("{c_name_str}_arg_ty_macro_{arg}"),
                format!("{c_name_str}_arg_ty_typedef_{arg}"),
            )
        })
        .unzip();

    // Argument types are encoded as a function-like macro. Calling this macro with an
    // identifier declares a variable (or function argument) of that type.  This allows using
    // any type, including more complex ones like arrays and function pointers.
    let c_args_typedef = concatcp(quote! {
        #(
            "\n",
            "#define ", #c_args_ty_macro, "(DECLARATOR)",
            {
                // Use a const fn to introduce f_generics and f_where
                const fn get #f_generics() -> &'static str #f_where {
                    <#arg_tys as ::lisakmod_macros::inlinec::FfiType>::C_DECL
                }
                get()
            },
            "\n",
            "typedef ", #c_args_ty_macro, "(", #c_args_ty_typedef ,");"
        ),*
    })?;

    let c_args = if c_nr_args == 0 {
        quote! { "void "}
    } else {
        concatcp(quote! {
            #(
                #c_args_ty_typedef, " ", #c_args_name,
                #c_args_commas
            ),*
        })?
    };

    let rust_extern_args = quote! {
        #(
            #arg_names : <#arg_tys as ::lisakmod_macros::inlinec::FfiType>::FfiType
        ),*
    };

    let pre_c_code = pre_c_code.unwrap_or(quote! {""});
    let post_c_code = post_c_code.unwrap_or(quote! {""});

    let c_funcdef = match c_code {
        Some(c_code) => concatcp(quote! {
            #c_proto,
            "{\n#line ", #c_code_line, " \"", file!(), "\"\n",
            #c_code,
            "\n}\n",
        })?,
        None => quote! {""},
    };

    let c_header_out = concatcp(quote! {
        r#"
        #include <linux/types.h>

        #define BUILTIN_TY_DECL(ty, declarator) ty declarator
        #define PTR_TY_DECL(declarator) (*declarator)
        #define ARR_TY_DECL(N, declarator) ((declarator)[N])
        #define CONST_TY_DECL(declarator) const declarator
        #define ATTR_TY_DECL(attributes, declarator) attributes declarator
        #define FN_TY_DECL(args, declarator) ((declarator)args)
        "#,

        "#line ", #c_code_line, " \"", file!(), "\"\n",
        #c_args_typedef,

        // See comment on how arguments type are handled, as we do the same for the return
        // type.
        "\n#define ", #c_ret_ty, "(DECLARATOR)",
        {
            // Use a const fn to introduce f_generics and f_where
            const fn get #f_generics() -> &'static str #f_where {
                <#f_ret_ty as ::lisakmod_macros::inlinec::FfiType>::C_DECL
            }
            get()
        },
        "\n#define ", #c_proto, " ", #c_ret_ty, "(FN_TY_DECL((", #c_args, "), ATTR_TY_DECL(__nocfi, ", #c_name_str, ")))",
        "\n#line ", #c_code_line, " \"", file!(), "\"\n",
        #c_proto, ";\n",
    })?;

    let c_out = concatcp(quote! {
        "#line ", #pre_c_code_line, " \"", file!(), "\"\n",
        #pre_c_code,
        "\n",
        // Definition
        "#line ", #c_code_line, " \"", file!(), "\"\n",
        #c_funcdef,
        "#line ", #post_c_code_line, " \"", file!(), "\"\n",
        #post_c_code,
        "\n",
    })?;

    let rust_out = match rust_name {
        Some(rust_name) => quote! {
            extern "C" {
                #[link_name = #c_name_str]
                fn #rust_name #f_generics(#rust_extern_args) -> <#f_ret_ty as ::lisakmod_macros::inlinec::FfiType>::FfiType #f_where;
            }
        },
        None => quote! {},
    };
    Ok((c_out, c_header_out, rust_out))
}

pub fn cfunc(_attrs: TokenStream, code: TokenStream) -> Result<TokenStream, Error> {
    let input = syn::parse2::<CFuncInput>(code)?;

    let name = input.name;
    let f_ret_ty = input.f_ret_ty;
    let f_attrs = input.f_attrs;
    let f_args = input.f_args;
    let f_unsafety = input.f_unsafety;
    let f_generics = input.f_generics;
    let f_where = f_generics.where_clause.clone();

    let shim_name_str = format!("__lisa_c_shim_{name}_{}", get_random());
    let shim_name = format_ident!("{}", shim_name_str);
    let c_out = make_c_func(
        Some(&shim_name),
        &shim_name,
        Some(&f_generics),
        &f_args,
        &f_ret_ty,
        input.c_code,
    )?;

    let (arg_names, arg_tys): (Vec<_>, Vec<_>) = f_args.into_iter().unzip();
    let rust_args = quote! {
        #(
            #arg_names : #arg_tys
        ),*
    };
    let rust_extern_call_args = quote! {
        #(
            ::lisakmod_macros::inlinec::IntoFfi::into_ffi(#arg_names)
        ),*
    };

    let rust_out = quote! {
        #[inline]
        #(#f_attrs)*
        #f_unsafety fn #name #f_generics(#rust_args) -> #f_ret_ty #f_where {
            unsafe {
                ::lisakmod_macros::inlinec::FromFfi::from_ffi(
                    #shim_name(#rust_extern_call_args)
                )
            }
        }
    };

    let out = quote! {
        #c_out
        #rust_out
    };
    // eprintln!("{}", &out.to_string());
    Ok(out)
}

pub fn c_constant(args: TokenStream) -> Result<TokenStream, Error> {
    let span = Span::call_site();
    let args = syn::parse::Parser::parse2(Punctuated::<Expr, Token![,]>::parse_terminated, args)?;

    fn parse_arg(expr: Expr) -> Result<String, Error> {
        match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) => Ok(s.value()),
            Expr::Tuple(ExprTuple { elems, .. }) => {
                let elems = elems
                    .into_iter()
                    .map(parse_arg)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(elems.join(""))
            }
            Expr::Group(ExprGroup { expr, .. }) => parse_arg(*expr),
            // expr => panic!("{:?}", expr),
            expr => Err(Error::new(expr.span(), "Expected a string literal")),
        }
    }

    let args: Vec<_> = args
        .into_iter()
        .map(parse_arg)
        .collect::<Result<Vec<_>, _>>()?;

    let (headers, expr, default) = if args.len() == 3 {
        Ok((args[0].clone(), args[1].clone(), args[2].clone()))
    } else {
        Err(Error::new(
            span,
            "Expected 3 string literals: the header #includes, the expression, and a default value (as a C expression string) when no C toolchain is available (e.g. when running cargo check).",
        ))
    }?;

    let out = match std::env::var("LISA_EVAL_C") {
        Ok(cmd) => {
            use std::process::Command;
            let mut cmd = Command::new(cmd);

            cmd.arg(headers.clone()).arg(expr.clone());
            let out = cmd
                .output()
                .map_err(|e| Error::new(span, format!("Could not run C toolchain: {e}")))?;

            io::stderr().write_all(&out.stderr).unwrap();

            let out = std::str::from_utf8(&out.stdout)
                .map_err(|_| Error::new(span, "Could not decode toolchain output"))?;
            out.trim().to_owned()
        }
        Err(_) => default.to_owned(),
    };

    let out_expr = syn::parse_str::<Expr>(&out)?;
    let assert_cond = concatcp(quote! {
        "(", #expr, ") == (", #out, ")"
    })?;
    let out = quote! {
        {
            // Ensure that the constant value we got from the compiler matches what we will get in
            // the real world when running the code.
            ::lisakmod_macros::inlinec::c_static_assert!(
                #headers,
                #assert_cond
            );
            #out_expr
        }
    };
    Ok(out)
}

struct CExportInput {
    item_fn: ItemFn,
}

impl Parse for CExportInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_fn: ItemFn = input.parse()?;
        Ok(CExportInput { item_fn })
    }
}

struct CExportAttrs {
    link_name: Option<String>,
}

impl Parse for CExportAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;

        let mut link_name = None;
        for attr in attrs {
            match attr {
                Meta::List(ml) => match ml.path.get_ident() {
                    Some(ident) => match &*ident.to_string() {
                        "link_name" => {
                            link_name = Some(syn::parse2::<LitStr>(ml.tokens)?.value());
                        }
                        name => {
                            return Err(Error::new(ml.span(), format!("Unknown argument: {name}")))
                        }
                    },
                    _ => return Err(Error::new(ml.span(), "Invalid argument name".to_string())),
                },
                attr => return Err(Error::new(attr.span(), "Unknown argument".to_string())),
            }
        }

        Ok(CExportAttrs { link_name })
    }
}

pub fn cexport(attrs: TokenStream, code: TokenStream) -> Result<TokenStream, Error> {
    let input = syn::parse2::<CExportInput>(code)?;
    let attrs = syn::parse2::<CExportAttrs>(attrs)?;
    let mut item_fn = input.item_fn;
    let f_args = get_f_args(&item_fn)?;
    let f_ret_ty = get_f_ret_ty(&item_fn)?;
    let (arg_names, arg_tys): (Vec<_>, Vec<_>) = f_args.iter().cloned().unzip();

    fn ffi_ty(ty: Type) -> syn::Result<Type> {
        Ok(parse_quote! {
            <#ty as ::lisakmod_macros::inlinec::FfiType>::FfiType
        })
    }

    let new_args = item_fn
        .sig
        .inputs
        .into_iter()
        .map(|arg| match arg {
            FnArg::Typed(mut arg) => {
                arg.ty = Box::new(ffi_ty(*arg.ty.clone())?);
                Ok(FnArg::Typed(arg))
            }
            FnArg::Receiver(..) => Err(Error::new(
                arg.span(),
                "self is not allowed in inline C function parameter",
            )),
        })
        .collect::<Result<_, _>>()?;
    item_fn.sig.inputs = new_args;

    let ret_ty = get_f_ret_ty(&item_fn)?;
    item_fn.sig.output = ReturnType::Type(Default::default(), Box::new(ffi_ty(ret_ty.clone())?));

    let body = item_fn.block;
    item_fn.block = Box::new(parse_quote! {{
        #(
            let #arg_names: #arg_tys = ::lisakmod_macros::inlinec::FromFfi::from_ffi(#arg_names);
        );*
        <#ret_ty as ::lisakmod_macros::inlinec::IntoFfi>::into_ffi(
            #body
        )
    }});

    let name = item_fn.sig.ident;
    let rust_name = format_ident!("__lisa_rust_shim_{name}_{}", get_random());
    item_fn.sig.ident = rust_name.clone();

    // Make the Rust function unsafe since we call FromFfi::from_ffi()
    item_fn.sig.unsafety = Some(Default::default());
    item_fn.sig.abi = Some(Abi {
        extern_token: Default::default(),
        name: Some(LitStr::new("C", item_fn.span())),
    });
    item_fn.vis = Visibility::Public(Default::default());

    // Emit a C prototype for the Rust function, so that the C shim can call the Rust code.
    let (_, rust_func_c_proto, _) = _make_c_func(
        None,
        &rust_name,
        None,
        &f_args,
        &f_ret_ty,
        (None, None, None),
    )?;

    let call_rust_func = concatcp(quote! {
        "return ", stringify!(#rust_name), "(",
            #(
                stringify!(#arg_names),
            )*
        ");"
    })?;

    // We create a symbol picked up by rust_exported_symbols.py
    let mut export_markers = vec![_export_symbol(rust_name)?];

    // Give a unique name to the actual C function generated by default, so it cannot conflict with
    // anything else. It will appear as "name" in the Rust universe thanks to the #[link_name]
    // attribute.
    let c_name = match attrs.link_name {
        None => format!("__lisa_c_shim_{name}_{}", get_random()),
        Some(name) => {
            export_markers.push(_export_symbol(format_ident!("{name}"))?);
            name
        }
    };

    let c_name = format_ident!("{c_name}");

    // Make a C function with the FfiType types, so that it is ABI-compatible with the Rust
    // function we created. That C function is kind of useless, except in that it is C instead of
    // Rust and therefore is a valid target for function pointers when kCFI is enabled
    // (CONFIG_CFI_CLANG).
    let c_out = make_c_func(
        Some(&name),
        &c_name,
        None,
        &f_args,
        &f_ret_ty,
        (Some(rust_func_c_proto), Some(call_rust_func), None),
    )?;

    let out = quote! {
        #(
            #export_markers
        )*

        #c_out

        // The Rust function needs to be callable by the C code, so we need a non-mangled name.
        // However, we already added a unique cookie to that name, so there is no risk of clash.
        #[no_mangle]
        #item_fn
    };
    Ok(out)
}

pub fn export_symbol(args: TokenStream) -> Result<TokenStream, Error> {
    let ident = syn::parse2::<Ident>(args)?;
    _export_symbol(ident).map(Into::into)
}

fn _export_symbol(ident: Ident) -> Result<TokenStream, Error> {
    let marker = format_ident!("__export_rust_symbol_{ident}");

    Ok(quote! {
        const _:() = {
            #[used]
            #[no_mangle]
            static #marker: () = ();
        };
    })
}
