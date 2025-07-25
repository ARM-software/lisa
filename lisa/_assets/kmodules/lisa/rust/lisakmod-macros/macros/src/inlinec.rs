/* SPDX-License-Identifier: Apache-2.0 */

use std::io::{self, Write};

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    Abi, Attribute, Error, Expr, ExprGroup, ExprLit, ExprTuple, FnArg, Generics, Ident, ItemFn,
    ItemStatic, Lit, LitStr, Meta, Pat, Path, ReturnType, Stmt, StmtMacro, Token, Type, Visibility,
    parse::Parse, parse_quote, punctuated::Punctuated, spanned::Spanned, token,
};

use crate::misc::{_dump_to_binstore, concatcp, get_random};

fn _export_symbol(sym: Ident) -> Result<TokenStream, Error> {
    Ok(quote! {
        ::lisakmod_macros::misc::export_symbol!(#sym);
    })
}

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
    f_vis: Visibility,
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
        let f_vis = item_fn.vis;
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

        let c_code = match &snippets[..] {
            [code] => Ok((None, Some(code.clone()), None)),
            [pre, code] => Ok((Some(pre.clone()), Some(code.clone()), None)),
            [pre, code, post] => Ok((Some(pre.clone()), Some(code.clone()), Some(post.clone()))),
            _ => Err(Error::new(
                span,
                "An inline C function must contain either a single string expression with the C code in it or two string expressions (one output before the function and one for the body).",
            )),
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
            f_vis,
        })
    }
}

fn make_c_func(
    rust_name: Option<&Ident>,
    c_name: &Ident,
    f_generics: Option<&Generics>,
    f_args: &[(Ident, Type)],
    f_ret_ty: &Type,
    c_attrs: Vec<TokenStream>,
    c_code: (
        Option<TokenStream>,
        Option<TokenStream>,
        Option<TokenStream>,
    ),
) -> Result<TokenStream, Error> {
    let (c_out, c_header_out, rust_out) = _make_c_func(
        rust_name, c_name, f_generics, f_args, f_ret_ty, c_attrs, c_code,
    )?;

    let c_out = [
        _dump_to_binstore(&format!("c.code.{c_name}"), c_out)?,
        _dump_to_binstore(&format!("c.header.{c_name}"), c_header_out)?,
    ];

    Ok(quote! {
        #(#c_out)*
        #rust_out
    })
}

fn _make_c_func(
    rust_name: Option<&Ident>,
    c_name: &Ident,
    f_generics: Option<&Generics>,
    f_args: &[(Ident, Type)],
    f_ret_ty: &Type,
    mut c_attrs: Vec<TokenStream>,
    c_code: (
        Option<TokenStream>,
        Option<TokenStream>,
        Option<TokenStream>,
    ),
) -> Result<(TokenStream, TokenStream, TokenStream), Error> {
    let c_name_str: String = c_name.to_string();
    let f_where = f_generics.map(|x| x.where_clause.clone());

    let c_type_of = |ty| {
        (
            quote! {
                {
                    // Use a const fn to introduce f_generics and f_where
                    const fn get #f_generics() -> &'static str #f_where {
                        (
                            <#ty as ::lisakmod_macros::inlinec::FfiType>::C_TYPE
                        )
                    }
                    get()
                }
            },
            quote! {
                {
                    // Use a const fn to introduce f_generics and f_where
                    const fn get #f_generics() -> &'static str #f_where {
                        (
                            match <#ty as ::lisakmod_macros::inlinec::FfiType>::C_HEADER {
                                Some(c_header) => c_header,
                                None => "linux/types.h",
                            }
                        )
                    }
                    get()
                }
            },
        )
    };

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
    let (args_name, args_ty): (Vec<_>, Vec<_>) = f_args.iter().cloned().unzip();
    let (c_args, c_args_header) = if c_nr_args == 0 {
        (quote! { "void" }, quote! { "" })
    } else {
        let c_args_name: Vec<_> = args_name.iter().map(|name| name.to_string()).collect();
        let c_args_commas: Vec<_> = c_args_name
            .iter()
            .enumerate()
            .map(|(i, _)| if i == (c_nr_args - 1) { "" } else { ", " })
            .collect();

        let (c_args_ty, c_args_header): (Vec<_>, Vec<_>) = args_ty.iter().map(c_type_of).unzip();
        (
            concatcp(quote! {
                #(
                    "__typeof__(" , #c_args_ty, ") ", #c_args_name,
                    #c_args_commas
                ),*
            })?,
            concatcp(quote! {
                #(
                    "#include \"" , #c_args_header, "\"\n"
                ),*
            })?,
        )
    };

    let rust_extern_args = quote! {
        #(
            #args_name : <#args_ty as ::lisakmod_macros::inlinec::FfiType>::FfiType
        ),*
    };

    let pre_c_code = pre_c_code.unwrap_or(quote! {""});
    let post_c_code = post_c_code.unwrap_or(quote! {""});

    // Disabling CFI inside the function allows us to call anything we want, including Rust
    // functions if needed.
    c_attrs.push(quote! {"__nocfi"});

    let (c_ret_ty, c_ret_header) = c_type_of(f_ret_ty);
    let c_proto = concatcp(quote! {
        #c_args_header,
        "\n",
        "#include \"", #c_ret_header, "\"",
        "\n",
        #c_ret_ty, " ", #(#c_attrs, " "),* , " ", #c_name_str, "(", #c_args , ")",
    })?;

    let c_header_out = concatcp(quote! {
        "\n#line ", #c_code_line, " \"", file!(), "\"\n",
        #c_proto, ";\n",
    })?;

    let c_funcdef = match c_code {
        Some(c_code) => concatcp(quote! {
            #c_proto, "{\n",
            "#define FUNC_RET_TYPE ", #c_ret_ty, "\n",
            "#line ", #c_code_line, " \"", file!(), "\"\n",
            #c_code, "\n",
            "#undef FUNC_RET_TYPE\n",
            "}\n",
        })?,
        None => quote! {""},
    };

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
            #[cfg(not(any(test, feature = "test")))]
            unsafe extern "C" {
                #[link_name = #c_name_str]
                fn #rust_name #f_generics(#rust_extern_args) -> <#f_ret_ty as ::lisakmod_macros::inlinec::FfiType>::FfiType #f_where;
            }

            #[cfg(any(test, feature = "test"))]
            #[allow(unused)]
            fn #rust_name #f_generics(#rust_extern_args) -> <#f_ret_ty as ::lisakmod_macros::inlinec::FfiType>::FfiType #f_where {
                ::core::panic!("extern C function are not available during tests")
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
    let f_vis = input.f_vis;

    let shim_name_str = format!("__lisa_c_shim_{name}_{:0>39}", get_random());
    let shim_name = format_ident!("{}", shim_name_str);
    let c_out = make_c_func(
        Some(&shim_name),
        &shim_name,
        Some(&f_generics),
        &f_args,
        &f_ret_ty,
        Vec::new(),
        input.c_code,
    )?;

    let (args_name, args_ty): (Vec<_>, Vec<_>) = f_args.into_iter().unzip();
    let rust_args = quote! {
        #(
            #args_name : #args_ty
        ),*
    };
    let rust_extern_call_args = quote! {
        #(
            ::lisakmod_macros::inlinec::IntoFfi::into_ffi(#args_name)
        ),*
    };

    let rust_out = quote! {
        #[inline]
        #(#f_attrs)*
        #f_vis #f_unsafety fn #name #f_generics(#rust_args) -> #f_ret_ty #f_where {
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

pub fn cconstant(args: TokenStream) -> Result<TokenStream, Error> {
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

    let (headers, expr) = match &args[..] {
        [headers, expr] => Ok((headers, expr)),
        _ => Err(Error::new(
            span,
            "Expected 2 string literals: the header #includes and the C constant expression evaluate.",
        )),
    }?;

    let (out, static_assert) = match std::env::var("LISA_EVAL_C") {
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
            let assert_cond = concatcp(quote! {"(", #expr, ") == (", #out, ")"})?;
            let out = syn::parse_str::<Expr>(out.trim())?;
            (
                quote! {Some(#out)},
                quote! {
                    // Ensure that the constant value we got from the compiler matches what we will get in
                    // the real world when running the code.
                    ::lisakmod_macros::inlinec::c_static_assert!(
                        #headers,
                        #assert_cond
                    );
                },
            )
        }
        Err(_) => (quote! {None}, quote! {}),
    };

    Ok(quote! {
        {
            #static_assert
            #out
        }
    })
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

struct CAttrs {
    export_name: Option<String>,
    no_mangle: bool,
}

fn path_to_string(path: &Path) -> String {
    path.segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

impl Parse for CAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;

        let unknown = |span| Err(Error::new(span, "Unknown argument".to_string()));

        let mut export_name = None;
        let mut no_mangle = false;
        for attr in attrs {
            match attr {
                // Meta::List(ml) => match ml.path.get_ident() {
                // Some(ident) => match &*ident.to_string() {
                // "export_name" => {
                // export_name = Some(syn::parse2::<LitStr>(ml.tokens)?.value());
                // }
                // _ => return unknown(ml.span()),
                // },
                // _ => return unknown(ml.span()),
                // },
                Meta::List(ml) => return unknown(ml.span()),
                Meta::Path(path) => {
                    let span = path.span();
                    let path = path_to_string(&path);
                    match &*path {
                        "no_mangle" => {
                            no_mangle = true;
                        }
                        _ => return unknown(span),
                    }
                }

                Meta::NameValue(name_value) => {
                    let path = path_to_string(&name_value.path);
                    let span = name_value.span();
                    let expr = name_value.value;
                    match &*path {
                        "export_name" => match expr {
                            Expr::Lit(lit) => match lit.lit {
                                Lit::Str(lit) => {
                                    export_name = Some(lit.value());
                                }
                                lit => {
                                    return Err(Error::new(
                                        lit.span(),
                                        "Expected string literal".to_string(),
                                    ));
                                }
                            },
                            expr => {
                                return Err(Error::new(
                                    expr.span(),
                                    "Expected string literal".to_string(),
                                ));
                            }
                        },
                        _ => return unknown(span),
                    };
                }
            }
        }

        if export_name.is_some() && no_mangle {
            Err(Error::new(
                input.span(),
                "\"no_mangle\" and \"export_name\" attributes cannot be specified at the same time"
                    .to_string(),
            ))
        } else {
            Ok(CAttrs {
                export_name,
                no_mangle,
            })
        }
    }
}

pub fn cexport(attrs: TokenStream, code: TokenStream) -> Result<TokenStream, Error> {
    let input = syn::parse2::<CExportInput>(code)?;
    let attrs = syn::parse2::<CAttrs>(attrs)?;
    let mut item_fn = input.item_fn;
    let f_args = get_f_args(&item_fn)?;
    let f_ret_ty = get_f_ret_ty(&item_fn)?;
    let (args_name, args_ty): (Vec<_>, Vec<_>) = f_args.iter().cloned().unzip();

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
            let #args_name: #args_ty = ::lisakmod_macros::inlinec::FromFfi::from_ffi(#args_name);
        );*
        <#ret_ty as ::lisakmod_macros::inlinec::IntoFfi>::into_ffi(
            #body
        )
    }});

    let name = item_fn.sig.ident;
    let rust_name = format_ident!("__lisa_rust_shim_{name}_{:0>39}", get_random());
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
        Vec::new(),
        (None, None, None),
    )?;

    let arg_list = args_name
        .iter()
        .map(|arg| arg.to_string())
        .intersperse(", ".to_string());
    let call_rust_func = concatcp(quote! {
        "return ", stringify!(#rust_name), "(", #(#arg_list,)* ");"
    })?;

    // We create a symbol picked up by rust_exported_symbols.py
    let mut export_markers = vec![_export_symbol(rust_name)?];

    // Give a unique name to the actual C function generated by default, so it cannot conflict with
    // anything else. It will appear as "name" in the Rust universe thanks to the #[export_name]
    // attribute.
    let c_name = match attrs.export_name {
        None => match attrs.no_mangle {
            true => name.to_string(),
            false => format!("__lisa_c_shim_{name}_{:0>39}", get_random()),
        },
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
        vec![
            // In order for Rust code to be able to take the address of the C shim and then pass it
            // back to some other C code (e.g. by filling a function pointer callback in a struct),
            // we need to have __attribute__((cfi_canonical_jump_table)) on the function with
            // pre-kCFI kernels (e.g. 5.15). See details at:
            // https://clang.llvm.org/docs/ControlFlowIntegrity.html#fsanitize-cfi-canonical-jump-tables
            quote! {"__cficanonical"},
        ],
        (Some(rust_func_c_proto), Some(call_rust_func), None),
    )?;

    let out = quote! {
        #(
            #export_markers
        )*

        #c_out

        // The Rust function needs to be callable by the C code, so we need a non-mangled name.
        // However, we already added a unique cookie to its name if we want to keep it private, so
        // there is no risk of clash.
        #[unsafe(no_mangle)]
        #item_fn
    };
    Ok(out)
}

struct CStaticInput {
    name: Ident,
    c_code: (
        Option<TokenStream>,
        Option<TokenStream>,
        Option<TokenStream>,
    ),
    static_attrs: Vec<Attribute>,
    ty: Type,
    vis: Visibility,
}

impl Parse for CStaticInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let item_static: ItemStatic = input.parse()?;
        let span = item_static.span();
        let static_attrs = item_static.attrs;
        let name = item_static.ident;
        let ty = *item_static.ty;
        let vis = item_static.vis;

        let snippets = match *item_static.expr {
            Expr::Tuple(ExprTuple { elems, .. }) => {
                let mut snippets = Vec::new();
                for item in elems {
                    snippets.push(match item {
                        Expr::Lit(lit) => match lit.lit {
                            Lit::Str(lit) => Ok(quote! {#lit}),
                            lit => Err(Error::new(
                                lit.span(),
                                "Expected string literal or macro invocation",
                            )),
                        },
                        Expr::Macro(mac) => Ok(quote! {#mac}),
                        expr => Err(Error::new(
                            expr.span(),
                            "Expected string literal or macro invocation",
                        )),
                    }?);
                }
                Ok(snippets)
            }
            expr => Err(Error::new(
                expr.span(),
                "C static must be defined by a tuple of string constants",
            )),
        }?;

        let c_code = match &snippets[..] {
            [code] => Ok((None, Some(code.clone()), None)),
            [pre, code] => Ok((Some(pre.clone()), Some(code.clone()), None)),
            [pre, code, post] => Ok((Some(pre.clone()), Some(code.clone()), Some(post.clone()))),
            _ => Err(Error::new(
                span,
                "An inline C static must contain either a single string expression with the C code in it or two string expressions (one output before the function and one for the definition). The C macro STATIC_VARIABLE must be used to refer to the name of the C variable.",
            )),
        }?;

        Ok(Self {
            name,
            c_code,
            static_attrs,
            ty,
            vis,
        })
    }
}

pub fn cstatic(attrs: TokenStream, code: TokenStream) -> Result<TokenStream, Error> {
    let input = syn::parse2::<CStaticInput>(code)?;
    let attrs = syn::parse2::<CAttrs>(attrs)?;

    let name = input.name;
    let static_attrs = input.static_attrs;
    let ty = input.ty;
    let vis = input.vis;
    let (pre_c_code, c_code, post_c_code) = input.c_code;

    let empty = quote! {""};
    let (pre_c_code, pre_c_code_line) = match &pre_c_code {
        Some(tokens) => (tokens, tokens.span().start().line),
        None => (&empty, 0),
    };

    let c_code_line = match &c_code {
        Some(tokens) => tokens.span().start().line,
        None => pre_c_code_line,
    };
    let (post_c_code, post_c_code_line) = match &post_c_code {
        Some(tokens) => (tokens, tokens.span().start().line),
        None => (&empty, c_code_line),
    };

    let pre_c_code_line = pre_c_code_line.to_string();
    let c_code_line = c_code_line.to_string();
    let post_c_code_line = post_c_code_line.to_string();

    let c_name = match attrs.export_name {
        None => match attrs.no_mangle {
            true => name.to_string(),
            false => format!("__lisa_c_shim_{name}_{:0>39}", get_random()),
        },
        Some(name) => name,
    };
    let c_typecheck_name = format!("{c_name}_typecheck_{:0>39}", get_random());

    let c_type = quote! { <#ty as ::lisakmod_macros::inlinec::FfiType>::C_TYPE };
    let c_header = quote! {
        match <#ty as ::lisakmod_macros::inlinec::FfiType>::C_HEADER {
            Some(c_header) => c_header,
            None => "linux/types.h",
        }
    };

    let section = format!("c.code.{c_name}");
    let c_out = concatcp(quote! {
        "#define STATIC_VARIABLE ", #c_name, "\n",
        "\n#line ", #pre_c_code_line, " \"", file!(), "\"\n",
        #pre_c_code,
        "\n#line ", #c_code_line, " \"", file!(), "\"\n",
        "#include \"", #c_header, "\"\n",
        "\n#line ", #c_code_line, " \"", file!(), "\"\n",
        #c_code,

        // Check that the variable created by the user has the correct type by taking its address
        // in a pointer to the type we expect.
        "static  __attribute__ ((unused)) __typeof__(", #c_type, ") *", #c_typecheck_name, " = &", #c_name, ";\n",

        "\n#line ", #post_c_code_line, " \"", file!(), "\"\n",
        #post_c_code,
        "#undef STATIC_VARIABLE\n"
    })?;

    let c_out = _dump_to_binstore(&section, c_out)?;

    Ok(quote! {
        #c_out

        const _: () = {
            const fn check_ffi_type<T>()
            where
                T: ::lisakmod_macros::inlinec::FfiType<FfiType = T>,
                T: ::lisakmod_macros::inlinec::FromFfi,
                T: ::lisakmod_macros::inlinec::IntoFfi
            {}
            check_ffi_type::<#ty>();
        };

        unsafe extern "C" {
            #[link_name = #c_name]
            #(#static_attrs)*
            #vis static #name: #ty;
        }

        #[cfg(any(test, feature = "test"))]
        const _: () = {
            #[unsafe(export_name = #c_name)]
            static placeholder: u8 = 0;
        };
    })
}
