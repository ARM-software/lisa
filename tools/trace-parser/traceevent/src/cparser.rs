// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2024, ARM Limited and contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use core::{
    fmt,
    fmt::{Debug, Formatter},
    ops::Deref,
    str::from_utf8,
};
use std::{rc::Rc, string::String as StdString, sync::Arc};

use bytemuck::cast_slice;
use itertools::Itertools as _;
use nom::{
    branch::alt,
    bytes::complete::{is_not, tag, take},
    character::complete::{alpha1, alphanumeric1, char, u64 as dec_u64},
    combinator::{all_consuming, cut, fail, map_res, not, opt, recognize, success},
    error::{context, FromExternalError},
    multi::{fold_many1, many0, many0_count, many1, many_m_n, separated_list0, separated_list1},
    number::complete::u8,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    AsBytes, Finish as _, Parser,
};
use smartstring::alias::String;

use crate::{
    cinterp::{
        new_dyn_evaluator, BasicEnv, Bitmap, CompileEnv, CompileError, EvalEnv, EvalError,
        Evaluator, InterpError, ParseEnv, Value,
    },
    error::convert_err_impl,
    grammar::{grammar, PackratGrammar as _},
    header::{Abi, Endianness, FileSize, Identifier, LongSize},
    parser::{
        error, failure, hex_u64, lexeme, map_res_cut, parenthesized, success_with, FromParseError,
        VerboseParseError,
    },
    scratch::{OwnedScratchBox, ScratchVec},
    str::Str,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Void,

    Bool,

    U8,
    I8,

    U16,
    I16,

    U32,
    I32,

    U64,
    I64,

    // Opaque type of variables
    Variable(Identifier),

    // Complete black box used in cases where we want to completely hide any
    // information about the type.
    Unknown,

    Typedef(Box<Type>, Identifier),
    Enum(Box<Type>, Identifier),
    Struct(Identifier),
    Union(Identifier),
    Function(Box<Type>, Vec<ParamDeclaration>),

    Pointer(Box<Type>),

    Array(Box<Type>, ArrayKind),
    DynamicScalar(Box<Type>, DynamicKind),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicKind {
    Dynamic,
    DynamicRel,
}

#[derive(Debug, Clone)]
pub enum ArrayKind {
    Fixed(Result<FileSize, Box<InterpError>>),
    Dynamic(DynamicKind),
    ZeroLength,
}

impl PartialEq for ArrayKind {
    fn eq(&self, other: &ArrayKind) -> bool {
        match (self, other) {
            (ArrayKind::Fixed(Ok(x1)), ArrayKind::Fixed(Ok(x2))) => x1 == x2,
            (ArrayKind::Fixed(Err(_)), ArrayKind::Fixed(Err(_))) => true,
            (ArrayKind::Dynamic(kind1), ArrayKind::Dynamic(kind2)) => kind1 == kind2,
            (ArrayKind::ZeroLength, ArrayKind::ZeroLength) => true,
            _ => false,
        }
    }
}

impl Eq for ArrayKind {}

#[derive(Clone)]
pub struct Declarator {
    identifier: Option<Identifier>,
    // Use Rc<> so that Declarator can be Clone, which is necessary to for the
    // packrat cache.
    modify_typ: Rc<dyn Fn(Type) -> Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    pub identifier: Identifier,
    pub typ: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamDeclaration {
    pub identifier: Option<Identifier>,
    pub typ: Type,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Uninit,
    Variable(Type, Identifier),

    InitializerList(Vec<Expr>),
    DesignatedInitializer(Box<Expr>, Box<Expr>),
    CompoundLiteral(Type, Vec<Expr>),

    IntConstant(Type, u64),
    CharConstant(Type, u64),
    EnumConstant(Type, Identifier),
    StringLiteral(String),

    Addr(Box<Expr>),
    Deref(Box<Expr>),
    Plus(Box<Expr>),
    Minus(Box<Expr>),
    Tilde(Box<Expr>),
    Bang(Box<Expr>),
    Cast(Type, Box<Expr>),
    SizeofType(Type),
    SizeofExpr(Box<Expr>),
    PreInc(Box<Expr>),
    PreDec(Box<Expr>),
    PostInc(Box<Expr>),
    PostDec(Box<Expr>),

    MemberAccess(Box<Expr>, Identifier),
    FuncCall(Box<Expr>, Vec<Expr>),
    Subscript(Box<Expr>, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),

    ExtensionMacro(Arc<ExtensionMacroDesc>),
    ExtensionMacroCall(ExtensionMacroCall),

    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),

    Eq(Box<Expr>, Box<Expr>),
    NEq(Box<Expr>, Box<Expr>),
    LoEq(Box<Expr>, Box<Expr>),
    HiEq(Box<Expr>, Box<Expr>),
    Hi(Box<Expr>, Box<Expr>),
    Lo(Box<Expr>, Box<Expr>),

    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),

    LShift(Box<Expr>, Box<Expr>),
    RShift(Box<Expr>, Box<Expr>),
    BitAnd(Box<Expr>, Box<Expr>),
    BitOr(Box<Expr>, Box<Expr>),
    BitXor(Box<Expr>, Box<Expr>),

    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    CommaExpr(Vec<Expr>),

    Evaluated(Type, Value<'static>),
}

impl Expr {
    pub(crate) fn record_field(field: Identifier) -> Self {
        Expr::MemberAccess(
            // The extra Deref(Addr(...)) layer is important as it accurately
            // matches what the C sources would be, i.e. REC->field. This allows
            // comparing this expression for equality with parsed source to
            // check if we got a typical field access.
            Box::new(Expr::Deref(Box::new(Expr::Addr(Box::new(Expr::Variable(
                Type::Variable("REC".into()),
                "REC".into(),
            )))))),
            field,
        )
    }
    #[inline]
    pub(crate) fn is_record_field(&self, field: &str) -> bool {
        // This must match the record_field() definition
        if let Expr::MemberAccess(expr, _field) = self {
            if _field == field {
                if let Expr::Deref(expr) = (*expr).deref() {
                    if let Expr::Addr(expr) = (*expr).deref() {
                        if let Expr::Variable(Type::Variable(name), name2) = (*expr).deref() {
                            if name == "REC" && name2 == "REC" {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }
}

#[derive(Clone)]
pub struct ExtensionMacroCall {
    pub args: Vec<u8>,
    pub desc: Arc<ExtensionMacroDesc>,
    pub compiler: ExtensionMacroCallCompiler,
}

impl Debug for ExtensionMacroCall {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.write_fmt(format_args!(
            "ExtensionMacroCall{{ typ: {:?}, call: {}({}) }}",
            &self.compiler.ret_typ,
            &self.desc.name,
            &StdString::from_utf8_lossy(&self.args)
        ))
    }
}

impl PartialEq<Self> for ExtensionMacroCall {
    fn eq(&self, other: &Self) -> bool {
        self.compiler.ret_typ == other.compiler.ret_typ
            && self.desc == other.desc
            && self.args == other.args
    }
}

impl Eq for ExtensionMacroCall {}

type Compiler = dyn for<'ceref, 'ce> Fn(
        &'ceref (dyn CompileEnv<'ce> + 'ceref),
    ) -> Result<Box<dyn Evaluator>, CompileError>
    + Send
    + Sync;

#[derive(Clone)]
pub struct ExtensionMacroCallCompiler {
    pub ret_typ: Type,
    pub compiler: Arc<Compiler>,
}

type FunctionLikeExtensionMacroParser = Box<
    dyn for<'a> Fn(&'a dyn ParseEnv, &str) -> Result<ExtensionMacroCallCompiler, CParseError>
        + Send
        + Sync,
>;

pub(crate) enum ExtensionMacroKind {
    FunctionLike {
        parser: FunctionLikeExtensionMacroParser,
    },
    ObjectLike {
        value: Value<'static>,
        typ: Type,
    },
}

pub struct ExtensionMacroDesc {
    name: Identifier,
    pub(crate) kind: ExtensionMacroKind,
}

impl ExtensionMacroDesc {
    #[inline]
    pub fn new_function_like(name: Identifier, parser: FunctionLikeExtensionMacroParser) -> Self {
        ExtensionMacroDesc {
            name,
            kind: ExtensionMacroKind::FunctionLike { parser },
        }
    }

    #[inline]
    pub fn new_object_like(name: Identifier, typ: Type, value: Value<'static>) -> Self {
        ExtensionMacroDesc {
            name,
            kind: ExtensionMacroKind::ObjectLike { value, typ },
        }
    }
}

impl PartialEq<Self> for ExtensionMacroDesc {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for ExtensionMacroDesc {}

impl Debug for ExtensionMacroDesc {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("ExtensionMacroDesc")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CParseError {
    #[error("Could not decode UTF-8 string: {0}")]
    DecodeUtf8(StdString),
    #[error("No identifier found in a declaration of type {0:?}")]
    DeclarationWithoutIdentifier(Type),
    #[error("Found nested __data_loc in type: {0:?}")]
    NestedDataLoc(Type),
    #[error("The outermost type of a dynamic array must be an array: {0:?}")]
    DataLocArrayNotArray(Type),

    // Kind of redundant with DataLocWithoutIdentifier but the different error
    // message might be genuinely important considered this is not valid ISO C
    // syntax to start with.
    #[error(
        "A __data_loc array declaration is expected to have an identifier following the last []: {0:?}"
    )]
    DataLocArrayWithoutIdentifier(Type),
    #[error("Found no identifier in the scalar __data_loc declaration")]
    DataLocWithoutIdentifier,
    #[error(
        "Found ambiguous identifiers in the scalar __data_loc declaration: \"{0}\" or \"{1}\""
    )]
    DataLocAmbiguousIdentifier(Identifier, Identifier),

    #[error("Invalid type name (incompatible int/long/short/char usage)")]
    InvalidTypeName,

    #[error(
        "Invalid integer constant ({0}): the value does not fit in the range of any of the allowed types"
    )]
    InvalidIntegerConstant(u64),

    #[error("Character value is out of range ({0}), only 8 bit values are supported")]
    CharOutOfRange(u64),

    #[error("Invalid variable identifier \"{0}\". Only the REC identifier is recognized as a variable, every other identifier is assumed to be an enumeration constant")]
    InvalidVariableIdentifier(Identifier),

    #[error("Could not guess the type of the expression: {0:?}")]
    CouldNotGuessType(Expr),

    #[error("Unsupported C construct: {0}")]
    UnsupportedConstruct(StdString),

    #[error("Object-like macro {0:?} cannot be called since it is not a function-like macro")]
    CannotCallObjectLikeMacro(Identifier),

    #[error("This snippet might use unsupported C constructs, could not recognize input: {0}")]
    ParseError(VerboseParseError),

    #[error("Could not interpret expression: {0}")]
    InterpError(Box<InterpError>),
}

convert_err_impl!(InterpError, InterpError, CParseError);
convert_err_impl!(EvalError, InterpError, CParseError);
convert_err_impl!(CompileError, InterpError, CParseError);

impl<I, I2> FromParseError<I, nom::error::VerboseError<I2>> for CParseError
where
    I: AsRef<[u8]>,
    I2: AsRef<[u8]>,
{
    fn from_parse_error(input: I, err: &nom::error::VerboseError<I2>) -> Self {
        CParseError::ParseError(VerboseParseError::new(input, err))
    }
}

fn eval_unsigned(expr: Expr, abi: &Abi) -> Result<FileSize, InterpError> {
    expr.eval_const(abi, |res| {
        let x = res?;
        match x {
            Value::U64Scalar(x) => Ok(x),
            Value::I64Scalar(x) if x > 0 => Ok(x.unsigned_abs()),
            val => Err(InterpError::EvalError(Box::new(EvalError::IllegalType(
                val.into_static().ok(),
            )))),
        }
    })
}

macro_rules! apply_cparser {
    ($penv:expr, $input:expr, $parser:expr) => {{
        let ctx = CGrammarCtx::new($penv);
        CGrammar::apply_rule($parser, $input.as_bytes(), &ctx).map(move |(_, x)| x)
    }};
}

fn print_array_hex(separator: &'static str) -> Result<ExtensionMacroKind, CParseError> {
    Ok(ExtensionMacroKind::FunctionLike {
        parser: Box::new(move |penv, input| {
            apply_cparser!(
                penv,
                input,
                map_res_cut(
                    tuple((
                        // Array to format
                        CGrammar::assignment_expr(),
                        lexeme(char(',')),
                        // Array size
                        CGrammar::assignment_expr(),
                    )),
                    |(val, _, array_size)| {
                        let compiler = Arc::new(move |cenv: &dyn CompileEnv<'_>| {
                            let cval = val.clone().compile(&cenv)?;
                            let carray_size = array_size.clone().compile(&cenv)?;

                            let eval = new_dyn_evaluator(move |env: &_| {
                                let array_size = match carray_size.eval(env)? {
                                    Value::U64Scalar(x) => Ok(x),
                                    Value::I64Scalar(x) => Ok(x as u64),
                                    val => Err(EvalError::IllegalType(val.into_static().ok())),
                                }?;

                                macro_rules! print_array {
                                    ($arr:expr) => {{
                                        let writer = OwnedScratchBox::new_in(
                                            move |out: &mut dyn fmt::Write| {
                                                let mut closure = || -> Result<(), fmt::Error> {
                                                    let mut first = true;
                                                    for x in $arr
                                                        .into_iter()
                                                        .take(array_size.try_into().unwrap())
                                                    {
                                                        if !first {
                                                            out.write_str(separator)?;
                                                        }
                                                        write!(out, "{:02x}", x)?;
                                                        first = false;
                                                    }
                                                    Ok(())
                                                };
                                                closure()
                                                    .expect("could not render hex array to string")
                                            },
                                            env.scratch(),
                                        );

                                        Ok(Value::Str(Str::new_procedural(writer)))
                                    }};
                                }

                                match cval.eval(env)? {
                                    Value::Raw(_, arr) => print_array!(arr.iter()),
                                    Value::Str(s) => print_array!(s.bytes().chain([0])),

                                    Value::U8Array(arr) => print_array!(arr.iter()),
                                    Value::I8Array(arr) => print_array!(arr.iter()),

                                    val => Err(EvalError::IllegalType(val.into_static().ok())),
                                }
                            });
                            Ok(eval)
                        });

                        Ok(ExtensionMacroCallCompiler {
                            ret_typ: Type::Pointer(Box::new(penv.abi().char_typ())),
                            compiler,
                        })
                    },
                )
            )
        }),
    })
}

fn print_symbolic<const EXACT_MATCH: bool>() -> Result<ExtensionMacroKind, CParseError> {
    Ok(ExtensionMacroKind::FunctionLike {
        parser: Box::new(move |penv, input| {
            apply_cparser!(
                penv,
                input,
                map_res_cut(
                    tuple((
                        // value to format
                        CGrammar::assignment_expr(),
                        lexeme(char(',')),
                        // delimiter
                        move |input| {
                            if EXACT_MATCH {
                                success(Expr::Uninit).parse(input)
                            } else {
                                terminated(CGrammar::assignment_expr(), lexeme(char(',')))
                                    .parse(input)
                            }
                        },
                        // flags
                        terminated(
                            separated_list0(
                                lexeme(char(',')),
                                delimited(
                                    // E.g. {(1 << 1), "I_DIRTY_DATASYNC"}
                                    lexeme(char('{')),
                                    separated_pair(
                                        CGrammar::assignment_expr(),
                                        lexeme(char(',')),
                                        CGrammar::assignment_expr(),
                                    ),
                                    lexeme(char('}')),
                                ),
                            ),
                            opt(lexeme(char(','))),
                        ),
                    )),
                    |(val, _, delim, flags)| {
                        let flags = flags
                            .into_iter()
                            .filter_map(|(mask, flag)| -> Option<Result<_, _>> {
                                || -> Result<Option<_>, _> {
                                    let mask: u64 = mask.eval_const(penv.abi(), |x| match x? {
                                        Value::U64Scalar(x) => Ok(x),
                                        Value::I64Scalar(x) => Ok(x as u64),
                                        val => Err(InterpError::EvalError(Box::new(
                                            EvalError::IllegalType(val.into_static().ok()),
                                        ))),
                                    })?;

                                    let flag: Option<String> =
                                        flag.eval_const(penv.abi(), |s| {
                                            match s? {
                                                // We can end up with a null pointer value, which is
                                                // simply the terminator of the array. It is normally
                                                // provided by __print_symbolic() itself but some
                                                // events specify it explicitly themselves, especially
                                                // when the array is shared with some other formatting
                                                // code.
                                                Value::U64Scalar(0) | Value::I64Scalar(0) => {
                                                    Ok(None)
                                                }
                                                s => match s.to_str() {
                                                    Some(s) => Ok(Some(s.into())),
                                                    None => Err(InterpError::EvalError(Box::new(
                                                        EvalError::IllegalType(
                                                            s.into_static().ok(),
                                                        ),
                                                    ))),
                                                },
                                            }
                                        })?;
                                    match flag {
                                        Some(flag) => Ok(Some((mask, flag))),
                                        None => Ok(None),
                                    }
                                }()
                                .transpose()
                            })
                            .collect::<Result<Vec<_>, InterpError>>()?;

                        let compiler = Arc::new(move |cenv: &dyn CompileEnv<'_>| {
                            let cval = val.clone().compile(&cenv)?;
                            #[allow(clippy::type_complexity)]
                            let cdelim: Box<
                                dyn Fn(&dyn EvalEnv) -> Result<String, EvalError> + Send + Sync,
                            > = if EXACT_MATCH {
                                Box::new(|_env| Ok("".into()))
                            } else {
                                let cdelim = delim.clone().compile(&cenv)?;
                                Box::new(move |env| {
                                    let cdelim = cdelim.eval(env)?;
                                    let cdelim = cdelim.deref_ptr(env)?;
                                    match cdelim.to_str() {
                                        Some(s) => Ok(s.into()),
                                        None => {
                                            Err(EvalError::IllegalType(cdelim.into_static().ok()))
                                        }
                                    }
                                })
                            };

                            let flags = flags.clone();

                            let eval = new_dyn_evaluator(move |env: &_| {
                                let val = match cval.eval(env)? {
                                    Value::U64Scalar(x) => Ok(x),
                                    Value::I64Scalar(x) => Ok(x as u64),
                                    val => Err(EvalError::IllegalType(val.into_static().ok())),
                                }?;
                                let delim = cdelim(env)?;
                                let flags = flags.clone();

                                let writer = OwnedScratchBox::new_in(
                                    move |out: &mut dyn fmt::Write| {
                                        let mut closure = || -> Result<(), fmt::Error> {
                                            let mut first = true;
                                            let mut remaining = val;
                                            for (flag, s) in &flags {
                                                let found = if EXACT_MATCH {
                                                    val == *flag
                                                } else {
                                                    (val & flag) != 0
                                                };

                                                if found {
                                                    if !first {
                                                        write!(out, "{delim}")?;
                                                    }
                                                    write!(out, "{s}")?;

                                                    remaining &= !*flag;
                                                    first = false;

                                                    if EXACT_MATCH && found {
                                                        break;
                                                    }
                                                };
                                            }
                                            if remaining != 0 {
                                                if !first {
                                                    write!(out, "{delim}")?;
                                                }
                                                write!(out, "{remaining:#x}")?;
                                            }
                                            Ok(())
                                        };
                                        closure()
                                            .expect("could not render symbolic values to string")
                                    },
                                    env.scratch(),
                                );

                                // let f = crate::scratch::OwnedScratchBox_as_dyn!(writer, StringProducer);
                                // Ok(Value::Str(Str::new_owned("foo".into())))
                                Ok(Value::Str(Str::new_procedural(writer)))
                            });
                            Ok(eval)
                        });

                        Ok(ExtensionMacroCallCompiler {
                            ret_typ: Type::Pointer(Box::new(penv.abi().char_typ())),
                            compiler,
                        })
                    },
                )
            )
        }),
    })
}

fn object_like_macro(typ: Type, value: Value<'static>) -> Result<ExtensionMacroKind, CParseError> {
    Ok(ExtensionMacroKind::ObjectLike { value, typ })
}

fn resolve_extension_macro(name: &str) -> Result<ExtensionMacroKind, CParseError> {
    // TODO: support functions in https://elixir.free-electrons.com/linux/v6.6.9/source/include/uapi/linux/swab.h
    match name {
        "true" => object_like_macro(Type::Bool, Value::I64Scalar(1)),
        "false" => object_like_macro(Type::Bool, Value::I64Scalar(0)),
        "TASK_COMM_LEN" => object_like_macro(Type::I32, Value::I64Scalar(16)),

        "__builtin_expect" => Ok(ExtensionMacroKind::FunctionLike {
            parser: Box::new(move |penv, input| {
                apply_cparser!(
                    penv,
                    input,
                    map_res_cut(CGrammar::assignment_expr(), |expr| {
                        let ret_typ = expr.typ(penv)?;
                        Ok(ExtensionMacroCallCompiler {
                            ret_typ,
                            compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                expr.clone().compile(&cenv)
                            }),
                        })
                    })
                )
            }),
        }),

        "__builtin_constant_p" => Ok(ExtensionMacroKind::FunctionLike {
            parser: Box::new(move |_penv, _input| {
                Ok(ExtensionMacroCallCompiler {
                    ret_typ: Type::I32,
                    compiler: Arc::new(move |_| {
                        // As per the doc, 0 is an acceptable return value in any context:
                        // > A return of 0 does not indicate that the expression is not a
                        // > constant, but merely that GCC cannot prove it is a constant within
                        // > the constraints of the active set of optimization options.
                        //
                        // https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fconstant_005fp
                        let eval = new_dyn_evaluator(move |_| Ok(Value::I64Scalar(0)));
                        Ok(eval)
                    }),
                })
            }),
        }),

        "__builtin_choose_expr" => {
            Ok(ExtensionMacroKind::FunctionLike {
                parser: Box::new(move |penv, input| {
                    apply_cparser!(
                        penv,
                        input,
                        map_res_cut(
                            tuple((
                                // Condition
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Expr if condition is true
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Expr if condition is false
                                CGrammar::assignment_expr(),
                            )),
                            |(cond, _, if_true, _, if_false)| {
                                let env = BasicEnv::new(penv);
                                let cond = cond.compile(&env)?;
                                let cond = match cond.eval(&env)? {
                                    Value::U64Scalar(x) => Ok(x != 0),
                                    Value::I64Scalar(x) => Ok(x != 0),
                                    val => Err(EvalError::IllegalType(val.into_static().ok())),
                                }?;

                                let ret_expr = if cond { if_true } else { if_false };
                                let ret_typ = ret_expr.typ(penv)?;

                                Ok(ExtensionMacroCallCompiler {
                                    ret_typ,
                                    compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                        ret_expr.clone().compile(&cenv)
                                    }),
                                })
                            }
                        )
                    )
                }),
            })
        }

        "__print_ns_to_secs" => Ok(ExtensionMacroKind::FunctionLike {
            parser: Box::new(move |penv, input| {
                apply_cparser!(
                    penv,
                    input,
                    map_res_cut(CGrammar::assignment_expr(), |expr| {
                        Ok(ExtensionMacroCallCompiler {
                            ret_typ: Type::U64,
                            compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                let cexpr = expr.clone().compile(&cenv)?;

                                let eval =
                                    new_dyn_evaluator(move |env: &_| match cexpr.eval(env)? {
                                        Value::U64Scalar(x) => {
                                            Ok(Value::U64Scalar(x / 1_000_000_000))
                                        }
                                        Value::I64Scalar(x) => {
                                            Ok(Value::I64Scalar(x / 1_000_000_000))
                                        }
                                        val => Err(EvalError::IllegalType(val.into_static().ok())),
                                    });
                                Ok(eval)
                            }),
                        })
                    })
                )
            }),
        }),
        "__print_ns_without_secs" => Ok(ExtensionMacroKind::FunctionLike {
            parser: Box::new(move |penv, input| {
                apply_cparser!(
                    penv,
                    input,
                    map_res_cut(CGrammar::assignment_expr(), |expr| {
                        Ok(ExtensionMacroCallCompiler {
                            ret_typ: Type::U32,
                            compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                let cexpr = expr.clone().compile(&cenv)?;

                                let eval =
                                    new_dyn_evaluator(move |env: &_| match cexpr.eval(env)? {
                                        Value::U64Scalar(x) => {
                                            Ok(Value::U64Scalar(x % 1_000_000_000))
                                        }
                                        Value::I64Scalar(x) => {
                                            Ok(Value::I64Scalar(x % 1_000_000_000))
                                        }
                                        val => Err(EvalError::IllegalType(val.into_static().ok())),
                                    });
                                Ok(eval)
                            }),
                        })
                    })
                )
            }),
        }),
        "__get_sockaddr" | "__get_sockaddr_rel" => Ok(ExtensionMacroKind::FunctionLike {
            parser: Box::new(move |penv, input| {
                apply_cparser!(
                    penv,
                    input,
                    map_res_cut(CGrammar::identifier(), |field| {
                        let ret_typ = penv.field_typ(&field)?;
                        Ok(ExtensionMacroCallCompiler {
                            ret_typ,
                            compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                let cexpr = Expr::record_field(field.clone()).compile(&cenv)?;

                                let eval =
                                    new_dyn_evaluator(move |env: &_| match cexpr.eval(env)? {
                                        val @ Value::U8Array(_) => Ok(val),
                                        val => Err(EvalError::IllegalType(val.into_static().ok())),
                                    });
                                Ok(eval)
                            }),
                        })
                    })
                )
            }),
        }),
        "__get_dynamic_array" | "__get_rel_dynamic_array" | "__get_str" | "__get_rel_str" => {
            Ok(ExtensionMacroKind::FunctionLike {
                parser: Box::new(move |penv, input| {
                    apply_cparser!(
                        penv,
                        input,
                        map_res_cut(CGrammar::identifier(), |field| {
                            let ret_typ = penv.field_typ(&field)?;
                            Ok(ExtensionMacroCallCompiler {
                                ret_typ,
                                compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                    // Compile "__get_str(field)" as "REC->field", since the compiler
                                    // of REC->field will take care of getting the value and present
                                    // it as an array already.
                                    Expr::record_field(field.clone()).compile(&cenv)
                                }),
                            })
                        })
                    )
                }),
            })
        }
        "__get_dynamic_array_len" | "__get_rel_dynamic_array_len" => {
            Ok(ExtensionMacroKind::FunctionLike {
                parser: Box::new(move |penv, input| {
                    apply_cparser!(
                        penv,
                        input,
                        map_res_cut(CGrammar::identifier(), |field| {
                            Ok(ExtensionMacroCallCompiler {
                                ret_typ: penv.abi().long_typ(),
                                compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                    let expr = Expr::record_field(field.clone()).compile(&cenv)?;
                                    Ok(new_dyn_evaluator(move |env: &_| {
                                        match expr.eval(env)? {
                                            Value::Raw(_, arr) => Ok(arr.len()),
                                            Value::Bitmap(bitmap) => Ok(bitmap.len()),
                                            Value::Str(s) => Ok(s.len() + 1),

                                            Value::U8Array(arr) => Ok(arr.len()),
                                            Value::I8Array(arr) => Ok(arr.len()),

                                            Value::U16Array(arr) => Ok(arr.len() * 2),
                                            Value::I16Array(arr) => Ok(arr.len() * 2),

                                            Value::U32Array(arr) => Ok(arr.len() * 4),
                                            Value::I32Array(arr) => Ok(arr.len() * 4),

                                            Value::U64Array(arr) => Ok(arr.len() * 8),
                                            Value::I64Array(arr) => Ok(arr.len() * 8),

                                            val => {
                                                Err(EvalError::IllegalType(val.into_static().ok()))
                                            }
                                        }
                                        .map(|size| Value::U64Scalar(size.try_into().unwrap()))
                                    }))
                                }),
                            })
                        })
                    )
                }),
            })
        }
        "__get_bitmask" | "__get_rel_bitmask" | "__get_cpumask" | "__get_rel_cpumask" => {
            Ok(ExtensionMacroKind::FunctionLike {
                parser: Box::new(move |penv, input| {
                    apply_cparser!(
                        penv,
                        input,
                        map_res_cut(CGrammar::identifier(), |field| {
                            Ok(ExtensionMacroCallCompiler {
                                ret_typ: penv.abi().char_typ(),
                                compiler: Arc::new(move |cenv: &dyn CompileEnv| {
                                    let bitmap =
                                        Expr::record_field(field.clone()).compile(&cenv)?;
                                    let abi = cenv.abi().clone();
                                    Ok(new_dyn_evaluator(move |env: &_| {
                                        macro_rules! to_string {
                                            ($bitmap:expr) => {{
                                                let writer = OwnedScratchBox::new_in(
                                                    move |out: &mut dyn fmt::Write| {
                                                        write!(out, "{}", $bitmap).expect(
                                                            "Could not render bitmap to string",
                                                        )
                                                    },
                                                    env.scratch(),
                                                );

                                                Ok(Value::Str(Str::new_procedural(writer)))
                                            }};
                                        }

                                        let bitmap = bitmap.eval(env)?;
                                        match bitmap {
                                            Value::Bitmap(bitmap) => to_string!(bitmap),

                                            // Older kernels had cpumasks declared as a dynamic array of unsigned long:
                                            // __data_loc unsigned long[] target_cpus;
                                            //
                                            // More recent kernels now declare it this way:
                                            // __data_loc cpumask_t target_cpus
                                            //
                                            // Note that even recent kernels still use
                                            // unsigned long[] for other bitmasks.
                                            //
                                            // https://bugzilla.kernel.org/show_bug.cgi?id=217447
                                            Value::U32Array(arr)
                                                if abi.long_size == LongSize::Bits32 =>
                                            {
                                                let abi = abi.clone();
                                                to_string!(Bitmap::from_bytes(
                                                    cast_slice(&arr),
                                                    &abi
                                                ))
                                            }
                                            Value::U64Array(arr)
                                                if abi.long_size == LongSize::Bits64 =>
                                            {
                                                let abi = abi.clone();
                                                to_string!(Bitmap::from_bytes(
                                                    cast_slice(&arr),
                                                    &abi
                                                ))
                                            }
                                            val => {
                                                Err(EvalError::IllegalType(val.into_static().ok()))
                                            }
                                        }
                                    }))
                                }),
                            })
                        })
                    )
                }),
            })
        }
        "__print_flags" | "__print_flags_u64" => print_symbolic::<false>(),
        "__print_symbolic" | "__print_symbolic_u64" => print_symbolic::<true>(),

        "__print_hex" => print_array_hex(" "),
        "__print_hex_str" => print_array_hex(""),
        "__print_array" => {
            Ok(ExtensionMacroKind::FunctionLike {
                parser: Box::new(move |penv, input| {
                    apply_cparser!(
                        penv,
                        input,
                        map_res_cut(
                            tuple((
                                // Array to format
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Array size
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Item size
                                CGrammar::assignment_expr(),
                            )),
                            |(val, _, array_size, _, item_size)| {
                                let compiler = Arc::new(move |cenv: &dyn CompileEnv<'_>| {
                                    let cval = val.clone().compile(&cenv)?;
                                    let carray_size = array_size.clone().compile(&cenv)?;
                                    let citem_size = item_size.clone().compile(&cenv)?;

                                    let eval = new_dyn_evaluator(move |env: &_| {
                                        let item_size: usize = match citem_size.eval(env)? {
                                            Value::U64Scalar(x) => Ok(x),
                                            Value::I64Scalar(x) => Ok(x as u64),
                                            val => {
                                                Err(EvalError::IllegalType(val.into_static().ok()))
                                            }
                                        }?
                                        .try_into()
                                        .unwrap();

                                        let array_size: usize = match carray_size.eval(env)? {
                                            Value::U64Scalar(x) => Ok(x),
                                            Value::I64Scalar(x) => Ok(x as u64),
                                            val => {
                                                Err(EvalError::IllegalType(val.into_static().ok()))
                                            }
                                        }?
                                        .try_into()
                                        .unwrap();

                                        macro_rules! print_array {
                                            ($item_ty:ty, $arr:expr) => {{
                                                let item_size = core::mem::size_of::<$item_ty>();
                                                let writer = OwnedScratchBox::new_in(
                                                    move |out: &mut dyn fmt::Write| {
                                                        let mut closure = || {
                                                            write!(out, "[")?;
                                                            let mut first = true;
                                                            for x in $arr.into_iter().take(
                                                                array_size.try_into().unwrap(),
                                                            ) {
                                                                if !first {
                                                                    write!(out, ",")?;
                                                                }
                                                                let x: $item_ty = x;
                                                                write!(
                                                                    out,
                                                                    "{:#0size$x}",
                                                                    x,
                                                                    size = 2 + item_size * 2
                                                                )?;
                                                                first = false;
                                                            }
                                                            write!(out, "]")
                                                        };
                                                        closure().expect(
                                                            "could not render array to string",
                                                        )
                                                    },
                                                    env.scratch(),
                                                );

                                                (
                                                    item_size,
                                                    Ok(Value::Str(Str::new_procedural(writer))),
                                                )
                                            }};
                                        }

                                        let (real_item_size, printed) = match cval.eval(env)? {
                                            Value::Raw(_, arr) => {
                                                Ok(print_array!(u8, arr.iter().copied()))
                                            }
                                            Value::Bitmap(bitmap) => match &bitmap.chunk_size {
                                                LongSize::Bits32 => Ok(print_array!(
                                                    u32,
                                                    bitmap.into_iter().as_chunks().map(|x| x
                                                        .try_into()
                                                        .expect(
                                                            "Chunk requires more than 32 bits"
                                                        ))
                                                )),
                                                LongSize::Bits64 => Ok(print_array!(
                                                    u64,
                                                    bitmap.into_iter().as_chunks()
                                                )),
                                            },
                                            Value::Str(s) => {
                                                Ok(print_array!(u8, s.bytes().chain([0])))
                                            }

                                            Value::U8Array(arr) => {
                                                Ok(print_array!(u8, arr.iter().copied()))
                                            }
                                            Value::I8Array(arr) => {
                                                Ok(print_array!(i8, arr.iter().copied()))
                                            }

                                            Value::U16Array(arr) => {
                                                Ok(print_array!(u16, arr.iter().copied()))
                                            }
                                            Value::I16Array(arr) => {
                                                Ok(print_array!(i16, arr.iter().copied()))
                                            }

                                            Value::U32Array(arr) => {
                                                Ok(print_array!(u32, arr.iter().copied()))
                                            }
                                            Value::I32Array(arr) => {
                                                Ok(print_array!(i32, arr.iter().copied()))
                                            }

                                            Value::U64Array(arr) => {
                                                Ok(print_array!(u64, arr.iter().copied()))
                                            }
                                            Value::I64Array(arr) => {
                                                Ok(print_array!(i64, arr.iter().copied()))
                                            }

                                            val => {
                                                Err(EvalError::IllegalType(val.into_static().ok()))
                                            }
                                        }?;

                                        if real_item_size == item_size {
                                            printed
                                        } else {
                                            Err(EvalError::ExtensionMacroError {
                                                call: "__print_array(...)".into(),
                                                error: format!("Wrong size for array item. Expected {item_size} bytes, got {real_item_size} bytes")
                                            })
                                        }
                                    });
                                    Ok(eval)
                                });

                                Ok(ExtensionMacroCallCompiler {
                                    ret_typ: Type::Pointer(Box::new(penv.abi().char_typ())),
                                    compiler,
                                })
                            },
                        )
                    )
                }),
            })
        }
        "__print_hex_dump" => {
            Ok(ExtensionMacroKind::FunctionLike {
                parser: Box::new(move |penv, input| {
                    apply_cparser!(
                        penv,
                        input,
                        map_res_cut(
                            tuple((
                                // Prefix string
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Prefix type
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Row size
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Group size
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Buffer
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Length
                                CGrammar::assignment_expr(),
                                lexeme(char(',')),
                                // Ascii
                                CGrammar::assignment_expr(),
                            )),
                            |(
                                prefix_str,
                                _,
                                prefix_type,
                                _,
                                row_size,
                                _,
                                _group_size,
                                _,
                                buf,
                                _,
                                length,
                                _,
                                ascii,
                            )| {
                                let compiler = Arc::new(move |cenv: &dyn CompileEnv<'_>| {
                                    let endianness = cenv.abi().endianness;

                                    let cprefix_str = prefix_str.clone().compile(&cenv)?;
                                    let cprefix_type = prefix_type.clone().compile(&cenv)?;
                                    let crow_size = row_size.clone().compile(&cenv)?;
                                    // Group size is ignored, as using any group size
                                    // different than the underlying buffer type is
                                    // undefined behavior. Therefore we can simply look at
                                    // the kind of array we get at runtime and format it
                                    // normally.
                                    let cbuf = buf.clone().compile(&cenv)?;
                                    let clength = length.clone().compile(&cenv)?;
                                    let cascii = ascii.clone().compile(&cenv)?;

                                    let eval = new_dyn_evaluator(move |env: &_| {
                                        macro_rules! eval_int {
                                            ($expr:expr) => {{
                                                let val: usize = match $expr.eval(env)? {
                                                    Value::U64Scalar(x) => Ok(x),
                                                    Value::I64Scalar(x) => Ok(x as u64),
                                                    val => Err(EvalError::IllegalType(
                                                        val.into_static().ok(),
                                                    )),
                                                }?
                                                .try_into()
                                                .unwrap();
                                                val
                                            }};
                                        }

                                        let prefix_str: Option<String> =
                                            match cprefix_str.eval(env)? {
                                                Value::U64Scalar(0) => None,
                                                Value::I64Scalar(0) => None,
                                                val => {
                                                    let val = val.deref_ptr(env)?;
                                                    Some(match val.to_str() {
                                                        Some(s) => Ok(s.into()),
                                                        None => Err(EvalError::IllegalType(
                                                            val.into_static().ok(),
                                                        )),
                                                    }?)
                                                }
                                            };

                                        let prefix_type = eval_int!(cprefix_type);
                                        let row_size = eval_int!(crow_size);

                                        let length = eval_int!(clength);
                                        let ascii = eval_int!(cascii) != 0;

                                        macro_rules! print_array {
                                            ($item_ty:ty, $iter:expr) => {{
                                                let item_size = core::mem::size_of::<$item_ty>();
                                                let items_per_row = row_size / item_size;
                                                let writer = OwnedScratchBox::new_in(
                                                    move |out: &mut dyn fmt::Write| {
                                                        let mut closure = || -> Result<(), fmt::Error> {
                                                            for (row_i, row) in $iter
                                                                .into_iter()
                                                                .take(length)
                                                                .chunks(items_per_row)
                                                                .into_iter()
                                                                .enumerate()
                                                            {
                                                                write!(out, "\n")?;

                                                                // enum {
                                                                //     DUMP_PREFIX_NONE,
                                                                //     DUMP_PREFIX_ADDRESS,
                                                                //     DUMP_PREFIX_OFFSET
                                                                // };
                                                                match prefix_type {
                                                                    // We cannot handle
                                                                    // DUMP_PREFIX_ADDRESS
                                                                    // since the address
                                                                    // would be meaningless
                                                                    // (the address inside
                                                                    // the buffer), so we
                                                                    // treat it as the
                                                                    // offset.
                                                                    1 | 2 => write!(
                                                                        out,
                                                                        "{}{:#02x}: ",
                                                                        prefix_str.as_deref().unwrap_or(""),
                                                                        row_i * items_per_row
                                                                    )?,
                                                                    _ => match prefix_str.as_deref() {
                                                                        Some(prefix_str) => write!(out, "{prefix_str}: ")?,
                                                                        None => (),
                                                                    }
                                                                }

                                                                macro_rules! print_hex {
                                                                    ($buf:expr) => {{
                                                                        for (i, x) in $buf.enumerate() {
                                                                            // Ensure the type we got
                                                                            // passed matches what we
                                                                            // effectively get.
                                                                            let x: $item_ty = x;
                                                                            if i != 0 {
                                                                                write!(out, " ")?;
                                                                            }
                                                                            write!(
                                                                                out,
                                                                                "{:0size$x}",
                                                                                x,
                                                                                size = item_size * 2
                                                                            )?;
                                                                        }
                                                                        Ok(())
                                                                    }};
                                                                }

                                                                if ascii {
                                                                    let mut vec = ScratchVec::with_capacity_in(
                                                                        items_per_row,
                                                                        env.scratch(),
                                                                    );
                                                                    vec.extend(row);

                                                                    print_hex!(vec.iter().copied())?;

                                                                    write!(out, " ")?;
                                                                    for item in vec {
                                                                        let bytes = match endianness {
                                                                            Endianness::Little => {
                                                                                item.to_le_bytes()
                                                                            }
                                                                            Endianness::Big => {
                                                                                item.to_be_bytes()
                                                                            }
                                                                        };
                                                                        for x in bytes {
                                                                            let x = x as char;
                                                                            write!(
                                                                                out,
                                                                                "{}",
                                                                                if x.is_control() {
                                                                                    '.'
                                                                                } else {
                                                                                    x
                                                                                }
                                                                            )?;
                                                                        }
                                                                    }
                                                                } else {
                                                                    print_hex!(row)?;
                                                                }
                                                            }
                                                            Ok(())
                                                        };
                                                        closure().expect("could not render array to string")
                                                    },
                                                    env.scratch(),
                                                );

                                                Ok(Value::Str(Str::new_procedural(writer)))
                                            }};
                                        }

                                        match cbuf.eval(env)? {
                                            Value::Bitmap(bitmap) => match &bitmap.chunk_size {
                                                LongSize::Bits32 => print_array!(
                                                    u32,
                                                    bitmap.into_iter().as_chunks().map(|x| x
                                                        .try_into()
                                                        .expect(
                                                            "Chunk requires more than 32 bits"
                                                        ))
                                                ),
                                                LongSize::Bits64 => print_array!(
                                                    u64,
                                                    bitmap.into_iter().as_chunks()
                                                ),
                                            },
                                            Value::Raw(_, arr) => {
                                                print_array!(u8, arr.iter().copied())
                                            }
                                            Value::Str(s) => {
                                                print_array!(u8, s.bytes().chain([0]))
                                            }

                                            Value::U8Array(arr) => {
                                                print_array!(u8, arr.iter().copied())
                                            }
                                            Value::I8Array(arr) => {
                                                print_array!(i8, arr.iter().copied())
                                            }

                                            Value::U16Array(arr) => {
                                                print_array!(u16, arr.iter().copied())
                                            }
                                            Value::I16Array(arr) => {
                                                print_array!(i16, arr.iter().copied())
                                            }

                                            Value::U32Array(arr) => {
                                                print_array!(u32, arr.iter().copied())
                                            }
                                            Value::I32Array(arr) => {
                                                print_array!(i32, arr.iter().copied())
                                            }

                                            Value::U64Array(arr) => {
                                                print_array!(u64, arr.iter().copied())
                                            }
                                            Value::I64Array(arr) => {
                                                print_array!(i64, arr.iter().copied())
                                            }

                                            val => {
                                                Err(EvalError::IllegalType(val.into_static().ok()))
                                            }
                                        }
                                    });
                                    Ok(eval)
                                });

                                Ok(ExtensionMacroCallCompiler {
                                    ret_typ: Type::Pointer(Box::new(penv.abi().char_typ())),
                                    compiler,
                                })
                            },
                        )
                    )
                }),
            })
        }
        id => Err(CParseError::InvalidVariableIdentifier(id.into())),
    }
}

#[derive(Clone)]
pub struct CGrammarCtx<'a> {
    pub penv: &'a dyn ParseEnv,
}

impl<'a> CGrammarCtx<'a> {
    pub fn new(penv: &'a dyn ParseEnv) -> Self {
        CGrammarCtx { penv }
    }

    #[inline]
    fn abi(&self) -> &Abi {
        self.penv.abi()
    }
}

#[inline]
pub fn is_identifier<I>(i: I) -> bool
where
    I: nom::AsBytes,
{
    all_consuming(recognize(identifier::<_, ()>()))(i.as_bytes())
        .finish()
        .is_ok()
}

// https://port70.net/~nsz/c/c11/n1570.html#6.4.2.1
#[inline]
pub fn identifier<I, E>() -> impl nom::Parser<I, Identifier, E>
where
    I: nom::AsBytes
        + Clone
        + nom::InputTake
        + nom::Offset
        + nom::Slice<core::ops::RangeTo<usize>>
        + nom::InputLength
        + nom::InputIter<Item = u8>
        + nom::InputTakeAtPosition
        + for<'a> nom::Compare<&'a str>,
    <I as nom::InputTakeAtPosition>::Item: Clone + nom::AsChar,
    E: FromExternalError<I, CParseError> + nom::error::ParseError<I>,
{
    map_res_cut(
        lexeme(recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))),
        // For some reason rustc fails to infer CGrammar properly, even
        // though it looks like it should. Maybe this will be fixed in
        // the future.
        |s: I| {
            from_utf8(s.as_bytes())
                .map_err(|err| CParseError::DecodeUtf8(err.to_string()))
                .map(|s| s.into())
        },
    )
}

fn keyword<'a, I, E>(name: &'a str) -> impl nom::Parser<I, I, E> + 'a
where
    E: nom::error::ParseError<I> + 'a,
    I: nom::AsBytes
        + Clone
        + nom::InputTake
        + nom::Offset
        + nom::Slice<core::ops::RangeTo<usize>>
        + nom::InputLength
        + nom::InputIter<Item = u8>
        + nom::InputTakeAtPosition
        + for<'b> nom::Compare<&'b str>
        + 'a,
    <I as nom::InputTakeAtPosition>::Item: Clone + nom::AsChar,
    E: FromExternalError<I, CParseError> + nom::error::ParseError<I>,
{
    let mut inner = all_consuming(lexeme(tag(name)));
    let mut identifier = recognize(identifier());
    move |input: I| {
        let (input, id) = identifier.parse(input)?;
        let (_, x) = inner.parse(id)?;
        Ok((input, x))
    }
}

fn escape_sequence<I, E>() -> impl nom::Parser<I, u8, E>
where
    I: Clone
        + AsBytes
        + nom::InputTake
        + nom::InputLength
        + nom::InputIter<Item = u8>
        + nom::InputTakeAtPosition<Item = u8>
        + nom::Slice<std::ops::RangeFrom<usize>>,
    E: FromExternalError<I, CParseError> + nom::error::ParseError<I> + nom::error::ContextError<I>,
{
    context(
        "escape sequence",
        preceded(
            char('\\'),
            alt((
                // Simple escape sequence
                alt((
                    char('"').map(|_| b'\"'),
                    char('\'').map(|_| b'\''),
                    char('\\').map(|_| b'\\'),
                    char('n').map(|_| b'\n'),
                    char('t').map(|_| b'\t'),
                    char('a').map(|_| 0x07u8),
                    char('b').map(|_| 0x08u8),
                    char('e').map(|_| 0x1Bu8),
                    char('f').map(|_| 0x0Cu8),
                    char('r').map(|_| 0x0Du8),
                    char('v').map(|_| 0x0Bu8),
                    char('?').map(|_| 0x3Fu8),
                )),
                // Hexadecimal escape sequence
                preceded(
                    char('x'),
                    map_res_cut(hex_u64, |x| {
                        x.try_into().map_err(|_| CParseError::CharOutOfRange(x))
                    }),
                ),
                // Octal escape sequence
                map_res_cut(
                    many_m_n(
                        1,
                        3,
                        alt((
                            char('0'),
                            char('1'),
                            char('2'),
                            char('3'),
                            char('4'),
                            char('5'),
                            char('6'),
                            char('7'),
                        )),
                    ),
                    |digits| {
                        let zero: u64 = b'0'.into();
                        let digits = digits.into_iter().rev();
                        let mut x: u64 = 0;
                        let mut n = 1;
                        for digit in digits {
                            let digit: u64 = digit.into();
                            let digit: u64 = digit - zero;
                            x += digit * n;
                            n *= 8;
                        }
                        x.try_into().map_err(|_| CParseError::CharOutOfRange(x))
                    },
                ),
            )),
        ),
    )
}

pub fn string_literal<I, E>() -> impl nom::Parser<I, Expr, E>
where
    I: Clone
        + AsBytes
        + nom::InputTake
        + nom::Slice<std::ops::RangeFrom<usize>>
        + nom::InputLength
        + nom::InputIter<Item = u8>
        + nom::InputTakeAtPosition<Item = u8>
        + for<'a> nom::Compare<&'a str>,
    E: FromExternalError<I, CParseError> + nom::error::ContextError<I> + nom::error::ParseError<I>,
{
    many1(map_res_cut(
        tuple((
            context(
                "string encoding prefix",
                lexeme(opt(alt((tag("u8"), tag("u"), tag("U"), tag("L"))))),
            ),
            lexeme(delimited(
                char('"'),
                cut(context(
                    "string char sequence",
                    // Regrettably, type inference with map_res() breaks
                    // and we are forced to spell out the type of input,
                    // including a reference to a lifetime introduced
                    // inside the grammar!() macro.
                    |mut input: I| {
                        let mut string: String = String::new();
                        loop {
                            let res = escape_sequence::<_, E>().parse(input.clone());
                            if let Ok((_input, c)) = res {
                                string.push(c.into());
                                input = _input;
                                continue;
                            }

                            // Parse runs of non-escaped chars
                            let res = is_not::<_, _, ()>(r#"\""#).parse(input.clone());
                            match res {
                                Ok((_input, s)) => {
                                    input = _input;
                                    let s = match from_utf8(s.as_bytes()) {
                                        Err(err) => {
                                            return error(
                                                input,
                                                CParseError::DecodeUtf8(err.to_string()),
                                            )
                                        }
                                        Ok(s) => s,
                                    };
                                    string.push_str(s);
                                    continue;
                                }
                                _ => return Ok((input, string)),
                            }
                        }
                    },
                )),
                char('"'),
            )),
        )),
        |(prefix, string)| match prefix {
            Some(_) => Err(CParseError::UnsupportedConstruct(
                "string encoding prefix syntax is not supported".into(),
            )),
            None => Ok(string),
        },
    ))
    .map(|seqs: Vec<String>| {
        let mut s = String::new();
        for seq in seqs {
            s.push_str(&seq);
        }
        Expr::StringLiteral(s)
    })
}

// CGrammar inspired by N1570 ISO C draft (latest C11 draft):
// https://port70.net/~nsz/c/c11/n1570.html#A.2.1
grammar! {
    name: pub CGrammar,
    ctx: CGrammarCtx<'i>,
    error: CParseError,
    rules: {

        // https://port70.net/~nsz/c/c11/n1570.html#6.4.2.1
        rule identifier() -> Identifier {
            identifier()
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.3p1
        rule type_qualifier() -> () {
            alt((
                keyword("const"),
                keyword("restrict"),
                keyword("volatile"),
                keyword("_Atomic"),
            )).map(|_| ())
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.6
        rule declarator(abstract_declarator: bool) -> Declarator {
            lexeme(
                alt((
                    context(
                        "pointer",
                        preceded(
                            pair(
                                lexeme(char('*')),
                                many0(Self::type_qualifier()),
                            ),
                            Self::declarator(abstract_declarator),
                        ).map(|declarator| {
                            // Apply the declarator's modification last, since they have the least
                            // precedence (arrays)
                            let modify_typ = Rc::new(move |typ| (declarator.modify_typ)(Type::Pointer(Box::new(typ))));

                            Declarator {
                                modify_typ,
                                ..declarator
                            }
                        })
                    ),
                    Self::direct_declarator(abstract_declarator),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.6
        #[leftrec] rule direct_declarator(abstract_declarator: bool) -> Declarator {
            let name = if abstract_declarator {
                "abstract direct declarator"
            } else {
                "direct declarator"
            };

            let _id = || {
                lexeme(Self::identifier().map(|id| Declarator {
                    identifier: Some(id),
                    modify_typ: Rc::new(|typ| typ),
                }))
            };
            let id = || {
                move |input| {
                    if abstract_declarator {
                        alt((
                            _id(),
                            success_with(|| Declarator {
                                identifier: None,
                                modify_typ: Rc::new(|typ| typ),
                            }),
                        ))
                        .parse(input)
                    } else {
                        _id().parse(input)
                    }
                }
            };

            let paren = || {
                context(
                    "parenthesized",
                    lexeme(parenthesized(Self::declarator(abstract_declarator))),
                )
            };

            let parameter_declaration = || lexeme(context(
                "parameter declaration",
                lexeme(pair(
                    Self::declaration_specifier(),
                    // We only have to deal with declarations containing only one
                    // declarator, i.e. we only handle "int foo" and not "int foo, bar;"
                    alt((
                        Self::declarator(true),
                        Self::declarator(false),
                    ))
                )).map(|(typ, declarator)| {
                    ParamDeclaration {
                        typ: (declarator.modify_typ)(typ),
                        identifier: declarator.identifier,
                    }
                }),
            ));

            let function = context(
                "function",
                pair(
                    Self::direct_declarator(abstract_declarator),
                    context(
                        "parameter list",
                        lexeme(parenthesized(
                            separated_list0(
                                lexeme(char(',')),
                                parameter_declaration(),
                            )
                        ))
                    )
                ).map(|(declarator, params)| {
                    let modify_typ = Rc::new(move |typ| (declarator.modify_typ)(
                        Type::Function(
                            Box::new(typ),
                            params.clone()
                        )
                    ));
                    Declarator {
                        modify_typ,
                        identifier: declarator.identifier,
                    }
                })
            );

            let array = context(
                "array",
                tuple((
                    Self::grammar_ctx(),
                    Self::direct_declarator(abstract_declarator),
                    context(
                        "array size",
                        lexeme(delimited(
                            char('['),
                            preceded(
                                delimited(
                                    opt(keyword("static")),
                                    many0(
                                        Self::type_qualifier()
                                    ),
                                    opt(keyword("static")),
                                ),
                                lexeme(opt(Self::assignment_expr())),
                            ),
                            char(']'),
                        )),
                    ),
                )),
            ).map(
                |(ctx, declarator, array_size)| {
                    let array_size = match array_size {
                        Some(x) => {
                            match eval_unsigned(x, ctx.abi()) {
                                Ok(0) => ArrayKind::ZeroLength,
                                Ok(x) => ArrayKind::Fixed(Ok(x)),
                                Err(err) => ArrayKind::Fixed(Err(Box::new(err))),
                            }
                        },
                        None => ArrayKind::ZeroLength,
                    };
                    let modify_typ = Rc::new(move |typ| (declarator.modify_typ)(Type::Array(Box::new(typ), array_size.clone())));
                    Declarator {
                        modify_typ,
                        identifier: declarator.identifier,
                    }
                }
            );

            lexeme(
                context(name,
                    alt((
                        array, function, paren(), id()
                    ))
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7
        rule declaration_specifier() -> Type {
            let iso = lexeme(move |mut input| {
                let abi = Self::get_ctx(&input).abi();
                let long_size = abi.long_size;
                let char_typ = abi.char_typ();

                #[derive(Debug, Clone, Copy)]
                enum DeclSignedness {
                    Signed,
                    Unsigned,
                    // This is important to represent unknown signedness so that we can
                    // differentiate a lone "signed" from nothing, as a lone "signed" is
                    // equivalent to "signed int"
                    Unknown,
                }

                #[derive(Debug, Clone)]
                enum State {
                    Unknown(DeclSignedness),
                    Char(DeclSignedness),
                    Short(DeclSignedness),
                    Int(DeclSignedness),
                    Long(DeclSignedness),
                    LongLong(DeclSignedness),
                }

                // Tokens that we simply discard as they don't impact layout or pretty
                // representation
                let discard_parser = || {
                    context(
                        "discarded",
                        many0_count(
                            alt((
                                keyword("extern").map(|_| ()),
                                keyword("static").map(|_| ()),
                                keyword("auto").map(|_| ()),
                                keyword("register").map(|_| ()),
                                keyword("_Thread_local").map(|_| ()),
                                Self::type_qualifier(),
                            ))
                        )
                    )
                };

                // Parse the tokens using a state machine to deal with various
                // combinations of "signed int" "int signed" "signed", "long unsigned
                // long" etc.
                let mut state = State::Unknown(DeclSignedness::Unknown);

                loop {
                    // tokens we simply ignore
                    (input, _) = discard_parser().parse(input)?;

                    macro_rules! fsm {
                        ($($tag:expr => $transition:expr,)*) => {
                            |input| {
                                let (input, id) = Self::identifier().parse(input)?;

                                match &*id {
                                    $(
                                        $tag => {
                                            let res: Result<State, CParseError> = $transition;
                                            Ok((input, res))
                                        }
                                    ),*
                                    _ => fail(input)
                                }
                            }
                        }
                    }
                    let res = lexeme::<_, _, (), _>(fsm! {
                        "signed" => Ok(match &state {
                            State::Unknown(_) => State::Unknown(DeclSignedness::Signed),
                            State::Char(_) => State::Char(DeclSignedness::Signed),
                            State::Short(_) => State::Short(DeclSignedness::Signed),
                            State::Int(_) => State::Int(DeclSignedness::Signed),
                            State::Long(_) => State::Long(DeclSignedness::Signed),
                            State::LongLong(_) => State::LongLong(DeclSignedness::Signed),
                        }),
                        "unsigned" => Ok(match &state {
                            State::Unknown(_) => State::Unknown(DeclSignedness::Unsigned),
                            State::Char(_) => State::Char(DeclSignedness::Unsigned),
                            State::Short(_) => State::Short(DeclSignedness::Unsigned),
                            State::Int(_) => State::Int(DeclSignedness::Unsigned),
                            State::Long(_) => State::Long(DeclSignedness::Unsigned),
                            State::LongLong(_) => State::LongLong(DeclSignedness::Unsigned),
                        }),

                        "char" => match &state {
                            State::Unknown(signedness) => Ok(State::Char(*signedness)),
                            x@State::Char(_) => Ok(x.clone()),
                            _ => Err(CParseError::InvalidTypeName),
                        },
                        "short" => match &state {
                            State::Unknown(signedness) => Ok(State::Short(*signedness)),
                            State::Int(signedness) => Ok(State::Short(*signedness)),
                            _ => Err(CParseError::InvalidTypeName),
                        },
                        "int" => match &state {
                            State::Unknown(signedness) => Ok(State::Int(*signedness)),
                            State::Char(_) => Err(CParseError::InvalidTypeName),
                            State::Int(_) => Err(CParseError::InvalidTypeName),
                            x => Ok(x.clone()),
                        },
                        "long" => match &state {
                            State::Unknown(signedness) => Ok(State::Long(*signedness)),
                            State::Int(signedness) => Ok(State::Long(*signedness)),
                            State::Long(signedness) => Ok(State::LongLong(*signedness)),
                            _ => Err(CParseError::InvalidTypeName),
                        },
                    })
                    .parse(input.clone());

                    (input, state) = match res {
                        Ok((i, Ok(x))) => (i, x),
                        Ok((i, Err(err))) => return failure(i, err),
                        // We stop parsing when we can't recognize anything anymore.
                        // Either we hit something else (e.g. an "(" or "[") or we
                        // simply encountered a user-defined type.
                        Err(_) => break,
                    }
                }
                let (input, typ) = match state {
                    // If we did not hit any of "int", "signed" etc, then it's just a
                    // user-defined type that we can consume now.
                    State::Unknown(DeclSignedness::Unknown) => lexeme(alt((
                        context(
                            "struct",
                            preceded(keyword("struct"), Self::identifier())
                                .map(Type::Struct),
                        ),
                        context(
                            "enum",
                            preceded(keyword("enum"), Self::identifier())
                                .map(|id| Type::Enum(Box::new(Type::Unknown), id)),
                        ),
                        context(
                            "union",
                            preceded(keyword("union"), Self::identifier())
                                .map(Type::Union),
                        ),
                        context(
                            "struct/union/enum definition",
                            map_res_cut(
                                delimited(
                                    alt((
                                        keyword("struct"),
                                        keyword("union"),
                                        keyword("enum"),
                                    )),
                                    opt(Self::identifier()),
                                    lexeme(char('{')),
                                ),
                                |_| Err(CParseError::UnsupportedConstruct("struct/union/enum definition are not supported".into()))
                            )
                        ),
                        context(
                            "scalar",
                            Self::identifier().map(|id| {
                                match id.as_ref() {
                                    "void" => Type::Void,
                                    "_Bool" => Type::Bool,
                                    _ => {
                                        let typ = match id.as_ref() {
                                            "caddr_t" => Type::Pointer(Box::new(char_typ.clone())),
                                            "bool" => Type::Bool,

                                            "s8" | "__s8" | "int8_t"  => Type::I8,
                                            "u8" | "__u8" | "uint8_t" | "u_char" | "unchar" | "u_int8_t" => Type::U8,

                                            "s16" | "__s16" | "int16_t" => Type::I16,
                                            "u16" | "__u16" | "uint16_t" | "u_short" | "ushort" | "u_int16_t" | "__le16" | "__be16" | "__sum16" => Type::U16,

                                            "s32" | "__s32" | "int32_t" => Type::I32,
                                            "u32" | "__u32" | "uint32_t" | "u_int" | "uint" | "u_int32_t" | "gfp_t" | "slab_flags_t" | "fmode_t" | "OM_uint32" | "dev_t" | "nlink_t" | "__le32" | "__be32" | "__wsum" | "__poll_t" => Type::U32,

                                            "s64" | "__s64" | "int64_t" | "loff_t" => Type::I64,
                                            "u64" | "__u64" | "uint64_t" | "u_int64_t" | "sector_t" | "blkcnt_t" | "__le64" | "__be64" => Type::U64,

                                            "pid_t" => Type::I32,

                                            "u_long" | "ulong" | "off_t" | "ssize_t" | "ptrdiff_t" | "clock_t" | "irq_hw_number_t" => match long_size {
                                                LongSize::Bits32 => Type::I32,
                                                LongSize::Bits64 => Type::I64,
                                            },

                                            "uintptr_t" | "size_t" => match long_size {
                                                LongSize::Bits32 => Type::U32,
                                                LongSize::Bits64 => Type::U64,
                                            },

                                            _ => Type::Unknown,
                                        };
                                        Type::Typedef(Box::new(typ), id)
                                    }
                                }
                            }),
                        ),
                    )))
                    .parse(input),

                    // "signed" alone is equivalent to "signed int"
                    State::Unknown(DeclSignedness::Signed) => Ok((input, Type::I32)),
                    State::Unknown(DeclSignedness::Unsigned) => Ok((input, Type::U32)),

                    State::Char(DeclSignedness::Signed) => Ok((input, Type::I8)),
                    State::Char(DeclSignedness::Unsigned) => Ok((input, Type::U8)),

                    State::Char(DeclSignedness::Unknown) => Ok((input, char_typ)),

                    State::Short(DeclSignedness::Signed | DeclSignedness::Unknown) => {
                        Ok((input, Type::I16))
                    }
                    State::Short(DeclSignedness::Unsigned) => Ok((input, Type::U16)),

                    State::Int(DeclSignedness::Signed | DeclSignedness::Unknown) => {
                        Ok((input, Type::I32))
                    }
                    State::Int(DeclSignedness::Unsigned) => Ok((input, Type::U32)),

                    State::Long(DeclSignedness::Signed | DeclSignedness::Unknown) => Ok((
                        input,
                        match long_size {
                            LongSize::Bits32 => Type::I32,
                            LongSize::Bits64 => Type::I64,
                        },
                    )),
                    State::Long(DeclSignedness::Unsigned) => Ok((
                        input,
                        match long_size {
                            LongSize::Bits32 => Type::U32,
                            LongSize::Bits64 => Type::U64,
                        },
                    )),

                    State::LongLong(DeclSignedness::Signed | DeclSignedness::Unknown) => {
                        Ok((input, Type::I64))
                    }
                    State::LongLong(DeclSignedness::Unsigned) => Ok((input, Type::U64)),
                }?;

                let (input, _) = discard_parser().parse(input)?;
                Ok((input, typ))
            });

            let gnu_typeof = map_res_cut(
                pair(
                    Self::grammar_ctx(),
                    preceded(
                        alt((
                            keyword("typeof"),
                            keyword("__typeof__"),
                        )),
                        cut(parenthesized(
                            Self::expr(),
                        ))
                    )
                ),
                |(ctx, expr)| {
                    expr.typ(ctx.penv).map_err(|_| CParseError::CouldNotGuessType(expr))
                }
            );

            alt((gnu_typeof, iso))
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7
        rule declaration() -> Declaration {
            // Parser for ISO C
            let iso = context(
                "iso C declaration",
                map_res_cut(
                    lexeme(pair(
                        Self::declaration_specifier(),
                        // We only have to deal with declarations containing only one
                        // declarator, i.e. we only handle "int foo" and not "int foo, bar;"
                        Self::declarator(false),
                    )),
                    |(typ, declarator)| {
                        let typ = (declarator.modify_typ)(typ);
                        match declarator.identifier {
                            Some(identifier) => {
                                Ok(Declaration {
                                    identifier,
                                    typ
                                })
                            },
                            None => Err(CParseError::DeclarationWithoutIdentifier(typ))
                        }
                    },
                ),
            );

            // Invalid C syntax that ftrace outputs for its __data_loc and __rel_loc, e.g.:
            // __data_loc char[] name
            let data_loc = context(
                "__data_loc declaration",
                lexeme(
                    // Once we consumed "__data_loc", we don't want to allow
                    // backtracking back to the ISO C declaration, as we know it will
                    // never yield something sensible.
                    map_res_cut(
                        tuple((
                            alt((
                                keyword("__data_loc"),
                                keyword("__rel_loc"),
                            )),
                            Self::declaration_specifier(),
                            // This will be an abstract declarator, i.e. a declarator with
                            // no identifier (like parameters in a function prototype), as
                            // the name comes after the last "[]"
                            Self::declarator(true),
                            opt(context(
                                "__data_loc identifier",
                                lexeme(Self::identifier()),
                            )),
                        )),
                        |(kind, typ, abstract_declarator, identifier)| {
                            // Push the array sizes down the stack. The 2nd nested array takes the size of the 1st etc.
                            fn push_array_size(
                                typ: Type,
                                kind: ArrayKind,
                            ) -> Result<(bool, Type), CParseError> {
                                match typ {
                                    Type::Array(_, ArrayKind::Dynamic(_)) | Type::DynamicScalar(..) => {
                                        Err(CParseError::NestedDataLoc(typ))
                                    }
                                    Type::Array(typ, kind_) => Ok((
                                        true,
                                        Type::Array(Box::new(push_array_size(*typ, kind_)?.1), kind),
                                    )),
                                    Type::Pointer(typ) => {
                                        let (_, typ) = push_array_size(*typ, kind)?;
                                        // If an array is behind a pointer, it can be ignored.
                                        Ok((false, Type::Pointer(Box::new(typ))))
                                    }
                                    _ => Ok((false, typ)),
                                }
                            }

                            let typ = (abstract_declarator.modify_typ)(typ);

                            // Remove the inner array, which is corresponding to the last "[]"
                            // parsed. It only acts as a separator between the type specifier
                            // and the identifier, and actually corresponds to a top-level
                            // dynamic array.

                            // The innermost array is a fixed "[]" that is part of the format.
                            // It is actually acting as a top-level array, so we push the array
                            // sizes down the stack and replace the top-level by a dynamic array.
                            let array_kind = match *kind.deref() {
                                b"__data_loc" => ArrayKind::Dynamic(DynamicKind::Dynamic),
                                b"__rel_loc" => ArrayKind::Dynamic(DynamicKind::DynamicRel),
                                _ => panic!("Unknown dynamic kind: {:?}", kind.deref()),
                            };
                            let (is_array, pushed_typ) = push_array_size(typ.clone(), array_kind)?;

                            let typ = if is_array {
                                match pushed_typ {
                                    typ@Type::Array(..) => Ok(typ),
                                    typ => Err(CParseError::DataLocArrayNotArray(typ))
                                }
                            } else {
                                let scalar_kind = match *kind.deref() {
                                    b"__data_loc" => DynamicKind::Dynamic,
                                    b"__rel_loc" => DynamicKind::DynamicRel,
                                    _ => panic!("Unknown dynamic kind: {:?}", kind.deref()),
                                };
                                Ok(Type::DynamicScalar(Box::new(typ), scalar_kind))
                            }?;

                            let identifier = if is_array {
                                match identifier {
                                    None => return Err(CParseError::DataLocArrayWithoutIdentifier(typ)),
                                    Some(id) => id,
                                }
                            } else {
                                match (abstract_declarator.identifier, identifier) {
                                    (Some(id), None) => id,
                                    (None, Some(id)) => id,
                                    (None, None) => {
                                        return Err(CParseError::DataLocWithoutIdentifier);
                                    }
                                    (Some(id1), Some(id2)) => {
                                        return Err(CParseError::DataLocAmbiguousIdentifier(id1, id2));
                                    },
                                }
                            };

                            Ok(Declaration {
                                identifier,
                                typ
                            })
                        },
                    ),
                ),
            );

            let parser = alt((data_loc, iso));
            context("declaration", lexeme(parser))
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.9p1
        rule initializer() -> Expr {
            lexeme(
                alt((
                    delimited(
                        lexeme(char('{')),
                        cut(Self::initializer_list().map(Expr::InitializerList)),
                        preceded(
                            lexeme(opt(char(','))),
                            lexeme(char('}')),
                        )
                    ),
                    Self::assignment_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.9p1
        rule initializer_list() -> Vec<Expr> {

            enum DesignatorKind {
                Subscript(Expr),
                Member(Identifier),
            }
            let designator = || {
                lexeme(
                    alt((
                        delimited(
                            lexeme(char('[')),
                            cut(Self::constant_expr()),
                            lexeme(char(']')),
                        ).map(DesignatorKind::Subscript),
                        preceded(
                            lexeme(char('.')),
                            cut(Self::identifier()),
                        ).map(DesignatorKind::Member),
                    )).map(|kind| {
                        move |parent| match kind {
                            DesignatorKind::Subscript(expr) => Expr::Subscript(Box::new(parent), Box::new(expr)),
                            DesignatorKind::Member(id) => Expr::MemberAccess(Box::new(parent), id),
                        }
                    })
                )
            };

            lexeme(
                separated_list1(
                    lexeme(char(',')),
                    alt((
                        separated_pair(
                            fold_many1(
                                designator(),
                                || Expr::Uninit,
                                |parent, combine| combine(parent)
                            ),
                            lexeme(char('=')),
                            cut(Self::initializer()),
                        ).map(|(designation, expr)| Expr::DesignatedInitializer(Box::new(designation), Box::new(expr))),
                        Self::initializer()
                    )),
                )
            )
        }

        rule balanced_paren() -> Span<'i, CGrammar> {
            lexeme(delimited(
                char('('),
                |input: Span<'i, Self> | {
                    let mut i = 0;
                    let mut level: usize = 1;


                    enum EscapingState {
                        Escaped,
                        Normal,
                    }
                    enum LiteralKind {
                        Str,
                        Char,
                    }
                    enum State {
                        Literal(LiteralKind, EscapingState),
                        Normal,
                    }
                    let mut state = State::Normal;

                    for (i_, c) in input.as_bytes().iter().copied().enumerate() {
                        i = i_;

                        match c {
                            // We do not match parenthesis inside string literals
                            b'"' => {
                                state = match state {
                                    State::Normal => State::Literal(LiteralKind::Str, EscapingState::Normal),
                                    State::Literal(LiteralKind::Str, EscapingState::Normal) => State::Normal,
                                    State::Literal(LiteralKind::Str, EscapingState::Escaped) => State::Literal(LiteralKind::Str, EscapingState::Normal),
                                    state@State::Literal(_, _) => state,
                                };
                            }
                            // We do not match parenthesis inside char literals
                            b'\'' => {
                                state = match state {
                                    State::Normal => State::Literal(LiteralKind::Char, EscapingState::Normal),
                                    State::Literal(LiteralKind::Char, EscapingState::Normal) => State::Normal,
                                    State::Literal(LiteralKind::Char, EscapingState::Escaped) => State::Literal(LiteralKind::Char, EscapingState::Normal),
                                    state@State::Literal(_, _) => state,
                                };
                            }
                            b'\\' => {
                                state = match state {
                                    State::Normal => State::Normal,
                                    State::Literal(kind, EscapingState::Normal) => State::Literal(kind, EscapingState::Escaped),
                                    State::Literal(kind, EscapingState::Escaped) => State::Literal(kind, EscapingState::Normal),
                                };
                            }
                            b'(' if matches!(state, State::Normal) => {level += 1;},
                            b')' if matches!(state, State::Normal) => {level -= 1;},
                            _ => ()
                        }
                        if level == 0 {
                            break
                        }
                    }
                    take(i).parse(input)
                },
                char(')'),
            ))
        }


        // https://port70.net/~nsz/c/c11/n1570.html#6.5.2p1
        #[leftrec] rule postfix_expr() -> Expr {
            lexeme(
                alt((
                    context("postinc expr",
                        terminated(
                            Self::postfix_expr(),
                            lexeme(tag("++")),
                        ).map(|expr| Expr::PostInc(Box::new(expr)))
                    ),
                    context("postdec expr",
                        terminated(
                            Self::postfix_expr(),
                            lexeme(tag("--")),
                        ).map(|expr| Expr::PostDec(Box::new(expr)))
                    ),
                    context("subscript expr",
                        tuple((
                            Self::postfix_expr(),
                            delimited(
                                lexeme(char('[')),
                                cut(Self::expr()),
                                lexeme(char(']')),
                            ),
                        )).map(|(array, index)| Expr::Subscript(Box::new(array), Box::new(index)))
                    ),
                    context("func call expr",
                        move |input| {
                            let (input, f) = Self::postfix_expr().parse(input)?;
                            let penv = Self::get_ctx(&input).penv;
                            let f = f.simplify(&BasicEnv::new(penv));
                            match f {
                                Expr::ExtensionMacro(desc) => {
                                    context(
                                        "extension function args",
                                        lexeme(|input: Span<'i, Self>| {
                                            match &desc.kind {
                                                ExtensionMacroKind::FunctionLike{parser} => {
                                                    let desc = Arc::clone(&desc);
                                                    map_res_cut(
                                                        Self::balanced_paren(),
                                                        move |args| {
                                                            let saved_args = (*args.deref()).to_vec();
                                                            let args = from_utf8(args.as_bytes()).map_err(|err| CParseError::DecodeUtf8(err.to_string()))?;
                                                            let compiler = parser(penv, args)?;
                                                            Ok(Expr::ExtensionMacroCall(
                                                                ExtensionMacroCall {
                                                                    args: saved_args,
                                                                    desc: Arc::clone(&desc),
                                                                    compiler,
                                                                }
                                                            ))
                                                        }
                                                    ).parse(input)
                                                }
                                                ExtensionMacroKind::ObjectLike{value: _, typ: _} => {
                                                    error(input, CParseError::CannotCallObjectLikeMacro(desc.name.clone()))
                                                },
                                            }
                                        })
                                    ).parse(input)
                                }
                                f => {
                                    map_res(
                                        parenthesized(
                                            separated_list0(
                                                lexeme(char(',')),
                                                Self::assignment_expr()
                                            )
                                        ),
                                        move |args| Ok(Expr::FuncCall(Box::new(f.clone()), args))
                                    ).parse(input)
                                }
                            }
                        }
                    ),
                    context("member access expr",
                        separated_pair(
                            Self::postfix_expr(),
                            lexeme(char('.')),
                            cut(Self::identifier()),
                        ).map(|(value, member)| Expr::MemberAccess(Box::new(value), member))
                    ),
                    context("deref member access expr",
                        separated_pair(
                            Self::postfix_expr(),
                            lexeme(tag("->")),
                            cut(Self::identifier()),
                        ).map(|(value, member)| Expr::MemberAccess(Box::new(Expr::Deref(Box::new(value))), member))
                    ),

                    context("compound literal",
                        tuple((
                            parenthesized(
                                Self::type_name(),
                            ),
                            delimited(
                                lexeme(char('{')),
                                cut(Self::initializer_list()),
                                preceded(
                                    lexeme(opt(char(','))),
                                    lexeme(char('}')),
                                )
                            ),
                        )).map(|(typ, init)| Expr::CompoundLiteral(typ, init))
                    ),
                    Self::primary_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.5
        #[leftrec] rule multiplicative_expr() -> Expr {
            lexeme(
                alt((
                    context("* expr",
                        separated_pair(
                            Self::multiplicative_expr(),
                            lexeme(char('*')),
                            Self::cast_expr(),
                        ).map(|(lop, rop)| Expr::Mul(Box::new(lop), Box::new(rop)))
                    ),
                    context("/ expr",
                        separated_pair(
                            Self::multiplicative_expr(),
                            lexeme(char('/')),
                            cut(Self::cast_expr()),
                        ).map(|(lop, rop)| Expr::Div(Box::new(lop), Box::new(rop)))
                    ),
                    context("% expr",
                        separated_pair(
                            Self::multiplicative_expr(),
                            lexeme(char('%')),
                            Self::cast_expr(),
                        ).map(|(lop, rop)| Expr::Mod(Box::new(lop), Box::new(rop)))
                    ),
                    Self::cast_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.6
        #[leftrec] rule additive_expr() -> Expr {
            lexeme(
                alt((
                    context("+ expr",
                        separated_pair(
                            Self::additive_expr(),
                            lexeme(char('+')),
                            Self::multiplicative_expr(),
                        ).map(|(lop, rop)| Expr::Add(Box::new(lop), Box::new(rop)))
                    ),
                    context("- expr",
                        separated_pair(
                            Self::additive_expr(),
                            lexeme(char('-')),
                            Self::multiplicative_expr(),
                        ).map(|(lop, rop)| Expr::Sub(Box::new(lop), Box::new(rop)))
                    ),
                    Self::multiplicative_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.7
        #[leftrec] rule shift_expr() -> Expr {
            lexeme(
                alt((
                    context("<< expr",
                        separated_pair(
                            Self::shift_expr(),
                            lexeme(tag("<<")),
                            Self::additive_expr(),
                        ).map(|(lop, rop)| Expr::LShift(Box::new(lop), Box::new(rop)))
                    ),
                    context(">> expr",
                        separated_pair(
                            Self::shift_expr(),
                            lexeme(tag(">>")),
                            Self::additive_expr(),
                        ).map(|(lop, rop)| Expr::RShift(Box::new(lop), Box::new(rop)))
                    ),
                    Self::additive_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.8
        #[leftrec] rule relational_expr() -> Expr {
            lexeme(
                alt((
                    context("<= expr",
                        separated_pair(
                            Self::relational_expr(),
                            lexeme(tag("<=")),
                            cut(Self::shift_expr()),
                        ).map(|(lop, rop)| Expr::LoEq(Box::new(lop), Box::new(rop)))
                    ),
                    context(">= expr",
                        separated_pair(
                            Self::relational_expr(),
                            lexeme(tag(">=")),
                            cut(Self::shift_expr()),
                        ).map(|(lop, rop)| Expr::HiEq(Box::new(lop), Box::new(rop)))
                    ),
                    context("< expr",
                        separated_pair(
                            Self::relational_expr(),
                            lexeme(char('<')),
                            Self::shift_expr(),
                        ).map(|(lop, rop)| Expr::Lo(Box::new(lop), Box::new(rop)))
                    ),
                    context("> expr",
                        separated_pair(
                            Self::relational_expr(),
                            lexeme(char('>')),
                            Self::shift_expr(),
                        ).map(|(lop, rop)| Expr::Hi(Box::new(lop), Box::new(rop)))
                    ),
                    Self::shift_expr(),
                ))
            )
        }


        // https://port70.net/~nsz/c/c11/n1570.html#6.5.9
        #[leftrec] rule equality_expr() -> Expr {
            lexeme(
                alt((
                    context("== expr",
                        separated_pair(
                            Self::equality_expr(),
                            lexeme(tag("==")),
                            cut(Self::relational_expr()),
                        ).map(|(lop, rop)| Expr::Eq(Box::new(lop), Box::new(rop)))
                    ),
                    context("!=",
                        separated_pair(
                            Self::equality_expr(),
                            lexeme(tag("!=")),
                            cut(Self::relational_expr()),
                        ).map(|(lop, rop)| Expr::NEq(Box::new(lop), Box::new(rop)))
                    ),
                    Self::relational_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.10
        #[leftrec] rule and_expr() -> Expr {
            lexeme(
                alt((
                    context("& expr",
                        separated_pair(
                            Self::and_expr(),
                            // Avoid recognizing "A && B" as "A & (&B)"
                            lexeme(terminated(char('&'), not(char('&')))),
                            Self::equality_expr(),
                        ).map(|(lop, rop)| Expr::BitAnd(Box::new(lop), Box::new(rop)))
                    ),
                    Self::equality_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.11
        #[leftrec] rule exclusive_or_expr() -> Expr {
            lexeme(
                alt((
                    context("^ expr",
                        separated_pair(
                            Self::exclusive_or_expr(),
                            lexeme(char('^')),
                            Self::and_expr(),
                        ).map(|(lop, rop)| Expr::BitXor(Box::new(lop), Box::new(rop)))
                    ),
                    Self::and_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.12
        #[leftrec] rule inclusive_or_expr() -> Expr {
            lexeme(
                alt((
                    context("| expr",
                        separated_pair(
                            Self::inclusive_or_expr(),
                            lexeme(char('|')),
                            Self::exclusive_or_expr(),
                        ).map(|(lop, rop)| Expr::BitOr(Box::new(lop), Box::new(rop)))
                    ),
                    Self::exclusive_or_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.13
        #[leftrec] rule logical_and_expr() -> Expr {
            lexeme(
                alt((
                    context("&& expr",
                        separated_pair(
                            Self::logical_and_expr(),
                            lexeme(tag("&&")),
                            Self::inclusive_or_expr(),
                        ).map(|(lop, rop)| Expr::And(Box::new(lop), Box::new(rop)))
                    ),
                    Self::inclusive_or_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.14
        #[leftrec] rule logical_or_expr() -> Expr {
            lexeme(
                alt((
                    context("|| expr",
                        separated_pair(
                            Self::logical_or_expr(),
                            lexeme(tag("||")),
                            cut(Self::logical_and_expr()),
                        ).map(|(lop, rop)| Expr::Or(Box::new(lop), Box::new(rop)))
                    ),
                    Self::logical_and_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule conditional_expr() -> Expr {
            lexeme(
                alt((
                    context("ternary expr",
                        separated_pair(
                            context("ternary cond expr",
                                Self::logical_or_expr()
                            ),
                            lexeme(char('?')),
                            cut(separated_pair(
                                context("ternary true expr",
                                    opt(Self::expr())
                                ),
                                lexeme(char(':')),
                                context("ternary false expr",
                                    cut(Self::conditional_expr())
                                ),
                            )),
                        ).map(|(cond, (true_, false_))| {
                            // GNU extension allows "cond ?: false_" that is equivalent to
                            // "cond ? cond : false_". The only difference is that side effects of
                            // evaluating "cond" are not repeated, but it does not matter to our
                            // interpreter since we don't support side effects anyway.
                            let true_ = true_.unwrap_or(cond.clone());
                            Expr::Ternary(Box::new(cond), Box::new(true_), Box::new(false_))
                        })
                    ),
                    Self::logical_or_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.6
        rule constant_expr() -> Expr {
            Self::conditional_expr()
        }

        // https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
        rule statement_expr() -> Expr {
            context(
                "GNU C statement expressions",
                map_res_cut(
                    preceded(
                        lexeme(char('(')),
                        lexeme(char('{')),
                    ),
                    |_| Err(CParseError::UnsupportedConstruct("GNU C statement expressions are not supported".into()))
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule assignment_expr() -> Expr {
            lexeme(
                alt((
                    Self::statement_expr(),
                    context("assignment",
                        tuple((
                            Self::unary_expr(),
                            lexeme(alt((
                                tuple((
                                    tag("="),
                                    Self::assignment_expr(),
                                )),
                                tuple((
                                    alt((
                                        tag("*="),
                                        tag("/="),
                                        tag("%="),
                                        tag("+="),
                                        tag("-="),
                                        tag("<<="),
                                        tag(">>="),
                                        tag("&="),
                                        tag("^="),
                                        tag("|="),
                                    )),
                                    cut(Self::assignment_expr()),
                                ))
                            )))
                        )).map(|(lexpr, (op, rexpr))| {
                            use Expr::*;
                            match &op.fragment()[..] {
                                b"=" => Assign(Box::new(lexpr), Box::new(rexpr)),
                                b"*=" => Assign(Box::new(lexpr.clone()), Box::new(Mul(Box::new(lexpr), Box::new(rexpr)))),
                                b"/=" => Assign(Box::new(lexpr.clone()), Box::new(Div(Box::new(lexpr), Box::new(rexpr)))),
                                b"%=" => Assign(Box::new(lexpr.clone()), Box::new(Mod(Box::new(lexpr), Box::new(rexpr)))),
                                b"+=" => Assign(Box::new(lexpr.clone()), Box::new(Add(Box::new(lexpr), Box::new(rexpr)))),
                                b"-=" => Assign(Box::new(lexpr.clone()), Box::new(Sub(Box::new(lexpr), Box::new(rexpr)))),
                                b"<<=" => Assign(Box::new(lexpr.clone()), Box::new(LShift(Box::new(lexpr), Box::new(rexpr)))),
                                b">>=" => Assign(Box::new(lexpr.clone()), Box::new(RShift(Box::new(lexpr), Box::new(rexpr)))),
                                b"&=" => Assign(Box::new(lexpr.clone()), Box::new(BitAnd(Box::new(lexpr), Box::new(rexpr)))),
                                b"^=" => Assign(Box::new(lexpr.clone()), Box::new(BitXor(Box::new(lexpr), Box::new(rexpr)))),
                                b"|=" => Assign(Box::new(lexpr.clone()), Box::new(BitOr(Box::new(lexpr), Box::new(rexpr)))),
                                _ => panic!("unhandled assignment operator")
                            }
                        })
                    ),
                    Self::conditional_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        #[leftrec] rule unary_expr() -> Expr {
            lexeme(
                alt((
                    context("preinc expr",
                            preceded(
                                lexeme(tag("++")),
                                cut(Self::unary_expr()),
                            ).map(|e| Expr::PreInc(Box::new(e)))
                    ),
                    context("predec expr",
                            preceded(
                                lexeme(tag("--")),
                                cut(Self::unary_expr()),
                            ).map(|e| Expr::PreDec(Box::new(e)))
                    ),
                    context("sizeof type",
                            preceded(
                                keyword("sizeof"),
                                parenthesized(
                                    Self::type_name(),
                                )
                            ).map(Expr::SizeofType)
                    ),
                    context("sizeof expr",
                            preceded(
                                keyword("sizeof"),
                                Self::unary_expr(),
                            ).map(|e| Expr::SizeofExpr(Box::new(e)))
                    ),
                    context("unary op expr",
                        tuple((
                            lexeme(
                                // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
                                alt((
                                    context("unary &", char('&').map(|_| Box::new(|e| Expr::Addr(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary *", char('*').map(|_| Box::new(|e| Expr::Deref(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary +", char('+').map(|_| Box::new(|e| Expr::Plus(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary -", char('-').map(|_| Box::new(|e| Expr::Minus(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary ~", char('~').map(|_| Box::new(|e| Expr::Tilde(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary !", char('!').map(|_| Box::new(|e| Expr::Bang(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                ))
                            ),
                            Self::cast_expr(),
                        )).map(|(modify, e)| modify(e))
                    ),
                    Self::postfix_expr(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.7p1
        rule type_name() -> Type {
            lexeme(
                tuple((
                    Self::declaration_specifier(),
                    Self::declarator(true),
                )).map(|(typ, abstract_declarator)|
                    (abstract_declarator.modify_typ)(typ)
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.4p1
        rule cast_expr() -> Expr {
            lexeme(
                alt((
                    context("cast expr",
                        tuple((
                            parenthesized(
                                Self::type_name(),
                            ),
                            Self::cast_expr(),
                        )).map(|(typ, e)| Expr::Cast(typ, Box::new(e)))
                    ),
                    Self::unary_expr(),
                ))
            )
        }

        rule escape_sequence() -> u8 {
            escape_sequence()
        }


        // https://port70.net/~nsz/c/c11/n1570.html#6.4.4.4
        rule char_constant() -> Expr {
            lexeme(
                context("character constant",
                    map_res_cut(
                        tuple((
                            context(
                                "char encoding prefix",
                                opt(alt((keyword("u8"), keyword("u"), keyword("U"), keyword("L")))),
                            ),
                            delimited(
                                char('\''),
                                cut(alt((
                                    Self::escape_sequence(),
                                    u8,
                                ))),
                                char('\''),
                            )
                        )),
                        |(prefix, c)| match prefix {
                            Some(_) => Err(CParseError::UnsupportedConstruct("string encoding prefix syntax is not supported".into())),
                            None => Ok(Expr::CharConstant(Type::I32, c.into())),
                        },
                    )
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.4.5
        rule string_literal() -> Expr {
            string_literal()
        }

        rule int_constant() -> Expr {
            enum Kind {
                Dec,
                Hex,
                Oct,
            }
            let value = || alt((
                preceded(
                    char('0'),
                    alt((
                        context("octal constant",
                            dec_u64.map(|mut x| (
                                Kind::Oct,
                                {
                                    let mut val = 0;
                                    for i in 0.. {
                                        let rem = x % 10;
                                        val += rem * 8_u64.pow(i);
                                        x /= 10;
                                        if x == 0 {
                                            break
                                        }
                                    }
                                    val
                                }
                            ))
                        ),
                        context("hex constant",
                            preceded(
                                alt((char('x'), char('X'))),
                                cut(hex_u64.map(|x| (Kind::Hex, x))),
                            )
                        ),
                    ))
                ),
                context("decimal constant", dec_u64.map(|x| (Kind::Dec, x))),
            ));

            enum ConstantSignedness {
                Unsigned,
            }
            enum Longness {
                Long,
                LongLong,
            }

            let unsigned_suffix = || alt((char('u'), char('U'))).map(|_| ConstantSignedness::Unsigned);
            let long_suffix = || alt((
                tag("ll").map(|_| Longness::LongLong),
                tag("LL").map(|_| Longness::LongLong),
                tag("l").map(|_| Longness::Long),
                tag("L").map(|_| Longness::Long),
            ));

            let suffix = || alt((
                pair(
                    opt(unsigned_suffix()),
                    opt(long_suffix()),
                ),
                pair(
                    opt(long_suffix()),
                    opt(unsigned_suffix()),
                ).map(|(x, y)| (y, x)),
            ));


            lexeme(
                map_res_cut(
                    tuple((
                        Self::grammar_ctx(),
                        value(),
                        suffix(),
                    )),
                    move |(ctx, (kind, x), suffix)| {
                        let abi = &ctx.abi();
                        let long_size = abi.long_size;

                        macro_rules! max_val {
                            (U32) => {u32::MAX.into()};
                            (U64) => {u64::MAX.into()};

                            (I32) => {i32::MAX.try_into().unwrap()};
                            (I64) => {i64::MAX.try_into().unwrap()};
                            (Long) => {
                                match long_size {
                                    LongSize::Bits32 => max_val!(I32),
                                    LongSize::Bits64 => max_val!(I64),
                                }
                            };
                            (ULong) => {
                                match long_size {
                                    LongSize::Bits32 => max_val!(U32),
                                    LongSize::Bits64 => max_val!(U64),
                                }
                            };
                        }


                        let (ulong_ctype, long_ctype) = match long_size {
                            LongSize::Bits32 => (Type::U32, Type::I32),
                            LongSize::Bits64 => (Type::U64, Type::I64),
                        };
                        macro_rules! c_typ {
                            (Long) => {long_ctype.clone()};
                            (ULong) => {ulong_ctype.clone()};
                            ($typ:ident) => {Type::$typ};
                        }

                        macro_rules! guess_typ {
                            ($x:ident, $($typ:ident),*) => {
                                match $x {
                                    $(
                                        x if x <= max_val!($typ) => Ok(c_typ!($typ)),
                                    )*
                                    x => Err(CParseError::InvalidIntegerConstant(x)),
                                }
                            }
                        }

                        use Longness::*;
                        use ConstantSignedness::Unsigned;

                        // Encodes this table:
                        // https://port70.net/~nsz/c/c11/n1570.html#6.4.4.1p5
                        let typ = match (kind, suffix) {
                            (Kind::Dec, (None, None)) => guess_typ!(x, I32, I64),
                            (Kind::Dec, (Some(Unsigned), None)) => guess_typ!(x, U32, U64),

                            (Kind::Dec, (None, Some(Long))) => guess_typ!(x, Long, I64),
                            (Kind::Dec, (Some(Unsigned), Some(Long))) => guess_typ!(x, ULong, U64),

                            (Kind::Dec, (None, Some(LongLong))) => guess_typ!(x, I64),
                            (Kind::Dec, (Some(Unsigned), Some(LongLong))) => guess_typ!(x, U64),

                            // Oct and Hex constant have different type inference rules
                            (_, (None, None)) => guess_typ!(x, I32, U32, Long, ULong, I64, U64),
                            (_, (Some(Unsigned), None)) => guess_typ!(x, U32, U64),

                            (_, (None, Some(Long))) => guess_typ!(x, Long, ULong, I64, U64),
                            (_, (Some(Unsigned), Some(Long))) => guess_typ!(x, ULong, U64),

                            (_, (None, Some(LongLong))) => guess_typ!(x, I64, U64),
                            (_, (Some(Unsigned), Some(LongLong))) => guess_typ!(x, U64),
                        }?;
                        Ok(Expr::IntConstant(typ, x))
                    }
                )
            )
        }

        rule enum_constant() -> Expr {
            lexeme(context("enum constant",
                Self::identifier().map(|id| Expr::EnumConstant(Type::I32, id))
            ))
        }

        rule constant() -> Expr {
            context("constant",
                alt((
                    Self::char_constant(),
                    Self::int_constant(),
                    Self::enum_constant(),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.1p1
        rule primary_expr() -> Expr
        {
            lexeme(
                alt((
                    parenthesized(
                        Self::expr()
                    ),
                    Self::string_literal(),
                    map_res(
                        Self::identifier(),
                        |id| match id.deref() {
                            "REC" => {
                                // Make a REC variable and then take its
                                // address, rather than making a pointer-typed
                                // variable. This will allow the interpreter to
                                // simplify "REC->x" as it will see "(&REC)->x"
                                // that will get turned into "(*&REC).x" and
                                // then "REC.x". Doing it this way plays nicely
                                // with constant folding.
                                let typ = Type::Variable(id.clone());
                                Ok(Expr::Addr(Box::new(Expr::Variable(typ, id))))
                            },
                            _ => {
                                let kind = resolve_extension_macro(&id)?;

                                Ok(Expr::ExtensionMacro(Arc::new(
                                    ExtensionMacroDesc {
                                        name: id,
                                        kind,
                                    }
                                )))
                            },
                        }
                    ),
                    Self::constant(),
                ))
            )
        }

        rule expr() -> Expr {
            lexeme(
                context("expression",
                    separated_list1(
                        lexeme(char(',')),
                        Self::assignment_expr(),
                    ).map(|mut exprs| {
                        if exprs.len() == 1 {
                            exprs.remove(0)
                        } else {
                            Expr::CommaExpr(exprs)
                        }
                    })
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        closure::closure, cparser, grammar::PackratGrammar, header::Signedness,
        parser::tests::test_parser,
    };

    #[test]
    fn expr_test() {
        fn test(src: &[u8], expected: Expr) {
            let abi = Abi {
                long_size: LongSize::Bits64,
                endianness: Endianness::Little,
                char_signedness: Signedness::Unsigned,
            };
            let parser = CGrammar::expr();
            let ctx = CGrammarCtx::new(&abi);
            let src = CGrammar::make_span(src, &ctx);
            test_parser(expected, src, parser);
        }

        use Expr::*;

        let rec_var = Addr(Box::new(Variable(
            Type::Variable("REC".into()),
            "REC".into(),
        )));

        let extension_compiler = Arc::new(closure!(
            (
                for<'ceref, 'ce, 'a> Fn(
                    &'ceref (dyn CompileEnv<'ce> + 'ceref),
                ) -> Result<Box<dyn Evaluator>, CompileError>
            ) + Send + Sync,
            |_| panic!("non-implemented compiler")
        ));

        // Decimal literal
        test(b"1", IntConstant(Type::I32, 1));
        test(b"42", IntConstant(Type::I32, 42));
        test(b" 1 ", IntConstant(Type::I32, 1));
        test(b" 42 ", IntConstant(Type::I32, 42));

        test(
            b"1125899906842624",
            IntConstant(Type::I64, 1125899906842624),
        );
        test(
            b"18446744073709551615u",
            IntConstant(Type::U64, 18446744073709551615),
        );

        // Octal literal
        test(b"01", IntConstant(Type::I32, 1));
        test(b"0777", IntConstant(Type::I32, 511));
        test(b"01234", IntConstant(Type::I32, 668));

        // Hexadecimal literal
        test(b"0x1", IntConstant(Type::I32, 1));
        test(b"0x1234", IntConstant(Type::I32, 4660));
        test(b"0x777", IntConstant(Type::I32, 1911));
        test(b"0X1", IntConstant(Type::I32, 1));
        test(b"0X1234", IntConstant(Type::I32, 4660));
        test(b"0X777", IntConstant(Type::I32, 1911));

        // Char constant
        test(b"'a'", CharConstant(Type::I32, 'a'.into()));
        test(br#"'\n'"#, CharConstant(Type::I32, '\n'.into()));
        test(br#"'\xff'"#, CharConstant(Type::I32, 0xff));
        test(br#"'\012'"#, CharConstant(Type::I32, 0o12));
        test(br#"'\0'"#, CharConstant(Type::I32, 0));

        // String literal
        test(br#""a""#, StringLiteral("a".into()));
        test(br#"" hello world ""#, StringLiteral(" hello world ".into()));
        test(
            br#""1 hello world ""#,
            StringLiteral("1 hello world ".into()),
        );
        test(
            br#""hello \n world""#,
            StringLiteral("hello \n world".into()),
        );
        test(br#""\n\t\\""#, StringLiteral("\n\t\\".into()));

        // Address of
        test(b" &1 ", Addr(Box::new(IntConstant(Type::I32, 1))));

        // Deref
        test(
            b" *&1 ",
            Deref(Box::new(Addr(Box::new(IntConstant(Type::I32, 1))))),
        );

        // Unary
        test(b"1", IntConstant(Type::I32, 1));
        test(b" 1 ", IntConstant(Type::I32, 1));
        test(b"+1", Plus(Box::new(IntConstant(Type::I32, 1))));
        test(b" + 1 ", Plus(Box::new(IntConstant(Type::I32, 1))));
        test(b"-1", Minus(Box::new(IntConstant(Type::I32, 1))));
        test(b" - 1 ", Minus(Box::new(IntConstant(Type::I32, 1))));
        test(b" ~ 1 ", Tilde(Box::new(IntConstant(Type::I32, 1))));
        test(b"!1 ", Bang(Box::new(IntConstant(Type::I32, 1))));
        test(b" ! 1 ", Bang(Box::new(IntConstant(Type::I32, 1))));

        // Cast
        test(
            b"(int)1 ",
            Cast(Type::I32, Box::new(IntConstant(Type::I32, 1))),
        );
        test(
            b"(type)1 ",
            Cast(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                Box::new(IntConstant(Type::I32, 1)),
            ),
        );
        test(
            b"(type)(1) ",
            Cast(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                Box::new(IntConstant(Type::I32, 1)),
            ),
        );
        test(
            b"-(int)1 ",
            Minus(Box::new(Cast(
                Type::I32,
                Box::new(IntConstant(Type::I32, 1)),
            ))),
        );
        test(
            b"-(int)(unsigned long)1 ",
            Minus(Box::new(Cast(
                Type::I32,
                Box::new(Cast(Type::U64, Box::new(IntConstant(Type::I32, 1)))),
            ))),
        );
        test(
            b"(__typeof__(42))1 ",
            Cast(Type::I32, Box::new(IntConstant(Type::I32, 1))),
        );

        test(
            b"(__typeof__(42ULL))1 ",
            Cast(Type::U64, Box::new(IntConstant(Type::I32, 1))),
        );

        // Sizeof type
        test(b"sizeof(unsigned long)", SizeofType(Type::U64));
        test(
            b"sizeof (s32)",
            SizeofType(Type::Typedef(Box::new(Type::I32), "s32".into())),
        );
        test(b"sizeof (__typeof__(1))", SizeofType(Type::I32));
        test(b"sizeof (__typeof__(1ULL))", SizeofType(Type::U64));
        test(
            b"sizeof(struct page)",
            SizeofType(Type::Struct("page".into())),
        );

        // Sizeof expr
        test(
            b"sizeof(1)",
            SizeofExpr(Box::new(IntConstant(Type::I32, 1))),
        );

        test(
            b"sizeof(-(int)1)",
            SizeofExpr(Box::new(Minus(Box::new(Cast(
                Type::I32,
                Box::new(IntConstant(Type::I32, 1)),
            ))))),
        );
        test(
            b"sizeof - (int ) 1 ",
            SizeofExpr(Box::new(Minus(Box::new(Cast(
                Type::I32,
                Box::new(IntConstant(Type::I32, 1)),
            ))))),
        );

        // Pre-increment
        test(b"++ 42 ", PreInc(Box::new(IntConstant(Type::I32, 42))));
        test(
            b"++ sizeof - (int ) 1 ",
            PreInc(Box::new(SizeofExpr(Box::new(Minus(Box::new(Cast(
                Type::I32,
                Box::new(IntConstant(Type::I32, 1)),
            ))))))),
        );

        // Pre-decrement
        test(
            b"-- -42 ",
            PreDec(Box::new(Minus(Box::new(IntConstant(Type::I32, 42))))),
        );

        // Addition
        test(
            b"1+2",
            Add(
                Box::new(IntConstant(Type::I32, 1)),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );
        test(
            b" 1 + 2 ",
            Add(
                Box::new(IntConstant(Type::I32, 1)),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );
        test(
            b" (1) + (2) ",
            Add(
                Box::new(IntConstant(Type::I32, 1)),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );

        // Operator precedence
        test(
            b" 1 + 2 * 3",
            Add(
                Box::new(IntConstant(Type::I32, 1)),
                Box::new(Mul(
                    Box::new(IntConstant(Type::I32, 2)),
                    Box::new(IntConstant(Type::I32, 3)),
                )),
            ),
        );

        test(
            b" 1 * 2 + 3",
            Add(
                Box::new(Mul(
                    Box::new(IntConstant(Type::I32, 1)),
                    Box::new(IntConstant(Type::I32, 2)),
                )),
                Box::new(IntConstant(Type::I32, 3)),
            ),
        );

        test(
            b" 1 * 2 + 3 << 4",
            LShift(
                Box::new(Add(
                    Box::new(Mul(
                        Box::new(IntConstant(Type::I32, 1)),
                        Box::new(IntConstant(Type::I32, 2)),
                    )),
                    Box::new(IntConstant(Type::I32, 3)),
                )),
                Box::new(IntConstant(Type::I32, 4)),
            ),
        );

        test(
            b" 1 * 2 + 3 << 4 | 5",
            BitOr(
                Box::new(LShift(
                    Box::new(Add(
                        Box::new(Mul(
                            Box::new(IntConstant(Type::I32, 1)),
                            Box::new(IntConstant(Type::I32, 2)),
                        )),
                        Box::new(IntConstant(Type::I32, 3)),
                    )),
                    Box::new(IntConstant(Type::I32, 4)),
                )),
                Box::new(IntConstant(Type::I32, 5)),
            ),
        );

        // Function call
        test(
            b"f(1)",
            FuncCall(
                Box::new(EnumConstant(Type::I32, "f".into())),
                vec![IntConstant(Type::I32, 1)],
            ),
        );
        test(
            b" f (1, 2, 3) ",
            FuncCall(
                Box::new(EnumConstant(Type::I32, "f".into())),
                vec![
                    IntConstant(Type::I32, 1),
                    IntConstant(Type::I32, 2),
                    IntConstant(Type::I32, 3),
                ],
            ),
        );
        test(
            // This could be either a cast or a function call.
            b" (type)(1, 2, 3)",
            Cast(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                Box::new(CommaExpr(vec![
                    IntConstant(Type::I32, 1),
                    IntConstant(Type::I32, 2),
                    IntConstant(Type::I32, 3),
                ])),
            ),
        );

        // Subscript
        test(
            b"REC[1]",
            Subscript(
                Box::new(rec_var.clone()),
                Box::new(IntConstant(Type::I32, 1)),
            ),
        );
        test(
            b"REC[1][2]",
            Subscript(
                Box::new(Subscript(
                    Box::new(rec_var.clone()),
                    Box::new(IntConstant(Type::I32, 1)),
                )),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );

        // Member access
        test(
            b"REC.y",
            MemberAccess(Box::new(rec_var.clone()), "y".into()),
        );
        test(
            b" REC . y ",
            MemberAccess(Box::new(rec_var.clone()), "y".into()),
        );
        test(
            b"REC.y.z",
            MemberAccess(
                Box::new(MemberAccess(Box::new(rec_var.clone()), "y".into())),
                "z".into(),
            ),
        );
        test(
            b"REC->y",
            MemberAccess(Box::new(Deref(Box::new(rec_var.clone()))), "y".into()),
        );

        test(
            b"REC->y->z",
            MemberAccess(
                Box::new(Deref(Box::new(MemberAccess(
                    Box::new(Deref(Box::new(rec_var.clone()))),
                    "y".into(),
                )))),
                "z".into(),
            ),
        );

        // Compound literal
        test(
            b"(type){0}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![IntConstant(Type::I32, 0)],
            ),
        );
        test(
            b"(type){0, 1}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![IntConstant(Type::I32, 0), IntConstant(Type::I32, 1)],
            ),
        );
        test(
            b"(type){.x = 0}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![DesignatedInitializer(
                    Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                    Box::new(IntConstant(Type::I32, 0)),
                )],
            ),
        );
        test(
            b"(type){.x = 0, }",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![DesignatedInitializer(
                    Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                    Box::new(IntConstant(Type::I32, 0)),
                )],
            ),
        );
        test(
            b"(type){.x = {0, 1}}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![DesignatedInitializer(
                    Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                    Box::new(InitializerList(vec![
                        IntConstant(Type::I32, 0),
                        IntConstant(Type::I32, 1),
                    ])),
                )],
            ),
        );
        test(
            b"(type){.x = (type2){0}}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![DesignatedInitializer(
                    Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                    Box::new(CompoundLiteral(
                        Type::Typedef(Box::new(Type::Unknown), "type2".into()),
                        vec![IntConstant(Type::I32, 0)],
                    )),
                )],
            ),
        );
        test(
            b"(type){.x = {(type2){0}, (type3){1, 2}}, .y={3}, .z=4}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![
                    DesignatedInitializer(
                        Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                        Box::new(InitializerList(vec![
                            CompoundLiteral(
                                Type::Typedef(Box::new(Type::Unknown), "type2".into()),
                                vec![IntConstant(Type::I32, 0)],
                            ),
                            CompoundLiteral(
                                Type::Typedef(Box::new(Type::Unknown), "type3".into()),
                                vec![IntConstant(Type::I32, 1), IntConstant(Type::I32, 2)],
                            ),
                        ])),
                    ),
                    DesignatedInitializer(
                        Box::new(MemberAccess(Box::new(Uninit), "y".into())),
                        Box::new(InitializerList(vec![IntConstant(Type::I32, 3)])),
                    ),
                    DesignatedInitializer(
                        Box::new(MemberAccess(Box::new(Uninit), "z".into())),
                        Box::new(IntConstant(Type::I32, 4)),
                    ),
                ],
            ),
        );
        test(
            b"(type){.x[0] = 1}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![DesignatedInitializer(
                    Box::new(Subscript(
                        Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                        Box::new(IntConstant(Type::I32, 0)),
                    )),
                    Box::new(IntConstant(Type::I32, 1)),
                )],
            ),
        );
        test(
            b"(type){.x[0].y = 1}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![DesignatedInitializer(
                    Box::new(MemberAccess(
                        Box::new(Subscript(
                            Box::new(MemberAccess(Box::new(Uninit), "x".into())),
                            Box::new(IntConstant(Type::I32, 0)),
                        )),
                        "y".into(),
                    )),
                    Box::new(IntConstant(Type::I32, 1)),
                )],
            ),
        );
        test(
            b"(type){[0]= 1, 2}",
            CompoundLiteral(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                vec![
                    DesignatedInitializer(
                        Box::new(Subscript(
                            Box::new(Uninit),
                            Box::new(IntConstant(Type::I32, 0)),
                        )),
                        Box::new(IntConstant(Type::I32, 1)),
                    ),
                    IntConstant(Type::I32, 2),
                ],
            ),
        );

        // Comma operator
        test(
            b"(1,2)",
            CommaExpr(vec![IntConstant(Type::I32, 1), IntConstant(Type::I32, 2)]),
        );

        // Ambiguous cases

        // Amibiguity of is lifted by 6.4p4 stating that the tokenizer is
        // greedy, i.e. the following is tokenized as "1 ++ + 2":
        // https://port70.net/~nsz/c/c11/n1570.html#6.4p4
        test(
            b" 1 +++ 2 ",
            Add(
                Box::new(PostInc(Box::new(IntConstant(Type::I32, 1)))),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );
        test(
            b" 1 +++++ 2 ",
            Add(
                Box::new(PostInc(Box::new(PostInc(Box::new(IntConstant(
                    Type::I32,
                    1,
                )))))),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );

        test(
            b" 1 --- 2 ",
            Sub(
                Box::new(PostDec(Box::new(IntConstant(Type::I32, 1)))),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );
        test(
            b" 1 ----- 2 ",
            Sub(
                Box::new(PostDec(Box::new(PostDec(Box::new(IntConstant(
                    Type::I32,
                    1,
                )))))),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );

        // This is genuinely ambiguous: it can be either a cast to type "type"
        // of "+2" or the addition of a "type" variable and 2.
        // We parse it as a cast as the expressions we are interested in only
        // contain one variable (REC).
        test(
            b" (type) + (2) ",
            Cast(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                Box::new(Plus(Box::new(IntConstant(Type::I32, 2)))),
            ),
        );

        // Another ambiguous case: could be a function call or a cast. We decide
        // to treat that as a cast, since you can make a call without the extra
        // paren.
        test(
            b" (type)(2) ",
            Cast(
                Type::Typedef(Box::new(Type::Unknown), "type".into()),
                Box::new(IntConstant(Type::I32, 2)),
            ),
        );

        // More complex cases
        test(
            br#"(REC->prev_state & ((((0x0000 | 0x0001 | 0x0002) + 1) << 1) - 1)) ? __print_flags(REC->prev_state & ((((0x0000 | 0x0001 | 0x0002) + 1) << 1) - 1), "|", { 0x0001, "S" }, { 0x0002, "D" }) : "R", REC->prev_state & (((0x0000 | 0x0001 | 0x0002) + 1) << 1) ? "+" : """#,
            CommaExpr(vec![
                Ternary(
                    Box::new(BitAnd(
                        Box::new(MemberAccess(
                            Box::new(Deref(Box::new(
                                Addr(Box::new(
                                    Variable(Type::Variable("REC".into()), "REC".into())
                                ))
                            ))),
                            "prev_state".into()
                        )),
                        Box::new(Sub(
                            Box::new(LShift(
                                Box::new(Add(
                                    Box::new(BitOr(
                                        Box::new(BitOr(
                                            Box::new(IntConstant(Type::I32, 0)),
                                            Box::new(IntConstant(Type::I32, 1))
                                        )),
                                        Box::new(IntConstant(Type::I32, 2))
                                    )),
                                    Box::new(IntConstant(Type::I32, 1))
                                )),
                                Box::new(IntConstant(Type::I32, 1))
                            )),
                            Box::new(IntConstant(Type::I32, 1))
                        ))
                    )),
                    Box::new(ExtensionMacroCall(cparser::ExtensionMacroCall {
                        args: br#"REC->prev_state & ((((0x0000 | 0x0001 | 0x0002) + 1) << 1) - 1), "|", { 0x0001, "S" }, { 0x0002, "D" }"#.to_vec(),
                        desc: Arc::new(
                            ExtensionMacroDesc {
                                name: "__print_flags".into(),
                                kind: ExtensionMacroKind::FunctionLike {
                                    parser: Box::new(|_, _| panic!("non implemented parser"))
                                },
                            }
                        ),
                        compiler: ExtensionMacroCallCompiler {
                            ret_typ: Type::Pointer(Box::new(Type::U8)),
                            compiler: extension_compiler.clone()
                        }
                    })),
                    Box::new(StringLiteral("R".into()))
                ),
                Ternary(
                    Box::new(BitAnd(
                        Box::new(MemberAccess(
                            Box::new(Deref(
                                Box::new(Addr(
                                    Box::new(Variable(
                                        Type::Variable("REC".into()),
                                        "REC".into()
                                    ))
                                )),
                            )),
                            "prev_state".into()
                        )),
                        Box::new(LShift(
                            Box::new(Add(
                                Box::new(BitOr(
                                    Box::new(BitOr(
                                        Box::new(IntConstant(Type::I32, 0)),
                                        Box::new(IntConstant(Type::I32, 1))
                                    )),
                                    Box::new(IntConstant(Type::I32, 2))
                                )),
                                Box::new(IntConstant(Type::I32, 1))
                            )),
                            Box::new(IntConstant(Type::I32, 1))
                        ))
                    )),
                    Box::new(StringLiteral("+".into())),
                    Box::new(StringLiteral("".into()))
                )
            ])
        );

        test(br#"(("hello"))"#, StringLiteral("hello".into()));

        test(
            br#"(!!(sizeof(1 == 1)))"#,
            Bang(Box::new(Bang(Box::new(SizeofExpr(Box::new(Eq(
                Box::new(IntConstant(Type::I32, 1)),
                Box::new(IntConstant(Type::I32, 1)),
            ))))))),
        );

        test(
            br#"sizeof(typeof(2) *)"#,
            SizeofType(Type::Pointer(Box::new(Type::I32))),
        );
    }

    #[test]
    fn declaration_test() {
        fn test<'a>(decl: &'a [u8], id: &'a str, typ: Type) {
            let expected = Declaration {
                identifier: id.into(),
                typ,
            };
            let abi = Abi {
                long_size: LongSize::Bits64,
                endianness: Endianness::Little,
                char_signedness: Signedness::Unsigned,
            };
            let parser = CGrammar::declaration();
            let ctx = CGrammarCtx::new(&abi);
            let decl = CGrammar::make_span(decl, &ctx);
            test_parser(expected, decl, parser);
        }

        let u64_typ = Type::Typedef(Box::new(Type::U64), "u64".into());
        let s64_typ = Type::Typedef(Box::new(Type::I64), "s64".into());

        // Basic
        test(b"u64 foo", "foo", u64_typ.clone());
        test(b"u64 static_foo", "static_foo", u64_typ.clone());
        test(b"u64 static foo", "foo", u64_typ.clone());

        test(b" u64  \t foo\t", "foo", u64_typ.clone());
        test(b" const volatile u64 foo", "foo", u64_typ.clone());
        test(b" u64 const volatile foo", "foo", u64_typ.clone());
        test(
            b" const\t volatile  _Atomic u64  \t foo\t",
            "foo",
            u64_typ.clone(),
        );
        test(b"int interval", "interval", Type::I32);

        // Structs + Enum + Union
        test(
            b"struct mystruct foo",
            "foo",
            Type::Struct("mystruct".into()),
        );
        test(
            b"struct structmy foo",
            "foo",
            Type::Struct("structmy".into()),
        );
        test(
            b"enum mystruct foo",
            "foo",
            Type::Enum(Box::new(Type::Unknown), "mystruct".into()),
        );
        test(b"union mystruct foo", "foo", Type::Union("mystruct".into()));

        // Signed/Unsigned
        test(b"int signed extern const  foo  ", "foo", Type::I32);
        test(b"signed extern const  foo  ", "foo", Type::I32);
        test(
            b"_Atomic\t unsigned    extern const  foo  ",
            "foo",
            Type::U32,
        );
        test(b"int unsigned extern const  foo  ", "foo", Type::U32);
        test(
            b" long \t long unsigned extern const  foo  ",
            "foo",
            Type::U64,
        );

        test(
            b"int long long unsigned extern const  foo  ",
            "foo",
            Type::U64,
        );

        test(
            b"long extern int long unsigned const  foo  ",
            "foo",
            Type::U64,
        );

        // Pointers
        test(b"u64 *foo", "foo", Type::Pointer(Box::new(u64_typ.clone())));
        test(
            b" u64 * \tfoo ",
            "foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );
        test(b"u64 *foo", "foo", Type::Pointer(Box::new(u64_typ.clone())));
        test(
            b" u64 * \tfoo ",
            "foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );
        test(
            b"u64 * const foo",
            "foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );
        test(
            b" u64 * \tconst\tfoo ",
            "foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );
        test(
            b" long unsigned long * \tconst\tfoo ",
            "foo",
            Type::Pointer(Box::new(Type::U64)),
        );
        test(
            b" const volatile u64 * const foo",
            "foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );
        test(
            b" const\tvolatile u64 * \tconst\tfoo ",
            "foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );
        test(
            b" const\tvolatile u64 * const * \tconst\tfoo ",
            "foo",
            Type::Pointer(Box::new(Type::Pointer(Box::new(u64_typ.clone())))),
        );
        test(
            b" const\tvolatile u64 _Atomic * const * \tconst\tfoo ",
            "foo",
            Type::Pointer(Box::new(Type::Pointer(Box::new(u64_typ.clone())))),
        );
        test(
            b"struct callback_head * rhp",
            "rhp",
            Type::Pointer(Box::new(Type::Struct("callback_head".into()))),
        );

        test(
            b"u64 *static_foo",
            "static_foo",
            Type::Pointer(Box::new(u64_typ.clone())),
        );

        // Arrays
        test(
            b" u64 foo\t []",
            "foo",
            Type::Array(Box::new(u64_typ.clone()), ArrayKind::ZeroLength),
        );
        test(
            b" u64 foo\t []\t\t",
            "foo",
            Type::Array(Box::new(u64_typ.clone()), ArrayKind::ZeroLength),
        );
        test(
            b" u64 foo\t [124]",
            "foo",
            Type::Array(Box::new(u64_typ.clone()), ArrayKind::Fixed(Ok(124))),
        );
        test(
            b" u64 foo\t [static 124]",
            "foo",
            Type::Array(Box::new(u64_typ.clone()), ArrayKind::Fixed(Ok(124))),
        );
        test(
            b"u64 foo [static_bar]",
            "foo",
            Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Fixed(Err(Box::new(InterpError::CompileError(Box::new(
                    CompileError::ExprNotHandled(Expr::EnumConstant(
                        Type::I32,
                        "static_bar".into(),
                    )),
                ))))),
            ),
        );
        test(
            b" u64 (*foo) [1]",
            "foo",
            Type::Pointer(Box::new(Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Fixed(Ok(1)),
            ))),
        );
        test(
            b" u64 ((*foo)) [1]",
            "foo",
            Type::Pointer(Box::new(Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Fixed(Ok(1)),
            ))),
        );
        test(
            b" u64 (*foo[]) [1]",
            "foo",
            Type::Array(
                Box::new(Type::Pointer(Box::new(Type::Array(
                    Box::new(u64_typ.clone()),
                    ArrayKind::Fixed(Ok(1)),
                )))),
                ArrayKind::ZeroLength,
            ),
        );
        test(
            b" u64(*foo[]\t)[1]",
            "foo",
            Type::Array(
                Box::new(Type::Pointer(Box::new(Type::Array(
                    Box::new(u64_typ.clone()),
                    ArrayKind::Fixed(Ok(1)),
                )))),
                ArrayKind::ZeroLength,
            ),
        );
        test(
            b" u64 foo\t [A+B]\t\t",
            "foo",
            Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Fixed(Err(Box::new(InterpError::CompileError(Box::new(
                    CompileError::ExprNotHandled(Expr::EnumConstant(Type::I32, "A".into())),
                ))))),
            ),
        );

        // Nested arrays
        test(
            b" u64 foo\t [][]",
            "foo",
            Type::Array(
                Box::new(Type::Array(
                    Box::new(u64_typ.clone()),
                    ArrayKind::ZeroLength,
                )),
                ArrayKind::ZeroLength,
            ),
        );
        test(
            b" u64  foo\t [1][2][3]",
            "foo",
            Type::Array(
                Box::new(Type::Array(
                    Box::new(Type::Array(
                        Box::new(u64_typ.clone()),
                        ArrayKind::Fixed(Ok(3)),
                    )),
                    ArrayKind::Fixed(Ok(2)),
                )),
                ArrayKind::Fixed(Ok(1)),
            ),
        );
        test(
            b" u64 (*foo[3]) [2][1] ",
            "foo",
            Type::Array(
                Box::new(Type::Pointer(Box::new(Type::Array(
                    Box::new(Type::Array(
                        Box::new(u64_typ.clone()),
                        ArrayKind::Fixed(Ok(1)),
                    )),
                    ArrayKind::Fixed(Ok(2)),
                )))),
                ArrayKind::Fixed(Ok(3)),
            ),
        );

        // Function pointers
        test(
            b"int(*f)()",
            "f",
            Type::Pointer(Box::new(Type::Function(Box::new(Type::I32), Vec::new()))),
        );
        test(
            b" int ( * f ) ( ) ",
            "f",
            Type::Pointer(Box::new(Type::Function(Box::new(Type::I32), Vec::new()))),
        );
        test(
            b"int(*f)(unsigned int, s64 param1)",
            "f",
            Type::Pointer(Box::new(Type::Function(
                Box::new(Type::I32),
                vec![
                    ParamDeclaration {
                        identifier: None,
                        typ: Type::U32,
                    },
                    ParamDeclaration {
                        identifier: Some("param1".into()),
                        typ: s64_typ.clone(),
                    },
                ],
            ))),
        );
        test(
            b"foobar(*f)(unsigned int, s64 param1, pid_t(*param2)(u64))[1]",
            "f",
            Type::Pointer(Box::new(Type::Function(
                Box::new(Type::Array(
                    Box::new(Type::Typedef(Box::new(Type::Unknown), "foobar".into())),
                    ArrayKind::Fixed(Ok(1)),
                )),
                vec![
                    ParamDeclaration {
                        identifier: None,
                        typ: Type::U32,
                    },
                    ParamDeclaration {
                        identifier: Some("param1".into()),
                        typ: s64_typ.clone(),
                    },
                    ParamDeclaration {
                        identifier: Some("param2".into()),
                        typ: Type::Pointer(Box::new(Type::Function(
                            Box::new(Type::Typedef(Box::new(Type::I32), "pid_t".into())),
                            vec![ParamDeclaration {
                                identifier: None,
                                typ: u64_typ.clone(),
                            }],
                        ))),
                    },
                ],
            ))),
        );
        test(
            b"foobar(* const arr[2])(unsigned int, s64 param1, pid_t(*param2)(u64))[1]",
            "arr",
            Type::Array(
                Box::new(Type::Pointer(Box::new(Type::Function(
                    Box::new(Type::Array(
                        Box::new(Type::Typedef(Box::new(Type::Unknown), "foobar".into())),
                        ArrayKind::Fixed(Ok(1)),
                    )),
                    vec![
                        ParamDeclaration {
                            identifier: None,
                            typ: Type::U32,
                        },
                        ParamDeclaration {
                            identifier: Some("param1".into()),
                            typ: s64_typ.clone(),
                        },
                        ParamDeclaration {
                            identifier: Some("param2".into()),
                            typ: Type::Pointer(Box::new(Type::Function(
                                Box::new(Type::Typedef(Box::new(Type::I32), "pid_t".into())),
                                vec![ParamDeclaration {
                                    identifier: None,
                                    typ: u64_typ.clone(),
                                }],
                            ))),
                        },
                    ],
                )))),
                ArrayKind::Fixed(Ok(2)),
            ),
        );

        test(
            b"short (*(*foo)(int))(void)",
            "foo",
            Type::Pointer(Box::new(Type::Function(
                Box::new(Type::Pointer(Box::new(Type::Function(
                    Box::new(Type::I16),
                    vec![ParamDeclaration {
                        identifier: None,
                        typ: Type::Void,
                    }],
                )))),
                vec![ParamDeclaration {
                    identifier: None,
                    typ: Type::I32,
                }],
            ))),
        );

        // Scalar __data_loc
        test(
            b"__data_loc u64 foo",
            "foo",
            Type::DynamicScalar(Box::new(u64_typ.clone()), DynamicKind::Dynamic),
        );

        test(
            b"__data_loc cpumask_t mask",
            "mask",
            Type::DynamicScalar(
                Box::new(Type::Typedef(Box::new(Type::Unknown), "cpumask_t".into())),
                DynamicKind::Dynamic,
            ),
        );

        test(
            b"__data_loc u64* foo",
            "foo",
            Type::DynamicScalar(
                Box::new(Type::Pointer(Box::new(u64_typ.clone()))),
                DynamicKind::Dynamic,
            ),
        );

        test(
            b"__data_loc unsigned volatile * const foo",
            "foo",
            Type::DynamicScalar(
                Box::new(Type::Pointer(Box::new(Type::U32))),
                DynamicKind::Dynamic,
            ),
        );

        test(
            b"__data_loc u64 (*)[3] foo",
            "foo",
            Type::DynamicScalar(
                Box::new(Type::Pointer(Box::new(Type::Array(
                    Box::new(u64_typ.clone()),
                    ArrayKind::Fixed(Ok(3)),
                )))),
                DynamicKind::Dynamic,
            ),
        );

        test(
            b"__data_loc const u64 _Atomic ( * volatile)[3] foo",
            "foo",
            Type::DynamicScalar(
                Box::new(Type::Pointer(Box::new(Type::Array(
                    Box::new(u64_typ.clone()),
                    ArrayKind::Fixed(Ok(3)),
                )))),
                DynamicKind::Dynamic,
            ),
        );

        // Array __data_loc and __rel_loc
        test(
            b"__rel_loc u64[] foo",
            "foo",
            Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Dynamic(DynamicKind::DynamicRel),
            ),
        );

        test(
            b"__data_loc u64[] foo",
            "foo",
            Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Dynamic(DynamicKind::Dynamic),
            ),
        );

        test(
            b"   __data_loc\t u64  []foo",
            "foo",
            Type::Array(
                Box::new(u64_typ.clone()),
                ArrayKind::Dynamic(DynamicKind::Dynamic),
            ),
        );
        test(
            b"   __data_loc\t u64  [42][]foo",
            "foo",
            Type::Array(
                Box::new(Type::Array(
                    Box::new(u64_typ.clone()),
                    ArrayKind::Fixed(Ok(42)),
                )),
                ArrayKind::Dynamic(DynamicKind::Dynamic),
            ),
        );

        test(
            b"   __data_loc\t u64  [42][43][]foo",
            "foo",
            Type::Array(
                Box::new(Type::Array(
                    Box::new(Type::Array(
                        Box::new(u64_typ.clone()),
                        ArrayKind::Fixed(Ok(43)),
                    )),
                    ArrayKind::Fixed(Ok(42)),
                )),
                ArrayKind::Dynamic(DynamicKind::Dynamic),
            ),
        );

        test(
            b"   __data_loc u64 (*[3]) [2][]foo",
            "foo",
            Type::Array(
                Box::new(Type::Pointer(Box::new(Type::Array(
                    Box::new(Type::Array(
                        Box::new(u64_typ.clone()),
                        ArrayKind::Fixed(Ok(2)),
                    )),
                    ArrayKind::Fixed(Ok(3)),
                )))),
                ArrayKind::Dynamic(DynamicKind::Dynamic),
            ),
        );

        // All together
        test(
            b" const\tvolatile unsigned int volatile*const _Atomic* \tconst\tfoo [sizeof(struct foo)] \t[42] ",
            "foo",
            Type::Array(
                Box::new(Type::Array(
                    Box::new(Type::Pointer(Box::new(Type::Pointer(Box::new(
                        Type::U32,
                    ))))),
                    ArrayKind::Fixed(Ok(42)),
                )),
                ArrayKind::Fixed(Err(Box::new(InterpError::CompileError(Box::new(CompileError::UnknownSize(Type::Struct("foo".into()))))))),
            ),
        );
        test(
            b" const\tvolatile int volatile signed*const _Atomic* \tconst\tfoo [sizeof(struct foo)] \t[42] ",
            "foo",
            Type::Array(
                Box::new(Type::Array(
                    Box::new(Type::Pointer(Box::new(Type::Pointer(Box::new(
                        Type::I32,
                    ))))),
                    ArrayKind::Fixed(Ok(42)),
                )),
                ArrayKind::Fixed(Err(Box::new(InterpError::CompileError(Box::new(CompileError::UnknownSize(Type::Struct("foo".into()))))))),
            ),
        );

        test(
            b" __data_loc\tconst\tvolatile int volatile signed*const _Atomic* \tconst \t[] foo",
            "foo",
            Type::Array(
                Box::new(Type::Pointer(Box::new(Type::Pointer(Box::new(Type::I32))))),
                ArrayKind::Dynamic(DynamicKind::Dynamic),
            ),
        );

        test(
            b"  __data_loc \tconst\tvolatile int volatile signed*const _Atomic* \tconst\t [sizeof(struct foo)] \t[42] []foo",
            "foo",
            Type::Array(
            Box::new(Type::Array(
                Box::new(Type::Array(
                    Box::new(Type::Pointer(Box::new(Type::Pointer(Box::new(
                        Type::I32,
                    ))))),
                    ArrayKind::Fixed(Ok(42)),
                )),
                ArrayKind::Fixed(Err(Box::new(InterpError::CompileError(Box::new(CompileError::UnknownSize(Type::Struct("foo".into()))))))),
            ),
            ), ArrayKind::Dynamic(DynamicKind::Dynamic)));
    }
}
