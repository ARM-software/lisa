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

use core::cell::RefCell;
use std::rc::Rc;

use nom::{Finish as _, Parser};
use nom_locate::LocatedSpan;

use crate::{
    parser::{FromParseError, NomError},
    scratch::{OwnedScratchBox, ScratchAlloc},
};

pub type Span<'i, G> = LocatedSpan<
    &'i [u8],
    (
        &'i <G as PackratGrammar>::Ctx<'i>,
        Rc<RefCell<Vec<<G as PackratGrammar>::State<'i>>>>,
        Rc<ScratchAlloc>,
    ),
>;

#[derive(Clone)]
pub struct LocatedState<State> {
    pub state: State,
    pub pos: usize,
}

pub trait PackratGrammar {
    type Ctx<'i>: 'i;
    type State<'i>;
    type Error;

    #[inline]
    fn get_ctx<'iref, 'i: 'iref>(input: &'iref Span<'i, Self>) -> &'i Self::Ctx<'i> {
        input.extra.0
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn grammar_ctx<'i, E>(
    ) -> fn(Span<'i, Self>) -> nom::IResult<Span<'i, Self>, &'i Self::Ctx<'i>, E> {
        move |input: Span<'i, Self>| {
            let ctx = Self::get_ctx(&input);
            Ok((input, ctx))
        }
    }

    fn make_span<'i>(input: &'i [u8], ctx: &'i Self::Ctx<'i>) -> Span<'i, Self>
    where
        Self::State<'i>: Default + Clone,
        Self: Sized,
    {
        // Use len + 1 so that we can store a state for empty strings as well
        // (and for the last rule of the parse when we consumed the full input)
        let len = input.len() + 1;
        let vec = vec![Default::default(); len];
        let ctx = (
            ctx,
            Rc::new(RefCell::new(vec)),
            Rc::new(ScratchAlloc::new()),
        );
        LocatedSpan::new_extra(input, ctx)
    }

    #[inline]
    fn apply_rule<'i, 'p, O, E, P>(
        mut rule: P,
        input: &'i [u8],
        ctx: &'i Self::Ctx<'i>,
    ) -> Result<(&'i [u8], O), E>
    where
        E: FromParseError<&'i [u8], nom::error::VerboseError<&'i [u8]>>,
        P: 'p + Parser<Span<'i, Self>, O, NomError<E, nom::error::VerboseError<Span<'i, Self>>>>,
        <Self as PackratGrammar>::State<'i>: Default + Clone,
        Self: Sized,
        for<'a> Self::Ctx<'a>: 'a,
    {
        let span = Self::make_span(input, ctx);
        match rule.parse(span).finish() {
            Ok((remaining, x)) => Ok((*remaining.fragment(), x)),
            Err(err) => match err.data {
                None => {
                    let inner = nom::error::VerboseError {
                        errors: err
                            .inner
                            .errors
                            .into_iter()
                            .map(|(span, kind)| (*span.fragment(), kind))
                            .collect(),
                    };
                    Err(E::from_parse_error(input, &inner))
                }
                Some(err) => Err(err),
            },
        }
    }
}

#[derive(Default, Clone)]
pub enum PackratAction<'a, T> {
    // Keep that variant as the first, so that its discriminant is (probably) 0
    // so efficient zeroing can be used when initializing large amounts of
    // memory.
    #[default]
    Seed,

    // The size of this enum is critical for performance, as there will be one
    // value allocated in advance per input position and per left recursive
    // rule. This can lead to having to initialize large amounts of memory, so
    // it needs to be as small as possible. Therefore we put the data behind a
    // dynamic allocation.
    Succeed(OwnedScratchBox<'a, LocatedState<T>, Rc<ScratchAlloc>>),
    Fail,
}

macro_rules! __if_set_else {
    ({$($true:tt)*} {$($false:tt)*}) => {
        $($false)*
    };
    ({$($true:tt)*} {$($false:tt)*} $($_:tt)+) => {
        $($true)*
    };
}

pub(crate) use __if_set_else;

// Allow defining grammar production rules with most of the boilerplate
// removed and automatic context() added
macro_rules! grammar {
    (
        name: $vis:vis $grammar_name:ident,
        ctx: $ctx:ty,
        error: $grammar_error:ty,
        rules: {
            $( $(#[$rec:meta])? rule $name:ident $(<$($generics:tt $(: $bound:tt)?),*>)? ($($param:ident: $param_ty:ty),*) -> $ret:ty $body:block)*
        }) => {
        $vis struct $grammar_name ();

        // Create the state struct in a fresh scope so it will not
        // conflict with any other state structs. It also allows using "use"
        // without polluting the surrounding scope.
        const _: () = {
            use $crate::grammar::{PackratGrammar, PackratAction, Span};
            use ::nom::error::context;

            #[allow(non_camel_case_types)]
            #[derive(Default, Clone)]
            pub struct PackratState<'i> {
                $(
                    #[allow(dead_code)]
                    $name: $crate::grammar::__if_set_else! {
                        {
                            PackratAction<'i, $ret>
                        }
                        {()}
                        $($rec)?
                    },
                )*
                __internal_phantom_lifetime: std::marker::PhantomData<&'i ()>,
            }

            impl PackratGrammar for $grammar_name {
                // Using Rc<> allows cloning the LocatedSpan while sharing
                // the packrat state.
                type State<'i> = PackratState<'i>;
                type Error = $grammar_error;
                type Ctx<'i> = $ctx;
            }

            impl $grammar_name {
                $(
                    $vis fn $name<'i, 'ret, $($($generics $(: $bound)?,)*)? E>($($param: $param_ty),*) -> impl ::nom::Parser<Span<'i, $grammar_name>, $ret, E> + 'ret
                    where
                        E: 'ret
                        + ::nom::error::ParseError<Span<'i, $grammar_name>>
                        + ::nom::error::ContextError<Span<'i, $grammar_name>>
                        + ::nom::error::FromExternalError<Span<'i, $grammar_name>, $grammar_error>,
                        $($($generics: 'ret),*)?
                    {
                        // Wrap the body in a closure to avoid recursive type issues
                        // when a rule is recursive, and add a context for free.
                        //
                        // Also, this allows to implement packrat parsing
                        // modified to support left recursive grammar.
                        move |input: Span<'i, $grammar_name>| {
                            $crate::grammar::__if_set_else! {
                                {
                                    use std::{
                                        ops::Deref as _,
                                    };

                                    let parser = move |input| $body.parse(input);
                                    let orig_pos = input.location_offset();
                                    let mut packrat = input.extra.1.deref().borrow_mut();
                                    let state = &mut packrat[orig_pos].$name;

                                    match state {
                                        PackratAction::Seed => {
                                            // Will make any recursive invocation of the rule fail
                                            *state = PackratAction::Fail;
                                            drop(packrat);

                                            // Parse once, with no recursive call allowed to
                                            // succeed. This provides the seed result that
                                            // will be reinjected at the next attempt.
                                            let mut res = context(concat!(stringify!($name), " (seed)"), parser).parse(input.clone())?;

                                            loop {
                                                let (i, seed) = &res;
                                                let pos = i.location_offset();

                                                {
                                                    // Re-borrow the RefCell so that it does not
                                                    // appear as borrowed when running the rule.
                                                    let mut packrat = input.extra.1.deref().borrow_mut();
                                                    let state = &mut packrat[orig_pos].$name;
                                                    // Set the seed, which will make any
                                                    // recursive call to that rule succeed with
                                                    // that result.
                                                    *state = PackratAction::Succeed(
                                                        $crate::scratch::OwnedScratchBox::new_in(
                                                            $crate::grammar::LocatedState {
                                                                state: seed.clone(),
                                                                pos,
                                                            },
                                                            ::std::rc::Rc::clone(&input.extra.2),
                                                        )
                                                    );
                                                }

                                                // Parse again with the seed in place, so
                                                // that recursive call succeed and we can
                                                // try to match what comes after
                                                let res2 = context(concat!(stringify!($name), " (reparse)"), parser).parse(input.clone())?;

                                                let (i2, _x2) = &res2;
                                                let pos2 = i2.location_offset();

                                                // If we consumed the whole input, we have
                                                // the best match possible.
                                                if i2.fragment().len() == 0 {
                                                    return Ok(res2)
                                                } else if pos >= pos2 {
                                                    return Ok(res)
                                                // If this resulted in a longer match, take
                                                // it and loop again. Otherwise, we found
                                                // the best match.
                                                } else {
                                                    res = res2;
                                                }
                                            }
                                        }
                                        PackratAction::Succeed(data) => {
                                            let val = (&data.state).clone();
                                            let pos = data.pos;
                                            drop(packrat);
                                            let (input, _) = ::nom::bytes::complete::take(pos - input.location_offset()).parse(input)?;
                                            context(concat!(stringify!($name), " (pre-parsed)"), success(val))(input)
                                        }
                                        PackratAction::Fail => {
                                            drop(packrat);
                                            context(concat!(stringify!($name), " (seed recursion block)"), fail)(input)
                                        }
                                    }
                                }
                                {$body.parse(input)}
                                $($rec)?
                            }

                        }
                    }

                )*
            }
        };
    }
}
pub(crate) use grammar;

// Allow importing the macro like any other item inside this crate

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::parser::{lexeme, parenthesized, tests::test_parser, to_str};

    #[test]
    fn packrat_test() {
        fn test(input: &[u8], expected: Ast) {
            let parser = TestGrammar::starting_symbol();
            let input = TestGrammar::make_span(input, &());
            test_parser(expected, input, parser);
        }

        #[derive(Debug, Clone, PartialEq)]
        enum Ast {
            Var(std::string::String),
            Add(Box<Ast>, Box<Ast>),
            Sub(Box<Ast>, Box<Ast>),
            Mul(Box<Ast>, Box<Ast>),
            Div(Box<Ast>, Box<Ast>),
        }

        use std::string::ToString;

        use nom::{
            branch::alt,
            bytes::complete::tag,
            character::complete::alpha1,
            combinator::{fail, recognize, success},
            multi::many1,
            sequence::separated_pair,
        };

        grammar! {
            name: TestGrammar,
            ctx: (),
            error: (),
            rules: {
                #[leftrec] rule literal() -> Ast {
                    lexeme(recognize(many1(alpha1))).map(|id: LocatedSpan<&[u8], _>| Ast::Var(to_str(id.fragment())))
                }

                #[leftrec] rule expr() -> Ast {
                    lexeme(alt((
                        context("add",
                            separated_pair(
                                Self::expr(),
                                tag("+"),
                                Self::term(),
                            ).map(|(op1, op2)| Ast::Add(Box::new(op1), Box::new(op2)))
                        ),
                        context("sub",
                            separated_pair(
                                Self::expr(),
                                tag("-"),
                                Self::term(),
                            ).map(|(op1, op2)| Ast::Sub(Box::new(op1), Box::new(op2)))
                        ),
                        Self::term(),
                    )))
                }

                #[leftrec] rule term() -> Ast {
                    lexeme(alt((
                        context("mul",
                            separated_pair(
                                Self::term(),
                                tag("*"),
                                Self::factor(),
                            ).map(|(op1, op2)| Ast::Mul(Box::new(op1), Box::new(op2)))
                        ),
                        context("div",
                            separated_pair(
                                Self::term(),
                                tag("/"),
                                Self::factor(),
                            ).map(|(op1, op2)| Ast::Div(Box::new(op1), Box::new(op2)))
                        ),
                        Self::factor(),
                    )))
                }


                #[leftrec] rule factor() -> Ast {
                    lexeme(alt((
                        Self::literal(),
                        context("paren",
                            parenthesized(Self::expr())
                        ),
                    )))
                }

                #[leftrec] rule starting_symbol() -> Ast {
                    Self::expr()
                }
            }
        }

        test(b" a ", Ast::Var("a".to_string()));
        test(b"((( a)) ) ", Ast::Var("a".to_string()));
        test(
            b"a+b",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Var("b".to_string())),
            ),
        );
        test(
            b"a + b",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Var("b".to_string())),
            ),
        );
        test(
            b"a + b ",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Var("b".to_string())),
            ),
        );
        test(
            b" a + b ",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Var("b".to_string())),
            ),
        );
        test(
            b" a + b",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Var("b".to_string())),
            ),
        );

        test(
            b" a + b+c ",
            Ast::Add(
                Box::new(Ast::Add(
                    Box::new(Ast::Var("a".to_string())),
                    Box::new(Ast::Var("b".to_string())),
                )),
                Box::new(Ast::Var("c".to_string())),
            ),
        );

        test(
            b"a+(b+c)",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Add(
                    Box::new(Ast::Var("b".to_string())),
                    Box::new(Ast::Var("c".to_string())),
                )),
            ),
        );

        test(
            b"(a+b)+c",
            Ast::Add(
                Box::new(Ast::Add(
                    Box::new(Ast::Var("a".to_string())),
                    Box::new(Ast::Var("b".to_string())),
                )),
                Box::new(Ast::Var("c".to_string())),
            ),
        );
        test(
            b"(a+b+c)",
            Ast::Add(
                Box::new(Ast::Add(
                    Box::new(Ast::Var("a".to_string())),
                    Box::new(Ast::Var("b".to_string())),
                )),
                Box::new(Ast::Var("c".to_string())),
            ),
        );
        test(
            b"(a+b*c)",
            Ast::Add(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Mul(
                    Box::new(Ast::Var("b".to_string())),
                    Box::new(Ast::Var("c".to_string())),
                )),
            ),
        );
        test(
            b"a*b+c*d",
            Ast::Add(
                Box::new(Ast::Mul(
                    Box::new(Ast::Var("a".to_string())),
                    Box::new(Ast::Var("b".to_string())),
                )),
                Box::new(Ast::Mul(
                    Box::new(Ast::Var("c".to_string())),
                    Box::new(Ast::Var("d".to_string())),
                )),
            ),
        );

        test(
            b"(a*b)/(c*d)",
            Ast::Div(
                Box::new(Ast::Mul(
                    Box::new(Ast::Var("a".to_string())),
                    Box::new(Ast::Var("b".to_string())),
                )),
                Box::new(Ast::Mul(
                    Box::new(Ast::Var("c".to_string())),
                    Box::new(Ast::Var("d".to_string())),
                )),
            ),
        );
        test(
            b"(a*(b/(c*d)))",
            Ast::Mul(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Div(
                    Box::new(Ast::Var("b".to_string())),
                    Box::new(Ast::Mul(
                        Box::new(Ast::Var("c".to_string())),
                        Box::new(Ast::Var("d".to_string())),
                    )),
                )),
            ),
        );

        test(
            b"a*b/c*d",
            Ast::Mul(
                Box::new(Ast::Div(
                    Box::new(Ast::Mul(
                        Box::new(Ast::Var("a".to_string())),
                        Box::new(Ast::Var("b".to_string())),
                    )),
                    Box::new(Ast::Var("c".to_string())),
                )),
                Box::new(Ast::Var("d".to_string())),
            ),
        );

        test(
            b"a-b/c*d",
            Ast::Sub(
                Box::new(Ast::Var("a".to_string())),
                Box::new(Ast::Mul(
                    Box::new(Ast::Div(
                        Box::new(Ast::Var("b".to_string())),
                        Box::new(Ast::Var("c".to_string())),
                    )),
                    Box::new(Ast::Var("d".to_string())),
                )),
            ),
        );
    }

    #[test]
    fn packrat_recursive_test() {
        fn test(input: &[u8], expected: Ast) {
            let parser = TestGrammar::starting_symbol();
            let input = TestGrammar::make_span(input, &());
            test_parser(expected, input, parser);
        }

        #[derive(Debug, Clone, PartialEq)]
        enum Ast {
            Constant(char),
            CommaExpr(Box<Ast>, Box<Ast>),
        }

        use nom::{
            branch::alt,
            character::complete::char,
            combinator::{fail, success},
            sequence::{delimited, separated_pair},
        };

        grammar! {
            name: TestGrammar,
            ctx: (),
            error: (),
            rules: {
                #[leftrec] rule primary_expr() -> Ast {
                    lexeme(
                        alt((
                            alt((
                                char('1'),
                                char('2'),
                            )).map(Ast::Constant),
                            delimited(
                                lexeme(char('(')),
                                Self::expr(),
                                lexeme(char(')')),
                            )
                        ))
                    )
                }

                #[leftrec] rule comma_expr() -> Ast {
                    lexeme(
                        alt((
                            separated_pair(
                                Self::primary_expr(),
                                lexeme(char(',')),
                                Self::comma_expr(),
                            ).map(|(x, y)| Ast::CommaExpr(Box::new(x), Box::new(y))),
                            Self::primary_expr(),
                        ))
                    )
                }

                #[leftrec] rule expr() -> Ast {
                    alt((
                        Self::comma_expr(),
                        Self::primary_expr(),
                    ))
                }
                #[leftrec] rule starting_symbol() -> Ast {
                    Self::expr()
                }
            }
        }

        use Ast::*;

        test(b"1", Constant('1'));
        test(b"2", Constant('2'));
        test(
            b"(1,2)",
            CommaExpr(Box::new(Constant('1')), Box::new(Constant('2'))),
        );
    }
}
