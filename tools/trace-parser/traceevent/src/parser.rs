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
    fmt::{Debug, Display, Formatter},
    ops::Range,
};
use std::string::String as StdString;

use nom::{
    bytes::complete::is_a,
    character::complete::{char, multispace0},
    combinator::all_consuming,
    error::{ContextError, ErrorKind, FromExternalError, ParseError},
    sequence::delimited,
    Finish as _, Parser,
};

pub trait FromParseError<I, E>: Sized {
    fn from_parse_error(input: I, err: &E) -> Self;
}

impl<E, I> FromParseError<I, E> for () {
    fn from_parse_error(_input: I, _err: &E) -> Self {}
}

#[derive(Clone, PartialEq)]
pub struct VerboseParseError {
    input: String,
    errors: Vec<(Range<usize>, nom::error::VerboseErrorKind)>,
}

impl VerboseParseError {
    pub fn from_input<I: AsRef<[u8]>>(input: I) -> Self {
        let input = input.as_ref();
        let input = StdString::from_utf8_lossy(input).into_owned();
        VerboseParseError {
            input,
            errors: vec![],
        }
    }
    pub fn new<I: AsRef<[u8]>, I2: AsRef<[u8]>>(
        input: I,
        err: &nom::error::VerboseError<I2>,
    ) -> Self {
        match core::str::from_utf8(input.as_ref()) {
            Err(err) => VerboseParseError {
                input: format!("<utf-8 decoding error: {err}>"),
                errors: vec![],
            },
            Ok(input) => {
                let errors = err
                    .errors
                    .iter()
                    .map(|(s, k)| {
                        let s = s.as_ref();
                        let offset = s.as_ptr() as usize - input.as_ptr() as usize;
                        let size = s.len();
                        let range = offset..(offset + size);
                        (range, k.clone())
                    })
                    .collect();

                VerboseParseError {
                    input: input.into(),
                    errors,
                }
            }
        }
    }
}

impl Eq for VerboseParseError {}

impl Debug for VerboseParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        write!(f, "VerboseParseError {{{self}}}",)
    }
}

impl Display for VerboseParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        let input = self.input.as_str();
        let mut seen_context = false;
        let inner = nom::error::VerboseError {
            errors: self
                .errors
                .iter()
                // Preserve the leaf-most levels that don't have a
                // context, but after the first context is
                // encountered, display only levels with a context.
                // This makes the path much easier to follow if all
                // relevant levels are annotated correctly.
                .filter(|(_, kind)| match kind {
                    nom::error::VerboseErrorKind::Context(..) => {
                        seen_context = true;
                        true
                    }
                    _ => !seen_context,
                })
                .map(|(range, k)| (&input[range.clone()], k.clone()))
                .collect(),
        };
        write!(
            f,
            "Error while parsing:\n{}\n{}\n",
            input,
            &nom::error::convert_error(input, inner)
        )?;
        Ok(())
    }
}

/// Tie together a nom error and some user-defined data.
#[derive(Debug)]
pub struct NomError<T, E> {
    /// User-defined data.
    pub data: Option<T>,
    /// nom error, such as [nom::error::Error]
    pub inner: E,
}

impl<T, E> NomError<T, E> {
    #[inline]
    fn from_inner(inner: E) -> Self {
        NomError { data: None, inner }
    }

    pub fn into_external<I, O, T2, F, E2>(self, input: I, mut convert: F) -> nom::IResult<I, O, E2>
    where
        F: FnMut(T) -> T2,
        E: ParseError<I>,
        E2: ParseError<I> + FromExternalError<I, T2>,
    {
        match self.data {
            Some(data) => error(input, convert(data)),
            None => Err(nom::Err::Error(E2::from_error_kind(
                input,
                nom::error::ErrorKind::Fail,
            ))),
        }
    }
}

impl<I, T, E> ParseError<I> for NomError<T, E>
where
    I: Clone,
    E: ParseError<I>,
    T: FromParseError<I, E>,
{
    #[inline]
    fn from_error_kind(input: I, kind: ErrorKind) -> Self {
        NomError::from_inner(E::from_error_kind(input, kind))
    }

    #[inline]
    fn from_char(input: I, c: char) -> Self {
        NomError::from_inner(E::from_char(input, c))
    }

    #[inline]
    fn append(input: I, kind: ErrorKind, other: Self) -> Self {
        NomError {
            inner: E::append(input.clone(), kind, other.inner),
            data: other.data,
        }
    }

    #[inline]
    fn or(self, other: Self) -> Self {
        NomError {
            data: other.data,
            inner: self.inner.or(other.inner),
        }
    }
}

impl<I, T, E> FromExternalError<I, T> for NomError<T, E>
where
    E: ParseError<I>,
{
    #[inline]
    fn from_external_error(input: I, kind: ErrorKind, e: T) -> Self {
        NomError {
            data: Some(e),
            inner: E::from_error_kind(input, kind),
        }
    }
}

impl<I, T, E> ContextError<I> for NomError<T, E>
where
    E: ContextError<I>,
{
    #[inline]
    fn add_context(input: I, ctx: &'static str, other: Self) -> Self {
        NomError {
            data: other.data,
            inner: E::add_context(input, ctx, other.inner),
        }
    }
}

//////////////
// Conversions
//////////////

pub fn to_str(s: &[u8]) -> StdString {
    StdString::from_utf8_lossy(s).to_string()
}

//////////////////////
// Parsers
//////////////////////

pub fn hex_u64<I, E>(input: I) -> nom::IResult<I, u64, E>
where
    E: ParseError<I>,
    I: Clone,
    I: nom::AsBytes + nom::InputIter + nom::InputTakeAtPosition<Item = u8>,
{
    is_a(&b"0123456789abcdefABCDEF"[..])
        .map(|x: I| {
            x.as_bytes()
                .iter()
                .rev()
                .enumerate()
                .map(|(k, v)| -> u64 {
                    let v: char = (*v).into();
                    let v: u64 = v.to_digit(16).unwrap_or(0).into();
                    v << (k * 4)
                })
                .sum()
        })
        .parse(input)
}

//////////////////////
// Generic combinators
//////////////////////

pub trait NomParserExt<I, O, E, NE>: nom::Parser<I, O, NomError<E, NE>> {
    #[inline]
    fn parse_finish(&mut self, input: I) -> Result<O, E>
    where
        I: nom::InputLength + Clone + Debug,
        NE: Debug + ParseError<I>,
        E: Debug + FromParseError<I, NE>,
    {
        let mut parser = all_consuming(|input| self.parse(input));
        match parser.parse(input.clone()).finish() {
            Err(err) => match err.data {
                None => Err(E::from_parse_error(input, &err.inner)),
                Some(err) => Err(err),
            },
            Ok((_, x)) => Ok(x),
        }
    }
}

impl<I, O, E, NE, P> NomParserExt<I, O, E, NE> for P where P: nom::Parser<I, O, NomError<E, NE>> {}

#[allow(unused)]
pub fn print<I, O, E, P>(name: &'static str, mut inner: P) -> impl nom::Parser<I, O, E>
where
    E: ParseError<I>,
    P: nom::Parser<I, O, E>,
    I: core::convert::AsRef<[u8]> + Clone,
    O: Debug,
{
    move |input: I| {
        let (i, x) = inner.parse(input.clone())?;
        println!(
            "{name} input={:?} out={x:?} new_input={:?}",
            to_str(input.as_ref()),
            to_str(i.as_ref())
        );
        Ok((i, x))
    }
}

pub fn lexeme<I, O, E, P>(inner: P) -> impl nom::Parser<I, O, E>
where
    E: ParseError<I>,
    P: nom::Parser<I, O, E>,
    I: Clone + nom::InputLength + nom::InputIter + nom::InputTake + nom::InputTakeAtPosition,
    <I as nom::InputIter>::Item: Clone + nom::AsChar,
    <I as nom::InputTakeAtPosition>::Item: Clone + nom::AsChar,
{
    delimited(multispace0, inner, multispace0)
}

pub fn parenthesized<I, O, E, P>(parser: P) -> impl nom::Parser<I, O, E>
where
    P: nom::Parser<I, O, E>,
    E: ParseError<I>,
    I: nom::Slice<std::ops::RangeFrom<usize>>
        + nom::InputIter
        + Clone
        + nom::InputLength
        + nom::InputTake
        + nom::InputTakeAtPosition,
    <I as nom::InputIter>::Item: Clone + nom::AsChar,
    <I as nom::InputTakeAtPosition>::Item: Clone + nom::AsChar,
{
    delimited(lexeme(char('(')), parser, lexeme(char(')')))
}

pub fn map_res_cut<I: Clone, O1, O2, E: FromExternalError<I, E2>, E2, F, G>(
    mut parser: F,
    mut f: G,
) -> impl nom::Parser<I, O2, E>
where
    F: Parser<I, O1, E>,
    G: FnMut(O1) -> Result<O2, E2>,
{
    move |input: I| {
        let i = input.clone();
        let (input, x) = parser.parse(input)?;
        match f(x) {
            Ok(x) => Ok((input, x)),
            Err(err) => Err(nom::Err::Failure(E::from_external_error(
                i,
                ErrorKind::MapRes,
                err,
            ))),
        }
    }
}

// // Not available in nom 7 but maybe will be there in nom 8:
// // https://github.com/rust-bakery/nom/issues/1422
// pub fn map_err<P, F, I, O, E, E2, MappedE>(mut parser: P, f: F) -> impl nom::Parser<I, O, E2>
// where
//     P: nom::Parser<I, O, E>,
//     E: ParseError<I>,
//     E2: ParseError<I> + FromExternalError<I, MappedE>,
//     F: Fn(E) -> MappedE,
//     I: Clone,
// {
//     move |input: I| match parser.parse(input.clone()) {
//         Err(nom::Err::Error(e)) => Err(nom::Err::Error(E2::from_external_error(
//             input,
//             ErrorKind::Fail,
//             f(e),
//         ))),
//         Err(nom::Err::Failure(e)) => Err(nom::Err::Failure(E2::from_external_error(
//             input,
//             ErrorKind::Fail,
//             f(e),
//         ))),
//         Err(nom::Err::Incomplete(x)) => Err(nom::Err::Incomplete(x)),
//         Ok(x) => Ok(x),
//     }
// }

pub fn success_with<F, I, O, E>(mut f: F) -> impl FnMut(I) -> nom::IResult<I, O, E>
where
    F: FnMut() -> O,
    E: ParseError<I>,
{
    move |input: I| Ok((input, f()))
}

#[inline]
pub fn error<I, O, E, E2>(input: I, err: E) -> nom::IResult<I, O, E2>
where
    E2: FromExternalError<I, E>,
{
    Err(nom::Err::Error(E2::from_external_error(
        input,
        ErrorKind::Fail,
        err,
    )))
}

#[inline]
pub fn failure<I, O, E, E2>(input: I, err: E) -> nom::IResult<I, O, E2>
where
    E2: FromExternalError<I, E>,
{
    Err(nom::Err::Failure(E2::from_external_error(
        input,
        ErrorKind::Fail,
        err,
    )))
}

// pub fn null_terminated_str_parser<'a, E>() -> impl nom::Parser<&'a [u8], &'a [u8], E>
// where
//     E: ParseError<&'a [u8]>,
// {
//     terminated(
//         take_until(&[0][..]),
//         // Consume the null terminator
//         tag([0]),
//     )
// }

#[cfg(test)]
pub(crate) mod tests {
    use nom::{error::VerboseError, Finish as _};

    use super::*;

    pub trait DisplayErr {
        fn display_err(&self) -> StdString;
    }
    pub trait DisplayErrViaDisplay {}

    impl DisplayErrViaDisplay for crate::cparser::CParseError {}
    impl DisplayErrViaDisplay for crate::header::HeaderError {}
    impl DisplayErrViaDisplay for crate::print::PrintFmtError {}

    impl<T> DisplayErr for T
    where
        T: DisplayErrViaDisplay + Display,
    {
        fn display_err(&self) -> StdString {
            format!("{}", self)
        }
    }

    impl DisplayErr for () {
        fn display_err(&self) -> StdString {
            "".into()
        }
    }

    // Work-around this issue:
    // https://github.com/rust-bakery/nom/issues/1619
    // This function _must_ preserve the address of buf, as
    // nom::error::convert_error() relies on VerboseError input stack to be
    // pointer into the overall input. Otherwise, pointer arithmetic will
    // make no sense and it will either display non-sensical substrings or
    // panic.
    pub fn zero_copy_to_str(buf: &[u8]) -> &str {
        std::str::from_utf8(buf).unwrap()
    }

    pub fn run_parser<I, O, T, P>(input: I, parser: P) -> O
    where
        O: Debug + PartialEq,
        P: Parser<I, O, NomError<T, VerboseError<I>>>,
        I: nom::AsBytes + nom::InputLength + Clone,
        T: DisplayErr + FromParseError<I, nom::error::VerboseError<I>>,
    {
        let mut parser = all_consuming(parser);
        let parsed = parser.parse(input.clone()).finish();
        let input = zero_copy_to_str(input.as_bytes());
        match parsed {
            Ok((_, parsed)) => parsed,
            Err(err) => {
                // Convert input from &[u8] to &str so convert_error() can
                // display it.
                let mut seen_context = false;
                let inner = VerboseError {
                    errors: err
                        .inner
                        .errors
                        .iter()
                        // Preserve the leaf-most levels that don't have a
                        // context, but after the first context is
                        // encountered, display only levels with a context.
                        // This makes the path much easier to follow if all
                        // relevant levels are annotated correctly.
                        .filter(|(_, kind)| match kind {
                            nom::error::VerboseErrorKind::Context(..) => {
                                seen_context = true;
                                true
                            }
                            _ => !seen_context,
                        })
                        .map(|(s, err)| (zero_copy_to_str(s.as_bytes()), err.clone()))
                        .collect(),
                };
                let loc = nom::error::convert_error(input, inner);
                let err_data = match err.data {
                    Some(data) => data.display_err(),
                    None => "<unknown parse error>".into(),
                };
                panic!("Could not parse {input:?}: {err_data} :\n{loc}")
            }
        }
    }

    pub fn test_parser<I, O, T, P>(expected: O, input: I, parser: P)
    where
        O: Debug + PartialEq,
        T: DisplayErr + FromParseError<I, nom::error::VerboseError<I>>,
        P: Parser<I, O, NomError<T, VerboseError<I>>>,
        I: nom::AsBytes + nom::InputLength + Clone,
    {
        let parsed = run_parser(input.clone(), parser);

        let input = zero_copy_to_str(input.as_bytes());
        assert_eq!(parsed, expected, "while parsing: {input:?}");
    }
}
