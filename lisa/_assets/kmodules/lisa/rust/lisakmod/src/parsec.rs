/* SPDX-License-Identifier: GPL-2.0 */

use alloc::vec::Vec;
use core::{
    convert::Infallible,
    fmt,
    ops::{ControlFlow, FromResidual, Try},
};

#[derive(Debug, Clone)]
pub enum ParseResult<I, T, E> {
    Success { remainder: I, x: T },
    Failure { err: E },
    // Hard error, cannot be recovered from.
    Error { err: E },
}

impl<I, T, E> ParseResult<I, T, E> {
    #[inline]
    pub fn into_result(self) -> Result<T, crate::error::Error>
    where
        I: fmt::Debug,
        T: fmt::Debug,
        E: fmt::Display,
    {
        match self {
            ParseResult::Success { x, .. } => Ok(x),
            ParseResult::Failure { err } => Err(crate::error::error!("Parse failed: {err}")),
            ParseResult::Error { err } => Err(crate::error::error!("Parse errored: {err}")),
        }
    }

    #[inline]
    pub fn unwrap_success(self) -> T
    where
        I: fmt::Debug,
        T: fmt::Debug,
        E: fmt::Debug,
    {
        match self {
            ParseResult::Success { x, .. } => x,
            _ => panic!("Parse failed: {self:?}"),
        }
    }

    #[inline]
    pub fn from_result(input: I, res: Result<T, E>) -> ParseResult<I, T, E> {
        match res {
            Ok(x) => ParseResult::Success {
                x,
                remainder: input,
            },
            Err(err) => ParseResult::Failure { err },
        }
    }
}

impl<I, T, E> FromResidual for ParseResult<I, T, E> {
    #[inline]
    fn from_residual(residual: ParseResult<I, Infallible, E>) -> Self {
        match residual {
            ParseResult::Failure { err } => ParseResult::Failure { err },
            ParseResult::Error { err } => ParseResult::Error { err },
        }
    }
}

impl<I, T, E> Try for ParseResult<I, T, E> {
    type Output = (I, T);
    type Residual = ParseResult<I, Infallible, E>;

    fn from_output(output: Self::Output) -> Self {
        let (remainder, x) = output;
        ParseResult::Success { remainder, x }
    }
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            ParseResult::Success { x, remainder } => ControlFlow::Continue((remainder, x)),
            ParseResult::Failure { err } => ControlFlow::Break(ParseResult::Failure { err }),
            ParseResult::Error { err } => ControlFlow::Break(ParseResult::Error { err }),
        }
    }
}

pub trait Parser<I, T, E>
where
    I: Input,
    E: Error,
{
    fn parse(&mut self, input: I) -> ParseResult<I, T, E>;

    #[inline]
    fn then<T2, F, P>(mut self, mut f: F) -> impl Parser<I, T2, E>
    where
        Self: Sized,
        F: FnMut(T) -> P,
        P: Parser<I, T2, E>,
    {
        ClosureParser::new(move |input| {
            let (remainder, x) = self.parse(input)?;
            let mut then = f(x);
            then.parse(remainder)
        })
    }

    #[inline]
    fn or<F, P>(mut self, mut p: P) -> impl Parser<I, T, E>
    where
        Self: Sized,
        P: Parser<I, T, E>,
    {
        ClosureParser::new(move |mut input: I| {
            let saved = input.save_pos();
            let res = self.parse(input);
            match res {
                ParseResult::Success { remainder, x } => ParseResult::Success { remainder, x },
                ParseResult::Failure { .. } => p.parse(saved),
                ParseResult::Error { err } => ParseResult::Error { err },
            }
        })
    }

    fn map_cut<T2, F>(mut self, mut f: F) -> impl Parser<I, T2, E>
    where
        Self: Sized,
        F: FnMut(T) -> Result<T2, E>,
    {
        ClosureParser::new(move |input: I| {
            let (remainder, x) = self.parse(input)?;
            match f(x) {
                Ok(x) => ParseResult::Success { remainder, x },
                Err(err) => ParseResult::Error { err },
            }
        })
    }
}

impl<I, T, E, P> Parser<I, T, E> for &mut P
where
    I: Input,
    E: Error,
    P: Parser<I, T, E>,
{
    #[inline]
    fn parse(&mut self, input: I) -> ParseResult<I, T, E> {
        (*self).parse(input)
    }
}

pub struct ClosureParser<F>(F);
impl<F> ClosureParser<F> {
    #[inline]
    pub fn new(f: F) -> ClosureParser<F> {
        ClosureParser(f)
    }
}

impl<I, T, E, F> Parser<I, T, E> for ClosureParser<F>
where
    I: Input,
    E: Error,
    F: FnMut(I) -> ParseResult<I, T, E>,
{
    #[inline]
    fn parse(&mut self, input: I) -> ParseResult<I, T, E> {
        (self.0)(input)
    }
}

pub trait Error: fmt::Debug {
    fn from_msg(msg: &'static str) -> Self;
}

impl Error for &str {
    #[inline]
    fn from_msg(msg: &'static str) -> Self {
        msg
    }
}

impl Error for () {
    #[inline]
    fn from_msg(_: &'static str) -> Self {}
}

impl Error for crate::error::Error {
    #[inline]
    fn from_msg(msg: &'static str) -> Self {
        crate::error::error!("{msg}")
    }
}

pub trait Input: fmt::Debug {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
    fn save_pos(&mut self) -> Self;
    fn pos(&self) -> usize;

    // FIXME: find a better API that makes it more straightforward to use the "prev" input, rather
    // than making it easy to hit the subtle bug where one more item is consumed in the input.
    fn to_iter(&mut self) -> impl Iterator<Item = (Self, Self::Item)>
    where
        Self: Sized,
    {
        core::iter::from_fn(move || {
            let prev = self.save_pos();
            self.next().map(|item| (prev, item))
        })
    }
}

#[derive(Debug)]
pub struct BytesInput<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BytesInput<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> BytesInput<'a> {
        BytesInput { bytes, pos: 0 }
    }
}

impl<'a> Input for BytesInput<'a> {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.bytes.get(self.pos) {
            Some(b) => {
                self.pos += 1;
                Some(*b)
            }
            None => None,
        }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn save_pos(&mut self) -> Self {
        BytesInput {
            bytes: self.bytes,
            pos: self.pos,
        }
    }
}

#[inline]
pub fn any<I, E>() -> impl Parser<I, I::Item, E>
where
    I: Input,
    E: Error,
{
    ClosureParser::new(move |mut input: I| match input.next() {
        None => ParseResult::Failure {
            err: E::from_msg("Expected an item but reached end of input"),
        },
        Some(item) => ParseResult::Success {
            remainder: input,
            x: item,
        },
    })
}

#[inline]
pub fn tag<Tag, I, E>(tag: Tag) -> impl Parser<I, I::Item, E>
where
    I: Input<Item: PartialEq<Tag>>,
    E: Error,
{
    ClosureParser::new(move |input: I| {
        let (remainder, item) = any().parse(input)?;
        if item == tag {
            ParseResult::Success { remainder, x: item }
        } else {
            ParseResult::Failure {
                err: E::from_msg("Could not recognize expected tag"),
            }
        }
    })
}

#[inline]
pub fn not_tag<Tag, I, E>(tag: Tag) -> impl Parser<I, I::Item, E>
where
    I: Input<Item: PartialEq<Tag>>,
    E: Error,
{
    ClosureParser::new(move |input: I| {
        let (remainder, item) = any().parse(input)?;
        if item != tag {
            ParseResult::Success { remainder, x: item }
        } else {
            ParseResult::Failure {
                err: E::from_msg("Could not recognize expected tag"),
            }
        }
    })
}

#[inline]
pub fn tags<'a, Tag, TagI, I, E>(tags: TagI) -> impl Parser<I, (), E>
where
    Tag: 'a + Clone,
    TagI: IntoIterator<Item = &'a Tag> + Clone,
    I: Input<Item: PartialEq<Tag>>,
    E: Error,
{
    ClosureParser::new(move |mut input: I| {
        for _tag in tags.clone().into_iter().cloned() {
            match tag(_tag).parse(input) {
                ParseResult::Success { remainder, .. } => {
                    input = remainder;
                }
                ParseResult::Error { err } => return ParseResult::Error { err },
                ParseResult::Failure { err } => return ParseResult::Failure { err },
            }
        }

        ParseResult::Success {
            remainder: input,
            x: (),
        }
    })
}

#[inline]
pub fn many_min_max<I, T, E, Out, P>(
    min: Option<usize>,
    max: Option<usize>,
    mut parser: P,
) -> impl Parser<I, Out, E>
where
    I: Input,
    E: Error,
    P: Parser<I, T, E>,
    Out: FromIterator<T>,
{
    ClosureParser::new(move |mut input: I| {
        let mut len = 0;
        let mut saved = input.save_pos();
        let mut input = Some(input);
        let iter = core::iter::from_fn(|| {
            let mut _input: I = input.take().unwrap();
            saved = _input.save_pos();
            match parser.parse(_input) {
                ParseResult::Success { remainder, x } => {
                    input = Some(remainder);
                    len += 1;
                    Some(x)
                }
                _ => None,
            }
        });
        let out = iter.collect();
        if let Some(min) = min {
            if len < min {
                return ParseResult::Failure {
                    err: E::from_msg("Not enough items consumed"),
                };
            }
        }
        if let Some(max) = max {
            if len > max {
                return ParseResult::Failure {
                    err: E::from_msg("Too many items consumed"),
                };
            }
        }
        ParseResult::Success {
            remainder: saved,
            x: out,
        }
    })
}

#[inline]
pub fn many<I, T, E, Out, P>(parser: P) -> impl Parser<I, Out, E>
where
    I: Input,
    E: Error,
    P: Parser<I, T, E>,
    Out: FromIterator<T>,
{
    many_min_max(None, None, parser)
}

#[inline]
pub fn recognize<I, T, E, P>(mut parser: P) -> impl Parser<I, Vec<I::Item>, E>
where
    I: Input,
    E: Error,
    P: Parser<I, T, E>,
{
    ClosureParser::new(move |mut input: I| {
        let saved = input.save_pos();
        let (remainder, _) = parser.parse(input)?;

        let len = remainder.pos() - saved.pos();
        let mut out = Vec::with_capacity(len);
        let mut input = saved;
        for _ in 0..len {
            let (_input, item) = any().parse(input)?;
            input = _input;
            out.push(item);
        }
        ParseResult::Success {
            remainder: input,
            x: out,
        }
    })
}

#[inline]
pub fn cut<I, T, E, P>(mut parser: P) -> impl Parser<I, T, E>
where
    I: Input,
    E: Error,
    P: Parser<I, T, E>,
{
    ClosureParser::new(move |input: I| match parser.parse(input) {
        res @ (ParseResult::Success { .. } | ParseResult::Error { .. }) => res,
        ParseResult::Failure { err } => ParseResult::Error { err },
    })
}

#[inline]
pub fn or<I, T, E, P, Iter>(parsers: Iter) -> impl Parser<I, T, E>
where
    I: Input,
    E: Error,
    P: Parser<I, T, E>,
    for<'a> &'a Iter: IntoIterator<Item = &'a mut P>,
{
    ClosureParser::new(move |mut input: I| {
        for p in &parsers {
            match p.parse(input.save_pos()) {
                ParseResult::Success { remainder, x } => {
                    return ParseResult::Success { remainder, x };
                }
                ParseResult::Error { err } => return ParseResult::Error { err },
                _ => {}
            }
        }

        ParseResult::Failure {
            err: E::from_msg("No alternative matched"),
        }
    })
}

// #[inline]
// pub fn and<'a, I, T, E, P, Iter>(parsers: Iter) -> impl Parser<I, (), E>
// where
// I: Input,
// E: Error,
// P: 'a + Parser<I, T, E>,
// Iter: IntoIterator<Item = P> + Clone,
// {
// ClosureParser::new(move |mut input: I| {
// for p in parsers.clone() {
// match p.parse(input) {
// ParseResult::Success { remainder, .. } => {
// input = remainder;
// }
// ParseResult::Error { err } => return ParseResult::Error { err },
// ParseResult::Failure { err } => return ParseResult::Failure { err },
// }
// }

// ParseResult::Success {
// remainder: input,
// x: (),
// }
// })
// }

#[inline]
pub fn u64_decimal<I, E>() -> impl Parser<I, u64, E>
where
    I: Input<Item: Into<char>>,
    E: Error,
{
    // Since decimal number writing system works in "big endian" (most significant digit first), we
    // assign to it the highest possible power of ten we can represent with the output type, and at
    // the end we divide the result to scale it back to the real value.
    let pow_max = u64::MAX.ilog(10);
    let mag_max: u64 = 10u64.pow(pow_max);

    ClosureParser::new(move |mut input: I| {
        let mut found = false;
        let mut acc: u64 = 0;
        let mut mag = mag_max;
        for (prev, item) in input.to_iter() {
            let item: char = item.into();
            let item = match item {
                '0' => 0,
                '1' => 1,
                '2' => 2,
                '3' => 3,
                '4' => 4,
                '5' => 5,
                '6' => 6,
                '7' => 7,
                '8' => 8,
                '9' => 9,
                _ => {
                    if found {
                        return ParseResult::Success {
                            remainder: prev,
                            x: acc / mag,
                        };
                    } else {
                        return ParseResult::Failure {
                            err: E::from_msg("Expected a decimal number"),
                        };
                    }
                }
            };
            found = true;
            mag /= 10;
            acc += item * mag;
        }
        if found {
            ParseResult::Success {
                remainder: input,
                x: acc / mag,
            }
        } else {
            ParseResult::Failure {
                err: E::from_msg("Expected a digit but reached end of input"),
            }
        }
    })
}

#[inline]
pub fn whitespace<I, E>() -> impl Parser<I, (), E>
where
    I: Input<Item: Into<char>>,
    E: Error,
{
    ClosureParser::new(move |mut input: I| {
        for (prev, item) in input.to_iter() {
            let item: char = item.into();
            match item {
                ' ' | '\n' | '\t' => {}
                _ => {
                    return ParseResult::Success {
                        remainder: prev,
                        x: (),
                    };
                }
            }
        }
        ParseResult::Success {
            remainder: input,
            x: (),
        }
    })
}

#[inline]
pub fn eof<I, E>() -> impl Parser<I, (), E>
where
    I: Input,
    E: Error,
{
    ClosureParser::new(move |mut input: I| match input.next() {
        None => ParseResult::Success {
            remainder: input,
            x: (),
        },
        Some(_) => ParseResult::Failure {
            err: E::from_msg("Expected end of input but found more input"),
        },
    })
}
