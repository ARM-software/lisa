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

//! `C` language interpreter.
//!
//! The interpreter implemented in this module is used for:
//! 1. Simplifying and validating field types when they involve constant expressions (e.g. array
//!    size)
//! 2. Interpreting the printk-style format string arguments used to pretty-print ftrace events.
//!
//! # Limitations
//!
//! * The interpreter only supports expressions. Statements such as loops etc are not supported.
//! * Limited pointer arithmetic: the C language support for arrays based on pointer arithmetic
//!   makes writing a symbolic interpreter challenging. This interpreter assumes the code will no
//!   play tricks such as converting a 1D array into a 3D array an just lookup item 0.
//! * Limited pointer cast support: ISO C is actually quite conservative in the casts that do not
//!   lead to undefined behavior. We take a similar approach. All pointer casts are simplified
//!   symbolically and implementation-defined behavior might not match what a typical C compiler
//!   might give (e.g. target endianness may not be observed as expected when illegally casting
//!   pointers between incompatible integer types).
//!

use core::{fmt, num::Wrapping, ops::Deref};
use std::{string::String as StdString, sync::Arc};

use bytemuck::cast_slice;
use thiserror::Error;

use crate::{
    array::Array,
    buffer::BufferError,
    cparser,
    cparser::{ArrayKind, Expr, ExtensionMacroKind, ParseEnv, Type},
    error::convert_err_impl,
    header::{Abi, Address, Endianness, FileSize, Header, Identifier, LongSize, Signedness},
    print::{PrintError, PrintFmtError},
    scratch::{OwnedScratchBox, ScratchAlloc, ScratchBox},
    str::Str,
};

/// Errors while compiling.
///
/// See also: [EvalError] and [InterpError]
#[derive(Error, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum CompileError {
    #[error("Cannot this handle expression in its context: {0:?}")]
    ExprNotHandled(Expr),

    #[error("Cannot dereference an expression of type {0:?}: {1:?}")]
    CannotDeref(Type, Expr),

    #[error("Type is not an array: {0:?}")]
    NotAnArray(Type),

    #[error("Type not supported as array item: {0:?}")]
    InvalidArrayItem(Type),

    #[error("Size of this type is unknown: {0:?}")]
    UnknownSize(Type),

    #[error("Non arithmetic operand used with arithmetic operator")]
    NonArithmeticOperand(Type),

    #[error("Mismatching types in operands of {0:?}: {1:?} and {2:?}")]
    MismatchingOperandType(Expr, Type, Type),

    #[error("Cannot cast between incompatible pointer types: {0:?} => {1:?}")]
    IncompatiblePointerCast(Type, Type),

    #[error("The field \"{0}\" does not exist")]
    UnknownField(String),

    #[error("Values of this type cannot be decoded from a buffer: {0:?}")]
    NonDecodableType(Type),
}

/// Errors while evaluating an expression.
///
/// See also: [CompileError] and [InterpError]
#[derive(Error, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum EvalError {
    #[error("Illegal type of value: {}", match .0 {
        Some(val) => val.to_string(),
        None => "<no value>".into()
    })]
    IllegalType(Option<Value<'static>>),

    #[error("Cannot convert this value to a signed as it is too big: {0}")]
    CannotConvertToSigned(u64),

    #[error("Attempted to index a scalar value: {}", match .0 {
        Some(val) => val.to_string(),
        None => "<no value>".into()
    })]
    CannotIndexScalar(Option<Value<'static>>),

    #[error("Array index out of bonds: {0}")]
    OutOfBondIndex(usize),

    #[error("Could not dereference address: {0}")]
    CannotDeref(Address),

    #[error("Event data not available")]
    NoEventData,

    #[error("No header available")]
    NoHeader,

    #[error("Error while evaluating extension macro call \"{}\": {}", .call, .error)]
    ExtensionMacroError { call: StdString, error: StdString },

    #[error("Error while decoding buffer: {0}")]
    BufferError(Box<BufferError>),

    #[error("Error while parsing a vbin buffer format: {0}")]
    PrintFmtError(Box<PrintFmtError>),

    #[error("Error while evaluating a vbin buffer: {0}")]
    PrintError(Box<PrintError>),
}

convert_err_impl!(BufferError, BufferError, EvalError);
convert_err_impl!(PrintFmtError, PrintFmtError, EvalError);
convert_err_impl!(PrintError, PrintError, EvalError);

/// Generic interpreter-related error type.
///
/// See also: [CompileError] and [EvalError]
#[derive(Error, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum InterpError {
    #[error("Could not compile: {0}")]
    CompileError(Box<CompileError>),
    #[error("Could not evaluate: {0}")]
    EvalError(Box<EvalError>),
}
convert_err_impl!(EvalError, EvalError, InterpError);
convert_err_impl!(CompileError, CompileError, InterpError);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SockAddrFamily {
    Ipv4,
    Ipv6,
}

impl SockAddrFamily {
    #[inline]
    fn from_raw(code: u16) -> Result<Self, BufferError> {
        match code {
            2 => Ok(SockAddrFamily::Ipv4),
            10 => Ok(SockAddrFamily::Ipv6),
            _ => Err(BufferError::UnknownSockAddrFamily(code)),
        }
    }
}

/// Kind of socket address.
///
/// See also [SockAddr].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SockAddrKind {
    Full,
    Ipv4AddrOnly,
    Ipv6AddrOnly,
}

/// In-kernel socket address value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SockAddr<'a> {
    family: SockAddrFamily,
    kind: SockAddrKind,
    endianness: Endianness,
    data: &'a [u8],
}

#[derive(thiserror::Error, Debug, PartialEq, Eq, Clone)]
#[non_exhaustive]
pub enum SockAddrError {
    #[error("Could not convert value")]
    CannotConvert,
}

macro_rules! get_array {
    ($slice:expr, $len:expr) => {{
        let slice: &[u8] = $slice;
        let slice = slice.get(..$len).ok_or(SockAddrError::CannotConvert)?;
        let arr: [u8; $len] = slice.try_into().map_err(|_| SockAddrError::CannotConvert)?;
        arr
    }};
}

impl<'a> SockAddr<'a> {
    /// Create a [SockAddr] from the in-kernel byte encoding.
    #[inline]
    pub fn from_bytes(
        data: &'a [u8],
        endianness: Endianness,
        kind: SockAddrKind,
    ) -> Result<Self, BufferError> {
        let family = match kind {
            SockAddrKind::Full => {
                let (_data, family) = endianness
                    .parse_u16(data)
                    .map_err(|_| BufferError::SockAddrTooSmall)?;
                SockAddrFamily::from_raw(family)
            }
            SockAddrKind::Ipv4AddrOnly => Ok(SockAddrFamily::Ipv4),
            SockAddrKind::Ipv6AddrOnly => Ok(SockAddrFamily::Ipv6),
        }?;

        Ok(SockAddr {
            family,
            kind,
            data,
            endianness,
        })
    }

    /// Attempt to convert `self` to a [std::net::SocketAddr].
    ///
    /// This could fail if the socket address is not a full IPv4 or IPv6 socket address, including
    /// port information. For example, a Unix domain socket will not be converted successfully.
    pub fn to_socketaddr(&self) -> Result<std::net::SocketAddr, SockAddrError> {
        match (&self.kind, &self.family) {
            (SockAddrKind::Full, SockAddrFamily::Ipv4) => {
                let port = u16::from_be_bytes(get_array!(&self.data[2..], 2));

                // The kernel structs use network endianness but the user
                // might pass a little endian buffer and ask for that
                // explicitly.
                let (_, addr) = self
                    .endianness
                    .parse_u32(&self.data[4..])
                    .map_err(|_| SockAddrError::CannotConvert)?;

                Ok(std::net::SocketAddr::V4(std::net::SocketAddrV4::new(
                    addr.into(),
                    port,
                )))
            }
            (SockAddrKind::Full, SockAddrFamily::Ipv6) => {
                let port = u16::from_be_bytes(get_array!(&self.data[2..], 2));
                let flowinfo = u32::from_be_bytes(get_array!(&self.data[4..], 4));
                let addr = u128::from_be_bytes(get_array!(&self.data[8..], 16));
                let (_, scope_id) = self
                    .endianness
                    .parse_u32(&self.data[24..])
                    .map_err(|_| SockAddrError::CannotConvert)?;

                Ok(std::net::SocketAddr::V6(std::net::SocketAddrV6::new(
                    addr.into(),
                    port,
                    flowinfo,
                    scope_id,
                )))
            }
            _ => Err(SockAddrError::CannotConvert),
        }
    }

    pub fn to_ipaddr(&self) -> Result<std::net::IpAddr, SockAddrError> {
        match self.to_socketaddr() {
            Ok(sockaddr) => Ok(sockaddr.ip()),
            _ => match (&self.kind, &self.family) {
                (SockAddrKind::Ipv4AddrOnly, SockAddrFamily::Ipv4) => {
                    // The kernel structs use network endianness but the user
                    // might pass a little endian buffer and ask for that
                    // explicitly.
                    let (_, addr) = self
                        .endianness
                        .parse_u32(self.data)
                        .map_err(|_| SockAddrError::CannotConvert)?;
                    let addr: std::net::Ipv4Addr = addr.into();
                    Ok(addr.into())
                }

                (SockAddrKind::Ipv6AddrOnly, SockAddrFamily::Ipv6) => {
                    let data = get_array!(&self.data, 16);
                    // struct in6_addr is always encoded in big endian. The
                    // h/n/b/l printk specifiers are documented to be ignored
                    // for IPv6
                    let addr = u128::from_be_bytes(data);
                    let addr: std::net::Ipv6Addr = addr.into();
                    Ok(addr.into())
                }
                _ => panic!("Inconsistent sockaddr kind and family"),
            },
        }
    }
}

impl<'a> fmt::Display for SockAddr<'a> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        // Format of the structs described at:
        // https://www.gnu.org/software/libc/manual/html_node/Internet-Address-Formats.html The
        // order of struct members is different in the kernel struct.
        match self.to_socketaddr() {
            Ok(addr) => fmt::Display::fmt(&addr, f),
            Err(err) => write!(f, "ERROR<{err:?}>"),
        }
    }
}

/// Kernel bitmap value.
///
/// bitmaps are used for some types such as `cpumask_t`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bitmap<'a> {
    data: &'a [u8],
    pub(crate) chunk_size: LongSize,
    endianness: Endianness,
}

impl<'a> Bitmap<'a> {
    #[inline]
    pub(crate) fn from_bytes<'abi>(data: &'a [u8], abi: &'abi Abi) -> Self {
        let chunk_size: usize = abi.long_size.into();
        assert!(data.len() % chunk_size == 0);
        Bitmap {
            data,
            chunk_size: abi.long_size,
            endianness: abi.endianness,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'a> fmt::Display for Bitmap<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let mut range_start = None;
        let mut prev = None;
        let mut sep = "";

        let mut print_range = |range_start, prev, sep| match range_start {
            Some(range_start) if range_start == prev => {
                write!(f, "{sep}{prev}")
            }
            None => write!(f, "{sep}{prev}"),
            Some(range_start) => {
                write!(f, "{sep}{range_start}-{prev}")
            }
        };

        for curr in self {
            match prev {
                None => range_start = Some(curr),
                Some(prev) => {
                    if curr != prev + 1 {
                        print_range(range_start, prev, sep)?;
                        sep = ",";
                        range_start = Some(curr);
                    }
                }
            };
            prev = Some(curr);
        }
        if let Some(prev) = prev {
            print_range(range_start, prev, sep)?
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a Bitmap<'a> {
    type Item = <BitmapIterator<'a> as Iterator>::Item;
    type IntoIter = BitmapIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        BitmapIterator {
            bitmap: self,
            curr_chunk: None,
            next_chunk_index: 0,
            bit_index: 0,
        }
    }
}

pub struct BitmapIterator<'a> {
    bitmap: &'a Bitmap<'a>,

    curr_chunk: Option<u64>,
    next_chunk_index: usize,
    bit_index: usize,
}

impl<'a> BitmapIterator<'a> {
    fn next_chunk(&mut self) -> Option<u64> {
        let chunk_size = self.bitmap.chunk_size;
        let chunk_usize: usize = chunk_size.into();
        let data = self.bitmap.data;
        let base = self.next_chunk_index * chunk_usize;
        if base < data.len() {
            self.bit_index = 0;
            self.next_chunk_index += 1;
            let chunk = &data[base..base + chunk_usize];

            Some(match chunk_size {
                LongSize::Bits64 => {
                    let chunk = chunk.try_into().unwrap();
                    match self.bitmap.endianness {
                        Endianness::Little => u64::from_le_bytes(chunk),
                        Endianness::Big => u64::from_be_bytes(chunk),
                    }
                }
                LongSize::Bits32 => {
                    let chunk = chunk.try_into().unwrap();
                    match self.bitmap.endianness {
                        Endianness::Little => u32::from_le_bytes(chunk) as u64,
                        Endianness::Big => u32::from_be_bytes(chunk) as u64,
                    }
                }
            })
        } else {
            None
        }
    }

    /// Iterate over bytes in the bitmap.
    ///
    /// Endianness is controlled such that the first bit in the bitmap is the first bit of the
    /// first chunk.
    #[inline]
    pub fn as_bytes(&'a mut self) -> impl Iterator<Item = u8> + 'a {
        let mut curr = None;
        let max_idx = match self.bitmap.chunk_size {
            LongSize::Bits32 => 4,
            LongSize::Bits64 => 8,
        };
        let mut idx = max_idx;
        core::iter::from_fn(move || {
            if idx == max_idx {
                curr = self.next_chunk();
                idx = 0;
            }
            match curr {
                None => None,
                Some(chunk) => {
                    let x = (0xFF & (chunk >> (idx * 8))) as u8;
                    idx += 1;
                    Some(x)
                }
            }
        })
    }

    /// Iterate over each bit in the bitmap, with a `true` value if the bit is set, `false`
    /// otherwise.
    #[inline]
    pub fn as_bits(&'a mut self) -> impl Iterator<Item = bool> + 'a {
        let mut curr = None;
        let max_idx = match self.bitmap.chunk_size {
            LongSize::Bits32 => 32,
            LongSize::Bits64 => 64,
        };
        let mut idx = max_idx;
        core::iter::from_fn(move || {
            if idx == max_idx {
                curr = self.next_chunk();
                idx = 0;
            }
            match curr {
                None => None,
                Some(chunk) => {
                    let x = (0b1 & (chunk >> idx)) != 0;
                    idx += 1;
                    Some(x)
                }
            }
        })
    }
}

impl<'a> Iterator for BitmapIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let chunk_size: usize = self.bitmap.chunk_size.into();
        loop {
            match self.curr_chunk {
                Some(chunk) => {
                    if self.bit_index < chunk_size - 1 {
                        let bit_index = self.bit_index;
                        self.bit_index += 1;

                        let is_set = (chunk & (1 << bit_index)) != 0;
                        if is_set {
                            let global_index = bit_index + (self.next_chunk_index - 1) * chunk_size;
                            break Some(global_index);
                        }
                    } else {
                        self.curr_chunk = Some(self.next_chunk()?);
                    }
                }
                None => {
                    self.curr_chunk = Some(self.next_chunk()?);
                }
            }
        }
    }
}

/// Runtime value manipulated by the `C` interpeter.
///
/// # Integers
///
/// Integers are all represented by 64 bits signed ([Value::I64Scalar]) and unsigned
/// ([Value::U64Scalar]) values. The original size of the `C` type is available at the [Type] level
/// so there is little need to actually limit the value range itself.
///
/// # Pointers
///
/// `C` pointers fall into a few categories:
/// 1. Pointers to known arrays are represented by an array value at runtime (e.g.
///    [Value::U8Array]).
/// 2. `char *` values are represented by [Value::Str]. Such value will act as being
///    null-terminated and can only represent valid UTF-8.
/// 3. Pointers to known scalar values: this does not happen often in ftrace.  They are currently
///    only supported when appearing in `*&x` that gets simplified into `x`, or `&x` that is
///    represented by a symbolic address [Value::Addr].
/// 4. Pointers to unknown values. They are represented as an [Value::U64Scalar] at runtime.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Value<'a> {
    /// Unsigned integer value.
    U64Scalar(u64),
    /// Signed integer value.
    I64Scalar(i64),

    /// Similar to [Value::U8Array] but will act as a null-terminated string, and is
    /// guaranteed to be utf-8 encoded.
    Str(Str<'a>),

    /// Array of unsigned 8 bits integers.
    U8Array(Array<'a, u8>),
    /// Array of signed 8 bits integers.
    I8Array(Array<'a, i8>),

    /// Array of unsigned 16 bits integers.
    U16Array(Array<'a, u16>),
    /// Array of signed 16 bits integers.
    I16Array(Array<'a, i16>),

    /// Array of unsigned 32 bits integers.
    U32Array(Array<'a, u32>),
    /// Array of signed 32 bits integers.
    I32Array(Array<'a, i32>),

    /// Array of unsigned 64 bits integers.
    U64Array(Array<'a, u64>),
    /// Array of signed 64 bits integers.
    I64Array(Array<'a, i64>),

    /// Variable with unknown value. In ftrace format string arguments, it is used for `REC`, which
    /// is usually replaced by an event field access when it appears in the pattern
    /// `REC->myeventfield`.
    Variable(Identifier),

    /// Symbolic address of a value. Not much can be done with it apart from dereferencing it.
    Addr(ScratchBox<'a, Value<'a>>),

    /// Kernel bitmap, such as `cpumask_t`
    Bitmap(Bitmap<'a>),

    /// Kernel `struct sockaddr`.
    ///
    /// We don't use [std::net::SocketAddr] as we need to be able to represent any socket type the
    /// kernel can handle, which goes beyond `IP`.
    SockAddr(SockAddr<'a>),

    /// Value that could not be decoded to a more specific type. This will allow the consumer deal
    /// with the decoding, but provide limited support inside this library (e.g. in terms of
    /// printing).
    /// Any consumer relying on such value needs to keep up to date as new versions of this library
    /// might provide a more specific [Value] variant at some point and stop returning a
    /// [Value::Raw] for a given [Type].
    Raw(Arc<Type>, Array<'a, u8>),

    /// When nothing else is more appropriate, a value can simply be unknown.
    Unknown,
}

impl<'a> Value<'a> {
    /// Treat the value as a pointer and try to dereference it in the `env` evaluation environment.
    ///
    /// * [Value::Addr] layer will be removed.
    /// * [Value::Str] layer will stay as-is. `char *` values are represented by a [Value::Str] at
    ///   runtime and dereferencing them is expected to provide the actual char array, which is
    ///   also represented by the same [Value::Str]. If we treated such `char *` deref as returning
    ///   the first character of the string, we would forever loose the knowledge that this char is
    ///   the first char of a string, and any subsequent operation on the value (such as getting
    ///   its address to form a "new" string) would not work as expected.
    /// * Similarly to [Value::Str] all array variants stay equal to themselves.
    #[inline]
    pub fn deref_ptr<'ee, EE>(&'a self, env: &'ee EE) -> Result<Value<'a>, EvalError>
    where
        'ee: 'a,
        EE: EvalEnv<'ee> + ?Sized,
    {
        match self {
            Value::Addr(sbox) => Ok(sbox.deref().clone()),
            Value::Str(s) => Ok(Value::Str(Str::new_borrowed(s))),

            Value::U64Scalar(addr) => env.deref_static(*addr),
            Value::I64Scalar(addr) => env.deref_static(*addr as u64),

            Value::U8Array(arr) => Ok(Value::U8Array(Array::Borrowed(arr))),
            Value::I8Array(arr) => Ok(Value::I8Array(Array::Borrowed(arr))),

            Value::U16Array(arr) => Ok(Value::U16Array(Array::Borrowed(arr))),
            Value::I16Array(arr) => Ok(Value::I16Array(Array::Borrowed(arr))),

            Value::U32Array(arr) => Ok(Value::U32Array(Array::Borrowed(arr))),
            Value::I32Array(arr) => Ok(Value::I32Array(Array::Borrowed(arr))),

            Value::U64Array(arr) => Ok(Value::U64Array(Array::Borrowed(arr))),
            Value::I64Array(arr) => Ok(Value::I64Array(Array::Borrowed(arr))),
            val => Err(EvalError::IllegalType(val.clone().into_static().ok())),
        }
    }

    /// Iterate over the bytes of array-like values, including [Value::Raw].
    /// [Value::Str] iterator will yield the null terminator of the C string.
    pub fn to_bytes(&self) -> Option<impl IntoIterator<Item = u8> + '_> {
        use Value::*;

        let (add_null, slice) = match self {
            Str(s) => Some((true, s.as_bytes())),
            Raw(_, arr) => Some((false, arr.deref())),

            U8Array(arr) => Some((false, arr.deref())),
            I8Array(arr) => Some((false, cast_slice(arr))),

            U16Array(arr) => Some((false, cast_slice(arr))),
            I16Array(arr) => Some((false, cast_slice(arr))),

            U32Array(arr) => Some((false, cast_slice(arr))),
            I32Array(arr) => Some((false, cast_slice(arr))),

            U64Array(arr) => Some((false, cast_slice(arr))),
            I64Array(arr) => Some((false, cast_slice(arr))),
            _ => None,
        }?;
        let mut iter = slice.iter();
        Some(core::iter::from_fn(move || match iter.next().copied() {
            Some(x) => Some(x),
            None if add_null => Some(0),
            _ => None,
        }))
    }

    /// Convert char array values to a [&str].
    pub fn to_str(&self) -> Option<&str> {
        macro_rules! from_array {
            ($s:expr) => {
                if let Some(s) = $s.split(|c| *c == 0).next() {
                    if let Ok(s) = std::str::from_utf8(s) {
                        Some(s)
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
        }
        match self {
            Value::U8Array(s) => from_array!(s),
            Value::I8Array(s) => from_array!(cast_slice(s)),
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Create a static value, which is sometimes necessary to store in some containers.
    ///
    /// Some [Value] variants are only available as a data-borrowing flavor and are not convertible
    /// to a `'static` value.
    pub fn into_static(self) -> Result<Value<'static>, Value<'a>> {
        use Value::*;

        macro_rules! array {
            ($variant:ident, $arr:expr) => {
                Ok($variant($arr.into_static()))
            };
        }
        match self {
            U64Scalar(x) => Ok(U64Scalar(x)),
            I64Scalar(x) => Ok(I64Scalar(x)),

            Str(s) => Ok(Str(s.into_static())),

            U8Array(arr) => array!(U8Array, arr),
            I8Array(arr) => array!(I8Array, arr),

            U16Array(arr) => array!(U16Array, arr),
            I16Array(arr) => array!(I16Array, arr),

            U32Array(arr) => array!(U32Array, arr),
            I32Array(arr) => array!(I32Array, arr),

            U64Array(arr) => array!(U64Array, arr),
            I64Array(arr) => array!(I64Array, arr),

            Raw(typ, arr) => Ok(Raw(typ, arr.into_static())),
            Addr(addr) => {
                let addr = addr.deref().clone();
                let addr = addr.into_static()?;
                Ok(Addr(ScratchBox::Arc(Arc::new(addr))))
            }
            Variable(id) => Ok(Variable(id)),
            Unknown => Ok(Unknown),

            // The only bitmaps that exist are created by the kernel and stored
            // in a field, they are never synthesized by any expression that
            // could be evaluated ahead of time.
            bitmap @ Bitmap(_) => Err(bitmap),
            sockaddr @ SockAddr(_) => Err(sockaddr),
        }
    }

    /// Treat the value as an array and lookup the `i` item in it.
    ///
    /// [Value::U64Scalar] and [Value::I64Scalar] are treated as pointers.
    fn get<EE: EvalEnv<'a> + ?Sized>(self, env: &'a EE, i: usize) -> Result<Value<'a>, EvalError> {
        let (derefed, val) = match self {
            Value::U64Scalar(addr) => (true, env.deref_static(addr)?),
            Value::I64Scalar(addr) => (true, env.deref_static(addr as u64)?),
            Value::Addr(val) => (true, val.into_inner()),
            val => (false, val),
        };

        macro_rules! match_ {
            ($(($array_ctor:tt, $scalar_ctor:tt)),*) => {
                match val {
                    $(
                        Value::$array_ctor(vec) => {
                            match vec.deref().get(i) {
                                None => Err(EvalError::OutOfBondIndex(i)),
                                Some(x) => Ok(Value::$scalar_ctor(x.clone().into()))
                            }
                        }
                    ),*
                    Value::Str(s) => {
                        match s.as_bytes().get(i) {
                            None => {
                                if i == s.len() {
                                    Ok(Value::U64Scalar(0))
                                } else {
                                    Err(EvalError::OutOfBondIndex(i))
                                }
                            }
                            Some(c) => Ok(Value::U64Scalar((*c).into())),
                        }
                    }
                    val if derefed && i == 0 => Ok(val),
                    val => Err(EvalError::CannotIndexScalar(val.into_static().ok()))
                }
            }
        }
        match_! {
            (I8Array, I64Scalar),
            (U8Array, U64Scalar),

            (I16Array, I64Scalar),
            (U16Array, U64Scalar),

            (I32Array, I64Scalar),
            (U32Array, U64Scalar),

            (I64Array, I64Scalar),
            (U64Array, U64Scalar)
        }
    }
}

impl<'a> fmt::Display for Value<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        macro_rules! display {
            ($x:expr) => {{
                fmt::Display::fmt(&$x, f)
            }};
        }

        match self {
            Value::U64Scalar(x) => display!(x),
            Value::I64Scalar(x) => display!(x),
            Value::Str(x) => display!(x),
            Value::U8Array(x) => display!(x),
            Value::I8Array(x) => display!(x),
            Value::U16Array(x) => display!(x),
            Value::I16Array(x) => display!(x),
            Value::U32Array(x) => display!(x),
            Value::I32Array(x) => display!(x),
            Value::U64Array(x) => display!(x),
            Value::I64Array(x) => display!(x),
            Value::Variable(x) => display!(x),
            Value::Addr(x) => write!(f, "<ADDRESS OFF<{}>>", x.deref()),
            Value::Bitmap(x) => display!(x),
            Value::SockAddr(x) => display!(x),
            Value::Raw(typ, data) => {
                write!(f, "<RAW DATA {typ:?} [")?;
                for (i, x) in data.iter().enumerate() {
                    if i != 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{x:#04x}")?;
                }
                write!(f, "]>")?;
                Ok(())
            }
            Value::Unknown => write!(f, "<UNKNOWN>"),
        }
    }
}

/// Compilation environment.
pub trait CompileEnv<'ce>: EvalEnv<'ce> + ParseEnv
where
    Self: 'ce,
{
    /// Provide an [Evaluator] that will give the value of the `id` event field when evaluated.
    fn field_getter(&self, id: &str) -> Result<Box<dyn Evaluator>, CompileError>;
}

/// Evaluation environment.
pub trait EvalEnv<'ee>
where
    Self: 'ee + Send + Sync,
{
    /// Dereference a static value at the given address.
    ///
    /// This could for example be a string in the string table stored in the header.
    fn deref_static(&self, _addr: u64) -> Result<Value<'_>, EvalError>;
    /// Binary content of the current event record being processed.
    fn event_data(&self) -> Result<&[u8], EvalError>;

    /// [ScratchAlloc] available while interpreting an expression.
    fn scratch(&self) -> &ScratchAlloc;

    /// Current [Header].
    fn header(&self) -> Result<&Header, EvalError>;
}

impl<'ee, 'eeref> EvalEnv<'eeref> for &'eeref (dyn CompileEnv<'ee> + 'ee) {
    #[inline]
    fn deref_static(&self, addr: u64) -> Result<Value<'_>, EvalError> {
        (*self).deref_static(addr)
    }

    #[inline]
    fn event_data(&self) -> Result<&[u8], EvalError> {
        (*self).event_data()
    }

    #[inline]
    fn scratch(&self) -> &ScratchAlloc {
        (*self).scratch()
    }

    #[inline]
    fn header(&self) -> Result<&Header, EvalError> {
        (*self).header()
    }
}

impl<'ce, 'ceref> ParseEnv for &'ceref (dyn ParseEnv + 'ce) {
    fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
        (*self).field_typ(id)
    }
    fn abi(&self) -> &Abi {
        (*self).abi()
    }
}

impl<'ce, 'ceref> ParseEnv for &'ceref (dyn CompileEnv<'ce> + 'ce) {
    fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
        (*self).field_typ(id)
    }
    fn abi(&self) -> &Abi {
        (*self).abi()
    }
}

impl<'ce, 'ceref> CompileEnv<'ceref> for &'ceref (dyn CompileEnv<'ce> + 'ce) {
    fn field_getter(&self, id: &str) -> Result<Box<dyn Evaluator>, CompileError> {
        (*self).field_getter(id)
    }
}

pub(crate) struct BasicEnv<'pe, PE: ?Sized> {
    scratch: ScratchAlloc,
    penv: &'pe PE,
}

impl<'pe, PE: ?Sized> BasicEnv<'pe, PE> {
    pub fn new(penv: &'pe PE) -> Self {
        BasicEnv {
            penv,
            scratch: ScratchAlloc::new(),
        }
    }
}

impl<'ee, PE> EvalEnv<'ee> for BasicEnv<'ee, PE>
where
    PE: ?Sized + Send + Sync,
{
    #[inline]
    fn scratch(&self) -> &ScratchAlloc {
        &self.scratch
    }

    fn header(&self) -> Result<&Header, EvalError> {
        Err(EvalError::NoHeader)
    }

    fn deref_static(&self, addr: Address) -> Result<Value<'_>, EvalError> {
        Err(EvalError::CannotDeref(addr))
    }
    fn event_data(&self) -> Result<&[u8], EvalError> {
        Err(EvalError::NoEventData)
    }
}

impl<'pe, PE> ParseEnv for BasicEnv<'pe, PE>
where
    PE: ParseEnv + ?Sized + Send + Sync,
{
    #[inline]
    fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
        self.penv.field_typ(id)
    }
    #[inline]
    fn abi(&self) -> &Abi {
        self.penv.abi()
    }
}

impl<'ce, PE> CompileEnv<'ce> for BasicEnv<'ce, PE>
where
    PE: ParseEnv + ?Sized + Send + Sync,
{
    #[inline]
    fn field_getter(&self, id: &str) -> Result<Box<dyn Evaluator>, CompileError> {
        Err(CompileError::UnknownField(id.into()))
    }
}

/// [EvalEnv] used to manipulate [Value] (e.g. call [Value::deref_ptr]) in the context of a given
/// event record.
pub struct BufferEnv<'a> {
    scratch: &'a ScratchAlloc,
    header: &'a Header,
    data: &'a [u8],
}
impl<'a> BufferEnv<'a> {
    /// Create a [BufferEnv] from the given event record buffer `data`.
    pub fn new(scratch: &'a ScratchAlloc, header: &'a Header, data: &'a [u8]) -> Self {
        BufferEnv {
            scratch,
            header,
            data,
        }
    }
}

impl<'ee> EvalEnv<'ee> for BufferEnv<'ee> {
    #[inline]
    fn scratch(&self) -> &ScratchAlloc {
        self.scratch
    }

    #[inline]
    fn deref_static(&self, addr: u64) -> Result<Value<'_>, EvalError> {
        self.header.deref_static(addr)
    }

    fn event_data(&self) -> Result<&[u8], EvalError> {
        Ok(self.data)
    }

    fn header(&self) -> Result<&Header, EvalError> {
        Ok(self.header)
    }
}

/// Arithmetic information about a [Type]
pub(crate) struct ArithInfo<'a> {
    typ: &'a Type,
    /// type rank as defined in the C standard
    rank: u32,
    width: FileSize,
    signed: Type,
    unsigned: Type,
}

impl<'a> ArithInfo<'a> {
    #[inline]
    pub fn is_signed(&self) -> bool {
        self.typ == &self.signed
    }

    #[inline]
    pub fn signedness(&self) -> Signedness {
        if self.is_signed() {
            Signedness::Signed
        } else {
            Signedness::Unsigned
        }
    }
}

impl Type {
    /// Returns `true` if the type is an arithmetic type.
    #[inline]
    pub(crate) fn is_arith(&self) -> bool {
        self.arith_info().is_some()
    }

    pub(crate) fn arith_info(&self) -> Option<ArithInfo> {
        let typ = self.resolve_wrapper();

        use Type::*;
        match typ {
            U8 | I8 | Bool => Some(ArithInfo {
                typ,
                rank: 0,
                signed: I8,
                unsigned: U8,
                width: 8,
            }),
            U16 | I16 => Some(ArithInfo {
                typ,
                rank: 1,
                signed: I16,
                unsigned: U16,
                width: 16,
            }),
            U32 | I32 => Some(ArithInfo {
                typ,
                rank: 2,
                signed: I32,
                unsigned: U32,
                width: 32,
            }),
            U64 | I64 => Some(ArithInfo {
                typ,
                rank: 3,
                signed: I64,
                unsigned: U64,
                width: 64,
            }),
            _ => None,
        }
    }

    /// Integer promotion
    pub(crate) fn promote(self) -> Type {
        match self.arith_info() {
            Some(info) => {
                if info.width <= 32 {
                    if info.is_signed() {
                        Type::I32
                    } else {
                        Type::U32
                    }
                } else {
                    self
                }
            }
            None => self,
        }
    }

    /// Access the nested inner type of an [Type::Typedef] or the underlying integer type of an
    /// [Type::Enum].
    pub fn resolve_wrapper(&self) -> &Self {
        match self {
            Type::Typedef(typ, _) | Type::Enum(typ, _) => typ.resolve_wrapper(),
            _ => self,
        }
    }

    /// Decay an [Type::Array] to a [Type::Pointer].
    pub fn decay_to_ptr(self) -> Type {
        match self {
            Type::Array(typ, ..) => Type::Pointer(typ),
            typ => typ,
        }
    }
}

type ArithConverter = dyn for<'a> Fn(Value<'a>) -> Result<Value<'a>, EvalError> + Send + Sync;

#[inline]
fn convert_arith(dst: &Type) -> Result<Box<ArithConverter>, CompileError> {
    macro_rules! convert {
        ($typ:ty, $ctor:ident) => {
            Ok(Box::new(|x| {
                let x = match x {
                    Value::I64Scalar(x) => x as $typ,
                    Value::U64Scalar(x) => x as $typ,
                    val => return Err(EvalError::IllegalType(val.into_static().ok())),
                };
                Ok(Value::$ctor(x.into()))
            }))
        };
    }

    use Type::*;
    match dst.resolve_wrapper() {
        Bool => convert!(u8, U64Scalar),
        I8 => convert!(i8, I64Scalar),
        U8 => convert!(u8, U64Scalar),
        I16 => convert!(i16, I64Scalar),
        U16 => convert!(u16, U64Scalar),
        I32 => convert!(i32, I64Scalar),
        U32 => convert!(u32, U64Scalar),
        I64 => convert!(i64, I64Scalar),
        U64 => convert!(u64, U64Scalar),
        typ => Err(CompileError::NonArithmeticOperand(typ.clone())),
    }
}

fn usual_arith_conv(lhs: Type, rhs: Type) -> Result<Type, CompileError> {
    let lhs = lhs.promote();
    let rhs = rhs.promote();

    match (lhs.arith_info(), rhs.arith_info()) {
        (Some(lhs_info), Some(rhs_info)) => Ok({
            if lhs == rhs {
                lhs
            } else if lhs_info.is_signed() == rhs_info.is_signed() {
                if lhs_info.rank > rhs_info.rank {
                    lhs
                } else {
                    rhs
                }
            } else {
                let (styp, styp_info, utyp, utyp_info) = if lhs_info.is_signed() {
                    (&lhs, lhs_info, &rhs, rhs_info)
                } else {
                    (&rhs, rhs_info, &lhs, lhs_info)
                };

                if utyp_info.rank >= styp_info.rank {
                    utyp.clone()
                } else if styp_info.width > utyp_info.width {
                    styp.clone()
                } else {
                    styp_info.unsigned
                }
            }
        }),
        (None, _) => Err(CompileError::NonArithmeticOperand(lhs)),
        (_, None) => Err(CompileError::NonArithmeticOperand(rhs)),
    }
}

#[inline]
fn convert_arith_ops(
    _abi: &Abi,
    lhs: Type,
    rhs: Type,
) -> Result<(Type, Box<ArithConverter>, Box<ArithConverter>), CompileError> {
    let typ = usual_arith_conv(lhs.clone(), rhs.clone())?;
    Ok((typ.clone(), convert_arith(&typ)?, convert_arith(&typ)?))
}

#[inline]
fn convert_arith_op<'ce, CE>(
    cenv: &CE,
    expr: &Expr,
) -> Result<(Type, Box<ArithConverter>), CompileError>
where
    CE: CompileEnv<'ce>,
{
    let typ = expr.typ(cenv)?;
    let promoted = typ.promote();
    Ok((promoted.clone(), convert_arith(&promoted)?))
}

impl Type {
    /// Byte-size of the type.
    pub fn size(&self, abi: &Abi) -> Result<FileSize, CompileError> {
        let typ = self.resolve_wrapper();
        use Type::*;
        match typ {
            Pointer(_) => Ok(abi.long_size.into()),
            Array(typ, size) => match size {
                ArrayKind::Fixed(Ok(size)) => {
                    let item = typ.size(abi)?;
                    Ok(size * item)
                }
                _ => Err(CompileError::UnknownSize(*typ.clone())),
            },
            _ => {
                let info = typ
                    .arith_info()
                    .ok_or_else(|| CompileError::UnknownSize(typ.clone()))?;
                Ok(info.width / 8)
            }
        }
    }

    fn to_arith(&self, abi: &Abi) -> Result<Type, CompileError> {
        match self.resolve_wrapper() {
            Type::Pointer(_) | Type::Array(..) => match abi.long_size {
                LongSize::Bits32 => Ok(Type::U32),
                LongSize::Bits64 => Ok(Type::U64),
            },
            typ => {
                // Check it's an arithmetic type
                typ.arith_info()
                    .ok_or_else(|| CompileError::NonArithmeticOperand(typ.clone()))?;
                Ok(typ.clone())
            }
        }
    }
}

impl Expr {
    /// Typecheck the expression and return its [Type].
    pub fn typ<PE>(&self, penv: &PE) -> Result<Type, CompileError>
    where
        PE: ParseEnv + ?Sized,
    {
        use Expr::*;

        let abi = penv.abi();
        let recurse = |expr: &Expr| expr.typ(penv);

        match self {
            Evaluated(typ, _) => Ok(typ.clone()),
            Uninit => Ok(Type::Unknown),
            Variable(typ, _id) => Ok(typ.clone()),

            InitializerList(_) => Ok(Type::Unknown),
            DesignatedInitializer(_, init) => recurse(init),
            CompoundLiteral(typ, _) => Ok(typ.clone()),

            IntConstant(typ, _) | CharConstant(typ, _) | EnumConstant(typ, _) => Ok(typ.clone()),
            StringLiteral(str) => {
                let len: u64 = str.len().try_into().unwrap();
                // null terminator
                let len = len + 1;
                Ok(Type::Array(
                    Box::new(abi.char_typ()),
                    ArrayKind::Fixed(Ok(len)),
                ))
            }

            Addr(expr) => Ok(Type::Pointer(Box::new(recurse(expr)?))),
            Deref(expr) => match recurse(expr)?.resolve_wrapper() {
                Type::Pointer(typ) | Type::Array(typ, _) => Ok(*typ.clone()),
                typ => Err(CompileError::CannotDeref(typ.clone(), *expr.clone())),
            },
            Plus(expr) | Minus(expr) | Tilde(expr) => Ok(recurse(expr)?.promote()),
            Bang(_) => Ok(Type::I32),
            Cast(typ, _) => Ok(typ.clone()),
            SizeofType(..) | SizeofExpr(_) => Ok(match &abi.long_size {
                LongSize::Bits32 => Type::U32,
                LongSize::Bits64 => Type::U64,
            }),
            PreInc(expr) | PreDec(expr) | PostInc(expr) | PostDec(expr) => recurse(expr),

            MemberAccess(expr, member) => match recurse(expr)?.resolve_wrapper() {
                Type::Variable(id) if id == "REC" => Ok(penv.field_typ(member)?),
                _ => Ok(Type::Unknown),
            },
            FuncCall(..) => Ok(Type::Unknown),
            Subscript(expr, idx) => {
                let idx = recurse(idx)?;
                match idx.arith_info() {
                    Some(info) => Ok(info),
                    None => Err(CompileError::NonArithmeticOperand(idx)),
                }?;

                match recurse(expr)?.resolve_wrapper() {
                    Type::Array(typ, _) | Type::Pointer(typ) => Ok(*typ.clone()),
                    typ => Err(CompileError::NotAnArray(typ.clone())),
                }
            }

            Assign(_lhs, rhs) => recurse(rhs),

            Eq(..) | NEq(..) | LoEq(..) | HiEq(..) | Hi(..) | Lo(..) | And(..) | Or(..) => {
                Ok(Type::I32)
            }

            LShift(expr, _) | RShift(expr, _) => Ok(recurse(expr)?.promote()),

            Mul(lhs, rhs)
            | Div(lhs, rhs)
            | Mod(lhs, rhs)
            | Add(lhs, rhs)
            | Sub(lhs, rhs)
            | BitAnd(lhs, rhs)
            | BitOr(lhs, rhs)
            | BitXor(lhs, rhs) => {
                let lhs = recurse(lhs)?.resolve_wrapper().clone().decay_to_ptr();
                let rhs = recurse(rhs)?.resolve_wrapper().clone().decay_to_ptr();
                match (&lhs, &rhs) {
                    (Type::Pointer(_lhs), _rhs) if _rhs.is_arith() => Ok(lhs),
                    (_lhs, Type::Pointer(_rhs)) if _lhs.is_arith() => Ok(rhs),
                    _ => usual_arith_conv(lhs, rhs),
                }
            }
            Ternary(_, lhs, rhs) => {
                let lhstyp = recurse(lhs)?.resolve_wrapper().clone().decay_to_ptr();
                let rhstyp = recurse(rhs)?.resolve_wrapper().clone().decay_to_ptr();

                fn is_null_ptr_cst(expr: &Expr) -> bool {
                    match expr {
                        Expr::IntConstant(_, 0) => true,
                        Expr::Cast(Pointer(pointee), expr) if pointee.deref() == &Void => {
                            is_null_ptr_cst(expr)
                        }
                        _ => false,
                    }
                }

                use Type::*;
                match usual_arith_conv(lhstyp.clone(), rhstyp.clone()) {
                    Ok(typ) => Ok(typ),
                    Err(_) => match (&lhstyp, &rhstyp) {
                        (lhstyp_, rhstyp_) if lhstyp_ == rhstyp_ => Ok(lhstyp),
                        (Pointer(_), _) if is_null_ptr_cst(rhs) => Ok(lhstyp),
                        (_, Pointer(_)) if is_null_ptr_cst(lhs) => Ok(rhstyp),
                        (Pointer(inner), Pointer(_)) | (Pointer(_), Pointer(inner))
                            if inner.deref() == &Void =>
                        {
                            Ok(Pointer(Box::new(Void)))
                        }
                        (Pointer(lhstyp_), Pointer(rhstyp_))
                            if lhstyp_.resolve_wrapper() == rhstyp_.resolve_wrapper() =>
                        {
                            Ok(Pointer(Box::new(lhstyp)))
                        }
                        _ => Err(CompileError::MismatchingOperandType(
                            self.clone(),
                            lhstyp,
                            rhstyp,
                        )),
                    },
                }
            }
            CommaExpr(exprs) => recurse(exprs.last().unwrap()),

            ExtensionMacro(desc) => match &desc.kind {
                ExtensionMacroKind::ObjectLike { typ, .. } => Ok(typ.clone()),
                ExtensionMacroKind::FunctionLike { .. } => Ok(Type::Unknown),
            },
            ExtensionMacroCall(cparser::ExtensionMacroCall { compiler, .. }) => {
                Ok(compiler.ret_typ.clone())
            }
        }
    }
}

/// Evaluate an expression to a [Value] when given an evaluation environment [EvalEnv].
pub trait Evaluator: Send + Sync {
    fn eval<'eeref, 'ee>(
        &self,
        env: &'eeref (dyn EvalEnv<'ee> + 'eeref),
    ) -> Result<Value<'eeref>, EvalError>;
}

impl<F> Evaluator for F
where
    F: for<'ee, 'eeref> Fn(&'eeref (dyn EvalEnv<'ee> + 'eeref)) -> Result<Value<'eeref>, EvalError>
        + Send
        + Sync,
{
    fn eval<'eeref, 'ee>(
        &self,
        env: &'eeref (dyn EvalEnv<'ee> + 'eeref),
    ) -> Result<Value<'eeref>, EvalError> {
        self(env)
    }
}

// TODO: the day Rust infers correctly HRTB for closures, this won't be necessary anymore
#[inline]
pub(crate) fn new_dyn_evaluator<F>(f: F) -> Box<dyn Evaluator>
where
    F: for<'ee, 'eeref> Fn(&'eeref (dyn EvalEnv<'ee> + 'eeref)) -> Result<Value<'eeref>, EvalError>
        + Send
        + Sync
        + 'static,
{
    Box::new(f)
}

impl Expr {
    /// Evaluate the expression assuming it is a constant expression.
    ///
    /// Note that this will evaluate any expression that does not need extra input from an
    /// [EvalEnv] (such as an event field value). This covers more than what is considered a const
    /// expr in C.
    pub fn eval_const<T, F>(self, abi: &Abi, f: F) -> T
    where
        F: for<'a> FnOnce(Result<Value<'a>, InterpError>) -> T,
    {
        let env = BasicEnv::new(abi);
        let eval = || -> Result<_, InterpError> {
            let eval = self.compile(&env)?;
            Ok(eval.eval(&env)?)
        };
        f(eval())
    }

    /// Simplify the expression into a normal form.
    ///
    /// Some basic expression rewriting will be done, along with constant folding.
    pub fn simplify<'ce, CE>(self, cenv: &'ce CE) -> Expr
    where
        CE: CompileEnv<'ce>,
    {
        let compiled = self.clone().compile(cenv);
        self._do_simplify(cenv, compiled)
    }

    fn _simplify<'ce, CE>(self, cenv: &'ce CE) -> Expr
    where
        CE: CompileEnv<'ce>,
    {
        let compiled = self.clone()._compile(cenv);
        self._do_simplify(cenv, compiled)
    }

    fn _do_simplify<'ce, CE>(
        self,
        cenv: &'ce CE,
        compiled: Result<Box<dyn Evaluator>, CompileError>,
    ) -> Expr
    where
        CE: CompileEnv<'ce>,
    {
        match compiled {
            Ok(eval) => match self.typ(cenv) {
                Ok(typ) => match eval.eval(cenv) {
                    Ok(value) => match value.into_static() {
                        Ok(value) => Expr::Evaluated(typ, value),
                        Err(_) => self,
                    },
                    Err(_) => self,
                },
                Err(_) => self,
            },
            Err(_) => self,
        }
    }

    /// Compile the expression to an [Evaluator] that will then be used to actually evaluate the
    /// expression in a given [EvalEnv].
    ///
    /// This staged compilation/evaluation system saves a lot of computation time when evaluating
    /// the same expression over and over again.
    pub fn compile<'ce, CE>(self, cenv: &'ce CE) -> Result<Box<dyn Evaluator>, CompileError>
    where
        CE: CompileEnv<'ce>,
    {
        // Type check the AST. This should be done only once on the root node, so any recursive
        // compilation invocations are done via _compile() to avoid re-doing it and avoid an O(N^2)
        // complexity
        self.typ(cenv)?;
        self._compile(cenv)
    }

    fn _compile<'ce, CE>(self, cenv: &'ce CE) -> Result<Box<dyn Evaluator>, CompileError>
    where
        CE: CompileEnv<'ce>,
    {
        use Expr::*;
        let abi = cenv.abi();
        let cannot_handle = |expr| Err(CompileError::ExprNotHandled(expr));
        let recurse = |expr: Expr| expr._compile(cenv);
        let simplify = |expr: Expr| expr._simplify(cenv);
        let uintptr_t = abi.ulong_typ();

        fn to_signed(x: u64) -> Result<i64, EvalError> {
            x.try_into()
                .map_err(|_| EvalError::CannotConvertToSigned(x))
        }

        fn multiply_by_pointee_size(pointee: Type, expr: Expr) -> Expr {
            match pointee {
                Type::Void | Type::U8 | Type::I8 => expr,
                _ => Mul(Box::new(expr), Box::new(SizeofType(pointee))),
            }
        }

        macro_rules! binop {
            ($ctor:expr, $lhs:expr, $rhs:expr, $op:expr) => {{
                let lhs = *$lhs;
                let rhs = *$rhs;
                let eval_lhs = recurse(lhs.clone())?;
                let eval_rhs = recurse(rhs.clone())?;

                let lhstyp = lhs.typ(cenv)?;
                let rhstyp = rhs.typ(cenv)?;

                let lhsisarith = lhstyp.is_arith();
                let rhsisarith = rhstyp.is_arith();

                let ctor: Option<fn(Box<Expr>, Box<Expr>) -> Expr> = $ctor;
                match (lhstyp, lhsisarith, rhstyp, rhsisarith, ctor) {
                    // Binary operation between two arithmetic types
                    (lhs, true, rhs, true, _) => {
                        let (_typ, conv_lhs, conv_rhs) = convert_arith_ops(abi, lhs, rhs)?;

                        Ok(new_dyn_evaluator(move |env| {
                            let lhs = conv_lhs(eval_lhs.eval(env)?)?;
                            let rhs = conv_rhs(eval_rhs.eval(env)?)?;

                            match (lhs, rhs) {
                                (Value::U64Scalar(x), Value::U64Scalar(y)) =>
                                {
                                    #[allow(clippy::redundant_closure_call)]
                                    Ok(Value::U64Scalar($op(Wrapping(x), Wrapping(y)).0))
                                }
                                (Value::I64Scalar(x), Value::I64Scalar(y)) =>
                                {
                                    #[allow(clippy::redundant_closure_call)]
                                    Ok(Value::I64Scalar($op(Wrapping(x), Wrapping(y)).0))
                                }
                                (val, _) => Err(EvalError::IllegalType(val.into_static().ok())),
                            }
                        }))
                    }
                    // Pointer arithmetic
                    (Type::Pointer(_lhstyp), _, _rhstyp, true, Some(ctor)) => {
                        // Convert "A op B" where type(A) = Pointer(Ta) and type(B) is arithmetic
                        // into:
                        // (Ta*)((u64)A + B * sizeof(TA))
                        recurse(Cast(
                            Type::Pointer(_lhstyp.clone()),
                            Box::new(ctor(
                                Box::new(Cast(uintptr_t.clone(), Box::new(lhs))),
                                Box::new(multiply_by_pointee_size(*_lhstyp, rhs)),
                            )),
                        ))
                    }

                    (_lhstyp, true, Type::Pointer(_rhstyp), _, Some(ctor)) => recurse(Cast(
                        Type::Pointer(_rhstyp.clone()),
                        Box::new(ctor(
                            Box::new(multiply_by_pointee_size(*_rhstyp, lhs)),
                            Box::new(Cast(uintptr_t.clone(), Box::new(rhs))),
                        )),
                    )),
                    (lhs, false, _, _, _) => Err(CompileError::NonArithmeticOperand(lhs)),
                    (_, _, rhs, false, _) => Err(CompileError::NonArithmeticOperand(rhs)),
                }
            }};
        }

        macro_rules! comp {
            ($lhs:expr, $rhs:expr, $op:expr) => {{
                let lhs = *$lhs;
                let rhs = *$rhs;
                let eval_lhs = recurse(lhs.clone())?;
                let eval_rhs = recurse(rhs.clone())?;

                let lhs = lhs.typ(cenv)?.to_arith(abi)?;
                let rhs = rhs.typ(cenv)?.to_arith(abi)?;

                let (_typ, conv_lhs, conv_rhs) = convert_arith_ops(abi, lhs, rhs)?;

                Ok(new_dyn_evaluator(move |env| {
                    let lhs = conv_lhs(eval_lhs.eval(env)?)?;
                    let rhs = conv_rhs(eval_rhs.eval(env)?)?;

                    match (lhs, rhs) {
                        (Value::U64Scalar(x), Value::U64Scalar(y)) =>
                        {
                            #[allow(clippy::redundant_closure_call)]
                            Ok(Value::I64Scalar($op(x, y)))
                        }
                        (Value::I64Scalar(x), Value::I64Scalar(y)) =>
                        {
                            #[allow(clippy::redundant_closure_call)]
                            Ok(Value::I64Scalar($op(x, y)))
                        }
                        (val, _) => Err(EvalError::IllegalType(val.into_static().ok())),
                    }
                }))
            }};
        }

        macro_rules! shift {
            ($lhs:expr, $rhs:expr, $op:expr) => {{
                let lhs = *$lhs;
                let rhs = *$rhs;
                let eval_lhs = recurse(lhs.clone())?;
                let eval_rhs = recurse(rhs.clone())?;

                let (_typ, conv_lhs) = convert_arith_op(cenv, &lhs)?;
                let (_, conv_rhs) = convert_arith_op(cenv, &rhs)?;

                Ok(new_dyn_evaluator(move |env| {
                    let lhs = conv_lhs(eval_lhs.eval(env)?)?;
                    let rhs = conv_rhs(eval_rhs.eval(env)?)?;

                    match (lhs, rhs) {
                        (Value::U64Scalar(x), Value::U64Scalar(y)) =>
                        {
                            #[allow(clippy::redundant_closure_call)]
                            Ok(Value::U64Scalar($op(x, y)))
                        }
                        (Value::U64Scalar(x), Value::I64Scalar(y)) =>
                        {
                            #[allow(clippy::redundant_closure_call)]
                            Ok(Value::U64Scalar($op(x, y)))
                        }

                        (Value::I64Scalar(x), Value::U64Scalar(y)) =>
                        {
                            #[allow(clippy::redundant_closure_call)]
                            Ok(Value::I64Scalar($op(x, y)))
                        }
                        (Value::I64Scalar(x), Value::I64Scalar(y)) =>
                        {
                            #[allow(clippy::redundant_closure_call)]
                            Ok(Value::I64Scalar($op(x, y)))
                        }
                        (val, _) => Err(EvalError::IllegalType(val.into_static().ok())),
                    }
                }))
            }};
        }

        let eval = match self {
            Evaluated(_typ, value) => Ok(new_dyn_evaluator(move |_| Ok(value.clone()))),
            Variable(_typ, id) => Ok(new_dyn_evaluator(move |_| Ok(Value::Variable(id.clone())))),

            MemberAccess(expr, member) => {
                let expr = simplify(*expr);
                match &expr {
                    Variable(_, id) | Evaluated(_, Value::Variable(id)) if id == "REC" => {
                        cenv.field_getter(&member)
                    }
                    _ => cannot_handle(expr),
                }
            }

            expr @ (Uninit
            | InitializerList(_)
            | DesignatedInitializer(..)
            | CompoundLiteral(..)) => cannot_handle(expr),
            IntConstant(typ, x) | CharConstant(typ, x) => {
                let typ = typ.to_arith(abi)?;
                let info = match typ.arith_info() {
                    Some(info) => Ok(info),
                    None => Err(CompileError::NonArithmeticOperand(typ)),
                }?;
                Ok(if info.is_signed() {
                    new_dyn_evaluator(move |_| Ok(Value::I64Scalar(to_signed(x)?)))
                } else {
                    new_dyn_evaluator(move |_| Ok(Value::U64Scalar(x)))
                })
            }
            StringLiteral(s) => {
                let s: Arc<str> = Arc::from(s.as_ref());
                Ok(new_dyn_evaluator(move |_| {
                    Ok(Value::Str(Str::new_arc(Arc::clone(&s))))
                }))
            }
            expr @ EnumConstant(..) => cannot_handle(expr),
            SizeofType(typ) => {
                let size = Ok(Value::U64Scalar(typ.size(abi)?));
                Ok(new_dyn_evaluator(move |_| size.clone()))
            }
            SizeofExpr(expr) => {
                let typ = expr.typ(cenv)?;
                recurse(SizeofType(typ))
            }
            Cast(typ, expr) => {
                let expr = *expr;
                let typ = typ.decay_to_ptr();
                let expr_typ = expr.typ(cenv)?.decay_to_ptr();

                let expr_typ: &Type = expr_typ.resolve_wrapper();
                let typ: &Type = typ.resolve_wrapper();

                match (typ, expr) {
                    (typ, expr) if typ == expr_typ => recurse(expr),
                    // Chains of cast (T1*)(T2*)...x is equivalent to (T1*)x
                    (Type::Pointer(_), Cast(Type::Pointer(_), expr)) => {
                        recurse(Cast(typ.clone(), expr))
                    }
                    (typ, expr) => match (typ, expr_typ) {
                        (
                            Type::Pointer(typ),
                            Type::Pointer(expr_typ) | Type::Array(expr_typ, _),
                        ) => {
                            let typ: &Type = typ.deref().resolve_wrapper();
                            let expr_typ = expr_typ.resolve_wrapper();
                            match (expr_typ, typ) {
                                (expr_typ, typ) if typ == expr_typ => recurse(expr),
                                // (void *)(T *)x is treated the same as (T *)x
                                (_, Type::Void) => recurse(expr),

                                // For integer types:
                                // T x;
                                // (T2*)&x == &(T2)x
                                // Note that this is only well defined if T2 is char in
                                // first approximation
                                (
                                    Type::Bool
                                    | Type::U8
                                    | Type::I8
                                    | Type::U16
                                    | Type::I16
                                    | Type::U32
                                    | Type::I32
                                    | Type::I64
                                    | Type::U64,
                                    typ,
                                ) if typ == &Type::Bool || typ == &Type::U8 || typ == &Type::I8 => {
                                    recurse(Addr(Box::new(Cast(
                                        typ.clone(),
                                        Box::new(Deref(Box::new(expr))),
                                    ))))
                                }
                                (expr_typ, typ) => Err(CompileError::IncompatiblePointerCast(
                                    expr_typ.clone(),
                                    typ.clone(),
                                )),
                            }
                        }
                        (typ, _expr_typ) => {
                            // Convert potential pointers to an integer type for the
                            // sake of value conversion.
                            let typ = typ.to_arith(abi)?;
                            let conv = convert_arith(&typ)?;
                            let eval = recurse(expr)?;
                            Ok(new_dyn_evaluator(move |x| conv(eval.eval(x)?)))
                        }
                    },
                }
            }
            Plus(expr) => {
                let expr = *expr;
                let (_typ, conv) = convert_arith_op(cenv, &expr)?;
                let eval = recurse(expr)?;

                Ok(new_dyn_evaluator(move |x| conv(eval.eval(x)?)))
            }
            Minus(expr) => {
                let (_typ, conv) = convert_arith_op(cenv, &expr)?;

                macro_rules! negate {
                    ($value:expr) => {
                        match $value {
                            Value::I64Scalar(x) => conv(Value::I64Scalar(-x)),
                            Value::U64Scalar(x) => conv(Value::I64Scalar(-(x as i64))),
                            val => Err(EvalError::IllegalType(val.into_static().ok())),
                        }
                    };
                }

                let eval = recurse(*expr)?;
                match eval.eval(cenv) {
                    Err(_) => Ok(new_dyn_evaluator(move |env| {
                        let value = eval.eval(env)?;
                        negate!(value)
                    })),
                    Ok(value) => {
                        let value = negate!(value);
                        Ok(new_dyn_evaluator(move |_| value.clone()))
                    }
                }
            }
            Tilde(expr) => {
                let expr = *expr;
                let (typ, _conv) = convert_arith_op(cenv, &expr)?;
                let eval = recurse(expr)?;

                macro_rules! complement {
                    ($unsigned:ty, $signed:ty) => {
                        Ok(new_dyn_evaluator(move |env| match eval.eval(env)? {
                            Value::I64Scalar(x) => Ok(Value::I64Scalar((!(x as $signed)) as i64)),
                            Value::U64Scalar(x) => Ok(Value::U64Scalar((!(x as $unsigned)) as u64)),
                            val => Err(EvalError::IllegalType(val.into_static().ok())),
                        }))
                    };
                }

                use Type::*;
                match typ {
                    Bool => complement!(u8, i8),
                    U8 | I8 => complement!(u8, i8),
                    U16 | I16 => complement!(u16, i16),
                    U32 | I32 => complement!(u32, i32),
                    U64 | I64 => complement!(u64, i64),
                    _ => Err(CompileError::NonArithmeticOperand(typ)),
                }
            }
            Bang(expr) => {
                let eval = recurse(*expr)?;

                Ok(new_dyn_evaluator(move |env| match eval.eval(env)? {
                    Value::U64Scalar(x) => Ok(Value::I64Scalar((x == 0).into())),
                    Value::I64Scalar(x) => Ok(Value::I64Scalar((x == 0).into())),
                    val => Err(EvalError::IllegalType(val.into_static().ok())),
                }))
            }

            Addr(expr) => {
                let eval = recurse(*expr)?;
                Ok(new_dyn_evaluator(move |env| {
                    let val = eval.eval(env)?;
                    let val = ScratchBox::Owned(OwnedScratchBox::new_in(val, env.scratch()));
                    Ok(Value::Addr(val))
                }))
            }
            Deref(expr) => recurse(Subscript(expr, Box::new(IntConstant(Type::I32, 0)))),

            // Since there can be sequence points inside an expression in a number
            // of ways, we would need a mutable environment to keep track of it, so
            // ignore it for now as this does not seem to be used in current
            // kernels.
            // https://port70.net/~nsz/c/c11/n1570.html#C
            expr @ (PostInc(_) | PostDec(_) | PreInc(_) | PreDec(_)) => cannot_handle(expr),
            expr @ Assign(..) => cannot_handle(expr),

            Ternary(cond, lhs, rhs) => {
                let lhs = *lhs;
                let rhs = *rhs;

                let eval_cond = recurse(*cond)?;
                let eval_lhs = recurse(lhs.clone())?;
                let eval_rhs = recurse(rhs.clone())?;

                let lhs_typ = lhs.typ(cenv)?;
                let rhs_typ = rhs.typ(cenv)?;

                let lhs_info = lhs_typ.arith_info();
                let rhs_info = rhs_typ.arith_info();

                match (lhs_info, rhs_info) {
                    (Some(_), Some(_)) => {
                        let (_, conv_lhs, conv_rhs) = convert_arith_ops(abi, lhs_typ, rhs_typ)?;
                        Ok(new_dyn_evaluator(move |env| match eval_cond.eval(env)? {
                            Value::U64Scalar(0) | Value::I64Scalar(0) => {
                                conv_rhs(eval_rhs.eval(env)?)
                            }
                            _ => conv_lhs(eval_lhs.eval(env)?),
                        }))
                    }
                    _ => Ok(new_dyn_evaluator(move |env| match eval_cond.eval(env)? {
                        Value::U64Scalar(0) | Value::I64Scalar(0) => eval_rhs.eval(env),
                        _ => eval_lhs.eval(env),
                    })),
                }
            }
            CommaExpr(mut exprs) => recurse(exprs.pop().unwrap()),

            Subscript(expr, idx) => {
                let expr = *expr;
                let eval_idx = recurse(*idx)?;
                let eval_expr = recurse(expr.clone())?;

                match eval_idx.eval(cenv) {
                    // If we access element 0 at compile time, that is simply
                    // dereferencing the value as a pointer.
                    Ok(Value::U64Scalar(0) | Value::I64Scalar(0)) => {
                        let expr_typ = expr.typ(cenv)?;
                        let expr_typ = expr_typ.resolve_wrapper();

                        match expr_typ {
                            // Dereferencing a pointer to array gives the array,
                            // which in most contexts will behave like the
                            // address to the array when manipulated except when
                            // used with & (address of first array element) and
                            // sizeof operators (number of elements in the
                            // array). Dereferencing such address will work with
                            // this interpreter, and the sizeof() implementation
                            // is done by inspecting the type only.
                            Type::Pointer(inner) if matches!(inner.deref(), Type::Array(..)) => {
                                recurse(Expr::Cast(inner.deref().clone(), Box::new(expr)))
                            }
                            Type::Pointer(typ) | Type::Array(typ, ..) => {
                                // We might need the conversion as it is legal to cast e.g. an int* to a char*
                                let conv = convert_arith(typ).unwrap_or(Box::new(|x| Ok(x)));
                                Ok(new_dyn_evaluator(move |env| {
                                    conv(eval_expr.eval(env)?.get(env, 0)?)
                                }))
                            }
                            _ => cannot_handle(expr),
                        }
                    }
                    _ => Ok(new_dyn_evaluator(move |env| {
                        let idx: u64 = match eval_idx.eval(env)? {
                            Value::U64Scalar(x) => Ok(x),
                            Value::I64Scalar(x) => Ok(x as u64),
                            val => Err(EvalError::IllegalType(val.into_static().ok())),
                        }?;
                        let idx: usize = idx.try_into().unwrap();
                        eval_expr.eval(env)?.get(env, idx)
                    })),
                }
            }

            Mul(lhs, rhs) => binop!(None, lhs, rhs, |x, y| x * y),
            Div(lhs, rhs) => binop!(None, lhs, rhs, |x, y| x / y),
            Mod(lhs, rhs) => binop!(None, lhs, rhs, |x, y| x % y),
            Add(lhs, rhs) => binop!(Some(Add), lhs, rhs, |x, y| x + y),
            Sub(lhs, rhs) => binop!(Some(Sub), lhs, rhs, |x, y| x - y),

            BitAnd(lhs, rhs) => binop!(None, lhs, rhs, |x, y| x & y),
            BitOr(lhs, rhs) => binop!(None, lhs, rhs, |x, y| x | y),
            BitXor(lhs, rhs) => binop!(None, lhs, rhs, |x, y| x ^ y),

            Eq(lhs, rhs) => comp!(lhs, rhs, |x, y| (x == y).into()),
            NEq(lhs, rhs) => comp!(lhs, rhs, |x, y| (x != y).into()),
            LoEq(lhs, rhs) => comp!(lhs, rhs, |x, y| (x <= y).into()),
            HiEq(lhs, rhs) => comp!(lhs, rhs, |x, y| (x >= y).into()),
            Lo(lhs, rhs) => comp!(lhs, rhs, |x, y| (x < y).into()),
            Hi(lhs, rhs) => comp!(lhs, rhs, |x, y| (x > y).into()),

            And(lhs, rhs) => comp!(lhs, rhs, |x, y| ((x != 0) && (y != 0)).into()),
            Or(lhs, rhs) => comp!(lhs, rhs, |x, y| ((x != 0) || (y != 0)).into()),

            LShift(lhs, rhs) => shift!(lhs, rhs, |x, y| x << y),
            RShift(lhs, rhs) => shift!(lhs, rhs, |x, y| x >> y),

            ExtensionMacro(desc) => {
                let kind = &desc.kind;
                match kind {
                    ExtensionMacroKind::ObjectLike { value, .. } => {
                        let value = value.clone();
                        Ok(new_dyn_evaluator(move |_env| Ok(value.clone())))
                    }
                    // We cannot do anything with a bare function-like macro, it has
                    // to be applied to an expression, at which point the parser
                    // gives us a ExtensionMacroCall.
                    ExtensionMacroKind::FunctionLike { .. } => cannot_handle(ExtensionMacro(desc)),
                }
            }
            ExtensionMacroCall(call) => (call.compiler.compiler)(cenv),

            expr @ FuncCall(..) => cannot_handle(expr),
        }?;

        // Compile-time evaluation, if that succeeds we simply replace the evaluator
        // by a closure that clones the precomputed value.
        match eval.eval(cenv) {
            Ok(value) => match value.into_static() {
                Err(_) => Ok(eval),
                Ok(value) => Ok(new_dyn_evaluator(move |_| Ok(value.clone()))),
            },
            Err(_err) => Ok(eval),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use nom::AsBytes;

    use super::*;
    use crate::{
        cparser::{CGrammar, CGrammarCtx, DynamicKind},
        grammar::PackratGrammar as _,
        parser::tests::{run_parser, zero_copy_to_str},
    };

    #[derive(Clone, Copy)]
    enum Stage {
        Compile,
        Run,
    }

    struct TestEnv {
        abi: Abi,
        scratch: ScratchAlloc,
        stage: Arc<Mutex<Stage>>,
    }

    impl ParseEnv for TestEnv {
        fn abi(&self) -> &Abi {
            &self.abi
        }

        #[inline]
        fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
            match id {
                "runtime_u32_field" => Ok(Type::U32),
                "runtime_zero_field" => Ok(Type::U32),
                "u32_field" => Ok(Type::U32),
                "u32_array_field" => Ok(Type::Array(Box::new(Type::U32), ArrayKind::Fixed(Ok(2)))),
                "u32_dynarray_field" => Ok(Type::Array(
                    Box::new(Type::U32),
                    ArrayKind::Dynamic(DynamicKind::Dynamic),
                )),
                "str_field" => Ok(Type::Pointer(Box::new(Type::U8))),

                "string_field" => Ok(Type::Pointer(Box::new(Type::U8))),
                "owned_string_field" => Ok(Type::Pointer(Box::new(Type::U8))),
                "runtime_u32_ptr" => Ok(Type::Pointer(Box::new(Type::U32))),
                "runtime_void_ptr" => Ok(Type::Pointer(Box::new(Type::Void))),
                "runtime_char_ptr" => Ok(Type::Pointer(Box::new(Type::U8))),
                id => Err(CompileError::UnknownField(id.into())),
            }
        }
    }

    impl<'ce> CompileEnv<'ce> for TestEnv {
        #[inline]
        fn field_getter(&self, id: &str) -> Result<Box<dyn Evaluator>, CompileError> {
            match id {
                "runtime_u32_field" => {
                    let stage = Arc::clone(&self.stage);
                    Ok(new_dyn_evaluator(move |_env| {
                        match *stage.lock().unwrap() {
                            Stage::Compile => Err(EvalError::NoEventData),
                            Stage::Run => Ok(Value::U64Scalar(44)),
                        }
                    }))
                }
                "runtime_zero_field" => {
                    let stage = Arc::clone(&self.stage);
                    Ok(new_dyn_evaluator(move |_env| {
                        match *stage.lock().unwrap() {
                            Stage::Compile => Err(EvalError::NoEventData),
                            Stage::Run => Ok(Value::U64Scalar(0)),
                        }
                    }))
                }
                "u32_field" => Ok(new_dyn_evaluator(|_| Ok(Value::U64Scalar(42)))),
                "u32_array_field" => Ok(new_dyn_evaluator(|_| {
                    Ok(Value::U32Array([42, 43].as_ref().into()))
                })),
                "u32_dynarray_field" => Ok(new_dyn_evaluator(|_| {
                    Ok(Value::U32Array([42, 43].as_ref().into()))
                })),
                "str_field" => Ok(new_dyn_evaluator(|_| {
                    Ok(Value::Str(Str::new_owned("hello world".into())))
                })),

                "string_field" => {
                    let s = Arc::from("foobar");
                    Ok(new_dyn_evaluator(move |_| {
                        Ok(Value::Str(Str::new_arc(Arc::clone(&s))))
                    }))
                }
                "owned_string_field" => {
                    let array = Value::Str(Str::new_owned(("foobar").into()));
                    Ok(new_dyn_evaluator(move |_| Ok(array.clone())))
                }
                "runtime_u32_ptr" | "runtime_void_ptr" | "runtime_char_ptr" => {
                    Ok(new_dyn_evaluator(|_| Ok(Value::U64Scalar(42))))
                }
                id => Err(CompileError::UnknownField(id.into())),
            }
        }
    }

    impl<'ee> EvalEnv<'ee> for TestEnv {
        // #[inline]
        // fn field_getter<EE: EvalEnv>(&self, id: &str) -> Result<Box<dyn Fn(&EE) -> Result<Value, EvalError>>, CompileError> {
        //     Ok(Box::new(|_| Ok(Value::U64Scalar(42))))
        // }

        fn header(&self) -> Result<&Header, EvalError> {
            Err(EvalError::NoHeader)
        }

        fn deref_static(&self, addr: Address) -> Result<Value<'_>, EvalError> {
            match addr {
                42 => Ok(Value::Str(Str::new_borrowed("hello world"))),
                43 => Ok(Value::U64Scalar(105)),
                44 => Ok(Value::U64Scalar(257)),
                45 => Ok(Value::U64Scalar(44)),
                46 => Ok(Value::U64Scalar(45)),
                47 => Ok(Value::U32Array(Array::Borrowed(&[30, 31, 32, 33]))),
                addr => Err(EvalError::CannotDeref(addr)),
            }
        }

        fn event_data(&self) -> Result<&[u8], EvalError> {
            Err(EvalError::NoEventData)
        }

        #[inline]
        fn scratch(&self) -> &ScratchAlloc {
            &self.scratch
        }
    }

    #[test]
    fn interp_test() {
        fn test(src: &[u8], expected: Value<'_>) {
            let stage = Arc::new(Mutex::new(Stage::Compile));
            let abi = Abi {
                long_size: LongSize::Bits64,
                endianness: Endianness::Little,
                char_signedness: Signedness::Unsigned,
            };
            let env = TestEnv {
                scratch: ScratchAlloc::new(),
                stage: Arc::clone(&stage),
                abi,
            };
            let parser = CGrammar::expr();
            let ctx = CGrammarCtx::new(&env);
            let input = CGrammar::make_span(src, &ctx);
            let ast = run_parser(input.clone(), parser);
            let input = zero_copy_to_str(input.as_bytes());
            let compiled = ast
                .compile(&env)
                .unwrap_or_else(|err| panic!("Error while compiling {input:?}: {err}"));

            *stage.lock().unwrap() = Stage::Run;

            let expr = compiled
                .eval(&env)
                .unwrap_or_else(|err| panic!("Error while interpreting {input:?}: {err}"));
            assert_eq!(expr, expected, "while interpreting {input:}")
        }

        fn signed(x: i64) -> Value<'static> {
            Value::I64Scalar(x)
        }
        fn unsigned(x: u64) -> Value<'static> {
            Value::U64Scalar(x)
        }
        fn addr(x: Value<'static>) -> Value<'static> {
            Value::Addr(ScratchBox::Arc(Arc::new(x)))
        }

        let hello_world = Value::Str(Str::new_arc(Arc::from("hello world")));

        // Literals
        test(b"0", signed(0));
        test(b"1", signed(1));
        test(b"-1", signed(-1));
        test(b"-1u", unsigned(4294967295));
        test(b"-(1u)", unsigned(4294967295));
        test(b"-(-(-(1u)))", unsigned(4294967295));
        test(b"-(-(1u))", unsigned(1));
        test(b"-(-(1UL))", unsigned(1));

        test(br#""hello world""#, hello_world.clone());

        test(b"true", signed(1));
        test(b"false", signed(0));

        // Basic arithmetic
        test(b"(-1)", signed(-1));
        test(b"1+2", signed(3));
        test(b"1u+2u", unsigned(3));
        test(b"1+2u", unsigned(3));
        test(b"1u+2", unsigned(3));
        test(b"(uint16_t)1u+(s32)2", unsigned(3));

        test(b"-1+2", signed(1));
        test(b"(-1)+2", signed(1));
        test(b"2+(-1)", signed(1));
        test(b"2+((-1)*4)", signed(-2));
        test(b"(-1)*4", signed(-4));
        test(b"1+TASK_COMM_LEN", signed(17));
        test(b"-TASK_COMM_LEN + 1", signed(-15));
        test(b"-(s32)TASK_COMM_LEN + 1", signed(-15));

        test(b"-1-2", signed(-3));
        test(b"1-TASK_COMM_LEN", signed(-15));
        test(b"-TASK_COMM_LEN - 1", signed(-17));

        test(b"-TASK_COMM_LEN - 1u", unsigned(4294967279));

        test(b"10 % 2", signed(0));
        test(b"11 % 2", signed(1));
        test(b"-11 % 2", signed(-1));
        test(b"11 % -2", signed(1));
        test(b"-11 % -2", signed(-1));
        test(b"((s64)(-11)) % ((s16)(-2))", signed(-1));

        test(b"42 + ((-1) * 4)", signed(42 - 4));
        test(b"((  signed long)42) + ((-1) * 4)", signed(42 - 4));
        test(b"((unsigned long)42) - 4", unsigned(42 - 4));
        test(
            b"((unsigned long)42) + ((unsigned long)-4)",
            unsigned(42 - 4),
        );
        test(b"((unsigned long)42) + (0-4)", unsigned(42 - 4));
        test(b"((unsigned long)42) + (-4)", unsigned(42 - 4));
        test(b"((unsigned long)42) + ((-1) * 4)", unsigned(42 - 4));

        // Pointer arithmetic
        test(
            b"(unsigned int*)(((unsigned long)(unsigned int*)42) -4)",
            unsigned(42 - 4),
        );
        test(
            b"(unsigned int*)(((unsigned long)(unsigned int*)42) + (-4))",
            unsigned(42 - 4),
        );
        test(
            b"((unsigned long)(unsigned int*)42) + ((-1) * 4)",
            unsigned(42 - 4),
        );
        test(
            b"(unsigned int*)(((unsigned long)(unsigned int*)42) + ((-1) * 4))",
            unsigned(42 - 4),
        );

        test(b"REC->runtime_u32_ptr", unsigned(42));
        test(b"REC->runtime_u32_ptr + 1", unsigned(42 + 4));
        test(b"REC->runtime_u32_ptr - 1", unsigned(42 - 4));
        test(b"REC->runtime_u32_ptr + (-1)", unsigned(42 - 4));
        test(b"1 + REC->runtime_u32_ptr", unsigned(42 + 4));
        test(b"(-1) + 42", signed(42 - 1));
        test(b"(-1) + REC->runtime_u32_ptr", unsigned(42 - 4));

        test(b"REC->runtime_void_ptr", unsigned(42));
        test(b"REC->runtime_void_ptr + 1", unsigned(42 + 1));
        test(b"REC->runtime_void_ptr - 1", unsigned(42 - 1));
        test(b"REC->runtime_void_ptr + (-1)", unsigned(42 - 1));
        test(b"1 + REC->runtime_void_ptr", unsigned(42 + 1));
        test(b"(-1) + REC->runtime_void_ptr", unsigned(42 - 1));

        test(b"REC->runtime_char_ptr", unsigned(42));
        test(b"REC->runtime_char_ptr + 1", unsigned(42 + 1));
        test(b"REC->runtime_char_ptr - 1", unsigned(42 - 1));
        test(b"REC->runtime_char_ptr + (-1)", unsigned(42 - 1));
        test(b"1 + REC->runtime_char_ptr", unsigned(42 + 1));
        test(b"(-1) + REC->runtime_char_ptr", unsigned(42 - 1));

        // Integer overflow
        test(b"1 == 1", signed(1));
        test(b"1 == 2", signed(0));
        test(b"-1 == 4294967295", signed(0));
        test(b"-1u == 4294967295", signed(1));
        test(b"-1 == 4294967295u", signed(1));
        test(b"-1u == 4294967295u", signed(1));
        test(b"(u64)-1u == (unsigned int)4294967295u", signed(1));

        // Comparisons
        test(b"1 > 2", signed(0));
        test(b"2 > 1", signed(1));
        test(b"1 > -1u", signed(0));
        test(b"-1u > 1", signed(1));
        test(b"-1u < 1", signed(0));
        test(b"(u32)-1u > (s32)1", signed(1));

        // Shifts
        test(b"2 >> 1", signed(1));
        test(b"-2 >> 1", signed(-1));
        test(b"2 << 1", signed(4));
        test(b"-2 << 1", signed(-4));
        test(b"(s8)-2 << (s64)1", signed(-4));
        test(b"(s8)-2 << (u64)1", signed(-4));

        // Bitwise not
        test(b"~0", signed(-1));
        test(b"~0u", unsigned(4294967295));
        test(b"~(u8)0u", unsigned(4294967295));
        test(b"~((u32)0)", unsigned(4294967295));
        test(b"~0 == -1", signed(1));
        test(b"(s8)~0 == -1", signed(1));
        test(b"(u32)~0 == -1u", signed(1));
        test(b"(u64)~0 == -1ull", signed(1));

        // Logical not
        test(b"!0", signed(1));
        test(b"!1", signed(0));
        test(b"!42", signed(0));
        test(b"!(s32)42", signed(0));
        test(b"!(u32)42", signed(0));

        // Logical or
        test(b"1 && 2", signed(1));

        // Ternary
        test(b"1 ? 1 : 0", signed(1));
        test(b"0 ? 1 : 0", signed(0));
        test(b"0 ? 1 : 0u", unsigned(0));
        test(b"-12 ? 42u : 0", unsigned(42));
        test(b"(s32)-12 ? (u8)42 : 0", unsigned(42));
        test(b"1 ? (int *)42 : (int *)43", unsigned(42));
        test(b"1 ? (int *)42 : (void *)43", unsigned(42));
        test(b"1 ? (void *)42 : (int *)43", unsigned(42));
        test(b"1 ? (void *)42 : (void *)43", unsigned(42));
        test(b"1 ? (int *)42 : (int *)0", unsigned(42));
        test(b"1 ? (int *)42 : (void *)0", unsigned(42));
        test(b"1 ? (void *)42 : (int *)0", unsigned(42));
        test(b"1 ? (void *)42 : (void *)0", unsigned(42));
        test(b"1 ? (int *)0 : (int *)0", unsigned(0));
        test(b"1 ? (int *)0 : (void *)0", unsigned(0));
        test(b"1 ? (void *)0 : (int *)0", unsigned(0));
        test(b"1 ? (void *)0 : (void *)0", unsigned(0));
        test(b"1 ? (int *)0 : (int *)42", unsigned(0));
        test(b"1 ? (int *)0 : (void *)42", unsigned(0));
        test(b"1 ? (void *)0 : (int *)42", unsigned(0));
        test(b"1 ? (void *)0 : (void *)42", unsigned(0));

        // Casts
        test(b"(int)0", signed(0));
        test(b"(s8)0", signed(0));
        test(b"(unsigned int)0", unsigned(0));
        test(b"(u32)0", unsigned(0));
        test(b"(unsigned int)-1", unsigned(4294967295));
        test(b"(u32)-1", unsigned(4294967295));
        test(b"(unsigned int)(unsigned char)-1", unsigned(255));
        test(b"(u32)(u8)-1", unsigned(255));
        test(b"(int)(unsigned int)-1", signed(-1));
        test(b"(s32)(u64)-1", signed(-1));
        test(b"(int)4294967295", signed(-1));
        test(b"(s32)4294967295", signed(-1));
        test(b"(int*)&1", addr(Value::I64Scalar(1)));
        test(b"(s32*)&1", addr(Value::I64Scalar(1)));
        test(b"(int*)1", unsigned(1));
        test(b"(s16*)1", unsigned(1));
        test(b"(void *)1ull", unsigned(1));
        test(b"((__u16)(__le16)1)", unsigned(1));

        // Sizeof type
        test(b"sizeof(char)", unsigned(1));
        test(b"sizeof(int)", unsigned(4));
        test(b"sizeof(unsigned long)", unsigned(8));
        test(b"sizeof(int *)", unsigned(8));
        test(b"sizeof(u8)", unsigned(1));
        test(b"sizeof(u64)", unsigned(8));
        test(b"sizeof(u8 *)", unsigned(8));
        test(b"sizeof(typeof(1))", unsigned(4));
        test(b"sizeof(typeof(1ull))", unsigned(8));
        test(b"sizeof(typeof(1) *)", unsigned(8));
        test(b"sizeof(REC->u32_field)", unsigned(4));
        test(b"sizeof(typeof(REC->u32_field))", unsigned(4));

        // Sizeof expr
        test(b"sizeof(1)", unsigned(4));
        test(b"sizeof 1", unsigned(4));
        test(b"sizeof(1l)", unsigned(8));
        test(b"sizeof((long)1)", unsigned(8));
        test(b"sizeof((u64)1)", unsigned(8));
        test(b"sizeof(&1)", unsigned(8));
        test(b"sizeof((u8)&1)", unsigned(1));
        test(b"sizeof(*(unsigned int (*)[10])50)", unsigned(40));

        // Address and deref
        test(b"&1", addr(signed(1)));
        test(b"(void *)&1", addr(signed(1)));
        test(b"*(void *)&1", signed(1));
        test(b"*(u8*)(void *)&1", unsigned(1));
        test(b"*(u8*)(void *)&257", unsigned(1));
        test(b"*(u8*)(void *)&257ull", unsigned(1));
        test(b"*(u64*)(void*)(u64*)(void *)&1ull", unsigned(1));
        test(b"*(u64*)(u8*)(void*)(u64*)(void *)&1ull", unsigned(1));
        test(b"*(u32*)(u8*)(void*)(u32*)(void *)&1u", unsigned(1));
        test(b"&(u32)1", addr(unsigned(1)));
        test(b"&REC->runtime_u32_field", addr(unsigned(44)));
        test(b"*(unsigned int *)REC->runtime_u32_field", unsigned(257));
        test(b"*(u32 *)REC->runtime_u32_field", unsigned(257));
        test(b"*(signed int *)REC->runtime_u32_field", signed(257));
        test(b"*(s32 *)REC->runtime_u32_field", signed(257));
        test(b"*(unsigned int *)&REC->runtime_u32_field", unsigned(44));
        test(b"*(u32 *)&REC->runtime_u32_field", unsigned(44));
        test(b"(signed int)*&REC->runtime_u32_field", signed(44));
        test(b"(s32)*&REC->runtime_u32_field", signed(44));
        test(b"*&1", signed(1));
        test(b"*&*&1", signed(1));
        test(b"(s32)*&*&1", signed(1));
        test(b"*(&1)", signed(1));
        test(b"*(2, &1)", signed(1));
        test(b"*(0 ? &1 : &2)", signed(2));
        test(b"*(1 ? &1 : &2)", signed(1));
        test(b"*(1 ? &(s32)1 : &(s32)2)", signed(1));
        test(b"*(1 ? &(s32)1 : &(int)2)", signed(1));

        test(b"*(char *)42", unsigned(104));
        test(b"*(u8 *)42", unsigned(104));
        test(b"*(unsigned char *)42", unsigned(104));
        test(b"*(signed char *)42", signed(104));
        test(b"*(s8 *)42", signed(104));
        test(b"*(unsigned long *)43", unsigned(105));
        test(b"*(u64 *)43", unsigned(105));
        test(b"*(char *)44", unsigned(1));
        test(b"*(u8 *)44", unsigned(1));
        test(b"*(char *)(int *)(short *)44", unsigned(1));
        test(b"*(char *)(s32 *)(short *)44", unsigned(1));
        test(b"*(u8 *)(int *)(short *)44", unsigned(1));
        test(b"*(u8 *)(s32 *)(short *)44", unsigned(1));
        test(b"*(u8 *)(s32 *)(s16 *)44", unsigned(1));
        test(b"*(char *)(int *)(s16 *)44", unsigned(1));
        test(b"*(u8 *)(int *)(s16 *)44", unsigned(1));

        test(b"((char *)42)[0]", unsigned(104));
        test(b"((u8 *)42)[0]", unsigned(104));
        test(b"((char *)42)[1]", unsigned(101));
        test(b"((u8 *)42)[1]", unsigned(101));
        test(b"*(int *)44", signed(257));
        test(b"*(s32 *)44", signed(257));
        test(b"*(unsigned int *)44", unsigned(257));
        test(b"*(u32 *)44", unsigned(257));
        test(b"*(unsigned int *)47", unsigned(30));
        test(b"( *(unsigned int (*)[10])47 )[1]", unsigned(31));
        test(b"( *(u32 (*)[10])47 )[1]", unsigned(31));
        test(b"((u32 *)47)[1]", unsigned(31));

        test(b"**(unsigned int **)45", unsigned(257));
        test(b"**(u32 **)45", unsigned(257));
        test(b"( *(unsigned int * (*)[10])44 )[0]", unsigned(257));
        test(b"( *(u32 * (*)[10])44 )[0]", unsigned(257));

        // Array
        test(b"(&1)[0]", signed(1));
        test(b"((s32*)&1)[0]", signed(1));
        test(b"(42 ? &1 : &2)[0]", signed(1));
        test(b"(42 ? (s8*)&1 : (s8*)&2)[0]", signed(1));
        test(b"(0 ? &1 : &2)[0]", signed(2));
        test(b"(0 ? (s8*)&1 : (signed char*)&2)[0]", signed(2));
        test(b"(REC->runtime_zero_field ? &1 : &2)[0]", signed(2));
        test(b"((s8)REC->runtime_zero_field ? &1 : &2)[0]", signed(2));

        // Field access
        test(b"REC->u32_field", unsigned(42));
        test(b"(u64)REC->u32_field", unsigned(42));
        test(b"(*&REC) -> u32_field", unsigned(42));
        test(b"(*(0 ? &(REC) : &(REC))) -> u32_field", unsigned(42));
        test(b"(*(1 ? &(REC) : &(REC))) -> u32_field", unsigned(42));

        test(b"sizeof(REC->u32_array_field)", unsigned(4 * 2));
        test(b"sizeof((int [2])REC->u32_array_field)", unsigned(4 * 2));
        test(b"sizeof((u8 [2])REC->u32_array_field)", unsigned(2));
        test(b"REC->u32_array_field[0]", unsigned(42));
        test(b"*REC->u32_array_field", unsigned(42));

        test(b"REC->u32_dynarray_field[0]", unsigned(42));
        test(b"((u32 *)REC->u32_dynarray_field)[0]", unsigned(42));
        test(b"REC->u32_dynarray_field[1]", unsigned(43));
        test(b"((u32 *)REC->u32_dynarray_field)[1]", unsigned(43));
        test(b"*REC->u32_dynarray_field", unsigned(42));
        test(b"*(u32*)REC->u32_dynarray_field", unsigned(42));

        test(b"*REC->owned_string_field", unsigned(102));
        test(b"REC->owned_string_field[6]", unsigned(0));
        test(b"((char *)REC->owned_string_field)[6]", unsigned(0));
        test(b"REC->str_field", hello_world.clone());
        test(b"(char *)REC->str_field", hello_world.clone());
        // Unfortunately, it is not easy to preserve the Array value
        // across a &* chain, as this would either necessitate to not simplify
        // the *& chains in the sub-expression, or be very brittle and strictly
        // match &* with nothing in-between which is quite useless.
        //
        // So we end up with dereferencing the Array, which provides its
        // first item, and then we take the address of that.
        test(b"*&*REC->str_field", unsigned(104));
        test(b"*(u8*)&*REC->str_field", unsigned(104));
        test(b"*REC->str_field", unsigned(104));
        test(b"*(u8*)REC->str_field", unsigned(104));
        test(b"REC->str_field[0]", unsigned(104));
        test(b"((u8*)REC->str_field)[0]", unsigned(104));
        test(b"REC->str_field[1]", unsigned(101));
        test(b"((u8*)REC->str_field)[1]", unsigned(101));
        test(b"REC->str_field[6]", unsigned(119));
        test(b"((u8*)REC->str_field)[6]", unsigned(119));
        test(b"REC->str_field[11]", unsigned(0));
        test(b"((u8*)REC->str_field)[11]", unsigned(0));

        test(b"*(signed char*)REC->str_field", signed(104));
        test(b"*(s8*)REC->str_field", signed(104));
        test(b"(int)*REC->str_field", signed(104));
        test(b"(s32)*REC->str_field", signed(104));
        test(b"(int)REC->str_field[0]", signed(104));
        test(b"(s32)REC->str_field[0]", signed(104));

        // Combined
        test(b"(65536/((1UL) << 12) + 1)", unsigned(17));
        test(b"(65536/((1UL) << 12) + (s32)1)", unsigned(17));

        test(b"*(int*)(&(-1))", signed(-1));
        test(b"*(s32*)((s32 *)&(-1))", signed(-1));
        test(b"*(unsigned int*)(&-1u)", unsigned(4294967295));
        test(b"*(u32 *)(&-1u)", unsigned(4294967295));
        test(b"*(unsigned int*)(char*)(&-1u)", unsigned(4294967295));
        test(b"*(u32 *)(u8 *)(&-1u)", unsigned(4294967295));
        test(b"*(unsigned int *)(u8 *)(&-1u)", unsigned(4294967295));
        // This is not UB since any value can be accessed via a char pointer:
        // https://port70.net/~nsz/c/c11/n1570.html#6.5p7
        test(b"*(unsigned char*)(int*)(&(-1))", unsigned(255));
        test(b"*(u8 *)(int*)(&(-1))", unsigned(255));

        test(b"(int*)1 == (int*)1", signed(1));
        test(b"(s32*)1 == (s32*)1", signed(1));
        test(b"(int*)1 == (char*)1", signed(1));
        test(b"(s32*)1 == (u8*)1", signed(1));

        test(b"(int)(int*)1 == 1", signed(1));
        test(b"(s32)(s32*)1 == 1", signed(1));
        test(b"(int)(s32*)1 == 1", signed(1));
        test(b"(s32)(int*)1 == 1", signed(1));
        test(b"(int)(int*)1 == 2", signed(0));
        test(b"(s32)(int*)1 == 2", signed(0));
        test(b"(int)(s32*)1 == 2", signed(0));
        test(b"(s32)(s32*)1 == 2", signed(0));
        test(b"(char)(int*)256 == 0", signed(1));
        test(b"(u8)(s32*)256 == 0", signed(1));
        test(b"(signed char)(s32*)256 == 0", signed(1));

        test(
            b"*((char)(int*)256 == 0 ? (&42, &43) : &2) == 43",
            signed(1),
        );
        test(
            b"*((u8)(s32*)256 == 0 ? ((s32*)&42, &43) : &(s32)2) == (s64)43",
            signed(1),
        );

        test(b"1 ? '*' : ' '", signed(42));
        test(b"(1 && 2) ? '*' : ' '", signed(42));
        test(b"1 && 2 ? '*' : ' '", signed(42));
        test(b"(int) 1 && (int) 2 ? '*' : ' '", signed(42));

        // Extension macros
        test(b"__builtin_expect(42) + 1", signed(43));
        test(b"__builtin_constant_p(sizeof(struct page))", signed(0));
        test(br#"__builtin_constant_p("foo")"#, signed(0));
        test(br#"__builtin_constant_p("(a)")"#, signed(0));
        test(br#"__builtin_constant_p("(a")"#, signed(0));
        test(br#"__builtin_constant_p("a)")"#, signed(0));
        test(br#"__builtin_constant_p("a)\"")"#, signed(0));
        test(br#"__builtin_constant_p("\"a)")"#, signed(0));
        test(br#"__builtin_constant_p(')')"#, signed(0));
        test(br#"__builtin_constant_p('\'', "'")"#, signed(0));
        test(br#"__builtin_choose_expr(1, 42, 43)"#, signed(42));
        test(br#"__builtin_choose_expr(0, 42, 43)"#, signed(43));
        test(br#"__builtin_choose_expr(1, 42u, 43)"#, unsigned(42));
        test(br#"__builtin_choose_expr(0, 42u, 43)"#, signed(43));
        test(br#"__builtin_choose_expr(1, 42, 43u)"#, signed(42));
        test(br#"__builtin_choose_expr(0, 42, 43u)"#, unsigned(43));
        test(
            br#"sizeof(__builtin_choose_expr(0, (u8)42, (u32)43u))"#,
            unsigned(4),
        );
        test(
            br#"sizeof(__builtin_choose_expr(1, (u8)42, (u32)43u))"#,
            unsigned(1),
        );
        test(
            br#"sizeof(__builtin_choose_expr(0, (u8)42, REC->u32_field))"#,
            unsigned(4),
        );
    }
}
