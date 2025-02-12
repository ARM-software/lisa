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

//! trace.dat header parsing.
//!
//! The user-visible entry type is [`Header`].

use core::{
    borrow::Borrow,
    ffi::CStr,
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    ops::Deref as _,
    str::from_utf8,
};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    io,
    io::Error as IoError,
    ops::DerefMut as _,
    rc::Rc,
    string::String as StdString,
    sync::{Arc, RwLock},
};

use itertools::izip;
use nom::{
    branch::alt,
    bytes::complete::{is_a, is_not, tag},
    character::complete::{
        char, multispace0, multispace1, u16 as txt_u16, u32 as txt_u32, u64 as txt_u64,
    },
    combinator::{all_consuming, flat_map, iterator, map_res, opt, rest},
    error::context,
    multi::{fold_many0, many0, separated_list0},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
    Finish as _, Parser,
};
use once_cell::sync::OnceCell;
use smartstring::alias::String;

use crate::{
    array::Array,
    buffer::{Buffer, BufferError, FieldDecoder},
    cinterp::{
        new_dyn_evaluator, BasicEnv, CompileEnv, CompileError, EvalEnv, EvalError, Evaluator, Value,
    },
    closure::closure,
    compress::{Decompressor, DynDecompressor, ZlibDecompressor, ZstdDecompressor},
    cparser::{
        identifier, is_identifier, string_literal, ArrayKind, CGrammar, CGrammarCtx, Declaration,
        Expr, ExtensionMacroCall, ExtensionMacroCallCompiler, ExtensionMacroDesc, ParseEnv, Type,
    },
    error::convert_err_impl,
    grammar::PackratGrammar as _,
    io::{BorrowingCursor, BorrowingRead},
    nested_pointer::NestedPointer,
    parser::{
        error, failure, hex_u64, lexeme, map_res_cut, to_str, FromParseError, NomError,
        NomParserExt as _, VerboseParseError,
    },
    print::{PrintAtom, PrintFmtError, PrintFmtStr, PrintSpecifier, StringWriter},
    scratch::{ScratchAlloc, ScratchVec},
    str::Str,
};

/// Type alias for a memory address contained in the trace.
///
/// We cannot use [usize] since this would represent a memory address on the host running the
/// parser, which may be of a different architecture than the system that produced the trace.
pub type Address = u64;
/// Type alias for an offset in memory. This provides more helpful signatures than using [Address]
/// for everything.
pub type AddressOffset = Address;
/// Alias for the size of an object in memory.
pub type AddressSize = Address;
/// Alias for a CPU ID.
pub type Cpu = u32;
/// Alias for a process ID (PID).
pub type Pid = u32;
/// Alias for a nanosecond timestamp.
pub type Timestamp = u64;
/// Alias for an offset to a [Timestamp].
pub type TimeOffset = i64;
/// Alias for an ELF symbol name.
pub type SymbolName = String;
/// Alias for a Linux task name (also known as "comm" in various places).
pub type TaskName = String;
/// Alias for a C programming language identifier.
pub type Identifier = String;
/// Alias for an ftrace event ID.
pub type EventId = u16;

/// Alias for an offset from the beginning of a file.
pub type FileOffset = u64;
/// Alias for the size of a file.
pub type FileSize = FileOffset;

/// Alias for an offset from the beginning of memory, on the machine running the parser.
pub type MemOffset = usize;
/// Alias for the size of an object in memory, on the machine running the parser.
pub type MemSize = MemOffset;
/// Alias for the alignment of an object in memory, on the machine running the parser.
pub type MemAlign = MemOffset;

/// Alias for a trace.dat section ID
pub type SectionId = u16;

/// Encode the endianness of a piece of data.
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum Endianness {
    Big,
    Little,
}

macro_rules! parse_N {
    ($name:ident, $typ:ty) => {
        #[doc = concat!("Parse a [", stringify!($typ), "] from a buffer, following the `Self` endianness.")]
        #[inline]
        pub fn $name<'a>(&self, input: &'a [u8]) -> Result<(&'a [u8], $typ), io::Error> {
            let arr = input
                .get(..std::mem::size_of::<$typ>())
                .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?;
            let arr: [u8; std::mem::size_of::<$typ>()] = arr.try_into().unwrap();
            let x = match self {
                Endianness::Big => <$typ>::from_be_bytes(arr),
                Endianness::Little => <$typ>::from_le_bytes(arr),
            };
            let input = &input[std::mem::size_of::<$typ>()..];
            Ok((input, x))
        }
    };
}

impl Endianness {
    /// Return the native endianness of the machine running this library.
    fn native() -> Self {
        if cfg!(target_endian = "big") {
            Endianness::Big
        } else if cfg!(target_endian = "little") {
            Endianness::Little
        } else {
            panic!("Cannot handle endianness")
        }
    }

    /// Returns [true] if `Self` is the native endianness.
    pub fn is_native(&self) -> bool {
        self == &Self::native()
    }

    parse_N!(parse_u64, u64);
    parse_N!(parse_u32, u32);
    parse_N!(parse_u16, u16);
    parse_N!(parse_u8, u8);
}

/// Size of the *long* C type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LongSize {
    /// 4 bytes long
    Bits32,
    /// 8 bytes long
    Bits64,
}

/// Convert the size of the *long* type to a size in bytes.
impl From<LongSize> for u64 {
    fn from(size: LongSize) -> Self {
        match size {
            LongSize::Bits32 => 4,
            LongSize::Bits64 => 8,
        }
    }
}

/// Convert the size of the *long* type to a size in bytes.
impl From<LongSize> for usize {
    fn from(size: LongSize) -> Self {
        match size {
            LongSize::Bits32 => 4,
            LongSize::Bits64 => 8,
        }
    }
}

/// Convert the a size in bytes to a [LongSize].
impl TryFrom<usize> for LongSize {
    /// If the conversion fails, the byte size is returned.
    type Error = usize;

    fn try_from(size: usize) -> Result<Self, Self::Error> {
        match size {
            4 => Ok(LongSize::Bits32),
            8 => Ok(LongSize::Bits64),
            x => Err(x),
        }
    }
}

/// Whether a number is signed or unsigned.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Signedness {
    Signed,
    Unsigned,
}

impl Display for Signedness {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        let s = match self {
            Signedness::Signed => "signed",
            Signedness::Unsigned => "unsigned",
        };
        f.write_str(s)
    }
}

/// Encodes ABI details necessary to parse a trace.dat file.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Abi {
    /// Endianness in the kernel of the machine that generated the trace.
    pub endianness: Endianness,
    /// Long size in the kernel of the machine that generated the trace.
    pub long_size: LongSize,
    /// Whether a *char* C value is signed or unsigned in the kernel of the machine that generated
    /// the trace.
    pub char_signedness: Signedness,
}

macro_rules! abi_parse_N {
    ($name:ident, $typ:ty) => {
        #[doc = concat!("Parse a [", stringify!($typ), "] from a buffer.")]
        #[inline]
        pub fn $name<'a>(&self, input: &'a [u8]) -> Result<(&'a [u8], $typ), io::Error> {
            self.endianness.$name(input)
        }
    };
}

impl Abi {
    abi_parse_N!(parse_u64, u64);
    abi_parse_N!(parse_u32, u32);
    abi_parse_N!(parse_u16, u16);
    abi_parse_N!(parse_u8, u8);

    /// Resolve the type of a *char* to a fixed-size and fixed-signedness type according to the
    /// ABI.
    #[inline]
    pub fn char_typ(&self) -> Type {
        match self.char_signedness {
            Signedness::Unsigned => Type::U8,
            Signedness::Signed => Type::I8,
        }
    }

    /// Resolve the type of a *long* to a fixed-size type according to the
    /// ABI.
    #[inline]
    pub fn long_typ(&self) -> Type {
        match self.long_size {
            LongSize::Bits32 => Type::I32,
            LongSize::Bits64 => Type::I64,
        }
    }

    /// Resolve the type of a *unsigned long* to a fixed-size type according to the
    /// ABI.
    #[inline]
    pub fn ulong_typ(&self) -> Type {
        match self.long_size {
            LongSize::Bits32 => Type::U32,
            LongSize::Bits64 => Type::U64,
        }
    }

    /// Parse an *unsigned long* from a buffer.
    pub fn parse_ulong<'a>(&self, input: &'a [u8]) -> Result<(&'a [u8], u64), io::Error> {
        match self.long_size {
            LongSize::Bits32 => self
                .parse_u32(input)
                .map(|(remaining, x)| (remaining, x.into())),
            LongSize::Bits64 => self.parse_u64(input),
        }
    }
}

/// Basic [ParseEnv] instance that does not contain any event-specific information.
impl ParseEnv for Abi {
    #[inline]
    fn abi(&self) -> &Abi {
        self
    }

    fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
        Err(CompileError::UnknownField(id.into()))
    }
}

/// ID of a buffer in a trace.dat file.
///
/// There is typically one buffer per CPU, but extra buffer instances can exist, e.g. if the user
/// called `trace-cmd record -B mybuffer`. In that scenario, an extra buffer per CPU will be
/// created for that instance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferId {
    /// CPU ID associated to that buffer
    pub cpu: Cpu,
    /// Named of the buffer. For the implicit top-level buffer, the name is an empty string.
    pub name: String,
}

/// Location of a buffer in the trace.dat file.
#[derive(Debug, Clone)]
pub struct BufferLocation {
    pub id: BufferId,
    pub offset: FileOffset,
    pub size: FileSize,
}

/// Header for the [trace.dat v6 format](https://www.trace-cmd.org/Documentation/trace-cmd/trace-cmd.dat.v6.5.html).
#[derive(Debug, Clone)]
pub(crate) struct HeaderV6 {
    pub(crate) kernel_abi: Abi,
    pub(crate) page_size: FileSize,
    pub(crate) event_descs: Vec<EventDesc>,
    pub(crate) kallsyms: BTreeMap<Address, SymbolName>,
    pub(crate) str_table: BTreeMap<Address, String>,
    pub(crate) pid_comms: BTreeMap<Pid, TaskName>,
    pub(crate) options: Vec<Options>,
    pub(crate) top_level_buffer_locations: Vec<BufferLocation>,
    pub(crate) nr_cpus: Cpu,
}

/// Header for the [trace.dat v6 format](https://www.trace-cmd.org/Documentation/trace-cmd/trace-cmd.dat.v7.5.html).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct HeaderV7 {
    pub(crate) kernel_abi: Abi,
    pub(crate) page_size: FileSize,
    pub(crate) event_descs: Vec<EventDesc>,
    pub(crate) kallsyms: BTreeMap<Address, SymbolName>,
    pub(crate) str_table: BTreeMap<Address, String>,
    pub(crate) pid_comms: BTreeMap<Pid, TaskName>,
    pub(crate) options: Vec<Options>,
    pub(crate) nr_cpus: Cpu,
}

#[derive(Debug, Clone)]
enum VersionedHeader {
    V6(HeaderV6),
    V7(HeaderV7),
}

/// Main struct representing a trace.dat header, irrespective of the file format version in use.
#[derive(Clone)]
pub struct Header {
    // We have this inner layer so the publicly exposed struct is completely
    // opaque. An enum cannot be opaque.
    inner: VersionedHeader,
    typ_decoders: Arc<RwLock<BTreeMap<Type, Arc<dyn FieldDecoder>>>>,
}

impl Debug for Header {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("Header")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

macro_rules! attr {
    ($header:expr, $attr:ident) => {
        match $header {
            Header {
                inner:
                    VersionedHeader::V6(HeaderV6 { $attr, .. })
                    | VersionedHeader::V7(HeaderV7 { $attr, .. }),
                ..
            } => $attr,
        }
    };
}

impl Header {
    /// Dereference an address in the string table embedded in the header.
    ///
    /// This string table is typically used to store a single copy of static strings referenced by
    /// some *char \** event fields.
    #[inline]
    pub fn deref_static(&self, addr: Address) -> Result<Value<'_>, EvalError> {
        match attr!(self, str_table).get(&addr) {
            Some(s) => Ok(Value::Str(Str::new_borrowed(s.deref()))),
            None => Err(EvalError::CannotDeref(addr)),
        }
    }

    /// Returns an iterator of [EventDesc] for all the ftrace events defined in the header.
    #[inline]
    pub fn event_descs(&self) -> impl IntoIterator<Item = &EventDesc> {
        attr!(self, event_descs)
    }

    #[inline]
    pub fn event_desc_by_id(&self, id: EventId) -> Option<&EventDesc> {
        self.event_descs()
            .into_iter()
            .find(move |desc| desc.id == id)
    }

    #[inline]
    pub fn event_desc_by_name(&self, name: &str) -> Option<&EventDesc> {
        self.event_descs()
            .into_iter()
            .find(move |desc| desc.name == name)
    }

    /// ABI of the kernel that generated the ftrace this header is representing.
    #[inline]
    pub fn kernel_abi(&self) -> &Abi {
        attr!(self, kernel_abi)
    }

    /// Lookup the task name of the given PID in the PID/name table stored in the header.
    #[inline]
    pub fn comm_of(&self, pid: Pid) -> Option<&TaskName> {
        attr!(self, pid_comms).get(&pid)
    }

    /// Loookup the symbol offset, size and name at address `addr`.
    ///
    /// The data is coming from the copy of `/proc/kallsyms` embedded in the header.
    pub fn sym_at(&self, addr: Address) -> Option<(AddressOffset, Option<AddressSize>, &str)> {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        if addr == 0 {
            None
        } else {
            let map = attr!(self, kallsyms);
            let next_addr = map
                .range((Excluded(addr), Unbounded))
                .next()
                .map(|(addr, _)| addr);
            map.range((Unbounded, Included(addr)))
                .last()
                .map(|(base, s)| {
                    let size = next_addr.map(|next| next - addr);
                    let offset = addr - base;
                    (offset, size, s.deref())
                })
        }
    }

    /// Number of CPUs with a buffer in that trace.
    #[inline]
    pub fn nr_cpus(&self) -> Cpu {
        *attr!(self, nr_cpus)
    }

    /// Header options encoded in the header.
    #[inline]
    pub fn options(&self) -> impl IntoIterator<Item = &Options> {
        attr!(self, options)
    }

    /// Parsed content of `/proc/kallsyms` encoded in the header.
    ///
    /// Note: The addresses may not be the real addresses for security reasons depending on the
    /// kernel configuration.
    #[inline]
    pub fn kallsyms(&self) -> impl IntoIterator<Item = (Address, &str)> {
        attr!(self, kallsyms).iter().map(|(k, v)| (*k, v.deref()))
    }

    /// Content of the PID/task name table as an iterator.
    #[inline]
    pub fn pid_comms(&self) -> impl IntoIterator<Item = (Pid, &str)> {
        attr!(self, pid_comms).iter().map(|(k, v)| (*k, v.deref()))
    }

    /// Returns a timestamp fixup closure based on he header options that can affect it.
    pub(crate) fn timestamp_fixer(&self) -> impl Fn(Timestamp) -> Timestamp {
        let mut offset_signed: i64 = 0;
        let mut offset_unsigned: u64 = 0;
        let mut _multiplier: u32 = 1;
        let mut _shift: u32 = 0;

        for opt in self.options() {
            match opt {
                Options::TimeOffset(offset) => {
                    offset_signed += *offset;
                }
                Options::TSC2NSec {
                    multiplier,
                    shift,
                    offset,
                } => {
                    offset_unsigned += *offset;
                    _multiplier = *multiplier;
                    _shift = *shift;
                }
                _ => (),
            }
        }

        move |ts: Timestamp| {
            let ts: u128 = ts.into();
            let ts = (ts * _multiplier as u128) >> _shift;
            let ts = ts as u64;
            ts.saturating_add_signed(offset_signed) + offset_unsigned
        }
    }

    /// Vector of buffers found in that trace.dat file.
    pub fn buffers<'i, 'h, 'a: 'i + 'h, I: BorrowingRead + Send + 'i>(
        &'a self,
        input: Box<I>,
    ) -> Result<Vec<Buffer<'i, 'h>>, BufferError> {
        match &self.inner {
            VersionedHeader::V6(header) => header.buffers(self, input),
            VersionedHeader::V7(header) => header.buffers(self, input),
        }
    }

    #[inline]
    fn fixup_event_descs(&mut self) {
        let mut header = self.clone();
        // Ensure we won't accidentally lookup an event descriptor with a broken parent link on the
        // header clone. This also saves some memory.
        *attr!(&mut header, event_descs) = Vec::new();

        let header: Arc<_> = header.into();
        for event_desc in attr!(self, event_descs) {
            event_desc.header = Some(Arc::clone(&header))
        }
    }

    /// Name of the clock that was used in that ftrace session, if available in the trace.dat file.
    /// <div class="warning">
    /// trace.dat v6 has a creative encoding for the clock which is not handled here. This will
    /// therefore return [None] for that version.
    /// </div>
    pub fn clock(&self) -> Option<&str> {
        let mut parser = nom::sequence::preceded(
            nom::bytes::complete::take_till(|c| c == '['),
            nom::sequence::delimited(
                nom::character::complete::char('['),
                nom::bytes::complete::take_till(|c| c == ']'),
                nom::character::complete::char(']'),
            ),
        );

        for opt in self.options() {
            if let Options::TraceClock(clock) = opt {
                return match nom::Parser::<_>::parse(&mut parser, clock.deref()).finish() {
                    Ok((_, clock)) => Some(clock),
                    Err(()) => None,
                };
            }
        }
        None
    }

    /// Unique identifier of the tracing session that lead to that trace.dat.
    pub fn trace_id(&self) -> Option<StdString> {
        for opt in self.options() {
            if let Options::TraceId(id) = opt {
                return Some(id.to_string());
            }
        }
        None
    }
}

/// Binary format of a trace event field.
#[derive(Clone)]
pub struct FieldFmt {
    /// C declaration of that field considered as a struct member.
    ///
    /// This includes the name of the field.
    pub declaration: Declaration,
    /// Offset of the field in the binary content of an event.
    pub offset: MemOffset,
    /// Size of the field in the binary content of an event.
    pub size: MemSize,

    pub(crate) decoder: Arc<dyn FieldDecoder>,
}

// This instance is used for testing for now. We compare everything except the
// decoder, which cannot be compared as it is some sort of closure.
impl PartialEq<Self> for FieldFmt {
    fn eq(&self, other: &Self) -> bool {
        self.declaration == other.declaration
            && self.offset == other.offset
            && self.size == other.size
    }
}

impl Debug for FieldFmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("FieldFmt")
            .field("declaration", &self.declaration)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Binary format of a struct, typically used for an ftrace event.
#[derive(Debug, Clone, PartialEq)]
pub struct StructFmt {
    /// Vector of [FieldFmt], one for each struct member.
    pub fields: Vec<FieldFmt>,
}

impl StructFmt {
    pub fn field_by_name<Q>(&self, name: &Q) -> Option<&FieldFmt>
    where
        Q: ?Sized,
        Identifier: Borrow<Q> + PartialEq<Q>,
    {
        self.fields
            .iter()
            .find(|&field| &field.declaration.identifier == name.borrow())
    }
}

fn fixup_c_type(
    typ: Type,
    size: MemSize,
    signedness: Signedness,
    abi: &Abi,
) -> Result<Type, HeaderError> {
    let inferred_size = typ.size(abi).ok();
    let inferred_signedness = typ.arith_info().map(|info| info.signedness());

    if let Some(inferred_size) = inferred_size {
        if let Ok(size) = size.try_into() {
            if inferred_size != size {
                return Err(HeaderError::InvalidTypeSize {
                    typ,
                    inferred_size,
                    size,
                });
            }
        }
    }

    if let Some(inferred_signedness) = inferred_signedness {
        if inferred_signedness != signedness {
            return Err(HeaderError::InvalidTypeSign {
                typ,
                inferred_signedness,
                signedness,
            });
        }
    }

    // Primarily fixup Enum where the underlying type is unknown to the C
    // parser, but here we have a chance of actually fixing it up since we
    // usually know the size and signednessness of the type.
    fn fixup(typ: Type, size: Option<MemSize>, signedness: Signedness) -> Type {
        match typ {
            Type::Array(typ, ArrayKind::Fixed(Ok(0))) if size == Some(0) => Type::Array(
                Box::new(fixup(*typ, None, signedness)),
                ArrayKind::ZeroLength,
            ),
            Type::Array(typ, ArrayKind::Fixed(array_size)) => {
                let item_size = match (size, &array_size) {
                    (Some(size), Ok(array_size)) => {
                        let array_size: usize = (*array_size).try_into().unwrap();
                        Some(size / array_size)
                    }
                    _ => None,
                };
                Type::Array(
                    Box::new(fixup(*typ, item_size, signedness)),
                    ArrayKind::Fixed(array_size),
                )
            }
            Type::Array(typ, kind @ ArrayKind::Dynamic(_)) => {
                Type::Array(Box::new(fixup(*typ, None, signedness)), kind)
            }

            Type::Typedef(typ, id) => Type::Typedef(Box::new(fixup(*typ, size, signedness)), id),
            Type::Enum(typ, id) => Type::Enum(Box::new(fixup(*typ, size, signedness)), id),
            Type::Unknown => match (size.map(|x| x * 8), signedness) {
                (Some(8), Signedness::Unsigned) => Type::U8,
                (Some(8), Signedness::Signed) => Type::I8,

                (Some(16), Signedness::Unsigned) => Type::U16,
                (Some(16), Signedness::Signed) => Type::I16,

                (Some(32), Signedness::Unsigned) => Type::U32,
                (Some(32), Signedness::Signed) => Type::I32,

                (Some(64), Signedness::Unsigned) => Type::U64,
                (Some(64), Signedness::Signed) => Type::I64,
                _ => Type::Unknown,
            },
            typ => typ,
        }
    }

    Ok(fixup(typ, Some(size), signedness))
}

type HeaderNomError<'a> = NomError<HeaderError, nom_language::error::VerboseError<&'a [u8]>>;

/// Parse the struct format of an ftrace event as reported in
/// `/sys/kernel/tracing/events/*/*/format`
#[inline(never)]
fn parse_struct_fmt<'a, PE: ParseEnv>(
    penv: &PE,
    skip_fixup: bool,
    input: &'a [u8],
) -> nom::IResult<&'a [u8], StructFmt, HeaderNomError<'a>> {
    terminated(
        separated_list0(
            char('\n'),
            map_res_cut(
                preceded(
                    lexeme(tag(&b"field:"[..])),
                    separated_pair(
                        is_not(";"),
                        char(';'),
                        terminated(
                            separated_list0(
                                char(';'),
                                separated_pair(
                                    preceded(is_a("\t "), is_not("\n:").map(to_str)),
                                    char(':'),
                                    is_not(";").map(to_str),
                                ),
                            ),
                            char(';'),
                        ),
                    ),
                ),
                move |(declaration, props)| {
                    let props = BTreeMap::from_iter(props);
                    macro_rules! get {
                        ($name:expr) => {
                            props
                                .get($name)
                                .expect(concat!("Expected field property", $name))
                                .parse()
                                .expect("Failed to parse field property value")
                        };
                    }

                    let (_, mut declaration) = CGrammar::apply_rule(
                        all_consuming(CGrammar::declaration()),
                        declaration,
                        &CGrammarCtx::new(penv),
                    )
                    .map_err(|_| HeaderError::InvalidDeclaration)?;

                    let signedness = {
                        let signed: u8 = get!("signed");
                        if signed > 0 {
                            Signedness::Signed
                        } else {
                            Signedness::Unsigned
                        }
                    };
                    let size = get!("size");
                    if !skip_fixup {
                        declaration.typ =
                            fixup_c_type(declaration.typ, size, signedness, penv.abi())?;
                    }

                    Ok(FieldFmt {
                        declaration,
                        offset: get!("offset"),
                        size,
                        decoder: Arc::new(closure!(
                            (
                                for<'d> Fn(
                                    &'d [u8],
                                    &'d [u8],
                                    &'d Header,
                                    &'d ScratchAlloc,
                                )
                                -> Result<Value<'d>, BufferError>
                            ),
                            |_, _, _, _| Ok(Value::Unknown)
                        )),
                    })
                },
            ),
        ),
        opt(char('\n')),
    )
    .map(|fields| StructFmt { fields })
    .parse(input)
}

/// Parse header_event spec in the header
#[inline(never)]
fn parse_header_event(input: &[u8]) -> nom::IResult<&[u8], (), HeaderNomError<'_>> {
    map_res(
        preceded(
            opt(lexeme(preceded(char('#'), many0(is_not("\n"))))),
            fold_many0(
                terminated(
                    alt((
                        separated_pair(
                            |input| match identifier::<_, ()>().parse(input) {
                                Ok(id) => Ok(id),
                                Err(_) => error(input, HeaderError::InvalidCIdentifier),
                            },
                            char(':'),
                            delimited(
                                opt(pair(lexeme(tag("type")), lexeme(tag("==")))),
                                lexeme(txt_u64),
                                opt(lexeme(tag("bits"))),
                            ),
                        )
                        .map(|(id, n)| match (id.as_ref(), n) {
                            ("type_len", 5) => Ok(()),
                            ("type_len", x) => Err(HeaderError::InvalidEventHeader {
                                field: "type_len".into(),
                                value: x.to_string(),
                            }),

                            ("time_delta", 27) => Ok(()),
                            ("time_delta", x) => Err(HeaderError::InvalidEventHeader {
                                field: "time_delta".into(),
                                value: x.to_string(),
                            }),

                            ("array", 32) => Ok(()),
                            ("array", x) => Err(HeaderError::InvalidEventHeader {
                                field: "array".into(),
                                value: x.to_string(),
                            }),

                            ("padding", 29) => Ok(()),
                            ("padding", x) => Err(HeaderError::InvalidEventHeader {
                                field: "padding".into(),
                                value: x.to_string(),
                            }),

                            ("time_extend", 30) => Ok(()),
                            ("time_extend", x) => Err(HeaderError::InvalidEventHeader {
                                field: "time_extend".into(),
                                value: x.to_string(),
                            }),

                            ("time_stamp", 31) => Ok(()),
                            ("time_stamp", x) => Err(HeaderError::InvalidEventHeader {
                                field: "time_stamp".into(),
                                value: x.to_string(),
                            }),
                            _ => Ok(()),
                        }),
                        preceded(
                            (
                                lexeme(tag("data")),
                                lexeme(tag("max")),
                                lexeme(tag("type_len")),
                                lexeme(tag("==")),
                            ),
                            lexeme(txt_u64).map(|bits| match bits {
                                28 => Ok(()),
                                x => Err(HeaderError::InvalidEventHeader {
                                    field: "data max type_len".into(),
                                    value: x.to_string(),
                                }),
                            }),
                        ),
                    )),
                    opt(many0(char('\n'))),
                ),
                || Ok(()),
                |acc, i| match acc {
                    Ok(..) => i,
                    Err(..) => acc,
                },
            ),
        ),
        // Simplify the return type by "promoting" validation errors into a parse
        // error.
        |res| res,
    )
    .parse(input)
}

/// Descriptor of an ftrace event.
#[derive(Debug, Clone)]
pub struct EventDesc {
    /// Name of the ftrace event.
    ///
    /// This does not include the subsystem name.
    pub name: String,
    /// Unique ID of that event in the header.
    ///
    /// It is not unique accross files.
    pub id: EventId,
    /// Binary and print format of the event.
    // Use a OnceCell so that we can mutate it in place in order to lazily parse
    // the format and memoize the result.
    fmt: OnceCell<Result<EventFmt, HeaderError>>,
    /// Raw format in ASCII as encoded in the header.
    raw_fmt: Vec<u8>,
    /// Backlink to the header that defines that event.
    header: Option<Arc<Header>>,
}

/// Combines binary and print format of an ftrace event
#[derive(Clone)]
pub struct EventFmt {
    /// Binary format of the event encoded as a struct memory dump.
    struct_fmt: Result<StructFmt, HeaderError>,
    /// Print format of an ftrace event.
    ///
    /// This includes a [PrintFmtStr] to represent a (parsed) printk-style format string and a list
    /// of [Evaluator] objects for each value to interpolate in the format string.
    #[allow(clippy::type_complexity)]
    print_fmt_args:
        Result<(PrintFmtStr, Vec<Result<Arc<dyn Evaluator>, CompileError>>), HeaderError>,
}

impl EventFmt {
    /// Binary format of the event struct.
    pub fn struct_fmt(&self) -> Result<&StructFmt, HeaderError> {
        match &self.struct_fmt {
            Ok(x) => Ok(x),
            Err(err) => Err(err.clone()),
        }
    }

    /// Parsed printk-style format of the event.
    pub fn print_fmt(&self) -> Result<&PrintFmtStr, HeaderError> {
        match &self.print_fmt_args {
            Ok(x) => Ok(&x.0),
            Err(err) => Err(err.clone()),
        }
    }

    /// Evaluators for the arguments to interpolate in the printk-style format of the event.
    pub fn print_args(
        &self,
    ) -> Result<impl IntoIterator<Item = &Result<Arc<dyn Evaluator>, CompileError>>, HeaderError>
    {
        match &self.print_fmt_args {
            Ok(x) => Ok(&x.1),
            Err(err) => Err(err.clone()),
        }
    }
}

impl Debug for EventFmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("EventFmt")
            .field("struct_fmt", &self.struct_fmt())
            .field("print_fmt", &self.print_fmt())
            .finish_non_exhaustive()
    }
}

impl PartialEq<Self> for EventFmt {
    fn eq(&self, other: &Self) -> bool {
        self.struct_fmt() == other.struct_fmt() && self.print_fmt() == other.print_fmt()
    }
}

impl EventDesc {
    /// Raw ASCII format of the event as found in `/sys/kernel/tracing/events/*/*/format`
    #[inline]
    pub fn raw_fmt(&self) -> Result<&[u8], HeaderError> {
        // Allow for errors in case we decide to drop the raw_fmt once it has been parsed
        Ok(&self.raw_fmt)
    }

    #[inline]
    pub fn event_fmt(&self) -> Result<&EventFmt, HeaderError> {
        match self
            .fmt
            .get_or_init(|| parse_event_fmt(self.header(), &self.name, &self.raw_fmt))
        {
            Ok(x) => Ok(x),
            Err(err) => Err(err.clone()),
        }
    }

    // This method is private as the header that we get through here is not complete and we might
    // re-implement what depends on a full header to only depend on some bits. It can also panic in
    // some circumstances
    #[inline]
    fn header(&self) -> &Header {
        // Not having the parent link can happen if:
        // * We try to use the EventDesc before Header::fixup_event_descs() was called
        // * We try to use an EventDesc obtained from a copy of the header before the fixup was
        //   done. This can happen if soemone tries to access EventDesc attached to the header that
        //   was stored in an Rc<> to provide a parent to EventDesc attached to the primary Header
        self.header
            .as_ref()
            .expect("EventDesc does not have a parent Header link")
    }
}

impl PartialEq<Self> for EventDesc {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.name == other.name && self.raw_fmt == other.raw_fmt
    }
}
impl Eq for EventDesc {}

impl Hash for EventDesc {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state);
        self.name.hash(state);
        self.raw_fmt.hash(state);
    }
}

/// Compilation and evaluation environment attached to a header.
///
/// This provides access to the string table.
struct HeaderEnv<'h> {
    header: &'h Header,
    struct_fmt: &'h StructFmt,
    scratch: ScratchAlloc,
}

impl<'h> EvalEnv<'h> for HeaderEnv<'h> {
    #[inline]
    fn deref_static(&self, addr: u64) -> Result<Value<'_>, EvalError> {
        self.header.deref_static(addr)
    }

    #[inline]
    fn scratch(&self) -> &ScratchAlloc {
        &self.scratch
    }

    fn header(&self) -> Result<&Header, EvalError> {
        Ok(self.header)
    }

    fn event_data(&self) -> Result<&[u8], EvalError> {
        Err(EvalError::NoEventData)
    }
}

impl<'h> HeaderEnv<'h> {
    fn new(header: &'h Header, struct_fmt: &'h StructFmt) -> Self {
        HeaderEnv {
            header,
            struct_fmt,
            scratch: ScratchAlloc::new(),
        }
    }
}

impl ParseEnv for HeaderEnv<'_> {
    #[inline]
    fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
        for field in &self.struct_fmt.fields {
            if field.declaration.identifier == id {
                return Ok(field.declaration.typ.clone());
            }
        }
        Err(CompileError::UnknownField(id.into()))
    }
    #[inline]
    fn abi(&self) -> &Abi {
        self.header.kernel_abi()
    }
}

impl<'ce> CompileEnv<'ce> for HeaderEnv<'ce> {
    #[inline]
    fn field_getter(&self, id: &str) -> Result<Box<dyn Evaluator>, CompileError> {
        for field in &self.struct_fmt.fields {
            if field.declaration.identifier == id {
                let decoder = field.decoder.clone();
                let offset = field.offset;
                let end = offset + field.size;

                fn make_box<F>(f: F) -> Box<dyn Evaluator>
                where
                    F: for<'ee, 'eeref> Fn(
                            &'eeref (dyn EvalEnv<'ee> + 'eeref),
                        )
                            -> Result<Value<'eeref>, EvalError>
                        + Send
                        + Sync
                        + 'static,
                {
                    Box::new(f)
                }

                return Ok(make_box(move |env: &dyn EvalEnv<'_>| {
                    let event_data = env.event_data()?;
                    let header = env.header()?;
                    let field_data = &event_data[offset..end];
                    Ok(decoder.decode(event_data, field_data, header, env.scratch())?)
                }));
            }
        }
        Err(CompileError::UnknownField(id.into()))
    }
}

/// Parse event format as found in `/sys/kernel/tracing/events/*/*/format`
#[inline(never)]
fn parse_event_fmt<'a>(
    header: &'a Header,
    name: &'a str,
    input: &[u8],
) -> Result<EventFmt, HeaderError> {
    context(
            "event description",
            flat_map(
                context(
                    "event format",
                    preceded(
                        pair(lexeme(tag("format:")), multispace0),
                        |input| {
                            let penv = BasicEnv::new(header.kernel_abi());
                            parse_struct_fmt(&penv, false, input)
                        }
                    )
                    .map(|mut fmt| {
                        for field in &mut fmt.fields {
                            let typ = &field.declaration.typ;
                            let memo = header.typ_decoders.read().unwrap();
                            field.decoder = match memo.get(typ) {
                                Some(decoder) => Arc::clone(decoder),
                                None => {
                                    // Release the read lock so we can lock it to write again.
                                    drop(memo);
                                    let decoder = match typ.make_decoder(header) {
                                        Ok(parser) => parser,
                                        Err(err) => {
                                            let closure = closure!(
                                                (
                                                    for<'d> Fn(
                                                        &'d [u8],
                                                        &'d [u8],
                                                        &'d Header,
                                                        &'d ScratchAlloc,
                                                    )
                                                    -> Result<Value<'d>, BufferError>
                                                ),
                                                move |_, _, _, _| Err(err.clone().into())
                                            );
                                            Arc::new(closure) as Arc<dyn FieldDecoder>
                                        }
                                    };
                                    let mut memo = header.typ_decoders.write().unwrap();
                                    memo.insert(typ.clone(), decoder.clone());
                                    decoder
                                }
                            };
                        }
                        fmt
                    }),
                ),
                |struct_fmt| context(
                    "event print fmt",
                    preceded(
                        pair(multispace0, lexeme(tag("print fmt:"))),
                        map_res_cut(
                            move |input| {
                                let struct_fmt = struct_fmt.clone();
                                let cenv = HeaderEnv::new(header, &struct_fmt);
                                let (consumed, res) = match CGrammar::apply_rule(
                                    all_consuming(lexeme(CGrammar::expr())),
                                    input,
                                    &CGrammarCtx::new(&cenv),
                                ) {
                                    Ok((remaining, x)) => {
                                        let consumed = input.len() - remaining.len();
                                        (consumed, Ok(x))
                                    },
                                    Err(err) => (
                                        // Consume all the input so we don't
                                        // risk re-parsing the print fmt as
                                        // something else
                                        input.len(),
                                        Err(PrintFmtError::CParseError(Box::new(err)))
                                    )
                                };
                                Ok((&input[consumed..], (struct_fmt, res)))
                            },
                            |(struct_fmt, expr)| {
                                let get = move || -> Result<_, PrintFmtError> {
                                    let (fmt, exprs) = match expr? {
                                        Expr::CommaExpr(vec) => {
                                            let mut iter = vec.into_iter();
                                            let s = iter.next();
                                            let args: Vec<Expr> = iter.collect();

                                            let s = match s {
                                                Some(Expr::StringLiteral(s)) => Ok(s),
                                                Some(expr) => Err(PrintFmtError::NotAStringLiteral(expr)),
                                                None => panic!("CommaExpr is expected to contain at least one expression"),
                                            }?;
                                            Ok((s, args))
                                        }
                                        Expr::StringLiteral(s) => Ok((s, vec![])),
                                        expr => Err(PrintFmtError::NotAStringLiteral(expr)),
                                    }?;
                                    let fmt = PrintFmtStr::try_new(header, fmt.as_bytes())?;
                                    Ok((fmt, exprs))
                                };
                                // We are handling errors at the next stage, so we can create an
                                // EventFmt even if we did not manage to parse the print format
                                Ok((struct_fmt.clone(), get()))
                            },
                        ),
                    ),
                ),
            )
            .map(|(struct_fmt, print_fmt_args)| {
                // bprint format is a lie, so we deal with it by replacing
                // REC->fmt by an extension function call that formats REC->fmt
                // and REC->buf into a string.
                //
                // We only do the fixup if the format string contains a "%s"
                // called with a REC->fmt argument, so we stay
                // forward-compatible with future fixes.
                // https://bugzilla.kernel.org/show_bug.cgi?id=217357
                let fixup_arg = move |(expr, atom): (Expr, _)| {
                    match atom {
                        Some(&PrintAtom::Variable {print_spec: PrintSpecifier::Str, ..}) if name == "bprint" && expr.is_record_field("fmt") => {
                            macro_rules! compiler {
                                ($char_typ:expr) => {
                                    ExtensionMacroCallCompiler {
                                        ret_typ: Type::Pointer(Box::new($char_typ)),
                                        compiler: Arc::new(|cenv: &dyn CompileEnv| {
                                            let fmt = cenv.field_getter("fmt")?;
                                            let buf = cenv.field_getter("buf")?;
                                            Ok(new_dyn_evaluator({
                                                // Cache the parsed format string
                                                let fmt_map: RwLock<BTreeMap<Address, PrintFmtStr>> = RwLock::new(BTreeMap::new());
                                                move |env| {
                                                    let header = env.header()?;
                                                    // Get the format string and its parsed PrintFmtStr
                                                    let fmt_addr = match fmt.eval(env)? {
                                                        Value::U64Scalar(addr) => Ok(addr),
                                                        Value::I64Scalar(addr) => Ok(addr as u64),
                                                        val => Err(EvalError::IllegalType(val.into_static().ok()))
                                                    }?;

                                                    let mut _fmt_map_read;
                                                    let mut _fmt_map_write;
                                                    let fmt = {
                                                        _fmt_map_read = fmt_map.read().unwrap();
                                                        match _fmt_map_read.get(&fmt_addr) {
                                                            Some(fmt) => fmt,
                                                            None => {
                                                                drop(_fmt_map_read);
                                                                let fmt = env.deref_static(fmt_addr)?;
                                                                let fmt = match fmt.to_str() {
                                                                    Some(s) => Ok(s),
                                                                    None => Err(EvalError::IllegalType(fmt.into_static().ok())),
                                                                }?;

                                                                let fmt = PrintFmtStr::try_new(header, fmt.as_bytes())?;
                                                                _fmt_map_write = fmt_map.write().unwrap();
                                                                _fmt_map_write.entry(fmt_addr).or_insert(fmt)
                                                            }
                                                        }

                                                    };

                                                    // Get the vbin buffer
                                                    let buf = buf.eval(env)?;

                                                    match buf {
                                                        Value::U32Array(array) => {
                                                            let array: &[u32] = array.deref();

                                                            let mut vec = ScratchVec::new_in(env.scratch());
                                                            let mut writer = StringWriter::new(&mut vec);

                                                            // We don't attempt
                                                            // to use
                                                            // Str::new_procedural()
                                                            // as there are too
                                                            // many ways the
                                                            // interpolation
                                                            // could fail, and
                                                            // procedural
                                                            // strings cannot
                                                            // fail other than
                                                            // panicking
                                                            fmt.interpolate_vbin(header, env, &mut writer, array)?;
                                                            // This "leaks" the
                                                            // ScratchVec, but
                                                            // it will be freed
                                                            // after processing
                                                            // that event by
                                                            // whoever created
                                                            // "env"
                                                            Ok(Value::U8Array(Array::Borrowed(vec.leak())))
                                                        }
                                                        val => Err(EvalError::IllegalType(val.into_static().ok())),
                                                    }
                                                }
                                            }))
                                        })
                                    }
                                }
                            }

                            let compiler = compiler!(header.kernel_abi().char_typ());
                            let desc = ExtensionMacroDesc::new_function_like(
                                "__format_vbin_printf".into(),
                                Box::new(move |penv, _input| {
                                    Ok(compiler!(penv.abi().char_typ()))
                                })
                            );

                            Expr::ExtensionMacroCall(ExtensionMacroCall {
                                args: "REC->fmt, REC->buf, __get_zero_length_array_len(REC->buf)".into(),
                                desc: Arc::new(desc),
                                compiler
                            })
                        },
                        _ => expr
                    }
                };

                let cenv = HeaderEnv::new(header, &struct_fmt);
                let print_fmt_args = match print_fmt_args {
                    Ok((print_fmt, print_args)) => {
                        let print_args: Vec<_> = PrintAtom::zip_atoms(
                                print_args.into_iter(),
                                print_fmt.atoms.iter()
                            )
                            .into_iter()
                            .map(fixup_arg)
                            .map(|expr| Ok(
                                Arc::from(expr.compile(&cenv)?))
                            )
                            .collect();
                        Ok((
                            print_fmt,
                            print_args
                        ))
                    }
                    Err(err) => Err(HeaderError::PrintFmtError(Box::new(err))),
                };

                // TODO: We could possibly exploit e.g. __print_symbolic() to fixup the enum
                // variants, based on the print_fmt, but:
                // 1. The strings are arbitrary and cannot be expected to be valid identifier
                //    matching any original enum. That is probably fine but might be a bit
                //    surprising.
                // 2. We cannot count on the strings described by __print_symbolic() to be covering
                //    all the cases.
                EventFmt {
                    // For now, we just fail if we cannot at least parse the StructFmt, since the
                    // resulting EventFmt would be quite useless. This might change in the future
                    // if necessary.
                    struct_fmt: Ok(struct_fmt),
                    print_fmt_args,
                }
            }),
        )
        .parse_finish(input)
}

/// Parse the content of `/sys/kernel/tracing/events/*/*/format`
#[inline(never)]
fn parse_event_desc(input: &[u8]) -> nom::IResult<&[u8], EventDesc, HeaderNomError<'_>> {
    context(
        "event description",
        map_res_cut(
            (
                context(
                    "event name",
                    preceded(
                        lexeme(tag("name:")),
                        lexeme(terminated(is_not("\n"), char('\n'))),
                    ),
                ),
                context("event ID", preceded(lexeme(tag("ID:")), lexeme(txt_u16))),
                context("remainder", rest),
            ),
            |(name, id, fmt)| {
                Ok(EventDesc {
                    name: StdString::from_utf8_lossy(name).into(),
                    id,
                    fmt: OnceCell::new(),
                    // Store the unparsed content, as parsing is costly and only
                    // a handful of events will typically be actually used in
                    // the trace.
                    raw_fmt: fmt.to_vec(),
                    // Will be fixed up later
                    header: None,
                })
            },
        ),
    )
    .parse(input)
}

/// Parse content of `/proc/kallsyms`
#[inline(never)]
fn parse_kallsyms(
    input: &[u8],
) -> nom::IResult<&[u8], BTreeMap<Address, SymbolName>, HeaderNomError<'_>> {
    context("kallsyms", move |input| {
        let line = terminated(
            separated_pair(
                hex_u64,
                delimited(multispace1, is_not(" \t"), multispace1),
                map_res_cut(
                    pair(
                        is_not("\t\n"),
                        // The symbol name can be followed by \t[module_name],
                        // so we consume the \t in between to provide cleaner
                        // output.
                        opt(preceded(is_a(" \t"), is_not("\n"))),
                    ),
                    |(name, module)| match from_utf8(name) {
                        // Filter-out symbols starting with "$" as they are probably just mapping
                        // symbols that can sometimes have the same value as real function symbols,
                        // thereby breaking the output.  (see "ELF for the Arm© 64-bit Architecture
                        // (AArch64)" document).
                        // Also filter out all the compiler-generated symbols, e.g. ones that have
                        // a suffix as a result of some optimization pass.
                        Ok(name) if is_identifier(name) => Ok(Some(match module.map(from_utf8) {
                            Some(Ok(module)) => {
                                let mut full: SymbolName = name.into();
                                full.push_str(" ");
                                full.push_str(module);
                                full
                            }
                            _ => name.into(),
                        })),
                        Ok(_) => Ok(None),
                        Err(err) => Err(HeaderError::DecodeUtf8(err.to_string())),
                    },
                ),
            ),
            char('\n'),
        );

        let mut it = iterator(input, line);
        let parsed = it
            .by_ref()
            .filter_map(|item| match item {
                (addr, Some(name)) => Some(Ok((addr, name))),
                _ => None,
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let (input, _) = it.finish()?;
        Ok((input, parsed))
    })
    .parse(input)
}

/// Parse string table in trace.dat header.
///
/// The table is typically used to store printk-style format strings, but also string literals that
/// can be referenced by address in the event fields and print args expressions.
#[inline(never)]
fn parse_str_table(
    input: &[u8],
) -> nom::IResult<&[u8], BTreeMap<Address, String>, HeaderNomError<'_>> {
    context("trace_printk fmt", move |input| {
        let line = separated_pair(
            preceded(tag("0x"), hex_u64),
            lexeme(char(':')),
            move |input| {
                // Gather the line using a fast parser, otherwise invoking the C
                // parser on the whole input will allocate large amounts of
                // memory for the packrat state, out of which only the first few
                // tens of positions will be used. This can take seconds in
                // debug builds.
                let (input, line) = terminated(is_not("\n"), char('\n')).parse(input)?;
                let res: Result<(_, _), NomError<_, _>> =
                    all_consuming(string_literal()).parse(line).finish();
                match res {
                    Ok((_, Expr::StringLiteral(s))) => Ok((input, s)),
                    Ok((_, expr)) => failure(input, PrintFmtError::NotAStringLiteral(expr).into()),
                    Err(err) => err.into_external(input, |data| {
                        PrintFmtError::CParseError(Box::new(data)).into()
                    }),
                }
            },
        );
        let mut it = iterator(input, line);
        let parsed = it.by_ref().collect::<BTreeMap<_, _>>();
        let (input, _) = it.finish()?;
        Ok((input, parsed))
    })
    .parse(input)
}

/// Parse PID/task name tables.
#[inline(never)]
fn parse_pid_comms(input: &[u8]) -> nom::IResult<&[u8], BTreeMap<Pid, String>, HeaderNomError<'_>> {
    context("PID map", move |input| {
        let line = separated_pair(
            txt_u32,
            multispace1,
            map_res_cut(lexeme(is_not("\n")), |x| match from_utf8(x) {
                Ok(s) => Ok(s.into()),
                Err(err) => Err(HeaderError::DecodeUtf8(err.to_string())),
            }),
        );
        let mut it = iterator(input, line);
        let parsed = it.by_ref().collect::<BTreeMap<_, _>>();
        let (input, _) = it.finish()?;
        Ok((input, parsed))
    })
    .parse(input)
}

/// Error type used in [Header] methods and manipulation function.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum HeaderError {
    #[error("Bad magic found")]
    BadMagic,

    #[error("Could not decode UTF-8 string: {0}")]
    DecodeUtf8(StdString),

    #[error("Could not parse file format version: {0}")]
    InvalidVersion(StdString),

    #[error("Expected 0 or 1 for endianness, got: {0}")]
    InvalidEndianness(u8),

    #[error("Could not parse C declaration")]
    InvalidDeclaration,

    #[error("Could not parse C identifier")]
    InvalidCIdentifier,

    #[error(
        "Size of type \"{typ:?}\" was inferred to be {inferred_size} but kernel reported {size}"
    )]
    InvalidTypeSize {
        typ: Type,
        inferred_size: u64,
        size: u64,
    },

    #[error("Sign of type \"{typ:?}\" was inferred to be {inferred_signedness:?} but kernel reported {signedness}")]
    InvalidTypeSign {
        typ: Type,
        inferred_signedness: Signedness,
        signedness: Signedness,
    },

    #[error("Invalid long size: {0}")]
    InvalidLongSize(MemSize),

    #[error("Could not parse option {0} in header: {1}")]
    InvalidOption(u16, StdString),

    #[error("Compression codec not supported: {0}")]
    UnsupportedCompressionCodec(StdString),

    #[error("Compressed section was found but header specified no compression codec")]
    CompressedSectionWithoutCodec,

    #[error("Expected section ID {expected} but found {found}")]
    UnexpectedSection {
        expected: SectionId,
        found: SectionId,
    },

    #[error("Expected header page start")]
    ExpectedHeaderPage,

    #[error("Could not load section as it is too large: {0} bytes")]
    SectionTooLarge(u64),

    #[error("Could not load header page size as it is too large: {0} bytes")]
    PageHeaderSizeTooLarge(FileSize),

    #[error("Expected header event start")]
    ExpectedHeaderEvent,

    #[error("Could not load header event as it is too large: {0} bytes")]
    HeaderEventSizeTooLarge(FileSize),

    #[error("Could not load event format as it is too large: {0} bytes")]
    EventDescTooLarge(u64),

    #[error("Could not load kallsyms as it is too large: {0} bytes")]
    KallsymsTooLarge(u64),

    #[error("Could not load trace printk format strings table as it is too large: {0} bytes")]
    StrTableTooLarge(u64),

    #[error("Unexpected event header value \"{value}\" for field \"{field}\"")]
    InvalidEventHeader { field: StdString, value: StdString },

    #[error("Could not find the kernel long size")]
    LongSizeNotFound,

    #[error("Could not find the kernel char signedness")]
    CharSignednessNotFound,

    #[error("Data format not supported: {}", match .0 {
        None => "<unknown>",
        Some(s) => &s,
    })]
    UnsupportedDataFmt(Option<StdString>),

    #[error("Error while loading data: {0}")]
    IoError(Box<io::ErrorKind>),

    #[error("Could not parse header: {0}")]
    ParseError(Box<VerboseParseError>),

    #[error("Error while parsing printk format: {0}")]
    PrintFmtError(Box<PrintFmtError>),
}
convert_err_impl!(io::ErrorKind, IoError, HeaderError);
convert_err_impl!(PrintFmtError, PrintFmtError, HeaderError);

impl From<IoError> for HeaderError {
    fn from(err: IoError) -> HeaderError {
        err.kind().into()
    }
}

impl<I: AsRef<[u8]>, I2: AsRef<[u8]>> FromParseError<I, nom_language::error::VerboseError<I2>>
    for HeaderError
{
    fn from_parse_error(input: I, err: &nom_language::error::VerboseError<I2>) -> Self {
        HeaderError::ParseError(Box::new(VerboseParseError::new(input, err)))
    }
}

impl<I: AsRef<[u8]>> FromParseError<I, ()> for HeaderError {
    fn from_parse_error(input: I, _err: &()) -> Self {
        HeaderError::ParseError(Box::new(VerboseParseError::from_input(input)))
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TimeShiftCorrection {
    pub timestamp: Timestamp,
    pub scaling_ratio: u64,
    pub offset: TimeOffset,
}

/// Timestamp correction as encoded in trace.dat header option `TIME_SHIFT`
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TimeShiftCpuCorrection {
    pub cpu: Cpu,
    pub corrections: Vec<TimeShiftCorrection>,
}

/// Guest vCPU info as encoded in trace.dat header option `GUEST`
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GuestCpuInfo {
    pub cpu: Cpu,
    pub host_task_pid: Pid,
}

/// Options found in a [Header], regardless of the trace.dat format version.
///
/// Some options can only appear in some versions of the format.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Options {
    /// v6 only, defines a non-top-level instance
    #[non_exhaustive]
    Instance {
        name: String,
        offset: FileOffset,
    },

    /// v7 only, fully defines the location of a single ring buffer
    #[non_exhaustive]
    Buffer {
        cpu: Cpu,
        name: String,
        offset: FileOffset,
        size: FileSize,
        decomp: Option<DynDecompressor>,
        page_size: MemSize,
    },

    BufferText(),

    HeaderInfoLoc(FileOffset),
    FtraceEventsLoc(FileOffset),
    EventFormatsLoc(FileOffset),
    KallsymsLoc(FileOffset),
    PrintkLoc(FileOffset),
    CmdLinesLoc(FileOffset),
    TimeOffset(TimeOffset),

    #[non_exhaustive]
    TSC2NSec {
        multiplier: u32,
        shift: u32,
        offset: u64,
    },
    Date(String),

    CpuStat {
        cpu: Cpu,
        stat: String,
    },

    TraceClock(String),
    Uname(String),
    Hook(String),
    CpuCount(Cpu),
    Version(String),
    ProcMaps(String),
    TraceId(u64),
    #[non_exhaustive]
    TimeShift {
        peer_trace_id: u64,
        flags: u32,
        cpu_corrections: Vec<TimeShiftCpuCorrection>,
    },
    #[non_exhaustive]
    Guest {
        guest_name: String,
        guest_trace_id: u64,
        guests_cpu_info: Vec<GuestCpuInfo>,
    },
    #[non_exhaustive]
    Unknown {
        typ: u16,
        data: Vec<u8>,
    },
}

fn option_read_null_terminated(
    option_type: u16,
    option_data: &[u8],
) -> Result<(&[u8], &str), HeaderError> {
    match option_data {
        // Sometimes we get no data at all even though the doc states it
        // should be a null-terminated string:
        // https://bugzilla.kernel.org/show_bug.cgi?id=218430
        [] => Ok((&[], "")),
        _ => {
            let s = CStr::from_bytes_until_nul(option_data)
                .map_err(|err| HeaderError::InvalidOption(option_type, err.to_string()))?
                .to_str()
                .map_err(|err| HeaderError::InvalidOption(option_type, err.to_string()))?;
            let remaining = &option_data[s.len() + 1..];
            Ok((remaining, s))
        }
    }
}

fn option_parse_date(option_type: u16, date: &str) -> Result<TimeOffset, HeaderError> {
    let (date, sign) = if date.starts_with('-') {
        (date.trim_start_matches('-'), -1)
    } else {
        (date, 1)
    };

    let mut offset: TimeOffset = 0;
    for (prefix, base) in [
        (Some("0x"), 16),
        (Some("0X"), 16),
        (Some("0"), 8),
        (None, 10),
    ] {
        let (match_, date) = match prefix {
            Some(prefix) => (date.starts_with(prefix), date.trim_start_matches(prefix)),
            None => (true, date),
        };
        if match_ {
            offset = TimeOffset::from_str_radix(date, base)
                .map_err(|err| HeaderError::InvalidOption(option_type, err.to_string()))?;
            break;
        }
    }

    let offset = offset * sign;
    Ok(offset)
}

/// Decode options that are common between all trace.dat format versions.
fn shared_decode_option(
    abi: &Abi,
    option_type: u16,
    option_data: &[u8],
    cpu_stat_cpu: &mut Cpu,
) -> Result<Options, HeaderError> {
    Ok(match option_type {
        // DATE: id 1, size vary
        1 => {
            let (_, date) = option_read_null_terminated(option_type, option_data)?;
            Options::Date(date.into())
        }
        // CPUSTAT: id 2, size vary
        2 => {
            let cpu = *cpu_stat_cpu;
            *cpu_stat_cpu += 1;
            let (_, stat) = option_read_null_terminated(option_type, option_data)?;
            Options::CpuStat {
                cpu,
                stat: stat.into(),
            }
        }
        // TRACECLOCK: id 4, size vary
        4 => {
            // FIXME: v6 format has a strange way of storing the trace clock: if
            // the data is available, it will put a zero-sized TRACECLOCK
            // option, and the clock string will be encoded after the
            // "flyrecord\0" block:
            // https://bugzilla.kernel.org/show_bug.cgi?id=218430
            // For now, this will show up as an empty string on v6, and with the
            // correct value on v7
            let (_, clock) = option_read_null_terminated(option_type, option_data)?;
            Options::TraceClock(clock.into())
        }
        // UNAME: id 5, size vary
        5 => {
            let (_, uname) = option_read_null_terminated(option_type, option_data)?;
            Options::Uname(uname.into())
        }
        // HOOK: id 6, size vary
        6 => {
            let (_, hook) = option_read_null_terminated(option_type, option_data)?;
            Options::Hook(hook.into())
        }
        // OFFSET: id 7, size vary
        7 => {
            let (_, date) = option_read_null_terminated(option_type, option_data)?;
            Options::TimeOffset(option_parse_date(option_type, date)?)
        }
        // CPUCOUNT: id 8, size 4
        8 => {
            let (_, cpus) = abi.parse_u32(option_data)?;
            Options::CpuCount(cpus)
        }
        // VERSION: id 9, size vary
        9 => {
            let (_, version) = option_read_null_terminated(option_type, option_data)?;
            Options::Version(version.into())
        }
        // PROCMAPS: id 10, size vary
        10 => {
            let (_, procmaps) = option_read_null_terminated(option_type, option_data)?;
            Options::ProcMaps(procmaps.into())
        }
        // TRACEID: id 11, size 8
        11 => {
            let (_, id) = abi.parse_u64(option_data)?;
            Options::TraceId(id)
        }
        // TIME_SHIFT: id 12, size vary
        12 => {
            let (option_data, peer_trace_id) = abi.parse_u64(option_data)?;
            let (option_data, flags) = abi.parse_u32(option_data)?;
            let (option_data, nr_cpus) = abi.parse_u32(option_data)?;

            let mut cpu_corrections = Vec::new();
            let mut option_data = option_data;
            for cpu in 0..nr_cpus {
                let nr_corrections;
                (option_data, nr_corrections) = abi.parse_u32(option_data)?;
                let nr_corrections: usize = nr_corrections.try_into().unwrap();

                let mut timestamps = Vec::with_capacity(nr_corrections);
                let mut offsets = Vec::with_capacity(nr_corrections);
                let mut scaling_ratios = Vec::with_capacity(nr_corrections);

                for _ in 0..nr_corrections {
                    let x;
                    (option_data, x) = abi.parse_ulong(option_data)?;
                    timestamps.push(x);
                }
                for _ in 0..nr_corrections {
                    let x;
                    (option_data, x) = abi.parse_ulong(option_data)?;
                    offsets.push(x);
                }
                for _ in 0..nr_corrections {
                    let x;
                    (option_data, x) = abi.parse_ulong(option_data)?;
                    scaling_ratios.push(x);
                }
                let corrections: Vec<_> = izip!(timestamps, offsets, scaling_ratios,)
                    .map(|(timestamp, offset, scaling_ratio)| TimeShiftCorrection {
                        timestamp,
                        offset: offset.try_into().unwrap(),
                        scaling_ratio,
                    })
                    .collect();

                cpu_corrections.push(TimeShiftCpuCorrection { cpu, corrections });
            }
            Options::TimeShift {
                peer_trace_id,
                flags,
                cpu_corrections,
            }
        }
        // GUEST: id 13, size vary
        13 => {
            let (option_data, guest_name) = option_read_null_terminated(option_type, option_data)?;
            let (option_data, guest_trace_id) = abi.parse_u64(option_data)?;
            let (option_data, nr_cpus) = abi.parse_u32(option_data)?;

            let mut guests_cpu_info = Vec::new();
            let mut option_data = option_data;
            for _ in 0..nr_cpus {
                let cpu;
                let host_task_pid;
                (option_data, cpu) = abi.parse_u32(option_data)?;
                (option_data, host_task_pid) = abi.parse_u32(option_data)?;
                guests_cpu_info.push(GuestCpuInfo { cpu, host_task_pid })
            }
            Options::Guest {
                guest_name: guest_name.into(),
                guest_trace_id,
                guests_cpu_info,
            }
        }
        // TSC2NSEC: id 14, size 16
        14 => {
            let (option_data, multiplier) = abi.parse_u32(option_data)?;
            let (option_data, shift) = abi.parse_u32(option_data)?;
            let (_, offset) = abi.parse_u64(option_data)?;

            Options::TSC2NSec {
                multiplier,
                shift,
                offset,
            }
        }

        option_type => Options::Unknown {
            typ: option_type,
            data: option_data.to_vec(),
        },
    })
}

/// Decode options that are specific to trace.dat format v6.
fn v6_parse_options<I>(abi: &Abi, input: &mut I) -> Result<Vec<Options>, HeaderError>
where
    I: BorrowingRead,
{
    let endianness = abi.endianness;

    let mut options = Vec::new();
    let mut cpu_stat_cpu: Cpu = 0;
    loop {
        let option_type: u16 = input.read_int(endianness)?;
        if option_type == 0 {
            break;
        }
        let option_size: u32 = input.read_int(endianness)?;
        let option_data = input.read(option_size.try_into().unwrap())?;

        options.push(match option_type {
            // BUFFER: id 3, size vary
            3 => {
                let (option_data, offset) = abi.parse_u64(option_data)?;
                let (_, name) = option_read_null_terminated(option_type, option_data)?;
                Options::Instance {
                    name: name.into(),
                    offset,
                }
            }
            _ => shared_decode_option(abi, option_type, option_data, &mut cpu_stat_cpu)?,
        });
    }
    Ok(options)
}

/// Decode options that are specific to trace.dat format v7.
fn v7_parse_options<I>(
    abi: &Abi,
    decomp: &mut Option<DynDecompressor>,
    mut input: Box<I>,
) -> Result<(Box<I>, Vec<Options>), HeaderError>
where
    I: BorrowingRead,
{
    let mut options = Vec::new();
    let mut section_decomp = decomp.clone();

    loop {
        let section = {
            let (id, options) = v7_section(abi, &mut section_decomp, input.deref_mut())?;
            if id != 0 {
                Err(HeaderError::UnexpectedSection {
                    expected: 0,
                    found: id,
                })
            } else {
                Ok(options)
            }
        }?;
        let mut section_data = section.deref();

        macro_rules! read {
            ($meth:ident, $update:ident) => {{
                let (update, x) = abi.$meth($update)?;
                #[allow(unused_assignments)]
                {
                    $update = update;
                }
                x
            }};
        }

        let mut cpu_stat_cpu: Cpu = 0;
        loop {
            let option_type = read!(parse_u16, section_data);
            let option_size = read!(parse_u32, section_data);

            macro_rules! read_null_terminated {
                ($update:ident) => {{
                    let idx = $update.into_iter().position(|x| *x == 0).ok_or(
                        HeaderError::InvalidOption(
                            option_type,
                            "Could not find string null terminator".to_string(),
                        ),
                    )?;
                    let name = CStr::from_bytes_until_nul(&$update[..=idx])
                        .map_err(|err| HeaderError::InvalidOption(option_type, err.to_string()))?
                        .to_str()
                        .map_err(|err| HeaderError::InvalidOption(option_type, err.to_string()))?;
                    $update = &$update[idx + 1..];
                    name
                }};
            }

            match option_type {
                0 => {
                    let next_offset = read!(parse_u64, section_data);

                    if next_offset == 0 {
                        drop(section);
                        return Ok((input, options));
                    } else {
                        drop(section);
                        input = input.abs_seek(next_offset, None)?;
                        break;
                    }
                }
                option_type => {
                    let (mut option_data, section_data_) =
                        section_data.split_at(option_size.try_into().unwrap());
                    section_data = section_data_;

                    match option_type {
                        3 => {
                            let _offset = read!(parse_u64, option_data);
                            let name = read_null_terminated!(option_data);
                            let _clock = read_null_terminated!(option_data);
                            let page_size = read!(parse_u32, option_data);
                            let nr_cpus = read!(parse_u32, option_data);

                            for _i in 0..nr_cpus {
                                let cpu = read!(parse_u32, option_data);
                                let cpu_offset = read!(parse_u64, option_data);
                                let cpu_size = read!(parse_u64, option_data);

                                options.push(Options::Buffer {
                                    cpu,
                                    name: name.into(),
                                    offset: cpu_offset,
                                    size: cpu_size,
                                    decomp: decomp.clone(),
                                    page_size: page_size.try_into().unwrap(),
                                });
                            }
                        }

                        16 => {
                            options.push(Options::HeaderInfoLoc(read!(parse_u64, option_data)));
                        }
                        17 => {
                            options.push(Options::FtraceEventsLoc(read!(parse_u64, option_data)));
                        }
                        18 => {
                            options.push(Options::EventFormatsLoc(read!(parse_u64, option_data)));
                        }
                        19 => {
                            options.push(Options::KallsymsLoc(read!(parse_u64, option_data)));
                        }
                        20 => {
                            options.push(Options::PrintkLoc(read!(parse_u64, option_data)));
                        }
                        21 => {
                            options.push(Options::CmdLinesLoc(read!(parse_u64, option_data)));
                        }
                        _ => options.push(shared_decode_option(
                            abi,
                            option_type,
                            option_data,
                            &mut cpu_stat_cpu,
                        )?),
                    };
                }
            }
        }
    }
}

/// Parse a [Header] from a generic input.
pub fn header<I>(input: &mut I) -> Result<Header, HeaderError>
where
    I: BorrowingRead,
{
    input
        .read_tag(b"\x17\x08\x44tracing")?
        .map_err(|_| HeaderError::BadMagic)?;

    let version: u64 = {
        let version = input.read_null_terminated()?;
        let version = from_utf8(version).map_err(|err| HeaderError::DecodeUtf8(err.to_string()))?;
        version
            .parse()
            .map_err(|_| HeaderError::InvalidVersion(version.into()))
    }?;

    let mut header = match version {
        6 => {
            let (abi, page_size) = header_prefix(input)?;
            v6_header(abi, input, page_size)
        }
        7 => {
            let (abi, page_size) = header_prefix(input)?;
            v7_header(abi, input, page_size)
        }
        version => Err(HeaderError::InvalidVersion(version.to_string())),
    }?;
    header.fixup_event_descs();
    Ok(header)
}

fn header_prefix<I>(input: &mut I) -> Result<(Abi, FileSize), HeaderError>
where
    I: BorrowingRead,
{
    let endianness: u8 = input.read_int(Endianness::Little)?;
    let endianness = match endianness {
        0 => Ok(Endianness::Little),
        1 => Ok(Endianness::Big),
        x => Err(HeaderError::InvalidEndianness(x)),
    }?;

    let abi = Abi {
        // This should not be used until it's fixed to the correct value. We
        // don't use an Option<LongSize> as this would affect every downstream
        // consumer for the benefit of just one function.
        long_size: LongSize::Bits64,
        char_signedness: Signedness::Unsigned,
        endianness,
    };

    let _long_size: u8 = input.read_int(endianness)?;
    let page_size: u32 = input.read_int(endianness)?;
    let page_size: FileSize = page_size.into();

    Ok((abi, page_size))
}

macro_rules! make_read_int {
    ($input:expr, $abi:expr) => {
        macro_rules! read_int {
            () => {
                $input.read_int($abi.endianness)
            };
        }
    };
}

fn v7_header<I>(abi: Abi, input: &mut I, page_size: FileSize) -> Result<Header, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    let comp_codec = {
        let s = input.read_null_terminated()?;
        from_utf8(s)
            .map_err(|err| HeaderError::DecodeUtf8(err.to_string()))?
            .to_string()
    };

    let _comp_codec_version = {
        let s = input.read_null_terminated()?;
        from_utf8(s)
            .map_err(|err| HeaderError::DecodeUtf8(err.to_string()))?
            .to_string()
    };

    // TODO: does the version of the codec matter at all ?
    let mut decomp = match &*comp_codec {
        "none" => Ok(None),
        "zstd" => Ok(Some(DynDecompressor::new(ZstdDecompressor::new()))),
        "zlib" => Ok(Some(DynDecompressor::new(ZlibDecompressor::new()))),
        _ => Err(HeaderError::UnsupportedCompressionCodec(comp_codec)),
    }?;

    let options_offset: u64 = read_int!()?;

    let input = input.clone_and_seek(options_offset, None)?;

    let (mut input, options) = v7_parse_options(&abi, &mut decomp, input)?;

    let mut event_descs = Vec::new();
    let mut kallsyms = BTreeMap::new();
    let mut str_table = BTreeMap::new();
    let mut pid_comms = BTreeMap::new();
    let mut abi = abi;

    macro_rules! get_section {
        ($expected_id:expr, $offset:expr) => {{
            let expected = $expected_id;
            input = input.abs_seek($offset, None)?;
            let (found, section) = v7_section(&abi, &mut decomp, input.deref_mut())?;

            if expected != found {
                Err(HeaderError::UnexpectedSection { expected, found })
            } else {
                Ok(())
            }?;

            // Use Rc<ScratchBox<'a, T>> so that we can cheaply clone the
            // ScratchBox. We need NestedPointer to have Rc<ScratchBox<T>>
            // implement AsRef<T>. We need to pass the ScratchBox directly
            // instead of borrowing otherwise ownership of the actual box is
            // tedious (mainly because of the loop implying the need to
            // re-assign to the owning variable).
            let section = BorrowingCursor::new(NestedPointer::new(Rc::new(section)));
            Ok::<_, HeaderError>(section)
        }};
    }

    // Ensure we have accurate Abi information before doing anything else
    for option in &options {
        if let Options::HeaderInfoLoc(offset) = option {
            let mut section = get_section!(16, *offset)?;
            abi = parse_header_info_section(abi, &mut section)?;
        }
    }

    let mut nr_cpus = 0;
    for option in &options {
        match option {
            Options::FtraceEventsLoc(offset) => {
                let mut section = get_section!(17, *offset)?;
                event_descs.extend(parse_subsystem_event_formats(&abi, &mut section)?);
            }
            Options::EventFormatsLoc(offset) => {
                let mut section = get_section!(18, *offset)?;
                event_descs.extend(parse_event_formats_section(&abi, &mut section)?);
            }
            Options::KallsymsLoc(offset) => {
                let mut section = get_section!(19, *offset)?;
                kallsyms.extend(parse_kallsyms_section(&abi, &mut section)?);
            }
            Options::PrintkLoc(offset) => {
                let mut section = get_section!(20, *offset)?;
                str_table.extend(parse_printk_section(&abi, &mut section)?);
            }
            Options::CmdLinesLoc(offset) => {
                let mut section = get_section!(21, *offset)?;
                pid_comms.extend(parse_cmdlines_section(&abi, &mut section)?);
            }

            Options::Buffer { cpu, .. } => {
                nr_cpus = std::cmp::max(nr_cpus, *cpu);
            }
            Options::CpuCount(_nr_cpus) => {
                nr_cpus = std::cmp::max(nr_cpus, *_nr_cpus);
            }
            _ => (),
        }
    }

    Ok(Header {
        inner: VersionedHeader::V7(HeaderV7 {
            kernel_abi: abi,
            page_size,
            event_descs,
            kallsyms,
            str_table,
            pid_comms,
            options,
            nr_cpus,
        }),
        typ_decoders: Arc::new(RwLock::new(BTreeMap::new())),
    })
}

fn v7_section<'a, I, C>(
    abi: &Abi,
    decomp: &'a mut Option<C>,
    input: &'a mut I,
) -> Result<(SectionId, Cow<'a, [u8]>), HeaderError>
where
    I: BorrowingRead,
    C: Decompressor,
{
    make_read_int!(input, abi);

    let id: u16 = read_int!()?;
    let flags: u16 = read_int!()?;
    let compressed = flags & 0x1 != 0;

    // Description of the section, stored in a string table
    let _string_id: u32 = read_int!()?;
    let size: u64 = read_int!()?;
    let size: usize = size
        .try_into()
        .map_err(|_| HeaderError::SectionTooLarge(size))?;

    let data = if compressed {
        let compressed_size: u32 = read_int!()?;
        assert_eq!(compressed_size as usize, size - 8);

        let decompressed_size: u32 = read_int!()?;
        let decompressed_size: usize = decompressed_size
            .try_into()
            .map_err(|_| HeaderError::SectionTooLarge(decompressed_size.into()))?;
        let data = input.read(compressed_size.try_into().unwrap())?;
        match decomp {
            Some(decomp) => decomp.decompress(data, decompressed_size).map(Cow::Owned)?,
            None => Err(HeaderError::CompressedSectionWithoutCodec)?,
        }
    } else {
        input.read(size).map(Cow::Borrowed)?
    };

    Ok((id, data))
}

fn parse_event_formats_section<I>(abi: &Abi, input: &mut I) -> Result<Vec<EventDesc>, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    let mut event_descs: Vec<EventDesc> = Vec::new();
    let nr_event_systems: u32 = read_int!()?;

    for _ in 0..nr_event_systems {
        let _system_name = from_utf8(input.read_null_terminated()?)
            .map_err(|err| HeaderError::DecodeUtf8(err.to_string()))?;

        event_descs.extend(parse_subsystem_event_formats(abi, input)?)
    }
    Ok(event_descs)
}

fn parse_subsystem_event_formats<I>(abi: &Abi, input: &mut I) -> Result<Vec<EventDesc>, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    let mut event_descs: Vec<EventDesc> = Vec::new();
    let nr_event_descs: u32 = read_int!()?;

    for _ in 0..nr_event_descs {
        let desc_size: u64 = read_int!()?;
        let desc_size: usize = desc_size
            .try_into()
            .map_err(|_| HeaderError::EventDescTooLarge(desc_size))?;
        let desc = input.parse(desc_size, parse_event_desc)??;
        event_descs.push(desc);
    }
    Ok(event_descs)
}

fn parse_kallsyms_section<I>(
    abi: &Abi,
    input: &mut I,
) -> Result<BTreeMap<Address, SymbolName>, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    let kallsyms_size: u32 = read_int!()?;
    let kallsyms_size: usize = kallsyms_size
        .try_into()
        .map_err(|_| HeaderError::KallsymsTooLarge(kallsyms_size.into()))?;
    input.parse(kallsyms_size, parse_kallsyms)?
}

fn parse_printk_section<'a, 'abi: 'a, I>(
    abi: &'abi Abi,
    input: &'a mut I,
) -> Result<BTreeMap<Address, String>, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    let str_table_size: u32 = read_int!()?;
    let str_table_size: usize = str_table_size
        .try_into()
        .map_err(|_| HeaderError::StrTableTooLarge(str_table_size.into()))?;
    input.parse(str_table_size, parse_str_table)?
}

fn parse_cmdlines_section<I>(
    abi: &Abi,
    input: &mut I,
) -> Result<BTreeMap<Pid, TaskName>, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    let pid_comms_size: u64 = read_int!()?;
    let pid_comms_size: usize = pid_comms_size.try_into().unwrap();
    input.parse(pid_comms_size, parse_pid_comms)?
}

fn parse_header_info_section<I>(abi: Abi, input: &mut I) -> Result<Abi, HeaderError>
where
    I: BorrowingRead,
{
    make_read_int!(input, abi);

    // Header page
    input
        .read_tag(b"header_page\0")?
        .map_err(|_| HeaderError::ExpectedHeaderPage)?;
    let page_header_size_u64: u64 = read_int!()?;
    let page_header_size: usize = page_header_size_u64
        .try_into()
        .map_err(|_| HeaderError::PageHeaderSizeTooLarge(page_header_size_u64))?;

    let header_fields = input.parse(
        page_header_size,
        // Disable type check due to:
        // https://bugzilla.kernel.org/show_bug.cgi?id=216999
        |input| parse_struct_fmt(&BasicEnv::new(&abi), true, input),
    )??;

    // Fixup ABI with long_size
    let long_size = match header_fields.field_by_name("commit") {
        Some(commit) => commit.size.try_into().map_err(HeaderError::InvalidLongSize),
        None => Err(HeaderError::LongSizeNotFound),
    }?;

    let char_signedness = match header_fields.field_by_name("data") {
        Some(data) => match &data.declaration.typ {
            Type::U8 => Ok(Signedness::Unsigned),
            Type::I8 => Ok(Signedness::Unsigned),
            _ => Err(HeaderError::CharSignednessNotFound),
        },
        None => Err(HeaderError::CharSignednessNotFound),
    }?;

    let abi = Abi {
        long_size,
        char_signedness,
        ..abi
    };

    // Header event
    input
        .read_tag(b"header_event\0")?
        .map_err(|_| HeaderError::ExpectedHeaderEvent)?;
    let header_event_size: u64 = read_int!()?;
    let header_event_size: usize = header_event_size
        .try_into()
        .map_err(|_| HeaderError::HeaderEventSizeTooLarge(header_event_size))?;
    input.parse(header_event_size, parse_header_event)??;

    Ok(abi)
}

fn v6_header<I>(abi: Abi, input: &mut I, page_size: FileSize) -> Result<Header, HeaderError>
where
    I: BorrowingRead,
{
    let endianness = abi.endianness;
    let abi = parse_header_info_section(abi, input)?;

    let mut event_descs = parse_subsystem_event_formats(&abi, input)?;

    event_descs.extend(parse_event_formats_section(&abi, input)?);

    let kallsyms = parse_kallsyms_section(&abi, input)?;
    let str_table = parse_printk_section(&abi, input)?;
    let pid_comms = parse_cmdlines_section(&abi, input)?;

    let nr_cpus: u32 = input.read_int(endianness)?;

    let mut options = Vec::new();
    loop {
        let data_kind = input.read_null_terminated()?;
        match data_kind {
            b"options  " => {
                options.extend(v6_parse_options(&abi, input)?);
            }
            kind => {
                let kind = kind.to_owned();
                let top_level_buffer_locations = buffer_locations(&kind, nr_cpus, &abi, "", input)?;
                break Ok(Header {
                    inner: VersionedHeader::V6(HeaderV6 {
                        kernel_abi: abi,
                        page_size,
                        event_descs,
                        kallsyms,
                        str_table,
                        pid_comms,
                        top_level_buffer_locations,
                        options,
                        nr_cpus,
                    }),
                    typ_decoders: Arc::new(RwLock::new(BTreeMap::new())),
                });
            }
        }
    }
}

pub(crate) fn buffer_locations<I>(
    kind: &[u8],
    nr_cpus: Cpu,
    abi: &Abi,
    name: &str,
    input: &mut I,
) -> Result<Vec<BufferLocation>, HeaderError>
where
    I: BorrowingRead,
{
    let endianness = abi.endianness;
    match kind {
        b"flyrecord" => (0..nr_cpus)
            .map(|cpu| {
                let offset: u64 = input.read_int(endianness)?;
                let size: u64 = input.read_int(endianness)?;
                Ok(BufferLocation {
                    id: BufferId {
                        cpu,
                        name: name.into(),
                    },
                    offset,
                    size,
                })
            })
            .collect::<Result<Vec<_>, _>>(),
        b"latency  " => Err(HeaderError::UnsupportedDataFmt(Some("latency".into()))),
        kind => Err(HeaderError::UnsupportedDataFmt(
            from_utf8(kind).map(Into::into).ok(),
        )),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        parser::tests::test_parser,
        print::{PrintFlags, PrintPrecision, PrintWidth, VBinSpecifier},
    };

    #[derive(Debug, PartialEq)]
    struct EventDescContent {
        name: String,
        id: EventId,
        fmt: EventFmt,
    }

    #[test]
    fn event_desc_parser_test() {
        let abi = Abi {
            long_size: LongSize::Bits64,
            endianness: Endianness::Little,
            char_signedness: Signedness::Unsigned,
        };
        let test = |fmt: &[u8], expected: EventDescContent| {
            let header = Header {
                inner: VersionedHeader::V6(HeaderV6 {
                    kernel_abi: abi.clone(),
                    page_size: 4096,
                    event_descs: Vec::new(),
                    kallsyms: BTreeMap::new(),
                    str_table: BTreeMap::new(),
                    pid_comms: BTreeMap::new(),
                    options: Vec::new(),
                    top_level_buffer_locations: Vec::new(),
                    nr_cpus: 0,
                }),
                typ_decoders: Arc::new(RwLock::new(BTreeMap::new())),
            };
            let header = Arc::new(header);

            let parser = parse_event_desc.map(|mut desc| {
                desc.header = Some(Arc::clone(&header));

                EventDescContent {
                    name: desc.name.clone(),
                    id: desc.id,
                    fmt: desc
                        .event_fmt()
                        .cloned()
                        .expect("Error while computing EventFmt"),
                }
            });
            test_parser(expected, fmt, parser)
        };

        macro_rules! new_variable_atom {
            ($($args:expr),* $(,)?) => {
                PrintAtom::new_variable(
                    &abi,
                    $($args),*
                )
            }
        }

        let noop_decoder = Arc::new(closure!(
            (
                for<'d> Fn(
                    &'d [u8],
                    &'d [u8],
                    &'d Header,
                    &'d ScratchAlloc,
                )
                -> Result<Value<'d>, BufferError>
            ),
            |_, _, _, _| Ok(Value::Unknown)
        ));

        test(
            b"name: wakeup\nID: 3\nformat:\n\tfield:unsigned short common_type;\toffset:0;\tsize:2;\tsigned:0;\n\tfield:unsigned char common_flags;\toffset:2;\tsize:1;\tsigned:0;\n\tfield:unsigned char common_preempt_count;\toffset:3;\tsize:1;\tsigned:0;\n\tfield:int common_pid;\toffset:4;\tsize:4;\tsigned:1;\n\n\tfield:unsigned int prev_pid;\toffset:8;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_pid;\toffset:12;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_cpu;\toffset:16;\tsize:4;\tsigned:0;\n\tfield:unsigned char prev_prio;\toffset:20;\tsize:1;\tsigned:0;\n\tfield:unsigned char prev_state;\toffset:21;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_prio;\toffset:22;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_state;\toffset:23;\tsize:1;\tsigned:0;\n\nprint fmt: \"%u:%u:%u  ==+ %u:%u:%u \\\" \\t [%03u]\", __builtin_choose_expr(1, 55, 56)\n",
            EventDescContent {
                name: "wakeup".into(),
                id: 3,
                fmt: EventFmt {
                    print_fmt_args: Ok((PrintFmtStr {
                        vbin_decoders: OnceCell::new(),
                        atoms: vec![
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed(":".into()),
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed(":".into()),
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed("  ==+ ".into()),
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed(":".into()),
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed(":".into()),
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed(" \" \t [".into()),
                            new_variable_atom!(
                                VBinSpecifier::U32,
                                PrintSpecifier::Dec,
                                PrintFlags::ZeroPad,
                                PrintWidth::Fixed(3),
                                PrintPrecision::Unmodified,
                            ),
                            PrintAtom::Fixed("]".into()),

                        ],
                    }, vec![])),
                    struct_fmt: Ok(StructFmt {
                        fields: vec![
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_type".into(),
                                    typ: Type::U16,
                                },
                                offset: 0,
                                size: 2,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_flags".into(),
                                    typ: Type::U8,
                                },
                                offset: 2,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_preempt_count".into(),
                                    typ: Type::U8,
                                },
                                offset: 3,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_pid".into(),
                                    typ: Type::I32,
                                },
                                offset: 4,
                                size: 4,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "prev_pid".into(),
                                    typ: Type::U32,
                                },
                                offset: 8,
                                size: 4,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_pid".into(),
                                    typ: Type::U32,
                                },
                                offset: 12,
                                size: 4,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_cpu".into(),
                                    typ: Type::U32,
                                },
                                offset: 16,
                                size: 4,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "prev_prio".into(),
                                    typ: Type::U8,
                                },
                                offset: 20,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "prev_state".into(),
                                    typ: Type::U8,
                                },
                                offset: 21,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_prio".into(),
                                    typ: Type::U8,
                                },
                                offset: 22,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_state".into(),
                                    typ: Type::U8,
                                },
                                offset: 23,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                        ]
                    })
                }
            }

        );

        test(
            b"name: user_stack\nID: 12\nformat:\n\tfield:unsigned short common_type;\toffset:0;\tsize:2;\tsigned:0;\n\tfield:unsigned char common_flags;\toffset:2;\tsize:1;\tsigned:0;\n\tfield:unsigned char common_preempt_count;\toffset:3;\tsize:1;\tsigned:0;\n\tfield:int common_pid;\toffset:4;\tsize:4;\tsigned:1;\n\n\tfield:unsigned int tgid;\toffset:8;\tsize:4;\tsigned:0;\n\tfield:unsigned long caller[8];\toffset:16;\tsize:64;\tsigned:0;\n\nprint fmt: \"\\t=> %ps\", (void *)REC->caller[0], (void *)REC->caller[1]\n",
            EventDescContent {
                name: "user_stack".into(),
                id: 12,
                fmt: EventFmt {
                    print_fmt_args: Ok((PrintFmtStr {
                        vbin_decoders: OnceCell::new(),
                        atoms: vec![
                            PrintAtom::Fixed("\t=> ".into()),
                            new_variable_atom!(
                                VBinSpecifier::U64,
                                PrintSpecifier::Symbol,
                                PrintFlags::empty(),
                                PrintWidth::Unmodified,
                                PrintPrecision::Unmodified,
                            ),
                        ]
                    }, vec![])),
                    struct_fmt: Ok(StructFmt {
                        fields: vec![
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_type".into(),
                                    typ: Type::U16,
                                },
                                offset: 0,
                                size: 2,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_flags".into(),
                                    typ: Type::U8,
                                },
                                offset: 2,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_preempt_count".into(),
                                    typ: Type::U8,
                                },
                                offset: 3,
                                size: 1,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_pid".into(),
                                    typ: Type::I32,
                                },
                                offset: 4,
                                size: 4,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "tgid".into(),
                                    typ: Type::U32,
                                },
                                offset: 8,
                                size: 4,
                                decoder: noop_decoder.clone(),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "caller".into(),
                                    typ: Type::Array(Box::new(Type::U64), ArrayKind::Fixed(Ok(8))),
                                },
                                offset: 16,
                                size: 64,
                                decoder: noop_decoder.clone(),
                            },
                        ]
                    })
                }
            }
        );
    }
}
