use core::{
    borrow::Borrow,
    convert::TryFrom,
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
    rc::Rc,
    string::{String as StdString, ToString},
    sync::{Arc, RwLock},
};

use nom::{
    branch::alt,
    bytes::complete::{is_a, is_not, tag},
    character::complete::{
        char, multispace0, multispace1, u16 as txt_u16, u32 as txt_u32, u64 as txt_u64,
    },
    combinator::{all_consuming, iterator, map_res, opt, rest},
    error::context,
    multi::{fold_many0, many0, separated_list0},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Finish as _, Parser,
};
use once_cell::sync::OnceCell;
use smartstring::alias::String;

use crate::{
    array::Array,
    buffer::{Buffer, BufferError, FieldDecoder},
    cinterp::{CompileEnv, CompileError, EvalEnv, EvalError, new_dyn_evaluator, Evaluator, Value},
    closure::closure,
    compress::{Decompressor, DynDecompressor, ZlibDecompressor, ZstdDecompressor},
    cparser::{
        identifier, is_identifier, string_literal, ArrayKind, CGrammar, CGrammarCtx, Declaration,
        Expr, ExtensionMacroCall, ExtensionMacroCallCompiler, ExtensionMacroCallType,
        ExtensionMacroDesc, Type,
    },
    grammar::PackratGrammar as _,
    io::{BorrowingCursor, BorrowingRead},
    nested_pointer::NestedPointer,
    parser::{
        error, failure, hex_u64, lexeme, map_res_cut, to_str, FromParseError, NomError,
        NomParserExt as _, VerboseParseError,
    },
    print::{parse_print_fmt, PrintAtom, PrintFmtError, PrintFmtStr, PrintSpecifier, StringWriter},
    scratch::{ScratchAlloc, ScratchVec},
    str::Str,
};

pub type Address = u64;
pub type AddressOffset = Address;
pub type AddressSize = Address;
pub type CPU = u32;
pub type PID = u32;
pub type Timestamp = u64;
pub type TimeOffset = i64;
pub type SymbolName = String;
pub type TaskName = String;
pub type Identifier = String;
pub type EventId = u16;

pub type FileOffset = u64;
pub type FileSize = FileOffset;

pub type MemOffset = usize;
pub type MemSize = MemOffset;
pub type MemAlign = MemOffset;

pub type SectionId = u16;

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum Endianness {
    Big,
    Little,
}

macro_rules! parse_N {
    ($name:ident, $typ:ty) => {
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
    fn native() -> Self {
        if cfg!(target_endian = "big") {
            Endianness::Big
        } else if cfg!(target_endian = "little") {
            Endianness::Little
        } else {
            panic!("Cannot handle endianness")
        }
    }

    pub fn is_native(&self) -> bool {
        self == &Self::native()
    }

    parse_N!(parse_u64, u64);
    parse_N!(parse_u32, u32);
    parse_N!(parse_u16, u16);
    parse_N!(parse_u8, u8);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LongSize {
    Bits32,
    Bits64,
}

impl From<LongSize> for u64 {
    fn from(size: LongSize) -> Self {
        match size {
            LongSize::Bits32 => 4,
            LongSize::Bits64 => 8,
        }
    }
}
impl From<LongSize> for usize {
    fn from(size: LongSize) -> Self {
        match size {
            LongSize::Bits32 => 4,
            LongSize::Bits64 => 8,
        }
    }
}

impl TryFrom<usize> for LongSize {
    type Error = usize;

    fn try_from(size: usize) -> Result<Self, Self::Error> {
        match size {
            4 => Ok(LongSize::Bits32),
            8 => Ok(LongSize::Bits64),
            x => Err(x),
        }
    }
}

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
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Abi {
    pub endianness: Endianness,
    pub long_size: LongSize,
    pub char_signedness: Signedness,
}

macro_rules! abi_parse_N {
    ($name:ident, $typ:ty) => {
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

    #[inline]
    pub fn char_typ(&self) -> Type {
        match self.char_signedness {
            Signedness::Unsigned => Type::U8,
            Signedness::Signed => Type::I8,
        }
    }

    #[inline]
    pub fn long_typ(&self) -> Type {
        match self.long_size {
            LongSize::Bits32 => Type::I32,
            LongSize::Bits64 => Type::I64,
        }
    }

    #[inline]
    pub fn ulong_typ(&self) -> Type {
        match self.long_size {
            LongSize::Bits32 => Type::U32,
            LongSize::Bits64 => Type::U64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferId {
    pub cpu: CPU,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct BufferLocation {
    pub id: BufferId,
    pub offset: FileOffset,
    pub size: FileSize,
}

#[derive(Debug, Clone)]
pub(crate) struct HeaderV6 {
    pub(crate) kernel_abi: Abi,
    pub(crate) page_size: FileSize,
    pub(crate) event_descs: Vec<EventDesc>,
    pub(crate) kallsyms: BTreeMap<Address, SymbolName>,
    pub(crate) str_table: BTreeMap<Address, String>,
    pub(crate) pid_comms: BTreeMap<PID, TaskName>,
    pub(crate) options: Vec<Options>,
    pub(crate) top_level_buffer_locations: Vec<BufferLocation>,
    pub(crate) nr_cpus: CPU,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct HeaderV7 {
    pub(crate) kernel_abi: Abi,
    pub(crate) page_size: FileSize,
    pub(crate) event_descs: Vec<EventDesc>,
    pub(crate) kallsyms: BTreeMap<Address, SymbolName>,
    pub(crate) str_table: BTreeMap<Address, String>,
    pub(crate) pid_comms: BTreeMap<PID, TaskName>,
    pub(crate) options: Vec<Options>,
    pub(crate) nr_cpus: CPU,
}

#[derive(Debug, Clone)]
enum VersionedHeader {
    V6(HeaderV6),
    V7(HeaderV7),
}

#[derive(Debug, Clone)]
pub struct Header {
    // We have this inner layer so the publicly exposed struct is completely
    // opaque. An enum cannot be opaque.
    inner: VersionedHeader,
}

macro_rules! attr {
    ($header:expr, $attr:ident) => {
        match $header {
            Header {
                inner:
                    VersionedHeader::V6(HeaderV6 { $attr, .. })
                    | VersionedHeader::V7(HeaderV7 { $attr, .. }),
            } => $attr,
        }
    };
}

impl Header {
    #[inline]
    pub fn deref_static(&self, addr: u64) -> Result<Value<'_>, EvalError> {
        match attr!(self, str_table).get(&addr) {
            Some(s) => Ok(Value::Str(Str::new_borrowed(s.deref()))),
            None => Err(EvalError::CannotDeref),
        }
    }

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
    pub fn kernel_abi(&self) -> &Abi {
        attr!(self, kernel_abi)
    }

    #[inline]
    pub fn comm_of(&self, pid: PID) -> Option<&TaskName> {
        attr!(self, pid_comms).get(&pid)
    }

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

    #[inline]
    pub fn nr_cpus(&self) -> CPU {
        *attr!(self, nr_cpus)
    }

    #[inline]
    pub fn options(&self) -> impl IntoIterator<Item = &Options> {
        attr!(self, options)
    }

    #[inline]
    pub fn kallsyms(&self) -> impl IntoIterator<Item=(Address, &str)> {
        attr!(self, kallsyms).into_iter().map(|(k, v)| {
            (*k, v.deref())
        })
    }

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

    pub fn buffers<'a, I: BorrowingRead + Send + 'a>(
        &'a self,
        input: I,
    ) -> Result<Vec<Buffer<'a>>, BufferError> {
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
}

#[derive(Clone)]
pub struct FieldFmt {
    pub declaration: Declaration,
    pub offset: MemOffset,
    pub size: MemSize,

    pub decoder: Arc<dyn FieldDecoder>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct StructFmt {
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

type HeaderNomError<'a> = NomError<HeaderError, nom::error::VerboseError<&'a [u8]>>;

#[inline(never)]
fn parse_struct_fmt<'a>(
    abi: &Abi,
    skip_fixup: bool,
    input: &'a [u8],
) -> nom::IResult<&'a [u8], StructFmt, HeaderNomError<'a>> {
    terminated(
        separated_list0(
            char('\n'),
            map_res_cut(
                preceded(
                    lexeme(tag(b"field:")),
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
                        &CGrammarCtx::new(abi),
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
                        declaration.typ = fixup_c_type(declaration.typ, size, signedness, abi)?;
                    }

                    Ok(FieldFmt {
                        declaration,
                        offset: get!("offset"),
                        size,
                        decoder: Arc::new(()),
                    })
                },
            ),
        ),
        opt(char('\n')),
    )
    .map(|fields| StructFmt { fields })
    .parse(input)
}

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
                            tuple((
                                lexeme(tag("data")),
                                lexeme(tag("max")),
                                lexeme(tag("type_len")),
                                lexeme(tag("==")),
                            )),
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

#[derive(Debug, Clone)]
pub struct EventDesc {
    pub name: String,
    pub id: EventId,
    // Use a OnceCell so that we can mutate it in place in order to lazily parse
    // the format and memoize the result.
    fmt: OnceCell<Result<EventFmt, HeaderError>>,
    raw_fmt: Vec<u8>,
    header: Option<Arc<Header>>,
}

#[derive(Clone)]
pub struct EventFmt {
    struct_fmt: Result<StructFmt, HeaderError>,
    print_fmt: Result<PrintFmtStr, HeaderError>,
    print_args: Result<Vec<Result<Arc<dyn Evaluator>, CompileError>>, HeaderError>,
}

impl EventFmt {
    pub fn struct_fmt(&self) -> Result<&StructFmt, HeaderError> {
        match &self.struct_fmt {
            Ok(x) => Ok(x),
            Err(err) => Err(err.clone()),
        }
    }

    pub fn print_fmt(&self) -> Result<&PrintFmtStr, HeaderError> {
        match &self.print_fmt {
            Ok(x) => Ok(x),
            Err(err) => Err(err.clone()),
        }
    }

    pub fn print_args(&self) -> Result<&[Result<Arc<dyn Evaluator>, CompileError>], HeaderError> {
        match &self.print_args {
            Ok(x) => Ok(x),
            Err(err) => Err(err.clone()),
        }
    }
}

impl Debug for EventFmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("EventFmt")
            .field("struct_fmt", &self.struct_fmt)
            .field("print_fmt", &self.print_fmt)
            .finish_non_exhaustive()
    }
}

impl PartialEq<Self> for EventFmt {
    fn eq(&self, other: &Self) -> bool {
        self.struct_fmt == other.struct_fmt && self.print_fmt == other.print_fmt
    }
}

impl EventDesc {
    // Allow for errors in case we decide to drop the raw_fmt once it has been parsed
    #[inline]
    pub fn raw_fmt(&self) -> Result<&[u8], HeaderError> {
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

#[inline(never)]
fn parse_event_fmt<'a>(
    header: &'a Header,
    name: &'a str,
    input: &[u8],
) -> Result<EventFmt, HeaderError> {
    context(
            "event description",
            tuple((
                context(
                    "event format",
                    preceded(
                        pair(lexeme(tag("format:")), multispace0),
                        |input| parse_struct_fmt(header.kernel_abi(), false, input),
                    )
                    .map(|mut fmt| {
                        for field in &mut fmt.fields {
                            field.decoder =
                                Arc::new(match field.declaration.typ.make_decoder(header) {
                                    Ok(parser) => parser,
                                    Err(err) => {
                                        Box::new(closure!(
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
                                        ))
                                    }
                                });
                        }
                        fmt
                    }),
                ),
                context(
                    "event print fmt",
                    preceded(
                        pair(multispace0, lexeme(tag("print fmt:"))),
                        all_consuming(lexeme(map_res_cut(
                            move |input| {
                                match CGrammar::apply_rule(
                                    CGrammar::expr(),
                                    input,
                                    &CGrammarCtx::new(header.kernel_abi()),
                                ) {
                                    Ok((remaining, x)) => {
                                        let consumed = input.len() - remaining.len();
                                        Ok((&input[consumed..], x))
                                    },
                                    Err(err) => failure(
                                        input,
                                        HeaderError::InvalidPrintkFmt(Box::new(PrintFmtError::CParseError(Box::new(err)))),
                                    ),
                                }
                            },
                            |expr| {
                                let (fmt, exprs) = match expr {
                                    Expr::CommaExpr(vec) => {
                                        let mut iter = vec.into_iter();
                                        let s = iter.next();
                                        let args: Vec<Expr> = iter.collect();

                                        let s = match s {
                                            Some(Expr::StringLiteral(s)) => Ok(s),
                                            Some(_expr) => Err(HeaderError::InvalidPrintkFmt(Box::new(PrintFmtError::NotAStringLiteral))),
                                            None => panic!("CommaExpr is expected to contain at least one expression"),
                                        }?;
                                        Ok((s, args))
                                    }
                                    Expr::StringLiteral(s) => Ok((s, vec![])),
                                    _expr => Err(HeaderError::InvalidPrintkFmt(Box::new(PrintFmtError::NotAStringLiteral))),
                                }?;
                                let fmt = parse_print_fmt(header, fmt.as_bytes())
                                    .map_err(|err| HeaderError::InvalidPrintkFmt(Box::new(err)));
                                Ok((fmt, exprs))
                            },
                        ))),
                    ),
                ),
            ))
            .map(|(struct_fmt, (print_fmt, print_args))| {
                struct CEnv<'h> {
                    header: &'h Header,
                    struct_fmt: &'h StructFmt,
                    scratch: ScratchAlloc,
                }

                impl<'h> EvalEnv<'h> for CEnv<'h> {
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
                }

                impl<'ce> CompileEnv<'ce> for CEnv<'ce> {
                    #[inline]
                    fn field_typ(&self, id: &str) -> Result<Type, CompileError> {
                        for field in &self.struct_fmt.fields {
                            if field.declaration.identifier == id {
                                return Ok(field.declaration.typ.clone());
                            }
                        }
                        Err(CompileError::UnknownField)
                    }

                    #[inline]
                    fn field_getter(
                        &self,
                        id: &str,
                    ) -> Result<
                        Box<dyn Evaluator>,
                        CompileError,
                    > {
                        for field in &self.struct_fmt.fields {
                            if field.declaration.identifier == id {
                                let decoder = field.decoder.clone();
                                let offset = field.offset;
                                let end = offset + field.size;

                                fn make_box<F>(f: F) -> Box<dyn Evaluator>
                                where
                                    F: for<'ee, 'eeref> Fn(&'eeref (dyn EvalEnv<'ee> + 'eeref)) -> Result<Value<'eeref>, EvalError> + Send + Sync + 'static {
                                    Box::new(f)
                                }

                                return Ok(make_box(move |env: &dyn EvalEnv<'_>| {
                                    let event_data = env.event_data()?;
                                    let header = env.header()?;
                                    let field_data = &event_data[offset..end];
                                    Ok(decoder.decode(
                                        event_data,
                                        field_data,
                                        header,
                                        env.scratch(),
                                    )?)
                                }));
                            }
                        }
                        Err(CompileError::UnknownField)
                    }
                }

                let env = CEnv {
                    header,
                    struct_fmt: &struct_fmt,
                    scratch: ScratchAlloc::new(),
                };

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
                                        ret_typ: ExtensionMacroCallType::Type(Type::Pointer(Box::new($char_typ))),
                                        compiler: Arc::new(|cenv: &dyn CompileEnv, _abi: &_| {
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
                                                        _ => Err(EvalError::CannotDeref)
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
                                                                let fmt = fmt.to_str().ok_or(EvalError::IllegalType)?;

                                                                let fmt = parse_print_fmt(header, fmt.as_bytes())?;
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
                                                        _ => Err(EvalError::IllegalType),
                                                    }
                                                }
                                            }))
                                        })
                                    }
                                }
                            }

                            let char_typ = header.kernel_abi().char_typ();

                            let compiler = compiler!(char_typ.clone());
                            let desc = ExtensionMacroDesc::new_function_like(
                                "__format_vbin_printf".into(),
                                Box::new(move |input| {
                                    Ok((
                                        input,
                                        compiler!(char_typ.clone()),
                                    ))
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

                let print_args = match &print_fmt {
                    Ok(print_fmt) => {
                        Ok(PrintAtom::zip_atoms(
                                print_args.into_iter(),
                                print_fmt.atoms.iter()
                            )
                            .into_iter()
                            .map(fixup_arg)
                            .map(|expr| Ok(
                                Arc::from(expr.compile(&env, header.kernel_abi())?))
                            )
                            .collect())
                    }
                    Err(err) => Err(err.clone()),
                };
                // TODO: We could possibly exploit e.g. __print_symbolic() to fixup the enum
                // variants, based on the print_fmt, but:
                // 1. The strings are arbitrary and cannot be expected to be valid identifier
                //    matching any original enum. That is probably fine but might be a bit
                //    surprising.
                // 2. We cannot count on the strings described by __print_symbolic() cover all the
                //    cases.
                EventFmt {
                    // For now, we just fail if we cannot at least parse the StructFmt, since the
                    // resulting EventFmt would be quite useless. This might change in the future
                    // if necessary.
                    struct_fmt: Ok(struct_fmt),
                    print_fmt,
                    print_args,
                }
            }),
        )
        .parse_finish(input)
}

#[inline(never)]
fn parse_event_desc(input: &[u8]) -> nom::IResult<&[u8], EventDesc, HeaderNomError<'_>> {
    context(
        "event description",
        map_res_cut(
            tuple((
                context(
                    "event name",
                    preceded(
                        lexeme(tag("name:")),
                        lexeme(terminated(is_not("\n"), char('\n'))),
                    ),
                ),
                context("event ID", preceded(lexeme(tag("ID:")), lexeme(txt_u16))),
                context("remainder", rest),
            )),
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
                        Err(_) => Err(HeaderError::InvalidSymbolName),
                    },
                ),
            ),
            char('\n'),
        );

        let mut it = iterator(input, line);
        let parsed = it
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
                    Ok((_, _expr)) => failure(
                        input,
                        HeaderError::InvalidPrintkFmt(Box::new(PrintFmtError::NotAStringLiteral)),
                    ),
                    Err(err) => err.into_external(input, |data| {
                        HeaderError::InvalidPrintkFmt(Box::new(PrintFmtError::CParseError(
                            Box::new(data),
                        )))
                    }),
                }
            },
        );
        let mut it = iterator(input, line);
        let parsed = it.collect::<BTreeMap<_, _>>();
        let (input, _) = it.finish()?;
        Ok((input, parsed))
    })
    .parse(input)
}

#[inline(never)]
fn parse_pid_comms(input: &[u8]) -> nom::IResult<&[u8], BTreeMap<PID, String>, HeaderNomError<'_>> {
    context("PID map", move |input| {
        let line = separated_pair(
            txt_u32,
            multispace1,
            map_res_cut(lexeme(is_not("\n")), |x| match from_utf8(x) {
                Ok(s) => Ok(s.into()),
                Err(_) => Err(HeaderError::InvalidComm),
            }),
        );
        let mut it = iterator(input, line);
        let parsed = it.collect::<BTreeMap<_, _>>();
        let (input, _) = it.finish()?;
        Ok((input, parsed))
    })
    .parse(input)
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
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

    #[error("Could not parse symbol name")]
    InvalidSymbolName,

    #[error("Could not parse C identifier")]
    InvalidCIdentifier,

    #[error("{0}")]
    InvalidPrintkFmt(Box<PrintFmtError>),

    #[error("Could not parse printk argument")]
    InvalidPrintkArg,

    #[error("Could not parse command name")]
    InvalidComm,

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

    #[error("Invalid string literal")]
    InvalidStringLiteral,

    #[error("Invalid long size: {0}")]
    InvalidLongSize(MemSize),

    #[error("Invalid option in header")]
    InvalidOption,

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

    #[error("Could not load event formats as there is too many of them: {0}")]
    TooManyEventDescs(u32),

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

    #[error("Data format not supported")]
    UnsupportedDataFmt,

    #[error("Error while loading data: {0}")]
    IoError(io::ErrorKind),

    #[error("Could not parse header: {0}")]
    ParseError(Box<VerboseParseError>),
}

impl<I: AsRef<[u8]>, I2: AsRef<[u8]>> FromParseError<I, nom::error::VerboseError<I2>>
    for HeaderError
{
    fn from_parse_error(input: I, err: &nom::error::VerboseError<I2>) -> Self {
        HeaderError::ParseError(Box::new(VerboseParseError::new(input, err)))
    }
}

impl<I: AsRef<[u8]>> FromParseError<I, ()> for HeaderError {
    fn from_parse_error(input: I, _err: &()) -> Self {
        HeaderError::ParseError(Box::new(VerboseParseError::from_input(input)))
    }
}

impl From<io::Error> for HeaderError {
    fn from(err: io::Error) -> Self {
        HeaderError::IoError(err.kind())
    }
}

#[derive(Debug, Clone)]
pub enum Options {
    // v6 only, defines a non-top-level instance
    Instance {
        name: String,
        offset: FileOffset,
    },

    // v7 only, fully defines the location of a single ring buffer
    Buffer {
        cpu: CPU,
        name: String,
        offset: FileOffset,
        size: FileSize,
        decomp: Option<DynDecompressor>,
        page_size: MemSize,
    },

    // TODO: parse
    BufferText(),

    HeaderInfoLoc(FileOffset),
    FtraceEventsLoc(FileOffset),
    EventFormatsLoc(FileOffset),
    KallsymsLoc(FileOffset),
    PrintkLoc(FileOffset),
    CmdLinesLoc(FileOffset),
    TimeOffset(TimeOffset),

    TSC2NSec {
        multiplier: u32,
        shift: u32,
        offset: u64,
    },

    // TODO: parse
    Date(String),

    // TODO: parse
    CpuStat {
        cpu: CPU,
        stat: String,
    },

    // TODO: parse
    TraceClock(String),
    // TODO: parse
    Uname(String),
    // TODO: parse
    Hook(String),

    // TODO: parse
    CpuCount(CPU),

    // TODO: parse
    Version(String),

    // TODO: parse
    ProcMaps(String),

    // TODO: parse
    TraceId(String),

    // TODO: parse
    TimeShift(),

    // TODO: parse
    Guest(),

    Unknown {
        typ: u16,
        data: Vec<u8>,
    },
}

fn shared_decode_option(_abi: &Abi, option_type: u16, data: &[u8]) -> Result<Options, HeaderError> {
    Ok(Options::Unknown {
        typ: option_type,
        data: data.to_vec(),
    })
}

fn v6_parse_options<I>(abi: &Abi, input: &mut I) -> Result<Vec<Options>, HeaderError>
where
    I: BorrowingRead,
{
    let endianness = abi.endianness;

    fn parse_date(date: &str) -> Result<TimeOffset, HeaderError> {
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
                    .map_err(|_| HeaderError::InvalidOption)?;
                break;
            }
        }

        let offset = offset * sign;
        Ok(offset)
    }

    let mut options = Vec::new();
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
                let name = CStr::from_bytes_with_nul(option_data)
                    .map_err(|_| HeaderError::InvalidOption)?
                    .to_str()
                    .map_err(|_| HeaderError::InvalidOption)?;
                Options::Instance {
                    name: name.into(),
                    offset,
                }
            }
            // OFFSET: id 7, size vary
            7 => {
                let date = CStr::from_bytes_with_nul(option_data)
                    .map_err(|_| HeaderError::InvalidOption)?
                    .to_str()
                    .map_err(|_| HeaderError::InvalidOption)?;

                Options::TimeOffset(parse_date(date)?)
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
            _ => shared_decode_option(abi, option_type, option_data)?,
        });
    }
    Ok(options)
}

fn v7_parse_options<I>(
    abi: &Abi,
    decomp: &mut Option<DynDecompressor>,
    mut input: I,
) -> Result<(I, Vec<Options>), HeaderError>
where
    I: BorrowingRead,
{
    let mut options = Vec::new();
    let mut section_decomp = decomp.clone();

    loop {
        let section = {
            let (id, options) = v7_section(abi, &mut section_decomp, &mut input)?;
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

        macro_rules! read_null_terminated {
            ($update:ident) => {{
                let idx = $update
                    .into_iter()
                    .position(|x| *x == 0)
                    .ok_or(HeaderError::InvalidOption)?;
                let name = CStr::from_bytes_with_nul(&$update[..=idx])
                    .map_err(|_| HeaderError::InvalidOption)?
                    .to_str()
                    .map_err(|_| HeaderError::InvalidOption)?;
                $update = &$update[idx + 1..];
                name
            }};
        }

        loop {
            let option_type = read!(parse_u16, section_data);
            let option_size = read!(parse_u32, section_data);

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
                        _ => options.push(shared_decode_option(abi, option_type, option_data)?),
                    };
                }
            }
        }
    }
}

pub fn header<I>(input: &mut I) -> Result<Header, HeaderError>
where
    I: BorrowingRead,
{
    input.read_tag(b"\x17\x08\x44tracing", HeaderError::BadMagic)??;

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
    let nr_cpus = 0;
    let mut abi = abi;

    macro_rules! get_section {
        ($expected_id:expr, $offset:expr) => {{
            let expected = $expected_id;
            input = input.abs_seek($offset, None)?;
            let (found, section) = v7_section(&abi, &mut decomp, &mut input)?;

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
) -> Result<BTreeMap<PID, TaskName>, HeaderError>
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
    input.read_tag(b"header_page\0", HeaderError::ExpectedHeaderPage)??;
    let page_header_size_u64: u64 = read_int!()?;
    let page_header_size: usize = page_header_size_u64
        .try_into()
        .map_err(|_| HeaderError::PageHeaderSizeTooLarge(page_header_size_u64))?;

    let header_fields = input.parse(
        page_header_size,
        // Disable type check due to:
        // https://bugzilla.kernel.org/show_bug.cgi?id=216999
        |input| parse_struct_fmt(&abi, true, input),
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
    input.read_tag(b"header_event\0", HeaderError::ExpectedHeaderEvent)??;
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
                });
            }
        }
    }
}

pub(crate) fn buffer_locations<I>(
    kind: &[u8],
    nr_cpus: CPU,
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
        b"latency  " => Err(HeaderError::UnsupportedDataFmt),
        _ => Err(HeaderError::UnsupportedDataFmt),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        parser::tests::test_parser,
        print::{
            PrintAtom, PrintFlags, PrintFmtStr, PrintPrecision, PrintSpecifier, PrintWidth,
            VBinSpecifier,
        },
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

        test(
            b"name: wakeup\nID: 3\nformat:\n\tfield:unsigned short common_type;\toffset:0;\tsize:2;\tsigned:0;\n\tfield:unsigned char common_flags;\toffset:2;\tsize:1;\tsigned:0;\n\tfield:unsigned char common_preempt_count;\toffset:3;\tsize:1;\tsigned:0;\n\tfield:int common_pid;\toffset:4;\tsize:4;\tsigned:1;\n\n\tfield:unsigned int prev_pid;\toffset:8;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_pid;\toffset:12;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_cpu;\toffset:16;\tsize:4;\tsigned:0;\n\tfield:unsigned char prev_prio;\toffset:20;\tsize:1;\tsigned:0;\n\tfield:unsigned char prev_state;\toffset:21;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_prio;\toffset:22;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_state;\toffset:23;\tsize:1;\tsigned:0;\n\nprint fmt: \"%u:%u:%u  ==+ %u:%u:%u \\\" \\t [%03u]\", 55\n",
            EventDescContent {
                name: "wakeup".into(),
                id: 3,
                fmt: EventFmt {
                    print_args: Ok(vec![]),
                    print_fmt: Ok(PrintFmtStr {
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
                    }),
                    struct_fmt: Ok(StructFmt {
                        fields: vec![
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_type".into(),
                                    typ: Type::U16,
                                },
                                offset: 0,
                                size: 2,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_flags".into(),
                                    typ: Type::U8,
                                },
                                offset: 2,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_preempt_count".into(),
                                    typ: Type::U8,
                                },
                                offset: 3,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_pid".into(),
                                    typ: Type::I32,
                                },
                                offset: 4,
                                size: 4,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "prev_pid".into(),
                                    typ: Type::U32,
                                },
                                offset: 8,
                                size: 4,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_pid".into(),
                                    typ: Type::U32,
                                },
                                offset: 12,
                                size: 4,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_cpu".into(),
                                    typ: Type::U32,
                                },
                                offset: 16,
                                size: 4,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "prev_prio".into(),
                                    typ: Type::U8,
                                },
                                offset: 20,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "prev_state".into(),
                                    typ: Type::U8,
                                },
                                offset: 21,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_prio".into(),
                                    typ: Type::U8,
                                },
                                offset: 22,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "next_state".into(),
                                    typ: Type::U8,
                                },
                                offset: 23,
                                size: 1,
                                decoder: Arc::new(()),
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
                    print_args: Ok(vec![]),
                    print_fmt: Ok(PrintFmtStr {
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
                    }),
                    struct_fmt: Ok(StructFmt {
                        fields: vec![
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_type".into(),
                                    typ: Type::U16,
                                },
                                offset: 0,
                                size: 2,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_flags".into(),
                                    typ: Type::U8,
                                },
                                offset: 2,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_preempt_count".into(),
                                    typ: Type::U8,
                                },
                                offset: 3,
                                size: 1,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "common_pid".into(),
                                    typ: Type::I32,
                                },
                                offset: 4,
                                size: 4,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "tgid".into(),
                                    typ: Type::U32,
                                },
                                offset: 8,
                                size: 4,
                                decoder: Arc::new(()),
                            },
                            FieldFmt {
                                declaration: Declaration {
                                    identifier: "caller".into(),
                                    typ: Type::Array(Box::new(Type::U64), ArrayKind::Fixed(Ok(8))),
                                },
                                offset: 16,
                                size: 64,
                                decoder: Arc::new(()),
                            },
                        ]
                    })
                }
            }
        );
    }
}
