use core::{
    cmp::Ordering,
    fmt,
    fmt::{Debug, Error as FmtError, Write as _},
    str::from_utf8,
};
use std::{error::Error, io};

use bitflags::bitflags;
use itertools::Itertools as _;
use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{char, u64 as txt_u64},
    combinator::{cut, opt, success},
    error::{context, ContextError, FromExternalError, ParseError},
    multi::{many0, many1},
    sequence::{preceded, separated_pair, tuple},
    Parser,
};
use once_cell::sync::OnceCell;
use smartstring::alias::String;

use crate::{
    buffer::{BufferError, PrintArg, VBinDecoder},
    cinterp::{CompileError, EvalEnv, EvalError, InterpError, SockAddr, SockAddrKind, Value},
    cparser::CParseError,
    header::{Abi, Endianness, Header, LongSize, MemSize, Signedness},
    parser::{map_res_cut, FromParseError, NomParserExt as _, VerboseParseError},
    str::{InnerStr, Str},
};

#[derive(Debug, Clone)]
pub struct PrintFmtStr {
    pub atoms: Vec<PrintAtom>,
    pub(crate) vbin_decoders: OnceCell<Vec<VBinDecoder>>,
}

impl PartialEq<Self> for PrintFmtStr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.atoms == other.atoms
    }
}

impl Eq for PrintFmtStr {}

impl PartialOrd<Self> for PrintFmtStr {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrintFmtStr {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.atoms.cmp(&other.atoms)
    }
}

pub struct StringWriter<W> {
    inner: W,
}

impl<W> StringWriter<W> {
    #[inline]
    pub fn new(inner: W) -> Self {
        StringWriter { inner }
    }

    #[inline]
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: io::Write> io::Write for StringWriter<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W: io::Write> fmt::Write for StringWriter<W> {
    #[inline]
    fn write_str(&mut self, s: &str) -> Result<(), fmt::Error> {
        io::Write::write(self, s.as_bytes())
            .map(|_| ())
            .map_err(|_| fmt::Error)
    }
}

struct TrackedWriter<W> {
    inner: W,
    count: usize,
}

impl<W> TrackedWriter<W> {
    #[inline]
    fn new(inner: W) -> Self {
        TrackedWriter { count: 0, inner }
    }
}

impl<W: fmt::Write> fmt::Write for TrackedWriter<W> {
    #[inline]
    fn write_str(&mut self, s: &str) -> Result<(), fmt::Error> {
        self.inner.write_str(s)?;
        self.count += s.len();
        Ok(())
    }
}

struct SinkWriter;

impl fmt::Write for SinkWriter {
    #[inline]
    fn write_str(&mut self, _: &str) -> Result<(), fmt::Error> {
        Ok(())
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrintError {
    #[error("Could not parse printf format string: {0}")]
    PrintFmtError(Box<PrintFmtError>),

    #[error("Error while decoding buffer: {0}")]
    BufferError(Box<BufferError>),

    #[error("Error while interpreting expression: {0}")]
    InterpError(Box<InterpError>),

    #[error("Dynamic width in printf format is missing")]
    PrintFmtMissingWidth,

    #[error("Dynamic precision in printf format is missing")]
    PrintFmtMissingPrecision,

    #[error("Missing value to interpolate in the format string")]
    MissingValue,

    #[error("Address is not an integer")]
    NonNumericAddr,

    #[error("Value cannot be formatted as a buffer")]
    NotABuffer,

    #[error("Value cannot be formatted as a string")]
    NotAString,

    #[error("Value cannot be formatted as an integer")]
    NotAnInteger,

    #[error("Value cannot be formatted as a sockaddr")]
    NotASockAddr,

    #[error("Specifier not implemented: {0:?}")]
    SpecifierNotHandled(PrintSpecifier),

    #[error("Wrong size for array item. Expected {0} bytes, got {1} bytes")]
    WrongItemSize(MemSize, MemSize),

    #[error("Error while loading data: {0}")]
    IoError(io::ErrorKind),

    #[error("Error while formatting string: {0}")]
    FmtError(Box<FmtError>),
}
macro_rules! convert_err_impl {
    ($src:ident, $variant:ident, $dst:ident) => {
        impl From<$src> for $dst {
            fn from(err: $src) -> Self {
                $dst::$variant(Box::new(err.into()))
            }
        }
    };
}

impl From<io::Error> for PrintError {
    fn from(err: io::Error) -> Self {
        PrintError::IoError(err.kind())
    }
}

convert_err_impl!(PrintFmtError, PrintFmtError, PrintError);
convert_err_impl!(BufferError, BufferError, PrintError);
convert_err_impl!(InterpError, InterpError, PrintError);
convert_err_impl!(EvalError, InterpError, PrintError);
convert_err_impl!(CompileError, InterpError, PrintError);
convert_err_impl!(FmtError, FmtError, PrintError);

impl PrintFmtStr {
    pub fn interpolate_values<'v, 'ee, E, W, I, EE>(
        &self,
        header: &'v Header,
        env: &'ee EE,
        out: &mut W,
        values: I,
    ) -> Result<(), PrintError>
    where
        EE: EvalEnv<'ee> + ?Sized,
        E: Error,
        W: fmt::Write + ?Sized,
        I: IntoIterator<Item = Result<Value<'v>, E>>,
        PrintError: From<E>,
    {
        let mut values = values.into_iter();

        macro_rules! get_int {
            ($err:expr) => {
                match values.next() {
                    Some(Err(err)) => Err(err.into()),
                    Some(Ok(Value::U64Scalar(x))) => Ok(x.try_into().unwrap()),
                    Some(Ok(Value::I64Scalar(x))) if x >= 0 => {
                        Ok(x.unsigned_abs().try_into().unwrap())
                    }
                    _ => Err($err),
                }
            };
        }

        let print_values = self.atoms.iter().filter_map(|atom| match atom {
            PrintAtom::Fixed(_) => None,
            PrintAtom::Variable {
                width, precision, ..
            } => Some(|| -> Result<PrintArg<'_>, PrintError> {
                let width = match width {
                    PrintWidth::Dynamic => Some(get_int!(PrintError::PrintFmtMissingWidth)?),
                    _ => None,
                };
                let precision = match precision {
                    PrintPrecision::Dynamic => {
                        Some(get_int!(PrintError::PrintFmtMissingPrecision)?)
                    }
                    _ => None,
                };
                let value = values.next().ok_or(PrintError::MissingValue)??;
                Ok(PrintArg {
                    value,
                    width,
                    precision,
                })
            }()),
        });
        self.interpolate_into::<PrintError, _, _, _>(header, env, out, print_values)
    }

    pub fn interpolate_vbin<'v, 'ee, W, EE>(
        &self,
        header: &'v Header,
        env: &'ee EE,
        out: &mut W,
        buf: &'v [u32],
    ) -> Result<(), PrintError>
    where
        W: fmt::Write + ?Sized,
        EE: EvalEnv<'ee> + ?Sized,
    {
        self.interpolate_into(
            header,
            env,
            out,
            self.vbin_fields(header, env.scratch(), buf),
        )
    }

    fn interpolate_into<'v, 'ee, E, W, I, EE>(
        &self,
        header: &'v Header,
        env: &'ee EE,
        out: &mut W,
        values: I,
    ) -> Result<(), PrintError>
    where
        E: Error,
        EE: EvalEnv<'ee> + ?Sized,
        W: fmt::Write + ?Sized,
        I: IntoIterator<Item = Result<PrintArg<'v>, E>>,
        PrintError: From<E>,
    {
        let out = &mut TrackedWriter::new(out);
        let mut values = values.into_iter();

        let mut print_variable = |out: &mut TrackedWriter<&mut W>,
                                  print_spec: &PrintSpecifier,
                                  flags: &PrintFlags,
                                  width: &PrintWidth,
                                  precision: &PrintPrecision|
         -> Result<_, PrintError> {
            let item = values.next().ok_or(PrintError::MissingValue)??;
            let vanilla = print_spec == &PrintSpecifier::Dec
                && width == &PrintWidth::Unmodified
                && precision == &PrintPrecision::Unmodified
                && flags.is_empty();

            let width = match width {
                PrintWidth::Fixed(x) => Ok(*x),
                PrintWidth::Unmodified => Ok(0),
                PrintWidth::Dynamic => match item.width {
                    None => Err(PrintError::PrintFmtMissingWidth),
                    Some(x) => Ok(x),
                },
            }?;

            let precision = match precision {
                PrintPrecision::Fixed(x) => Ok(Some(*x)),
                PrintPrecision::Unmodified => Ok(None),
                PrintPrecision::Dynamic => match item.precision {
                    None => Err(PrintError::PrintFmtMissingPrecision),
                    Some(x) => Ok(Some(x)),
                },
            }?;

            #[derive(Debug)]
            enum Justification {
                Left,
                Right,
            }

            #[allow(clippy::enum_variant_names)]
            #[derive(Debug)]
            enum Sign {
                OnlyNeg,
                PosAndNeg,
                BlankAndNeg,
            }

            #[derive(Debug)]
            enum BasePrefix {
                None,
                Oct,
                LowerHex,
                UpperHex,
            }

            #[derive(Debug)]
            enum Padding {
                Zero,
                Space,
            }

            let justification = if flags.contains(PrintFlags::LeftJustify) {
                Justification::Left
            } else {
                Justification::Right
            };

            let sign = if flags.contains(PrintFlags::PositiveSign) {
                Sign::PosAndNeg
            } else if flags.contains(PrintFlags::SignPlaceholder) {
                Sign::BlankAndNeg
            } else {
                Sign::OnlyNeg
            };

            let base_prefix = if flags.contains(PrintFlags::BasePrefix) {
                match print_spec {
                    PrintSpecifier::UpperHex => BasePrefix::UpperHex,
                    PrintSpecifier::Hex => BasePrefix::LowerHex,
                    PrintSpecifier::Oct => BasePrefix::Oct,
                    _ => BasePrefix::None,
                }
            } else {
                BasePrefix::None
            };

            let padding = if flags.contains(PrintFlags::ZeroPad) {
                Padding::Zero
            } else {
                Padding::Space
            };

            // eprintln!("VALUE {val:?} just={justification:?} sign={sign:?} base={base_prefix:?} pad={padding:?} width={width:?} precision={precision:?}");

            macro_rules! print_left {
                ($x:expr, $print_spec:expr, $width:expr, $precision:expr, $discount_prefix:expr, $out:expr) => {{
                    let x = $x;
                    let print_spec = &$print_spec;
                    let width = $width;
                    let precision = $precision;
                    let discount_prefix: bool = $discount_prefix;
                    let out: &mut _ = $out;

                    let start = out.count;

                    match print_spec {
                        PrintSpecifier::Dec =>
                        {
                            #[allow(unused_comparisons)]
                            if x < 0 {
                                write!(out, "-")
                            } else {
                                match sign {
                                    Sign::OnlyNeg => Ok(()),
                                    Sign::PosAndNeg => write!(out, "+"),
                                    Sign::BlankAndNeg => write!(out, " "),
                                }
                            }
                        }
                        _ => match base_prefix {
                            BasePrefix::LowerHex => write!(out, "0x"),
                            BasePrefix::UpperHex => write!(out, "0X"),
                            BasePrefix::Oct => write!(out, "0"),
                            BasePrefix::None => Ok(()),
                        },
                    }?;

                    let precision = precision.unwrap_or(0);
                    let so_far = out.count - start;
                    let precision = if discount_prefix && so_far <= precision {
                        precision - so_far
                    } else {
                        precision
                    };

                    #[allow(unused_comparisons)]
                    let abs = if x < 0 { 0 - x } else { x };
                    match print_spec {
                        PrintSpecifier::Hex => {
                            write!(out, "{:0>precision$x}", abs)
                        }
                        PrintSpecifier::UpperHex => {
                            write!(out, "{:0>precision$X}", abs)
                        }
                        PrintSpecifier::Oct => {
                            write!(out, "{:0>precision$o}", abs)
                        }
                        _ => write!(out, "{:0>precision$}", abs),
                    }?;

                    let so_far = out.count - start;
                    if width > so_far {
                        write!(out, "{: >pad$}", "", pad = width - so_far)?;
                    };
                    Ok(())
                }};
            }

            macro_rules! print_integral {
                ($x:expr, $print_spec:expr) => {{
                    let x = $x;
                    let print_spec = $print_spec;
                    match justification {
                        // If width == 0, justification is irrelevant so
                        // we can use the simpler path.
                        Justification::Right if width > 0 => {
                            let (discount_prefix, precision) = match precision {
                                Some(precision) => (false, Some(precision)),
                                None => (
                                    // If we use the width as precision, the
                                    // precision will have to be reduced by
                                    // the amount of prefix otherwise the
                                    // overall width will not be respected.
                                    //
                                    // We still use the width as precision
                                    // as this allows interposing extra
                                    // zeros between the prefix and the
                                    // value.
                                    true,
                                    match padding {
                                        Padding::Zero => Some(width),
                                        Padding::Space => Some(0),
                                    },
                                ),
                            };

                            let sink = &mut TrackedWriter::new(SinkWriter);
                            print_left!(x, print_spec, 0, precision, discount_prefix, sink)?;
                            if width > sink.count {
                                write!(out, "{: <pad$}", "", pad = width - sink.count)?;
                            };
                            print_left!(x, print_spec, 0, precision, discount_prefix, out)
                        }
                        _ => print_left!(x, print_spec, width, precision, false, out),
                    }
                }};
            }

            let mut print_str = |s: &str| {
                let s = match precision {
                    Some(x) => s.get(..x).unwrap_or(s),
                    None => s,
                };
                // Remove trainling "\n". They appear in "print" event when userspace used
                // "echo" to write to the trace_marker file.
                let s = s.trim_end_matches('\n');
                match justification {
                    Justification::Left => write!(out, "{s: <width$}"),
                    Justification::Right => write!(out, "{s: >width$}"),
                }
            };

            let print_symbol =
                |out: &mut TrackedWriter<&mut W>, addr, show_offset| -> Result<_, PrintError> {
                    let addr = match addr {
                        Value::U64Scalar(x) => Ok(x),
                        Value::I64Scalar(x) => Ok(x as u64),
                        addr => match addr.to_str() {
                            Some(s) => return Ok(write!(out, "{s}")?),
                            None => Err(PrintError::NonNumericAddr),
                        },
                    }?;
                    Ok(match header.sym_at(addr) {
                        Some((offset, size, name)) => {
                            if show_offset {
                                match size {
                                    Some(size) => write!(out, "{name}+{offset:#x}/{size:#x}"),
                                    None => write!(out, "{name}+{offset:#x}"),
                                }
                            } else {
                                write!(out, "{name}")
                            }
                        }
                        None => write!(out, "{addr:#x}"),
                    }?)
                };

            // T this is a bit crude, as we display addresses always
            // the same way, without taking into account user hints.
            let print_sockaddr =
                |out: &mut TrackedWriter<&mut W>, s: &SockAddr| match s.to_socketaddr() {
                    Ok(addr) => Ok(write!(out, "{addr}")?),
                    _ => match s.to_ipaddr() {
                        Ok(addr) => Ok(write!(out, "{addr}")?),
                        _ => Err(PrintError::NotASockAddr),
                    },
                };

            macro_rules! print_hex_buffer {
                ($out:expr, $arr:expr, $sep:expr) => {{
                    let arr = $arr;
                    let sep = $sep;

                    let n = if width == 0 { usize::MAX } else { width };
                    for (i, x) in arr.into_iter().take(n).enumerate() {
                        if i != 0 {
                            match sep {
                                HexBufferSeparator::C => write!($out, "{}", ':'),
                                HexBufferSeparator::D => write!($out, "{}", '-'),
                                HexBufferSeparator::N => Ok(()),
                            }?;
                        }
                        write!($out, "{x:02x}")?;
                    }
                    Ok(())
                }};
            }

            macro_rules! handle_sockaddr {
                ($out:expr, $kind:expr, $val:expr, $endianness:expr) => {{
                    let val: &Value = $val;
                    let endianness: Endianness = $endianness;

                    let temp;
                    let sockaddr: &SockAddr = match val {
                        // This is not pretty but should work well
                        // enough: this function can be called on
                        // multiple paths:
                        //
                        // * When formatting an event after having
                        //   decoded its fields. In that case, we get a
                        //   U8Array that contains the raw sockaddr, and
                        //   possibly in the future we will get a
                        //   Value::SockAddr
                        // * When formatting a bprint buffer: we will
                        //   get a Value::Str, since the sockaddr has
                        //   already been rendered to a string by the
                        //   kernel.
                        //
                        // Normally, U8Array and Str can be used
                        // interchangeably, but in this case we need to
                        // disambiguate between the 2 cases.
                        Value::Str(s) => return Ok(print_str(&s)?),
                        Value::SockAddr(addr) => addr,
                        Value::U8Array(arr) => {
                            temp =
                                SockAddr::from_bytes(&arr, endianness, $kind).map_err(|err| {
                                    let err: PrintError = err.into();
                                    err
                                })?;
                            &temp
                        }
                        _ => Err(PrintError::NotASockAddr)?,
                    };
                    Ok(print_sockaddr($out, sockaddr)?)
                }};
            }

            let val = item.value;
            Ok(match print_spec {
                PrintSpecifier::Hex
                | PrintSpecifier::UpperHex
                | PrintSpecifier::Dec
                | PrintSpecifier::Oct => match val {
                    // Fast path if vanilla. This should accomodate the vast
                    // majority of fields.
                    Value::U64Scalar(x) if vanilla => write!(out, "{x}"),
                    Value::I64Scalar(x) if vanilla => write!(out, "{x}"),

                    Value::U64Scalar(x) => print_integral!(x, print_spec),
                    Value::I64Scalar(x) => print_integral!(x, print_spec),

                    _ => Err(PrintError::NotAnInteger)?,
                },
                PrintSpecifier::Str => match val {
                    Value::Str(Str {
                        inner: InnerStr::Procedural(memo),
                    }) => {
                        memo.seed.write(out);
                        Ok(())
                    }
                    _ => match val.deref_ptr(env)?.to_str() {
                        Some(s) => print_str(s),
                        None => Err(PrintError::NotAString)?,
                    },
                },
                PrintSpecifier::Symbol => Ok(print_symbol(out, val, false)?),
                PrintSpecifier::SymbolWithOffset => Ok(print_symbol(out, val, true)?),
                PrintSpecifier::HexBuffer(sep) => match val.deref_ptr(env)?.to_bytes() {
                    Some(arr) => print_hex_buffer!(out, arr, *sep),
                    _ => Err(PrintError::NotABuffer)?,
                },

                PrintSpecifier::IpGeneric(endianness) => {
                    handle_sockaddr!(out, SockAddrKind::Full, &val, *endianness)
                }
                PrintSpecifier::Ipv4(endianness) => {
                    handle_sockaddr!(out, SockAddrKind::Ipv4AddrOnly, &val, *endianness)
                }
                PrintSpecifier::Ipv6 => {
                    handle_sockaddr!(out, SockAddrKind::Ipv6AddrOnly, &val, Endianness::Big)
                }

                PrintSpecifier::Resource
                | PrintSpecifier::DmaAddr
                | PrintSpecifier::PhysAddr
                | PrintSpecifier::EscapedBuffer
                | PrintSpecifier::MacAddress
                | PrintSpecifier::Uuid
                | PrintSpecifier::Kobject
                | PrintSpecifier::PageFlags
                | PrintSpecifier::VmaFlags
                | PrintSpecifier::GfpFlags
                | PrintSpecifier::Clock
                | PrintSpecifier::NetdevFeatures
                | PrintSpecifier::Bitmap
                | PrintSpecifier::Dentry
                | PrintSpecifier::BlockDevice
                | PrintSpecifier::VaFormat => match val.to_str() {
                    // Fallback on displaying as a string. We might get
                    // a string value from vbin-encoded data, where the
                    // kernel-rendered string was dumped in the buffer.
                    // This allows using trace_printk() with arguments
                    // that are passed by reference and then formatted
                    // by the kernel, ready to be displayed by us.
                    Some(s) => print_str(s),
                    None => match val {
                        Value::U64Scalar(x) => print_integral!(x, PrintSpecifier::Hex),
                        Value::I64Scalar(x) => print_integral!(x, PrintSpecifier::Hex),
                        Value::Raw(_typ, arr) => {
                            print_hex_buffer!(out, arr.iter(), HexBufferSeparator::C)
                        }
                        _ => Err(PrintError::SpecifierNotHandled(print_spec.clone()))?,
                    },
                },
            }?)
        };

        let mut res = Ok(());
        for atom in &self.atoms {
            match atom {
                PrintAtom::Fixed(s) => write!(out, "{s}")?,
                PrintAtom::Variable {
                    print_spec,
                    flags,
                    width,
                    precision,
                    ..
                } => {
                    if let Err(err) = print_variable(out, print_spec, flags, width, precision) {
                        // Write the error message inplace in the output and then return the last
                        // one as well. If we are printing to stdout, this allows getting values
                        // for the other fields and then let the caller decide whether the
                        // processing should be interrupted.
                        write!(out, "<{err}>")?;
                        res = Err(err);
                    }
                }
            }
        }
        res
    }
}

/// Format specifier as understood by vbin_printf()
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum VBinSpecifier {
    U8,
    I8,

    U16,
    I16,

    U32,
    I32,

    U64,
    I64,

    Str,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrintWidth {
    Unmodified,
    Fixed(usize),
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrintPrecision {
    Unmodified,
    Fixed(usize),
    Dynamic,
}

bitflags! {
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PrintFlags: u8 {
        // #
        const BasePrefix = 1;

        // -
        const LeftJustify = 2;

        // +
        const PositiveSign = 4;

        // space
        const SignPlaceholder = 8;

        // 0
        const ZeroPad = 16;
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum HexBufferSeparator {
    C,
    D,
    N,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrintSpecifier {
    Hex,
    UpperHex,
    Dec,
    Oct,
    Str,

    Symbol,
    SymbolWithOffset,

    // %*ph
    HexBuffer(HexBufferSeparator),

    Resource,
    DmaAddr,
    PhysAddr,
    EscapedBuffer,
    MacAddress,
    Ipv4(Endianness),
    Ipv6,
    IpGeneric(Endianness),
    Uuid,
    Kobject,
    PageFlags,
    VmaFlags,
    GfpFlags,
    Clock,
    NetdevFeatures,
    Bitmap,
    Dentry,
    BlockDevice,
    VaFormat,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrintAtom {
    Fixed(String),
    Variable {
        vbin_spec: VBinSpecifier,
        print_spec: PrintSpecifier,
        flags: PrintFlags,
        width: PrintWidth,
        precision: PrintPrecision,
    },
}

impl PrintAtom {
    #[inline]
    pub(crate) fn new_variable(
        _abi: &Abi,
        vbin_spec: VBinSpecifier,
        print_spec: PrintSpecifier,
        flags: PrintFlags,
        width: PrintWidth,
        precision: PrintPrecision,
    ) -> Self {
        PrintAtom::Variable {
            vbin_spec,
            print_spec,
            flags: PrintFlags::from_iter(flags),
            width,
            precision,
        }
    }

    pub(crate) fn zip_atoms<
        'atom,
        T,
        I1: Iterator<Item = T>,
        I2: Iterator<Item = &'atom PrintAtom>,
    >(
        mut args: I1,
        mut atoms: I2,
    ) -> impl IntoIterator<Item = (T, Option<&'atom PrintAtom>)> {
        let mut count = 0;
        core::iter::from_fn(move || {
            let arg = args.next()?;
            if count > 0 {
                count -= 1;
                Some((arg, None))
            } else {
                loop {
                    match atoms.next() {
                        Some(PrintAtom::Fixed(_)) => continue,
                        curr @ Some(PrintAtom::Variable {
                            ref width,
                            ref precision,
                            ..
                        }) => {
                            count = 0;
                            count += match width {
                                PrintWidth::Dynamic => 1,
                                _ => 0,
                            };
                            count += match precision {
                                PrintPrecision::Dynamic => 1,
                                _ => 0,
                            };
                            break Some((arg, curr));
                        }
                        None => break Some((arg, None)),
                    }
                }
            }
        })
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrintFmtError {
    #[error("Expected string literal")]
    NotAStringLiteral,

    #[error("Could not parse the printk format string: {0}")]
    CParseError(Box<CParseError>),

    #[error("Could not decode string as utf-8: {0}")]
    DecodeUtf8(core::str::Utf8Error),

    #[error("Illegal specifier: {0}")]
    IllegalSpecifier(String),
}

impl<I: AsRef<[u8]>, I2: AsRef<[u8]>> FromParseError<I, nom::error::VerboseError<I2>>
    for PrintFmtError
{
    fn from_parse_error(input: I, err: &nom::error::VerboseError<I2>) -> Self {
        PrintFmtError::CParseError(Box::new(CParseError::ParseError(VerboseParseError::new(
            input, err,
        ))))
    }
}

impl<I: AsRef<[u8]>> FromParseError<I, ()> for PrintFmtError {
    fn from_parse_error(input: I, _err: &()) -> Self {
        PrintFmtError::CParseError(Box::new(CParseError::ParseError(
            VerboseParseError::from_input(input),
        )))
    }
}

impl From<core::str::Utf8Error> for PrintFmtError {
    fn from(err: core::str::Utf8Error) -> Self {
        PrintFmtError::DecodeUtf8(err)
    }
}

// This function has been split out and unnecessary levels of alt((...)) have
// been added to avoid hitting a compile time blowup (from <2s to type check to
// 25s, also leading to an absolute disaster in release profile)
fn specifier<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<&'a [u8], Result<(VBinSpecifier, PrintSpecifier), PrintFmtError>, E> + 'abi
where
    E: 'abi
        + ParseError<&'a [u8]>
        + ContextError<&'a [u8]>
        + FromExternalError<&'a [u8], PrintFmtError>
        + Debug,
    'a: 'abi,
{
    let (vbin_long, vbin_ulong) = match abi.long_size {
        LongSize::Bits32 => (VBinSpecifier::I32, VBinSpecifier::U32),
        LongSize::Bits64 => (VBinSpecifier::I64, VBinSpecifier::U64),
    };
    let vbin_char = match abi.char_signedness {
        Signedness::Unsigned => VBinSpecifier::U8,
        Signedness::Signed => VBinSpecifier::I8,
    };

    let ip_endianness = || {
        alt((char('h'), char('n'), char('b'), char('l'))).map(|letter| match letter {
            'h' => abi.endianness,
            'b' | 'n' => Endianness::Big,
            'l' => Endianness::Little,
            _ => panic!("Unknown endianness"),
        })
    };

    move |input| {
        alt((
            char('s').map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Str))),
            context(
                "integer",
                alt((
                    alt((
                        alt((
                            tag("hhu").map(|_| Ok((VBinSpecifier::U16, PrintSpecifier::Dec))),
                            tag("hhx").map(|_| Ok((VBinSpecifier::U16, PrintSpecifier::Hex))),
                            tag("hhX").map(|_| Ok((VBinSpecifier::U16, PrintSpecifier::UpperHex))),
                            tag("hho").map(|_| Ok((VBinSpecifier::U16, PrintSpecifier::Oct))),
                        )),
                        alt((
                            tag("hhd").map(|_| Ok((VBinSpecifier::I16, PrintSpecifier::Dec))),
                            tag("hhi").map(|_| Ok((VBinSpecifier::I16, PrintSpecifier::Dec))),
                            tag("hh").map(|_| Ok((VBinSpecifier::I16, PrintSpecifier::Dec))),
                        )),
                    )),
                    alt((
                        alt((
                            tag("hu").map(|_| Ok((VBinSpecifier::U8, PrintSpecifier::Dec))),
                            tag("hx").map(|_| Ok((VBinSpecifier::U8, PrintSpecifier::Hex))),
                            tag("hX").map(|_| Ok((VBinSpecifier::U8, PrintSpecifier::UpperHex))),
                            tag("ho").map(|_| Ok((VBinSpecifier::U8, PrintSpecifier::Oct))),
                        )),
                        alt((
                            tag("hd").map(|_| Ok((VBinSpecifier::I8, PrintSpecifier::Dec))),
                            tag("hi").map(|_| Ok((VBinSpecifier::I8, PrintSpecifier::Dec))),
                            char('h').map(|_| Ok((VBinSpecifier::I8, PrintSpecifier::Dec))),
                            char('c').map(|_| Ok((vbin_char.clone(), PrintSpecifier::Str))),
                        )),
                    )),
                    alt((
                        alt((char('d'), char('i')))
                            .map(|_| Ok((VBinSpecifier::I32, PrintSpecifier::Dec))),
                        char('u').map(|_| Ok((VBinSpecifier::U32, PrintSpecifier::Dec))),
                        char('o').map(|_| Ok((VBinSpecifier::U32, PrintSpecifier::Oct))),
                        char('x').map(|_| Ok((VBinSpecifier::U32, PrintSpecifier::Hex))),
                        char('X').map(|_| Ok((VBinSpecifier::U32, PrintSpecifier::UpperHex))),
                    )),
                    alt((
                        tag("ld"),
                        tag("li"),
                        tag("Ld"),
                        tag("Li"),
                        tag("z"),
                        tag("zd"),
                    ))
                    .map(|_| Ok((vbin_long.clone(), PrintSpecifier::Dec))),
                    alt((
                        alt((tag("lu"), tag("Lu"), tag("zu")))
                            .map(|_| Ok((vbin_ulong.clone(), PrintSpecifier::Dec))),
                        alt((tag("lx"), tag("Lx"), tag("zx")))
                            .map(|_| Ok((vbin_ulong.clone(), PrintSpecifier::Hex))),
                        alt((tag("lX"), tag("LX"), tag("zX")))
                            .map(|_| Ok((vbin_ulong.clone(), PrintSpecifier::UpperHex))),
                        alt((tag("lo"), tag("Lo"), tag("zo")))
                            .map(|_| Ok((vbin_ulong.clone(), PrintSpecifier::Oct))),
                    )),
                    alt((
                        alt((tag("lld"), tag("lli")))
                            .map(|_| Ok((VBinSpecifier::I64, PrintSpecifier::Dec))),
                        tag("llu").map(|_| Ok((VBinSpecifier::U64, PrintSpecifier::Dec))),
                        tag("llx").map(|_| Ok((VBinSpecifier::U64, PrintSpecifier::Hex))),
                        tag("llX").map(|_| Ok((VBinSpecifier::U64, PrintSpecifier::UpperHex))),
                        tag("llo").map(|_| Ok((VBinSpecifier::U64, PrintSpecifier::Oct))),
                    )),
                )),
            ),
            preceded(
                char('p'),
                cut(context(
                    "pointer",
                    alt((
                        // Some pointers encoded as a pre-rendered string in the
                        // vbin_printf buffer, so they use the VBinSpecifier::Str .
                        // Others are regular pointers and can be looked up in
                        // kallsyms.
                        alt((
                            alt((
                                alt((char('f'), char('s')))
                                    .map(|_| Ok((vbin_ulong.clone(), PrintSpecifier::Symbol))),
                                alt((tag("F"), tag("SR"), tag("S"))).map(|_| {
                                    Ok((vbin_ulong.clone(), PrintSpecifier::SymbolWithOffset))
                                }),
                                tag("B").map(|_| {
                                    Ok((VBinSpecifier::Str, PrintSpecifier::SymbolWithOffset))
                                }),
                            )),
                            alt((
                                alt((char('r'), char('R')))
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Resource))),
                                preceded(
                                    char('a'),
                                    alt((
                                        char('d').map(|_| {
                                            Ok((VBinSpecifier::Str, PrintSpecifier::DmaAddr))
                                        }),
                                        opt(char('p')).map(|_| {
                                            Ok((VBinSpecifier::Str, PrintSpecifier::PhysAddr))
                                        }),
                                    )),
                                ),
                                preceded(
                                    char('E'),
                                    many0(alt((
                                        char('a'),
                                        char('c'),
                                        char('n'),
                                        char('o'),
                                        char('p'),
                                        char('s'),
                                    ))),
                                )
                                .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::EscapedBuffer))),
                            )),
                            alt((
                                preceded(
                                    char('h'),
                                    alt((
                                        char('C').map(|_| HexBufferSeparator::C),
                                        char('D').map(|_| HexBufferSeparator::D),
                                        char('N').map(|_| HexBufferSeparator::N),
                                    )),
                                )
                                .map(|sep| {
                                    Ok((VBinSpecifier::Str, PrintSpecifier::HexBuffer(sep)))
                                }),
                                preceded(char('M'), opt(alt((char('R'), char('F')))))
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::MacAddress))),
                                preceded(char('m'), opt(char('R')))
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::MacAddress))),
                            )),
                            alt((
                                separated_pair(
                                    alt((char('i'), char('I'))),
                                    char('4'),
                                    opt(ip_endianness()),
                                )
                                .map(|(_, endianness)| {
                                    let endianness = endianness.unwrap_or(Endianness::Big);
                                    Ok((VBinSpecifier::Str, PrintSpecifier::Ipv4(endianness)))
                                }),
                                alt((tag("I6"), tag("i6"), tag("I6c")))
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Ipv6))),
                                preceded(
                                    alt((char('i'), char('I'))).map(|_| ()),
                                    preceded(
                                        char('S'),
                                        many0(alt((
                                            alt((char('p'), char('f'), char('s'), char('c')))
                                                .map(|_| None),
                                            ip_endianness().map(Some),
                                        ))),
                                    ),
                                )
                                .map(|flags| {
                                    let endianness = flags
                                        .into_iter()
                                        .flatten()
                                        .last()
                                        .unwrap_or(Endianness::Big);
                                    Ok((VBinSpecifier::Str, PrintSpecifier::IpGeneric(endianness)))
                                }),
                            )),
                            preceded(char('U'), alt((char('b'), char('B'), char('l'), char('L'))))
                                .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Uuid))),
                            tuple((
                                alt((char('d'), char('D'))),
                                opt(alt((char('2'), char('3'), char('4')))),
                            ))
                            .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Dentry))),
                            alt((
                                char('g')
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::BlockDevice))),
                                char('V')
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::VaFormat))),
                                preceded(
                                    tag("OF"),
                                    opt(alt((
                                        char('f'),
                                        char('n'),
                                        char('p'),
                                        char('P'),
                                        char('c'),
                                        char('C'),
                                        char('F'),
                                    ))),
                                )
                                .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Kobject))),
                            )),
                            alt((
                                preceded(char('C'), opt(alt((char('n'), char('r')))))
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Clock))),
                                preceded(char('b'), opt(char('l')))
                                    .map(|_| Ok((VBinSpecifier::Str, PrintSpecifier::Bitmap))),
                                preceded(
                                    char('G'),
                                    alt((
                                        char('p').map(|_| PrintSpecifier::PageFlags),
                                        char('g').map(|_| PrintSpecifier::GfpFlags),
                                        char('v').map(|_| PrintSpecifier::VmaFlags),
                                    )),
                                )
                                .map(|specifier| Ok((VBinSpecifier::Str, specifier))),
                                tag("NF").map(|_| {
                                    Ok((VBinSpecifier::Str, PrintSpecifier::NetdevFeatures))
                                }),
                            )),
                        )),
                        // Simple pointers encoded as integers in vbin_printf buffer
                        alt((char('x').map(|_| ()), char('K').map(|_| ()), success(())))
                            .map(|_x| Ok((vbin_ulong.clone(), PrintSpecifier::Hex))),
                    )),
                )),
            ),
            alt((char('n'), char('e'), char('f'), char('g'), char('a')))
                .map(|c| Err(PrintFmtError::IllegalSpecifier(String::from_iter([c])))),
        ))
        .parse(input)
    }
}

// Formats documented there:
// https://www.kernel.org/doc/Documentation/printk-formats.txt
// Plus some specifiers that are undocumented
fn print_fmt_parser<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<&'a [u8], PrintFmtStr, E> + 'abi
where
    E: 'abi
        + ParseError<&'a [u8]>
        + ContextError<&'a [u8]>
        + FromExternalError<&'a [u8], PrintFmtError>
        + Debug,
    'a: 'abi,
{
    move |input| {
        let (_vbin_long, _vbin_ulong) = match abi.long_size {
            LongSize::Bits32 => (VBinSpecifier::I32, VBinSpecifier::U32),
            LongSize::Bits64 => (VBinSpecifier::I64, VBinSpecifier::U64),
        };

        let flags = || {
            context(
                "flag",
                alt((
                    char('-').map(|_| PrintFlags::LeftJustify),
                    char('+').map(|_| PrintFlags::PositiveSign),
                    char(' ').map(|_| PrintFlags::SignPlaceholder),
                    char('#').map(|_| PrintFlags::BasePrefix),
                    char('0').map(|_| PrintFlags::ZeroPad),
                )),
            )
        };

        let precision = || {
            alt((
                preceded(
                    char('.'),
                    context(
                        "precision",
                        alt((
                            map_res_cut(preceded(char('*'), specifier(abi)), |spec| {
                                let (vbin_spec, print_spec) = spec?;
                                Ok((vbin_spec, print_spec, PrintPrecision::Dynamic))
                            }),
                            map_res_cut(tuple((opt(txt_u64), specifier(abi))), |(width, spec)| {
                                let (vbin_spec, print_spec) = spec?;
                                // No value after the dot is same as an explicit 0
                                let width = width.unwrap_or(0);
                                let width = width.try_into().unwrap();
                                Ok((vbin_spec, print_spec, PrintPrecision::Fixed(width)))
                            }),
                        )),
                    ),
                ),
                map_res_cut(specifier(abi), |spec| {
                    let (vbin_spec, print_spec) = spec?;
                    Ok((vbin_spec, print_spec, PrintPrecision::Unmodified))
                }),
            ))
        };

        let width = || {
            alt((
                context(
                    "width",
                    alt((
                        map_res_cut(
                            preceded(char('*'), precision()),
                            |(vbin_spec, print_spec, precision)| {
                                Ok((vbin_spec, print_spec, precision, PrintWidth::Dynamic))
                            },
                        ),
                        map_res_cut(
                            tuple((txt_u64, precision())),
                            |(width, (vbin_spec, print_spec, precision))| {
                                Ok((
                                    vbin_spec,
                                    print_spec,
                                    precision,
                                    PrintWidth::Fixed(width.try_into().unwrap()),
                                ))
                            },
                        ),
                    )),
                ),
                map_res_cut(precision(), |(vbin_spec, print_spec, precision)| {
                    Ok((vbin_spec, print_spec, precision, PrintWidth::Unmodified))
                }),
            ))
        };

        let mut parser = context(
            "printk format string",
            map_res_cut(
                many0(alt((
                    preceded(
                        char('%'),
                        context(
                            "specifier",
                            alt((
                                char('%').map(|_| PrintAtom::Fixed("%".into())),
                                tuple((many1(flags()), width())).map(
                                    |(flags, (vbin_spec, print_spec, precision, width))| {
                                        let flags = PrintFlags::from_iter(flags);
                                        PrintAtom::new_variable(
                                            abi, vbin_spec, print_spec, flags, width, precision,
                                        )
                                    },
                                ),
                                width().map(|(vbin_spec, print_spec, precision, width)| {
                                    PrintAtom::new_variable(
                                        abi,
                                        vbin_spec,
                                        print_spec,
                                        PrintFlags::empty(),
                                        width,
                                        precision,
                                    )
                                }),
                            )),
                        ),
                    ),
                    context(
                        "fixed",
                        map_res_cut(is_not("%"), |s: &[u8]| {
                            Ok(PrintAtom::Fixed(from_utf8(s)?.into()))
                        }),
                    ),
                ))),
                |atoms: Vec<PrintAtom>| -> Result<PrintFmtStr, PrintFmtError> {
                    // Merge consecutive PrintAtom::Fixed together
                    let mut merged = Vec::with_capacity(atoms.len());
                    atoms
                        .iter()
                        .group_by(|x| matches!(x, PrintAtom::Fixed(_)))
                        .into_iter()
                        .map(|(key, group)| {
                            if key {
                                let merged_s = group
                                    .map(|x| match x {
                                        PrintAtom::Fixed(s) => s,
                                        _ => panic!("Expected fixed atom"),
                                    })
                                    .collect();
                                merged.push(PrintAtom::Fixed(merged_s))
                            } else {
                                merged.extend(group.cloned())
                            }
                        })
                        .for_each(drop);

                    Ok(PrintFmtStr {
                        atoms: merged,
                        vbin_decoders: OnceCell::new(),
                    })
                },
            ),
        );
        parser.parse(input)
    }
}

#[inline]
pub fn parse_print_fmt(header: &Header, fmt: &[u8]) -> Result<PrintFmtStr, PrintFmtError> {
    print_fmt_parser::<crate::parser::NomError<_, nom::error::VerboseError<_>>>(header.kernel_abi())
        .parse_finish(fmt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{header::Endianness, parser::tests::test_parser};

    #[test]
    fn print_fmt_test() {
        let abi = Abi {
            long_size: LongSize::Bits64,
            endianness: Endianness::Little,
            char_signedness: Signedness::Unsigned,
        };

        let test = |src: &[u8], expected: Vec<PrintAtom>| {
            let expected = PrintFmtStr {
                vbin_decoders: OnceCell::new(),
                atoms: expected,
            };
            test_parser(expected, src, print_fmt_parser(&abi))
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
            b"%u",
            vec![new_variable_atom!(
                VBinSpecifier::U32,
                PrintSpecifier::Dec,
                PrintFlags::empty(),
                PrintWidth::Unmodified,
                PrintPrecision::Unmodified,
            )],
        );

        test(
            b"%03d",
            vec![new_variable_atom!(
                VBinSpecifier::I32,
                PrintSpecifier::Dec,
                PrintFlags::ZeroPad,
                PrintWidth::Fixed(3),
                PrintPrecision::Unmodified,
            )],
        );

        test(
            b"%#016x",
            vec![new_variable_atom!(
                VBinSpecifier::U32,
                PrintSpecifier::Hex,
                PrintFlags::BasePrefix | PrintFlags::ZeroPad,
                PrintWidth::Fixed(16),
                PrintPrecision::Unmodified,
            )],
        );
        test(
            b"%#016.42x",
            vec![new_variable_atom!(
                VBinSpecifier::U32,
                PrintSpecifier::Hex,
                PrintFlags::BasePrefix | PrintFlags::ZeroPad,
                PrintWidth::Fixed(16),
                PrintPrecision::Fixed(42),
            )],
        );
        test(
            b"%#016.*x",
            vec![new_variable_atom!(
                VBinSpecifier::U32,
                PrintSpecifier::Hex,
                PrintFlags::BasePrefix | PrintFlags::ZeroPad,
                PrintWidth::Fixed(16),
                PrintPrecision::Dynamic,
            )],
        );

        test(
            b"%px",
            vec![new_variable_atom!(
                VBinSpecifier::U64,
                PrintSpecifier::Hex,
                PrintFlags::empty(),
                PrintWidth::Unmodified,
                PrintPrecision::Unmodified,
            )],
        );

        test(
            b"%p",
            vec![new_variable_atom!(
                VBinSpecifier::U64,
                PrintSpecifier::Hex,
                PrintFlags::empty(),
                PrintWidth::Unmodified,
                PrintPrecision::Unmodified,
            )],
        );

        test(
            b"%pap",
            vec![new_variable_atom!(
                VBinSpecifier::Str,
                PrintSpecifier::PhysAddr,
                PrintFlags::empty(),
                PrintWidth::Unmodified,
                PrintPrecision::Unmodified,
            )],
        );

        test(
            b"%pad",
            vec![new_variable_atom!(
                VBinSpecifier::Str,
                PrintSpecifier::DmaAddr,
                PrintFlags::empty(),
                PrintWidth::Unmodified,
                PrintPrecision::Unmodified,
            )],
        );

        test(
            b"foo %pa",
            vec![
                PrintAtom::Fixed("foo ".into()),
                new_variable_atom!(
                    VBinSpecifier::Str,
                    PrintSpecifier::PhysAddr,
                    PrintFlags::empty(),
                    PrintWidth::Unmodified,
                    PrintPrecision::Unmodified,
                ),
            ],
        );

        test(
            b"foo %u bar %px baz %%%pS",
            vec![
                PrintAtom::Fixed("foo ".into()),
                new_variable_atom!(
                    VBinSpecifier::U32,
                    PrintSpecifier::Dec,
                    PrintFlags::empty(),
                    PrintWidth::Unmodified,
                    PrintPrecision::Unmodified,
                ),
                PrintAtom::Fixed(" bar ".into()),
                new_variable_atom!(
                    VBinSpecifier::U64,
                    PrintSpecifier::Hex,
                    PrintFlags::empty(),
                    PrintWidth::Unmodified,
                    PrintPrecision::Unmodified,
                ),
                PrintAtom::Fixed(" baz %".into()),
                new_variable_atom!(
                    VBinSpecifier::U64,
                    PrintSpecifier::SymbolWithOffset,
                    PrintFlags::empty(),
                    PrintWidth::Unmodified,
                    PrintPrecision::Unmodified,
                ),
            ],
        );
    }
}
