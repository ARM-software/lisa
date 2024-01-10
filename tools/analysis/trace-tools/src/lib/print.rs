use std::{
    error::Error,
    io::{Error as IoError, Write},
    ops::Deref as _,
};

use traceevent::{
    buffer::{flyrecord, BufferError, EventVisitor},
    cinterp::{BufferEnv, CompileError, Value},
    header::{EventDesc, FieldFmt, Header, HeaderError},
    io::BorrowingRead,
    print::{PrintError, StringWriter},
    scratch::ScratchAlloc,
};

use crate::convert_err_impl;

#[allow(clippy::enum_variant_names)]
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum MainError {
    #[error("Error while loading data: {0}")]
    IoError(std::io::Error),

    #[error("Error while parsing header: {0}")]
    HeaderError(HeaderError),

    #[error("Error while parsing buffer: {0}")]
    BufferError(BufferError),

    #[error("Error while parsing pretty printing: {0}")]
    PrintError(PrintError),

    #[error("Unexpected type for PID value")]
    PIDTypeError,
}

convert_err_impl!(IoError, MainError);
convert_err_impl!(HeaderError, MainError);
convert_err_impl!(BufferError, MainError);

pub fn print_events<R: BorrowingRead + Send, W: Write>(
    header: &Header,
    reader: R,
    mut out: W,
) -> Result<(), Box<dyn Error>> {
    let mut nr = 0;
    let scratch = &mut ScratchAlloc::new();

    let buffers = header.buffers(reader).unwrap();
    let buf_id_len = buffers
        .iter()
        .map(|buf| buf.id.name.len())
        .max()
        .unwrap_or(0);

    struct EventInfo<'h> {
        pid_fmt: Option<&'h FieldFmt>,
    }
    impl<'h> EventInfo<'h> {
        fn from_event_desc(_header: &'h Header, desc: &'h EventDesc) -> Self {
            let get = || {
                let struct_fmt = &desc.event_fmt().ok()?.struct_fmt().ok()?;
                let pid_fmt = struct_fmt.field_by_name("common_pid")?;
                Some(pid_fmt)
            };
            let pid_fmt = get();

            EventInfo { pid_fmt }
        }
    }

    macro_rules! visit {
        ($visitor:expr) => {{
            let visitor = $visitor;
            nr += 1;
            let ts = visitor.timestamp;
            let buf_id = visitor.buffer_id;
            let header = &visitor.header;
            let data = &visitor.data;
            let name = &visitor.event_name()?;
            let desc = visitor.event_desc()?;
            let info: &EventInfo = visitor.event_user_data()?;

            let unknown_field = || -> BufferError { CompileError::UnknownField.into() };
            let pid_fmt = info.pid_fmt.ok_or_else(unknown_field)?;

            let pid = visitor.field_by_fmt(pid_fmt)?;
            let pid = match pid {
                Value::I64Scalar(x) => Ok(x.try_into().unwrap()),
                _ => Err(MainError::PIDTypeError),
            }?;

            let comm = match pid {
                0 => "<idle>",
                pid => visitor
                    .header
                    .comm_of(pid)
                    .map(|s| s.deref())
                    .unwrap_or("<...>"),
            };

            match buf_id.name.deref() {
                "" => write!(
                    &mut out,
                    "{:<len$}",
                    "",
                    len = if buf_id_len > 0 { buf_id_len + 2 } else { 0 }
                )?,
                name => write!(&mut out, "{name:<len$}: ", len = buf_id_len)?,
            }

            let buf_cpu = buf_id.cpu;
            let ts_sec = ts / 1_000_000_000;
            let ts_dec = ts % 1_000_000_000;
            write!(
                &mut out,
                "{comm:>16}-{pid:<5} [{buf_cpu:0>3}] {ts_sec}.{ts_dec:0>9}: {}:{:>name_pad$} ",
                name,
                "",
                name_pad = 20usize.saturating_sub(name.len()),
            )?;

            let env = BufferEnv::new(scratch, header, data);

            // rustc fails to infer Higher Rank Trait Bound (HRTB) for the lifetime of references
            // passed as parameters, so we force them to be
            fn ensure_hrtb<T1, T2, U, F>(f: F) -> F
            where
                F: FnMut(&mut T1, &T2) -> U,
            {
                f
            }

            let mut print_raw = ensure_hrtb(|out: &mut W, visitor: &EventVisitor<_, _>| {
                for (fmt, val) in &mut visitor.fields()? {
                    let val = val?;
                    // let val = Value::U64Scalar(0xffff800009db03f8);
                    let derefed = val.deref_ptr(&env);
                    let val = match derefed {
                        Ok(val) => val,
                        Err(_) => {
                            drop(derefed);
                            val
                        }
                    };
                    let field_name = &fmt.declaration.identifier;
                    match val.to_str() {
                        Some(s) => write!(out, " {field_name}={s}")?,
                        None => write!(out, " {field_name}={val}")?,
                    }
                }
                Ok(())
            });

            let mut print_pretty = ensure_hrtb(|out: &mut W, _visitor: &EventVisitor<_, _>| {
                let print_fmt = &desc.event_fmt()?.print_fmt()?;
                let print_args = &desc.event_fmt()?.print_args()?;

                let print_args = print_args.iter().map(|spec| -> Result<_, PrintError> {
                    match spec {
                        Err(err) => Err(err.clone().into()),
                        // Transmit the error to interpolate_values() so it can display it
                        // at the right spot. It will then fail with the error after having
                        // printed to the output.
                        Ok(eval) => eval.eval(&env).map_err(Into::into),
                    }
                });
                let mut out = StringWriter::new(out);
                print_fmt
                    .interpolate_values(header, &env, &mut out, print_args)
                    .map_err(MainError::PrintError)
            });

            let pretty = true;
            if pretty {
                match print_pretty(&mut out, &visitor) {
                    Ok(x) => Ok(x),
                    Err(err) => {
                        write!(&mut out, "ERROR WHILE PRETTY PRINTING: {err}")?;
                        print_raw(&mut out, &visitor)
                    }
                }
            } else {
                print_raw(&mut out, &visitor)
            }
        }};
    }

    macro_rules! print_event {
        () => {{
            |record| -> Result<(), MainError> {
                let display_err = |out: &mut W, err: MainError| {
                    write!(out, "ERROR: {err}")?;
                    Ok(())
                };

                let res = match record {
                    Ok(visitor) => {
                        match visit!(visitor) {
                            // Ignore any error while printing the event, as any C evaluation error will have
                            // been embedded in the printed string already.
                            Err(MainError::PrintError(_)) => Ok(()),
                            // Recoverable errors that affected one event only
                            Err(err @ MainError::HeaderError(_) | err @ MainError::BufferError(_)) => {
                                display_err(&mut out, err)
                            }
                            res => res,
                        }
                    }
                    Err(err) => match err {
                        // Display recoverable errors
                        err @ BufferError::LostEvents(..) => display_err(&mut out, err.into()),
                        // Propagate non-recoverable ones
                        err => Err(err.into()),
                    },
                };

                // Reduce the overhead of reseting the scratch allocator.
                if (nr % 16) == 0 {
                    scratch.reset();
                }
                writeln!(&mut out)?;
                res
            }
        }};
    }
    let res =
        flyrecord(buffers, print_event!(), EventInfo::from_event_desc)?.into_iter().collect::<Result<(), _>>();
    Ok(res?)
}
