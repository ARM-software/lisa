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

use std::{io::Write, ops::Deref as _};

use traceevent::{
    buffer::{flyrecord, BufferError, EventVisitor},
    cinterp::{BufferEnv, CompileError, Value},
    header::{EventDesc, FieldFmt, Header, HeaderError},
    io::BorrowingReadCore,
    print::{PrintError, StringWriter},
    scratch::ScratchAlloc,
};

use crate::error::DynMultiError;

#[allow(clippy::enum_variant_names)]
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum MainError {
    #[error("Error while loading data: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Error while parsing header: {0}")]
    HeaderError(#[from] HeaderError),

    #[error("Error while parsing buffer: {0}")]
    BufferError(#[from] BufferError),

    #[error("Error while parsing pretty printing: {0}")]
    PrintError(#[from] PrintError),

    #[error("Unexpected type for PID value {}", match .0 {
        Some(val) => val.to_string(),
        None => "<unavailable>".into()
    })]
    UnexpectedPidType(Option<Value<'static>>),
}

pub fn print_events<R: BorrowingReadCore + Send, W: Write>(
    header: &Header,
    reader: Box<R>,
    mut out: W,
    raw: bool,
) -> Result<(), DynMultiError> {
    let mut nr = 0;
    let scratch = &mut ScratchAlloc::new();

    let buffers = header.buffers(reader)?;
    let buf_id_len = buffers
        .iter()
        .map(|buf| buf.id.name.len())
        .max()
        .unwrap_or(0);

    struct EventCtx<'h> {
        pid_fmt: Option<&'h FieldFmt>,
    }
    impl<'h> EventCtx<'h> {
        fn from_event_desc(_header: &'h Header, desc: &'h EventDesc) -> Self {
            let get = || {
                let struct_fmt = &desc.event_fmt().ok()?.struct_fmt().ok()?;
                let pid_fmt = struct_fmt.field_by_name("common_pid")?;
                Some(pid_fmt)
            };
            let pid_fmt = get();

            EventCtx { pid_fmt }
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
            let ctx: &EventCtx = visitor.event_ctx()?;

            let pid_fmt = ctx.pid_fmt.ok_or_else(|| {
                let err: BufferError = CompileError::UnknownField("common_pid".into()).into();
                err
            })?;

            let pid = visitor.field_by_fmt(pid_fmt)?;
            let pid = match pid {
                Value::I64Scalar(x) => Ok(x.try_into().unwrap()),
                val => Err(MainError::UnexpectedPidType(val.into_static().ok())),
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

            let mut print_raw = ensure_hrtb(|out: &mut W, visitor: &EventVisitor<_, _>| -> Result<(), MainError> {
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

            let mut print_pretty = ensure_hrtb(|out: &mut W, _visitor: &EventVisitor<_, _>| -> Result<(), MainError>{
                let print_fmt = &desc.event_fmt()?.print_fmt()?;
                let print_args = desc.event_fmt()?.print_args()?;

                let print_args = print_args.into_iter().map(|spec| -> Result<_, PrintError> {
                    match spec {
                        Err(err) => Err(err.clone().into()),
                        // Transmit the error to interpolate_values() so it can display it
                        // at the right spot. It will then fail with the error after having
                        // printed to the output.
                        Ok(eval) => eval.eval(&env).map_err(Into::into),
                    }
                });
                let mut out = StringWriter::new(out);
                Ok(print_fmt
                    .interpolate_values(header, &env, &mut out, print_args)?)
            });

            if raw {
                print_raw(&mut out, &visitor)
            } else {
                match print_pretty(&mut out, &visitor) {
                    Ok(x) => Ok(x),
                    Err(err) => {
                        print_raw(&mut out, &visitor)?;
                        write!(&mut out, "\nError while pretty printing event {name}, fell back on raw printing: {err}")?;
                        // This error affected a single event and we displayed it, so we can just
                        // swallow it and not propagate
                        Ok(())
                    }
                }
            }
        }};
    }

    macro_rules! print_event {
        () => {{
            |record| -> Result<(), MainError> {
                let display_err = |out: &mut W, err: MainError, desc: Option<&EventDesc>| {
                    let name = match desc {
                        Some(desc) => desc.name.deref(),
                        None => &"<unknown>",
                    };

                    writeln!(out, "Error while processing event {name}: {err}")?;
                    Ok(())
                };

                let res = match record {
                    Ok(visitor) => {
                        match (|| visit!(&visitor))() {
                            err @ Err(MainError::IoError(..)) => err,

                            // Recoverable errors that affected one event only
                            Err(err) => display_err(&mut out, err, visitor.event_desc().ok()),
                            res => res,
                        }
                    }
                    Err(err) => match err {
                        // Display recoverable errors
                        err @ BufferError::LostEvents(..) => {
                            display_err(&mut out, err.into(), None)
                        }
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
    let res: Result<_, MainError> =
        flyrecord(buffers, print_event!(), EventCtx::from_event_desc).map_err(Into::into);
    res?.into_iter().collect::<Result<(), _>>()?;
    Ok(())
}
