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
    cell::{Cell, RefCell, RefMut},
    convert::identity,
    ops::{Deref, DerefMut},
};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    fs::File,
    io::Write,
    path::PathBuf,
    rc::Rc,
    sync::Arc,
};

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow_array::{
    RecordBatch,
    array::Array,
    builder::{
        ArrayBuilder, BinaryBuilder, BooleanBuilder, Int8Builder, Int16Builder, Int32Builder,
        Int64Builder, ListBuilder, StringBuilder, UInt8Builder, UInt16Builder, UInt32Builder,
        UInt64Builder,
    },
};
use arrow_schema::ArrowError;
use crossbeam::{
    channel::{Sender, bounded},
    thread::{Scope, ScopedJoinHandle, scope},
};
use nom::{Finish as _, Parser as _};
use parquet::{
    arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions},
    basic::{Compression, Encoding},
    errors::ParquetError,
    file::properties::{
        EnabledStatistics, WriterProperties, WriterPropertiesBuilder, WriterVersion,
    },
    format::{KeyValue, SortingColumn},
    schema::types::ColumnPath,
};
use serde::Serialize;
use traceevent::{
    buffer::{BufferError, EventVisitor},
    cinterp::{EvalEnv, EvalError, Value},
    cparser::{ArrayKind, Type, identifier},
    header::{Address, EventDesc, EventId, FieldFmt, Header, HeaderError, LongSize, Timestamp},
    io::BorrowingReadCore,
    print::{PrintArg, PrintAtom, PrintFmtError, PrintFmtStr, VBinSpecifier},
};

use crate::error::DynMultiError;

#[allow(clippy::enum_variant_names)]
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum MainError {
    #[error("Error while loading data: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Error while creating JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Error while parsing header: {0}")]
    HeaderError(#[from] HeaderError),

    #[error("Error while parsing buffer: {0}")]
    BufferError(#[from] BufferError),

    #[error("Error while interpreting event data: {0}")]
    EvalError(#[from] EvalError),

    #[error("Error while parsing print format string: {0}")]
    PrintFmtError(#[from] PrintFmtError),

    #[error("Arrow error: {0}")]
    ArrowError(#[from] ArrowError),

    #[error("Parquet error: {0}")]
    ParquetError(#[from] ParquetError),

    #[error("Type not handled: {0:?}")]
    TypeNotHandled(Box<Type>),

    #[error("Arrow data type not handled: {0:?}")]
    ArrowDataTypeNotHandled(Box<DataType>),

    #[error(
        "Runtime data cannot be used according to the column storage schema: {}",
        match .0 {
            Some(x) => x.to_string(),
            None => "<unavailable>".to_string(),
        }
    )]
    DataMismatchingSchema(Option<Box<Value<'static>>>),

    #[error("Missing field")]
    MissingField,

    #[error("This print format string does not describe a meta event")]
    NotAMetaEvent,

    #[error("Error while processing {} field {}: {}", .0.field_name, .0.event_name.as_deref().unwrap_or("<unknown event>"), .0.error)]
    FieldError(Box<FieldError>),

    #[error("Writer thread error")]
    WriterThreadError,
}

#[derive(Debug)]
pub struct FieldError {
    event_name: Option<String>,
    field_name: String,
    error: MainError,
}

impl MainError {
    fn with_field(self, event_name: Option<&str>, field_name: &str) -> Self {
        MainError::FieldError(Box::new(FieldError {
            event_name: event_name.map(Into::into),
            field_name: field_name.into(),
            error: self,
        }))
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct EventMetadata {
    event: String,
    errors: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    nr_rows: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,

    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Metadata<'h> {
    header: &'h Header,
    events_info: Option<Vec<EventMetadata>>,
    time_range: Option<(Timestamp, Timestamp)>,
}

impl Metadata<'_> {
    pub fn dump<W: Write>(self, mut writer: W) -> Result<(), MainError> {
        let header = self.header;
        let events_info = self.events_info;
        let time_range = self.time_range;

        let mut json_value = serde_json::json!({
            "pid-comms": header.pid_comms().collect::<Vec<_>>(),
            "cpus-count": header.nr_cpus(),
            "symbols-address": header.kallsyms().map(|(addr, syms)| (addr, syms.collect())).collect::<Vec<(Address, Vec<&str>)>>(),
            // We cannot provide all events the kernel support here, as most of them
            // were not enabled during the trace.
            // "available-events": header.event_descs().into_iter().map(|desc| desc.name.deref()).collect::<Vec<&str>>(),
        });

        if let Some(events_info) = events_info {
            json_value["events-info"] = serde_json::to_value(events_info)?;
        }

        if let Some(time_range) = time_range {
            json_value["time-range"] = vec![time_range.0, time_range.1].into();
        }

        if let Some(id) = header.trace_id() {
            json_value["trace-id"] = id.into();
        }
        if let Some(clock) = header.clock() {
            json_value["trace-clock"] = clock.into();
        }

        Ok(writer.write_all(json_value.to_string().as_bytes())?)
    }
}

#[derive(Clone, Debug)]
enum EventCtx<T> {
    Selected(T),
    NotSelected,
}

#[derive(Clone, Debug)]
struct SharedState<'scope, 'scopeenv>(Rc<RefCell<Result<ReadState<'scope, 'scopeenv>, MainError>>>);

impl<'scope, 'scopeenv> SharedState<'scope, 'scopeenv> {
    fn new(x: Result<ReadState<'scope, 'scopeenv>, MainError>) -> Self {
        SharedState(Rc::new(RefCell::new(x)))
    }

    #[inline]
    fn borrow_mut<'a>(
        &'a self,
    ) -> impl DerefMut<Target = Result<ReadState<'scope, 'scopeenv>, MainError>> + 'a {
        RefCell::borrow_mut(&self.0)
    }

    #[inline]
    fn into_inner(self) -> Result<Result<ReadState<'scope, 'scopeenv>, MainError>, Self> {
        match Rc::try_unwrap(self.0) {
            Ok(refcell) => Ok(refcell.into_inner()),
            Err(inner) => Err(SharedState(inner)),
        }
    }
}

pub fn dump_events<R, FTimestamp>(
    header: &Header,
    reader: Box<R>,
    mut modify_timestamps: FTimestamp,
    only_events: Option<Vec<String>>,
    row_group_size: usize,
    compression: Compression,
    max_errors: usize,
) -> Result<Metadata<'_>, DynMultiError>
where
    FTimestamp: FnMut(Timestamp) -> Timestamp,
    R: BorrowingReadCore + Send,
{
    // Size of a chunk accumulated in memory before handing it over to the parquet-writing thread.
    // This is independent from the row group size, which may be bigger.
    let chunk_size: usize = 64 * 1024;

    let only_events = &only_events;
    // TODO: EventId might not be enough if we extend the API to deal with buffers from multiple
    // traces
    //
    // Keep the per-event state in a map that is shared between all buffers. Otherwise, we would
    // end up with a state per-event and per-buffer, which is not what we want.
    type StateMap<'scope, 'scopeenv> = BTreeMap<EventId, EventCtx<SharedState<'scope, 'scopeenv>>>;

    macro_rules! chunk_append {
        ($($arms:expr),*) => {
            loop {
                $(
                    $arms;
                )*

                default_error!()
            }
        }
    }

    macro_rules! make_macros {
        ($scrutinee:expr) => {
            let scrutinee = $scrutinee;

            macro_rules! default_error {
                () => {{
                    break Err(MainError::DataMismatchingSchema(
                        scrutinee.1.into_static().map(Box::new).ok(),
                    ));
                }};
            }

            macro_rules! basic {
                ($pat:pat => $expr:expr) => {
                    if let $pat = scrutinee {
                        let xs = $expr;
                        break Ok(xs.len());
                    }
                };
            }

            macro_rules! integer {
                ($arr_ctor:path, $val_ctor:path, $f:expr) => {
                    if let ($arr_ctor(xs), x) = scrutinee {
                        match x {
                            $val_ctor(x) => {
                                #[allow(clippy::redundant_closure_call)]
                                xs.append_value($f(x));
                                break Ok(xs.len());
                            }
                            x => {
                                break Err(MainError::DataMismatchingSchema(
                                    x.into_static().map(Box::new).ok(),
                                ));
                            }
                        }
                    }
                };
            }
        };
    }

    let creator = format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));

    let make_props_builder = || {
        WriterProperties::builder()
            .set_encoding(Encoding::PLAIN)
            .set_compression(compression)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_created_by(creator.clone())
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_max_row_group_size(row_group_size)
        // TODO: Revisit enabling bloom filters. As of March 2024, polars cannot make use of it
        // anyway so not really worth it. Also, we already use dictionary encoding for strings,
        // which allows to quickly check if page contains a given string. It's not much of a
        // use for smaller scalar types.
        // .set_bloom_filter_enabled(true)
    };

    scope(move |scope| {
        let mut count: u64 = 0;
        let buffers = header.buffers(reader)?;
        let mut state_map = StateMap::new();
        let mut time_range = (None, None);

        let mut make_ctx = |header: &_, event_desc: &EventDesc| {
            let id = event_desc.id;
            match state_map.entry(id) {
                Entry::Vacant(entry) => {
                    let select = match only_events {
                        None => true,
                        Some(only_events) => only_events.iter().any(|selected| {
                            event_desc.name.deref() == selected.deref()
                        })
                    };
                    let state = if select {
                        let state = {
                            ReadState::new(header, event_desc, chunk_size, &event_desc.name, make_props_builder(), scope, max_errors)
                        };

                        EventCtx::Selected(SharedState::new(state))
                    } else {
                        EventCtx::NotSelected
                    };
                    entry.insert(state.clone());
                    state
                }
                Entry::Occupied(entry) => entry.get().clone(),
            }
        };

        let events = traceevent::buffer::flyrecord(
            buffers,
            {
                let time_range = &mut time_range;
                let count = &mut count;
                let mut slice_scratch = Vec::new();
                move |res: Result<EventVisitor<_, EventCtx<SharedState>>, _>| -> Result<(), MainError> {
                    match res {
                        Ok(visitor) => {
                            *count += 1;
                            // This needs to happen regardless of whether the event is selected,
                            // otherwise the resulting timestamps would vary based on the set of
                            // events selected, making caching of the parquet files problematic.
                            let ts = modify_timestamps(visitor.timestamp);
                            *time_range = match time_range {
                                (None, _) => (Some(ts), None),
                                (first@Some(_), _) => (*first, Some(ts)),
                            };

                            match visitor.event_ctx()? {
                                EventCtx::Selected(state) => {
                                    let mut state = state.borrow_mut();
                                    let state = state.deref_mut();
                                    match state {
                                        // We ignore the error here as it will be reported at the end when finalizing
                                        // the files.
                                        Err(_) => Ok(()),
                                        Ok(state) => {
                                            let buf_id = visitor.buffer_id;
                                            let event_desc = visitor.event_desc()?;

                                            let cpu = buf_id.cpu;
                                            let len: Cell<usize> = Cell::new(0);

                                            let mut table_state = state.process_fields(
                                                &visitor,
                                                |field_name, col, val| -> Result<(), MainError> {
                                                    let val = val?;

                                                    macro_rules! cast {
                                                        ($from:ty, $to:ty) => {
                                                            |x: $from| -> $to {
                                                                let x: $to = match x.try_into() {
                                                                    Ok(x) => x,
                                                                    Err(_) => {
                                                                        let conv =
                                                                            concat!("from ", stringify!($from), " to ", stringify!($to));
                                                                        panic!(
                                                                            "Cannot convert {}.{}={x} at t={ts} on CPU {cpu} {conv}",
                                                                            event_desc.name, field_name,
                                                                        )
                                                                    }
                                                                };
                                                                x
                                                            }
                                                        };
                                                    }

                                                    make_macros!((col, val));
                                                    len.replace(chunk_append! {
                                                        // Integers
                                                        integer!(FieldArray::I8,  Value::I64Scalar, cast!(i64, i8)),
                                                        integer!(FieldArray::I16, Value::I64Scalar, cast!(i64, i16)),
                                                        integer!(FieldArray::I32, Value::I64Scalar, cast!(i64, i32)),
                                                        integer!(FieldArray::I64, Value::I64Scalar, identity),

                                                        integer!(FieldArray::U8,  Value::U64Scalar, cast!(u64, u8)),
                                                        integer!(FieldArray::U16, Value::U64Scalar, cast!(u64, u16)),
                                                        integer!(FieldArray::U32, Value::U64Scalar, cast!(u64, u32)),
                                                        integer!(FieldArray::U64, Value::U64Scalar, identity),

                                                        // String
                                                        basic!((FieldArray::Str(xs), x) => {
                                                            xs.append_option(x.deref_ptr(&visitor.buffer_env())?.to_str());
                                                            xs
                                                        }),

                                                        // Bool
                                                        basic!((FieldArray::Bool(xs), Value::I64Scalar(x)) => {
                                                            xs.append_option(Some(x != 0));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Bool(xs), Value::U64Scalar(x)) => {
                                                            xs.append_option(Some(x != 0));
                                                            xs
                                                        }),

                                                        // Binary
                                                        basic!((FieldArray::Binary(xs), Value::U8Array(x)) => {
                                                            xs.append_value(x);
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::I8Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::U16Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::I16Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::U32Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::I32Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::U64Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::I64Array(x)) => {
                                                            xs.append_value(bytemuck::cast_slice(&x));
                                                            xs
                                                        }),

                                                        // Lists
                                                        basic!((FieldArray::ListBool(xs), Value::U8Array(x)) => {
                                                            xs.append_value(x.iter().map(|x| Some(*x != 0)));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListBool(xs), Value::I8Array(x)) => {
                                                            xs.append_value(x.iter().map(|x| Some(*x != 0)));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListU8(xs), Value::U8Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListI8(xs), Value::I8Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),

                                                        basic!((FieldArray::ListU16(xs), Value::U16Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListI16(xs), Value::I16Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),

                                                        basic!((FieldArray::ListU32(xs), Value::U32Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListI32(xs), Value::I32Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),

                                                        basic!((FieldArray::ListU64(xs), Value::U64Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListI64(xs), Value::I64Array(x)) => {
                                                            xs.append_value(x.iter().copied().map(Some));
                                                            xs
                                                        }),

                                                        // Bitmap
                                                        basic!((FieldArray::ListU8(xs), Value::Bitmap(x)) => {
                                                            xs.append_value(x.into_iter().as_bytes().map(Some));
                                                            xs
                                                        }),
                                                        basic!((FieldArray::Binary(xs), Value::Bitmap(x)) => {
                                                            slice_scratch.clear();
                                                            slice_scratch.extend(x.into_iter().as_bytes());
                                                            xs.append_value(&slice_scratch);
                                                            xs
                                                        }),
                                                        basic!((FieldArray::ListBool(xs), Value::Bitmap(x)) => {
                                                            xs.append_value(x.into_iter().as_bits().map(Some));
                                                            xs
                                                        })

                                                    }?);
                                                    Ok(())
                                                },
                                                only_events,
                                            )?;

                                            table_state.fixed_cols.time.append_value(ts);
                                            table_state.fixed_cols.cpu.append_value(cpu);
                                            table_state.nr_rows += 1;

                                            if len.get() >= chunk_size {
                                                let chunk = table_state.extract_batch()?;
                                                table_state.sender.send(chunk).map_err(|_| MainError::WriterThreadError)?;
                                            }
                                            Ok(())
                                        }
                                    }
                                }
                                _ => Ok(())
                            }
                        }
                        Err(err) => Err(err.into()),
                    }
                }
            },
            &mut make_ctx,
        )?;

        let mut errors = Vec::new();
        let mut push_global_err = |err: Result<(), MainError>| if let Err(err) = err {
            limited_append(&mut errors, err, max_errors);
        };

        // Even if there were some errors, we should have pushed some None values so that all
        // columns are of the same length, so the file can be finalized.
        events.into_iter().map(&mut push_global_err).for_each(drop);
        eprintln!("Found {count} event records in this trace");

        // TODO: We can do that the day we can know what event was asked to be recorded in the
        // trace-cmd invocation. Then we can legitimately say that there was no occurence of an
        // event. Until then, we consider that if we did not hit any occurence, the event was not
        // specified on trace-cmd CLI and was never enabled in the first place.
        //
        // Ensure we have a file for each event that was asked for as long as that event is
        // actually available in the trace header.
        // if let Some(only_events) = only_events {
            // for event in only_events {
                // header.event_desc_by_name(event).map(|event_desc| make_ctx(header, event_desc));
            // }
        // }

        let mut handles = Vec::new();
        let mut events_info = Vec::new();

        while let Some((id, ctx)) = state_map.pop_first() {
            push_global_err(match ctx {
                EventCtx::Selected(read_state) => {
                    // There shouldn't be any other clone of the Rc<> at this point so we can
                    // safely unwrap.
                    match read_state.into_inner().unwrap() {
                        Ok(read_state) => {
                            for mut read_state in read_state.drain_states() {
                                let res = match read_state.extract_batch() {
                                    Ok(chunk) => {
                                        read_state.sender.send(chunk).unwrap();
                                        // Drop the sender which will close the channel so that the writer thread will
                                        // know it's time to finish.
                                        drop(read_state.sender);
                                        handles.push(read_state.handle);
                                        eprintln!("File written successfully {}", read_state.name);
                                        Ok(())
                                    }
                                    Err(err) => Err(err),
                                };
                                read_state.errors.extend_errors([res]);

                                let errors = read_state.errors.errors;
                                if !errors.is_empty() {
                                    eprintln!("Errors encountered while dumping event {}, see meta.json for details", read_state.name);
                                }

                                events_info.push(EventMetadata {
                                    event: read_state.name.to_string(),
                                    nr_rows: Some(read_state.nr_rows),
                                    path: Some(read_state.path),
                                    format: Some("parquet".into()),
                                    errors: errors.into_iter().map(|err| err.to_string()).collect::<Vec<_>>(),
                                });
                            }
                            Ok(())
                        }
                        Err(err) => {
                            match header.event_desc_by_id(id) {
                                Some(desc) => {
                                    events_info.push(EventMetadata {
                                        event: desc.name.to_string(),
                                        nr_rows: None,
                                        path: None,
                                        format: Some("parquet".into()),
                                        errors: vec![err.to_string()],
                                    });
                                    Ok(())
                                },
                                // If we cannot get the associated event name, we just turn it into
                                // a global error.
                                _ => Err(err)
                            }
                        }
                    }
                }
                // Still register the events that we did not select, as they indicate what event is
                // available in that trace.
                EventCtx::NotSelected => {
                    match header.event_desc_by_id(id) {
                        Some(desc) => {
                            events_info.push(EventMetadata {
                                event: desc.name.to_string(),
                                nr_rows: None,
                                path: None,
                                format: None,
                                errors: vec![],
                            });
                            Ok(())
                        },
                        // This is not an event anyone requested, and we can't find its name, so
                        // that means no-one can request it. Since this is in-effect unselectable,
                        // we just ignore it.
                        None => Ok(())
                    }
                }
            })
        }

        for handle in handles {
            handle.join().expect("Writer thread panicked")?;
        }

        let time_range = match time_range {
            (Some(start), Some(end)) => (start, end),
            (Some(start), None) => (start, start),
            (None, None) => (0, 0),
            (None, Some(end)) => panic!("Time time_range has an end ({end}) but not a start"),
        };

        let metadata = Metadata {
            header,
            events_info: Some(events_info),
            time_range: Some(time_range),
        };

        if errors.is_empty() {
            Ok(metadata)
        } else {
            Err(DynMultiError::new(errors))
        }
    }).unwrap()
}

pub fn dump_metadata<R, W>(
    header: &Header,
    reader: Box<R>,
    writer: W,
    keys: Option<Vec<String>>,
    max_errors: usize,
) -> Result<(), DynMultiError>
where
    W: Write,
    R: BorrowingReadCore + Send,
{
    let scan_trace = match keys {
        Some(keys) => keys
            .iter()
            .any(|item| item == "available-events" || item == "time-range"),
        None => false,
    };

    // Some metadata require a full scan of the trace
    let metadata = if scan_trace {
        dump_events(
            header,
            reader,
            |ts| ts,
            // Do not create any parquet file.
            Some(vec![]),
            0,
            Compression::UNCOMPRESSED,
            max_errors,
        )?
    } else {
        Metadata {
            header,
            time_range: None,
            events_info: None,
        }
    };

    Ok(metadata.dump(writer)?)
}

#[derive(Debug)]
enum FieldArray {
    U8(UInt8Builder),
    U16(UInt16Builder),
    U32(UInt32Builder),
    U64(UInt64Builder),

    I8(Int8Builder),
    I16(Int16Builder),
    I32(Int32Builder),
    I64(Int64Builder),

    Bool(BooleanBuilder),
    // Using i32 means strings and binary blobs have to be smaller than 2GB, which should be fine
    Binary(BinaryBuilder),
    Str(StringBuilder),

    ListBool(ListBuilder<BooleanBuilder>),

    ListU8(ListBuilder<UInt8Builder>),
    ListU16(ListBuilder<UInt16Builder>),
    ListU32(ListBuilder<UInt32Builder>),
    ListU64(ListBuilder<UInt64Builder>),

    ListI8(ListBuilder<Int8Builder>),
    ListI16(ListBuilder<Int16Builder>),
    ListI32(ListBuilder<Int32Builder>),
    ListI64(ListBuilder<Int64Builder>),
}

impl FieldArray {
    fn finish(mut self) -> Arc<dyn Array> {
        match &mut self {
            FieldArray::U8(xs) => ArrayBuilder::finish(xs),
            FieldArray::U16(xs) => ArrayBuilder::finish(xs),
            FieldArray::U32(xs) => ArrayBuilder::finish(xs),
            FieldArray::U64(xs) => ArrayBuilder::finish(xs),

            FieldArray::I8(xs) => ArrayBuilder::finish(xs),
            FieldArray::I16(xs) => ArrayBuilder::finish(xs),
            FieldArray::I32(xs) => ArrayBuilder::finish(xs),
            FieldArray::I64(xs) => ArrayBuilder::finish(xs),

            FieldArray::Bool(xs) => ArrayBuilder::finish(xs),
            FieldArray::Str(xs) => ArrayBuilder::finish(xs),
            FieldArray::Binary(xs) => ArrayBuilder::finish(xs),

            FieldArray::ListBool(xs) => ArrayBuilder::finish(xs),

            FieldArray::ListU8(xs) => ArrayBuilder::finish(xs),
            FieldArray::ListU16(xs) => ArrayBuilder::finish(xs),
            FieldArray::ListU32(xs) => ArrayBuilder::finish(xs),
            FieldArray::ListU64(xs) => ArrayBuilder::finish(xs),

            FieldArray::ListI8(xs) => ArrayBuilder::finish(xs),
            FieldArray::ListI16(xs) => ArrayBuilder::finish(xs),
            FieldArray::ListI32(xs) => ArrayBuilder::finish(xs),
            FieldArray::ListI64(xs) => ArrayBuilder::finish(xs),
        }
    }

    fn append_null(&mut self) {
        match self {
            FieldArray::U8(xs) => xs.append_null(),
            FieldArray::U16(xs) => xs.append_null(),
            FieldArray::U32(xs) => xs.append_null(),
            FieldArray::U64(xs) => xs.append_null(),

            FieldArray::I8(xs) => xs.append_null(),
            FieldArray::I16(xs) => xs.append_null(),
            FieldArray::I32(xs) => xs.append_null(),
            FieldArray::I64(xs) => xs.append_null(),

            FieldArray::Bool(xs) => xs.append_null(),
            FieldArray::Str(xs) => xs.append_null(),
            FieldArray::Binary(xs) => xs.append_null(),

            FieldArray::ListBool(xs) => xs.append_null(),

            FieldArray::ListU8(xs) => xs.append_null(),
            FieldArray::ListU16(xs) => xs.append_null(),
            FieldArray::ListU32(xs) => xs.append_null(),
            FieldArray::ListU64(xs) => xs.append_null(),

            FieldArray::ListI8(xs) => xs.append_null(),
            FieldArray::ListI16(xs) => xs.append_null(),
            FieldArray::ListI32(xs) => xs.append_null(),
            FieldArray::ListI64(xs) => xs.append_null(),
        }
    }
}

#[derive(Debug)]
struct FixedCols {
    time: UInt64Builder,
    cpu: UInt32Builder,
}

impl FixedCols {
    fn new(chunk_size: usize) -> Self {
        FixedCols {
            time: UInt64Builder::with_capacity(chunk_size),
            cpu: UInt32Builder::with_capacity(chunk_size),
        }
    }

    fn arrow_fields() -> impl Iterator<Item = Field> {
        [
            Field::new("common_ts", DataType::UInt64, false),
            Field::new("common_cpu", DataType::UInt32, false),
        ]
        .into_iter()
    }

    fn into_finished(mut self) -> impl Iterator<Item = Arc<dyn Array>> {
        [
            ArrayBuilder::finish(&mut self.time),
            ArrayBuilder::finish(&mut self.cpu),
        ]
        .into_iter()
    }
}

#[derive(Debug)]
struct ReadState<'scope, 'scopeenv> {
    variant: ReadStateVariant<'scope>,
    chunk_size: usize,
    scope: &'scope Scope<'scopeenv>,
    max_errors: usize,
}

type MetaEventEntry<'scope> =
    Rc<EventCtx<Result<(RefCell<TableState<'scope>>, PrintFmtStr), MainError>>>;

#[derive(Debug)]
enum ReadStateVariant<'scope> {
    Generic(TableState<'scope>),
    BPrint {
        common_pid_fmt: FieldFmt,
        fmt_fmt: FieldFmt,
        buf_fmt: FieldFmt,
        generic: TableState<'scope>,
        // We have a table indexed by the address of the format string since that is what we get
        // from the bprint event.
        meta_events_by_addr: BTreeMap<Address, MetaEventEntry<'scope>>,
        // However, we don't want to accidentally create 2 identical meta events if 2 or more
        // independent format strings have the exact same format (e.g. the user copy-pasted some
        // calls to trace_printk() in various places). For this purpose, we also maintain a map
        // indexed by the format string content, which is used to populate the by-address map.
        meta_events_by_fmt: BTreeMap<PrintFmtStr, MetaEventEntry<'scope>>,
    },
}

impl<'scope, 'scopeenv> ReadState<'scope, 'scopeenv>
where
    'scopeenv: 'scope,
{
    fn new(
        header: &Header,
        event_desc: &EventDesc,
        chunk_size: usize,
        name: &str,
        props_builder: WriterPropertiesBuilder,
        scope: &'scope Scope<'scopeenv>,
        max_errors: usize,
    ) -> Result<Self, MainError> {
        let (full_schema, fields_schema) = Self::make_event_desc_schemas(header, event_desc)?;
        let clock = header.clock();

        let props_builder = props_builder.set_key_value_metadata(Some(vec![
            // FIXME: this is not correct for meta events like the ones extracted from BPrint
            KeyValue::new("FTRACE:event".to_string(), event_desc.name.to_string()),
            KeyValue::new(
                "FTRACE:event_format".to_string(),
                match event_desc.raw_fmt() {
                    Ok(s) => std::str::from_utf8(s).map(Into::into).ok(),
                    Err(_) => None,
                },
            ),
            KeyValue::new("FTRACE:clock".to_string(), clock.map(ToString::to_string)),
            KeyValue::new("FTRACE:trace_id".to_string(), header.trace_id()),
        ]));

        let monotonic_clocks = [
            "local", "global", "uptime", "x86-tsc", "mono", "mono_raw", "counter",
        ];

        // Some clocks guarantee that the timestamps will monotonically increase
        let sorted_ts = match clock {
            Some(clock) => monotonic_clocks.contains(&clock),
            None => false,
        };

        // If the timestamps are monotonic, we declare the common_ts column as being sorted in
        // ascending order. This hint can be used by parquet readers to speed up
        // selection/filtering (e.g. using binary search)
        let mut props_builder = if sorted_ts {
            match full_schema.index_of("common_ts") {
                Ok(idx) => props_builder.set_sorting_columns(Some(vec![SortingColumn {
                    column_idx: idx.try_into().unwrap(),
                    descending: false,
                    nulls_first: false,
                }])),
                Err(_) => props_builder,
            }
        } else {
            props_builder
        };

        // Set dictionary encoding for all string fields, as the text data logged in ftrace is
        // typically _very_ repetitive (e.g. task name, cgroup name etc).
        for field in &full_schema.fields {
            if field.data_type() == &DataType::Utf8 {
                props_builder = props_builder.set_column_dictionary_enabled(
                    ColumnPath::new(vec![field.name().to_string()]),
                    true,
                );
            }
        }

        // TODO: figure out if it's worth setting set_data_page_size_limit()
        let options = ArrowWriterOptions::new().with_properties(props_builder.build());

        let state = TableState::new(
            full_schema,
            fields_schema,
            options,
            chunk_size,
            name,
            scope,
            max_errors,
        )?;

        let variant = match event_desc.name.deref() {
            event_name @ "bprint" => {
                let struct_fmt = &event_desc.event_fmt()?.struct_fmt()?;

                macro_rules! field_fmt {
                    ($struct_fmt:expr, $name:expr) => {{
                        let field_name = $name;
                        $struct_fmt.field_by_name(field_name).ok_or_else(|| {
                            MainError::MissingField.with_field(Some(event_name), field_name)
                        })
                    }};
                }

                let fmt_fmt = field_fmt!(struct_fmt, "fmt")?;
                let buf_fmt = field_fmt!(struct_fmt, "buf")?;
                let common_pid_fmt = field_fmt!(struct_fmt, "common_pid")?;

                ReadStateVariant::BPrint {
                    fmt_fmt: fmt_fmt.clone(),
                    buf_fmt: buf_fmt.clone(),
                    common_pid_fmt: common_pid_fmt.clone(),
                    generic: state,
                    meta_events_by_addr: BTreeMap::new(),
                    meta_events_by_fmt: BTreeMap::new(),
                }
            }
            _ => ReadStateVariant::Generic(state),
        };
        Ok(ReadState {
            variant,
            scope,
            chunk_size,
            max_errors,
        })
    }

    fn process_fields<'ret, 'i, 'h, 'edm, InitDescF, Ctx, F>(
        &'ret mut self,
        visitor: &'ret EventVisitor<'i, 'h, 'edm, InitDescF, Ctx>,
        mut f: F,
        only_events: &Option<Vec<String>>,
    ) -> Result<
        impl DerefMut<Target = TableState<'scope>> + 'ret + use<'scope, 'ret, InitDescF, Ctx, F>,
        MainError,
    >
    where
        'i: 'ret,
        'h: 'ret,
        'scope: 'ret,
        InitDescF: 'h + FnMut(&'h Header, &'h EventDesc) -> Ctx,
        F: FnMut(&str, &mut FieldArray, Result<Value<'_>, BufferError>) -> Result<(), MainError>,
    {
        enum DerefMutWrapper<'a, T> {
            RefMut(&'a mut T),
            RcRefMut(RefMut<'a, T>),
        }

        impl<T> Deref for DerefMutWrapper<'_, T> {
            type Target = T;
            fn deref(&self) -> &T {
                match self {
                    DerefMutWrapper::RefMut(x) => x,
                    DerefMutWrapper::RcRefMut(x) => x.deref(),
                }
            }
        }

        impl<T> DerefMut for DerefMutWrapper<'_, T> {
            fn deref_mut(&mut self) -> &mut T {
                match self {
                    DerefMutWrapper::RefMut(x) => x,
                    DerefMutWrapper::RcRefMut(x) => x.deref_mut(),
                }
            }
        }

        let mut handle_error =
            |visitor: &EventVisitor<'i, 'h, 'edm, InitDescF, Ctx>, name, col: &mut _, val| {
                let res = f(name, col, val);
                match res {
                    Err(err) => {
                        col.append_null();
                        Err(err.with_field(visitor.event_name().ok(), name))
                    }
                    _ => Ok(()),
                }
            };

        macro_rules! generic_iter {
            ($table_state:expr, $visitor:expr) => {{
                let table_state = $table_state;
                let visitor = $visitor;

                let field_cols = table_state.field_cols.iter_mut();
                // We want to go through all the columns so that we have a chance to append None
                // values in places we had an error, and when we are done we return the last error.
                // This way, all columns should have the same length and we will still be able to
                // dump to parquet.

                table_state.errors.extend_errors(
                    visitor
                        .fields()?
                        .into_iter()
                        .zip(field_cols)
                        .map(|((fmt, val), col)| {
                            handle_error(visitor, fmt.declaration.identifier.deref(), col, val)
                        }),
                );
                Ok(DerefMutWrapper::RefMut(table_state))
            }};
        }

        macro_rules! bprint_meta_iter {
            ($meta_event_entry:expr, $visitor:expr, $buf_fmt:expr, $common_pid_fmt:expr) => {{
                let visitor = $visitor;
                let buf_fmt = $buf_fmt;
                let common_pid_fmt = $common_pid_fmt;

                let buf = visitor.field_by_fmt(buf_fmt)?;

                match buf {
                    Value::U32Array(array) => {
                        let (table_state, print_fmt) = $meta_event_entry;
                        let mut table_state: RefMut<'ret, _> = RefCell::borrow_mut(table_state);
                        let mut _table_state = table_state.deref_mut();

                        let pid = visitor.field_by_fmt(common_pid_fmt)?;

                        _table_state.errors.extend_errors(
                            print_fmt
                                .vbin_fields(visitor.header, visitor.scratch(), &array)
                                .into_iter()
                                .chain([Ok(PrintArg {
                                    value: pid,
                                    width: None,
                                    precision: None,
                                })])
                                .zip(_table_state.field_cols.iter_mut())
                                .map(|(res, col)| {
                                    handle_error(
                                        visitor,
                                        &_table_state.name,
                                        col,
                                        res.map(|print_arg| print_arg.value),
                                    )
                                }),
                        );

                        Ok(DerefMutWrapper::RcRefMut(table_state))
                    }
                    val => Err(MainError::EvalError(EvalError::IllegalType(
                        val.into_static().ok(),
                    ))),
                }
            }};
        }

        match &mut self.variant {
            ReadStateVariant::Generic(state) => generic_iter!(state, visitor),
            ReadStateVariant::BPrint {
                generic,
                fmt_fmt,
                buf_fmt,
                common_pid_fmt,
                meta_events_by_addr,
                meta_events_by_fmt,
                ..
            } => {
                let fmt = visitor.field_by_fmt(fmt_fmt)?;
                let addr = match fmt {
                    Value::U64Scalar(addr) => Ok(addr),
                    Value::I64Scalar(addr) => Ok(addr as u64),
                    _ => Err(EvalError::CannotDeref(0)),
                }?;

                macro_rules! handle {
                    ($res:expr) => {{
                        match Rc::as_ref($res) {
                            EventCtx::Selected(Ok(entry)) => {
                                bprint_meta_iter!(entry, visitor, buf_fmt, common_pid_fmt)
                            }
                            _ => generic_iter!(generic, visitor),
                        }
                    }};
                }
                match meta_events_by_addr.entry(addr) {
                    // We have a recorded attempt to treat it as a meta event that did not succeed,
                    // so we treat it like a regular bprint text event.
                    Entry::Occupied(entry) => handle!(entry.into_mut()),
                    Entry::Vacant(entry) => {
                        let header = visitor.header;
                        let env = visitor.buffer_env();

                        let parse_print_fmt = || -> Result<PrintFmtStr, MainError> {
                            let print_fmt = env.deref_static(addr)?;
                            let print_fmt = match print_fmt.to_str() {
                                Some(s) => Ok(s),
                                None => Err(EvalError::IllegalType(print_fmt.into_static().ok())),
                            }?;
                            Ok(PrintFmtStr::try_new(header, print_fmt.as_bytes())?)
                        };

                        let make_schema =
                            |print_fmt: PrintFmtStr| match Self::make_print_fmt_schemas(
                                header, &print_fmt,
                            ) {
                                Ok((meta_event_name, full_schema, fields_schema)) => {
                                    let meta_event_name = format!("trace_printk@{meta_event_name}");

                                    let select = match only_events {
                                        None => true,
                                        Some(only_events) => only_events.iter().any(|selected| {
                                            meta_event_name.deref() == selected.deref()
                                        }),
                                    };

                                    if select {
                                        match TableState::new(
                                            full_schema,
                                            fields_schema,
                                            // FIXME: the key/value metadata will have the source
                                            // event name "bprint" instead of the meta event name
                                            generic.options.clone(),
                                            self.chunk_size,
                                            &meta_event_name,
                                            self.scope,
                                            self.max_errors,
                                        ) {
                                            Ok(state) => EventCtx::Selected(Ok((
                                                RefCell::new(state),
                                                print_fmt,
                                            ))),
                                            Err(err) => EventCtx::Selected(Err(err)),
                                        }
                                    } else {
                                        EventCtx::NotSelected
                                    }
                                }
                                Err(_) => EventCtx::NotSelected,
                            };

                        let new = match parse_print_fmt() {
                            Ok(print_fmt) => {
                                // Find an already-created meta event that would have the same
                                // print format string, therefore the same schema.
                                match meta_events_by_fmt.entry(print_fmt) {
                                    Entry::Occupied(entry) => Rc::clone(entry.get()),
                                    Entry::Vacant(entry) => {
                                        let new = Rc::new(make_schema(entry.key().clone()));
                                        entry.insert(Rc::clone(&new));
                                        new
                                    }
                                }
                            }
                            Err(_) => Rc::new(EventCtx::Selected(Err(MainError::NotAMetaEvent))),
                        };

                        handle!(entry.insert(new))
                    }
                }
            }
        }
    }

    fn drain_states(self) -> impl Iterator<Item = TableState<'scope>> + use<'scope> {
        match self.variant {
            ReadStateVariant::Generic(state) => {
                Box::new([state].into_iter()) as Box<dyn Iterator<Item = _>>
            }
            ReadStateVariant::BPrint {
                generic,
                mut meta_events_by_fmt,
                meta_events_by_addr,
                ..
            } => {
                // Ensure we kill all the Rc that could be pointing at the meta event entries
                // before trying to unwrap the Rc
                drop(meta_events_by_addr);
                Box::new([generic].into_iter().chain(
                    std::iter::from_fn(move || meta_events_by_fmt.pop_first()).filter_map(
                        |(_, entry)| match Rc::into_inner(entry).unwrap() {
                            EventCtx::Selected(entry) => {
                                let (table_state, _) = entry.ok()?;
                                let table_state = RefCell::into_inner(table_state);
                                Some(table_state)
                            }
                            _ => None,
                        },
                    ),
                )) as Box<dyn Iterator<Item = _>>
            }
        }
    }

    fn make_event_desc_schemas(
        header: &Header,
        event_desc: &EventDesc,
    ) -> Result<(Schema, Schema), MainError> {
        let struct_fmt = &event_desc.event_fmt()?.struct_fmt()?;
        let fields = &struct_fmt.fields;
        Self::make_schemas(
            &event_desc.name,
            header,
            fields.iter().map(|fmt| {
                (
                    fmt.declaration.identifier.to_string(),
                    fmt.declaration.typ.clone(),
                )
            }),
        )
    }

    fn make_print_fmt_schemas(
        header: &Header,
        fmt: &PrintFmtStr,
    ) -> Result<(String, Schema, Schema), MainError> {
        let field_name_parser = || {
            nom::sequence::preceded(
                nom::multi::many0(nom::character::complete::char(' ')),
                nom::sequence::terminated(identifier(), nom::character::complete::char('=')),
            )
        };

        let mut event_name = None;
        let mut field_name = None;

        let fields = fmt.atoms.iter().enumerate().filter_map(|(i, atom)| {
            if i == 0 {
                match atom {
                    PrintAtom::Fixed(fixed) => {
                        let res = nom::combinator::all_consuming(nom::sequence::separated_pair(
                            identifier(),
                            nom::character::complete::char(':'),
                            field_name_parser(),
                        ))
                        .parse(fixed.as_bytes())
                        .finish();
                        match res {
                            Ok((_, (_event_name, _field_name))) => {
                                field_name = Some(_field_name);
                                event_name = Some(_event_name);
                                None
                            }
                            Err(()) => Some(Err(MainError::NotAMetaEvent)),
                        }
                    }
                    _ => Some(Err(MainError::NotAMetaEvent)),
                }
            } else {
                match atom {
                    PrintAtom::Fixed(fixed) => {
                        let fixed = fixed.as_bytes();
                        match nom::combinator::all_consuming(nom::character::complete::multispace1)
                            .parse(fixed)
                            .finish()
                        {
                            Ok(_) => None,
                            Err(()) => match nom::combinator::all_consuming(field_name_parser())
                                .parse(fixed)
                                .finish()
                            {
                                Err(()) => Some(Err(MainError::NotAMetaEvent)),
                                Ok((_, name)) => {
                                    field_name = Some(name);
                                    None
                                }
                            },
                        }
                    }
                    PrintAtom::Variable { vbin_spec, .. } => {
                        let typ = match vbin_spec {
                            VBinSpecifier::U8 => Type::U8,
                            VBinSpecifier::I8 => Type::I8,

                            VBinSpecifier::U16 => Type::U16,
                            VBinSpecifier::I16 => Type::I16,

                            VBinSpecifier::U32 => Type::U32,
                            VBinSpecifier::I32 => Type::I32,

                            VBinSpecifier::U64 => Type::U64,
                            VBinSpecifier::I64 => Type::I64,

                            VBinSpecifier::Str => Type::Array(
                                Box::new(header.kernel_abi().char_typ()),
                                ArrayKind::ZeroLength,
                            ),
                        };
                        Some(match &field_name {
                            None => Err(MainError::NotAMetaEvent),
                            Some(name) => Ok((name.deref().into(), typ)),
                        })
                    }
                }
            }
        });
        let fields: Result<Vec<_>, MainError> = fields.collect();
        let mut fields = fields?;
        fields.push(("common_pid".into(), Type::I32));
        let event_name = event_name.ok_or(MainError::NotAMetaEvent)?;
        let (full_schema, fields_schema) = Self::make_schemas(&event_name, header, fields)?;
        Ok((event_name, full_schema, fields_schema))
    }

    fn make_schemas<FieldsIterator>(
        event_name: &str,
        header: &Header,
        fields: FieldsIterator,
    ) -> Result<(Schema, Schema), MainError>
    where
        FieldsIterator: IntoIterator<Item = (String, Type)>,
    {
        let char_typ = header.kernel_abi().char_typ();
        let long_size = header.kernel_abi().long_size;

        let field_cols = fields.into_iter().map(|(name, typ)| {
            fn guess_typ(
                typ: &Type,
                char_typ: &Type,
                long_size: &LongSize,
            ) -> Result<DataType, MainError> {
                let recurse = |typ| guess_typ(typ, char_typ, long_size);
                match typ {
                    Type::Bool => Ok(DataType::Boolean),
                    Type::U8 => Ok(DataType::UInt8),
                    Type::U16 => Ok(DataType::UInt16),
                    Type::U32 => Ok(DataType::UInt32),
                    Type::U64 => Ok(DataType::UInt64),
                    Type::I8 => Ok(DataType::Int8),
                    Type::I16 => Ok(DataType::Int16),
                    Type::I32 => Ok(DataType::Int32),
                    Type::I64 => Ok(DataType::Int64),

                    // char [] are considered as strings
                    Type::Array(inner, _) | Type::Pointer(inner) if &**inner == char_typ => {
                        Ok(DataType::Utf8)
                    }

                    // u8 [] are considered as byte buffer
                    Type::Array(inner, _) | Type::Pointer(inner)
                        if matches!(
                        &**inner, Type::Typedef(_, name) if name == "u8") =>
                    {
                        Ok(DataType::Binary)
                    }

                    Type::Array(inner, _) | Type::Pointer(inner)
                        if matches!(
                            inner.resolve_wrapper(),
                            Type::Bool
                                | Type::U8
                                | Type::I8
                                | Type::U16
                                | Type::I16
                                | Type::U32
                                | Type::I32
                                | Type::U64
                                | Type::I64
                        ) =>
                    {
                        Ok(DataType::new_list(recurse(inner)?, true))
                    }

                    Type::Pointer(..) => match long_size {
                        LongSize::Bits32 => Ok(DataType::UInt32),
                        LongSize::Bits64 => Ok(DataType::UInt64),
                    },

                    Type::Typedef(_, id) if id.deref() == "cpumask_t" => {
                        Ok(DataType::new_list(DataType::Boolean, true))
                    }

                    // TODO: try to do symbolic resolution of enums somehow, maybe with BTF
                    // Do we want that always ? What about conversion from other formats where the
                    // enum is not available ? Maybe that should be left to a Python function,
                    // hooked with the BTF parser, and BTF available in platform info.
                    Type::Typedef(typ, _) | Type::Enum(typ, _) | Type::DynamicScalar(typ, _) => {
                        recurse(typ)
                    }

                    typ => Err(MainError::TypeNotHandled(Box::new(typ.clone()))),
                }
            }
            let typ = guess_typ(&typ, &char_typ, &long_size)
                .map_err(|err| err.with_field(Some(event_name), &name))?;
            Ok(Field::new(name, typ, true))
        });
        let field_cols: Result<Vec<_>, MainError> = field_cols.collect();
        let field_cols = field_cols?;

        let fields_schema = Schema::new(Fields::from(field_cols.clone()));
        let full_schema = Schema::new(Fields::from(
            FixedCols::arrow_fields()
                .chain(field_cols)
                .collect::<Vec<_>>(),
        ));
        Ok((full_schema, fields_schema))
    }
}

#[derive(Debug)]
struct TableState<'scope> {
    name: String,
    path: PathBuf,
    chunk_size: usize,
    fields_schema: Schema,
    fixed_cols: FixedCols,
    field_cols: Vec<FieldArray>,
    nr_rows: u64,

    sender: Sender<RecordBatch>,
    handle: ScopedJoinHandle<'scope, Result<(), MainError>>,
    errors: TableErrors,

    full_schema: Arc<Schema>,
    options: ArrowWriterOptions,
}

impl<'scope> TableState<'scope> {
    fn new(
        full_schema: Schema,
        fields_schema: Schema,
        options: ArrowWriterOptions,
        chunk_size: usize,
        name: &str,
        scope: &'scope Scope,
        max_errors: usize,
    ) -> Result<Self, MainError> {
        let full_schema = Arc::new(full_schema);
        let (fixed_cols, field_cols) = Self::make_cols(name, &fields_schema, chunk_size)?;

        let path = PathBuf::from(format!("{name}.parquet"));
        let file = File::create(&path)?;
        let (sender, receiver) = bounded(128);

        let mut writer =
            ArrowWriter::try_new_with_options(file, full_schema.clone(), options.clone())?;
        let write_thread = move |_: &_| -> Result<(), MainError> {
            for batch in receiver.iter() {
                writer.write(&batch)?;
            }
            writer.close()?;
            Ok(())
        };

        let handle = scope.spawn(write_thread);

        Ok(TableState {
            field_cols,
            fixed_cols,
            fields_schema,
            sender,
            name: name.to_string(),
            handle,
            path,
            errors: TableErrors::new(max_errors),
            nr_rows: 0,
            chunk_size,

            full_schema,
            options,
        })
    }

    fn make_cols(
        name: &str,
        schema: &Schema,
        chunk_size: usize,
    ) -> Result<(FixedCols, Vec<FieldArray>), MainError> {
        let make_col = |field: &Field| match &field.data_type() {
            DataType::Int8 => Ok(FieldArray::I8(Int8Builder::with_capacity(chunk_size))),
            DataType::Int16 => Ok(FieldArray::I16(Int16Builder::with_capacity(chunk_size))),
            DataType::Int32 => Ok(FieldArray::I32(Int32Builder::with_capacity(chunk_size))),
            DataType::Int64 => Ok(FieldArray::I64(Int64Builder::with_capacity(chunk_size))),

            DataType::UInt8 => Ok(FieldArray::U8(UInt8Builder::with_capacity(chunk_size))),
            DataType::UInt16 => Ok(FieldArray::U16(UInt16Builder::with_capacity(chunk_size))),
            DataType::UInt32 => Ok(FieldArray::U32(UInt32Builder::with_capacity(chunk_size))),
            DataType::UInt64 => Ok(FieldArray::U64(UInt64Builder::with_capacity(chunk_size))),

            DataType::Boolean => Ok(FieldArray::Bool(BooleanBuilder::with_capacity(chunk_size))),
            DataType::Utf8 => Ok(FieldArray::Str(StringBuilder::with_capacity(
                chunk_size,
                chunk_size * 16,
            ))),
            DataType::Binary => Ok(FieldArray::Binary(BinaryBuilder::with_capacity(
                chunk_size,
                chunk_size * 16,
            ))),

            DataType::List(field) if field.data_type() == &DataType::Boolean => Ok(
                FieldArray::ListBool(ListBuilder::new(BooleanBuilder::with_capacity(chunk_size))),
            ),

            DataType::List(field) if field.data_type() == &DataType::UInt8 => Ok(
                FieldArray::ListU8(ListBuilder::new(UInt8Builder::with_capacity(chunk_size))),
            ),
            DataType::List(field) if field.data_type() == &DataType::UInt16 => Ok(
                FieldArray::ListU16(ListBuilder::new(UInt16Builder::with_capacity(chunk_size))),
            ),
            DataType::List(field) if field.data_type() == &DataType::UInt32 => Ok(
                FieldArray::ListU32(ListBuilder::new(UInt32Builder::with_capacity(chunk_size))),
            ),
            DataType::List(field) if field.data_type() == &DataType::UInt64 => Ok(
                FieldArray::ListU64(ListBuilder::new(UInt64Builder::with_capacity(chunk_size))),
            ),

            DataType::List(field) if field.data_type() == &DataType::Int8 => Ok(
                FieldArray::ListI8(ListBuilder::new(Int8Builder::with_capacity(chunk_size))),
            ),
            DataType::List(field) if field.data_type() == &DataType::Int16 => Ok(
                FieldArray::ListI16(ListBuilder::new(Int16Builder::with_capacity(chunk_size))),
            ),
            DataType::List(field) if field.data_type() == &DataType::Int32 => Ok(
                FieldArray::ListI32(ListBuilder::new(Int32Builder::with_capacity(chunk_size))),
            ),
            DataType::List(field) if field.data_type() == &DataType::Int64 => Ok(
                FieldArray::ListI64(ListBuilder::new(Int64Builder::with_capacity(chunk_size))),
            ),

            typ => Err(MainError::ArrowDataTypeNotHandled(Box::new((*typ).clone()))),
        };

        let fields: Result<Vec<_>, MainError> = schema
            .fields
            .iter()
            .map(|field| make_col(field).map_err(|err| err.with_field(Some(name), field.name())))
            .collect();
        let fields = fields?;

        let fixed = FixedCols::new(chunk_size);
        Ok((fixed, fields))
    }

    fn extract_batch(&mut self) -> Result<RecordBatch, MainError> {
        let (mut fixed_cols, mut field_cols) =
            Self::make_cols(&self.name, &self.fields_schema, self.chunk_size)?;

        assert_eq!(field_cols.len(), self.field_cols.len());
        core::mem::swap(&mut self.field_cols, &mut field_cols);
        core::mem::swap(&mut self.fixed_cols, &mut fixed_cols);

        Ok(RecordBatch::try_new(
            Arc::clone(&self.full_schema),
            fixed_cols
                .into_finished()
                .chain(field_cols.into_iter().map(|col| col.finish()))
                .collect(),
        )?)
    }
}

#[derive(Debug)]
struct TableErrors {
    errors: Vec<MainError>,
    max_errors: usize,
}

fn limited_append<T>(vec: &mut Vec<T>, x: T, max_errors: usize) {
    if vec.len() <= max_errors {
        vec.push(x);
    }
}

impl TableErrors {
    fn new(max_errors: usize) -> Self {
        TableErrors {
            errors: Vec::new(),
            max_errors,
        }
    }
    fn extend_errors<I: IntoIterator<Item = Result<(), MainError>>>(&mut self, iter: I) {
        for res in iter.into_iter() {
            if let Err(err) = res {
                limited_append(&mut self.errors, err, self.max_errors);
            }
        }
    }
}
