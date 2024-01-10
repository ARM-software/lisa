use core::{
    cell::{Cell, RefCell, RefMut},
    convert::identity,
    ops::{Deref, DerefMut},
};
use std::{
    collections::{btree_map::Entry, BTreeMap},
    error::Error,
    fs::File,
    io::{Error as IoError, Write as _},
    rc::Rc,
    sync::Arc,
    path::PathBuf,
};

use arrow2::{
    array::{
        Array, MutableArray, MutableBinaryArray, MutableBooleanArray, MutableListArray,
        MutablePrimitiveArray, MutableUtf8Array, TryPush as _,
    },
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    error::Error as ArrowError,
    io::parquet::write::{
        CompressionOptions, Encoding, FileWriter, RowGroupIterator, Version, WriteOptions,
    },
};
// use polars_error::PolarsError as ArrowError;
use crossbeam::{
    channel::{bounded, Sender},
    thread::{scope, Scope, ScopedJoinHandle},
};
use nom::{Finish as _, Parser as _};
use traceevent::{
    self,
    buffer::{BufferError, EventVisitor},
    cinterp::{EvalEnv, EvalError, Value},
    cparser::{identifier, ArrayKind, Type},
    header::{
        Address, EventDesc, EventId, FieldFmt, Header, HeaderError, LongSize, Timestamp, CPU,
    },
    io::BorrowingRead,
    print::{PrintAtom, PrintFmtError, PrintFmtStr, VBinSpecifier},
};

use crate::convert_err_impl;

// This size is a sweet spot. If in doubt, it's best to have chunks that are too big than too
// small, as smaller chunks can wreak performances and might also mean more work when consuming the
// file. In my experiments, 16 * 1024 was a transition point between good and horrible performance.
// Note that this chunk size is expressed in terms of number of rows, independently from the size
// of the rows themselves.
const CHUNK_SIZE: usize = 64 * 1024;

type ArrayChunk = Chunk<Arc<dyn Array>>;

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

    #[error("Error while interpreting event data: {0}")]
    EvalError(EvalError),

    #[error("Error while parsing print format string: {0}")]
    PrintFmtError(PrintFmtError),

    #[error("Arrow error: {0}")]
    ArrowError(ArrowError),

    #[error("Field type not handled: {0:?}")]
    TypeNotHandled(Box<Type>),

    #[error("Arrow data type not handled: {0:?}")]
    ArrowDataTypeNotHandled(Box<DataType>),

    #[error("Runtime data cannot be used according to the column storage schema")]
    DataMismatchingSchema,

    #[error("Expected field {field_name} to be available in event {event_name}")]
    MissingField {
        event_name: String,
        field_name: String,
    },

    #[error("Expected an integer value")]
    NotAnInteger,

    #[error("This print format string does not describe a meta event")]
    NotAMetaEvent,
}

convert_err_impl!(IoError, MainError);
convert_err_impl!(HeaderError, MainError);
convert_err_impl!(BufferError, MainError);
convert_err_impl!(ArrowError, MainError);
convert_err_impl!(EvalError, MainError);
convert_err_impl!(PrintFmtError, MainError);

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
    reader: R,
    mut modify_timestamps: FTimestamp,
) -> Result<(), Box<dyn Error>>
where
    FTimestamp: FnMut(Timestamp) -> Timestamp + Send,
    R: BorrowingRead + Send,
{
    let options = WriteOptions {
        write_statistics: true,
        compression: CompressionOptions::Zstd(None),
        version: Version::V2,
        data_pagesize_limit: None,
    };

    // TODO: EventId might not be enough if we extend the API to deal with buffers from multiple
    // traces
    //
    // Keep the per-event state in a map that is shared between all buffers. Otherwise, we would
    // end up with a state per-event and per-buffer, which is not what we want.
    type StateMap<'scope, 'scopeenv> = BTreeMap<EventId, SharedState<'scope, 'scopeenv>>;

    macro_rules! chunk_append {
        ($($arms:expr),*) => {
            loop {
                $(
                    $arms;
                )*

                break Err(MainError::DataMismatchingSchema);
            }
        }
    }

    macro_rules! make_macros {
        ($scrutinee:expr) => {
            let scrutinee = $scrutinee;

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
                                xs.push(Some($f(x)));
                                break Ok(xs.len());
                            }
                            _ => break Err(MainError::DataMismatchingSchema),
                        }
                    }
                };
            }
        };
    }

    scope(move |scope| -> Result<_, MainError> {
        eprintln!("SPAWNED");

        let mut count: u64 = 0;
        let _count = &mut count;
        let buffers = header.buffers(reader)?;
        let mut state_map = StateMap::new();
        let mut window = (None, None);

        let events = traceevent::buffer::flyrecord(
            buffers,
            {
                let window = &mut window;
                move |res| -> Result<(), MainError> {
                    match res {
                        Ok(visitor) => {
                            *_count += 1;
                            let state: &SharedState = visitor.event_user_data()?;
                            let mut state = state.borrow_mut();
                            let state = state.deref_mut();
                            match state {
                                // We ignore the error here as it will be reported at the end when finalizing
                                // the files.
                                Err(_) => Ok(()),
                                Ok(state) => {
                                    // This needs to happen regardless of whether the event is selected, otherwise the
                                    // resulting timestamps would vary based on the set of events selected, making caching of
                                    // the parquet files problematic.
                                    let ts = modify_timestamps(visitor.timestamp);
                                    *window = match window {
                                        (None, _) => (Some(ts), None),
                                        (first@Some(_), _) => (*first, Some(ts)),
                                    };

                                    let buf_id = visitor.buffer_id;
                                    let event_desc = visitor.event_desc()?;

                                    let cpu = buf_id.cpu;
                                    let len: Cell<usize> = Cell::new(0);

                                    let (mut table_state, res) = state.with_fields(
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
                                                integer!(FieldArray::I8,  Value::I64Scalar, cast!(i64, i8)),
                                                integer!(FieldArray::I16, Value::I64Scalar, cast!(i64, i16)),
                                                integer!(FieldArray::I32, Value::I64Scalar, cast!(i64, i32)),
                                                integer!(FieldArray::I64, Value::I64Scalar, identity),

                                                integer!(FieldArray::U8,  Value::U64Scalar, cast!(u64, u8)),
                                                integer!(FieldArray::U16, Value::U64Scalar, cast!(u64, u16)),
                                                integer!(FieldArray::U32, Value::U64Scalar, cast!(u64, u32)),
                                                integer!(FieldArray::U64, Value::U64Scalar, identity),

                                                basic!((FieldArray::Str(xs), x) => {
                                                    xs.push(x.deref_ptr(&visitor.buffer_env())?.to_str());
                                                    xs
                                                }),

                                                basic!((FieldArray::Bool(xs), Value::I64Scalar(x)) => {
                                                    xs.push(Some(x != 0));
                                                    xs
                                                }),
                                                basic!((FieldArray::Bool(xs), Value::U64Scalar(x)) => {
                                                    xs.push(Some(x != 0));
                                                    xs
                                                }),
                                                // Expected Binary
                                                basic!((FieldArray::Binary(xs), Value::U8Array(x)) => {
                                                    xs.push(Some(x));
                                                    xs
                                                }),
                                                basic!((FieldArray::Binary(xs), Value::I8Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                }),


                                                // Lists
                                                basic!((FieldArray::ListU8(xs), Value::U8Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),
                                                basic!((FieldArray::ListI8(xs), Value::I8Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),

                                                basic!((FieldArray::ListU16(xs), Value::U16Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),
                                                basic!((FieldArray::ListI16(xs), Value::I16Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),

                                                basic!((FieldArray::ListU32(xs), Value::U32Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),
                                                basic!((FieldArray::ListI32(xs), Value::I32Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),

                                                basic!((FieldArray::ListU64(xs), Value::U64Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),
                                                basic!((FieldArray::ListI64(xs), Value::I64Array(x)) => {
                                                    xs.try_push(Some(x.iter().copied().map(Some)))?;
                                                    xs
                                                }),

                                                // Unexpected Binary
                                                basic!((FieldArray::Binary(xs), Value::U16Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                }),
                                                basic!((FieldArray::Binary(xs), Value::I16Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                }),
                                                basic!((FieldArray::Binary(xs), Value::U32Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                }),
                                                basic!((FieldArray::Binary(xs), Value::I32Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                }),
                                                basic!((FieldArray::Binary(xs), Value::U64Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                }),
                                                basic!((FieldArray::Binary(xs), Value::I64Array(x)) => {
                                                    xs.push(Some(bytemuck::cast_slice(&x)));
                                                    xs
                                                })
                                            }?);
                                            Ok(())
                                        }
                                    )?;

                                    table_state.fixed_cols.time.push(Some(ts));
                                    table_state.fixed_cols.cpu.push(Some(cpu));

                                    if len.get() >= CHUNK_SIZE {
                                        let chunk = table_state.extract_chunk()?;
                                        table_state.sender.send(chunk).unwrap();
                                    }
                                    res
                                }
                            }
                        }
                        Err(err) => Err(err.into()),
                    }
                }
            },

            |header, event_desc: &EventDesc| {
                let id = event_desc.id;
                match state_map.entry(id) {
                    Entry::Vacant(entry) => {
                        let state = {
                            ReadState::new(header, event_desc, options, &event_desc.name, scope)
                        };

                        let state = SharedState::new(state);
                        entry.insert(state.clone());
                        state
                    }
                    Entry::Occupied(entry) => entry.get().clone(),
                }
            }
        )?;

        // Even if there were some errors, we should have pushed some None values so that all
        // columns are of the same length, so the file can be finalized.
        let res = last_err(events.into_iter());
        eprintln!("COUNT {count}");

        let mut handles = Vec::new();
        let mut events_path = Vec::new();

        while let Some((id, read_state)) = state_map.pop_first() {
            let read_state = read_state.into_inner().unwrap();

            match read_state {
                Ok(read_state) => {
                    for mut read_state in read_state.drain_states() {
                        let chunk = read_state.extract_chunk()?;
                        read_state.sender.send(chunk).unwrap();
                        // Drop the sender which will close the channel so that the writer thread will
                        // know it's time to finish.
                        drop(read_state.sender);
                        handles.push(read_state.handle);
                        eprintln!("FILE WRITTEN SUCCESSFULLY {}", read_state.name);
                        events_path.push(serde_json::json!({
                            "event": read_state.name,
                            "path": read_state.path,
                            "format": "parquet",
                        }))
                    }
                }
                Err(err) => {
                    let event_desc = header.event_desc_by_id(id);
                    eprintln!(
                        "ERROR WHEN CREATING FILE FOR EVENT ID {} ({:?}): {err}",
                        id,
                        event_desc.map(|desc| &desc.name)
                    );
                }
            }
        }

        for handle in handles {
            handle.join().expect("Writer thread panicked")?;
        }

        {
            let json_value = serde_json::json!({
                "events": events_path,
                "cpus-count": header.nr_cpus(),
                "time-range": window,
                "symbols-address": header.kallsyms().into_iter().collect::<Vec<_>>(),
            });

            let mut meta = File::create("meta.json")?;
            meta.write_all(json_value.to_string().as_bytes())?
        }
        res
    })
    .unwrap()?;
    Ok(())
}

#[derive(Debug)]
enum FieldArray {
    U8(MutablePrimitiveArray<u8>),
    U16(MutablePrimitiveArray<u16>),
    U32(MutablePrimitiveArray<u32>),
    U64(MutablePrimitiveArray<u64>),

    I8(MutablePrimitiveArray<i8>),
    I16(MutablePrimitiveArray<i16>),
    I32(MutablePrimitiveArray<i32>),
    I64(MutablePrimitiveArray<i64>),

    Bool(MutableBooleanArray),
    // Using i32 means strings and binary blobs have to be smaller than 2GB, which should be fine
    Binary(MutableBinaryArray<i32>),
    Str(MutableUtf8Array<i32>),

    ListU8(MutableListArray<i32, MutablePrimitiveArray<u8>>),
    ListU16(MutableListArray<i32, MutablePrimitiveArray<u16>>),
    ListU32(MutableListArray<i32, MutablePrimitiveArray<u32>>),
    ListU64(MutableListArray<i32, MutablePrimitiveArray<u64>>),

    ListI8(MutableListArray<i32, MutablePrimitiveArray<i8>>),
    ListI16(MutableListArray<i32, MutablePrimitiveArray<i16>>),
    ListI32(MutableListArray<i32, MutablePrimitiveArray<i32>>),
    ListI64(MutableListArray<i32, MutablePrimitiveArray<i64>>),
}

impl FieldArray {
    fn into_arc(self) -> Arc<dyn Array> {
        match self {
            FieldArray::U8(xs) => xs.into_arc(),
            FieldArray::U16(xs) => xs.into_arc(),
            FieldArray::U32(xs) => xs.into_arc(),
            FieldArray::U64(xs) => xs.into_arc(),

            FieldArray::I8(xs) => xs.into_arc(),
            FieldArray::I16(xs) => xs.into_arc(),
            FieldArray::I32(xs) => xs.into_arc(),
            FieldArray::I64(xs) => xs.into_arc(),

            FieldArray::Bool(xs) => xs.into_arc(),
            FieldArray::Str(xs) => xs.into_arc(),
            FieldArray::Binary(xs) => xs.into_arc(),

            FieldArray::ListU8(xs) => xs.into_arc(),
            FieldArray::ListU16(xs) => xs.into_arc(),
            FieldArray::ListU32(xs) => xs.into_arc(),
            FieldArray::ListU64(xs) => xs.into_arc(),

            FieldArray::ListI8(xs) => xs.into_arc(),
            FieldArray::ListI16(xs) => xs.into_arc(),
            FieldArray::ListI32(xs) => xs.into_arc(),
            FieldArray::ListI64(xs) => xs.into_arc(),
        }
    }

    fn push_null(&mut self) {
        match self {
            FieldArray::U8(xs) => xs.push_null(),
            FieldArray::U16(xs) => xs.push_null(),
            FieldArray::U32(xs) => xs.push_null(),
            FieldArray::U64(xs) => xs.push_null(),

            FieldArray::I8(xs) => xs.push_null(),
            FieldArray::I16(xs) => xs.push_null(),
            FieldArray::I32(xs) => xs.push_null(),
            FieldArray::I64(xs) => xs.push_null(),

            FieldArray::Bool(xs) => xs.push_null(),
            FieldArray::Str(xs) => xs.push_null(),
            FieldArray::Binary(xs) => xs.push_null(),

            FieldArray::ListU8(xs) => xs.push_null(),
            FieldArray::ListU16(xs) => xs.push_null(),
            FieldArray::ListU32(xs) => xs.push_null(),
            FieldArray::ListU64(xs) => xs.push_null(),

            FieldArray::ListI8(xs) => xs.push_null(),
            FieldArray::ListI16(xs) => xs.push_null(),
            FieldArray::ListI32(xs) => xs.push_null(),
            FieldArray::ListI64(xs) => xs.push_null(),
        }
    }
}

#[derive(Debug)]
struct FixedCols {
    time: MutablePrimitiveArray<Timestamp>,
    cpu: MutablePrimitiveArray<CPU>,
}

impl FixedCols {
    fn new() -> Self {
        FixedCols {
            time: MutablePrimitiveArray::with_capacity(CHUNK_SIZE),
            cpu: MutablePrimitiveArray::with_capacity(CHUNK_SIZE),
        }
    }

    fn arrow_fields() -> impl Iterator<Item = Field> {
        [
            Field::new("common_ts", DataType::UInt64, false),
            Field::new("common_cpu", DataType::UInt32, false),
        ]
        .into_iter()
    }

    fn into_arcs(self) -> impl Iterator<Item = Arc<dyn Array>> {
        [self.time.into_arc(), self.cpu.into_arc()].into_iter()
    }
}

#[derive(Debug)]
struct ReadState<'scope, 'scopeenv> {
    variant: ReadStateVariant<'scope>,
    options: WriteOptions,
    scope: &'scope Scope<'scopeenv>,
}

type MetaEventEntry<'scope> = Rc<Result<(RefCell<TableState<'scope>>, PrintFmtStr), MainError>>;

#[derive(Debug)]
enum ReadStateVariant<'scope> {
    Generic(TableState<'scope>),
    BPrint {
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
        options: WriteOptions,
        name: &str,
        scope: &'scope Scope<'scopeenv>,
    ) -> Result<Self, MainError> {
        let (full_schema, fields_schema) = Self::make_event_desc_schemas(header, event_desc)?;
        let state = TableState::new(full_schema, fields_schema, options, name, scope)?;

        let variant = match event_desc.name.deref() {
            name @ "bprint" => {
                let struct_fmt = &event_desc.event_fmt()?.struct_fmt()?;

                let fmt_fmt = struct_fmt
                    .field_by_name("fmt")
                    .ok_or(MainError::MissingField {
                        event_name: name.into(),
                        field_name: "fmt".into(),
                    })?;

                let buf_fmt = struct_fmt
                    .field_by_name("buf")
                    .ok_or(MainError::MissingField {
                        event_name: name.into(),
                        field_name: "buf".into(),
                    })?;

                ReadStateVariant::BPrint {
                    fmt_fmt: fmt_fmt.clone(),
                    buf_fmt: buf_fmt.clone(),
                    generic: state,
                    meta_events_by_addr: BTreeMap::new(),
                    meta_events_by_fmt: BTreeMap::new(),
                }
            }
            _ => ReadStateVariant::Generic(state),
        };
        Ok(ReadState {
            variant,
            options,
            scope,
        })
    }

    fn with_fields<'ret, 'i, 'h, 'edm, InitDescF, T, F>(
        &'ret mut self,
        visitor: &'ret EventVisitor<'i, 'h, 'edm, InitDescF, T>,
        mut f: F,
    ) -> Result<
        (
            impl DerefMut<Target = TableState<'scope>> + 'ret,
            Result<(), MainError>,
        ),
        MainError,
    >
    where
        'i: 'ret,
        'h: 'ret,
        'scope: 'ret,
        InitDescF: 'h + FnMut(&'h Header, &'h EventDesc) -> T,
        F: FnMut(&str, &mut FieldArray, Result<Value<'_>, BufferError>) -> Result<(), MainError>,
    {
        enum DerefMutWrapper<'a, T> {
            RefMut(&'a mut T),
            RcRefMut(RefMut<'a, T>),
        }

        impl<'a, T> Deref for DerefMutWrapper<'a, T> {
            type Target = T;
            fn deref(&self) -> &T {
                match self {
                    DerefMutWrapper::RefMut(x) => x,
                    DerefMutWrapper::RcRefMut(x) => x.deref(),
                }
            }
        }

        impl<'a, T> DerefMut for DerefMutWrapper<'a, T> {
            fn deref_mut(&mut self) -> &mut T {
                match self {
                    DerefMutWrapper::RefMut(x) => x,
                    DerefMutWrapper::RcRefMut(x) => x.deref_mut(),
                }
            }
        }

        let mut handle_error = |name, col: &mut _, val| {
            let res = f(name, col, val);
            match res {
                Err(err) => {
                    col.push_null();
                    Err(err)
                }
                _ => Ok(()),
            }
        };

        macro_rules! generic_iter {
            ($table_state:expr, $visitor:expr) => {{
                let table_state = $table_state;
                let visitor = $visitor;

                // We want to go through all the columns so that we have a chance to append None
                // values in places we had an error, and when we are done we return the last error.
                // This way, all columns should have the same length and we will still be able to
                // dump to parquet.
                let res = last_err(
                    visitor
                        .fields()?
                        .into_iter()
                        .zip(table_state.field_cols.iter_mut())
                        .map(|((fmt, val), col)| {
                            handle_error(fmt.declaration.identifier.deref(), col, val)
                        }),
                );
                Ok((DerefMutWrapper::RefMut(table_state), res))
            }};
        }

        macro_rules! bprint_meta_iter {
            ($meta_event_entry:expr, $visitor:expr, $buf_fmt:expr) => {{
                let visitor = $visitor;
                let buf_fmt = $buf_fmt;

                let buf = visitor.field_by_fmt(buf_fmt)?;

                match buf {
                    Value::U32Array(array) => {
                        let (table_state, print_fmt) = $meta_event_entry;
                        let mut table_state: RefMut<'ret, _> = RefCell::borrow_mut(table_state);
                        let mut _table_state = table_state.deref_mut();

                        let res = last_err(
                            visitor
                                .vbin_fields(print_fmt, &array)
                                .into_iter()
                                .zip(_table_state.field_cols.iter_mut())
                                .map(|(res, col)| {
                                    handle_error(
                                        &_table_state.name,
                                        col,
                                        res.map(|print_arg| print_arg.value),
                                    )
                                }),
                        );

                        Ok((DerefMutWrapper::RcRefMut(table_state), res))
                    }
                    _ => Err(MainError::EvalError(EvalError::IllegalType)),
                }
            }};
        }

        match &mut self.variant {
            ReadStateVariant::Generic(state) => generic_iter!(state, visitor),
            ReadStateVariant::BPrint {
                generic,
                fmt_fmt,
                buf_fmt,
                meta_events_by_addr,
                meta_events_by_fmt,
                ..
            } => {
                let fmt = visitor.field_by_fmt(fmt_fmt)?;
                let addr = match fmt {
                    Value::U64Scalar(addr) => Ok(addr),
                    Value::I64Scalar(addr) => Ok(addr as u64),
                    _ => Err(EvalError::CannotDeref),
                }?;

                macro_rules! handle {
                    ($res:expr) => {{
                        match Rc::as_ref($res) {
                            Ok(entry) => {
                                bprint_meta_iter!(entry, visitor, buf_fmt)
                            }
                            Err(_) => generic_iter!(generic, visitor),
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
                            let print_fmt = print_fmt.to_str().ok_or(EvalError::IllegalType)?;
                            Ok(traceevent::print::parse_print_fmt(
                                header,
                                print_fmt.as_bytes(),
                            )?)
                        };

                        let make_schema = |print_fmt: PrintFmtStr| {
                            let (meta_event_name, full_schema, fields_schema) =
                                Self::make_print_fmt_schemas(header, &print_fmt)?;

                            let state = TableState::new(
                                full_schema,
                                fields_schema,
                                self.options,
                                &format!("trace_printk@{meta_event_name}"),
                                self.scope,
                            )?;
                            Ok((RefCell::new(state), print_fmt))
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
                            Err(_) => Rc::new(Err(MainError::NotAMetaEvent)),
                        };

                        handle!(entry.insert(new))
                    }
                }
            }
        }
    }

    fn drain_states(self) -> impl Iterator<Item = TableState<'scope>> {
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
                        |(_, entry)| {
                            let (table_state, _) = Rc::into_inner(entry).unwrap().ok()?;
                            let table_state = RefCell::into_inner(table_state);
                            Some(table_state)
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
                            Err(()) => None,
                        }
                    }
                    _ => None,
                }
            } else {
                match atom {
                    PrintAtom::Fixed(fixed) => {
                        let _ = nom::combinator::all_consuming(field_name_parser())
                            .parse(fixed.as_bytes())
                            .finish()
                            .map(|(_, name)| {
                                field_name = Some(name);
                            });
                        None
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
        let (full_schema, fields_schema) = Self::make_schemas(header, fields?)?;
        let event_name = event_name.ok_or(MainError::NotAMetaEvent)?;
        Ok((event_name.into(), full_schema, fields_schema))
    }

    fn make_schemas<FieldsIterator>(
        header: &Header,
        fields: FieldsIterator,
    ) -> Result<(Schema, Schema), MainError>
    where
        FieldsIterator: IntoIterator<Item = (String, Type)>,
    {
        let char_typ = header.kernel_abi().char_typ();
        let long_size = header.kernel_abi().long_size;

        let field_cols = fields.into_iter().map(|(name, typ)| {
            fn guess_typ(typ: &Type, char_typ: &Type, long_size: &LongSize) -> Result<DataType, MainError> {
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
                    Type::Array(inner, _) | Type::Pointer(inner) if &**inner == char_typ => Ok(DataType::Utf8),

                    // u8 [] are considered as byte buffer
                    Type::Array(inner, _) | Type::Pointer(inner) if matches!(&**inner, Type::Typedef(_, name) if name == "u8") => Ok(DataType::Binary),

                    Type::Array(inner, _) | Type::Pointer(inner) if matches!(
                        inner.resolve_wrapper(),
                        Type::Bool | Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64
                    ) => Ok(DataType::List(Box::new(Field::new(
                        "",
                        guess_typ(inner, char_typ, long_size)?,
                        true,
                    )))),

                    Type::Pointer(..) => match long_size {
                        LongSize::Bits32 => Ok(DataType::UInt32),
                        LongSize::Bits64 => Ok(DataType::UInt64),
                    },

                    // TODO: try to do symbolic resolution of enums somehow, maybe with BTF
                    // Do we want that always ? What about conversion from other formats where the
                    // enum is not available ? Maybe that should be left to a Python function,
                    // hooked with the BTF parser, and BTF available in platform info.
                    Type::Typedef(typ, _) | Type::Enum(typ, _) => guess_typ(typ, char_typ, long_size),

                    // TODO: handle DynamicScalar such as cpumasks, either as extension types or
                    // simply as DataType::Binary, decoded to be little endian and 8bit-word based
                    // instead of the weird chunked format of the kernel.
                    // Or maybe kernel bitmap can be turned into a bitmap type of parquet if it is
                    // supported. We need to check polars/pandas support for those as well.
                    typ => Err(MainError::TypeNotHandled(Box::new(typ.clone()))),
                }
            }
            let typ = guess_typ(&typ, &char_typ, &long_size)?;
            Ok(Field::new(name, typ, true))
        });
        let field_cols: Result<Vec<_>, MainError> = field_cols.collect();
        let field_cols = field_cols?;

        let fields_schema = Schema::from(field_cols.clone());
        let full_schema = Schema::from(
            FixedCols::arrow_fields()
                .chain(field_cols)
                .collect::<Vec<_>>(),
        );
        Ok((full_schema, fields_schema))
    }
}

#[derive(Debug)]
struct TableState<'scope> {
    name: String,
    path: PathBuf,
    fields_schema: Schema,
    fixed_cols: FixedCols,
    field_cols: Vec<FieldArray>,

    sender: Sender<ArrayChunk>,
    handle: ScopedJoinHandle<'scope, Result<(), MainError>>,
}

struct EventWriteState {
    full_schema: Schema,
    options: WriteOptions,
    writer: FileWriter<File>,
    count: u64,
}

impl<'scope> TableState<'scope> {
    fn new(
        full_schema: Schema,
        fields_schema: Schema,
        options: WriteOptions,
        name: &str,
        scope: &'scope Scope,
    ) -> Result<Self, MainError> {
        let (fixed_cols, field_cols) = Self::make_cols(&fields_schema)?;

        let path = PathBuf::from(format!("{}.parquet", name));
        let file = File::create(&path)?;
        let writer = FileWriter::try_new(file, full_schema.clone(), options)?;
        let (sender, receiver) = bounded(128);

        let mut write_state = EventWriteState {
            full_schema,
            options,
            writer,
            count: 0,
        };
        let write_thread = move |_: &_| -> Result<_, MainError> {
            for chunk in receiver.iter() {
                write_state.dump_to_file(chunk)?;
            }
            write_state.writer.end(None)?;
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
        })
    }

    fn make_cols(schema: &Schema) -> Result<(FixedCols, Vec<FieldArray>), MainError> {
        macro_rules! make_array {
            ($variant:path) => {
                Ok($variant(MutablePrimitiveArray::with_capacity(CHUNK_SIZE)))
            };
        }
        let make_col = |field: &Field| match &field.data_type {
            DataType::Int8 => make_array!(FieldArray::I8),
            DataType::Int16 => make_array!(FieldArray::I16),
            DataType::Int32 => make_array!(FieldArray::I32),
            DataType::Int64 => make_array!(FieldArray::I64),

            DataType::UInt8 => make_array!(FieldArray::U8),
            DataType::UInt16 => make_array!(FieldArray::U16),
            DataType::UInt32 => make_array!(FieldArray::U32),
            DataType::UInt64 => make_array!(FieldArray::U64),

            DataType::Boolean => Ok(FieldArray::Bool(MutableBooleanArray::with_capacity(
                CHUNK_SIZE,
            ))),
            DataType::Utf8 => Ok(FieldArray::Str(MutableUtf8Array::with_capacity(CHUNK_SIZE))),
            DataType::Binary => Ok(FieldArray::Binary(MutableBinaryArray::with_capacity(
                CHUNK_SIZE,
            ))),

            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::UInt8,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListU8(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }
            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::UInt16,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListU16(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }
            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::UInt32,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListU32(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }
            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::UInt64,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListU64(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }

            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::Int8,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListI8(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }
            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::Int16,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListI16(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }
            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::Int32,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListI32(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }
            DataType::List(field)
                if matches!(
                    field.deref(),
                    Field {
                        data_type: DataType::Int64,
                        ..
                    }
                ) =>
            {
                Ok(FieldArray::ListI64(MutableListArray::with_capacity(
                    CHUNK_SIZE,
                )))
            }

            typ => Err(MainError::ArrowDataTypeNotHandled(Box::new(typ.clone()))),
        };

        let fields: Result<Vec<_>, MainError> = schema.fields.iter().map(make_col).collect();
        let fields = fields?;

        let fixed = FixedCols::new();
        Ok((fixed, fields))
    }

    fn extract_chunk(&mut self) -> Result<ArrayChunk, MainError> {
        let (mut fixed_cols, mut field_cols) = Self::make_cols(&self.fields_schema)?;

        assert_eq!(field_cols.len(), self.field_cols.len());
        core::mem::swap(&mut self.field_cols, &mut field_cols);
        core::mem::swap(&mut self.fixed_cols, &mut fixed_cols);

        Ok(Chunk::new(
            fixed_cols
                .into_arcs()
                .chain(field_cols.into_iter().map(|col| col.into_arc()))
                .collect(),
        ))
    }
}

impl EventWriteState {
    fn dump_to_file(&mut self, chunk: ArrayChunk) -> Result<(), MainError> {
        self.count += 1;

        let row_groups = RowGroupIterator::try_new(
            [Ok(chunk)].into_iter(),
            &self.full_schema,
            self.options,
            self.full_schema
                .fields
                .iter()
                .map(|_| vec![Encoding::Plain])
                .collect(),
        )?;

        for group in row_groups {
            let group = group?;
            self.writer.write(group)?;
        }
        Ok(())
    }
}

fn last_err<E, I: Iterator<Item = Result<(), E>>>(iter: I) -> Result<(), E> {
    let mut res = Ok(());
    for x in iter {
        if x.is_err() {
            res = x;
        }
    }
    res
}

// fn main2() -> Result<(), ArrowError> {
// // declare arrays
// let a = Int8Array::from(&[Some(1), None, Some(3)]);
// let b = Int32Array::from(&[Some(2), None, Some(6)]);

// // declare a schema with fields
// let schema = Schema::from(vec![
// Field::new("c2", DataType::Int32, true),
// Field::new("c1", DataType::Int8, true),
// ]);

// // declare chunk
// let chunk = Chunk::new(vec![a.arced(), b.arced()]);

// // write to parquet (probably the fastest implementation of writing to parquet out there)
// let options = WriteOptions {
// write_statistics: false,
// compression: CompressionOptions::Snappy,
// version: Version::V1,
// data_pagesize_limit: None,
// };

// let row_groups = RowGroupIterator::try_new(
// vec![Ok(chunk)].into_iter(),
// &schema,
// options,
// vec![vec![Encoding::Plain], vec![Encoding::Plain]],
// )?;

// // anything implementing `std::io::Write` works
// // let mut file = vec![];
// let file = File::create("hello.pq").unwrap();
// let mut writer = FileWriter::try_new(file, schema, options)?;

// // Write the file.
// for group in row_groups {
// writer.write(group?)?;
// }
// let _ = writer.end(None)?;
// Ok(())
// }
