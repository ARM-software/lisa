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

use std::{error::Error, fs::File, io::Write, path::PathBuf, process::ExitCode};

#[cfg(target_arch = "x86_64")]
use mimalloc::MiMalloc;
#[global_allocator]
#[cfg(target_arch = "x86_64")]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{Parser, Subcommand, ValueEnum};
use lib::error::DynMultiError;
use parquet::basic::{Compression as ParquetCompression, ZstdLevel};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, value_name = "ERRORS_JSON")]
    errors_json: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Debug, ValueEnum)]
#[clap(rename_all = "lower")]
enum Compression {
    Snappy,
    Lz4,
    Zstd,
    Uncompressed,
}

impl From<Compression> for ParquetCompression {
    fn from(compression: Compression) -> ParquetCompression {
        match compression {
            Compression::Snappy => ParquetCompression::SNAPPY,
            Compression::Zstd => ParquetCompression::ZSTD(
                ZstdLevel::try_new(3).expect("Invalid zstd compression level"),
            ),
            // LZ4 codec is deprecated, so use LZ4_RAW instead:
            // https://parquet.apache.org/docs/file-format/data-pages/compression/
            Compression::Lz4 => ParquetCompression::LZ4_RAW,
            Compression::Uncompressed => ParquetCompression::UNCOMPRESSED,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
#[clap(rename_all = "lower")]
enum TraceFormat {
    TraceDat,
}

#[derive(Subcommand)]
enum Command {
    HumanReadable {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

        #[arg(long, value_name = "TRACE FORMAT")]
        trace_format: TraceFormat,

        #[arg(long, value_name = "RAW")]
        raw: bool,
    },
    Parquet {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

        #[arg(long, value_name = "TRACE FORMAT")]
        trace_format: TraceFormat,

        #[arg(long, value_name = "EVENT")]
        event: Option<Vec<String>>,

        #[arg(long)]
        unique_timestamps: bool,

        #[arg(long, default_value = "snappy")]
        compression: Compression,

        // Large row group:
        //     * Good for disk I/O.
        //     * Bad for network I/O if only a small part of the row group is needed.
        //     * Good for metadata size, as the total groups metadata can be quite large on really
        //       large files, and is loaded eagerly by polars and datafusion, leading to really bad
        //       memory consumption.
        #[arg(long, default_value_t=1024 * 1024)]
        row_group_size: usize,

        // Cap the amount of errors accumulated so we don't end up with ridiculously large memory
        // consumption or JSON files
        #[arg(long, default_value_t = 256)]
        max_errors: usize,
    },
    CheckHeader {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

        #[arg(long, value_name = "TRACE FORMAT")]
        trace_format: TraceFormat,
    },
    Metadata {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

        #[arg(long, value_name = "TRACE FORMAT")]
        trace_format: TraceFormat,

        #[arg(long, value_name = "KEY")]
        key: Option<Vec<String>>,

        // Cap the amount of errors accumulated so we don't end up with ridiculously large memory
        // consumption or JSON files
        #[arg(long, default_value_t = 256)]
        max_errors: usize,
    },
}

impl Command {
    fn trace_format(&self) -> Option<TraceFormat> {
        match self {
            Command::HumanReadable { trace_format, .. } => Some(trace_format.clone()),
            Command::Parquet { trace_format, .. } => Some(trace_format.clone()),
            Command::CheckHeader { trace_format, .. } => Some(trace_format.clone()),
            Command::Metadata { trace_format, .. } => Some(trace_format.clone()),
        }
    }
}

fn _main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let stdout = std::io::stdout().lock();
    let mut out = std::io::BufWriter::with_capacity(1024 * 1024, stdout);

    let res: Result<(), DynMultiError> = match cli.command.trace_format() {
        #[cfg(feature = "tracedat")]
        Some(TraceFormat::TraceDat) => tracedat_main(&cli, &mut out),

        _ => Err(DynMultiError::from_string("File format not handled".into())),
    };

    out.flush()?;

    if let Err(err) = &res {
        eprintln!("Errors happened while processing the trace: {err}");
    }

    if let Some(path) = &cli.errors_json {
        let errors = match &res {
            Err(err) => err.errors().map(|err| err.to_string()).collect(),
            Ok(_) => Vec::new(),
        };
        let mut file = File::create(path)?;
        let json_value = serde_json::json!({
            "errors": errors,
        });
        file.write_all(json_value.to_string().as_bytes())?;
    }
    match res {
        Ok(_) => Ok(()),
        Err(_) => Err("Errors happened".into()),
    }
}

fn main() -> ExitCode {
    match _main() {
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
        Ok(_) => ExitCode::from(0),
    }
}

#[cfg(feature = "tracedat")]
fn tracedat_main<W>(cli: &Cli, out: &mut W) -> Result<(), DynMultiError>
where
    W: Write,
{
    use std::ops::DerefMut as _;

    use lib::tracedat::{
        check::check_header,
        parquet::dump::{dump_events, dump_metadata},
        print::print_events,
    };
    use traceevent::{
        header,
        header::{Header, Timestamp},
        io::MmapFile,
    };

    let open_trace = |path| -> Result<(Header, Box<_>), DynMultiError> {
        let file = std::fs::File::open(path)?;
        let reader = unsafe { MmapFile::new(file) }?;
        let mut reader = Box::new(reader);
        let header = header::header(reader.deref_mut())?;
        Ok((header, reader))
    };

    match &cli.command {
        Command::HumanReadable { trace, raw, .. } => {
            let (header, reader) = open_trace(trace)?;
            print_events(&header, reader, out, *raw)
        }
        Command::Parquet {
            trace,
            event,
            unique_timestamps,
            compression,
            row_group_size,
            max_errors,
            ..
        } => {
            let (header, reader) = open_trace(trace)?;
            let make_ts: Box<dyn FnMut(_) -> _> = if *unique_timestamps {
                let make_unique_timestamps = {
                    let mut prev = 0;
                    move |ts: Timestamp| {
                        // Ensure there is at least 2ns of difference between each timestamp, so that we never
                        // end up with duplicated timestamps once converted to f64 (due to rounding errors).
                        prev = std::cmp::max(ts, prev + 2);
                        prev
                    }
                };
                Box::new(make_unique_timestamps)
            } else {
                Box::new(|ts| ts)
            };

            let compression = compression.clone().into();
            match dump_events(
                &header,
                reader,
                make_ts,
                event.clone(),
                *row_group_size,
                compression,
                *max_errors,
            ) {
                Ok(metadata) => Ok(metadata.dump(File::create("meta.json")?)?),
                Err(err) => Err(err),
            }
        }
        Command::CheckHeader { trace, .. } => {
            let (header, _) = open_trace(trace)?;
            check_header(&header, out)
        }
        Command::Metadata {
            trace,
            key,
            max_errors,
            ..
        } => {
            let (header, reader) = open_trace(trace)?;
            dump_metadata(&header, reader, out, key.clone(), *max_errors)
        }
    }
}
