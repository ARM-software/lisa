use std::{error::Error, fs::File, io::Write, path::PathBuf, process::ExitCode};

#[cfg(target_arch = "x86_64")]
use mimalloc::MiMalloc;
use traceevent::{header, header::Timestamp};
#[global_allocator]
#[cfg(target_arch = "x86_64")]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{Parser, Subcommand, ValueEnum};
use lib::{
    check::check_header,
    error::DynMultiError,
    parquet::{dump_events, dump_metadata},
    print::print_events,
};
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

#[derive(Subcommand)]
enum Command {
    HumanReadable {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

        #[arg(long, value_name = "RAW")]
        raw: bool,
    },
    Parquet {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

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
    },
    CheckHeader {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,
    },
    Metadata {
        #[arg(long, value_name = "TRACE")]
        trace: PathBuf,

        #[arg(long, value_name = "KEY")]
        key: Option<Vec<String>>,
    },
}

fn _main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let open_trace = |path| -> Result<_, Box<dyn Error>> {
        let file = std::fs::File::open(path)?;
        let mut reader = unsafe { traceevent::io::MmapFile::new(file) }?;
        let header = header::header(&mut reader)?;
        Ok((header, reader))
    };

    let make_unique_timestamps = {
        let mut prev = 0;
        move |ts: Timestamp| {
            // Ensure there is at least 2ns of difference between each timestamp, so that we never
            // end up with duplicated timestamps once converted to f64 (due to rounding errors).
            prev = std::cmp::max(ts, prev + 2);
            prev
        }
    };

    let stdout = std::io::stdout().lock();
    let mut out = std::io::BufWriter::with_capacity(1024 * 1024, stdout);

    let res: Result<_, DynMultiError> = match cli.command {
        Command::HumanReadable { trace, raw } => {
            let (header, reader) = open_trace(trace)?;
            print_events(&header, reader, &mut out, raw)
        }
        Command::Parquet {
            trace,
            event,
            unique_timestamps,
            compression,
            row_group_size,
        } => {
            let (header, reader) = open_trace(trace)?;
            let make_ts: Box<dyn FnMut(_) -> _> = if unique_timestamps {
                Box::new(make_unique_timestamps)
            } else {
                Box::new(|ts| ts)
            };

            let compression = match compression {
                Compression::Snappy => ParquetCompression::SNAPPY,
                Compression::Zstd => ParquetCompression::ZSTD(
                    ZstdLevel::try_new(3).expect("Invalid zstd compression level"),
                ),
                // LZ4 codec is deprecated, so use LZ4_RAW instead:
                // https://parquet.apache.org/docs/file-format/data-pages/compression/
                Compression::Lz4 => ParquetCompression::LZ4_RAW,
                Compression::Uncompressed => ParquetCompression::UNCOMPRESSED,
            };
            match dump_events(&header, reader, make_ts, event, row_group_size, compression) {
                Ok(metadata) => Ok(metadata.dump(File::create("meta.json")?)?),
                Err(err) => Err(err),
            }
        }
        Command::CheckHeader { trace } => {
            let (header, _) = open_trace(trace)?;
            check_header(&header, &mut out)
        }
        Command::Metadata { trace, key } => {
            let (header, reader) = open_trace(trace)?;
            dump_metadata(&header, reader, &mut out, key)
        }
    };
    out.flush()?;

    if let Err(err) = &res {
        eprintln!("Errors happened while processing the trace: {err}");
    }

    if let Some(path) = &cli.errors_json {
        let errors = match &res {
            Err(err) => err
                .errors()
                .into_iter()
                .map(|err| err.to_string())
                .collect(),
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
