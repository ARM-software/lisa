use std::{error::Error, io::Write, ops::Deref};

#[cfg(target_arch = "x86_64")]
use mimalloc::MiMalloc;
use traceevent::{header, header::Timestamp};
#[global_allocator]
#[cfg(target_arch = "x86_64")]
static GLOBAL: MiMalloc = MiMalloc;

use lib::{check_header, parquet::dump_events, print::print_events};

enum Action {
    HumanReadable,
    Parquet,
    CheckHeader,
}

fn main() -> Result<(), Box<dyn Error>> {
    let stdout = std::io::stdout().lock();
    let mut out = std::io::BufWriter::with_capacity(1024 * 1024, stdout);

    let args: Vec<String> = std::env::args().collect();
    eprintln!("CLI args: {args:?}");

    let path = &args[1];
    let file = std::fs::File::open(path).unwrap();
    let mut reader = unsafe { traceevent::io::MmapFile::new(file) }.unwrap();
    let header = header::header(&mut reader).expect("failed to parse header");

    let action = &args[2];
    let action = match action.deref() {
        "txt" => Action::HumanReadable,
        "pq" => Action::Parquet,
        "check-header" => Action::CheckHeader,
        _ => panic!("invalid format"),
    };

    // TODO: make that optional
    let unique_timestamps = {
        // We make the timestamp unique assuming it will be manipulated as an f64 by consumers
        let mut last = 0.0;
        move |mut ts: Timestamp| {
            let mut _ts = ts as f64;
            while _ts == last {
                ts += 1;
                _ts = ts as f64;
            }
            last = _ts;
            ts
        }
    };

    let res = match action {
        Action::HumanReadable => print_events(&header, reader, &mut out),
        Action::Parquet => dump_events(&header, reader, unique_timestamps),
        Action::CheckHeader => check_header(&header, &mut out),
    };

    out.flush()?;
    if let Err(err) = &res {
        println!("Unrecoverable error: {err}")
    }
    res
}
