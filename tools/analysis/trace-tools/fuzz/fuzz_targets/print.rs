#![no_main]
#[cfg(target_arch = "x86_64")]
use mimalloc::MiMalloc;
use traceevent::header;
#[global_allocator]
#[cfg(target_arch = "x86_64")]
static GLOBAL: MiMalloc = MiMalloc;

use std::io::Write;

use libfuzzer_sys::fuzz_target;

use traceevent;
use lib::{check_header, parquet::dump_events, print::print_events};


fuzz_target!(|data: &[u8]| {
    // Speedup the test by not writing anything anywhere
    let mut out = std::io::sink();

    let mut run = move || {
        let mut reader: traceevent::io::BorrowingCursor<_> = data.into();
        let header = header::header(&mut reader)?;
        let res = print_events(&header, reader, &mut out);
        res
    };

    let _ = run();

});

