pub mod parquet;
pub mod print;

use std::{error::Error, io::Write};

use traceevent::header::Header;

macro_rules! convert_err_impl {
    ($src:ident, $dst:ident) => {
        impl From<$src> for $dst {
            fn from(err: $src) -> Self {
                $dst::$src(err)
            }
        }
    };
}
pub(crate) use convert_err_impl;

pub fn check_header<W: Write>(header: &Header, mut out: W) -> Result<(), Box<dyn Error>> {
    for desc in header.event_descs() {
        writeln!(&mut out, "Checking event \"{}\" format", desc.name)?;

        let raw_fmt = std::str::from_utf8(desc.raw_fmt()?)?;
        match desc.event_fmt() {
            Err(err) => {
                writeln!(
                    &mut out,
                    "    Error while parsing event format: {err}:\n{raw_fmt}"
                )
            }
            Ok(fmt) => {
                match fmt.print_args() {
                    Ok(print_args) => {
                        print_args.iter().enumerate().try_for_each(|(i, res)| match res {
                            Err(err) => {
                                writeln!(&mut out, "    Error while compiling printk argument #{i}: {err}:\n{raw_fmt}")
                            }
                            Ok(_) => Ok(()),
                        })?;
                        Ok(())
                    }
                    Err(err) => {
                        writeln!(&mut out, "    Error while parsing event print format arguments: {err}:\n{raw_fmt}")
                    }
                }
            }
        }?;
    }
    Ok(())
}
