use std::io::Write;

use traceevent::header::Header;

use crate::error::DynMultiError;

pub fn check_header<W: Write>(header: &Header, mut out: W) -> Result<(), DynMultiError> {
    for desc in header.event_descs() {
        writeln!(&mut out, "Checking event \"{}\" format:", desc.name)?;

        let raw_fmt = std::str::from_utf8(desc.raw_fmt()?)?;
        writeln!(&mut out, "{raw_fmt}")?;
        match desc.event_fmt() {
            Err(err) => {
                writeln!(&mut out, "    Error while parsing event format: {err}")
            }
            Ok(fmt) => {
                let _ = fmt.struct_fmt().map_err(|err| {
                    writeln!(
                        &mut out,
                        "    Error while parsing event struct format: {err}"
                    )
                });

                match fmt.print_args() {
                    Ok(print_args) => {
                        print_args
                            .into_iter()
                            .enumerate()
                            .try_for_each(|(i, res)| match res {
                                Err(err) => {
                                    writeln!(
                                        &mut out,
                                        "    Error while compiling printk argument #{i}: {err}"
                                    )
                                }
                                Ok(_) => Ok(()),
                            })?;
                        Ok(())
                    }
                    Err(err) => {
                        writeln!(
                            &mut out,
                            "    Error while parsing event print format arguments: {err}"
                        )
                    }
                }
            }
        }?;
        writeln!(&mut out)?;
    }
    Ok(())
}
