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

use std::io::Write;

use traceevent::header::Header;

use crate::error::DynMultiError;

pub fn check_header<W: Write>(header: &Header, mut out: W) -> Result<(), DynMultiError> {
    for desc in header.event_descs() {
        writeln!(&mut out, "Checking event \"{}\" (ID {}) format:", desc.name, desc.id)?;

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

    writeln!(&mut out, "Header options:")?;
    for option in header.options() {
        writeln!(&mut out, "  {option:?}")?;
    }
    Ok(())
}
