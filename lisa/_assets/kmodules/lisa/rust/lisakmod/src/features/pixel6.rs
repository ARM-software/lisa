/* SPDX-License-Identifier: GPL-2.0 */

// use crate::{
// features::{Feature, Visibility},
// lifecycle::new_lifecycle,
// };
//
use alloc::{
    format,
    string::{String, ToString as _},
    sync::Arc,
    vec,
    vec::Vec,
};

use embedded_io::{Seek as _, Write as _};

use crate::{
    error::{Error, ResultExt as _, error},
    features::{FeaturesConfig, FeaturesService, define_feature, wq::WqFeature},
    lifecycle::new_lifecycle,
    parsec::{self, ClosureParser, ParseResult, Parser},
    runtime::{
        fs::{File, OpenFlags},
        traceevent::new_event,
        wq,
    },
};

type DeviceId = u32;
type ChanId = u32;

#[derive(Debug, PartialEq)]
struct Sample {
    ts: u64,
    value: u64,
    chan: ChanId,
    chan_name: String,
}

fn emeter_parser<F, I>(f: &mut F) -> impl Parser<I, (), Error>
where
    I: parsec::Input<Item = u8>,
    F: FnMut(Sample) -> Result<(), Error>,
{
    use crate::parsec::{eof, many, tags, u64_decimal, whitespace};
    ClosureParser::new(move |input: I| {
        let (input, _) = tags(b"t=").parse(input)?;
        let (input, _ts) = u64_decimal().parse(input)?;
        let (input, _) = whitespace().parse(input)?;
        let (input, res): (_, Result<(), Error>) = many(emeter_row_parser(f)).parse(input)?;
        let (input, res) = ParseResult::from_result(input, res)?;
        let (input, _) = whitespace().parse(input)?;
        let (input, _) = eof().parse(input)?;
        ParseResult::Success {
            remainder: input,
            x: res,
        }
    })
}

fn emeter_row_parser<F, I>(f: &mut F) -> impl Parser<I, Result<(), Error>, Error>
where
    I: parsec::Input<Item = u8>,
    F: FnMut(Sample) -> Result<(), Error>,
{
    use crate::parsec::{many, not_tag, tag, tags, u64_decimal, whitespace};
    ClosureParser::new(move |input: I| {
        let (input, _) = tags(b"CH").parse(input)?;
        let (input, chan) = u64_decimal()
            .map_cut(|x| {
                x.try_into()
                    .map_err(|err| error!("Could not convert u64 to ChanId: {err}"))
            })
            .parse(input)?;
        let (input, _) = tags(b"(T=").parse(input)?;
        let (input, ts) = u64_decimal().parse(input)?;
        let (input, _) = tag(b')').parse(input)?;

        let (input, _) = tag(b'[').parse(input)?;

        let (input, chan_name) = many(not_tag(b']'))
            .map_cut(|s| {
                String::from_utf8(s)
                    .map_err(|err| error!("Could not convert channel name to UTF-8 string: {err}"))
            })
            .parse(input)?;

        let (input, _) = tag(b']').parse(input)?;
        let (input, _) = tag(b',').parse(input)?;
        let (input, _) = whitespace().parse(input)?;
        let (input, value) = u64_decimal().parse(input)?;

        let (input, _) = whitespace().parse(input)?;

        let sample = Sample {
            ts,
            value,
            chan,
            chan_name,
        };

        ParseResult::Success {
            remainder: input,
            x: f(sample),
        }
    })
}

struct Device {
    value_file: File,
    rate_file: File,
    config: DeviceConfig,
}

impl Device {
    fn parse_samples<F>(&mut self, f: &mut F) -> Result<(), Error>
    where
        F: FnMut(Sample) -> Result<(), Error>,
    {
        self.value_file.rewind()?;
        let samples = self.value_file.read_to_end()?;
        let input = parsec::BytesInput::new(&samples);
        let mut parser = emeter_parser(f);
        parser.parse(input).into_result()
    }
}

struct DeviceConfig {
    id: DeviceId,
    folder: String,
    hardware_sampling_rate_us: u64,
}

struct Pixel6EmeterConfig {
    devices: Vec<DeviceConfig>,
}

// /* There is no point in setting this value to less than 8 times what is written
//  * in usec to POWER_METER_RATE_FILE
//  */
// const POWER_METER_SAMPLING_RATE_MS: u64 = 50;

define_feature! {
    struct Pixel6Emeter,
    name: "lisa__pixel6_emeter",
    visibility: Public,
    Service: (),
    Config: (),
    dependencies: [WqFeature],
    init: |configs| {
        Ok((
            FeaturesConfig::new(),
            new_lifecycle!(|services| {
                let services: FeaturesService = services;
                let wq = services.get::<WqFeature>()
                    .expect("Could not get service for WqFeature")
                    .wq();

                let hardware_sampling_rate_us = 500;
                let config = Pixel6EmeterConfig {
                    devices: vec![
                        DeviceConfig {
                           id: 0,
                           folder: "/sys/bus/iio/devices/iio:device0/".into(),
                           hardware_sampling_rate_us,
                        },
                        DeviceConfig {
                           id: 1,
                           folder: "/sys/bus/iio/devices/iio:device1/".into(),
                           hardware_sampling_rate_us,
                        },
                    ]
                };

                let mut devices = config.devices.into_iter().map(|device_config| {
                    let value_file = File::open(
                        &(device_config.folder.to_string() + "/energy_value"),
                        OpenFlags::ReadOnly,
                        0,
                    )?;
                    let mut rate_file = File::open(
                        &(device_config.folder.to_string() + "/sampling_rate"),
                        OpenFlags::WriteOnly,
                        0,
                    )?;

                    // Note that this is the hardware sampling rate. Software will only see an
                    // updated value every 8 hardware periods
                    writeln!(rate_file, "{}", device_config.hardware_sampling_rate_us).map_err(|err| error!("Could not write to sampling_rate file: {err}"))?;

                    let device = Device {
                        value_file,
                        rate_file,
                        config: device_config,
                    };
                    Ok(device)
                }).collect::<Result<Vec<_>, Error>>()?;

                let emit = new_event! {
                    lisa__pixel6_emeter,
                    fields: {
                        ts: u64,
                        device: DeviceId,
                        chan: ChanId,
                        chan_name: &str,
                        value: u64,
                    }
                }?;

                // There is no point in setting this value to less than 8 times what is written in
                // usec to the sampling_rate file, as the hardware will only expose a new value
                // every 8 hardware periods.
                let software_sampling_rate_us = hardware_sampling_rate_us / 8;

                let work_item = wq::new_work_item!(wq, move |work| {
                    let process_device = |device: &mut Device| -> Result<(), Error> {
                        let device_id = device.config.id;
                        device.parse_samples(&mut |sample: Sample| {
                            emit(sample.ts, device_id, sample.chan, &sample.chan_name, sample.value);
                            Ok(())
                        })?;
                        Ok(())
                    };
                    for device in &mut devices {
                        process_device(device)
                            .with_context(|| format!("Could not read samples from device {}", device.config.id))
                            .print_err();
                    }
                    work.enqueue(software_sampling_rate_us);
                });
                work_item.enqueue(software_sampling_rate_us);
                yield_!(Ok(Arc::new(())));
                Ok(())
            }),
        ))
    },
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::parsec;

    #[test]
    fn test_emeter_parser() {
        let input = b"t=473848\nCH42(T=473848)[S10M_VDD_TPU], 3161249\nCH1(T=473848)[VSYS_PWR_MODEM], 48480309\nCH2(T=473848)[VSYS_PWR_RFFE], 9594393\nCH3(T=473848)[S2M_VDD_CPUCL2], 28071872\nCH4(T=473848)[S3M_VDD_CPUCL1], 17477139\nCH5(T=473848)[S4M_VDD_CPUCL0], 113447446\nCH6(T=473848)[S5M_VDD_INT], 12543588\nCH7(T=473848)[S1M_VDD_MIF], 25901660\n";
        let input = parsec::BytesInput::new(input);
        let mut samples = Vec::new();
        let mut f = |sample| {
            samples.push(sample);
            Ok(())
        };
        let mut parser = emeter_parser(&mut f);
        let res = parser.parse(input);
        res.unwrap_success();

        let expected = vec![
            Sample {
                ts: 473848,
                value: 3161249,
                chan: 42,
                chan_name: "S10M_VDD_TPU".into(),
            },
            Sample {
                ts: 473848,
                value: 48480309,
                chan: 1,
                chan_name: "VSYS_PWR_MODEM".into(),
            },
            Sample {
                ts: 473848,
                value: 9594393,
                chan: 2,
                chan_name: "VSYS_PWR_RFFE".into(),
            },
            Sample {
                ts: 473848,
                value: 28071872,
                chan: 3,
                chan_name: "S2M_VDD_CPUCL2".into(),
            },
            Sample {
                ts: 473848,
                value: 17477139,
                chan: 4,
                chan_name: "S3M_VDD_CPUCL1".into(),
            },
            Sample {
                ts: 473848,
                value: 113447446,
                chan: 5,
                chan_name: "S4M_VDD_CPUCL0".into(),
            },
            Sample {
                ts: 473848,
                value: 12543588,
                chan: 6,
                chan_name: "S5M_VDD_INT".into(),
            },
            Sample {
                ts: 473848,
                value: 25901660,
                chan: 7,
                chan_name: "S1M_VDD_MIF".into(),
            },
        ];

        drop(parser);
        for (_sample, _expected) in samples.into_iter().zip(expected) {
            assert_eq!(_sample, _expected);
        }
    }

    #[test]
    fn test_emeter_row_parser() {
        let input = b"CH42(T=473848)[S10M_VDD_TPU], 3161249";
        let input = parsec::BytesInput::new(input);
        let mut sample = None;
        let mut f = |_sample| {
            sample = Some(_sample);
            Ok(())
        };
        let mut parser = emeter_row_parser(&mut f);
        let res = parser.parse(input);
        let _ = res.unwrap_success();
        drop(parser);
        let parsed = sample.unwrap();
        let expected = Sample {
            ts: 473848,
            value: 3161249,
            chan: 42,
            chan_name: "S10M_VDD_TPU".into(),
        };
        assert_eq!(parsed, expected);
    }
}
