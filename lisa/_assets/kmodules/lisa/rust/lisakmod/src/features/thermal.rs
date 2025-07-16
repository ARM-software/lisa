/* SPDX-License-Identifier: GPL-2.0 */

// use crate::{
// features::{Feature, Visibility},
// lifecycle::new_lifecycle,
// };
//
use alloc::{ffi::CString, format, string::String, sync::Arc, vec::Vec};
use core::{
    cell::UnsafeCell,
    ffi::{CStr, c_int, c_uint},
    fmt,
    ptr::NonNull,
};

use itertools::Itertools as _;
use lisakmod_macros::inlinec::{NegativeError, PtrError, cfunc, incomplete_opaque_type};

use crate::{
    error::{Error, ResultExt as _, error},
    features::{
        DependenciesSpec, FeatureResources, FeaturesService, ProvidedFeatureResources,
        define_feature, wq::WqFeature,
    },
    lifecycle::new_lifecycle,
    query::query_type,
    runtime::{traceevent::new_event, wq},
};

query_type! {
    #[derive(Clone)]
    struct ThermalZoneConfig {
        name: String,
        sampling_period_us: u64,
    }
}

query_type! {
    #[derive(Clone)]
    struct ThermalConfig {
        zones: Vec<ThermalZoneConfig>,
    }
}

impl ThermalConfig {
    fn merge<'a, I>(iter: I) -> ThermalConfig
    where
        I: Iterator<Item = &'a Self>,
    {
        let zones: Vec<_> = iter.flat_map(|config| &config.zones).cloned().collect();
        ThermalConfig { zones }
    }
}

type ThermalZoneId = u32;
type Temperature = i32;

incomplete_opaque_type!(
    struct _CThermalZoneDevice,
    "struct thermal_zone_device",
    "linux/thermal.h"
);

#[repr(transparent)]
struct CThermalZoneDevice(UnsafeCell<_CThermalZoneDevice>);

impl CThermalZoneDevice {
    fn from_unsafe_cell_ref(tzd: &UnsafeCell<_CThermalZoneDevice>) -> &Self {
        // SAFETY: We can safely transmute between CThermalZoneDevice and UnsafeCell<_CThermalZoneDevice> as
        // they have the same layout thanks to repr(transparent)
        unsafe {
            core::mem::transmute::<&UnsafeCell<_CThermalZoneDevice>, &CThermalZoneDevice>(tzd)
        }
    }
}

unsafe impl Send for CThermalZoneDevice {}
// SAFETY: struct thermal_zone_device has its lock infrastructure to be usable as Sync
unsafe impl Sync for CThermalZoneDevice {}

struct ThermalZone {
    id: ThermalZoneId,
    tzd: &'static CThermalZoneDevice,
    config: ThermalZoneConfig,
}

impl fmt::Debug for ThermalZone {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThermalZone")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl ThermalZone {
    fn from_config(config: ThermalZoneConfig) -> Result<ThermalZone, Error> {
        #[cfunc]
        fn by_name(name: &CStr) -> Result<NonNull<UnsafeCell<_CThermalZoneDevice>>, PtrError> {
            r#"
            #include <linux/thermal.h>
            "#;

            r#"
            return thermal_zone_get_zone_by_name(name);
            "#
        }

        #[cfunc]
        fn get_id(tzd: &UnsafeCell<_CThermalZoneDevice>) -> c_uint {
            r#"
            #include <linux/thermal.h>
            #include "introspection.h"
            "#;

            r#"
            // From kernel v6.4, we have that function
            #if HAS_SYMBOL(thermal_zone_device_id)
                return thermal_zone_device_id(tzd);
            // In earlier versions, struct thermal_zone_device is defined in the public header.
            #else
                return tzd->id;
            #endif
            "#
        }

        let c_name = CString::new(&*config.name)
            .map_err(|err| error!("Could not convert thermal zone name to C string: {err}"))?;
        let tzd = by_name(c_name.as_c_str())
            .map_err(|err| error!("Could not resolve thermal zone: {err}"))?;

        // FIXME: is that assumption on thermal lifetime true ?
        // SAFETY: The thermal zone lives forever
        let tzd: &'static UnsafeCell<_CThermalZoneDevice> = unsafe { tzd.as_ref() };
        let id = get_id(tzd);
        // SAFETY: We can safely transmute between CThermalZoneDevice and UnsafeCell<_CThermalZoneDevice> as
        // they have the same layout thanks to repr(transparent)
        let tzd = CThermalZoneDevice::from_unsafe_cell_ref(tzd);
        Ok(ThermalZone { config, tzd, id })
    }

    fn get_temperature(&self) -> Result<i32, Error> {
        #[cfunc]
        fn get_temperature(
            tzd: &UnsafeCell<_CThermalZoneDevice>,
            temperature: &mut c_int,
        ) -> Result<c_uint, NegativeError<c_uint>> {
            r#"
            #include <linux/thermal.h>
            "#;

            r#"
            return thermal_zone_get_temp(tzd, temperature);
            "#
        }

        let mut temperature: c_int = 0;
        match get_temperature(&self.tzd.0, &mut temperature) {
            Err(err) => Err(error!("{err}")),
            Ok(0) => Ok(()),
            Ok(_) => unreachable!(),
        }?;
        Ok(temperature)
    }
}

define_feature! {
    struct ThermalFeature,
    name: "thermal",
    // FIXME: set to Public once the lifetime of the pointer returned by
    // thermal_zone_get_zone_by_name() is figured out (probably needs some refcounting)
    visibility: Private,
    Service: (),
    Config: ThermalConfig,
    dependencies: [WqFeature],
    resources: || {
        FeatureResources {
            provided: ProvidedFeatureResources {
                ftrace_events: ["lisa__thermal".into()].into()
            }
        }
    },
    init: |configs| {
        let config = ThermalConfig::merge(configs);
        Ok((
            DependenciesSpec::new(),
            new_lifecycle!(|services| {
                let services: FeaturesService = services;
                let wq = services.get::<WqFeature>()
                    .expect("Could not get service for WqFeature")
                    .wq();

                let mut zones = config.zones.into_iter()
                    .map(ThermalZone::from_config)
                    .collect::<Result<Vec<_>, Error>>()?;

                if zones.is_empty() {
                    Err(error!("No thermal zone was requested"))
                } else {
                    Ok(())
                }?;

                #[allow(non_snake_case)]
                let trace_lisa__thermal = new_event! {
                    lisa__thermal,
                    fields: {
                        id: ThermalZoneId,
                        name: &str,
                        temp: Temperature,
                    }
                }?;

                let process_zone = move |zone: &mut ThermalZone| -> Result<(), Error> {
                    trace_lisa__thermal(zone.id, &zone.config.name, zone.get_temperature()?);
                    Ok(())
                };

                let key = |zone: &ThermalZone| zone.config.sampling_period_us;
                zones.sort_by_key(key);
                let mut works: Vec<_> = zones
                    .into_iter()
                    .chunk_by(key)
                    .into_iter()
                    .map(|(sampling_period_us, zones)| {
                        let mut zones: Vec<_> = zones.collect();
                        let process_zone = &process_zone;
                        Ok(wq::new_work_item!(wq, move |work| {
                            for zone in &mut zones {
                                process_zone(zone)
                                    .with_context(|| format!("Could not read temperature from zone {}", zone.config.name))
                                    .print_err();
                            }
                            work.enqueue(sampling_period_us);
                        }))
                    }).collect::<Result<_, Error>>()?;

                for work in &mut works {
                    work.enqueue(0);
                }
                yield_!(Ok(Arc::new(())));
                Ok(())
            }),
        ))
    },
}
