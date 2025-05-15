/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{collections::BTreeSet, ffi::CString, string::String, sync::Arc, vec::Vec};
use core::{
    ffi::{CStr, c_char, c_int, c_uchar},
    ptr::NonNull,
};

use itertools::Itertools as _;
use lisakmod_macros::inlinec::{c_realchar, cfunc};
use serde::{Deserialize, Serialize};

use crate::{
    error::error,
    features::{DependenciesSpec, FeatureResources, ProvidedFeatureResources, define_feature},
    lifecycle::new_lifecycle,
};

pub fn legacy_features() -> impl Iterator<Item = &'static str> {
    #[cfunc]
    fn nth(i: &mut usize) -> Option<NonNull<c_realchar>> {
        r#"
        #include "features.h"
        "#;

        r#"
        const struct feature* base = __lisa_features_start;
        const struct feature* stop = __lisa_features_stop;
        size_t len = stop - base;

        while (1) {
            if (*i >= len) {
                return NULL;
            } else {
                const struct feature* nth = base + *i;
                *i += 1;
                if (nth->__internal) {
                    continue;
                } else {
                    return nth->name;
                }
            }
        }
        "#
    }

    let mut i: usize = 0;
    core::iter::from_fn(move || {
        nth(&mut i).map(|ptr| {
            let ptr = ptr.as_ptr();
            unsafe { CStr::from_ptr(ptr as *const c_char) }
                .to_str()
                .expect("Invalid UTF-8")
        })
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LegacyConfig {
    pub features: BTreeSet<String>,
}

impl Default for LegacyConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LegacyConfig {
    pub fn new() -> LegacyConfig {
        LegacyConfig {
            features: BTreeSet::new(),
        }
    }

    fn merge<'a, I>(configs: I) -> LegacyConfig
    where
        I: Iterator<Item = &'a LegacyConfig>,
    {
        LegacyConfig {
            features: configs
                .flat_map(|config| &config.features)
                .cloned()
                .collect(),
        }
    }

    pub fn new_all() -> LegacyConfig {
        LegacyConfig {
            features: legacy_features().map(|s| s.into()).collect(),
        }
    }
}

define_feature! {
    pub struct LegacyFeatures,
    name: "__legacy_features",
    visibility: Private,
    Service: (),
    Config: LegacyConfig,
    dependencies: [],
    resources: || {
        let events = legacy_features().filter_map(|name| {
            name.strip_prefix("event__").map(|event| event.into())
        }).collect();

        FeatureResources {
            provided: ProvidedFeatureResources {
                ftrace_events: events,
            }
        }
    },
    init: |configs| {
        let config = LegacyConfig::merge(configs);

        Ok((
            DependenciesSpec::new(),
            new_lifecycle!(|services| {
                #[cfunc]
                fn list_kernel_features() {
                    r#"
                    #include <linux/kernel.h>
                    #include <linux/printk.h>
                    #include "introspection.h"
                    "#;

                    r#"
                    pr_info("Kernel features detected. This will impact the module features that are available:\n");
                    const char *kernel_feature_names[] = {__KERNEL_FEATURE_NAMES};
                    const bool kernel_feature_values[] = {__KERNEL_FEATURE_VALUES};
                    for (size_t i=0; i < ARRAY_SIZE(kernel_feature_names); i++) {
                        pr_info("  %s: %s\n", kernel_feature_names[i], kernel_feature_values[i] ? "enabled" : "disabled");
                    }
                    "#
                }

                #[cfunc]
                fn start(features: *const *const c_realchar, len: usize) -> Result<(), c_int> {
                    r#"
                    #include "features.h"
                    "#;

                    r#"
                    return init_features(features, len);
                    "#
                }


                #[cfunc]
                fn stop() -> Result<(), c_int> {
                    r#"
                    #include "features.h"
                    "#;

                    r#"
                    return deinit_features();
                    "#
                }

                list_kernel_features();

                // Drop the features name Vec before yield_!() as Vec<*const c_uchar> is not Send.
                yield_!({
                    let features: Vec<CString> = config.features
                        .iter()
                        .sorted()
                        .map(|s| CString::new(&**s))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| error!("Could not convert String to CString: {err}"))?;
                    let features: Vec<*const c_uchar> = features
                        .iter()
                        .map(|s| s.as_ptr() as *const c_uchar)
                        .collect();

                    // SAFETY: We must not return early here, otherwise we will never run stop(),
                    // which will leave tracepoint probes installed after modexit, leading to a
                    // kernel crash
                    match start(
                        features.as_ptr() as *const *const c_realchar,
                        features.len(),
                    ) {
                        Err(code) => Err(error!("Failed to start legacy C features: {code}")),
                        Ok(()) => Ok(Arc::new(()))
                    }
                });

                stop().map_err(|code| error!("Failed to stop legacy C features: {code}"))
            })
        ))
    },
}
