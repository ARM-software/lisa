/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    ptr::NonNull,
};

use lisakmod_macros::inlinec::cfunc;

use crate::{
    error::error,
    features::{FeaturesConfig, define_feature},
    lifecycle::new_lifecycle,
    runtime::printk::pr_err,
};

#[cfunc]
fn features_array() -> Option<NonNull<c_void>> {
    r#"
    #include <linux/module.h>
    #include "features.h"

    static char *features[MAX_FEATURES];
    static unsigned int features_len = 0;
    module_param_array(features, charp, &features_len, 0);
    MODULE_PARM_DESC(features, "Comma-separated list of features to enable. Available features are printed when loading the module");
    "#;

    r#"
    return features;
    "#
}

#[cfunc]
fn features_array_len() -> c_uint {
    r#"
    static unsigned int features_len;
    "#;

    r#"
    return features_len;
    "#
}

pub fn module_param_features() -> Option<impl Iterator<Item = &'static str>> {
    let len = match features_array_len() {
        0 => None,
        x => Some(x),
    }?;

    let ptr = features_array()?;
    let ptr = ptr.as_ptr() as *const *const c_char;

    let slice = unsafe { core::slice::from_raw_parts(ptr, len as usize) };
    Some(slice.iter().map(|s| {
        unsafe { CStr::from_ptr(*s) }
            .to_str()
            .expect("Invalid UTF-8 in feature name")
    }))
}

define_feature! {
    struct LegacyFeatures,
    name: "__legacy_features",
    visibility: Private,
    Service: (),
    Config: (),
    dependencies: [],
    init: |configs| {
        Ok((
            FeaturesConfig::new(),
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
                fn start(features: Option<NonNull<c_void>>, len: c_uint) -> Result<(), c_int> {
                    r#"
                    #include "features.h"
                    "#;

                    r#"
                    return init_features(len ? features : NULL , len);
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

                // We must not bail out here, otherwise we will never run stop(), which will leave
                // tracepoint probes installed after modexit, leading to a kernel crash
                if let Err(code) = start(features_array(), features_array_len()) { pr_err!("Failed to start legacy C features: {code}") }
                yield_!(Ok(Arc::new(())));
                stop().map_err(|code| error!("Failed to stop legacy C features: {code}"))
            })
        ))
    },
}
