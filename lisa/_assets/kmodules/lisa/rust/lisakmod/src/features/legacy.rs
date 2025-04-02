/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};
use core::ffi::c_int;

use lisakmod_macros::inlinec::cfunc;

use crate::{
    error::error, features::define_feature, lifecycle::new_lifecycle, runtime::printk::pr_err,
};

define_feature! {
    struct LegacyFeatures,
    name: "__legacy_features",
    visibility: Public,
    Service: (),
    Config: (),
    dependencies: [],
    init: |configs| {
        Ok(new_lifecycle!(|services| {
            #[cfunc]
            fn start() -> Result<(), c_int> {
                r#"
                #include <linux/module.h>
                #include "features.h"
                #include "introspection.h"

                static char *features[MAX_FEATURES];
                static unsigned int features_len = 0;
                module_param_array(features, charp, &features_len, 0);
                MODULE_PARM_DESC(features, "Comma-separated list of features to enable. Available features are printed when loading the module");
                "#;

                r#"
                pr_info("Kernel features detected. This will impact the module features that are available:\n");
                const char *kernel_feature_names[] = {__KERNEL_FEATURE_NAMES};
                const bool kernel_feature_values[] = {__KERNEL_FEATURE_VALUES};
                for (size_t i=0; i < ARRAY_SIZE(kernel_feature_names); i++) {
                    pr_info("  %s: %s\n", kernel_feature_names[i], kernel_feature_values[i] ? "enabled" : "disabled");
                }

                return init_features(features_len ? features : NULL , features_len);
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

            // We must not bail out here, otherwise we will never run stop(), which will leave
            // tracepoint probes installed after modexit, leading to a kernel crash
            if let Err(code) = start() { pr_err!("Failed to start legacy C features: {code}") }
            yield_!(Ok(Arc::new(())));
            stop().map_err(|code| error!("Failed to stop legacy C features: {code}"))
        }))
    },
}
