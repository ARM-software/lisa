/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};
use core::ffi::c_int;

use crate::{
    error::Error,
    features::{
        Feature, FeaturesConfig, FeaturesService, Visibility, define_feature,
        events::define_event_feature, features_lifecycle,
    },
    lifecycle::{LifeCycle, new_lifecycle},
    runtime::printk::pr_info,
};

// define_feature! {
    // struct Feature1,
    // name: "feature1",
    // visibility: Public,
    // Service: (),
    // Config: (),
    // dependencies: [Feature2],
    // init: |configs| {
        // Ok(new_lifecycle!(|services| {
            // let services: FeaturesService = services;
            // pr_info!("FEATURE1 start");
            // let feature2_service = services.get::<Feature2>();
            // pr_info!("FEATURE2 service: {feature2_service:?}");
            // yield_!(Ok(Arc::new(())));
            // pr_info!("FEATURE1 stop");
            // Ok(())
        // }))
    // },
// }

// #[derive(Debug)]
// struct Feature2Service {
    // a: u32,
// }

// define_feature! {
    // struct Feature2,
    // name: "feature2",
    // visibility: Public,
    // Service: Feature2Service,
    // Config: (),
    // dependencies: [],
    // init: |configs| {
        // Ok(new_lifecycle!(|services| {
            // pr_info!("FEATURE2 start");
            // let service = Feature2Service { a: 42 };
            // yield_!(Ok(Arc::new(service)));
            // pr_info!("FEATURE2 stop");
            // Ok(())
        // }))
    // },
// }

// define_event_feature!(struct myevent);

pub fn module_main() -> LifeCycle<(), (), c_int> {
    let select = |feature: &dyn Feature| feature.visibility() == Visibility::Public;
    // let select = |feature: &dyn Feature| feature.name() == "feature2";

    features_lifecycle(select)
}
