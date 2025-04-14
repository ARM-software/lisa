/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};

pub use crate::runtime::wq::*;
use crate::{
    features::{FeaturesConfig, define_feature},
    lifecycle::new_lifecycle,
};

#[derive(Debug)]
pub struct WqService {
    wq: Wq,
}

impl WqService {
    fn new() -> WqService {
        WqService { wq: Wq::new() }
    }

    pub fn wq(&self) -> &Wq {
        &self.wq
    }
}

define_feature! {
    pub struct WqFeature,
    name: "__wq",
    visibility: Private,
    Service: WqService,
    Config: (),
    dependencies: [],
    init: |configs| {
        Ok((
            FeaturesConfig::new(),
            new_lifecycle!(|services| {
                yield_!(Ok(Arc::new(WqService::new())));
                Ok(())
            })
        ))
    },
}
