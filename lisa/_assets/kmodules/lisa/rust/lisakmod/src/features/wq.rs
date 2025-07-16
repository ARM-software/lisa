/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};

pub use crate::runtime::wq::*;
use crate::{
    error::Error,
    features::{DependenciesSpec, define_feature},
    lifecycle::new_lifecycle,
};

#[derive(Debug)]
pub struct WqService {
    wq: Wq,
}

impl WqService {
    fn new() -> Result<WqService, Error> {
        Ok(WqService {
            wq: Wq::new("lisa_features")?,
        })
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
    resources: Default::default,
    init: |_| {
        Ok((
            DependenciesSpec::new(),
            new_lifecycle!(|_| {
                yield_!(Ok(Arc::new(WqService::new()?)));
                Ok(())
            })
        ))
    },
}
