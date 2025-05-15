/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};

pub use crate::runtime::tracepoint::*;
use crate::{
    features::{DependenciesSpec, define_feature},
    lifecycle::new_lifecycle,
};

#[derive(Debug)]
pub struct TracepointService {
    dropper: ProbeDropper,
}

impl TracepointService {
    fn new() -> TracepointService {
        TracepointService {
            dropper: ProbeDropper::new(),
        }
    }

    pub fn probe_dropper(&self) -> &ProbeDropper {
        &self.dropper
    }
}

define_feature! {
    pub struct TracepointFeature,
    name: "__tp",
    visibility: Private,
    Service: TracepointService,
    Config: (),
    dependencies: [],
    init: |configs| {
        Ok((
            DependenciesSpec::new(),
            new_lifecycle!(|services| {
                yield_!(Ok(Arc::new(TracepointService::new())));
                Ok(())
            })
        ))
    },
}
