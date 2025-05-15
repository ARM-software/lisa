/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec, vec::Vec};

use itertools::Itertools as _;
use serde::{Deserialize, Serialize};

use crate::{
    error::Error,
    features::{
        DependenciesSpec, DependencySpec, Feature, FeatureId, FeaturesService, Visibility,
        all_features,
        legacy::{LegacyConfig, LegacyFeatures},
        register_feature,
    },
    lifecycle::{LifeCycle, new_lifecycle},
};

pub struct AllFeatures;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "kebab-case")]
pub struct AllFeaturesConfig {
    pub best_effort: bool,
}

impl AllFeaturesConfig {
    fn merge<'a, I>(mut iter: I) -> AllFeaturesConfig
    where
        I: Iterator<Item = &'a Self>,
    {
        AllFeaturesConfig {
            best_effort: iter.all(|conf| conf.best_effort),
        }
    }
}

impl AllFeatures {
    pub const NAME: &'static str = "all";
}

fn default_features() -> impl Iterator<Item = Arc<dyn Feature>> {
    all_features().filter(|feat| {
        (feat.name() != AllFeatures::NAME) && (feat.visibility() == Visibility::Public)
    })
}

impl Feature for AllFeatures {
    type Service = ();
    type Config = AllFeaturesConfig;

    fn name(&self) -> &str {
        Self::NAME
    }

    fn visibility(&self) -> Visibility {
        Visibility::Public
    }

    fn dependencies(&self) -> Vec<FeatureId> {
        default_features().map(|feat| feat.__id()).collect()
    }

    fn configure(
        &self,
        configs: &mut dyn Iterator<Item = &Self::Config>,
    ) -> Result<
        (
            DependenciesSpec,
            LifeCycle<FeaturesService, Arc<Self::Service>, Error>,
        ),
        Error,
    > {
        let config = AllFeaturesConfig::merge(configs);
        let mut spec = DependenciesSpec::new();
        let mandatory = !config.best_effort;
        for feat in default_features() {
            feat.__push_no_config(&mut spec, mandatory)?;
        }

        let legacy_config = LegacyConfig::new_all();
        let legacy_spec = match mandatory {
            true => DependencySpec::Mandatory {
                configs: vec![legacy_config],
            },
            false => DependencySpec::Optional {
                configs: vec![legacy_config],
            },
        };
        spec.insert::<LegacyFeatures>(legacy_spec);

        Ok((
            spec,
            new_lifecycle!(|services| {
                yield_!(Ok(Arc::new(())));
                Ok(())
            }),
        ))
    }
}
register_feature!(AllFeatures);
