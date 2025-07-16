/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};

use crate::{
    error::Error,
    features::{
        DependenciesSpec, Feature, FeatureId, FeaturesService, Visibility, all_features,
        register_feature,
    },
    lifecycle::{LifeCycle, new_lifecycle},
    query::query_type,
};

pub struct AllFeatures;

query_type! {
    #[derive(Clone)]
    pub struct AllFeaturesConfig {
        pub best_effort: bool,
    }
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

    fn id(&self) -> FeatureId {
        FeatureId::new::<Self>()
    }

    fn visibility(&self) -> Visibility {
        Visibility::Public
    }

    fn dependencies(&self) -> Vec<FeatureId> {
        default_features().map(|feat| feat.id()).collect()
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

        Ok((
            spec,
            new_lifecycle!(|_| {
                yield_!(Ok(Arc::new(())));
                Ok(())
            }),
        ))
    }
}
register_feature!(AllFeatures);
