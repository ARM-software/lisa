/* SPDX-License-Identifier: GPL-2.0 */

pub mod all;
pub mod events;
pub mod pixel6;
pub mod pmu;
pub mod tests;
pub mod thermal;
pub mod wq;

use alloc::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::{
    any::{Any, TypeId, type_name},
    ffi::{CStr, c_char, c_int},
    fmt,
    fmt::Debug,
    ptr::NonNull,
};

use linkme::distributed_slice;
use lisakmod_macros::inlinec::{c_realchar, cfunc};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, MultiResult, error},
    graph::{Cursor, DfsPostTraversal, Graph, TraversalDirection},
    lifecycle::{self, FinishedKind, LifeCycle, new_lifecycle},
    query::query_type,
    runtime::{
        printk::{pr_debug, pr_info},
        sync::Lock,
    },
    typemap,
};

pub enum DependencySpec<Feat>
where
    Feat: Feature,
{
    Optional {
        configs: Vec<<Feat as Feature>::Config>,
    },
    Mandatory {
        configs: Vec<<Feat as Feature>::Config>,
    },
    Disabled,
}

impl<Feat> Clone for DependencySpec<Feat>
where
    Feat: Feature,
    <Feat as Feature>::Config: Clone,
{
    fn clone(&self) -> DependencySpec<Feat> {
        match self {
            DependencySpec::Optional { configs } => DependencySpec::Optional {
                configs: configs.clone(),
            },
            DependencySpec::Mandatory { configs } => DependencySpec::Mandatory {
                configs: configs.clone(),
            },
            DependencySpec::Disabled => DependencySpec::Disabled,
        }
    }
}

impl<Feat> fmt::Debug for DependencySpec<Feat>
where
    Feat: Feature,
    <Feat as Feature>::Config: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencySpec::Optional { configs } => f
                .debug_struct("DependencySpec::Optional")
                .field("configs", configs)
                .finish(),
            DependencySpec::Mandatory { configs } => f
                .debug_struct("DependencySpec::Mandatory")
                .field("configs", configs)
                .finish(),
            DependencySpec::Disabled => f.debug_struct("Disabled").finish(),
        }
    }
}

typemap::make_index!(pub DependenciesSpecIndex, Send);
impl<Feat> typemap::KeyOf<DependenciesSpecIndex> for Feat
where
    Feat: 'static + Feature,
    <Feat as Feature>::Config: Debug + Clone,
{
    type Value = DependencySpec<Feat>;
}
pub type DependenciesSpec = typemap::TypeMap<DependenciesSpecIndex>;

typemap::make_index!(pub FeaturesServiceIndex, Send);
impl<Feat> typemap::KeyOf<FeaturesServiceIndex> for Feat
where
    Feat: 'static + Feature,
{
    type Value = Arc<<Feat as Feature>::Service>;
}

pub struct FeaturesService {
    map: typemap::TypeMap<FeaturesServiceIndex>,
}

impl FeaturesService {
    #[inline]
    fn new() -> FeaturesService {
        FeaturesService {
            map: typemap::TypeMap::new(),
        }
    }

    #[inline]
    pub fn get<Feat>(&self) -> Option<&<Feat as Feature>::Service>
    where
        Feat: 'static + Feature,
    {
        self.map.get::<Feat>().map(|service| &**service)
    }

    #[inline]
    pub fn insert<Feat>(&mut self, service: Arc<<Feat as Feature>::Service>)
    where
        Feat: 'static + Feature,
    {
        self.map.insert::<Feat>(service)
    }
}

type LifeCycleAlias<Feat> = LifeCycle<FeaturesService, Arc<<Feat as Feature>::Service>, Error>;

// Tie together a LifeCycle for a Feature and its associated Service type to guide type inference
// and allow impl blocks, unlike LifeCycleAlias.
pub struct FeatureLifeCycle<Feat: Feature> {
    inner: LifeCycleAlias<Feat>,
}

impl<Feat: Feature> FeatureLifeCycle<Feat> {
    fn new(lifecycle: LifeCycleAlias<Feat>) -> Self {
        FeatureLifeCycle { inner: lifecycle }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
enum FeatureIdInner {
    Normal {
        type_id: TypeId,
        type_name: &'static str,
    },
    Legacy {
        name: &'static str,
    },
}

#[derive(PartialOrd, Ord, PartialEq, Eq, Debug)]
pub struct FeatureId {
    inner: FeatureIdInner,
}

impl FeatureId {
    fn new<Feat: 'static + Feature>() -> Self {
        FeatureId {
            inner: FeatureIdInner::Normal {
                type_id: TypeId::of::<Feat>(),
                type_name: type_name::<Feat>(),
            },
        }
    }
}

query_type! {
    #[derive(Default, Serialize)]
    pub struct ProvidedFeatureResources {
        ftrace_events: BTreeSet<String>,
    }
}

query_type! {
    #[derive(Default, Serialize)]
    pub struct FeatureResources {
        provided: ProvidedFeatureResources,
    }
}

pub type GenericConfig = serde_json::Value;

mod private {
    use alloc::{boxed::Box, format, sync::Arc, vec::Vec};

    use super::*;
    use crate::{
        error::{Error, ResultExt as _, error},
        runtime::sync::{Mutex, new_static_lockdep_class},
    };

    pub struct LiveFeature {
        lifecycle: Mutex<Box<dyn Any + Send>>,
        feature: Arc<dyn Feature + Send + Sync>,
        children_config: DependenciesSpec,
        // configs are a type-erased Box<Vec<<Feat as Feature>::Config>>
        config: Box<dyn Any + Send>,
    }

    pub enum SelectedReason {
        ByUser,
        AsDep,
    }

    pub enum Selected<T> {
        Selected(T, SelectedReason),
        NotSelected(Arc<dyn Feature + Send + Sync>),
    }

    macro_rules! lock_lifecycle {
        ($Feat:ident, $live:expr, $guard:ident, $lifecycle:ident) => {
            let mut $guard = $live.lifecycle.lock();
            let lifecycle: &mut FeatureLifeCycle<$Feat> = $guard
                .downcast_mut()
                .expect("A LiveFeature for the wrong Feature was given");
            let $lifecycle = &mut lifecycle.inner;
        };
    }

    // Split some methods in a super trait so we can have a blanket implementation for those, so the
    // user only needs to implement the fundamental methods on the concrete types.
    #[diagnostic::on_unimplemented(
        message = "Implement the Feature trait for `{Self}` and it will gain a blanket implementation of BlanketFeature automatically"
    )]
    #[doc(hidden)]
    pub trait BlanketFeature {
        fn __push_no_config(
            &self,
            configs: &mut DependenciesSpec,
            optional: bool,
        ) -> Result<(), Error>;

        fn __push_user_config(
            &self,
            configs: &mut DependenciesSpec,
            config: GenericConfig,
        ) -> Result<(), Error>;

        fn __configure(
            self: Arc<Self>,
            parents: &mut dyn Iterator<Item = &Result<Selected<LiveFeature>, Error>>,
            from_user: &DependenciesSpec,
        ) -> Result<LiveFeature, Error>;

        fn __start(
            &self,
            live: &LiveFeature,
            parent: Option<&LiveFeature>,
            config: &DependenciesSpec,
            selected_reason: &SelectedReason,
            parents_service: &mut FeaturesService,
            start_children: &mut dyn FnMut(&mut FeaturesService) -> Result<(), Error>,
        ) -> Result<(), Error>;

        fn __stop(
            &self,
            live: &LiveFeature,
            stop_parents: &mut dyn FnMut() -> Result<(), Error>,
        ) -> Result<(), Error>;

        fn __config_schema(&self, gen_: &mut SchemaGenerator) -> Schema;
    }

    impl<Feat> BlanketFeature for Feat
    where
        Feat: 'static + Feature + Send + Sync,
        <Feat as Feature>::Config: Debug + for<'a> Deserialize<'a> + Send + Clone,
    {
        fn __config_schema(&self, gen_: &mut SchemaGenerator) -> Schema {
            gen_.subschema_for::<<Feat as Feature>::Config>()
        }

        fn __push_no_config(
            &self,
            configs: &mut DependenciesSpec,
            mandatory: bool,
        ) -> Result<(), Error> {
            let spec = match mandatory {
                true => DependencySpec::Mandatory {
                    configs: Vec::new(),
                },
                false => DependencySpec::Optional {
                    configs: Vec::new(),
                },
            };
            configs.insert::<Self>(spec);
            Ok(())
        }

        fn __push_user_config(
            &self,
            configs: &mut DependenciesSpec,
            config: GenericConfig,
        ) -> Result<(), Error> {
            // Always allow "null" as config, which will translate to no config being pushed on the
            // stack. It is then up to the feature to deal with an empty set of configs and reject
            // it if it does not make sense.
            let config: Option<<Self as Feature>::Config> = serde_json::from_value(config)
                .map_err(|err| error!("Could not load config of feature {}: {err}", self.name()))?;
            match configs.get_mut::<Self>() {
                None => {
                    configs.insert::<Self>(DependencySpec::Mandatory {
                        configs: config.as_slice().into(),
                    });
                }
                Some(spec) => match spec {
                    DependencySpec::Optional {
                        configs: feat_configs,
                    }
                    | DependencySpec::Mandatory {
                        configs: feat_configs,
                    } => {
                        if let Some(config) = config {
                            feat_configs.push(config)
                        }
                    }
                    DependencySpec::Disabled => {}
                },
            }
            Ok(())
        }
        fn __configure(
            self: Arc<Self>,
            parents: &mut dyn Iterator<Item = &Result<Selected<LiveFeature>, Error>>,
            from_user: &DependenciesSpec,
        ) -> Result<LiveFeature, Error> {
            let name = self.name();
            let mut from_parents = Vec::new();
            for parent in parents {
                match parent {
                    Ok(Selected::Selected(parent, _)) => {
                        if let Some(spec) = parent.children_config.get::<Self>() {
                            match spec {
                                DependencySpec::Mandatory { configs }
                                | DependencySpec::Optional { configs } => {
                                    from_parents.extend(configs.iter().cloned())
                                }
                                DependencySpec::Disabled => {}
                            }
                        }
                    }
                    Ok(Selected::NotSelected(_)) => {}
                    Err(_) => {
                        return Err(error!("Could not configure parent of feature {name}"));
                    }
                }
            }
            let from_user = match from_user.get::<Self>() {
                Some(
                    DependencySpec::Mandatory { configs } | DependencySpec::Optional { configs },
                ) => configs,
                _ => &Vec::new(),
            };
            let for_us: Vec<<Self as Feature>::Config> =
                from_user.iter().cloned().chain(from_parents).collect();
            let (children_config, lifecycle) = self
                .configure(&mut for_us.iter())
                .with_context(|| format!("Failed to configure feature {name}"))?;
            let lifecycle = FeatureLifeCycle::<Feat>::new(lifecycle);
            new_static_lockdep_class!(LIVE_FEATURE_LIFECYCLE_LOCKDEP_CLASS);
            let lifecycle = Mutex::new(
                Box::new(lifecycle) as Box<dyn Any + Send>,
                LIVE_FEATURE_LIFECYCLE_LOCKDEP_CLASS.clone(),
            );

            Ok(LiveFeature {
                feature: self as Arc<dyn Feature + Send + Sync>,
                lifecycle,
                children_config,
                config: Box::new(for_us),
            })
        }

        fn __start(
            &self,
            live: &LiveFeature,
            parent: Option<&LiveFeature>,
            config: &DependenciesSpec,
            selected_reason: &SelectedReason,
            parents_service: &mut FeaturesService,
            start_children: &mut dyn FnMut(&mut FeaturesService) -> Result<(), Error>,
        ) -> Result<(), Error> {
            let mut start = move || {
                lock_lifecycle!(Feat, live, guard, lifecycle);

                let mut register_service = |service: &_| {
                    parents_service.insert::<Feat>(Arc::clone(service));
                };

                match lifecycle.state() {
                    lifecycle::State::Init | lifecycle::State::Finished(_, Ok(())) => {
                        let mut children_service = FeaturesService::new();

                        // Release the lock before recursing in the children so that lockdep does not
                        // get upset
                        drop(guard);
                        start_children(&mut children_service)?;

                        // INVARIANT: if a child failed to start, then we do not attempt starting the
                        // current level. That means that a started node is always reachable from the
                        // leaves by following a path of other started nodes. Therefore, a
                        // partially-started graph can always be stopped by traversing it from the
                        // leaves.

                        lock_lifecycle!(Feat, live, guard, lifecycle);

                        pr_info!("Starting feature {}", self.name());
                        let service = lifecycle
                            .start(children_service)
                            .with_context(|| format!("Failed to start feature {}", self.name()))?;
                        pr_debug!("Started feature {}", self.name());

                        register_service(service);
                        Ok(())
                    }
                    lifecycle::State::Started(Ok(service)) => {
                        register_service(service);
                        Ok(())
                    }
                    lifecycle::State::Started(Err(err)) => Err(err),
                    // Disallow re-starting a feature that failed to finish properly, as external
                    // resources may be in an unknown state.
                    lifecycle::State::Finished(_, Err(err)) => Err(err.context(format!(
                        "Cannot restart feature {} as it may be in a broken state",
                        self.name()
                    ))),
                }
            };
            match (selected_reason, config.get::<Self>()) {
                (SelectedReason::ByUser, _)
                | (_, Some(DependencySpec::Mandatory { .. }) | None) => start(),
                (_, Some(DependencySpec::Optional { .. })) => match start() {
                    Ok(()) => Ok(()),
                    Err(err) => {
                        pr_info!(
                            "Error while starting {} as an optional dependency of {}: {err:#}",
                            &live.feature.name(),
                            match parent {
                                Some(parent) => parent.feature.name(),
                                None => "<root>",
                            }
                        );
                        Ok(())
                    }
                },
                (_, Some(DependencySpec::Disabled)) => Ok(()),
            }
        }

        fn __stop(
            &self,
            live: &LiveFeature,
            stop_parents: &mut dyn FnMut() -> Result<(), Error>,
        ) -> Result<(), Error> {
            lock_lifecycle!(Feat, live, guard, lifecycle);

            match lifecycle.state() {
                // We rely on the invariant that if a feature was not started, none of its parent
                // can be started either so we don't need to explore that way.
                lifecycle::State::Started(_) => {
                    // Release the lock before recursing in the parents so that lockdep does not
                    // get upset
                    drop(guard);
                    // Bail out if we cannot stop the parents, as it would be unsafe to stop ourselves if a
                    // parent has failed to stop and might still be relying on us.
                    stop_parents()?;

                    pr_info!("Stopping feature {}", self.name());
                    lock_lifecycle!(Feat, live, guard, lifecycle);
                    lifecycle
                        .stop()
                        .with_context(|| format!("Failed to stop feature {}", self.name()))?;
                    pr_debug!("Stopped feature {}", self.name());
                    Ok(())
                }
                lifecycle::State::Init => Ok(()),
                // If the feature did an early exit, either it was successful or an error was
                // returned, but it will have already been passed back to the caller back then, so
                // make stopping it a successful no-op.
                lifecycle::State::Finished(FinishedKind::Early, _) => Ok(()),
                lifecycle::State::Finished(FinishedKind::Normal, res) => res.clone(),
            }
        }
    }

    pub fn start_features(graph: &Graph<Selected<LiveFeature>>) -> Result<(), Error> {
        fn process(
            parent: Option<&LiveFeature>,
            cursor: &Cursor<'_, Selected<LiveFeature>>,
            parents_service: &mut FeaturesService,
        ) -> Result<(), Error> {
            match cursor.value() {
                Selected::Selected(live, reason) => {
                    let mut start_children = |children_service: &mut _| {
                        cursor.children().try_for_each(|child_cursor| {
                            process(Some(live), &child_cursor, children_service).with_context(
                                || match child_cursor.value() {
                                    Selected::Selected(_, SelectedReason::ByUser) => {
                                        "Error while starting user-requested feature".into()
                                    }
                                    _ => format!(
                                        "Error while starting children of feature {}",
                                        &live.feature.name()
                                    ),
                                },
                            )
                        })
                    };
                    let empty_config = DependenciesSpec::new();
                    let config = match parent {
                        Some(parent) => &parent.children_config,
                        None => &empty_config,
                    };
                    live.feature.__start(
                        live,
                        parent,
                        config,
                        reason,
                        parents_service,
                        &mut start_children,
                    )
                }
                Selected::NotSelected(_) => cursor
                    .children()
                    .map(|child| process(parent, &child, &mut FeaturesService::new()))
                    .collect::<MultiResult<(), Error>>()
                    .into_result(),
            }
        }

        let mut parents_service = FeaturesService::new();
        graph
            .roots()
            .map(|root| process(None, &root, &mut parents_service))
            .collect::<MultiResult<(), Error>>()
            .into_result()
    }

    pub fn stop_features(graph: &Graph<Selected<LiveFeature>>) -> Result<(), Error> {
        fn process(cursor: Cursor<Selected<LiveFeature>>) -> Result<(), Error> {
            match cursor.value() {
                Selected::Selected(live, _) => {
                    let mut stop_parents = || {
                        cursor
                            .parents()
                            .map(process)
                            // Stop all the features we can and only then propagate all the issues we
                            // encountered.
                            .collect::<MultiResult<(), Error>>()
                            .into_result()
                            .with_context(|| {
                                format!(
                                    "Error while stopping parents of feature {}",
                                    &live.feature.name()
                                )
                            })
                    };
                    live.feature.__stop(live, &mut stop_parents)
                }
                Selected::NotSelected(_) => cursor
                    .parents()
                    .map(process)
                    .collect::<MultiResult<(), Error>>()
                    .into_result(),
            }
        }

        graph
            .leaves()
            .map(process)
            .collect::<MultiResult<(), Error>>()
            .into_result()
    }
}

pub trait Feature: Send + Sync + private::BlanketFeature {
    // Add Self: Sized bound for all associated types so that Feature is dyn-usable.
    type Service: Send + Sync + Debug
    where
        Self: Sized;
    type Config: Send + JsonSchema
    where
        Self: Sized;

    fn name(&self) -> &str;
    fn visibility(&self) -> Visibility;

    fn id(&self) -> FeatureId;

    #[allow(clippy::type_complexity)]
    fn configure(
        &self,
        configs: &mut dyn Iterator<Item = &Self::Config>,
    ) -> Result<
        (
            DependenciesSpec,
            LifeCycle<FeaturesService, Arc<Self::Service>, Error>,
        ),
        Error,
    >
    where
        Self: Sized;

    #[inline]
    fn dependencies(&self) -> Vec<FeatureId> {
        Vec::new()
    }

    #[inline]
    fn resources(&self) -> FeatureResources {
        Default::default()
    }
}

fn push_user_configs(
    features: &[Arc<dyn Feature>],
    stack: &mut DependenciesSpec,
    configs: Vec<BTreeMap<String, GenericConfig>>,
) -> Result<(), Error> {
    let features_by_name: BTreeMap<&str, &dyn Feature> =
        features.iter().map(|feat| (feat.name(), &**feat)).collect();

    for config in configs {
        for (feat_name, feat_config) in config {
            match features_by_name.get(&*feat_name) {
                None => Err(error!("Unknown feature: {feat_name}")),
                Some(feat) => match feat.visibility() {
                    Visibility::Public => feat.__push_user_config(stack, feat_config),
                    Visibility::Private => Err(error!(
                        "Feature \"{feat_name}\" cannot be explicitly enabled as it is private"
                    )),
                },
            }?;
        }
    }
    Ok(())
}

pub fn features_lifecycle<Select>(
    select: Select,
    base_config: DependenciesSpec,
    configs: Vec<BTreeMap<String, GenericConfig>>,
) -> Result<LifeCycle<(), (), Error>, Error>
where
    Select: Fn(&dyn Feature) -> bool,
{
    let features: Vec<_> = all_features().collect();
    let graph: Graph<Arc<dyn Feature>> = Graph::new(
        features
            .iter()
            .map(|feature| (feature.id(), feature.dependencies(), Arc::clone(feature))),
    );

    let graph = graph.dfs_map(DfsPostTraversal::new(
        TraversalDirection::FromLeaves,
        |value: Arc<dyn Feature>, mut parents: &mut dyn Iterator<Item = &private::Selected<_>>| {
            if select(&*value) {
                private::Selected::Selected(value, private::SelectedReason::ByUser)
            } else if (&mut parents)
                .any(|parent| matches!(parent, private::Selected::Selected(_, _)))
            {
                private::Selected::Selected(value, private::SelectedReason::AsDep)
            } else {
                private::Selected::NotSelected(value)
            }
        },
    ));

    let mut stacked_configs = base_config;
    push_user_configs(&features, &mut stacked_configs, configs)?;
    let configure = |feature: private::Selected<Arc<dyn Feature>>,
                     parents: &mut dyn Iterator<
        Item = &Result<private::Selected<private::LiveFeature>, Error>,
    >| {
        match feature {
            private::Selected::Selected(feature, reason) => feature
                .__configure(parents, &stacked_configs)
                .map(|live| private::Selected::Selected(live, reason)),
            private::Selected::NotSelected(feature) => Ok(private::Selected::NotSelected(feature)),
        }
    };
    let graph = graph.dfs_map(DfsPostTraversal::new(
        TraversalDirection::FromLeaves,
        configure,
    ));

    let graph = Into::<Result<_, _>>::into(graph)?;

    Ok(new_lifecycle!(|_| {
        // If this fail, we simply yield the error instead of doing an early return. This way, we
        // ensure the 2nd part of the lifecycle will execute, so we get to stop the partially
        // started feature graph (some features may have not failed to start and need to be
        // stopped).
        yield_!(private::start_features(&graph));
        private::stop_features(&graph)
    }))
}

macro_rules! register_feature {
    ($type:ident) => {
        const _: () = {
            #[::linkme::distributed_slice($crate::features::__FEATURES)]
            fn register() -> ::alloc::sync::Arc<dyn $crate::features::Feature> {
                ::alloc::sync::Arc::new($type)
            }
        };
    };
}
#[allow(unused_imports)]
pub(crate) use register_feature;

macro_rules! define_feature {
    (
        $vis:vis struct $type:ident,
        name: $name:expr,
        visibility: $visibility:ident,
        Service: $service:ty,
        Config: $config:ty,
        dependencies: [$($dep:ty),* $(,)?],
        resources: $resources:expr,
        init: $init:expr,
    ) => {
        $vis struct $type;
        impl $type {
            pub const NAME: &'static str = $name;
            pub const VISBILITY: $crate::features::Visibility =
                $crate::features::Visibility::$visibility;
        }
        $crate::features::register_feature!($type);

        impl $crate::features::Feature for $type {
            type Service = $service;
            type Config = $config;

            fn name(&self) -> &str {
                Self::NAME
            }

            fn visibility(&self) -> $crate::features::Visibility {
                Self::VISBILITY
            }

            fn id(&self) -> $crate::features::FeatureId {
                $crate::features::FeatureId::new::<Self>()
            }

            fn dependencies(&self) -> Vec<$crate::features::FeatureId> {
                [$(
                    $crate::features::FeatureId::new::<$dep>()
                ),*].into()
            }

            fn resources(&self) -> $crate::features::FeatureResources {
                ($resources)()
            }

            fn configure(
                &self,
                configs: &mut dyn ::core::iter::Iterator<Item = &Self::Config>,
            ) -> ::core::result::Result<
                (
                    $crate::features::DependenciesSpec,
                    $crate::lifecycle::LifeCycle<
                        $crate::features::FeaturesService,
                        ::alloc::sync::Arc<Self::Service>,
                        $crate::error::Error
                    >,
                ),
                $crate::error::Error,
            >
            {
                $init(configs)
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use define_feature;

#[distributed_slice]
pub static __FEATURES: [fn() -> Arc<dyn Feature>];

pub fn all_features() -> impl Iterator<Item = Arc<dyn Feature>> {
    __FEATURES.into_iter().map(|f| f()).chain(legacy_features())
}

struct LegacyFeature {
    name: &'static str,
}

impl Feature for LegacyFeature {
    type Service = ();
    type Config = ();

    fn name(&self) -> &str {
        self.name
    }
    fn visibility(&self) -> Visibility {
        Visibility::Public
    }

    fn id(&self) -> FeatureId {
        FeatureId {
            inner: FeatureIdInner::Legacy { name: self.name },
        }
    }

    #[inline]
    fn resources(&self) -> FeatureResources {
        match self.name.strip_prefix("event__") {
            Some(event) => FeatureResources {
                provided: ProvidedFeatureResources {
                    ftrace_events: [event].into_iter().map(Into::into).collect(),
                },
            },
            None => Default::default(),
        }
    }

    #[allow(clippy::type_complexity)]
    fn configure(
        &self,
        _configs: &mut dyn Iterator<Item = &Self::Config>,
    ) -> Result<
        (
            DependenciesSpec,
            LifeCycle<FeaturesService, Arc<Self::Service>, Error>,
        ),
        Error,
    > {
        let name = self.name;
        let name = CString::new(name)
            .map_err(|err| error!("Could not convert feature name to CString: {err}"))?;
        Ok((
            Default::default(),
            new_lifecycle!(|_| {
                #[cfunc]
                fn start(feature: *const c_realchar) -> Result<(), c_int> {
                    r#"
                    #include "features.h"
                    "#;

                    r#"
                    return init_feature(feature);
                    "#
                }

                #[cfunc]
                fn stop(feature: *const c_realchar) -> Result<(), c_int> {
                    r#"
                    #include "features.h"
                    "#;

                    r#"
                    return deinit_feature(feature);
                    "#
                }

                yield_!({
                    // SAFETY: We must not return early here, otherwise we will never run stop(),
                    // which will leave tracepoint probes installed after modexit, leading to a
                    // kernel crash
                    match start(name.as_ptr() as *const c_realchar) {
                        Err(code) => {
                            Err(error!("Failed to start legacy C features {name:?}: {code}"))
                        }
                        Ok(()) => Ok(Arc::new(())),
                    }
                });

                stop(name.as_ptr() as *const c_realchar)
                    .map_err(|code| error!("Failed to stop legacy C feature {name:?}: {code}"))
            }),
        ))
    }
}

fn legacy_features() -> impl Iterator<Item = Arc<dyn Feature>> {
    fn names() -> impl Iterator<Item = &'static str> {
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

    names().map(|name| Arc::new(LegacyFeature { name }) as Arc<dyn Feature>)
}
