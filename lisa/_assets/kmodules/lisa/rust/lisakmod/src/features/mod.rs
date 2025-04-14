/* SPDX-License-Identifier: GPL-2.0 */

pub mod events;
pub mod legacy;
pub mod pixel6;
pub mod tests;
pub mod tracepoint;
pub mod wq;

use alloc::{sync::Arc, vec::Vec};
use core::{
    any::{Any, TypeId, type_name},
    ffi::c_int,
    fmt::Debug,
};

use linkme::distributed_slice;

use crate::{
    error::{Error, MultiResult},
    graph::{Cursor, DfsPostTraversal, Graph, TraversalDirection},
    lifecycle::{self, FinishedKind, LifeCycle, new_lifecycle},
    runtime::{
        printk::{pr_debug, pr_err, pr_info},
        sync::Lock,
    },
    typemap,
};

typemap::make_index!(pub FeaturesConfigIndex, Send);
impl<Feat> typemap::KeyOf<FeaturesConfigIndex> for Feat
where
    Feat: 'static + Feature,
{
    type Value = Vec<<Feat as Feature>::Config>;
}
pub type FeaturesConfig = typemap::TypeMap<FeaturesConfigIndex>;

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
pub struct FeatureId {
    type_id: TypeId,
    type_name: &'static str,
}

impl FeatureId {
    fn new<Feat: 'static + Feature>() -> Self {
        FeatureId {
            type_id: TypeId::of::<Feat>(),
            type_name: type_name::<Feat>(),
        }
    }
}

mod private {
    use alloc::{boxed::Box, format, sync::Arc, vec::Vec};

    use super::*;
    use crate::{
        error::{Error, ResultExt as _, error},
        runtime::sync::{LockdepClass, Mutex},
    };

    pub struct LiveFeature {
        lifecycle: Mutex<Box<dyn Any + Send>>,
        feature: Arc<dyn Feature + Send + Sync>,
        children_config: FeaturesConfig,
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
        fn __configure(
            self: Arc<Self>,
            parents: &mut dyn Iterator<Item = &Result<Option<LiveFeature>, Error>>,
            from_user: &FeaturesConfig,
        ) -> Result<LiveFeature, Error>;

        fn __start(
            &self,
            live: &LiveFeature,
            parents_service: &mut FeaturesService,
            start_children: &mut dyn FnMut(&mut FeaturesService) -> Result<(), Error>,
        ) -> Result<(), Error>;

        fn __stop(
            &self,
            live: &LiveFeature,
            stop_parents: &mut dyn FnMut() -> Result<(), Error>,
        ) -> Result<(), Error>;

        fn __id(&self) -> FeatureId;
    }

    impl<Feat> BlanketFeature for Feat
    where
        Feat: 'static + Feature + Send + Sync,
    {
        fn __configure(
            self: Arc<Self>,
            parents: &mut dyn Iterator<Item = &Result<Option<LiveFeature>, Error>>,
            from_user: &FeaturesConfig,
        ) -> Result<LiveFeature, Error> {
            let name = self.name();
            let mut from_parents = Vec::new();
            for parent in parents {
                match parent {
                    Ok(Some(parent)) => {
                        if let Some(configs) = parent.children_config.get::<Self>() {
                            from_parents.extend(configs.iter())
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        return Err(error!("Could not configure parent of feature {name}"));
                    }
                }
            }
            let empty = Vec::new();
            let from_user = from_user.get::<Self>().unwrap_or(&empty);
            let mut for_us = from_user.iter().chain(from_parents);
            let (children_config, lifecycle) = self
                .configure(&mut for_us)
                .with_context(|| format!("Failed to configure feature {name}"))?;
            let lifecycle = FeatureLifeCycle::<Feat>::new(lifecycle);
            let lifecycle = Mutex::new(
                Box::new(lifecycle) as Box<dyn Any + Send>,
                LockdepClass::new(),
            );

            Ok(LiveFeature {
                feature: self as Arc<dyn Feature + Send + Sync>,
                lifecycle,
                children_config,
            })
        }
        fn __id(&self) -> FeatureId {
            FeatureId::new::<Self>()
        }

        fn __start(
            &self,
            live: &LiveFeature,
            parents_service: &mut FeaturesService,
            start_children: &mut dyn FnMut(&mut FeaturesService) -> Result<(), Error>,
        ) -> Result<(), Error> {
            lock_lifecycle!(Feat, live, guard, lifecycle);

            let mut register_service = |service: &_| {
                parents_service.insert::<Feat>(Arc::clone(service));
            };

            match lifecycle.state() {
                lifecycle::State::Init | lifecycle::State::Finished(_, Ok(_)) => {
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

    pub fn start_features(graph: &Graph<Option<LiveFeature>>) -> Result<(), Error> {
        fn process(
            cursor: &Cursor<'_, Option<LiveFeature>>,
            parents_service: &mut FeaturesService,
        ) -> Result<(), Error> {
            match cursor.value() {
                Some(live) => {
                    let mut start_children = |children_service: &mut _| {
                        cursor
                            .children()
                            .try_for_each(|child| process(&child, children_service))
                            .with_context(|| {
                                format!(
                                    "Error while starting children of feature {}",
                                    &live.feature.name()
                                )
                            })
                    };
                    live.feature
                        .__start(live, parents_service, &mut start_children)
                }
                None => cursor
                    .children()
                    .map(|child| process(&child, &mut FeaturesService::new()))
                    .collect::<MultiResult<(), Error>>()
                    .into_result(),
            }
        }

        let mut parents_service = FeaturesService::new();
        graph
            .roots()
            .map(|root| process(&root, &mut parents_service))
            .collect::<MultiResult<(), Error>>()
            .into_result()
    }

    pub fn stop_features(graph: &Graph<Option<LiveFeature>>) -> Result<(), Error> {
        fn process(cursor: Cursor<Option<LiveFeature>>) -> Result<(), Error> {
            match cursor.value() {
                Some(live) => {
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
                None => cursor
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
    type Config: Send
    where
        Self: Sized;

    fn name(&self) -> &str;
    fn visibility(&self) -> Visibility;

    // This associated method allows getting a FeatureId when specifying the dependencies without
    // having a live instance of the feature.
    fn id() -> FeatureId
    where
        Self: 'static + Sized,
    {
        FeatureId::new::<Self>()
    }

    #[allow(clippy::type_complexity)]
    fn configure(
        &self,
        configs: &mut dyn Iterator<Item = &Self::Config>,
    ) -> Result<
        (
            FeaturesConfig,
            LifeCycle<FeaturesService, Arc<Self::Service>, Error>,
        ),
        Error,
    >
    where
        Self: Sized;

    fn dependencies(&self) -> Vec<FeatureId> {
        Vec::new()
    }
}

pub fn features_lifecycle<Select>(select: Select) -> LifeCycle<(), (), c_int>
where
    Select: Fn(&dyn Feature) -> bool,
{
    let graph: Graph<Arc<dyn Feature>> = Graph::new(
        __FEATURES
            .into_iter()
            .map(|f| f())
            .map(|feature| (feature.__id(), feature.dependencies(), feature)),
    );

    let graph = graph.dfs_map(DfsPostTraversal::new(
        TraversalDirection::FromLeaves,
        |value: Arc<dyn Feature>, mut parents: &mut dyn Iterator<Item = &Option<_>>| {
            // Select a feature if it has been directly selected or if any of its parent has been
            // selected.
            let selected = (&mut parents).any(|parent| parent.is_some()) || select(&*value);
            if selected { Some(value) } else { None }
        },
    ));

    let from_user = &FeaturesConfig::new();
    let configure =
        |feature: Option<Arc<dyn Feature>>,
         parents: &mut dyn Iterator<Item = &Result<Option<private::LiveFeature>, Error>>| {
            match feature {
                Some(feature) => feature.__configure(parents, from_user).map(Some),
                None => Ok(None),
            }
        };
    let graph = graph.dfs_map(DfsPostTraversal::new(
        TraversalDirection::FromLeaves,
        configure,
    ));

    let graph = Into::<Result<_, _>>::into(graph);
    let graph = graph.expect("Error while configuring features");

    new_lifecycle!(|_| {
        if let Err(err) = private::start_features(&graph) {
            pr_err!("Error while starting features: {err:#}");
        }

        yield_!(Ok(()));

        if let Err(err) = private::stop_features(&graph) {
            pr_err!("Error while stopping features: {err:#}");
        }
        Ok(())
    })
}

macro_rules! define_feature {
    (
        $vis:vis struct $type:ident,
        name: $name:expr,
        visibility: $visibility:ident,
        Service: $service:ty,
        Config: $config:ty,
        dependencies: [$($dep:ty),* $(,)?],
        init: $init:expr,
    ) => {
        $vis struct $type;

        const _: () = {
            #[::linkme::distributed_slice($crate::features::__FEATURES)]
            fn register() -> ::alloc::sync::Arc<dyn $crate::features::Feature> {
                ::alloc::sync::Arc::new($type)
            }
        };

        impl $crate::features::Feature for  $type {
            type Service = $service;
            type Config = $config;

            fn name(&self) -> &str {
                $name
            }
            fn visibility(&self) -> $crate::features::Visibility {
                $crate::features::Visibility::$visibility
            }

            fn dependencies(&self) -> Vec<$crate::features::FeatureId> {
                [$(<$dep>::id()),*].into()
            }

            fn configure(
                &self,
                configs: &mut dyn ::core::iter::Iterator<Item = &Self::Config>,
            ) -> ::core::result::Result<
                (
                    $crate::features::FeaturesConfig,
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
