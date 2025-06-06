/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, ResultExt as _, error},
    features::{
        DependenciesSpec, DependencySpec, Feature, FeatureResources, GenericConfig,
        all_features, features_lifecycle,
        legacy::{LegacyConfig, LegacyFeatures},
    },
    init::State,
    runtime::{
        sync::{Lock as _, Mutex, new_static_lockdep_class},
        sysfs::{BinFile, BinROContent, BinRWContent, Folder},
        wq::new_attached_work_item,
    },
    version::module_version,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum PopConfigN {
    All,
    #[serde(untagged)]
    N(usize),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Query {
    PushConfig(BTreeMap<String, GenericConfig>),
    PopConfig { n: PopConfigN },
    GetConfig,
    GetVersion,
    GetResources,
    StartFeatures,
    StopFeatures,
    CloseSession,
}

impl Query {
    fn execute(
        self,
        state: &Arc<State>,
        session: &mut QuerySession,
    ) -> Result<QuerySuccess, Error> {
        match self {
            Query::CloseSession => {
                // The eof_read() handler of the BinROContent will take care of destroying the
                // session when the userspace has finished reading the file
                session.finalized = true;
                Ok(QuerySuccess::None)
            }
            Query::GetVersion => Ok(QuerySuccess::GetVersion {
                checksum: module_version().into(),
            }),
            Query::PushConfig(config) => state.push_config(config).map(|()| QuerySuccess::None),
            Query::PopConfig { n } => match n {
                PopConfigN::N(n) => state.pop_configs(n),
                PopConfigN::All => state.pop_all_configs(),
            }
            .map(|i| QuerySuccess::PopConfig { remaining: i }),
            Query::StartFeatures => {
                let mut stack = state.config_stack()?;
                let to_enable: BTreeSet<_> = stack
                    .iter()
                    .flat_map(|config| config.keys().cloned())
                    .collect();

                // Split the features between the legacy ones and the non-legacy ones, so that
                // legacy features can be asked for the same way as non-legacy features. This
                // allows porting the legacy features in the future without breaking compat with
                // old configs.
                let legacy_features: BTreeSet<_> =
                    crate::features::legacy::legacy_features().collect();
                let (legacy_features, mut features): (BTreeSet<_>, BTreeSet<_>) = to_enable
                    .into_iter()
                    .partition(|name| legacy_features.contains(&**name));

                if !legacy_features.is_empty() {
                    features.insert(LegacyFeatures::NAME.into());
                }

                // Remove the legacy features from the config, as they can't have any config option
                // and to allow strict checking of the content against the features that do exist.
                for config in &mut stack {
                    for legacy_feature in &legacy_features {
                        config.remove(legacy_feature);
                    }
                }

                let mut legacy_config = LegacyConfig::new();
                legacy_config.features.extend(legacy_features);
                let mut base_config = DependenciesSpec::new();
                base_config.insert::<LegacyFeatures>(DependencySpec::Mandatory {
                    configs: vec![legacy_config],
                });

                let select = |feature: &dyn Feature| match feature.name() {
                    name if features.contains(name) => name != LegacyFeatures::NAME,
                    _ => false,
                };

                state.restart(move || features_lifecycle(select, base_config, stack))?;
                Ok(QuerySuccess::None)
            }
            Query::StopFeatures => state.stop().map(|()| QuerySuccess::None),
            Query::GetConfig => {
                let stack = state.config_stack()?;
                Ok(QuerySuccess::GetConfig { config: stack })
            }
            Query::GetResources => Ok(QuerySuccess::GetResources {
                features: all_features()
                    .map(|feature| (feature.name().into(), feature.resources()))
                    .collect(),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QuerySuccess {
    PopConfig {
        remaining: usize,
    },
    GetConfig {
        config: Vec<BTreeMap<String, GenericConfig>>,
    },
    GetVersion {
        checksum: String,
    },
    GetResources {
        features: BTreeMap<String, FeatureResources>,
    },
    #[serde(untagged)]
    None,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QueryResult {
    Success(QuerySuccess),
    Error(Error),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum QueriesResult<E> {
    Executed(Vec<QueryResult>),
    Error(E),
}

impl<E> QueriesResult<E> {
    fn to_json_string(&self) -> String
    where
        E: Serialize,
    {
        let mut s = match serde_json::to_string(self) {
            Ok(s) => s,
            // This should always serialize without any error
            Err(err) => serde_json::to_string(&QueriesResult::Error(format!(
                "Could not serialize the query result to JSON: {err}"
            )))
            // Last resort
            .expect("Could not serialize the query result to JSON"),
        };
        s.push('\n');
        s
    }
}

pub type SessionId = u64;

// Do not allow even reading for anyone else than the owner, as reading triggers the
// execution of the query.
const MODE: u16 = 0o600;

pub struct QuerySession {
    #[allow(dead_code)]
    root: Folder,
    #[allow(dead_code)]
    execute_file: BinFile<BinROContent>,
    id: SessionId,
    finalized: bool,
}

impl QuerySession {
    pub fn new(root: &mut Folder, state: Arc<State>, id: SessionId) -> Result<QuerySession, Error> {
        let name = Self::__name(id);
        let mut root = Folder::new(root, &name)?;
        let query_file = BinFile::new(&mut root, "query", 0o644, 1024 * 1024, BinRWContent::new())?;

        let execute = {
            let state = Arc::clone(&state);
            move || {
                let res: Result<Vec<_>, Error> = query_file.ops().with_content(|query| {
                    let query = core::str::from_utf8(query)
                        .map_err(|err| error!("Could not interpret query as UTF8: {err}"))?;
                    let queries: Vec<Query> = serde_json::from_str(query)
                        .map_err(|err| error!("Could not parse query: {err}"))?;
                    Ok(queries
                        .into_iter()
                        .map(|query| {
                            match state.with_session(id, |session| {
                                if session.finalized {
                                    QueryResult::Error(error!(
                                        "Session ID {id} was already finalized"
                                    ))
                                } else {
                                    match query.execute(&state, session) {
                                        Ok(x) => QueryResult::Success(x),
                                        Err(err) => QueryResult::Error(err),
                                    }
                                }
                            }) {
                                Err(err) => QueryResult::Error(err),
                                Ok(res) => res,
                            }
                        })
                        .collect())
                });
                let res = match res {
                    Ok(xs) => QueriesResult::Executed(xs),
                    Err(err) => QueriesResult::Error(err),
                };
                let s = res.to_json_string();
                s.into_bytes()
            }
        };
        let execute_file = BinFile::new(
            &mut root,
            "execute",
            // Do not allow even reading for anyone else than the owner, as reading triggers the
            // execution of the query.
            MODE,
            1024 * 1024,
            BinROContent::new(execute, move || {
                match state.with_session(id, |session| session.finalized) {
                    Ok(true) => {
                        let wq = state.wq();
                        let state = Arc::clone(&state);
                        // It is not possible to remove kobjects from within the read handler (show() sysfs
                        // callback). All that could be done inline here is make the sysfs file/folder
                        // invisible but deallocation must take place somewhere else, so we use a workqueue
                        // for that.
                        new_attached_work_item!(
                            wq,
                            move |_work| {
                                state.close_session(id);
                            },
                            |work| {
                                work.enqueue(0);
                            }
                        );
                    }
                    Ok(false) => {}
                    err @ Err(_) => err.print_err(),
                }
            }),
        )?;

        Ok(QuerySession {
            root,
            execute_file,
            id,
            finalized: false,
        })
    }

    fn __name(id: SessionId) -> String {
        format!("{id}")
    }

    pub fn name(&self) -> String {
        Self::__name(self.id)
    }
}

pub struct QueryService {
    #[allow(dead_code)]
    root: Arc<Mutex<Folder>>,
    #[allow(dead_code)]
    new_session_file: BinFile<BinROContent>,
    #[allow(dead_code)]
    state: Arc<State>,
}

impl QueryService {
    pub fn new(state: Arc<State>) -> Result<QueryService, Error> {
        let mut root = Folder::sysfs_module_root();
        new_static_lockdep_class!(QUERIES_FOLDER_LOCKDEP_CLASS);
        let root = Arc::new(Mutex::new(
            Folder::new(&mut root, "queries")?,
            QUERIES_FOLDER_LOCKDEP_CLASS.clone(),
        ));

        let new_session = {
            let root = Arc::clone(&root);
            let state = Arc::clone(&state);
            move || {
                let name = state
                    .new_session(&mut root.lock(), Arc::clone(&state))
                    .expect("Could not create query session");
                name.into_bytes()
            }
        };
        let max_size = 1024 * 1024;
        let new_session_file = BinFile::new(
            &mut root.lock(),
            "new_session",
            // Do not allow even reading for anyone else than the owner, as reading triggers the
            // execution of the query.
            MODE,
            max_size,
            BinROContent::new(new_session, || {}),
        )?;

        Ok(QueryService {
            root,
            new_session_file,
            state,
        })
    }
}
