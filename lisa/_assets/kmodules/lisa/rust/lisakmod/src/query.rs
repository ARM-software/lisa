/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::String,
    sync::Arc,
    vec::Vec,
};

use schemars::{Schema, SchemaGenerator, json_schema, schema_for};
use serde::Serialize;

use crate::{
    error::{Error, ResultExt as _, error},
    features::{
        DependenciesSpec, Feature, FeatureResources, GenericConfig, Visibility, all_features,
        features_lifecycle,
    },
    init::State,
    runtime::{
        sync::{Lock as _, Mutex, new_static_lockdep_class},
        sysfs::{BinFile, BinROContent, BinRWContent, Folder},
        wq::new_attached_work_item,
    },
    version::module_version,
};

macro_rules! query_type {
    ($($tt:tt)*) => {
        #[derive(Debug, ::serde::Deserialize, ::schemars::JsonSchema)]
        #[serde(rename_all = "kebab-case")]
        $($tt)*
    }
}
pub(crate) use query_type;

query_type! {
    enum PopConfigN {
        All,
        #[serde(untagged)]
        N(usize),
    }
}

fn push_config_schema(gen_: &mut SchemaGenerator) -> Schema {
    let mut schema = json_schema!({
        "type": "object",
        "additionalProperties": false,
        "properties": {}
    });
    match schema.get_mut("properties") {
        Some(serde_json::Value::Object(properties)) => {
            let features = all_features().filter(|feat| feat.visibility() == Visibility::Public);
            for feature in features {
                let val = feature.__config_schema(gen_);
                let val = serde_json::to_value(val).unwrap();
                properties.insert(feature.name().into(), val);
            }
        }
        _ => unreachable!(),
    }
    schema
}

query_type! {
    #[derive(Clone, Serialize)]
    pub struct ConfigStackItem {
        #[schemars(schema_with = "push_config_schema")]
        config: BTreeMap<String, GenericConfig>,
        enable_features: BTreeSet<String>,
    }
}

query_type! {
    enum Query {
        PushFeaturesConfig(ConfigStackItem),
        PopFeaturesConfig { n: PopConfigN },
        GetFeaturesConfig,
        GetVersion,
        GetResources,
        GetSchemas,
        StartFeatures,
        StopFeatures,
        CloseSession,
    }
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
            Query::PushFeaturesConfig(query) => {
                state.push_config(query).map(|()| QuerySuccess::None)
            }
            Query::PopFeaturesConfig { n } => match n {
                PopConfigN::N(n) => state.pop_configs(n),
                PopConfigN::All => state.pop_all_configs(),
            }
            .map(|i| QuerySuccess::PopFeaturesConfig { remaining: i }),
            Query::StartFeatures => {
                let stack = state.config_stack()?;

                let allowed_features: BTreeMap<String, _> = all_features()
                    .map(|feature| (feature.name().into(), feature))
                    .collect();
                let features: BTreeSet<_> = stack
                    .iter()
                    .flat_map(|config| config.enable_features.clone())
                    .map(|name| match allowed_features.get(&name) {
                        Some(feature) => {
                            if feature.visibility() == Visibility::Public {
                                Ok(name)
                            } else {
                                Err(error!("Cannot enable private feature: {name}"))
                            }
                        }
                        None => Err(error!("Cannot enable inexistent feature: {name}")),
                    })
                    .collect::<Result<_, _>>()?;

                let configs: Vec<_> = stack.iter().map(|config| config.config.clone()).collect();

                let base_config = DependenciesSpec::new();
                let select = |feature: &dyn Feature| features.contains(feature.name());
                state.restart(move || features_lifecycle(select, base_config, configs))?;
                Ok(QuerySuccess::None)
            }
            Query::StopFeatures => state.stop().map(|()| QuerySuccess::None),
            Query::GetFeaturesConfig => {
                let stack = state.config_stack()?;
                Ok(QuerySuccess::GetFeaturesConfig(stack))
            }
            Query::GetResources => Ok(QuerySuccess::GetResources {
                features: all_features()
                    .map(|feature| (feature.name().into(), feature.resources()))
                    .collect(),
            }),
            Query::GetSchemas => Ok(QuerySuccess::GetSchemas {
                query: schema_for!(Query),
            }),
        }
    }
}

query_type! {
    #[derive(Serialize)]
    enum QuerySuccess {
        PopFeaturesConfig {
            remaining: usize,
        },
        GetFeaturesConfig(Vec<ConfigStackItem>),
        GetVersion {
            checksum: String,
        },
        GetResources {
            features: BTreeMap<String, FeatureResources>,
        },
        GetSchemas {
            query: Schema,
        },
        #[serde(untagged)]
        None,
    }
}

query_type! {
    #[derive(Serialize)]
    enum QueryResult {
        Success(QuerySuccess),
        Error(Error),
    }
}

query_type! {
    #[derive(Serialize)]
    enum QueriesResult<E> {
        Executed(Vec<QueryResult>),
        Error(E),
    }
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
