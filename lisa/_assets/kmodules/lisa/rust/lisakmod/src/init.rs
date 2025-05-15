/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{
    boxed::Box,
    collections::{BTreeMap, btree_map},
    string::String,
    sync::Arc,
    vec,
    vec::Vec,
};
use core::{
    ffi::c_int,
    ops::DerefMut,
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
};

use lisakmod_macros::inlinec::cfunc;

use crate::{
    error::{Error, error},
    features::{
        DependenciesSpec, DependencySpec, Feature, GenericConfig,
        all::{AllFeatures, AllFeaturesConfig},
        features_lifecycle,
    },
    lifecycle::{LifeCycle, new_lifecycle},
    query::{QueryService, QuerySession, SessionId},
    runtime::{
        printk::{pr_err, pr_info},
        sync::{Lock as _, LockdepClass, Mutex},
        sysfs::Folder,
        wq::Wq,
    },
    version::module_version,
};

// use alloc::{collections::BTreeMap, string::String, sync::Arc, vec, vec::Vec};
// use core::{
//     ffi::{c_int, c_ulong, c_void},
//     ops::DerefMut,
// };
//
// use lisakmod_macros::inlinec::cfunc;
//
// use crate::{
//     error::{Error, error},
//     features::{
//         DependenciesSpec, DependencySpec, Feature, FeaturesService, GenericConfig,
//         all::{AllFeatures, AllFeaturesConfig},
//         define_feature, features_lifecycle,
//         tracepoint::TracepointFeature,
//         wq::WqFeature,
//     },
//     lifecycle::{LifeCycle, new_lifecycle},
//     query::QueryService,
//     runtime::{
//         printk::{pr_err, pr_info},
//         sync::{Lock as _, LockdepClass, Mutex},
//     },
//     version::module_version,
// };
// define_feature! {
//     struct Feature1,
//     name: "feature1",
//     visibility: Public,
//     Service: (),
//     Config: (),
//     dependencies: [Feature2, TracepointFeature, WqFeature],
//     init: |configs| {
//         Ok((
//             DependenciesSpec::new(),
//             new_lifecycle!(|services| {
//                 let services: FeaturesService = services;
//                 let dropper = services.get::<TracepointFeature>()
//                     .expect("Could not get service for TracepointFeature")
//                     .probe_dropper();
//                 pr_info!("FEATURE1 start");
//
//                 use crate::runtime::tracepoint::Tracepoint;
//                 use core::sync::atomic::{AtomicUsize, Ordering};
//
//                 // let x = AtomicUsize::new(0);
//                 // let probe = new_probe!(
//                     // &dropper,
//                     // (_preempt: bool, _prev: *const c_void, _next:* const c_void, _prev_state: c_ulong) {
//                         // let x = x.fetch_add(1, Ordering::SeqCst);
//                         // crate::runtime::printk::pr_info!("SCHED_SWITCH {x}");
//                     // }
//                 // );
//
//                 let tp = unsafe {
//                     Tracepoint::<(bool, *const c_void, * const c_void, c_ulong)>::lookup("sched_switch").expect("tp not found")
//                 };
//
//                 // let registered = tp.register_probe(&probe);
//                 // drop(probe);
//                 // drop(dropper);
//                 // drop(registered);
//                 //
//
//                 use crate::runtime::wq;
//                 let wq = services.get::<WqFeature>()
//                     .expect("Could not get service for WqFeature")
//                     .wq();
//
//                 let work_item_n = AtomicUsize::new(1);
//                 let work_item = wq::new_work_item!(wq, move |work| {
//                     pr_info!("FROM WORKER");
//                     let n = work_item_n.fetch_add(1, Ordering::SeqCst);
//                     if n < 10 {
//                         work.enqueue(10);
//                     }
//                 });
//                 work_item.enqueue(0);
//                 msleep(100);
//                 // work_item.enqueue(0);
//
//
//                 use core::ffi::CStr;
//                 let f = crate::runtime::traceevent::new_event! {
//                     lisa__myevent2,
//                     fields: {
//                         field1: u8,
//                         field3: &CStr,
//                         field4: &str,
//                         field2: u64,
//                     }
//                 }?;
//
//                 use lisakmod_macros::inlinec::cfunc;
//                 #[cfunc]
//                 fn msleep(ms: u64) {
//                     r#"
//                     #include <linux/delay.h>
//                     "#;
//
//                     r#"
//                     msleep(ms);
//                     "#
//                 }
//
//                 fn run(cmd: &CStr) {
//                     #[cfunc]
//                     fn run(cmd: &CStr) -> Result<(), c_int> {
//                         r#"
//                         #include <linux/umh.h>
//                         "#;
//
//                         r#"
//                         char *envp[] = {
//                             "HOME=/",
//                             "PWD=/",
//                             "USER=root",
//                             "PATH=/:/sbin:/bin:/usr/sbin:/usr/bin",
//                             "SHELL=/bin/sh",
//                             NULL
//                         };
//
//                         char *argv[] = {
//                             "/bin/sh", "-c", (char *)cmd, NULL,
//                         };
//                         return call_usermodehelper(argv[0], argv, envp, UMH_WAIT_PROC);
//                         "#
//                     }
//                     let _ = run(cmd).map_err(|ret| pr_info!("Command {cmd:?} failed: {ret}"));
//                 }
//
//                 pr_info!("TRACING START");
//                 run(c"/trace-cmd reset -at > stdout.reset 2>&1");
//                 // run(c"/trace-cmd start -Bmybuffer -e all > stdout.start 2>&1");
//                 run(c"/trace-cmd start -e all > stdout.start 2>&1");
//                 pr_info!("TRACING STARTED");
//                 run(c"echo mymsg > /dev/kmsg");
//
//                 f(1, c"hello", "world", 42);
//                 f(2, c"hello2", "world2", 43);
//
//                 pr_info!("TRACING STOP");
//                 run(c"/trace-cmd stop -a > stdout.stop 2>&1");
//                 run(c"/trace-cmd extract -at > stdout.extract 2>&1");
//                 pr_info!("TRACING STOPED");
//
//
//                 use crate::runtime::sysfs::{KObjType, KObject};
//
//                 let root = KObject::sysfs_module_root();
//
//                 let kobj_type = Arc::new(KObjType::new());
//                 let mut kobject = KObject::new(kobj_type.clone());
//                 let mut kobject2 = KObject::new(kobj_type.clone());
//
//                 kobject.add(Some(&root), "folder1")
//                     .expect("Could not add kobject to sysfs");
//                 let kobject = kobject.publish().expect("Could not publish kobject");
//
//                 kobject2.add(Some(&kobject), "folder2")
//                     .expect("Could not add kobject to sysfs");
//                 let kobject2 = kobject2.publish().expect("Could not publish kobject");
//
//                 let feature2_service = services.get::<Feature2>();
//                 pr_info!("FEATURE2 service: {feature2_service:?}");
//                 yield_!(Ok(Arc::new(())));
//                 pr_info!("FEATURE1 stop");
//
//                 drop(f);
//                 drop(work_item);
//                 // drop(registered);
//                 Ok(())
//             })
//         ))
//     },
// }

// #[derive(Debug)]
// struct Feature2Service {
//     a: u32,
// }
//
// define_feature! {
//     struct Feature2,
//     name: "feature2",
//     visibility: Public,
//     Service: Feature2Service,
//     Config: (),
//     dependencies: [],
//     init: |configs| {
//         Ok((
//             DependenciesSpec::new(),
//             new_lifecycle!(|services| {
//                 pr_info!("FEATURE2 start");
//                 let service = Feature2Service { a: 42 };
//                 yield_!(Ok(Arc::new(service)));
//                 pr_info!("FEATURE2 stop");
//                 Ok(())
//             })
//         ))
//     },
// }
// define_event_feature!(struct myevent);

pub struct State {
    lifecycle: Mutex<Option<LifeCycle<(), (), Error>>>,
    config_stack: Mutex<Vec<BTreeMap<String, GenericConfig>>>,
    sessions: Mutex<BTreeMap<SessionId, QuerySession>>,
    session_id: AtomicU64,
    wq: Pin<Box<Wq>>,
}

fn pop_configs<S>(mut stack: S, n: usize) -> Result<usize, Error>
where
    S: DerefMut<Target = Vec<BTreeMap<String, GenericConfig>>>,
{
    let len = stack.len();
    if n > len {
        Err(error!(
            "Cannot pop {n} configs as only {len} are left in the stack"
        ))
    } else {
        let new_len = len.saturating_sub(n);
        stack.truncate(new_len);
        Ok(new_len)
    }
}

impl State {
    fn new() -> Result<State, Error> {
        Ok(State {
            lifecycle: Mutex::new(None, LockdepClass::new()),
            config_stack: Mutex::new(Vec::new(), LockdepClass::new()),
            sessions: Mutex::new(BTreeMap::new(), LockdepClass::new()),
            session_id: AtomicU64::new(0),
            wq: Box::pin(Wq::new("lisa_state")?),
        })
    }

    pub fn config_stack(&self) -> Result<Vec<BTreeMap<String, GenericConfig>>, Error> {
        Ok(self.config_stack.lock().clone())
    }

    pub fn push_config(&self, config: BTreeMap<String, GenericConfig>) -> Result<(), Error> {
        self.config_stack.lock().push(config);
        Ok(())
    }

    pub fn pop_configs(&self, n: usize) -> Result<usize, Error> {
        let stack = self.config_stack.lock();
        pop_configs(stack, n)
    }

    pub fn pop_all_configs(&self) -> Result<usize, Error> {
        let stack = self.config_stack.lock();
        let n = stack.len();
        pop_configs(stack, n)
    }

    pub fn restart<F>(&self, f: F) -> Result<(), Error>
    where
        F: FnOnce() -> Result<LifeCycle<(), (), Error>, Error>,
    {
        let mut lifecycle = self.lifecycle.lock();
        match &mut *lifecycle {
            None => Ok(()),
            Some(lifecycle) => lifecycle.stop(),
        }?;
        *lifecycle = Some(f()?);
        lifecycle.as_mut().unwrap().start(()).cloned()
    }

    pub fn stop(&self) -> Result<(), Error> {
        match &mut *self.lifecycle.lock() {
            None => Ok(()),
            Some(lifecycle) => lifecycle.stop(),
        }
    }

    pub fn new_session(&self, root: &mut Folder, state: Arc<State>) -> Result<String, Error> {
        let id = self.session_id.fetch_add(1, Ordering::Relaxed);
        let session = QuerySession::new(root, state, id)?;
        match self.sessions.lock().entry(id) {
            btree_map::Entry::Vacant(entry) => Ok(entry.insert(session).name()),
            _ => Err(error!("Session ID {id} already exists")),
        }
    }

    pub fn with_session<F, T>(&self, id: SessionId, f: F) -> Result<T, Error>
    where
        F: FnOnce(&mut QuerySession) -> T,
    {
        match self.sessions.lock().get_mut(&id) {
            Some(session) => Ok(f(session)),
            None => Err(error!("Could not find session ID {id}")),
        }
    }

    pub fn close_session(&self, id: SessionId) {
        // Ensure we drop the lock guard before dropping the removed value, otherwise we can get a
        // deadlock as we are taking session lock first then sysfs lock. When accessing control
        // files, the kernel takes the sysfs lock and then we take the session lock
        drop({
            let mut guard = self.sessions.lock();
            let x = guard.remove(&id);
            drop(guard);
            x
        })
    }

    pub fn finalize(&self) {
        // This is needed in order to break the Arc cycle:
        // State -> QuerySession -> State
        // which would otherwise prevent deallocation of the State.
        *self.sessions.lock() = BTreeMap::new();
        self.wq.clear_owned_work();
    }

    pub fn wq(&self) -> Pin<&Wq> {
        self.wq.as_ref()
    }
}

#[cfunc]
fn enable_all_param() -> bool {
    r#"
    #include <linux/module.h>

    static bool ___param_enable_all_features = true;
    module_param(___param_enable_all_features, bool, 0);
    MODULE_PARM_DESC(___param_enable_all_features, "If true, make a best effort attempt to enable all the features with no specific configuration upon module load.");
    "#;

    r#"
    return ___param_enable_all_features;
    "#
}

pub fn module_main() -> LifeCycle<(), (), c_int> {
    new_lifecycle!(|_| {
        let version = module_version();
        pr_info!("Loading Lisa module version {version}");

        let state = Arc::new(State::new().map_err(|err| {
            pr_err!("Error while creating the state workqueue: {err:#}");
            1
        })?);
        let query_service = QueryService::new(Arc::clone(&state)).map_err(|err| {
            pr_err!("Error while creating the query service: {err:#}");
            1
        })?;

        let enable_all = enable_all_param();

        // Legacy behavior: when loading the module without specifying any parameter, we load all
        // the features on a best-effort basis. This allows a basic user to get all what they can
        // that does not strictly require configuration out of the module.
        if enable_all {
            let mut spec = DependenciesSpec::new();
            spec.insert::<AllFeatures>(DependencySpec::Mandatory {
                configs: vec![AllFeaturesConfig { best_effort: true }],
            });

            let make_lifecycle =
                || features_lifecycle(|feat| feat.name() == AllFeatures::NAME, spec, Vec::new());

            // AFTER THIS POINT, NO MORE EARLY RETURN
            // Since we are going to create a LifeCycle, it would need to be stop()-ed before exiting,
            // so we don't want to do an early return and just drop it.

            yield_!(match state.restart(make_lifecycle) {
                Ok(()) => Ok(()),
                Err(err) => {
                    pr_err!("Error while starting features: {err:#}");
                    // Best-effort basis, so we don't return an error code. If we did so, this
                    // would prevent from unloading the module.
                    Ok(())
                }
            });
        } else {
            yield_!(Ok(()));
        }

        state.stop().map_err(|err| {
            pr_err!("Error while stopping features: {err:#}");
            // This is ignored as the module_exit() returns void
            0
        })?;

        state.finalize();
        pr_info!("Unloaded Lisa module version {version}");
        Ok(())
    })
}
