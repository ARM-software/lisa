/* SPDX-License-Identifier: GPL-2.0 */

use alloc::collections::BTreeSet;
use core::ffi::c_int;

use crate::{
    features::{Feature, Visibility, features_lifecycle},
    lifecycle::LifeCycle,
};

// FIXME: clean that up
// use alloc::{collections::BTreeSet, sync::Arc, vec::Vec};
// use core::ffi::{c_int, c_ulong, c_void};
//
// use crate::{
//     features::{
//         Feature, FeaturesConfig, FeaturesService, Visibility, define_feature, features_lifecycle,
//         tracepoint::TracepointFeature, wq::WqFeature,
//     },
//     lifecycle::{LifeCycle, new_lifecycle},
//     runtime::printk::pr_info,
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
//             FeaturesConfig::new(),
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
//
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
//             FeaturesConfig::new(),
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

pub fn module_main() -> LifeCycle<(), (), c_int> {
    // Always enable the C legacy features, as it is just a proxy to enable the C features.
    let select_legacy = |feature: &dyn Feature| feature.name() == "__legacy_features";
    let select_public = |feature: &dyn Feature| {
        select_legacy(feature) || feature.visibility() == Visibility::Public
    };

    let to_enable = crate::features::legacy::module_param_features();
    match to_enable {
        None => features_lifecycle(select_public),
        Some(features) => {
            let features: BTreeSet<_> = features.collect();
            let select = |feature: &dyn Feature| {
                select_legacy(feature)
                    || (select_public(feature) && features.contains(feature.name()))
            };
            features_lifecycle(select)
        }
    }
}
