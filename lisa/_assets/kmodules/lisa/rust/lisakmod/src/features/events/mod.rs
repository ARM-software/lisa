/* SPDX-License-Identifier: GPL-2.0 */

macro_rules! define_event_feature {
    (struct $type:ident) => {
        $crate::features::define_feature! {
            struct $type,
            name: concat!("event__", stringify!($type)),
            visibility: Public,
            Service: (),
            Config: (),
            dependencies: [],
            init: |_| {
                Ok($crate::lifecycle::new_lifecycle!(|services| {
                    $crate::runtime::printk::pr_info!("Enabling ftrace event: {}", stringify!($type));
                    yield_!(Ok(::alloc::sync::Arc::new(())));
                    $crate::runtime::printk::pr_info!("Disabling ftrace event: {}", stringify!($type));
                    Ok(())
                }))
            },
        }
    };
}
pub(crate) use define_event_feature;
