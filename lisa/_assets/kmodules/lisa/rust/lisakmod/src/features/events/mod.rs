/* SPDX-License-Identifier: GPL-2.0 */

macro_rules! define_event_feature {
    (struct $type:ident, $event_name:literal) => {
        $crate::features::define_feature! {
            struct $type,
            name: concat!("event__", $event_name),
            visibility: Public,
            Service: (),
            Config: (),
            dependencies: [],
            init: |_| {
                Ok($crate::lifecycle::new_lifecycle!(|services| {
                    yield_!(Ok(::alloc::sync::Arc::new(())));
                    Ok(())
                }))
            },
        }
    };
}
