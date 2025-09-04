/* SPDX-License-Identifier: GPL-2.0 */

// use crate::{
// features::{Feature, Visibility},
// lifecycle::new_lifecycle,
// };
//
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    sync::Arc,
    vec::Vec,
};
use core::{
    cell::UnsafeCell,
    ffi::{CStr, c_int, c_uint},
    pin::Pin,
    ptr::NonNull,
    sync::atomic::{AtomicU64, Ordering},
};

use itertools::Itertools as _;
use lisakmod_macros::inlinec::{NegativeError, PtrError, cconstant, cfunc, opaque_type};

use crate::{
    error::{Error, error},
    features::{DependenciesSpec, FeatureResources, ProvidedFeatureResources, define_feature},
    lifecycle::new_lifecycle,
    query::query_type,
    runtime::{
        cpumask::{CpuId, active_cpus, smp_processor_id},
        irqflags::local_irq_save,
        printk::pr_err,
        traceevent::{TracepointString, new_event, new_tracepoint_string},
        tracepoint::{Tracepoint, new_probe},
        version::kernel_version,
    },
};

type EventTyp = u32;

opaque_type!(
    struct CTaskStruct,
    "struct task_struct",
    "linux/sched.h"
);

opaque_type!(
    struct _CPerfEvent,
    "struct perf_event",
    "linux/perf_event.h"
);

#[derive(Debug)]
enum PerfEventState {
    Dead,
    Exit,
    Error,
    Off,
    Inactive,
    Active,
    #[allow(dead_code)]
    Unknown(c_int),
}

impl From<c_int> for PerfEventState {
    fn from(state: c_int) -> PerfEventState {
        macro_rules! convert {
            ($x:expr, $fallback_pat:pat => $fallback:expr, $($macro:literal => $val:expr),*, $(,)?) => {{
                (move |x| {
                    $(
                        if Some(&x) == Option::as_ref(&cconstant!("#include <linux/perf_event.h>", $macro)) {
                            return $val;
                        }
                    )*

                    let $fallback_pat = x;
                    $fallback
                })($x)
            }}
        }

        // Convert to an unsigned type, as some PERF_EVENT_STATE_* values are negative, which is
        // not supported by cconstant!()
        convert! {
            state as u32,
            val => PerfEventState::Unknown(val as c_int),
            "(u32)PERF_EVENT_STATE_DEAD" => PerfEventState::Dead,
            "(u32)PERF_EVENT_STATE_EXIT" => PerfEventState::Exit,
            "(u32)PERF_EVENT_STATE_ERROR" => PerfEventState::Error,
            "(u32)PERF_EVENT_STATE_OFF" => PerfEventState::Off,
            "(u32)PERF_EVENT_STATE_INACTIVE" => PerfEventState::Inactive,
            "(u32)PERF_EVENT_STATE_ACTIVE" => PerfEventState::Active,

        }
    }
}

query_type! {
    #[derive(Clone)]
    struct PmuConfig {
        events: BTreeSet<PerfEventDesc>,
    }
}

impl PmuConfig {
    fn merge<'a, I>(iter: I) -> PmuConfig
    where
        I: Iterator<Item = &'a Self>,
    {
        let events: BTreeSet<_> = iter.flat_map(|config| &config.events).cloned().collect();
        PmuConfig { events }
    }
}

query_type! {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct RawPerfEventId {
        id: u64,
        pmu_name: String,
    }
}

query_type! {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct GenericPerfEventId {
        name: String,
    }
}

query_type! {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum PerfEventId {
        Raw(RawPerfEventId),
        Generic(GenericPerfEventId),
    }
}

// Information dumped in the ftrace event to identify what event we are using.  The
// PerfEventAttr.id is not a great choice as it can also encode the PMU type when
// PERF_TYPE_HARDWARE is used and if the user wants the generic event to be enabled on a specific
// PMU.
enum PerfEventRuntimeId {
    Raw {
        id: u64,
        pmu_name: String,
    },
    Generic {
        // Use a TracepointString so that we only pay for a pointer-sized field rather than the
        // full name copied inside the event.
        name: TracepointString,
    },
}

// Our version of struct perf_event_attr, that can then be turned into a set of values used to
// populate an actual perf_event_attr.
struct PerfEventAttr {
    typ: EventTyp,
    id: u64,
    runtime_id: PerfEventRuntimeId,
}

impl PerfEventId {
    fn attr(&self) -> Result<PerfEventAttr, Error> {
        match self {
            PerfEventId::Generic(id) => {
                macro_rules! match_event_data {
                    ($lookup:expr, $($name:expr => $id:literal),* $(,)?) => {{
                        let lookup = $lookup;
                        match lookup {
                            $(
                                $name => Ok((
                                    new_tracepoint_string!($name),
                                    cconstant!("#include <linux/perf_event.h>", $id).unwrap()
                                )),
                            )*
                            _ => Err(error!("Could not find perf event: {lookup}"))
                        }
                    }}
                }

                let (runtime_name, id) = match_event_data! {
                    &*id.name,
                    "cpu-cycles" => "PERF_COUNT_HW_CPU_CYCLES",
                    "instructions" => "PERF_COUNT_HW_INSTRUCTIONS",
                    "cache-references" => "PERF_COUNT_HW_CACHE_REFERENCES",
                    "cache-misses" => "PERF_COUNT_HW_CACHE_MISSES",
                    "branch-instructions" => "PERF_COUNT_HW_BRANCH_INSTRUCTIONS",
                    "branch-misses" => "PERF_COUNT_HW_BRANCH_MISSES",
                    "bus-cycles" => "PERF_COUNT_HW_BUS_CYCLES",
                    "stalled-cycles-frontend" => "PERF_COUNT_HW_STALLED_CYCLES_FRONTEND",
                    "stalled-cycles-backend" => "PERF_COUNT_HW_STALLED_CYCLES_BACKEND",
                    "ref-cycles" => "PERF_COUNT_HW_REF_CPU_CYCLES",
                }?;

                const PERF_TYPE_HARDWARE: u32 =
                    match cconstant!("#include <linux/perf_event.h>", "PERF_TYPE_HARDWARE") {
                        Some(x) => x,
                        None => 0,
                    };

                Ok(PerfEventAttr {
                    id,
                    typ: PERF_TYPE_HARDWARE,
                    runtime_id: PerfEventRuntimeId::Generic { name: runtime_name },
                })
            }
            PerfEventId::Raw(id) => {
                const PERF_TYPE_RAW: u32 =
                    match cconstant!("#include <linux/perf_event.h>", "PERF_TYPE_RAW") {
                        Some(x) => x,
                        None => 0,
                    };
                Ok(PerfEventAttr {
                    id: id.id,
                    // FIXME: we need to pass the actual PMU type matching the id.pmu_name string.
                    // Otherwise, we will end up with an event on _some_ PMU, that may or may not
                    // be the one we want. This would fail the pmu_name check down the line.
                    //
                    // Note that when we do that, we will also need to deal with the fact a given
                    // PMU may not support all CPUs, so we won't necessary be able to register such
                    // event on all CPUs.
                    //
                    typ: PERF_TYPE_RAW,
                    runtime_id: PerfEventRuntimeId::Raw {
                        id: id.id,
                        pmu_name: id.pmu_name.clone(),
                    },
                })
            }
        }
    }
}

query_type! {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum PerfEventTriggerTracepoint {
        #[serde(rename = "sched_switch")]
        SchedSwitch,
    }
}

query_type! {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum PerfEventTrigger {
        Tracepoint(PerfEventTriggerTracepoint),
    }
}

query_type! {
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct PerfEventDesc {
        id: PerfEventId,
        // FIXME: make use of that
        triggers: Vec<PerfEventTrigger>,
    }
}

#[repr(transparent)]
struct CPerfEventPtr(NonNull<UnsafeCell<_CPerfEvent>>);
unsafe impl Send for CPerfEventPtr {}
unsafe impl Sync for CPerfEventPtr {}

struct PerfEvent {
    desc: Arc<PerfEventDesc>,
    c_event: CPerfEventPtr,
    enabled: AtomicU64,
    cpu: CpuId,
}

impl Drop for PerfEvent {
    fn drop(&mut self) {
        #[cfunc]
        fn release(
            event: NonNull<UnsafeCell<_CPerfEvent>>,
        ) -> Result<c_uint, NegativeError<c_uint>> {
            r#"
            #include <linux/perf_event.h>
            #include "introspection.h"
            "#;
            r#"
            #
            #if HAS_SYMBOL(perf_event_release_kernel)
                return perf_event_release_kernel(event);
            #else
                return -ENOSYS;
            #endif
            "#
        }

        let enabled = self.enabled.load(Ordering::Relaxed);
        assert!(
            enabled == 0,
            "Cannot release perf event while not all PerfEventEnableGuard have been dropped"
        );

        release(self.c_event.0).expect("Could not release perf event");
    }
}

impl PerfEvent {
    fn from_desc(desc: Arc<PerfEventDesc>, cpu: CpuId) -> Result<PerfEvent, Error> {
        // FIXME: add overflow handler
        #[cfunc]
        fn perf_event_create_kernel_counter(
            id: u64,
            // typ is enum perf_type_id
            typ: EventTyp,
            cpu: c_int,
        ) -> Result<NonNull<UnsafeCell<_CPerfEvent>>, PtrError> {
            r#"
            #include <linux/perf_event.h>
            #include "introspection.h"
            "#;
            r#"
            #if HAS_SYMBOL(perf_event_create_kernel_counter)
                struct perf_event_attr attr = {
                    .type		= typ,
                    .size		= sizeof(struct perf_event_attr),
                    .pinned		= 1,
                    .disabled	= 1,
                    .config		= id,
                };

                return perf_event_create_kernel_counter(&attr, cpu, NULL, NULL, NULL);
            #else
                return PTR_ERR(-ENOSYS);
            #endif
            "#
        }

        let attr = desc.id.attr()?;

        let cpu_signed: c_int = cpu.try_into().unwrap();
        let c_event =
            perf_event_create_kernel_counter(attr.id, attr.typ, cpu_signed).map_err(|err| {
                error!(
                    "Could not allocate PMU counter for perf event {id:?}: {err}",
                    id = desc.id
                )
            })?;

        #[cfunc]
        unsafe fn get_pmu_name<'a>(event: NonNull<UnsafeCell<_CPerfEvent>>) -> &'a CStr {
            r#"
            #include <linux/perf_event.h>
            "#;
            r#"
            return event->pmu->name;
            "#
        }

        match &desc.id {
            PerfEventId::Raw(id) => {
                let expected = &id.pmu_name;
                // SAFETY: we don't hold onto the returned string beyond the life of the struct
                // perf_event it was taken from.
                let real = unsafe { get_pmu_name(c_event) };
                match real.to_str() {
                    Ok(real) => {
                        if real == expected {
                            Ok(())
                        } else {
                            Err(error!(
                                "Expected PMU type {expected} for raw perf event {id:?} but found: {real}"
                            ))
                        }
                    }
                    Err(_) => Err(error!("Could not convert PMU type {real:?} to Rust string")),
                }
            }
            _ => Ok(()),
        }?;

        Ok(PerfEvent {
            desc,
            cpu,
            c_event: CPerfEventPtr(c_event),
            enabled: AtomicU64::new(0),
        })
    }

    fn enable<'a>(&'a self) -> Result<PerfEventEnableGuard<'a>, Error> {
        let c_event = self.c_event().get_ref();

        #[cfunc]
        fn enable(event: &UnsafeCell<_CPerfEvent>) -> Result<c_uint, NegativeError<c_uint>> {
            r#"
            #include <linux/perf_event.h>
            #include "introspection.h"
            "#;
            r#"
            #if HAS_SYMBOL(perf_event_enable)
                perf_event_enable(event);
                return 0;
            #else
                return -ENOSYS;
            #endif
            "#
        }

        fn event_state(event: &UnsafeCell<_CPerfEvent>) -> PerfEventState {
            #[cfunc]
            fn event_state(event: &UnsafeCell<_CPerfEvent>) -> c_int {
                r#"
                #include <linux/perf_event.h>
                "#;
                r#"
                return event->state;
                "#
            }
            event_state(event).into()
        }

        // If there are multiple threads involved, the &Self we are working with here must have
        // been transmitted somehow, and that transmission channel should come with its own
        // synchronization barriers, so Ordering::Relaxed is enough.
        self.enabled.fetch_add(1, Ordering::Relaxed);
        enable(c_event).map_err(|err| error!("Could not enable perf event: {err}"))?;

        // Create the guard in all cases, so that perf_event_disable() is still called in case we
        // return an error.
        let guard = PerfEventEnableGuard { event: self };

        match event_state(c_event) {
            PerfEventState::Active => Ok(guard),
            PerfEventState::Error => Err(error!(
                "Perf event {id:?} state is PERF_EVENT_STATE_ERROR. Do you have enough counters available on this platform ?",
                id = self.desc.id
            )),
            state => Err(error!(
                "Perf event {id:?} is not active: {state:?}",
                id = self.desc.id
            )),
        }
    }

    fn c_event(&self) -> Pin<&UnsafeCell<_CPerfEvent>> {
        unsafe { Pin::new_unchecked(self.c_event.0.as_ref()) }
    }
}

struct PerfEventEnableGuard<'a> {
    event: &'a PerfEvent,
}

impl<'a> PerfEventEnableGuard<'a> {
    fn read(&self) -> Result<u64, Error> {
        // Disable IRQs so we don't risk re-entering on that CPU. Since we will only read a given
        // event on the CPU it was registered on, this means we can read that counter in any
        // context without fear. The only bad thing that could happen is if we call that function
        // from a context that could have interrupted an on-going pmu->read() call.
        let _guard = local_irq_save();

        if self.event.cpu == smp_processor_id() {
            // The approach taken is a *semi*-safe one as:
            // - the execution context is one as of the caller
            //   (__schedule) with preemption and interrupts being
            //   disabled
            // - the events being traced are per-CPU ones only
            // - kernel counter so no inheritance (no child events)
            // - counter is being read on/for a local CPU
            #[cfunc]
            fn read(event: &UnsafeCell<_CPerfEvent>) -> u64 {
                r#"
                #include <linux/perf_event.h>
                #include <linux/irqflags.h>
                #include <linux/types.h>
                "#;
                r#"
                // Refresh the count value
                event->pmu->read(event);

                // Read the now-refreshed value
                return (uint64_t)local64_read(&event->count);
                "#
            }
            Ok(read(self.event.c_event().get_ref()))
        } else {
            Err(error!("Cannot read a perf event for a remote CPU"))
        }
    }
}

impl<'a> Drop for PerfEventEnableGuard<'a> {
    fn drop(&mut self) {
        #[cfunc]
        fn disable(event: NonNull<UnsafeCell<_CPerfEvent>>) {
            r#"
            #include <linux/perf_event.h>
            #include "introspection.h"
            "#;
            r#"
            #if HAS_SYMBOL(perf_event_disable)
                return perf_event_disable(event);
            #endif
            "#
        }

        disable(self.event.c_event.0);
        // If there are multiple threads involved, the &Self we are working with here must have
        // been transmitted somehow, and that transmission channel should come with its own
        // synchronization barriers, so Ordering::Relaxed is enough. We won't attempt to destroy
        // the object until the reference is dropped.
        self.event.enabled.fetch_sub(1, Ordering::Relaxed);
    }
}

define_feature! {
    struct PmuFeature,
    name: "pmu",
    visibility: Public,
    Service: (),
    Config: PmuConfig,
    dependencies: [],
    resources: || {
        FeatureResources {
            provided: ProvidedFeatureResources {
                ftrace_events: [
                    "lisa__perf_event_raw".into(),
                    "lisa__perf_event_generic".into(),
                ].into()
            }
        }
    },
    init: |configs| {
        let config = PmuConfig::merge(configs);
        Ok((
            DependenciesSpec::new(),
            new_lifecycle!(|_| {
                let cpus: Vec<CpuId> = active_cpus().collect();
                let events = config.events
                    .into_iter()
                    .map(|desc| {
                        let desc = Arc::new(desc);

                        let events = cpus
                            .iter()
                            .copied()
                            .map(|cpu| PerfEvent::from_desc(desc.clone(), cpu))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(events)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;

                if events.is_empty() {
                    Err(error!("No perf event was requested"))
                } else {
                    Ok(())
                }?;

                let guards = events
                    .iter()
                    .flatten()
                    .map(|event| {
                        Ok((
                            event.desc.id.attr()?,
                            event.enable()?
                        ))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;

                let guard_cpu = |(_, guard): &(_, PerfEventEnableGuard<'_>)| guard.event.cpu;
                let per_cpu_guards = guards
                    .into_iter()
                    .sorted_by_key(guard_cpu)
                    .chunk_by(guard_cpu)
                    .into_iter()
                    .map(|(cpu, group)| (cpu, group.collect::<Vec<_>>()))
                    .collect::<BTreeMap<CpuId, Vec<_>>>();

                // Before 5.18, there is no prev_state last parameter, so we need to deal with that
                // as Pixel 6 kernel is 5.10
                if kernel_version() < (5, 18, 0) {
                    Err(error!("Kernels prior to v5.18.0 are not supported as the sched_switch tracepoint probe had a different signature"))
                } else {
                    Ok(())
                }?;

                if per_cpu_guards.is_empty() {
                    yield_!(Ok(Arc::new(())));
                } else {

                    #[allow(non_snake_case)]
                    let trace_lisa__perf_event_raw = new_event! {
                        lisa__perf_event_raw,
                        fields: {
                            // FIXME: once we have something like a a dynmaically-allocated
                            // TracepointString, use it for PMU name
                            pmu_name: &str,
                            event_id: u64,
                            value: u64,
                        }
                    }?;

                    #[allow(non_snake_case)]
                    let trace_lisa__perf_event_generic = new_event! {
                        lisa__perf_event_generic,
                        fields: {
                            // FIXME: Do we really want to have that string ? That makes the event
                            // 24 bytes long (+4 of buffer record header). If we used a u8 instead,
                            // it saves 7 bytes (+4 as well). That's a 25% space saving.
                            event_name: TracepointString,
                            value: u64,
                        }
                    }?;

                    let probe = new_probe!(
                        (_preempt: bool, _prev: *mut CTaskStruct, _next:* mut CTaskStruct, _prev_state: c_uint) {
                            let cpu = smp_processor_id();
                            if let Some(guards) = per_cpu_guards.get(&cpu) {
                                for (attr, guard) in guards {
                                    let value = guard.read();
                                    match value {
                                        Ok(value) => {
                                            match &attr.runtime_id {
                                                PerfEventRuntimeId::Raw{pmu_name, id} => {
                                                    trace_lisa__perf_event_raw(pmu_name, *id, value);
                                                }
                                                PerfEventRuntimeId::Generic{name} => {
                                                    trace_lisa__perf_event_generic(*name, value);
                                                }
                                            }
                                        }
                                        Err(err) => {
                                            pr_err!("{err:#}");
                                        }
                                    }
                                }
                            }
                        }
                    );

                    // SAFETY: sched_switch tracepoint has a static lifetime as it cannot suddenly
                    // disappear.
                    let tp = unsafe {
                        Tracepoint::<(bool, *mut CTaskStruct, *mut CTaskStruct, c_uint)>::lookup("sched_switch")
                    }.ok_or(error!("Could not find sched_switch tracepoint to attach to"))?;

                    let registered = tp.register_probe(&probe);

                    yield_!(Ok(Arc::new(())));

                    drop(registered);
                    drop(probe);
                }

                Ok(())
            }),
        ))
    },
}
