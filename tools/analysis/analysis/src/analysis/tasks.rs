use core::fmt::Debug;
use std::fmt::Write;

use ::macro_rules_attribute::apply;
use arrow2_convert::ArrowField;
use futures::stream::Stream;
use futures_async_stream::{for_await, stream};
pub(crate) use make_table_struct;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    analysis,
    analysis::{AnalysisResult, EventStream, SignalUpdate},
    const_event_req,
    event::{Comm, Event, EventData, Freq, Timestamp, CPU, PID},
    string::String,
};

#[stream(item = SignalUpdate<CPU, Freq>)]
async fn cpufreq(x: &Event) {
    match &x.data {
        EventData::EventCPUFrequency(fields) => yield SignalUpdate::Update(fields.cpu, fields.freq),
        _ => (), //yield SignalUpdate::Ignore,
    }
}

// async fn sum<S: EventStream>(stream: S) -> u64 {
// let mut acc: Freq = Freq::new(0);

// #[for_await]
// for x in stream.demux(cpufreq) {
//     match x {
//         SignalValue::Initial(_, cpu, (ts, freq)) => {
//             eprintln!("initial value: CPU {cpu:?} @ {ts:?} = {freq:?}")
//         }
//         SignalValue::Current(_cpu, (_ts, freq)) => acc += freq as u64,
//         SignalValue::Final(_, cpu, (ts, freq)) => {
//             eprintln!("last value: CPU {cpu:?} @ {ts:?} = {freq:?}")
//         }
//     }
// }
// for x in stream.demux(crate::analysis::splitter_union(&cpufreq, &cpufreq)) {
//     match x {
//         SignalValue::Initial(ts, cpu, (Some(freq), _)) => {
//             eprintln!("initial value: CPU {cpu:?} @ {ts:?} = {freq:?}")
//         }
//         SignalValue::Current(_ts, _cpu, (Some(freq), _)) => acc = acc + freq,
//         SignalValue::Final(ts, cpu, (Some(freq), _)) => {
//             eprintln!("last value: CPU {cpu:?} @ {ts:?} = {freq:?}")
//         }
//         _ => eprintln!("other"),
//     }
// }
// for x in stream.demux(&cpufreq) {
//     match x {
//         SignalValue::Initial(ts, cpu, freq) => {
//             eprintln!("initial value: CPU {cpu:?} @ {ts:?} = {freq:?}")
//         }
//         SignalValue::Current(_ts, _cpu, freq) => acc = acc + freq,
//         SignalValue::Final(ts, cpu, freq) => {
//             eprintln!("last value: CPU {cpu:?} @ {ts:?} = {freq:?}")
//         }
//         _ => eprintln!("other"),
//     }
// }
// let acc: u32 = acc.into();
// acc.into()
// }

#[stream(item = T::Item)]
async fn replicate<T>(n: u32, stream: T)
where
    T: Stream,
    T::Item: Clone,
{
    #[for_await]
    for x in stream {
        for _ in 0..n {
            yield x.clone();
        }
    }
}

async fn count<T: Stream>(stream: T) -> u64
where
    T::Item: Debug,
{
    let mut acc = 0;
    #[for_await]
    for _event in stream {
        // eprintln!("{:?}", _event);
        acc += 1;
    }
    // eprintln!("acc={}", acc);
    acc
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Task {
    Idle(CPU),
    Normal(PID),
}

impl Task {
    pub fn new(pid: &PID, cpu: &CPU) -> Task {
        if pid == &PID::new(0) {
            Task::Idle(cpu.clone())
        } else {
            Task::Normal(pid.clone())
        }
    }

    pub fn pid(&self) -> PID {
        match self {
            Task::Idle(_) => PID::new(0),
            Task::Normal(pid) => pid.clone(),
        }
    }
}

trait TaskID: Clone {
    fn new(task: &Task, comm: &Comm) -> Self;
    fn task(self: &Self) -> Option<Task>;
    fn comm(self: &Self) -> Option<Comm>;

    const STORES_COMM: bool;
}

impl TaskID for (Task, Comm) {
    fn new(task: &Task, comm: &Comm) -> Self {
        (task.clone(), comm.clone())
    }
    fn task(self: &Self) -> Option<Task> {
        Some(self.0.clone())
    }
    fn comm(self: &Self) -> Option<Comm> {
        Some(self.1.clone())
    }
    const STORES_COMM: bool = true;
}
impl TaskID for Comm {
    fn new(_task: &Task, comm: &Comm) -> Self {
        comm.clone()
    }
    fn task(self: &Self) -> Option<Task> {
        None
    }
    fn comm(self: &Self) -> Option<Comm> {
        Some(self.clone())
    }
    const STORES_COMM: bool = true;
}
impl TaskID for Task {
    fn new(task: &Task, _comm: &Comm) -> Self {
        task.clone()
    }
    fn task(self: &Self) -> Option<Task> {
        Some(self.clone())
    }
    fn comm(self: &Self) -> Option<Comm> {
        None
    }
    const STORES_COMM: bool = false;
}

// This is a limited set of state compared to what can be described in struct
// task_struct, but sched_switch will only dump states with only one bit set and
// a state listed in the TASK_REPORT mask
#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize, JsonSchema, ArrowField)]
#[arrow_field(type = "sparse")]
enum KernelTaskState {
    Running,
    Interruptible,
    Uninterruptible,
    Stopped,
    Traced,
    Dead,
    Zombie,
    Parked,
    // If the conversion failed and unknown flags are passed
    Unknown(u32),
}

impl From<u32> for KernelTaskState {
    fn from(flags: u32) -> Self {
        if flags == 0 {
            KernelTaskState::Running
        } else if flags & 0x1 != 0 {
            KernelTaskState::Interruptible
        } else if flags & 0x2 != 0 {
            KernelTaskState::Uninterruptible
        } else if flags & 0x4 != 0 {
            KernelTaskState::Stopped
        } else if flags & 0x8 != 0 {
            KernelTaskState::Traced
        } else if flags & 0x10 != 0 {
            KernelTaskState::Dead
        } else if flags & 0x20 != 0 {
            KernelTaskState::Zombie
        } else if flags & 0x40 != 0 {
            KernelTaskState::Parked
        } else if flags & 0x80 != 0 {
            // This one is really fishy: the code is the one of TASK_DEAD, but
            // TASK_DEAD is not part of TASK_REPORT and therefore should not
            // appear in the bitflags logged by sched_switch events. However,
            // there is another flag called TASK_IDLE_REPORT that happens to
            // have the exact same value (coincidence ?) and is not part of
            // TASK_REPORT either, but is set after the processing of the state
            // with TASK_REPORT. At the same time, the format string of
            // sched_switch explicitly maps TASK_DEAD to "I" letter. The stink
            // is real.
            //
            // SInce TASK_IDLE "state" is actually defined as
            // "TASK_UNINTERRUPTIBLE | TASK_NOLOAD", consider it as an
            // Uninterruptible sleep.
            KernelTaskState::Uninterruptible
            // Another fishy one: TASK_REPORT_MAX==0x100 value is used to report
            // running preempted tasks. Since we can already see that a task is
            // preempted with TaskState::Inactive(KernelTaskState::Running),
            // there is no need to map it to something else than Running.
        } else if flags & 0x100 != 0 {
            KernelTaskState::Running
        } else {
            KernelTaskState::Unknown(flags)
        }
    }
}

#[macro_export]
macro_rules! variant_structs {
    (
        // type_meta will be applied on all the variant type definitions created
        // by this macro. This ensures macros like Clone will work as expected,
        // since all the "children" types will get the #[derive()] as well.
        $( #[$type_meta:meta] )*
        $vis:vis enum $name:ident {
            $(
                // variant_meta will be applied on the variant itself, but not
                // on the payload type definition created for it. This allows
                // using e.g. #[serde(skip)] on some variants and retain the
                // expected behavior.
                $( #[$variant_meta:meta] )*
                $variant:ident

                // Named fields
                $(
                    {
                        $(
                            $named_field:ident: $named_field_ty:ty
                        ),*
                        $(,)?
                    }
                )?
                // Positional fields
                $(
                    (
                        $(
                            $unnamed_field_ty:ty
                        ),*
                        $(,)?
                    )
                )?
            ),*
            $(,)?
        }
    ) => {
        ::paste::paste! {
            variant_structs!(
                @variants1
                {$(#[$type_meta])*}
                $(
                    // If we did not have fields for the variant, we don't
                    // generate a struct.
                    $(
                        #[automatically_derived]
                        $vis struct [<$name $variant>] {
                            $(
                                $vis $named_field: $named_field_ty
                            ),*
                        }
                    )?
                    $(
                        #[automatically_derived]
                        $vis struct [<$name $variant>] (
                            $(
                                $vis $unnamed_field_ty
                            ),*
                        );

                    )?
                )*
            );


            // Define the enum itself.
            $( #[$type_meta] )*
            $vis enum $name {
                $(
                    $( #[$variant_meta] )*
                    $variant
                    $(
                        // Repeat the enclosing $(...)? as many times as
                        // $named_field_ty repeats. This can be 0 or 1 time,
                        // depending on whether there was any field to that
                        // variant.
                        ${ignore(named_field_ty)}
                        ([<$name $variant>])
                    )?
                    $(
                        ${ignore(unnamed_field_ty)}
                        ([<$name $variant>])
                    )?
                ),*
            }

            // Implement constructor for each variant, to avoid having to refer
            // to the new variant types all the time.
            #[automatically_derived]
            impl $name
            {
                $(
                    $(
                        #[allow(nonstandard_style)]
                        #[inline]
                        $vis fn [<new_ $variant>] ($($named_field: $named_field_ty),*) -> $name {
                            $name :: $variant
                            (
                                [<$name $variant>]
                                {
                                    $(
                                        $named_field
                                    ),*
                                }
                            )
                        }
                    )?
                    $(
                        #[allow(nonstandard_style)]
                        #[inline]
                        $vis fn [<new_ $variant>] ($([<x ${index()}>] : $unnamed_field_ty),*) -> $name {
                            $name :: $variant
                            (
                                [<$name $variant>]
                                (
                                    // Repeat as many times as $unnamed_field_ty
                                    // and at each repetition, create an
                                    // identifier that is "xN" with N being the
                                    // current repetition count.
                                    $(
                                        ${ignore(unnamed_field_ty)}
                                        [<x ${index()}>]
                                    ),*
                                )
                            )
                        }
                    )?
                )*
            }

            // Allow converting to and from the tuple equivalent to the tuple
            // struct.
            $(
                $(
                    #[automatically_derived]
                    impl From<[<$name $variant>]> for ($($unnamed_field_ty,)*) {
                        #[inline]
                        fn from(x: [<$name $variant>]) -> Self {
                            (
                                $(
                                    ${ignore(unnamed_field_ty)}
                                    x.${index()},
                                )*
                            )
                        }
                    }

                    #[automatically_derived]
                    impl From<($($unnamed_field_ty,)*)> for [<$name $variant>] {
                        #[inline]
                        fn from(x: ($($unnamed_field_ty,)*)) -> Self {
                            [<$name $variant>]
                            (
                                $(
                                    ${ignore(unnamed_field_ty)}
                                    x.${index()},
                                )*
                            )
                        }
                    }
                )?
            )*
        }
    };

    // These levels are only there to turn "meta [a, b, c, ...]" into "[meta a,
    // meta b, meta c, ...]" so that we can apply meta to each variant.
    // Otherwise, it would complain about repetition levels not matching.
    (@variants1 $meta:tt $($variants:item)*) => {
        variant_structs!(@variants2 $($meta $variants)*);
    };

    (@variants2 $({ $(#[$meta:meta])* } $variants:item)*) => {
        $(
            $(#[$meta])*
            $variants
        )*
    }
}

macro_rules! row_enum {
    ($item:item) => {
        // All the attributes applied below variant_structs will be applied to
        // each variant struct generated as well.
        #[apply(variant_structs)]
        #[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize, JsonSchema, ArrowField)]
        // We use type = "sparse" because type = "dense" is broken currently:
        // https://github.com/DataEngineeringLabs/arrow2-convert/issues/86
        #[arrow_field(type = "sparse")]
        $item
    };
}

macro_rules! row_struct {
    ($item:item) => {
        #[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize, JsonSchema, ArrowField)]
        $item
    };
}

#[apply(variant_structs)]
#[derive(Clone)]
enum Foo {
    Bar,
    Bar3(),
    Bar4 {},
    Bar5(u32),
    Bar1(u8, u64),
    Bar2 { old1: Comm, new1: Comm },
}

// #[apply(row_struct)]
// struct Foo2 {
//     x: u8,
// }

#[apply(row_enum)]
enum TaskFinishedReason {
    Zombie,
    Dead,
    Renamed { old: Comm, new: Comm },
}

#[apply(row_enum)]
enum TaskState {
    Waking {
        src_cpu: CPU,
        target_cpu: CPU,
    },
    Active {
        cpu: CPU,
    },
    // Preempted is TaskState::Inactive(KernelState::Running)
    Inactive {
        cpu: CPU,
        kernel_state: KernelTaskState,
    },
    Finished {
        reason: TaskFinishedReason,
    },
}

#[stream(item = SignalUpdate<T, TaskState>)]
async fn tasks_state<T: TaskID + 'static>(x: &Event) {
    match &x.data {
        EventData::EventSchedSwitch(fields) => {
            let cpu = fields.__cpu;
            let prev_kernel = KernelTaskState::from(fields.prev_state);

            let prev_task = Task::new(&fields.prev_pid, &cpu);
            let prev_id = T::new(&prev_task, &fields.prev_comm);

            let next_task = Task::new(&fields.next_pid, &cpu);
            let next_id = T::new(&next_task, &fields.next_comm);

            // The task is switched out in 2 cases only:
            // 1. It is dead
            // 2. It is preempted, or otherwise stopped (e.g. SIGSTOP)
            match prev_kernel {
                // We detect the end task using prev_state=Z. This avoids
                // requiring a sched_process_exit event that may not have been
                // collected in usual cases, and also makes manual trace
                // analysis somewhat easier and is less likely to break
                // developer assumptions (i.e. the task finishes the last time
                // it's switched out and not before). If someone wants something
                // based on sched_process_exit, they are likely having special
                // needs and they should probably make their own signal.
                //
                // Using sched_switch also conveniently allows detecting the end
                // of a task at the exact same timestamp another task takes
                // over. This can facilitate aligning things in a GUI without
                // having weird gaps.
                KernelTaskState::Zombie => {
                    yield SignalUpdate::FinishedWithUpdate(
                        prev_id,
                        TaskState::new_Finished(TaskFinishedReason::Zombie),
                    );
                }
                KernelTaskState::Dead => {
                    yield SignalUpdate::FinishedWithUpdate(
                        prev_id,
                        TaskState::new_Finished(TaskFinishedReason::Dead),
                    );
                }
                // We could still have KernelTaskState::Running, but that means
                // the task has been preempted.
                _ => yield SignalUpdate::Update(prev_id, TaskState::new_Inactive(cpu, prev_kernel)),
            }
            yield SignalUpdate::Update(next_id, TaskState::new_Active(cpu))
        }
        EventData::EventSchedWakeup(fields) => {
            let task = Task::new(&fields.pid, &fields.__cpu);
            yield SignalUpdate::Update(
                T::new(&task, &fields.comm),
                TaskState::new_Waking(fields.__cpu, fields.target_cpu),
            )
        }
        EventData::EventSchedWakeupNew(fields) => {
            let task = Task::new(&fields.pid, &fields.__cpu);
            yield SignalUpdate::Update(
                T::new(&task, &fields.comm),
                TaskState::new_Waking(fields.__cpu, fields.target_cpu),
            )
        }
        // If we don't store the comm, changes to the comm are irrelevant. If we
        // do store the comm and the task gets renamed, we consider this as the
        // end of the current task. Any subsequent reuse (by that PID or another
        // one) of the same comm will appear as a new task freshly created.
        EventData::EventTaskRename(fields) if T::STORES_COMM => {
            let task = Task::new(&fields.pid, &fields.__cpu);
            let id = T::new(&task, &fields.oldcomm);
            yield SignalUpdate::FinishedWithUpdate(
                id,
                TaskState::new_Finished(TaskFinishedReason::new_Renamed(
                    fields.oldcomm.clone(),
                    fields.newcomm.clone(),
                )),
            );
        }
        _ => {}
    }
}

// TODO: find a nicer way of declaring the events of a function
const_event_req!(
    TASKS_STATE_EVENT_REQ,
    ("sched_wakeup" and "sched_wakeup_new" and "sched_switch" and "task_rename")
);

#[stream(item = SignalUpdate<Task, Comm>)]
async fn tasks_comm(x: &Event) {
    fn process(cpu: CPU, pid: PID, comm: &Comm) -> SignalUpdate<Task, Comm> {
        let task = Task::new(&pid, &cpu);
        match task {
            Task::Idle(cpu) => {
                let comm = Comm::from({
                    let mut comm = String::new();
                    let cpu: u32 = cpu.into();
                    write!(comm, "swapper/{cpu}")
                        .expect("Formatting of CPU in swapper task name failed");
                    comm
                });
                SignalUpdate::Update(task, comm)
            }
            Task::Normal(_) => SignalUpdate::Update(task, comm.clone()),
        }
    }
    match &x.data {
        EventData::EventSchedSwitch(fields) => {
            let cpu = fields.__cpu;
            yield process(cpu, fields.prev_pid, &fields.prev_comm);
            yield process(cpu, fields.next_pid, &fields.next_comm);
        }
        // Optional event. If we don't have it, we will simply wait for
        // sched_wakeup to change the state.
        EventData::EventSchedWaking(fields) => {
            yield process(fields.target_cpu, fields.pid, &fields.comm);
        }
        EventData::EventSchedWakeup(fields) => {
            yield process(fields.target_cpu, fields.pid, &fields.comm);
        }
        EventData::EventSchedWakeupNew(fields) => {
            yield process(fields.target_cpu, fields.pid, &fields.comm);
        }
        EventData::EventTaskRename(fields) => {
            yield process(fields.__cpu, fields.pid, &fields.newcomm);
        }

        // Optional event. If we don't have it, we simply will have unbounded
        // signal and use more memory than strictly necessary.
        EventData::EventSchedProcessFree(fields) => {
            let task = Task::new(&fields.pid, &fields.__cpu);
            yield SignalUpdate::Finished(task);
        }
        _ => (),
    }
}

const_event_req!(
    TASKS_COMM_EVENT_REQ,
    ("sched_wakeup" and "sched_wakeup_new" and "sched_switch" and "task_rename" and (optional: "sched_waking", "sched_process_free"))
);

// impl<T0, T1, T2> ConcatTuple for (Option<(T0, T1)>, T2) {
//     type Flat = (Option<T0>, Option<T1>, T2);
//     #[inline(always)]
//     fn concat_tuple(self) -> Self::Flat {
//         match self {
//             (Some((a, b)), c) => (Some(a), Some(b), c),
//             (None, c) => (None, None, c),
//         }
//     }
// }

macro_rules! if_set_else {
    ({$($true:tt)*} {$($false:tt)*}) => {
        $($false)*
    };
    ({$($true:tt)*} {$($false:tt)*} $($_:tt)+) => {
        $($true)*
    };
}

macro_rules! sql {
    (@joinf INNER) => {$crate::analysis::inner_join};
    (@joinf LEFT_OUTER) => {$crate::analysis::left_join};
    (@joinf RIGHT_OUTER) => {$crate::analysis::right_join};
    (@joinf FULL_OUTER) => {$crate::analysis::full_join};

    (@signal_joinf INNER) => {$crate::analysis::inner_join_signal};
    (@signal_joinf LEFT_OUTER) => {$crate::analysis::left_join_signal};
    (@signal_joinf RIGHT_OUTER) => {$crate::analysis::right_join_signal};
    (@signal_joinf FULL_OUTER) => {$crate::analysis::full_join_signal};

    (
        SELECT $select_arg:pat_param in $select_block:block
        FROM ($from_stream:expr)
        $(JOIN_SIGNAL $signal_join_type:tt ($signal_join_rstream:expr))*
        $(JOIN $join_type:tt ($join_rstream:expr) ON $join_larg:pat_param in $join_lbody:block == $join_rarg:pat_param in $join_rbody:block )*
        $(WHERE $where_arg:pat_param in $where_body:block)?
        $(GROUP BY $groupby_arg:pat_param in $groupby_body:block AGGREGATE WITH $(KEY $groupby_with_arg:pat_param in $groupby_with_body:block)? $(($groupby_agg:expr))?)?
    ) => {
        {
            let stream = $from_stream;
            $(
                let stream = sql!(@signal_joinf $signal_join_type)(
                    stream,
                    $signal_join_rstream,
                );
            )*
            $(
                let stream = sql!(@joinf $join_type)(
                    stream,
                    |$join_larg| $join_lbody,
                    $join_rstream,
                    |$join_rarg| $join_rbody
                );
            )*

            if_set_else!(
                {
                    let stream = ::futures::stream::StreamExt::filter(stream, |x| {
                        ::futures::future::ready(
                            #[allow(irrefutable_let_patterns)]
                            if let $($where_arg)* = x {
                                // Allow using "expr?" syntax in $where_body
                                (move || {
                                    $($where_body)?
                                })()
                            } else {
                                false
                            }
                        )
                    });
                }
                {}
                $($where_body)*
            );
            let stream = ::futures::stream::StreamExt::filter_map(stream, |x| {
                ::futures::future::ready(
                    #[allow(irrefutable_let_patterns)]
                    if let $select_arg = x {
                        // Allow using "expr?" syntax in $select_block
                        (move || {
                            $select_block
                        })()
                    } else {
                        None
                    }
                )
            });
            if_set_else!(
                {
                    let stream = $crate::analysis::group_by(
                        stream,
                        |x| {
                            let $($groupby_arg)* = x;
                            $($groupby_body)?
                        },
                        |_k, stream|
                        if_set_else!(
                            {{
                                $($($groupby_agg)?)?(stream)
                            }}
                            {{
                                let $($($groupby_with_arg)*)* = (k, stream);
                                $($($groupby_with_body)?)?
                            }}
                            $($($groupby_agg)*)*
                        )
                    );
                }
                {}
                $($groupby_body)*
            );
            stream
        }
    };
}

// make__struct! {
//     #[derive(Clone, Debug)]
//     struct TasksStatesRow {
//         ts: Timestamp,
//         pid: PID,
//         comm: Option<Comm>,
//         state: TaskState,
//     }
// }

#[derive(Clone, Debug, ArrowField)]
#[apply(make_table_struct)]
struct TasksStatesRow {
    ts: Timestamp,
    pid: PID,
    comm: Option<Comm>,
    state: TaskState,
}

analysis! {
    name: tasks_states,
    events: ({TASKS_COMM_EVENT_REQ} and {TASKS_STATE_EVENT_REQ}),
    (stream: EventStream, _args: ()) {
        let states = stream.fork().demux(&tasks_state::<Task>);

        // TODO: find a way of filtering out Initial/Final values that are not
        // inside the window.
        //
        // => Actually not, what we need is _demux() to never emit Initial/Final
        // when not inside a window, and instead store the Initial/Final so that
        // it can be yielded at the beginning of the next window.
        let rows = sql!(
            SELECT x in {
                // if x.key() == &PID::new(5740) {
                //     eprintln!("taken={x:?}");
                // }


                let (ts, task, (state, comm)) = x.into();
                Some(TasksStatesRow { ts, pid: task.pid(), comm, state })
            }
            FROM (states)

            JOIN_SIGNAL LEFT_OUTER (stream.fork().demux(&tasks_comm))
            // GROUP BY row in {row.comm.clone()}
            // AGGREGATE WITH (count)
        );

        AnalysisResult::from_row_stream(rows)
        // |_| Box::pin(AnalysisResult::from_row_stream(rows))

        // let map = rows.await;
        // AnalysisResult::from_map(map)

        // AnalysisResult::new(Timestamp::new(9_000_000))
        // AnalysisResult::new(MyResult{
        //     x: 33,
        //     myts: vec![
        //         Timestamp::new(9_000_000),
        //         Timestamp::new(9_100_000),
        //     ],
        // })
    }
}
