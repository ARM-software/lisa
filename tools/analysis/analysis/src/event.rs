use core::{
    any::type_name,
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

use schemars::JsonSchema;
use serde::{de, Deserialize, Serialize};

use crate::{arrow::newtype_impl_arrow_field, string::String};

newtype_impl_arrow_field!(Comm, String);
newtype_impl_arrow_field!(Freq, u32);
newtype_impl_arrow_field!(CPU, u32);
newtype_impl_arrow_field!(Load, u32);
newtype_impl_arrow_field!(Util, u32);
newtype_impl_arrow_field!(PID, u64);
newtype_impl_arrow_field!(Prio, u64);
newtype_impl_arrow_field!(Timestamp, u64);

#[inline(always)]
fn optional<'de, In, NullValue, Out, D>(
    deserializer: D,
    null_value: NullValue,
) -> Result<Option<Out>, D::Error>
where
    D: de::Deserializer<'de>,
    In: Deserialize<'de> + PartialEq<NullValue> + Display,
    Out: TryFrom<In>,
{
    let x: In = de::Deserialize::deserialize(deserializer)?;
    if x == null_value {
        Ok(None)
    } else {
        match x.try_into() {
            Err(_) => Err(format!(
                "Could not convert {:?} to output format {:?}",
                type_name::<In>(),
                type_name::<Out>()
            ))
            .map_err(de::Error::custom),
            Ok(x) => Ok(Some(x)),
        }
    }
}

fn optional_str<'de, D, Out>(deserializer: D) -> Result<Option<Out>, D::Error>
where
    D: de::Deserializer<'de>,
    Out: TryFrom<String>,
{
    optional::<'_, String, _, _, _>(deserializer, "(null)")
}

fn optional_int<'de, D, Out>(deserializer: D) -> Result<Option<Out>, D::Error>
where
    D: de::Deserializer<'de>,
    Out: TryFrom<i64>,
{
    optional::<'_, i64, _, _, _>(deserializer, -1)
}

fn optional_nat<'de, D, Out>(deserializer: D) -> Result<Option<Out>, D::Error>
where
    D: de::Deserializer<'de>,
    Out: TryFrom<u64> + PartialOrd,
    Out::Error: Display,
{
    let x = optional::<'_, i64, _, u64, _>(deserializer, -1)?;
    Ok(match x {
        Some(x) => Some(x.try_into().map_err(de::Error::custom)?),
        None => None,
    })
}

#[derive(
    Serialize,
    Deserialize,
    JsonSchema,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Hash,
    Eq,
    Ord,
    PartialOrd,
    Default,
)]
pub struct EventID(u64);

impl EventID {
    pub fn new(id: u64) -> EventID {
        EventID(id)
    }
}

macro_rules! derive_arith {
    ($type:ty, $ctor:ident, $underlying:ty) => {
        impl From<$underlying> for $type {
            fn from(x: $underlying) -> $type {
                $ctor(x)
            }
        }

        impl From<$type> for $underlying {
            fn from(x: $type) -> $underlying {
                x.0
            }
        }

        impl Sub for $type {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                $ctor(self.0 - rhs.0)
            }
        }

        impl Add for $type {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                $ctor(self.0 + rhs.0)
            }
        }

        impl Mul for $type {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                $ctor(self.0 * rhs.0)
            }
        }

        impl Div for $type {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                $ctor(self.0 / rhs.0)
            }
        }
    };
}
#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn new(ts: u64) -> Timestamp {
        Timestamp(ts)
    }
}

impl From<u64> for Timestamp {
    fn from(x: u64) -> Timestamp {
        Timestamp(x)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Comm(String);

impl From<String> for Comm {
    fn from(x: String) -> Comm {
        Comm(x)
    }
}

impl From<&str> for Comm {
    fn from(x: &str) -> Comm {
        Comm(x.into())
    }
}

#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
#[serde(from = "u64")]
pub struct PID(u64);

impl PID {
    #[inline]
    pub fn new(pid: u64) -> PID {
        PID(pid)
    }
}

impl From<u64> for PID {
    fn from(x: u64) -> PID {
        PID::new(x)
    }
}

#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
pub struct Prio(u64);

impl Prio {
    pub fn new(prio: u64) -> Prio {
        Prio(prio)
    }
}

impl From<u64> for Prio {
    fn from(x: u64) -> Prio {
        Prio(x)
    }
}

#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
pub struct CPU(u32);

impl CPU {
    pub fn new(cpu: u32) -> CPU {
        CPU(cpu)
    }
}

impl From<u32> for CPU {
    fn from(x: u32) -> CPU {
        CPU(x)
    }
}

impl From<CPU> for u32 {
    fn from(x: CPU) -> u32 {
        x.0
    }
}

#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
pub struct Freq(u32);

impl Freq {
    pub fn new(freq: u32) -> Freq {
        Freq(freq)
    }
}
derive_arith!(Freq, Freq, u32);

#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
pub struct Load(u32);

impl Load {
    pub fn new(load: u32) -> Load {
        Load(load)
    }
}
derive_arith!(Load, Load, u32);

#[derive(
    Serialize, Deserialize, JsonSchema, Debug, Copy, Clone, PartialEq, Hash, Eq, Ord, PartialOrd,
)]
pub struct Util(u32);

impl Util {
    pub fn new(util: u32) -> Util {
        Util(util)
    }
}
derive_arith!(Util, Util, u32);

// TODO: figure out if we could be compatible with non-kernel events as well
// easily.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    #[serde(skip)]
    pub id: EventID,

    #[serde(rename = "__ts")]
    pub ts: Timestamp,
    #[serde(flatten)]
    pub data: EventData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "__type")]
#[non_exhaustive]
pub enum EventData {
    #[serde(rename = "sched_switch")]
    EventSchedSwitch(EventSchedSwitchFields),
    #[serde(rename = "sched_waking")]
    EventSchedWaking(EventSchedWakingFields),
    #[serde(rename = "sched_wakeup")]
    EventSchedWakeup(EventSchedWakeupFields),
    #[serde(rename = "sched_wakeup_new")]
    EventSchedWakeupNew(EventSchedWakeupNewFields),
    #[serde(rename = "task_rename")]
    EventTaskRename(EventTaskRenameFields),
    #[serde(rename = "sched_process_exit")]
    EventSchedProcessExit(EventSchedProcessExitFields),
    #[serde(rename = "sched_process_free")]
    EventSchedProcessFree(EventSchedProcessFreeFields),
    #[serde(rename = "sched_pelt_se")]
    EventSchedPELTSE(EventSchedPELTSEFields),

    #[serde(rename = "cpu_frequency")]
    EventCPUFrequency(EventCPUFrequencyFields),

    #[serde(rename = "__lisa_event_stream_start")]
    StartOfStream,
    #[serde(rename = "__lisa_event_stream_end")]
    EndOfStream,

    #[serde(other)]
    #[serde(rename = "__lisa_unknown_event")]
    UnknownEvent,
}

// "__comm":"<idle>",
// "__pid":0,
// "__cpu":4,
// "prev_comm":"swapper\/4",
// "next_comm":"rcu_preempt",
// "prev_state":0,
// "prev_pid":0,
// "next_pid":12,
// "next_prio":120,
// "prev_prio":120,

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedSwitchFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub prev_comm: Comm,
    pub next_comm: Comm,
    pub prev_state: u32,
    pub prev_pid: PID,
    pub next_pid: PID,
    pub next_prio: Prio,
    pub prev_prio: Prio,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedWakingFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub comm: Comm,
    pub pid: PID,
    pub prio: Prio,
    pub target_cpu: CPU,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedWakeupFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub comm: Comm,
    pub pid: PID,
    pub prio: Prio,
    pub target_cpu: CPU,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedWakeupNewFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub comm: Comm,
    pub pid: PID,
    pub prio: Prio,
    pub target_cpu: CPU,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventTaskRenameFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub pid: PID,
    pub oldcomm: Comm,
    pub newcomm: Comm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedProcessExitFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub pid: PID,
    pub comm: Comm,
    pub prio: Prio,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedProcessFreeFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub comm: Comm,
    pub pid: PID,
    pub prio: Prio,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventSchedPELTSEFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    pub load: Load,
    pub util: Util,
    pub update_time: Timestamp,
    #[serde(deserialize_with = "optional_str")]
    pub comm: Option<Comm>,
    #[serde(deserialize_with = "optional_str")]
    pub path: Option<String>,
    #[serde(deserialize_with = "optional_nat")]
    pub pid: Option<PID>,
    pub cpu: CPU,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventCPUFrequencyFields {
    pub __comm: Comm,
    pub __pid: PID,
    pub __cpu: CPU,
    #[serde(rename = "cpu_id")]
    pub cpu: CPU,
    #[serde(rename = "state")]
    pub freq: Freq,
}
