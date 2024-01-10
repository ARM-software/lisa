use std::fmt;

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub enum EventReq {
    #[serde(rename = "single")]
    SingleEvent(&'static str),
    #[serde(rename = "or")]
    OrGroup(&'static [EventReq]),
    #[serde(rename = "and")]
    AndGroup(&'static [EventReq]),
    #[serde(rename = "optional")]
    OptionalGroup(&'static [EventReq]),
    #[serde(rename = "dynamic")]
    DynamicGroup(&'static [EventReq]),
}

fn fmt_group(
    f: &mut fmt::Formatter,
    reqs: &[EventReq],
    op: &'static str,
    prefix: &'static str,
) -> fmt::Result {
    write!(f, "({}", prefix)?;
    let body = reqs
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(op);
    write!(f, "{}", body)?;
    write!(f, ")")
}

impl fmt::Display for EventReq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EventReq::SingleEvent(name) => write!(f, "{}", name),
            EventReq::OrGroup(reqs) => fmt_group(f, reqs, " or ", ""),
            EventReq::AndGroup(reqs) => fmt_group(f, reqs, " and ", ""),
            EventReq::OptionalGroup(reqs) => fmt_group(f, reqs, ", ", "optional: "),
            EventReq::DynamicGroup(reqs) => fmt_group(f, reqs, ", ", "one_group_of: "),
        }
    }
}

#[macro_export]
macro_rules! const_event_req {
    ($name:ident, $events:tt) => {
        const $name: $crate::eventreq::EventReq = $crate::event_req!($events);
    };
}

#[macro_export]
macro_rules! event_req {

    // Binary operators
    (@binop and) => {$crate::eventreq::EventReq::AndGroup};
    (@binop or) => {$crate::eventreq::EventReq::OrGroup};
    (@binop $x:tt) => {compile_error!(concat!("Unknown binary operator \"", stringify!($x), "\" in event requiremnts"))};

    // Group operators
    (@groupop one_group_of) => {$crate::eventreq::EventReq::DynamicGroup};
    (@groupop optional) => {$crate::eventreq::EventReq::OptionalGroup};
    (@groupop $x:tt) => {compile_error!(concat!("Unknown group operator \"", stringify!($x), "\" in event requiremnts"))};


    // Parse binary operatos
    (@nextbinop $a:tt) => {$a};
    (@nextbinop $a:tt $op:tt $b:tt) => {
        $crate::event_req!(@binop $op)(&[
            $a,
            $crate::event_req!($b)
        ])
    };
    (@nextbinop $pre:tt $op:tt $b:tt $($tail:tt)+) => {
        $crate::event_req!(
            @nextbinop
            (
                $crate::event_req!(@binop $op)(&[
                    $pre,
                    $crate::event_req!($b)
                ])
            )

            $($tail)*
        )
    };

    (@nextbinop $($x:tt)*) => {compile_error!(concat!("Unknown syntax \"", stringify!($($x)*), "\" in event requiremnts"))};


    // Parse single event
    ($event:literal) => { $crate::eventreq::EventReq::SingleEvent($event) };

    // Parse variable containing an EventReq
    ({$eventreq:expr}) => { $eventreq };

    // Parse group operators in the form "(XXX: a, b, c, ...)"
    (($op:tt : $($tail:tt),* )) => {
        $crate::event_req!(@groupop $op)(&[
            $(
                $crate::event_req!($tail),
            )*
        ])
    };

    // Parse parenthesized grouping
    (($($tail:tt)*)) => {
        $crate::event_req!($($tail)*)
    };


    // Parse left-associative binary operators
    ($event:tt $($tail:tt)+) => {
        $crate::event_req!(@nextbinop ($crate::event_req!($event)) $($tail)* )
    };

    ($x:tt) => {compile_error!(concat!("Unknown syntax \"", stringify!($x), "\" in event requiremnts"))};
}
