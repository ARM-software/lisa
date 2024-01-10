// The musl libc allocator is pretty slow, switching to mimalloc or jemalloc
// makes the resulting binary significantly faster, as we allocate pretty
// heavily when parsing JSON. mimalloc crate compiles much more quickly though.
#[cfg(target_arch = "x86_64")]
use mimalloc::MiMalloc;
#[global_allocator]
#[cfg(target_arch = "x86_64")]
static GLOBAL: MiMalloc = MiMalloc;

use core::{
    fmt::Debug,
    future::Future,
    iter::zip,
    task::{Context, Poll},
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{stdin, BufReader, Cursor, Read},
    path::PathBuf,
};

use ::futures::{future::join_all, pin_mut};
use analysis::{
    analysis::{get_analyses, AnalysisConf, EventWindow, HasKDim, TraceEventStream, WindowUpdate},
    event::{Event, EventData, EventID},
    futures::make_noop_waker,
};
use clap::Parser;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use serde_json::{value::Value, Deserializer};

#[derive(clap::Parser, Debug)]
struct Cli {
    #[clap(subcommand)]
    cmd: Subcommands,
}

#[derive(clap::Subcommand, Debug)]
enum Subcommands {
    Run {
        /// The path to the file to read
        #[clap(parse(from_os_str))]
        path: PathBuf,
        analyses: String,
        #[clap(long, default_value = "\"none\"")]
        window: String,
        #[clap(long)]
        out_path: PathBuf,
    },
    List,
}

pub fn do_run<SelectFn, R, F, T>(
    mut stream: TraceEventStream<SelectFn>,
    fut: F,
    reader: R,
) -> Result<impl Debug + Serialize, String>
where
    R: Read,
    F: Future<Output = T>,
    T: Debug + Serialize,
    SelectFn: FnMut(&Event) -> WindowUpdate<<Event as HasKDim>::KDim>,
{
    let events = Deserializer::from_reader(reader).into_iter::<Event>();

    pin_mut!(fut);

    let mut errors = vec![];
    let mut res = None;

    let waker = make_noop_waker();
    let mut ctx = Context::from_waker(&waker);

    for (id, event) in zip(1.., events) {
        let event = match event {
            Ok(Event {
                data: EventData::UnknownEvent,
                ..
            }) => continue,
            Ok(mut event) => {
                event.id = EventID::new(id);
                event
            }
            Err(error) => {
                errors.push((id, error));
                continue;
            }
        };

        stream.set_curr_event(event);
        match fut.as_mut().poll(&mut ctx) {
            Poll::Pending => (),
            Poll::Ready(x) => {
                res = Some(x);
                break;
            }
        }
    }

    if !errors.is_empty() {
        Err(format!("Errors while parsing events: {:?}", errors))
    } else {
        match res {
            None => Err("Coroutine did not finish when asked to".to_string()),
            Some(x) => Ok(x),
        }
    }
}

fn open_path(path: PathBuf) -> Result<Box<dyn Read>, String> {
    if path == PathBuf::from("-") {
        Ok(Box::new(BufReader::new(stdin())))
    } else {
        // Open the file in read-only mode
        let file = File::open(path.clone())
            .map_err(|x| format!("Could not open {}: {}", path.display(), x))?;

        Ok(match unsafe { Mmap::map(&file) } {
            // Divide by 2 the JSON overhead compared to BufReader
            Ok(mmap) => Box::new(Cursor::new(mmap)),
            Err(_) => Box::new(BufReader::with_capacity(8192 * 2, file)),
        })
    }
}

#[derive(Deserialize)]
struct CliAnalysis {
    name: String,
    args: Value,
}

fn run(path: PathBuf, out_path: PathBuf, analyses: String, window: String) -> Result<(), String> {
    let reader = open_path(path)?;
    let map = get_analyses();

    let window: EventWindow = serde_json::from_str(&window)
        .map_err(|x| format!("Could not decode window JSON: {}", x))?;

    let anas: Vec<CliAnalysis> =
        serde_json::from_str(&analyses).map_err(|x| format!("Could not analyses JSON: {}", x))?;

    let mut prev_selected = false;
    let stream = TraceEventStream::new(move |event: &Event| match window {
        EventWindow::None => match event {
            Event {
                data: EventData::StartOfStream,
                ..
            } => WindowUpdate::Start(event.to_kdim()),
            _ => WindowUpdate::None,
        },
        EventWindow::Time(start, end) => {
            let selected = event.ts >= start && event.ts <= end;
            let is_start = selected && !prev_selected;
            let is_end = !selected && prev_selected;
            prev_selected = selected;

            if is_start {
                WindowUpdate::Start(start)
            } else if is_end {
                // This window function only deals with a single window, so when
                // it's finished, the stream will run out.
                WindowUpdate::LastEnd(end)
            } else {
                WindowUpdate::None
            }
        }
    });

    let conf = AnalysisConf {
        stream: stream.clone(),
        out_path: out_path,
    };
    let mut futures = vec![];

    for ana in anas {
        let args = ana.args;
        let ana = map
            .get(&ana.name as &str)
            .ok_or(format!("Analysis does not exist: {}", ana.name))?;
        let fut = (ana.f)(&conf, &args);
        futures.push(fut);
    }

    let fut = join_all(futures);
    let fut = Box::pin(fut);
    let x = do_run(stream, fut, reader)?;
    let x = serde_json::ser::to_string(&x).map_err(|err| {
        format!(
            "JSON encoding error: {} occured when encoding:\n {:?}",
            err.to_string(),
            x
        )
    })?;
    println!("{}", x);
    Ok(())
}

fn list() -> Result<(), String> {
    let map: BTreeMap<_, _> =
        get_analyses::<fn(&Event) -> WindowUpdate<<Event as HasKDim>::KDim>>()
            .into_iter()
            .map(|(name, ana)| (name, BTreeMap::from([("eventreq", ana.eventreq)])))
            .collect();

    let map =
        serde_json::ser::to_string(&map).map_err(|x| format!("could not encode JSON: {:?}", x))?;
    println!("{}", map);
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Cli::parse();
    match args.cmd {
        Subcommands::Run {
            path,
            analyses,
            window,
            out_path,
        } => run(path, out_path, analyses, window),
        Subcommands::List => list(),
    }
}

// fn main() {
//     // let events = vec![
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventFoo(EventFooFields { value: 42 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventBar(EventBarFields { value: 101 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventFoo(EventFooFields { value: 43 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EventBar(EventBarFields { value: 102 }),
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::EndOfStream { ts: 0 },
//     //     },
//     //     Event {
//     //         id: 0,
//     //         data: EventData::CloseStream,
//     //     },
//     // ];
//     // let events: Vec<EventData> = events.iter().map(|event| event.data).collect();

//     let path = "./trace.json";
//     // Open the file in read-only mode with buffer.
//     let file = File::open(path).unwrap();
//     let reader = BufReader::new(file);

//     // let events: Vec<EventData> = serde_json::from_reader(reader).unwrap();
//     let events = Deserializer::from_reader(reader).into_iter::<EventData>();

//     // for _ in 0..19 {
//     //     events.extend(events.clone());
//     // }

//     for event in events {
//         let event = event.unwrap();
//         println!("{:?}", event);
//     }

//     // println!("size={}", events.len());

//     // println!("{:?}", events);

//     // let s = serde_json::to_string(&events).unwrap();
//     // let events: Vec<EventData> = serde_json::from_str(&s).unwrap();
//     // println!("success {}", events.len());
// }
