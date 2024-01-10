use core::{
    cmp::max,
    fmt::Debug,
    future::Future,
    hint::unreachable_unchecked,
    mem::{replace, swap, take},
    pin::Pin,
};
use std::{
    cell::Cell,
    collections::{btree_map::Entry, BTreeMap},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex},
};

use arrow2_convert::{field::ArrowField, serialize::ArrowSerialize};
use erased_serde::serialize_trait_object;
use futures::{
    future,
    future::FutureExt,
    stream::{select, Fuse, Stream},
    task::{Context, Poll},
    StreamExt as FuturesStreamExt,
};
use futures_async_stream::{for_await, stream};
use pin_project::pin_project;
use schemars::{
    gen::SchemaGenerator,
    schema::{RootSchema, Schema},
    schema_for, JsonSchema,
};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::value::Value;

use crate::{
    event::{Event, EventData, EventID, Timestamp},
    eventreq::EventReq,
    futures::make_noop_waker,
};

macro_rules! make_table_struct {
    ($(#[$attr:meta])* struct $name:ident { $($field_name:ident : $field_type:ty),+ $(,)?} ) => {
        $(#[$attr])* struct $name { $($field_name: $field_type),+ }

        impl crate::analysis::Row for $name {
            type AsTuple = ($($field_type),+);
            type AsRefTuple<'a> = ($(&'a $field_type),+);

            fn columns() -> ::std::vec::Vec<&'static str> {
                vec![$(stringify!($field_name)),+]
            }
        }

        // tuple -> struct
        impl From<<$name as crate::analysis::Row>::AsTuple> for $name {
            fn from(x: <$name as crate::analysis::Row>::AsTuple) -> $name {
                let ($($field_name),+) = x;
                $name{ $($field_name),+ }
            }
        }

        // struct -> tuple
        impl From<$name> for <$name as crate::analysis::Row>::AsTuple {
            fn from(x: $name) -> <$name as crate::analysis::Row>::AsTuple {
                ($(x.$field_name),+)
            }
        }

        // ref struct -> tuple of ref
        impl<'a> From<&'a $name> for <$name as crate::analysis::Row>::AsRefTuple<'a> {
            fn from(x: &'a $name) -> <$name as crate::analysis::Row>::AsRefTuple<'a> {
                ($(&x.$field_name),+)
            }
        }
    }
}

mod tasks;

// Unsound type to test the perf impact of an actual Mutex
pub struct FakeMutex<T>(Cell<T>);
pub struct FakeMutexGuard<'a, T>(&'a FakeMutex<T>);

use core::ops::{Deref, DerefMut};

impl<'a, T> Deref for FakeMutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 .0.as_ptr() }
    }
}

impl<T> FakeMutex<T> {
    fn new(x: T) -> Self {
        FakeMutex(Cell::new(x))
    }
    #[inline]
    fn lock<'a>(&'a self) -> Option<FakeMutexGuard<'a, T>> {
        Some(FakeMutexGuard(&self))
    }
}

unsafe impl<T> Send for FakeMutex<T> {}
unsafe impl<T> Sync for FakeMutex<T> {}

type MyMutex<T> = Mutex<T>;
// type MyMutex<T> = FakeMutex<T>;

#[derive(Clone)]
pub struct RawEventStream {
    last_seen: EventID,
    closed: bool,
    // TODO: remove the Mutex somehow
    pub curr_event: Arc<MyMutex<Cell<Option<Event>>>>,
}

impl RawEventStream {
    pub fn new() -> RawEventStream {
        RawEventStream {
            last_seen: EventID::new(0),
            closed: false,
            // Encapsulate the current event into an Rc<Refcell<>> so that all
            // the RawEventStream derived from this one will share the same
            // "slot".
            curr_event: Arc::new(MyMutex::new(Cell::new(None))),
        }
    }
}

impl Stream for RawEventStream {
    type Item = Event;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.closed {
            Poll::Ready(None)
        } else {
            let mut closed = false;
            let mut last_seen = self.last_seen;

            let res = {
                let curr_event = self.curr_event.as_ref().lock().unwrap();

                // Temporarily replace the event in the stream by None, we will put
                // it back later.
                let event = curr_event.replace(None);

                let res = match &event {
                    None => Poll::Pending,
                    Some(event) => {
                        last_seen = event.id;
                        if self.last_seen == event.id {
                            Poll::Pending
                        } else {
                            closed = match event.data {
                                EventData::EndOfStream => true,
                                _ => false,
                            };
                            Poll::Ready(Some(event.clone()))
                        }
                    }
                };

                // Put the event back in the stream for another consumer to look at
                // it.
                curr_event.replace(event);
                res
            };

            self.last_seen = last_seen;
            self.closed = closed;
            // Make sure to call the waker once, so that the future will be
            // rescheduled for polling next time an event is available. Otherwise,
            // combinators like futures::future::join_all() will simply not attempt
            // to poll, thinking that the Future will not make progress.
            cx.waker().wake_by_ref();
            res
        }
    }
}

#[derive(Debug, Clone)]
pub enum SignalValue<KDim, K, V> {
    Initial(KDim, K, V),
    Current(KDim, K, V),
    Final(KDim, K, V),

    WindowStart(KDim, K, V),
    WindowEnd(KDim, K, V),
}

impl<KDim, K, V> SignalValue<KDim, K, V> {
    fn map_kv<F, K2, V2>(self, f: F) -> SignalValue<KDim, K2, V2>
    where
        F: FnOnce(K, V) -> (K2, V2),
    {
        use SignalValue::*;

        match self {
            Initial(kdim, k, v) => {
                let (k, v) = f(k, v);
                Initial(kdim, k, v)
            }
            Current(kdim, k, v) => {
                let (k, v) = f(k, v);
                Current(kdim, k, v)
            }
            Final(kdim, k, v) => {
                let (k, v) = f(k, v);
                Final(kdim, k, v)
            }
            WindowStart(kdim, k, v) => {
                let (k, v) = f(k, v);
                WindowStart(kdim, k, v)
            }
            WindowEnd(kdim, k, v) => {
                let (k, v) = f(k, v);
                WindowEnd(kdim, k, v)
            }
        }
    }

    fn map<F, V2>(self, f: F) -> SignalValue<KDim, K, V2>
    where
        F: FnOnce(V) -> V2,
    {
        self.map_kv(|k, v| (k, f(v)))
    }

    fn map_key<F, K2>(self, f: F) -> SignalValue<KDim, K2, V>
    where
        F: FnOnce(K) -> K2,
    {
        self.map_kv(|k, v| (f(k), v))
    }

    #[inline]
    fn key(&self) -> &K {
        let (_, k, _) = self.into();
        k
    }
    #[inline]
    fn value(&self) -> &V {
        let (_, _, v) = self.into();
        v
    }
    #[inline]
    fn kdim(&self) -> &KDim {
        let (kdim, _, _) = self.into();
        kdim
    }

    #[inline]
    fn into_key(self) -> K {
        let (_, k, _) = self.into();
        k
    }
    #[inline]
    fn into_value(self) -> V {
        let (_, _, v) = self.into();
        v
    }
    #[inline]
    fn into_kdim(self) -> KDim {
        let (kdim, _, _) = self.into();
        kdim
    }
}

impl<KDim, K, V> From<SignalValue<KDim, K, V>> for (KDim, K, V) {
    fn from(x: SignalValue<KDim, K, V>) -> Self {
        match x {
            SignalValue::Initial(kdim, k, v) => (kdim, k, v),
            SignalValue::Current(kdim, k, v) => (kdim, k, v),
            SignalValue::Final(kdim, k, v) => (kdim, k, v),
            SignalValue::WindowStart(kdim, k, v) => (kdim, k, v),
            SignalValue::WindowEnd(kdim, k, v) => (kdim, k, v),
        }
    }
}

impl<'a, KDim, K, V> From<&'a SignalValue<KDim, K, V>> for (&'a KDim, &'a K, &'a V) {
    fn from(x: &'a SignalValue<KDim, K, V>) -> Self {
        match x {
            SignalValue::Initial(kdim, k, v) => (kdim, k, v),
            SignalValue::Current(kdim, k, v) => (kdim, k, v),
            SignalValue::Final(kdim, k, v) => (kdim, k, v),
            SignalValue::WindowStart(kdim, k, v) => (kdim, k, v),
            SignalValue::WindowEnd(kdim, k, v) => (kdim, k, v),
        }
    }
}

pub trait SignalUpdateFn<K, V> {
    // The function will not be able to return the SignalUpdate::UpdateWith
    // variant, as this would require taking an arbitrary amount of generic
    // parameter.
    fn call(self, k: &K, v: Option<&V>) -> SignalUpdate<K, V, !>;
}

impl<K, V, F> SignalUpdateFn<K, V> for F
where
    F: FnOnce(&K, Option<&V>) -> SignalUpdate<K, V, !>,
{
    fn call(self, k: &K, v: Option<&V>) -> SignalUpdate<K, V, !> {
        self(k, v)
    }
}

impl<K, V> SignalUpdateFn<K, V> for ! {
    fn call(self, _: &K, _: Option<&V>) -> SignalUpdate<K, V, !> {
        self
    }
}

pub enum SignalUpdate<K, V, UpdateFn = !> {
    Update(K, V),
    UpdateWith(K, UpdateFn),
    Finished(K),
    FinishedWithUpdate(K, V),
}

pub trait Splitter<'a, I, K, V, UpdateFn> {
    type SplitStream: Send + Stream<Item = SignalUpdate<K, V, UpdateFn>> + 'a;
    fn split(self: &Self, x: &'a I) -> Self::SplitStream;
}

impl<'a, F, I, R, K, V, UpdateFn> Splitter<'a, I, K, V, UpdateFn> for F
where
    (F, I, R, K, V, UpdateFn): 'a,
    F: Fn(&'a I) -> R,
    R: Send + Stream<Item = SignalUpdate<K, V, UpdateFn>> + 'a,
{
    type SplitStream = impl Stream<Item = SignalUpdate<K, V, UpdateFn>>;
    fn split(&self, x: &'a I) -> Self::SplitStream {
        self(x)
    }
}

pub trait MultiplexedStream: Stream
where
    Self::Item: HasKDim,
{
    // SplitFn needs to be a parameter since the actual DemuxStream type will
    // embed a value of type SplitFn in implementations.
    type DemuxStream<K: Send, V: Send, SplitFn: for<'a> Splitter<'a, Self::Item, K, V, UpdateFn> + Send, UpdateFn: Send>: Send + Stream<
        Item = SignalValue<<Self::Item as HasKDim>::KDim, K, V>,
    >;
    fn demux<K, V, SplitFn, UpdateFn>(
        self,
        split: SplitFn,
    ) -> Self::DemuxStream<K, V, SplitFn, UpdateFn>
    where
        // TODO: remove that
        K: Debug,
        V: Debug,

        K: Send + Ord + Eq + Clone,
        V: Send + Clone + Eq,
        UpdateFn: Send + SignalUpdateFn<K, V>,
        SplitFn: for<'a> Splitter<'a, Self::Item, K, V, UpdateFn> + Send;
}

pub trait EventStream: MultiplexedStream<Item = Event> {
    fn fork(&self) -> Self;
}

#[pin_project]
#[derive(Clone)]
pub struct TraceEventStream<SelectFn> {
    #[pin]
    stream: RawEventStream,
    last_selected: bool,
    select: SelectFn,
}

impl<SelectFn> TraceEventStream<SelectFn> {
    pub fn new(select: SelectFn) -> Self {
        TraceEventStream {
            stream: RawEventStream::new(),
            last_selected: false,
            select,
        }
    }

    pub fn set_curr_event(&mut self, event: Event) {
        self.stream
            .curr_event
            .as_ref()
            .lock()
            .unwrap()
            .replace(Some(event));
    }
}

pub enum WindowUpdate<KDim> {
    None,
    Start(KDim),
    End(KDim),
    LastEnd(KDim),
}

impl<SelectFn> Stream for TraceEventStream<SelectFn>
where
    SelectFn: FnMut(&Event) -> WindowUpdate<<Event as HasKDim>::KDim>,
{
    type Item = Event;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            let polled = this.stream.as_mut().poll_next(cx);
            return match &polled {
                Poll::Ready(Some(x)) => {
                    let last_selected = *this.last_selected;
                    let selected = match (this.select)(x) {
                        WindowUpdate::None => last_selected,
                        WindowUpdate::Start(_) => true,
                        WindowUpdate::End(_) => false,
                        WindowUpdate::LastEnd(_) => return Poll::Ready(None),
                    };
                    *this.last_selected = selected;

                    if selected {
                        polled
                    } else {
                        continue;
                    }
                }
                _ => polled,
            };
        }
    }
}

impl<SelectFn> MultiplexedStream for TraceEventStream<SelectFn>
where
    SelectFn: Send + FnMut(&Event) -> WindowUpdate<<Event as HasKDim>::KDim>,
{
    type DemuxStream<
        K: Send,
        V: Send,
        SplitFn: for<'a> Splitter<'a, Self::Item, K, V, UpdateFn> + Send,
        UpdateFn: Send,
    > = impl Send + Stream<Item = SignalValue<<Self::Item as HasKDim>::KDim, K, V>>;

    fn demux<K, V, SplitFn, UpdateFn>(
        self,
        split: SplitFn,
    ) -> Self::DemuxStream<K, V, SplitFn, UpdateFn>
    where
        // TODO: remove that
        K: Debug,
        V: Debug,

        K: Send + Ord + Eq + Clone,
        V: Send + Clone + Eq,
        UpdateFn: Send + SignalUpdateFn<K, V>,
        SplitFn: for<'a> Splitter<'a, Self::Item, K, V, UpdateFn> + Send,
    {
        _demux(self.stream, self.select, split)
    }
}

impl<SelectFn> EventStream for TraceEventStream<SelectFn>
where
    SelectFn: Clone + Send + FnMut(&Event) -> WindowUpdate<<Event as HasKDim>::KDim>,
{
    fn fork(&self) -> Self {
        self.clone()
    }
}

pub trait HasKDim {
    type KDim: Clone;
    fn to_kdim(&self) -> Self::KDim;
}

impl HasKDim for Event {
    type KDim = Timestamp;
    fn to_kdim(&self) -> Self::KDim {
        self.ts
    }
}

impl<KDim: Clone, K, V> HasKDim for SignalValue<KDim, K, V> {
    type KDim = KDim;
    fn to_kdim(&self) -> Self::KDim {
        self.kdim().clone()
    }
}

impl<K, V1, V2, KDim: Clone> HasKDim
    for StreamPairItem<SignalValue<KDim, K, V1>, SignalValue<KDim, K, V2>>
{
    type KDim = KDim;
    fn to_kdim(&self) -> Self::KDim {
        match self {
            StreamPairItem::Left(x) => x.kdim().clone(),
            StreamPairItem::Right(x) => x.kdim().clone(),
        }
    }
}

#[stream(item = SignalValue<KDim, K, V>)]
async fn signal_dedup<KDim, K, V, S>(stream: S)
where
    K: Ord + Eq + Clone,
    V: PartialEq + Clone,
    KDim: Clone,
    S: Stream<Item = SignalValue<KDim, K, V>>,
{
    let mut map: BTreeMap<K, V> = BTreeMap::new();

    #[for_await]
    for x in stream {
        let do_yield = match &x {
            SignalValue::Current(_, k, v) => match map.entry(k.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(v.clone());
                    true
                }
                Entry::Occupied(mut entry) => {
                    let _v = entry.get();
                    if _v != v {
                        entry.insert(v.clone());
                        true
                    } else {
                        false
                    }
                }
            },
            SignalValue::Initial(_, k, v) => {
                map.insert(k.clone(), v.clone());
                true
            }
            SignalValue::Final(_, k, _) => {
                map.remove(k);
                true
            }
            _ => true,
        };

        if do_yield {
            yield x;
        }
    }
}

#[inline]
#[stream(item = SignalValue<<S::Item as HasKDim>::KDim, K, V>)]
async fn _demux<S, K, V, SplitFn, SelectFn, UpdateFn>(
    stream: S,
    mut select: SelectFn,
    splitter: SplitFn,
) where
    // TODO: remove that
    V: Debug,
    K: Debug,
    S::Item: Debug,

    K: Send + Ord + Eq + Clone,
    V: Send + Clone + Eq,
    S: Send + Stream,
    S::Item: Send + HasKDim,
    SelectFn: Send + FnMut(&S::Item) -> WindowUpdate<<S::Item as HasKDim>::KDim>,
    SplitFn: for<'a> Splitter<'a, S::Item, K, V, UpdateFn>,
    UpdateFn: Send + SignalUpdateFn<K, V>,
{
    #[derive(Clone)]
    enum Selection<KDim> {
        Selected,
        NotSelected(Option<KDim>),
    }

    // The compiler won't be able to inline the recursive call, but at least try
    // to entice it to inline the first level.
    #[inline(always)]
    fn process<K, V, UpdateFn, Item>(
        update: SignalUpdate<K, V, UpdateFn>,
        map: &mut BTreeMap<K, (Option<V>, Option<V>, bool)>,
        selected: Selection<Item::KDim>,
        x: &Item,
    ) -> Option<SignalValue<Item::KDim, K, V>>
    where
        K: Ord + Eq + Clone,
        V: Clone + Eq,
        Item: HasKDim,
        UpdateFn: SignalUpdateFn<K, V>,
    {
        match update {
            SignalUpdate::Update(k, v) => {
                let (scurr, sinitial, _) =
                    map.entry(k.clone()).or_insert_with(|| (None, None, false));

                *scurr = Some(v.clone());

                match selected {
                    Selection::Selected => {
                        *sinitial = None;
                        match scurr {
                            None => Some(SignalValue::Initial(x.to_kdim(), k, v)),
                            _ => Some(SignalValue::Current(x.to_kdim(), k, v)),
                        }
                    }
                    _ => {
                        // At the beginning of the next window, we
                        // will see the latest Initial.
                        *sinitial = Some(v);
                        None
                    }
                }
            }
            SignalUpdate::Finished(k) => {
                let entry = map.entry(k.clone());
                let mut temp;
                let mut temp2;
                let (scurr, sinitial, sfinal) = match entry {
                    Entry::Occupied(entry) => match selected {
                        Selection::Selected => {
                            temp = entry.remove();
                            &mut temp
                        }
                        _ => {
                            temp2 = entry;
                            temp2.get_mut()
                        }
                    },
                    Entry::Vacant(_) => {
                        temp = (None, None, false);
                        &mut temp
                    }
                };

                let scurr = replace(scurr, None);
                match selected {
                    Selection::Selected => {
                        scurr.map(|scurr| SignalValue::Final(x.to_kdim(), k, scurr))
                    }
                    Selection::NotSelected(last_window_end) => {
                        *sinitial = None;
                        // Preserve an existing Final value, so that we only save
                        // the earliest Final after the end of a window.
                        if *sfinal {
                            *sfinal = false;
                            // If it is before the beginning of the first
                            // window, we may not have any "last_window_end" to
                            // work with, in which case we simply do not emit
                            // any Final value. There would be little point in
                            // doing so anyway.
                            match last_window_end {
                                Some(last_window_end) => {
                                    scurr.map(|scurr| SignalValue::Final(last_window_end, k, scurr))
                                }
                                None => None,
                            }
                        } else {
                            None
                        }
                    }
                }
            }
            SignalUpdate::FinishedWithUpdate(k, v) => {
                let entry = map.entry(k.clone());
                let mut temp;
                let mut temp2;
                let (scurr, sinitial, sfinal) = match entry {
                    Entry::Occupied(entry) => match selected {
                        Selection::Selected => {
                            temp = entry.remove();
                            &mut temp
                        }
                        _ => {
                            temp2 = entry;
                            temp2.get_mut()
                        }
                    },
                    Entry::Vacant(_) => {
                        temp = (None, None, false);
                        &mut temp
                    }
                };

                let scurr = replace(scurr, None);
                match selected {
                    Selection::Selected => scurr.map(|_| SignalValue::Final(x.to_kdim(), k, v)),
                    Selection::NotSelected(last_window_end) => {
                        *sinitial = None;
                        // Preserve an existing Final value, so that we only save
                        // the earliest Final after the end of a window.
                        if *sfinal {
                            *sfinal = false;
                            match last_window_end {
                                Some(last_window_end) => {
                                    scurr.map(|_| SignalValue::Final(last_window_end, k, v))
                                }
                                None => None,
                            }
                        } else {
                            None
                        }
                    }
                }
            }
            SignalUpdate::UpdateWith(k, f) => {
                let scurr = match map.get(&k) {
                    Some((Some(v), _, _)) => Some(v),
                    _ => None,
                };
                process(f.call(&k, scurr), map, selected, x)
            }
        }
    }

    let mut map: BTreeMap<K, (Option<V>, Option<V>, bool)> = BTreeMap::new();
    let mut selected = Selection::NotSelected(None);

    #[for_await]
    for x in stream {
        match select(&x) {
            WindowUpdate::None => (),
            WindowUpdate::Start(kdim) => {
                // We don't care about the order, as signals are assumed to be
                // independent
                let mut iter = map.iter_mut().into_iter();

                loop {
                    let mut initial_item = None;
                    let mut final_item = None;
                    match iter.next() {
                        Some(i) => {
                            let (k, (v, sinitial, sfinal)) = i;
                            *sfinal = false;

                            let sinitial = take(sinitial);
                            if let Some(v) = sinitial {
                                initial_item =
                                    Some(SignalValue::Initial(kdim.clone(), k.clone(), v));
                            }

                            if let Some(v) = v {
                                final_item = Some(SignalValue::WindowStart(
                                    kdim.clone(),
                                    k.clone(),
                                    v.clone(),
                                ));
                            }
                        }
                        None => break,
                    }
                    if let Some(item) = initial_item {
                        yield item;
                    }
                    if let Some(item) = final_item {
                        yield item;
                    }
                }
                selected = Selection::Selected;
            }
            WindowUpdate::End(kdim) => {
                let mut iter = map.iter_mut().into_iter();
                loop {
                    let mut item = None;
                    match iter.next() {
                        Some(i) => {
                            let (k, (v, _, _)) = i;
                            if let Some(v) = v {
                                item = Some(SignalValue::WindowEnd(
                                    kdim.clone(),
                                    k.clone(),
                                    v.clone(),
                                ));
                            }
                        }
                        None => break,
                    }
                    if let Some(item) = item {
                        yield item;
                    }
                }
                selected = Selection::NotSelected(Some(kdim));
            }

            WindowUpdate::LastEnd(kdim) => {
                let mut iter = map.iter_mut().into_iter();
                loop {
                    let mut item = None;
                    match iter.next() {
                        Some(i) => {
                            let (k, (v, _, _)) = i;
                            if let Some(v) = v {
                                item = Some(SignalValue::WindowEnd(
                                    kdim.clone(),
                                    k.clone(),
                                    v.clone(),
                                ));
                            }
                        }
                        None => break,
                    }
                    if let Some(item) = item {
                        yield item;
                    }
                }
                // The stream is considered finished when we receive the
                // Finished update, as it marks the end of the last window.
                break;
            }
        }

        #[for_await]
        for update in splitter.split(&x) {
            match process(update, &mut map, selected.clone(), &x) {
                Some(_x) => {
                    yield _x;
                }
                None => (),
            }
        }
    }
}

#[derive(Debug, Clone)]
enum StreamPairItem<T1, T2> {
    Left(T1),
    Right(T2),
}

fn merge<S1, T1, S2, T2>(s1: S1, s2: S2) -> impl Stream<Item = StreamPairItem<T1, T2>>
where
    S1: Stream<Item = T1>,
    S2: Stream<Item = T2>,
{
    let s1 = s1.map(StreamPairItem::<T1, T2>::Left);
    let s2 = s2.map(StreamPairItem::<T1, T2>::Right);
    select(s1, s2)
}

#[stream(item = (T1, Option<T2>))]
pub async fn left_join<T1, S1, F1, T2, S2, F2, K>(s1: S1, mut f1: F1, s2: S2, mut f2: F2)
where
    S1: Stream<Item = T1>,
    S2: Stream<Item = T2>,
    F1: FnMut(&T1) -> Option<K>,
    F2: FnMut(&T2) -> Option<K>,
    K: Ord,
    T1: Clone,
    T2: Clone,
{
    let s1 = s1.map(StreamPairItem::<T1, T2>::Left);
    let s2 = s2.map(StreamPairItem::<T1, T2>::Right);

    let mut map = BTreeMap::<K, T2>::new();

    #[for_await]
    // Systematically poll s2 before polling s1, so that the map is as up to
    // date as can be.
    for x in select_ordered(s2, s1) {
        match x {
            StreamPairItem::Left(x) => {
                if let Some(k) = f1(&x) {
                    let entry = map.entry(k);
                    let right = match entry {
                        Entry::Occupied(entry) => Some(entry.get().clone()),
                        _ => None,
                    };
                    yield (x, right);
                }
            }
            StreamPairItem::Right(x) => {
                if let Some(k) = f2(&x) {
                    map.insert(k, x);
                }
            }
        }
    }
}

pub fn right_join<T1, S1, F1, T2, S2, F2, K>(
    s1: S1,
    f1: F1,
    s2: S2,
    f2: F2,
) -> impl Stream<Item = (Option<T1>, T2)>
where
    S1: Stream<Item = T1>,
    S2: Stream<Item = T2>,
    F1: FnMut(&T1) -> Option<K>,
    F2: FnMut(&T2) -> Option<K>,
    K: Ord,
    T1: Clone,
    T2: Clone,
{
    left_join(s2, f2, s1, f1).map(|(left, right)| (right, left))
}

#[derive(Clone, Debug)]
pub enum FullJoinItem<L, R> {
    Left(L),
    Right(R),
    Both(L, R),
}

impl<L, R> FullJoinItem<L, R> {
    fn left(&self) -> Option<&L> {
        use FullJoinItem::*;
        match self {
            Left(x) => Some(x),
            Right(_) => None,
            Both(l, _) => Some(l),
        }
    }

    fn right(&self) -> Option<&R> {
        use FullJoinItem::*;
        match self {
            Left(_) => None,
            Right(x) => Some(x),
            Both(_, r) => Some(r),
        }
    }

    fn both(self) -> Option<(L, R)> {
        use FullJoinItem::*;
        match self {
            Left(_) => None,
            Right(_) => None,
            Both(l, r) => Some((l, r)),
        }
    }
}
impl<KDim, T1, T2> HasKDim for FullJoinItem<T1, T2>
where
    T1: HasKDim<KDim = KDim>,
    T2: HasKDim<KDim = KDim>,
    KDim: Clone + Ord,
{
    type KDim = KDim;
    fn to_kdim(&self) -> Self::KDim {
        use FullJoinItem::*;
        match self {
            Left(x) => x.to_kdim(),
            Right(x) => x.to_kdim(),
            Both(l, r) => max(l.to_kdim(), r.to_kdim()),
        }
    }
}

// impl<L, R> From<FullJoinItem<L, R>> for (Option<L>, Option<R>) {
//     fn from(x: FullJoinItem<L, R>) -> Self {
//         x.any()
//     }
// }

#[stream(item = FullJoinItem<T1, T2>)]
pub async fn full_join<T1, S1, F1, T2, S2, F2, K>(s1: S1, mut f1: F1, s2: S2, mut f2: F2)
where
    S1: Stream<Item = T1>,
    S2: Stream<Item = T2>,
    F1: FnMut(&T1) -> Option<K>,
    F2: FnMut(&T2) -> Option<K>,
    K: Ord + Clone,
    T1: Clone,
    T2: Clone,

    // TODO: remove that
    T1: Debug,
    T2: Debug,
{
    // TODO: maybe merge both maps with (T1, T2) values ?
    let mut map1 = BTreeMap::<K, T1>::new();
    let mut map2 = BTreeMap::<K, T2>::new();

    #[for_await]
    for x in merge(s1, s2) {
        // Since we own the Vec for now, we are free to drain it and move the
        // items out of the Vec. This will not change its capacity so the memory
        // will be stay allocated, so we can later give it back to the Cell.
        match x {
            StreamPairItem::Left(x) => {
                if let Some(k) = f1(&x) {
                    map1.insert(k.clone(), x.clone());

                    let entry = map2.entry(k);
                    match entry {
                        Entry::Occupied(entry) => {
                            yield FullJoinItem::Both(x, entry.get().clone());
                        }
                        _ => yield FullJoinItem::Left(x),
                    }
                }
            }
            StreamPairItem::Right(x) => {
                if let Some(k) = f2(&x) {
                    map2.insert(k.clone(), x.clone());

                    let entry = map1.entry(k);
                    match entry {
                        Entry::Occupied(entry) => {
                            yield FullJoinItem::Both(entry.get().clone(), x);
                        }
                        _ => yield FullJoinItem::Right(x),
                    }
                }
            }
        }
    }
}

pub fn inner_join<T1, S1, F1, T2, S2, F2, K>(
    s1: S1,
    f1: F1,
    s2: S2,
    f2: F2,
) -> impl Stream<Item = (T1, T2)>
where
    S1: Stream<Item = T1>,
    S2: Stream<Item = T2>,
    F1: FnMut(&T1) -> Option<K>,
    F2: FnMut(&T2) -> Option<K>,
    K: Ord + Clone,
    T1: Clone,
    T2: Clone,

    // TODO: remove that
    T1: Debug,
    T2: Debug,
{
    full_join(s1, f1, s2, f2).filter_map(|x| {
        future::ready(match x {
            FullJoinItem::Both(l, r) => Some((l, r)),
            _ => None,
        })
    })
}

#[stream(item = SignalValue<KDim, K, (V1, Option<V2>)>)]
pub async fn left_join_signal<V1, S1, V2, S2, KDim, K>(s1: S1, s2: S2)
where
    S1: Stream<Item = SignalValue<KDim, K, V1>>,
    S2: Stream<Item = SignalValue<KDim, K, V2>>,
    K: Ord + Clone,
    KDim: Clone,
    V2: Clone,
{
    let s1 = s1.map(StreamPairItem::<S1::Item, S2::Item>::Left);
    let s2 = s2.map(StreamPairItem::<S1::Item, S2::Item>::Right);

    let mut map = BTreeMap::<K, V2>::new();

    #[for_await]
    // Systematically poll s2 before polling s1, so that the map is as up to
    // date as can be.
    for x in select_ordered(s2, s1) {
        match x {
            StreamPairItem::Left(x) => {
                let k = x.key().clone();
                let right = match map.entry(k) {
                    Entry::Occupied(entry) => Some(entry.get().clone()),
                    _ => None,
                };
                yield x.map(|v| (v, right));
            }
            StreamPairItem::Right(x) => {
                let k = x.key();
                match x {
                    SignalValue::WindowEnd(..) | SignalValue::Final(..) => {
                        map.remove(k);
                    }
                    _ => {
                        map.insert(k.clone(), x.value().clone());
                    }
                }
            }
        }
    }
}

pub fn right_join_signal<V1, S1, V2, S2, KDim, K>(
    s1: S1,
    s2: S2,
) -> impl Stream<Item = SignalValue<KDim, K, (Option<V1>, V2)>>
where
    S1: Stream<Item = SignalValue<KDim, K, V1>>,
    S2: Stream<Item = SignalValue<KDim, K, V2>>,
    K: Ord + Clone,
    KDim: Clone,
    V1: Clone,
{
    left_join_signal(s2, s1).map(|x| x.map(|(l, r)| (r, l)))
}

#[stream(item = SignalValue<KDim, K, FullJoinItem<V1, V2>>)]
pub async fn full_join_signal<K, KDim, V1, V2, S1, S2>(s1: S1, s2: S2)
where
    KDim: Clone,
    K: Ord + Clone,
    V1: Clone,
    V2: Clone,
    S1: Stream<Item = SignalValue<KDim, K, V1>>,
    S2: Stream<Item = SignalValue<KDim, K, V2>>,
{
    let mut state =
        BTreeMap::<K, FullJoinItem<SignalValue<KDim, K, V1>, SignalValue<KDim, K, V2>>>::new();

    #[for_await]
    for x in merge(s1, s2) {
        match x {
            StreamPairItem::Left(mut left) => {
                let entry = state.entry(left.key().clone());

                match entry {
                    Entry::Occupied(mut occupied_entry) => {
                        let state_ref = occupied_entry.get_mut();

                        match state_ref {
                            FullJoinItem::Both(_, right) | FullJoinItem::Right(right) => {
                                yield SignalValue::Current(
                                    left.kdim().clone(),
                                    left.key().clone(),
                                    FullJoinItem::Both(left.value().clone(), right.value().clone()),
                                );
                            }
                            FullJoinItem::Left(_) => {
                                yield left.clone().map(FullJoinItem::Left);
                            }
                        };

                        match left {
                            SignalValue::WindowEnd(..) | SignalValue::Final(..) => {
                                match state_ref {
                                    FullJoinItem::Both(_, right) => {
                                        // Ideally we would steal ownership of
                                        // "right" instead of cloning it, but
                                        // since get_mut() only gives access to
                                        // &mut and not an owned value we can't
                                        // do that.
                                        *state_ref = FullJoinItem::Right(right.clone());
                                    }
                                    FullJoinItem::Left(_) => {
                                        occupied_entry.remove();
                                    }
                                    _ => (),
                                }
                            }
                            _ => {
                                match state_ref {
                                    FullJoinItem::Both(_left, _) | FullJoinItem::Left(_left) => {
                                        swap(_left, &mut left);
                                    }
                                    FullJoinItem::Right(right) => {
                                        // Ideally we would steal ownership of
                                        // "right" instead of cloning it, but
                                        // since get_mut() only gives access to
                                        // &mut and not an owned value we can't
                                        // do that.
                                        *state_ref = FullJoinItem::Both(left, right.clone())
                                    }
                                };
                            }
                        };
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(FullJoinItem::Left(left.clone()));
                        yield left.map(FullJoinItem::Left);
                    }
                }
            }
            StreamPairItem::Right(mut right) => {
                let entry = state.entry(right.key().clone());

                match entry {
                    Entry::Occupied(mut occupied_entry) => {
                        let state_ref = occupied_entry.get_mut();

                        match state_ref {
                            FullJoinItem::Both(left, _) | FullJoinItem::Left(left) => {
                                yield SignalValue::Current(
                                    right.kdim().clone(),
                                    right.key().clone(),
                                    FullJoinItem::Both(left.value().clone(), right.value().clone()),
                                );
                            }
                            FullJoinItem::Right(_) => {
                                yield right.clone().map(FullJoinItem::Right);
                            }
                        };

                        match right {
                            SignalValue::WindowEnd(..) | SignalValue::Final(..) => {
                                match state_ref {
                                    FullJoinItem::Both(left, _) => {
                                        // Ideally we would steal ownership of
                                        // "left" instead of cloning it, but
                                        // since get_mut() only gives access to
                                        // &mut and not an owned value we can't
                                        // do that.
                                        *state_ref = FullJoinItem::Left(left.clone());
                                    }
                                    FullJoinItem::Right(_) => {
                                        occupied_entry.remove();
                                    }
                                    _ => (),
                                }
                            }
                            _ => {
                                match state_ref {
                                    FullJoinItem::Both(_, _right) | FullJoinItem::Right(_right) => {
                                        swap(_right, &mut right);
                                    }
                                    FullJoinItem::Left(left) => {
                                        // Ideally we would steal ownership of
                                        // "left" instead of cloning it, but
                                        // since get_mut() only gives access to
                                        // &mut and not an owned value we can't
                                        // do that.
                                        *state_ref = FullJoinItem::Both(left.clone(), right)
                                    }
                                };
                            }
                        };
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(FullJoinItem::Right(right.clone()));
                        yield right.map(FullJoinItem::Right);
                    }
                }
            }
        }
    }
}

#[stream(item = SignalValue<KDim, K, (V1, V2)>)]
pub async fn inner_join_signal<K, KDim, V1, V2, S1, S2>(s1: S1, s2: S2)
where
    KDim: Clone,
    K: Ord + Clone,
    V1: Clone,
    V2: Clone,
    S1: Stream<Item = SignalValue<KDim, K, V1>>,
    S2: Stream<Item = SignalValue<KDim, K, V2>>,

    // TODO: remove that
    SignalValue<KDim, K, V1>: Debug,
    SignalValue<KDim, K, V2>: Debug,
{
    let mut state =
        BTreeMap::<K, FullJoinItem<SignalValue<KDim, K, V1>, SignalValue<KDim, K, V2>>>::new();

    #[for_await]
    for x in merge(s1, s2) {
        match x {
            StreamPairItem::Left(left) => {
                let entry = state.entry(left.key().clone());

                match entry {
                    Entry::Occupied(mut occupied_entry) => {
                        let state_ref = occupied_entry.get_mut();

                        match state_ref {
                            FullJoinItem::Both(_left, SignalValue::WindowEnd(_, _, _right)) => {
                                match left {
                                    SignalValue::Current(..)
                                    | SignalValue::WindowStart(..)
                                    | SignalValue::Initial(..) => {
                                        *_left = left.clone();
                                    }
                                    SignalValue::WindowEnd(..) => {
                                        occupied_entry.remove();
                                    }
                                    SignalValue::Final(..) => {
                                        let (kdim, k, v) = left.into();
                                        // Simply move the value out of the entry
                                        let _right = match occupied_entry.remove() {
                                            FullJoinItem::Both(_, _right) => _right.into_value(),
                                            _ => unsafe { unreachable_unchecked() },
                                        };
                                        yield SignalValue::Final(kdim, k, (v, _right));
                                    }
                                }
                            }
                            FullJoinItem::Both(_left, _right) => match left {
                                SignalValue::Current(..) => {
                                    *_left = left.clone();
                                    let (kdim, k, v) = left.into();
                                    yield SignalValue::Current(
                                        kdim,
                                        k,
                                        (v, _right.value().clone()),
                                    );
                                }
                                SignalValue::WindowEnd(..) => {
                                    *_left = left.clone();
                                    let (kdim, k, v) = left.into();
                                    yield SignalValue::WindowEnd(
                                        kdim,
                                        k,
                                        (v, _right.value().clone()),
                                    );
                                }
                                SignalValue::Final(..) => {
                                    let new_state = FullJoinItem::Right(_right.clone());
                                    let (kdim, k, v) = left.into();
                                    yield SignalValue::Final(kdim, k, (v, _right.value().clone()));
                                    *state_ref = new_state;
                                }

                                _ => unreachable!(),
                            },
                            FullJoinItem::Left(_left) => match left {
                                SignalValue::Current(..)
                                | SignalValue::WindowStart(..)
                                | SignalValue::Initial(..) => {
                                    *_left = left.clone();
                                }
                                SignalValue::WindowEnd(..) | SignalValue::Final(..) => {
                                    occupied_entry.remove();
                                }
                            },
                            FullJoinItem::Right(_right) => match left {
                                SignalValue::Initial(..) => {
                                    let new_state =
                                        FullJoinItem::Both(left.clone(), _right.clone());
                                    let (kdim, k, v) = left.into();
                                    yield SignalValue::Initial(
                                        kdim,
                                        k,
                                        (v, _right.value().clone()),
                                    );
                                    *state_ref = new_state;
                                }
                                SignalValue::WindowStart(..) => {
                                    let new_state =
                                        FullJoinItem::Both(left.clone(), _right.clone());
                                    let (kdim, k, v) = left.into();
                                    yield SignalValue::WindowStart(
                                        kdim,
                                        k,
                                        (v, _right.value().clone()),
                                    );
                                    *state_ref = new_state;
                                }
                                _ => unreachable!(),
                            },
                        }
                    }
                    Entry::Vacant(vacant_entry) => match left {
                        SignalValue::Current(..)
                        | SignalValue::WindowStart(..)
                        | SignalValue::Initial(..) => {
                            vacant_entry.insert(FullJoinItem::Left(left));
                        }
                        _ => (),
                    },
                }
            }
            StreamPairItem::Right(right) => {
                let entry = state.entry(right.key().clone());

                match entry {
                    Entry::Occupied(mut occupied_entry) => {
                        let state_ref = occupied_entry.get_mut();

                        match state_ref {
                            FullJoinItem::Both(SignalValue::WindowEnd(_, _, _left), _right) => {
                                match right {
                                    SignalValue::Current(..)
                                    | SignalValue::WindowStart(..)
                                    | SignalValue::Initial(..) => {
                                        *_right = right.clone();
                                    }
                                    SignalValue::WindowEnd(..) => {
                                        occupied_entry.remove();
                                    }
                                    SignalValue::Final(..) => {
                                        let (kdim, k, v) = right.into();
                                        // Simply move the value out of the entry
                                        let _left = match occupied_entry.remove() {
                                            FullJoinItem::Both(_left, _) => _left.into_value(),
                                            _ => unsafe { unreachable_unchecked() },
                                        };
                                        yield SignalValue::Final(kdim, k, (_left, v));
                                    }
                                }
                            }
                            FullJoinItem::Both(_left, _right) => match right {
                                SignalValue::Current(..) => {
                                    *_right = right.clone();
                                    let (kdim, k, v) = right.into();
                                    yield SignalValue::Current(kdim, k, (_left.value().clone(), v));
                                }
                                SignalValue::WindowEnd(..) => {
                                    *_right = right.clone();
                                    let (kdim, k, v) = right.into();
                                    yield SignalValue::WindowEnd(
                                        kdim,
                                        k,
                                        (_left.value().clone(), v),
                                    );
                                }
                                SignalValue::Final(..) => {
                                    let new_state = FullJoinItem::Left(_left.clone());
                                    let (kdim, k, v) = right.into();
                                    yield SignalValue::Final(kdim, k, (_left.value().clone(), v));
                                    *state_ref = new_state;
                                }

                                _ => unreachable!(),
                            },
                            FullJoinItem::Right(_right) => match right {
                                SignalValue::Current(..)
                                | SignalValue::WindowStart(..)
                                | SignalValue::Initial(..) => {
                                    *_right = right.clone();
                                }
                                SignalValue::WindowEnd(..) | SignalValue::Final(..) => {
                                    occupied_entry.remove();
                                }
                            },
                            FullJoinItem::Left(_left) => match right {
                                SignalValue::Initial(..) => {
                                    let new_state =
                                        FullJoinItem::Both(_left.clone(), right.clone());
                                    let (kdim, k, v) = right.into();
                                    yield SignalValue::Initial(kdim, k, (_left.value().clone(), v));
                                    *state_ref = new_state;
                                }
                                SignalValue::WindowStart(..) => {
                                    let new_state =
                                        FullJoinItem::Both(_left.clone(), right.clone());
                                    let (kdim, k, v) = right.into();
                                    yield SignalValue::WindowStart(
                                        kdim,
                                        k,
                                        (_left.value().clone(), v),
                                    );
                                    *state_ref = new_state;
                                }
                                _ => unreachable!(),
                            },
                        }
                    }
                    Entry::Vacant(vacant_entry) => match right {
                        SignalValue::Current(..)
                        | SignalValue::WindowStart(..)
                        | SignalValue::Initial(..) => {
                            vacant_entry.insert(FullJoinItem::Right(right));
                        }
                        _ => (),
                    },
                }
            }
        }
    }
}

pub struct GroupStream<T> {
    curr_x: Rc<Cell<Poll<Option<T>>>>,
}

impl<T> GroupStream<T> {
    fn new() -> GroupStream<T> {
        GroupStream {
            curr_x: Rc::new(Cell::new(Poll::Pending)),
        }
    }
}

impl<T> Stream for GroupStream<T>
where
    T: Clone,
{
    type Item = T;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let x = self.curr_x.replace(Poll::Pending);
        match &x {
            Poll::Ready(None) => {}
            _ => {
                cx.waker().wake_by_ref();
            }
        }
        x
    }
}

#[pin_project]
struct Group<Fut, T> {
    curr_x: Rc<Cell<Poll<Option<T>>>>,
    #[pin]
    future: Fut,
}

impl<Fut, T, O> Group<Fut, T>
where
    Fut: Future<Output = O>,
{
    fn new<F, K>(k: K, mut f: F) -> Group<Fut, T>
    where
        F: FnMut(K, GroupStream<T>) -> Fut,
    {
        let stream = GroupStream::new();
        let curr_x = stream.curr_x.clone();
        let future = f(k, stream);
        Group { curr_x, future }
    }

    fn push(self: Pin<&mut Self>, x: Option<T>) -> Option<O> {
        let waker = make_noop_waker();
        let mut ctx = Context::from_waker(&waker);

        let mut this = self.project();
        this.curr_x.set(Poll::Ready(x));
        loop {
            let res = this.future.as_mut().poll(&mut ctx);
            match res {
                Poll::Pending => {
                    // Check if the stream's latest item has been consumed yet.
                    // If not, we keep polling the future until it consumes it.
                    let curr_x = this.curr_x.replace(Poll::Pending);
                    match curr_x {
                        Poll::Pending => {
                            break None;
                        }
                        _ => {
                            this.curr_x.set(curr_x);
                        }
                    }
                }
                Poll::Ready(x) => break Some(x),
            }
        }
    }
}

pub async fn group_by<T, S, K, KF, Fut, CF, Result>(
    stream: S,
    mut keyf: KF,
    consume: CF,
) -> BTreeMap<K, Result>
where
    T: Clone,
    S: Stream<Item = T>,
    K: Ord + Clone,
    KF: FnMut(&T) -> K,
    Fut: Future<Output = Result>,
    CF: FnMut(K, GroupStream<T>) -> Fut + Clone,
{
    let mut group_map = BTreeMap::<K, Option<Pin<Box<Group<Fut, T>>>>>::new();
    let mut res_map = BTreeMap::<K, Result>::new();

    #[for_await]
    for x in stream {
        let k = keyf(&x);
        match group_map.entry(k.clone()) {
            Entry::Occupied(mut entry) => {
                let group = entry.get_mut();
                match group {
                    // Group already finished and result already in res_map,
                    // nothing to do.
                    None => {}
                    // We have a group already allocated, push some data into it
                    Some(group) => match group.as_mut().push(Some(x)) {
                        // If we have the final result for that group, insert it
                        // in res_map and then mark the group as completed using
                        // None in group_map
                        Some(x) => {
                            res_map.insert(k, x);
                            entry.insert(None);
                        }
                        None => {}
                    },
                }
            }
            Entry::Vacant(entry) => {
                let mut group = Box::pin(Group::new(k.clone(), consume.clone()));
                entry.insert(match group.as_mut().push(Some(x)) {
                    Some(x) => {
                        res_map.insert(k, x);
                        None
                    }
                    None => Some(group),
                });
            }
        }
    }

    // Close the stream for each aggregation function still consuming it.
    for (k, group) in group_map {
        match group {
            None => {}
            Some(mut group) => {
                let res = group.as_mut().push(None);
                match res {
                    None => panic!(
                        "The aggregation function failed to terminate when the stream was closed"
                    ),
                    Some(x) => {
                        res_map.insert(k.clone(), x);
                    }
                }
            }
        }
    }
    res_map
}

#[pin_project]
pub struct BatchStream<S>
where
    S: Stream,
{
    finished: bool,
    #[pin]
    inner: S,
    batch: Rc<Cell<Vec<S::Item>>>,
    batch_size: usize,
}

impl<S> BatchStream<S>
where
    S: Stream,
{
    fn new(stream: S) -> Self {
        BatchStream {
            inner: stream,
            finished: false,
            batch: Rc::new(Cell::new(Vec::new())),
            batch_size: 0,
        }
    }
}

impl<S> Stream for BatchStream<S>
where
    S: Stream,
    S::Item: Clone,
{
    type Item = Rc<Cell<Vec<S::Item>>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.as_mut().project();
        if *this.finished {
            Poll::Ready(None)
        } else {
            // Steal the Vec behind the Rc if as no-one else kept
            // a reference to it to avoid a new allocation.
            if let Some(vec) = Rc::get_mut(this.batch) {
                let vec = vec.get_mut();
                vec.clear();

                loop {
                    match this.inner.as_mut().poll_next(cx) {
                        Poll::Ready(Some(x)) => vec.push(x),
                        Poll::Pending => {
                            return if vec.is_empty() {
                                Poll::Pending
                            } else {
                                *this.batch_size = vec.len();
                                Poll::Ready(Some(this.batch.clone()))
                            }
                        }
                        Poll::Ready(None) => {
                            *this.finished = true;
                            return Poll::Ready(if vec.is_empty() {
                                None
                            } else {
                                *this.batch_size = vec.len();
                                Some(this.batch.clone())
                            });
                        }
                    }
                }
            } else {
                // // Otherwise, existing users can keep using the vector and
                // // we simply allocate a new one. Use the previous vector
                // // length for this new vector as it is quite likely that the
                // // stream is following a specific pattern of yielding N
                // // items at a time.
                let mut vec = Vec::with_capacity(*this.batch_size);
                loop {
                    match this.inner.as_mut().poll_next(cx) {
                        Poll::Ready(Some(x)) => vec.push(x),
                        Poll::Pending => {
                            return if vec.is_empty() {
                                Poll::Pending
                            } else {
                                *this.batch_size = vec.len();
                                let batch = Rc::new(Cell::new(vec));
                                *this.batch = batch.clone();
                                Poll::Ready(Some(batch))
                            }
                        }
                        Poll::Ready(None) => {
                            *this.finished = true;
                            return Poll::Ready(if vec.is_empty() {
                                None
                            } else {
                                *this.batch_size = vec.len();
                                let batch = Rc::new(Cell::new(vec));
                                *this.batch = batch.clone();
                                Some(batch)
                            });
                        }
                    }
                }
            }
        }
    }
}

pub trait StreamExt: Stream {
    // type JoinedStream<F1, T2, S2, F2, K>;

    // fn left_join<F1, T2, S2, F2, K>(
    //     self,
    //     f1: F1,
    //     s2: S2,
    //     f2: F2,
    // ) -> Self::JoinedStream<F1, T2, S2, F2, K>
    // where
    //     S2: Stream<Item = T2>,
    //     F1: FnMut(&Self::Item) -> K,
    //     F2: FnMut(&T2) -> K,
    //     K: Ord,
    //     Self::Item: Clone,
    //     Self: Sized,
    //     T2: Clone;

    // type ModifiedStream;
    fn apply<F, T>(self, f: F) -> T
    where
        Self: Sized,
        F: FnOnce(Self) -> T;

    fn batch(self) -> BatchStream<Self>
    where
        Self: Sized;
}

impl<S: Stream> StreamExt for S {
    #[inline(always)]
    fn apply<F, T>(self, f: F) -> T
    where
        F: FnOnce(Self) -> T,
    {
        f(self)
    }

    #[inline(always)]
    fn batch(self) -> BatchStream<Self> {
        BatchStream::new(self)
    }

    // type JoinedStream<F1, T2, S2, F2, K> = impl Stream<Item = (Self::Item, Option<T2>)>;
    // fn left_join<F1, T2, S2, F2, K>(
    //     self,
    //     f1: F1,
    //     s2: S2,
    //     f2: F2,
    // ) -> Self::JoinedStream<F1, T2, S2, F2, K>
    // where
    //     S2: Stream<Item = T2>,
    //     F1: FnMut(&Self::Item) -> K,
    //     F2: FnMut(&T2) -> K,
    //     K: Ord,
    //     Self::Item: Clone,
    //     Self: Sized,
    //     T2: Clone,
    // {
    //     left_join(self, f1, s2, f2)
    // }
}

#[pin_project]
pub struct SelectOrderedStream<S1, S2> {
    #[pin]
    s1: Fuse<S1>,
    #[pin]
    s2: Fuse<S2>,
}

pub fn select_ordered<S1, S2, T>(s1: S1, s2: S2) -> SelectOrderedStream<S1, S2>
where
    S1: Stream<Item = T>,
    S2: Stream<Item = T>,
{
    SelectOrderedStream {
        s1: s1.fuse(),
        s2: s2.fuse(),
    }
}

impl<S1, S2, T> Stream for SelectOrderedStream<S1, S2>
where
    S1: Stream<Item = T>,
    S2: Stream<Item = T>,
{
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        return match this.s1.poll_next(cx) {
            x @ Poll::Ready(Some(_)) => x,
            _ => this.s2.poll_next(cx),
        };
    }
}

type ErrorMsg = std::string::String;

#[derive(Clone, Debug, Serialize, PartialEq, Eq, JsonSchema)]
pub enum AnalysisError {
    #[serde(rename = "input")]
    Input(ErrorMsg),
    #[serde(rename = "serialization")]
    Serialization(ErrorMsg),
}

// Allow serializing any Box<dyn AnalysisValue>, which is not possible with
// serde alone since Serialize is not object-safe.
pub trait AnalysisValue: Debug + erased_serde::Serialize {
    fn get_schema(&self) -> RootSchema;
}
serialize_trait_object!(AnalysisValue);
impl<T: Serialize + Debug + JsonSchema> AnalysisValue for T {
    fn get_schema(&self) -> RootSchema {
        schema_for!(T)
    }
}

fn serialize_ok<S: Serializer>(x: &Box<dyn AnalysisValue + Send>, s: S) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct WithSchema<T> {
        schema: RootSchema,
        value: T,
    }
    WithSchema {
        schema: x.get_schema(),
        value: x,
    }
    .serialize(s)
}

#[derive(Debug, Serialize)]
pub enum AnalysisResult {
    #[serde(rename = "err")]
    Err(AnalysisError),
    #[serde(rename = "ok")]
    #[serde(serialize_with = "serialize_ok")]
    Ok(Box<dyn AnalysisValue + Send>),
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct Map<K, V>(Vec<(K, V)>);

impl<K, V> Map<K, V> {
    fn new(map: BTreeMap<K, V>) -> Self {
        Map(map.into_iter().collect())
    }
}

#[derive(Debug, Serialize, JsonSchema)]
struct InbandTable<Item> {
    columns: Vec<&'static str>,
    data: Vec<Item>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub enum OutofbandTableFormat {
    Feather,
}

#[derive(Debug, Serialize)]
pub struct OutofbandTable<P>
where
    P: AsRef<Path> + JsonSchema,
{
    path: P,
    format: OutofbandTableFormat,
    schema: RootSchema,
    columns: Vec<&'static str>,
}

// Hack: since RootSchema itself does not implement JsonSchema, use a
// placeholder implementation.
// https://github.com/GREsau/schemars/issues/191
const _: () = {
    #[derive(JsonSchema)]
    pub struct OutofbandTable<T> {
        _phantom: std::marker::PhantomData<T>,
    }

    impl<P> JsonSchema for crate::analysis::OutofbandTable<P>
    where
        P: AsRef<Path> + JsonSchema,
    {
        #[inline]
        fn schema_name() -> std::string::String {
            <OutofbandTable<P> as JsonSchema>::schema_name()
        }
        #[inline]
        fn json_schema(gen: &mut SchemaGenerator) -> Schema {
            <OutofbandTable<P> as JsonSchema>::json_schema(gen)
        }
        #[inline]
        fn is_referenceable() -> bool {
            <OutofbandTable<P> as JsonSchema>::is_referenceable()
        }
    }
};

pub trait Row {
    type AsTuple;
    type AsRefTuple<'a>;
    fn columns() -> Vec<&'static str>;
}

impl AnalysisResult {
    fn _new<T: AnalysisValue + JsonSchema + Send + 'static>(x: T) -> Self {
        AnalysisResult::Ok(Box::new(x))
    }

    pub fn new<T: AnalysisValue + JsonSchema + Send + 'static, EventStream>(
        x: T,
    ) -> AnalysisResultBuilder<EventStream> {
        Box::new(move |_conf, _name| Box::pin(async { Self::_new(x) }))
    }

    pub fn from_row_stream<S, Item, EventStream>(stream: S) -> AnalysisResultBuilder<EventStream>
    where
        S: Stream<Item = Item> + Send + 'static,
        Item: Row
            + Into<<Item as Row>::AsTuple>
            + ArrowSerialize
            + ArrowField<Type = Item>
            + Send
            + 'static,
        <Item as Row>::AsTuple: Serialize + JsonSchema + Debug + 'static,
    {
        Box::new(move |conf, name| {
            Box::pin(async move {
                // TODO: add a runtime switch for JSON vs feather output

                // Convert the records to tuples for efficient JSON encoding.
                // let data: Vec<Item::AsTuple> = stream.map(Into::into).collect().await;
                // let columns = Item::columns();
                // let data = InbandTable { data, columns };

                // use uuid::Uuid;
                // let id = Uuid::new_v4();
                let id = name;
                let path = format!("{id}.feather");
                let path = conf.out_path.join(path);

                let data: arrow2::error::Result<OutofbandTable<PathBuf>> = try {
                    crate::arrow::write_stream(&path, stream).await?;
                    OutofbandTable {
                        path: path,
                        format: OutofbandTableFormat::Feather,
                        columns: Item::columns(),
                        schema: schema_for!(Item::AsTuple),
                    }
                };

                match data {
                    Err(err) => AnalysisResult::Err(AnalysisError::Serialization(err.to_string())),
                    Ok(data) => Self::_new(data),
                }
            })
        })
    }

    pub async fn from_map<K, V, EventStream>(
        map: BTreeMap<K, V>,
    ) -> AnalysisResultBuilder<EventStream>
    where
        K: Serialize + JsonSchema + Debug + Send + 'static,
        V: Serialize + JsonSchema + Debug + Send + 'static,
    {
        Box::new(move |_conf, _name| Box::pin(async { Self::_new(Map::new(map)) }))
    }
}

type AnalysisResultBuilder<S> = Box<
    dyn FnOnce(
        AnalysisConf<S>,
        &'static str,
    ) -> Pin<Box<dyn Future<Output = AnalysisResult> + Send>>,
>;

#[derive(Clone, Debug)]
pub struct AnalysisConf<S> {
    pub stream: S,
    pub out_path: PathBuf,
}

pub struct Analysis<S: EventStream> {
    pub name: &'static str,
    pub eventreq: EventReq,
    pub f: Box<
        dyn for<'a> Fn(
                &'a AnalysisConf<S>,
                &Value,
            ) -> Pin<Box<dyn Future<Output = AnalysisResult> + Send + 'a>>
            + Send,
    >,
}

impl<S> Analysis<S>
where
    S: EventStream + Clone + Send + 'static,
{
    pub fn new<Fut, P, F>(name: &'static str, f: F, eventreq: EventReq) -> Analysis<S>
    where
        F: Fn(S, P) -> Fut + Send + 'static,
        Fut: Future<Output = AnalysisResultBuilder<S>> + Send + 'static,
        P: for<'de> Deserialize<'de> + 'static,
        S: EventStream,
    {
        Analysis {
            name,
            f: Box::new(move |conf, x| {
                Box::pin({
                    let conf = conf.clone();
                    match serde_json::value::from_value(x.clone()) {
                        Ok(x) => {
                            { f(conf.stream.fork(), x).then(move |x| x(conf, name)) }.right_future()
                        }
                        Err(error) => async move {
                            AnalysisResult::Err(AnalysisError::Input(error.to_string()))
                        }
                        .left_future(),
                    }
                })
            }),
            eventreq,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventWindow {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "time")]
    Time(Timestamp, Timestamp),
}

pub trait HasEventReq {
    fn eventreq() -> EventReq;
}

#[macro_export]
macro_rules! analysis {
    (name: $name:ident, events: $events:tt, ($stream:ident: EventStream, $param:ident: $param_ty:ty) $body:block) => {
        pub async fn $name<S: EventStream + 'static>(
            $stream: S,
            $param: $param_ty,
        ) -> $crate::analysis::AnalysisResultBuilder<S> {
            $body
        }

        #[allow(non_camel_case_types)]
        pub struct $name {}

        impl $crate::analysis::HasEventReq for $name {
            fn eventreq() -> $crate::eventreq::EventReq {
                $crate::event_req!($events)
            }
        }
    };
}

macro_rules! build_analyses_descriptors {
    ($($path:path),* $(,)?) => {
        [
            $(
                Analysis::new(
                    stringify!($path),
                    $path,
                    <$path as HasEventReq>::eventreq(),
                ),
            )*
        ]
    };
}

pub fn get_analyses<SelectFn>() -> BTreeMap<&'static str, Analysis<TraceEventStream<SelectFn>>>
where
    SelectFn: Clone + FnMut(&Event) -> WindowUpdate<<Event as HasKDim>::KDim> + Send + 'static,
{
    // The content of this file is a single call to
    // build_analyses_descriptors!() containing the reference to all the
    // analyses in the crate.
    let analyses = include!(concat!(env!("OUT_DIR"), "/analyses_list.rs"));

    let mut map = BTreeMap::new();
    analyses.map(|ana| map.insert(ana.name, ana));
    map
}
