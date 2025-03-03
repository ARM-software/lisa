/* SPDX-License-Identifier: GPL-2.0 */

use alloc::boxed::Box;
use core::{
    marker::{PhantomData, Unpin},
    pin::Pin,
    task::{Context, Poll, RawWakerVTable, Waker},
};

enum CoroutineOutput<Y, R> {
    Yielded(Y),
    Complete(R),
}

struct YieldSlot<T> {
    yielded: Option<T>,
}

impl<T> YieldSlot<T>
where
    T: Send,
{
    fn new() -> Self {
        YieldSlot { yielded: None }
    }

    // SAFETY: The Waker in Context must have the "data" pointer set to a pointer to YieldSlot.
    unsafe fn set_yielded(cx: &mut Context<'_>, yielded: T) {
        let this = cx.waker().data() as *mut Self;
        // SAFETY: the pointer is converted back to its original type as per the safety contract.
        // SAFETY: Since each waker is used in a single Context, if we have an &mut Context, we
        // also have an &mut on the waker it wraps.
        let this: &mut Self = unsafe { this.as_mut().unwrap() };
        this.yielded = Some(yielded);
    }
}

fn poll_lifecycle<T, R, F>(future: Pin<&mut F>) -> CoroutineOutput<T, R>
where
    T: Send,
    F: Future<Output = R> + ?Sized,
{
    let mut data = YieldSlot::<T>::new();
    let waker = {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| panic!("Cannot clone this waker"),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: The vtable functions trivially uphold the RawWakerVTable contract since they do
        // nothing apart from panicking.
        //
        // On another note, the data pointer will be used by YieldSlot::set_yielded(), so "data"
        // must be a pointer to YieldSlot.
        unsafe { Waker::new(&data as *const _ as *const (), &VTABLE) }
    };
    // SAFETY: The waker is used with a single Context, never shared between multiple contexts.
    let mut cx = Context::from_waker(&waker);

    match future.poll(&mut cx) {
        Poll::Ready(x) => CoroutineOutput::Complete(x),
        Poll::Pending => {
            let x = data
                .yielded
                .take()
                .expect("LifeCycle did not store any value in the YieldSlot");
            CoroutineOutput::Yielded(x)
        }
    }
}

// This type needs to be public as it is used in the new_lifecycle!() macro, but is really a
// private type. It is critical for safety that the user cannot create instances of that type
// without going through the yield_!() macro available inside the new_lifecycle!() body.
pub struct __YieldFuture<T> {
    yielded: Option<T>,
}

impl<T> Future for __YieldFuture<T>
where
    T: Send + Unpin,
{
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut().yielded.take() {
            // First call to poll, we pass the data via the YieldSlot and report ourselves as
            // Pending. This way, poll_lifecycle() will be notified and have a chance to read from
            // the YieldSlot.
            Some(yielded) => {
                // SAFETY: This type of future should only be used in a coroutine that is processed
                // with poll_lifecycle(). If that is the case, the YieldSlot::set_yielded()
                // invariant is upheld.
                unsafe {
                    YieldSlot::<T>::set_yielded(cx, yielded);
                }
                Poll::Pending
            }
            // Second call to poll, we unblock. The async function will catch that directly and
            // carry on until the next yield/return point. poll_lifecycle() will not see our Ready
            // state. It will only see as Ready the point at which the async function returns.
            None => Poll::Ready(()),
        }
    }
}

type LifeCycleFuture<E> = Box<dyn Future<Output = Result<(), E>> + Send>;
type LifeCycleFutureBuilder<A, E> = Box<dyn FnOnce(A) -> LifeCycleFuture<E> + Send>;

enum LifeCycleCoroutineState<A, Y, E> {
    Builder(LifeCycleFutureBuilder<A, E>),
    Yielded {
        future: Pin<LifeCycleFuture<E>>,
        res: Result<Y, E>,
    },
}

struct LifeCycleCoroutine<A, Y, E> {
    // Use Option<> so we can temporarily move out of the enum when updating the state.
    state: Option<LifeCycleCoroutineState<A, Y, E>>,
}

macro_rules! new_lifecycle {
    (|$arg:pat_param| $body:expr) => {{
        let funcs = $crate::lifecycle::__LifeCycleNewFuncs::new_funcs();
        let closure = ::alloc::boxed::Box::new(move |$arg| {
            ::alloc::boxed::Box::new(async move {
                macro_rules! yield_ {
                    ($expr:expr) => {
                        funcs.new_future($expr).await
                    };
                }

                $body
            }) as ::alloc::boxed::Box<dyn ::core::future::Future<Output = _> + Send>
        });

        // SAFETY: The new_future() function is introduced in scope with an identifer from the
        // macro, so its hygiene prevents the $body to use it apart from via the yield_!() macro.
        // Since the new_future() function is the only way a __YieldFuture can be built outside of
        // the current module, and yield_!() consumes the __YieldFuture immediately by awaiting it,
        // there is no risk of the user ever passing __YieldFuture values around between
        // lifecycles, which would lead to unsoundness if a __YieldFuture<T> is expected by
        // poll_lifecycle() but __YieldFuture<U> is awaited.
        unsafe { funcs.new_lifecycle(closure) }
    }};
}
pub(crate) use new_lifecycle;

enum InternalState<A, Y, E> {
    Ready(LifeCycleCoroutine<A, Y, E>),
    Complete(FinishedKind, Result<(), E>),
}

#[derive(Clone)]
pub enum FinishedKind {
    Early,
    Normal,
}

pub enum State<Y, E> {
    Init,
    Started(Result<Y, E>),
    Finished(FinishedKind, Result<(), E>),
}

pub struct LifeCycle<A, Y, E> {
    state: InternalState<A, Y, E>,
}

// FIXME: Is there a less verbose way of forcing unification of Y across __YieldFuture and
// LifeCycle than creating this empty struct __LifeCycleNewFuncs ?
// => Maybe we can just call a no-op function that takes the other functions as parameters with
// generic parameters to force unification of the otherwise-unrelated generics.

// This struct's only purpose is to provide a unification point between the function creating a
// future and the one creating the lifecycle, so that all the type variables are equal.
pub struct __LifeCycleNewFuncs<A, Y, E> {
    _marker: PhantomData<(A, Y, E)>,
}

impl<A, Y, E> Clone for __LifeCycleNewFuncs<A, Y, E> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<A, Y, E> Copy for __LifeCycleNewFuncs<A, Y, E> {}

impl<A, Y, E> __LifeCycleNewFuncs<A, Y, E> {
    pub fn new_funcs() -> Self {
        __LifeCycleNewFuncs {
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn new_future(self, yielded: Result<Y, E>) -> __YieldFuture<Result<Y, E>> {
        __YieldFuture {
            yielded: Some(yielded),
        }
    }

    /// # Safety
    ///
    /// This function must only be used via the new_lifecycle!() macro, as it is the only public
    /// gateway to poll_lifecycle(). new_lifecycle!() ensures via the yield_!() macro that it is
    /// impossible for the user to ever create a __YieldFuture in a non-controlled way, thereby
    /// preventing any scenario where a __YieldFuture<T> is awaited by poll_lifecycle() where it
    /// would expect a __YieldFuture<U>
    #[inline]
    pub unsafe fn new_lifecycle(self, builder: LifeCycleFutureBuilder<A, E>) -> LifeCycle<A, Y, E> {
        LifeCycle {
            state: InternalState::Ready(LifeCycleCoroutine {
                state: Some(LifeCycleCoroutineState::Builder(builder)),
            }),
        }
    }
}

impl<A, Y, E> LifeCycle<A, Y, E>
where
    Y: Send,
    E: Clone + Send,
{
    pub fn state(&self) -> State<&Y, E> {
        match &self.state {
            InternalState::Ready(coro) => match &coro.state.as_ref().unwrap() {
                LifeCycleCoroutineState::Builder(_) => State::Init,
                LifeCycleCoroutineState::Yielded { res, .. } => State::Started(match res {
                    Err(err) => Err(err.clone()),
                    Ok(x) => Ok(x),
                }),
            },
            InternalState::Complete(kind, res) => State::Finished(
                kind.clone(),
                match res {
                    Err(err) => Err(err.clone()),
                    Ok(()) => Ok(()),
                },
            ),
        }
    }

    pub fn start(&mut self, arg: A) -> Result<&mut Y, E> {
        match &mut self.state {
            InternalState::Ready(coro) => match coro.state.take().unwrap() {
                LifeCycleCoroutineState::Yielded { .. } => {
                    panic!("LifeCycle coroutine has already been started")
                }
                LifeCycleCoroutineState::Builder(f) => {
                    let mut future = Box::into_pin(f(arg));
                    match poll_lifecycle::<Result<Y, E>, _, _>(future.as_mut()) {
                        CoroutineOutput::Yielded(res) => {
                            coro.state = Some(LifeCycleCoroutineState::Yielded { future, res });
                            match &mut coro.state {
                                Some(LifeCycleCoroutineState::Yielded { res, .. }) => match res {
                                    Err(err) => Err(err.clone()),
                                    // SAFETY: Workaround NLL limitation where the compiler thinks
                                    // the returned borrow is active until the end of the function,
                                    // including on other paths where we need to modify self.state.
                                    // We work around it by transmuting the lifetime so they are
                                    // not considered conflicting anymore.
                                    Ok(x) => {
                                        Ok(unsafe { core::mem::transmute::<&mut Y, &mut Y>(x) })
                                    }
                                },
                                _ => unreachable!(),
                            }
                        }
                        CoroutineOutput::Complete(res) => match res {
                            Err(err) => {
                                self.state =
                                    InternalState::Complete(FinishedKind::Early, Err(err.clone()));
                                Err(err)
                            }
                            Ok(_) => panic!(
                                "LifeCycle coroutine is only allowed early returns (before yielding) if Result::Err() is returned."
                            ),
                        },
                    }
                }
            },
            _ => panic!("LifeCycle coroutine has already yielded or returned once."),
        }
    }

    pub fn stop(&mut self) -> Result<(), E> {
        match &mut self.state {
            InternalState::Ready(coro) => match coro.state.as_mut().unwrap() {
                LifeCycleCoroutineState::Builder(_) => {
                    panic!("LifeCycle::stop() was called before LifeCycle::start()")
                }
                LifeCycleCoroutineState::Yielded { future, .. } => {
                    match poll_lifecycle::<Result<Y, E>, _, _>(future.as_mut()) {
                        CoroutineOutput::Yielded(_) => {
                            panic!("LifeCycle coroutine yielded more than once")
                        }
                        CoroutineOutput::Complete(res) => {
                            self.state = InternalState::Complete(FinishedKind::Normal, res.clone());
                            res
                        }
                    }
                }
            },
            // If there was an early return, any potential error was already given to the caller,
            // so the stop part is treated as an infaillble no-op.
            InternalState::Complete(FinishedKind::Early, _) => Ok(()),
            // We make stop() idempotent
            InternalState::Complete(FinishedKind::Normal, res) => res.clone(),
        }
    }
}
