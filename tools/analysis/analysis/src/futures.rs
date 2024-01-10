use core::{
    future::Future,
    mem::swap,
    pin::Pin,
    ptr,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

pub struct JoinedFuture<F: Future> {
    futures: Vec<(Pin<Box<F>>, Poll<F::Output>)>,
}

impl<F: Future> JoinedFuture<F> {
    pub fn new<I>(futures: I) -> Self
    where
        I: IntoIterator<Item = F>,
        F: Future,
    {
        JoinedFuture {
            futures: futures
                .into_iter()
                .map(|fut| (Box::pin(fut), Poll::Pending))
                .collect(),
        }
    }
}

impl<F: Future> Future for JoinedFuture<F>
where
    F::Output: Unpin,
{
    type Output = Vec<F::Output>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut finished = true;

        for (fut, res) in self.futures.iter_mut() {
            if res.is_pending() {
                *res = fut.as_mut().poll(cx);
                finished &= res.is_pending();
            };
        }

        if finished {
            let mut futures = vec![];
            // Since we cannot clone the results, we "steal" their ownership by
            // swapping the vector of results with an empty vector. If we did
            // require Clone, they cannot be hidden behind a trait object
            // anymore so we want to avoid that).
            swap(&mut futures, &mut self.futures);
            Poll::Ready(
                futures
                    .into_iter()
                    .map(|x| match x.1 {
                        Poll::Ready(x) => x,
                        _ => panic!("Future not ready"),
                    })
                    .collect(),
            )
        } else {
            Poll::Pending
        }
    }
}

#[inline]
fn noop(_p: *const ()) {}

#[inline]
fn noop_clone(_p: *const ()) -> RawWaker {
    RawWaker::new(_p, &NOOP_WAKER_VTABLE)
}
const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

#[inline]
pub fn make_noop_waker() -> Waker {
    let data = ptr::null();
    let raw = RawWaker::new(data, &NOOP_WAKER_VTABLE);
    unsafe { Waker::from_raw(raw) }
}
