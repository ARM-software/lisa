/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{sync::Arc, vec::Vec};
use core::{error::Error as StdError, fmt};

use anyhow;

/// Cloneable error type suitable for memoization
#[derive(Clone)]
pub struct Error {
    inner: Arc<anyhow::Error>,
}

macro_rules! error {
    ($($args:tt)*) => {
        ::core::convert::Into::<$crate::error::Error>::into(::anyhow::anyhow!($($args)*))
    }
}

pub(crate) use error;

impl Error {
    #[inline]
    pub fn new<E>(err: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Error {
            inner: Arc::new(err.into()),
        }
    }

    #[inline]
    pub fn context<C>(self, context: C) -> Self
    where
        C: fmt::Display + Send + Sync + 'static,
    {
        <Result<(), Self> as anyhow::Context<_, _>>::context(Err(self), context)
            .unwrap_err()
            .into()
    }
}

impl From<anyhow::Error> for Error {
    #[inline]
    fn from(err: anyhow::Error) -> Error {
        Error {
            inner: Arc::new(err),
        }
    }
}

impl<E> From<MultiError<E>> for Error
where
    E: StdError + Send + Sync + 'static,
{
    #[inline]
    fn from(err: MultiError<E>) -> Error {
        Error::new(err)
    }
}

impl fmt::Display for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}
impl fmt::Debug for Error {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl StdError for Error {
    #[inline]
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.inner.source()
    }

    // Unstable API
    // #[inline]
    // fn provide<'a>(&'a self, request: &mut core::Error::Request<'a>) {
    // self.inner.provide()
    // }
}

impl embedded_io::Error for Error {
    #[inline]
    fn kind(&self) -> embedded_io::ErrorKind {
        embedded_io::ErrorKind::Other
    }
}

/// Mirror the anyhow::Context API, but returns the original Result<T, E> type rather than
/// Result<T, anyhow::Error>
pub trait ContextExt {
    fn context<C>(self, context: C) -> Self
    where
        C: fmt::Display + Send + Sync + 'static;

    fn with_context<C, F>(self, f: F) -> Self
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T> ContextExt for Result<T, Error> {
    #[inline]
    fn context<C>(self, context: C) -> Self
    where
        C: fmt::Display + Send + Sync + 'static,
    {
        Ok(<Self as anyhow::Context<_, _>>::context(self, context)?)
    }

    #[inline]
    fn with_context<C, F>(self, f: F) -> Self
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        Ok(<Self as anyhow::Context<_, _>>::with_context(self, f)?)
    }
}

pub struct MultiResult<T, E> {
    inner: Result<T, MultiError<E>>,
}

impl<T, E> MultiResult<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    #[inline]
    pub fn into_result(self) -> Result<T, Error> {
        self.inner.map_err(Into::into)
    }
}

impl<E> FromIterator<Result<(), E>> for MultiResult<(), E> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Result<(), E>>,
    {
        let mut empty = true;
        let mut errors = Vec::new();
        for x in iter {
            empty = false;
            if let Err(err) = x {
                errors.push(err);
            }
        }
        MultiResult {
            inner: if empty || errors.is_empty() {
                Ok(())
            } else {
                Err(errors.into_iter().collect())
            },
        }
    }
}

impl<T, E> From<MultiResult<T, E>> for Result<T, Error>
where
    E: StdError + Send + Sync + 'static,
{
    #[inline]
    fn from(multi: MultiResult<T, E>) -> Result<T, Error> {
        multi.into_result()
    }
}

impl<T, E> From<Result<T, MultiError<E>>> for MultiResult<T, E> {
    #[inline]
    fn from(res: Result<T, MultiError<E>>) -> MultiResult<T, E> {
        MultiResult { inner: res }
    }
}

pub struct MultiError<E> {
    inner: Vec<E>,
}

impl<E> FromIterator<E> for MultiError<E> {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = E>,
    {
        MultiError {
            inner: iter.into_iter().collect(),
        }
    }
}

impl<E: fmt::Display> fmt::Display for MultiError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut options = f.options();
        // Print each item with the alternate formatting ("{:#}") so that anyhow::Error is made
        // to show the source of the error.
        options.alternate(true);
        let mut out = alloc::string::String::new();

        let idt = "\n  ";

        // Create a new formatter so we can add indentation to the formatted content.
        for err in &self.inner {
            f.write_str(idt)?;

            let item_f = &mut fmt::Formatter::new(&mut out, options);
            err.fmt(item_f)?;

            let mut is_first = true;
            for chunk in out.split('\n') {
                if !is_first {
                    f.write_str(idt)?;
                }
                is_first = false;
                f.write_str(chunk)?;
            }

            out.clear();
        }
        f.write_str(&out)?;
        Ok(())
    }
}
impl<E: fmt::Debug> fmt::Debug for MultiError<E> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

// Do not implement StdError::source(), otherwise anyhow will start printing them on its own in
// some circumstances, which will result in a mess because we have a source for each of our
// item, and those sources are not particularly related to each other (they certainly do not
// form a chain from a logical perspective).
impl<E: StdError + 'static> StdError for MultiError<E> {}
