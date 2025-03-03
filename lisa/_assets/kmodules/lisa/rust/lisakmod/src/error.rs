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
    ($($args:tt)*) => { ::core::convert::Into::<$crate::error::Error>::into(::anyhow::anyhow!($($args)*)) }
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
    fn context<C>(self, context: C) -> Self
    where
        C: fmt::Display + Send + Sync + 'static,
    {
        <Result<(), Self> as anyhow::Context<_, _>>::context(Err(self), context)
            .unwrap_err()
            .into()
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Error {
        Error {
            inner: Arc::new(err),
        }
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

impl<T, E> ContextExt for Result<T, E>
where
    E: From<anyhow::Error> + Send + Sync + StdError + 'static,
{
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

/// Combine multiple errors in a single value, so they can later be inspected or combined together.
#[derive(Debug)]
pub struct AllErrors<E> {
    errors: Vec<E>,
}

impl<E> AllErrors<E>
where
    E: core::fmt::Display + Send + Sync + 'static,
{
    #[inline]
    pub fn combine(self, err: Error) -> Result<(), Error> {
        let errors = self.errors;
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.into_iter().fold(err, Error::context))
        }
    }
}

impl<T, E> FromIterator<Result<T, E>> for AllErrors<E> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Result<T, E>>,
    {
        let mut errors = Vec::new();
        for x in iter {
            if let Err(err) = x {
                errors.push(err);
            }
        }
        AllErrors { errors }
    }
}

impl<E> FromIterator<E> for AllErrors<E> {
    #[inline]
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = E>,
    {
        AllErrors {
            errors: iter.into_iter().collect(),
        }
    }
}
