use std::{error::Error, fmt};

#[derive(Debug)]
pub struct MultiError<E> {
    errors: Vec<E>,
}

impl<E> MultiError<E> {
    fn new<I: IntoIterator<Item = E>>(errors: I) -> Self {
        MultiError {
            errors: errors.into_iter().collect(),
        }
    }

    pub fn errors(&self) -> impl IntoIterator<Item = &E> {
        &self.errors
    }
}

impl<E: fmt::Display> fmt::Display for MultiError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for err in &self.errors {
            err.fmt(f)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct DynMultiError(MultiError<Box<dyn Error>>);

impl DynMultiError {
    pub fn new<E: Error + 'static, I: IntoIterator<Item = E>>(errors: I) -> Self {
        DynMultiError(MultiError::new(
            errors
                .into_iter()
                .map(|err| Box::new(err) as Box<dyn Error>),
        ))
    }
    pub fn errors(&self) -> impl IntoIterator<Item = &dyn Error> {
        self.0.errors().into_iter().map(AsRef::as_ref)
    }
}

impl fmt::Display for DynMultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.0.fmt(f)
    }
}

impl<E: Error + 'static> From<E> for DynMultiError {
    #[inline]
    fn from(error: E) -> Self {
        DynMultiError(MultiError {
            errors: vec![Box::new(error)],
        })
    }
}
