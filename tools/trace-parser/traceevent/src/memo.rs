use core::{
    fmt,
    ops::{Deref, DerefMut},
};

use once_cell::sync::OnceCell;

#[derive(Clone)]
pub(crate) struct Memo<T, Seed, F> {
    val: OnceCell<T>,
    pub seed: Seed,
    pub f: F,
}

impl<T, Seed, F> Memo<T, Seed, F>
where
    F: Fn(&Seed) -> T + Send + Sync,
{
    #[inline]
    pub fn new(seed: Seed, f: F) -> Self {
        Memo {
            val: OnceCell::new(),
            seed,
            f,
        }
    }

    // #[inline]
    // pub fn memoized(&self) -> Option<&T> {
    // self.val.get()
    // }

    // #[inline]
    // pub fn memoized_mut(&mut self) -> Option<&mut T> {
    // self.val.get_mut()
    // }

    #[inline]
    pub fn into_owned(mut self) -> T {
        let _ = self.eval();
        self.val.take().unwrap()
    }

    #[inline]
    fn eval(&self) -> &T {
        self.val.get_or_init(|| (self.f)(&self.seed))
    }
}

impl<T, Seed, F> Deref for Memo<T, Seed, F>
where
    F: Fn(&Seed) -> T + Send + Sync,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.eval()
    }
}

impl<T, Seed, F> DerefMut for Memo<T, Seed, F>
where
    F: Fn(&Seed) -> T + Send + Sync,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let _ = self.eval();
        self.val.get_mut().unwrap()
    }
}

impl<T, Seed, F> fmt::Debug for Memo<T, Seed, F> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("Memo").finish_non_exhaustive()
    }
}
