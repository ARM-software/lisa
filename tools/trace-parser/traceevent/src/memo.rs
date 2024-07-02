// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2024, ARM Limited and contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Memoized closures.
use core::{
    fmt,
    ops::{Deref, DerefMut},
};

use once_cell::sync::OnceCell;

/// Represent a memoized closure.
#[derive(Clone)]
pub(crate) struct Memo<T, Seed, F> {
    /// Memoized value.
    val: OnceCell<T>,
    /// Seed that will be passed to `f()` if `val == None`.
    pub seed: Seed,
    /// Function that will be called with `seed` to initialize `val`.
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

    #[allow(dead_code)]
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
