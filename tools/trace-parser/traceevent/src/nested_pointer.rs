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

use core::{fmt::Debug, ops::Deref};

#[derive(Clone, Debug)]
pub(crate) struct NestedPointer<Outer> {
    outer: Outer,
}
impl<Outer> NestedPointer<Outer> {
    #[inline]
    pub fn new(outer: Outer) -> Self {
        NestedPointer { outer }
    }

    // #[inline]
    // pub fn into_outer(self) -> Outer {
    // self.outer
    // }
}

impl<Outer> Deref for NestedPointer<Outer>
where
    Outer: Deref,
    Outer::Target: Deref,
{
    type Target = <Outer::Target as Deref>::Target;

    fn deref(&self) -> &Self::Target {
        &self.outer
    }
}

impl<T, Outer, Inner> AsRef<T> for NestedPointer<Outer>
where
    // We can't use AsRef here because otherwise the compiler might be left with
    // multiple choices of N. With Deref, only one choice is possible since
    // Target is an associated type.
    Outer: Deref<Target = Inner>,
    Inner: Deref<Target = T> + ?Sized,
    T: ?Sized,
{
    #[inline]
    fn as_ref(&self) -> &T {
        self.deref()
    }
}
