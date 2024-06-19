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

macro_rules! make_closure_coerce {
    ($name:ident, $bound1:tt $(+ $bounds2:tt)*) => {
        #[inline]
        fn $name<F>(f: F) -> F
            where
            F: $bound1 $(+ $bounds2)*
        {
            f
        }
    }
}
pub(crate) use make_closure_coerce;

macro_rules! make_closure_coerce_type {
    ($name:ident, $ty:ty) => {
        #[inline]
        fn $name(f: $ty) -> $ty {
            f
        }
    };
}
pub(crate) use make_closure_coerce_type;

// This is a workaround for the broken HRTB inference:
// https://github.com/rust-lang/rust/issues/41078
macro_rules! closure {
    ($bound1:tt $(+ $bounds2:tt)*, $closure:expr) => {
        {
            $crate::closure::make_closure_coerce!(coerce, $bound1 $(+ $bounds2)*);
            coerce($closure)
        }
    }
}
pub(crate) use closure;
