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

macro_rules! convert_err_impl {
    ($src:path, $variant:ident, $dst:ident) => {
        impl From<$src> for $dst {
            fn from(err: $src) -> Self {
                $dst::$variant(Box::new(err.into()))
            }
        }
    };
}
pub(crate) use convert_err_impl;
