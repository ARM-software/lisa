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
#![feature(trait_upcasting)]

//! This crate implements a parser for
//! [trace.dat](https://www.trace-cmd.org/Documentation/trace-cmd/trace-cmd.dat.v7.5.html) file
//! format. The decoding naturally streams so it is able to process arbitrarily large files.

mod closure;
mod compress;
mod error;
mod grammar;
mod iterator;
mod memo;
mod nested_pointer;
mod parser;

pub mod array;
pub mod buffer;
pub mod cinterp;
pub mod cparser;
pub mod header;
pub mod io;
pub mod print;
pub mod scratch;
pub mod str;
