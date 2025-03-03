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

//! Compression support for trace.dat v7 content

use core::{
    cell::RefCell,
    fmt::{Debug, Formatter},
};
use std::io;

use thread_local::ThreadLocal;

use crate::header::MemSize;

pub trait Decompressor: Send + Sync {
    fn decompress_into(&self, src: &[u8], dst: &mut [u8]) -> io::Result<()>;

    fn decompress<'a>(&'a mut self, src: &'a [u8], dst_count: MemSize) -> io::Result<Vec<u8>> {
        let mut buffer = vec![0; dst_count];
        self.decompress_into(src, &mut buffer)?;
        Ok(buffer)
    }

    fn to_dyn(&self) -> Box<dyn Decompressor>;
}

pub struct DynDecompressor {
    inner: Box<dyn Decompressor>,
}

impl Clone for DynDecompressor {
    #[inline]
    fn clone(&self) -> Self {
        DynDecompressor {
            inner: self.inner.to_dyn(),
        }
    }
}

impl DynDecompressor {
    pub fn new<D: Decompressor + 'static>(inner: D) -> Self {
        DynDecompressor {
            inner: Box::new(inner),
        }
    }
}

impl Debug for DynDecompressor {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("Decompressor").finish_non_exhaustive()
    }
}

impl Decompressor for DynDecompressor {
    #[inline]
    fn decompress_into(&self, src: &[u8], dst: &mut [u8]) -> io::Result<()> {
        self.inner.decompress_into(src, dst)
    }

    #[inline]
    fn to_dyn(&self) -> Box<dyn Decompressor> {
        Box::new(self.clone())
    }
}

/// Decompressor for `zstd` codec.
pub(crate) struct ZstdDecompressor {
    inner: ThreadLocal<RefCell<zstd::bulk::Decompressor<'static>>>,
}

impl ZstdDecompressor {
    pub fn new() -> Self {
        ZstdDecompressor {
            inner: ThreadLocal::new(),
        }
    }

    fn inner(&self) -> &RefCell<zstd::bulk::Decompressor<'static>> {
        self.inner.get_or(move || {
            RefCell::new(
                zstd::bulk::Decompressor::new().expect("Could not create zstd::bulk::Decompressor"),
            )
        })
    }
}

impl Decompressor for ZstdDecompressor {
    #[inline]
    fn to_dyn(&self) -> Box<dyn Decompressor> {
        Box::new(Self::new())
    }

    fn decompress_into(&self, src: &[u8], dst: &mut [u8]) -> io::Result<()> {
        let count = self.inner().borrow_mut().decompress_to_buffer(src, dst)?;
        if count == dst.len() {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Zstd expected {} bytes, decompressed {count}",
                dst.len()
            )))
        }
    }
}

// ruzstd is a pure-Rust implementation of zstd which can ease cross compiling, but at the moment
// we don't really need it so leave it at that.

// #[cfg(not(target_arch = "x86_64"))]
// pub(crate) struct ZstdDecompressor;

// #[cfg(not(target_arch = "x86_64"))]
// impl ZstdDecompressor {
// pub fn new() -> Self {
// ZstdDecompressor
// }
// }

// #[cfg(not(target_arch = "x86_64"))]
// impl Decompressor for ZstdDecompressor {
// #[inline]
// fn to_dyn(&self) -> Box<dyn Decompressor> {
// Box::new(Self::new())
// }

// fn decompress_into(&self, src: &[u8], dst: &mut [u8]) -> io::Result<()> {
// use std::io::Read as _;
// let mut decoder = ruzstd::StreamingDecoder::new(src).map_err(io::Error::other)?;
// decoder.read_exact(dst)
// }
// }

/// Decompressor for `zlib` codec.
pub(crate) struct ZlibDecompressor {
    inner: ThreadLocal<RefCell<libdeflater::Decompressor>>,
}

impl ZlibDecompressor {
    pub fn new() -> Self {
        ZlibDecompressor {
            inner: ThreadLocal::new(),
        }
    }
    pub fn inner(&self) -> &RefCell<libdeflater::Decompressor> {
        self.inner
            .get_or(|| RefCell::new(libdeflater::Decompressor::new()))
    }
}

impl Decompressor for ZlibDecompressor {
    #[inline]
    fn to_dyn(&self) -> Box<dyn Decompressor> {
        Box::new(Self::new())
    }

    fn decompress_into(&self, src: &[u8], dst: &mut [u8]) -> io::Result<()> {
        let count = self
            .inner()
            .borrow_mut()
            .zlib_decompress(src, dst)
            .map_err(io::Error::other)?;
        if count == dst.len() {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Zlib expected {} bytes, decompressed {count}",
                dst.len()
            )))
        }
    }
}
