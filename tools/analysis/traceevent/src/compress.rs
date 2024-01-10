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
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Zstd expected {} bytes, decompressed {count}", dst.len()),
            ))
        }
    }
}

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
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
        if count == dst.len() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Zlib expected {} bytes, decompressed {count}", dst.len()),
            ))
        }
    }
}
