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

//! IO layer.

use core::{
    fmt::Debug,
    mem::size_of,
    ops::{Deref, DerefMut, Range},
};
use std::{
    io,
    io::{BufRead, BufReader, ErrorKind, Read, Seek, SeekFrom},
    os::fd::AsRawFd,
    sync::{Arc, Mutex},
};

use nom::IResult;

use crate::{
    header::{Endianness, FileOffset, FileSize, MemOffset, MemSize},
    parser::{FromParseError, NomError, NomParserExt as _},
    scratch::{OwnedScratchBox, ScratchAlloc, ScratchVec},
};

// We have to know which of MemOffset and FileOffset is the biggest. This module assumes
// MemOffset <= FileOffset, so it converts MemOffset to FileOffset when a common denominator is
// required.
#[inline]
fn mem2file(x: MemOffset) -> FileOffset {
    x.try_into()
        .expect("Could not convert MemOffset to FileOffset")
}
#[inline]
fn file2mem(x: FileOffset) -> MemOffset {
    x.try_into()
        .expect("Could not convert FileOffset to MemOffset")
}

/// Core operations of a I/O input type that can be consumed by this library.
///
/// Emphasis is put on:
/// 1. Zero-copy operation. Readers that have to create copy will manage an intenal buffer and
///    still expose references.
/// 2. Being able to safely seek in the input at multiple locations at once without having to
///    re-open the input multiple times. This is required in order to parse each CPU buffer
///    concurrently, so that events can be re-ordered into a single stream of event.
/// 3. Being eventually object-safe. It currently is not since [BorrowingReadCore::abs_seek] takes
///    a [`Box<Self>`] but that restriction might be lifted one day as [`Box<Self>`] does not
///    require [Self] to be [Sized].
///
/// Note: This trait unfortunately is not (yet) object-safe, but since it does not require Self to
/// be Sized we are probably not too far away from it:
/// `<https://github.com/rust-lang/rust/issues/47649>`
pub trait BorrowingReadCore {
    /// Read the specified number of bytes
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]>;
    /// Read a null-terminated C-style string.
    fn read_null_terminated(&mut self) -> io::Result<&[u8]>;

    /// Clone the object.
    fn try_clone(&self) -> io::Result<Box<Self>>;
    /// Seek to an absolute location in the input.
    ///
    /// This consumes `self` to allow efficient implementations and prevent re-use of the original
    /// object that would now read from a different unexpected location.
    fn abs_seek(
        self: Box<Self>,
        offset: FileOffset,
        len: Option<FileSize>,
    ) -> io::Result<Box<Self>>;

    /// Combine [BorrowingReadCore::try_clone] and [BorrowingReadCore::abs_seek]
    #[inline]
    fn clone_and_seek(&self, offset: FileOffset, len: Option<FileSize>) -> io::Result<Box<Self>> {
        self.try_clone()?.abs_seek(offset, len)
    }
}

/// Convenience functions on top of [BorrowingReadCore].
///
/// Most consumers of this library will interact with this trait rather than [BorrowingReadCore].
///
/// Note: It is not possible to make that trait object-safe as methods take generic parameters, but
/// since they all have default implementation, it would be possible to implement that trait for
/// `dyn BorrowingReadCore`.
pub trait BorrowingRead: BorrowingReadCore {
    /// Apply a nom parser over a buffer starting at the current location and spanning `count`
    /// bytes.
    ///
    /// The outer [Result] layer deals with I/O errors, the inner layer reflects whether the parser
    /// was succeeded or not.
    #[inline]
    fn parse<P, O, E>(&mut self, count: MemSize, mut parser: P) -> io::Result<Result<O, E>>
    where
        // Sadly we can't take an for<'b>impl Parser<&'b [u8], _, _> as it
        // seems impossible to build real-world parsers complying with for<'b>.
        // In practice, the error type needs bounds involving 'b, and there is
        // no way to shove these bounds inside the scope of for<'b> (for now at
        // least). As a result, 'b needs to be a generic param on the parser
        // function and our caller gets to choose the lifetime, not us.
        P: for<'b> Fn(
            &'b [u8],
        )
            -> IResult<&'b [u8], O, NomError<E, nom::error::VerboseError<&'b [u8]>>>,
        E: for<'b> FromParseError<&'b [u8], nom::error::VerboseError<&'b [u8]>> + Debug,
    {
        let buf = self.read(count)?;
        Ok(parser.parse_finish(buf))
    }

    /// Read an integer from the input.
    #[inline]
    fn read_int<T>(&mut self, endianness: Endianness) -> io::Result<T>
    where
        T: DecodeBinary,
    {
        DecodeBinary::decode(self.read(size_of::<T>())?, endianness)
    }

    /// Read a given tag (sequence of integers, typically ASCII string) from the input.
    ///
    /// The outer [Result] layer deals with I/O errors, the inner layer reflects whether the tag
    /// was recognized or not.
    #[inline]
    fn read_tag<'b, T>(&mut self, tag: T) -> io::Result<Result<(), ()>>
    where
        T: IntoIterator<Item = &'b u8>,
        T::IntoIter: ExactSizeIterator,
    {
        let tag = tag.into_iter();
        let buff = self.read(tag.len())?;
        let eq = buff.iter().eq(tag);
        Ok(if eq { Ok(()) } else { Err(()) })
    }
}

impl<T: BorrowingReadCore> BorrowingRead for T {}

// impl<'a> BorrowingReadCore for &'a mut dyn BorrowingReadCore {
//     #[inline]
//     fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
//         (*self).read(count)
//     }
//     #[inline]
//     fn read_null_terminated(&mut self) -> io::Result<&[u8]> {
//         (*self).read_null_terminated()
//     }

//     #[inline]
//     fn try_clone(&self) -> io::Result<Self> {
//         (*self).try_clone()
//     }
//     #[inline]
//     fn abs_seek(self, offset: FileOffset, len: Option<FileSize>) -> io::Result<Self> {
//         (*self).abs_seek(offset, len)
//     }

//     #[inline]
//     fn parse<P, O, E>(&mut self, count: MemSize, mut parser: P) -> io::Result<Result<O, E>>
//     where
//         // Sadly we can't take an for<'b>impl Parser<&'b [u8], _, _> as it
//         // seems impossible to build real-world parsers complying with for<'b>.
//         // In practice, the error type needs bounds involving 'b, and there is
//         // no way to shove these bounds inside the scope of for<'b> (for now at
//         // least). As a result, 'b needs to be a generic param on the parser
//         // function and our caller gets to choose the lifetime, not us.
//         P: for<'b> Fn(
//             &'b [u8],
//         )
//             -> IResult<&'b [u8], O, NomError<E, nom::error::VerboseError<&[u8]>>>,
//         E: for<'b> FromParseError<&'b [u8], nom::error::VerboseError<&'b [u8]>> + Debug,
//     {
//         (*self).parse(count, parser)
//     }

//     #[inline]
//     fn read_int<T>(&mut self, endianness: Endianness) -> io::Result<T>
//     where
//         T: DecodeBinary,
//     {
//         (*self).read_int(endianness)
//     }

//     #[inline]
//     fn read_tag<'b, T>(&mut self, tag: T) -> io::Result<Result<(), ()>>
//     where
//         T: IntoIterator<Item = &'b u8>,
//         T::IntoIter: ExactSizeIterator,
//     {
//         (*self).read_tag(tag, or)
//     }

//     #[inline]
//     fn clone_and_seek(&self, offset: FileOffset, len: Option<FileSize>) -> io::Result<Self> {
//         (*self).clone_and_seek(offset, len)
//     }
// }

/// Newtype wrapper for [`AsRef<[u8]>`] that allows zero-copy operations from [BorrowingReadCore].
/// It is similar to what [std::io::Cursor] provides to [std::io::Read].
#[derive(Clone)]
pub struct BorrowingCursor<T> {
    inner: T,
    offset: MemOffset,
    len: MemSize,
}

impl<T> BorrowingCursor<T>
where
    T: AsRef<[u8]>,
{
    #[inline]
    pub fn new(inner: T) -> Self {
        BorrowingCursor {
            offset: 0,
            len: inner.as_ref().len(),
            inner,
        }
    }

    #[inline]
    fn buf(&self) -> &[u8] {
        self.inner.as_ref()
    }

    #[inline]
    fn max_offset(&self) -> MemOffset {
        self.offset + self.len
    }

    #[inline]
    fn range(&self) -> Range<usize> {
        self.offset..self.max_offset()
    }

    #[inline]
    fn slice(&self) -> &[u8] {
        &self.buf()[self.range()]
    }

    #[inline]
    fn advance(&mut self, count: MemOffset) -> io::Result<&[u8]> {
        if self.offset + count > self.max_offset() {
            Err(ErrorKind::UnexpectedEof.into())
        } else {
            let range = self.offset..(self.offset + count);

            self.offset += count;
            self.len -= count;

            Ok(&self.buf()[range])
        }
    }
}

impl<T> From<T> for BorrowingCursor<T>
where
    T: AsRef<[u8]>,
{
    #[inline]
    fn from(x: T) -> Self {
        BorrowingCursor::new(x)
    }
}

impl<T> BorrowingReadCore for BorrowingCursor<T>
where
    T: AsRef<[u8]> + Clone,
{
    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
        self.advance(count)
    }

    fn read_null_terminated(&mut self) -> io::Result<&[u8]> {
        match self.slice().iter().position(|x| *x == 0) {
            Some(end) => {
                let range = self.offset..(self.offset + end);
                self.advance(end + 1)?;
                Ok(&self.buf()[range])
            }
            None => {
                self.advance(self.len)?;
                Err(ErrorKind::UnexpectedEof.into())
            }
        }
    }

    #[inline]
    fn try_clone(&self) -> io::Result<Box<Self>> {
        Ok(Box::new(self.clone()))
    }

    fn abs_seek(
        mut self: Box<Self>,
        offset: FileOffset,
        len: Option<FileSize>,
    ) -> io::Result<Box<Self>> {
        #[inline]
        fn convert(x: FileOffset) -> io::Result<MemOffset> {
            x.try_into().map_err(|_| ErrorKind::UnexpectedEof.into())
        }

        let offset = convert(offset)?;
        let len = match len {
            Some(len) => convert(len),
            None => Ok(self.buf().len() - offset),
        }?;

        if offset + len > self.buf().len() {
            Err(ErrorKind::UnexpectedEof.into())
        } else {
            self.offset = offset;
            self.len = len;
            Ok(self)
        }
    }
}

struct Mmap {
    // Offset in the file matching the beginning of the memory area
    file_offset: FileOffset,
    // Current offset in the memory area.
    read_offset: MemOffset,
    // Length of the mmapped area. This could be smaller than the actual mmapped
    // area if the mmap was recycled with an adjusted length.
    len: MemSize,
    mmap: memmap2::Mmap,
}

impl Mmap {
    unsafe fn new<T>(file: &T, offset: FileOffset, mut len: MemSize) -> io::Result<Self>
    where
        T: AsRawFd,
    {
        //SAFETY: mmap is inherently unsafe as the memory content could change
        // without notice if the backing file is modified. We have to rely on
        // the user/OS being nice to us and not do that, or we might crash,
        // there is no way around it unfortunately.
        let mmap = loop {
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .offset(offset)
                    .len(len)
                    .map(file)
            };
            match mmap {
                Ok(mmap) => break Ok(mmap),
                Err(err) => {
                    len /= 2;
                    if len == 0 {
                        break Err(err);
                    }
                }
            }
        }?;

        // This MADV_WILLNEED is equivalent to MAP_POPULATE in terms of enabling read-ahead but
        // will not trigger a complete read in memory upon creation of the mapping. This
        // dramatically lowers the reported RES memory consumption at no performance cost.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let _ = mmap.advise(memmap2::Advice::Sequential);

        Ok(Mmap {
            len,
            file_offset: offset,
            read_offset: 0,
            mmap,
        })
    }

    #[inline]
    fn curr_offset(&self) -> FileOffset {
        self.file_offset + mem2file(self.read_offset)
    }

    #[inline]
    fn max_file_offset(&self) -> FileOffset {
        self.file_offset + mem2file(self.len)
    }

    #[inline]
    fn remaining(&self) -> MemSize {
        self.len - self.read_offset
    }

    #[inline]
    fn read(&mut self, count: MemSize) -> Option<&[u8]> {
        if self.remaining() >= count {
            let view = &self.mmap[self.read_offset..self.read_offset + count];
            self.read_offset += count;
            Some(view)
        } else {
            None
        }
    }

    fn abs_seek(mut self, offset: FileOffset, len: FileSize) -> Option<Self> {
        if offset >= self.file_offset && offset + len <= self.max_file_offset() {
            let delta = file2mem(offset - self.file_offset);
            self.read_offset = delta;
            self.len = delta + len.try_into().unwrap_or(MemSize::MAX);
            assert!(self.curr_offset() < self.max_file_offset());
            Some(self)
        } else {
            None
        }
    }
}

impl Deref for Mmap {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.mmap[self.read_offset..self.len]
    }
}

struct MmapFileInner<T> {
    // Use a Arc<Mutex<_>> so that we can clone the reference when creating
    // MmapFile
    file: Arc<Mutex<T>>,
    // Total file size. This is used to validate new length in order to avoid
    // creating mappings that would lead to a SIGBUS.
    file_len: FileSize,

    // Logical length of the area we want to mmap. The actual mmapped area at
    // any given time might be smaller.
    len: FileSize,
    // Original offset of the mmapped area. The current read offset is
    // maintained in Mmap.
    offset: FileOffset,

    mmap: Mmap,
}

impl<T> MmapFileInner<T> {
    #[inline]
    fn curr_offset(&self) -> FileOffset {
        self.mmap.curr_offset()
    }

    #[inline]
    fn max_offset(&self) -> FileOffset {
        self.offset + self.len
    }

    #[inline]
    unsafe fn remap(&mut self, offset: FileOffset) -> io::Result<Mmap>
    where
        T: AsRawFd,
    {
        // Try to map all the remainder of the file range of interest.
        let len = if offset > self.offset {
            self.len - (offset - self.offset)
        } else {
            self.len + (self.offset - offset)
        };

        // Saturate at the max size possible for a mmap
        let len: MemSize = len.try_into().unwrap_or(MemSize::MAX);
        Mmap::new(self.file.lock().unwrap().deref(), offset, len)
    }
}

/// Uses mmap to provide a [BorrowingRead]
///
/// In cases where the mmap syscall fails, it will transparently fallback to read syscalls.
pub struct MmapFile<T> {
    inner: MmapFileInner<T>,
    scratch: ScratchAlloc,
}

impl<T> MmapFile<T> {
    /// # Safety
    ///
    /// Undefined behavior will happen if the file is modified while it is opened from here,
    /// as Rust will not expect the underlying memory to change randomly.
    pub unsafe fn new(mut file: T) -> io::Result<MmapFile<T>>
    where
        T: AsRawFd + Seek,
    {
        let offset = 0;
        let len = file_len(&mut file)?;
        let file = Arc::new(Mutex::new(file));
        Self::from_cell(file, offset, None, len)
    }

    unsafe fn from_cell(
        file: Arc<Mutex<T>>,
        offset: FileOffset,
        len: Option<FileSize>,
        file_len: FileSize,
    ) -> io::Result<MmapFile<T>>
    where
        T: AsRawFd,
    {
        let len = len.unwrap_or(file_len - offset);

        // Check that we are not trying to mmap past the end of the
        // file, as mmap() will let us do it but we will get SIGBUS upon
        // access.
        if offset + len > file_len {
            Err(ErrorKind::UnexpectedEof.into())
        } else {
            let mmap_len = len.try_into().unwrap_or(MemSize::MAX);
            let mmap = Mmap::new(file.lock().unwrap().deref(), offset, mmap_len)?;

            Ok(MmapFile {
                inner: MmapFileInner {
                    file,
                    len,
                    file_len,
                    mmap,
                    offset,
                },
                scratch: ScratchAlloc::new(),
            })
        }
    }

    #[inline]
    fn clear_buffer(&mut self) {
        self.scratch.reset()
    }
}

impl<T> BorrowingReadCore for MmapFile<T>
where
    T: AsRawFd + Read + Seek,
{
    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
        self.clear_buffer();

        let max_offset = self.inner.max_offset();
        if self.inner.curr_offset() + mem2file(count) > max_offset {
            // Remap to consume all the remaining data to be consistent with
            // other BorrowingReadCore implementations.
            self.inner.mmap = unsafe { self.inner.remap(max_offset)? };
            Err(ErrorKind::UnexpectedEof.into())
        } else {
            // Workaround limitation of NLL borrow checker, see:
            // https://docs.rs/polonius-the-crab/latest/polonius_the_crab/
            macro_rules! this {
                () => {
                    // SAFETY: This is safe as long as we do not use "self"
                    // anymore from the first use of this!(). It has been
                    // checked with MIRI.
                    // If and when Polonius borrow checker becomes available,
                    // that unsafe{} block can simply be replaced by "self". It
                    // has been tested with RUSTFLAGS="-Z polonius".
                    {
                        #[allow(unused_unsafe)]
                        unsafe {
                            &mut *(self as *mut Self)
                        }
                    }
                };
            }

            macro_rules! inner {
                () => {
                    this!().inner
                };
            }

            for i in 0..2 {
                if let Some(slice) = inner!().mmap.read(count) {
                    return Ok(slice);
                } else if i == 0 {
                    // Not enough bytes left in the mmap, so we need to remap to catch
                    // up with the read offset.
                    inner!().mmap = unsafe { inner!().remap(inner!().curr_offset())? };
                }
            }
            // Remapping was not enough, we need to fallback on read()
            // syscall. We discard the mapping we just created, as it's
            // useless since it cannot service even the first read.

            let file_offset = inner!().curr_offset();
            let count_u64 = mem2file(count);

            // Remap for future reads after the syscall we are abount to do.
            inner!().mmap = unsafe { inner!().remap(file_offset + count_u64)? };
            assert!(inner!().curr_offset() == file_offset + count_u64);

            let mut file = inner!().file.lock().unwrap();
            file.seek(SeekFrom::Start(file_offset))?;

            let scratch = &this!().scratch;
            read(file.deref_mut(), count, scratch)
        }
    }

    #[inline]
    fn read_null_terminated(&mut self) -> io::Result<&[u8]> {
        self.clear_buffer();

        let find = |buf: &[u8]| buf.iter().position(|x| *x == 0);

        for i in 0..2 {
            match find(&self.inner.mmap) {
                Some(end) => {
                    let view = self.inner.mmap.read(end + 1).unwrap();
                    // Remove the null terminator from the view
                    let view = &view[..view.len() - 1];
                    return Ok(view);
                }
                // Update the mapping to catch up with the current read offset and try again.
                None => {
                    // We scanned the entire content of the current mmap and
                    // there won't be anything else to mmap after that, so we
                    // reached the end.
                    if self.inner.curr_offset() + mem2file(self.inner.mmap.remaining())
                        >= self.inner.max_offset()
                    {
                        // Consume all the remaining data to be consistent with
                        // other implementations.
                        assert!(self.inner.mmap.read(self.inner.mmap.remaining()).is_some());
                        return Err(ErrorKind::UnexpectedEof.into());
                    } else if i == 0 {
                        self.inner.mmap = unsafe { self.inner.remap(self.inner.curr_offset())? };
                    }
                }
            }
        }

        // We failed to find the pattern in the area covered by mmap, so try
        // again with read() syscall.

        let out = {
            let mut file = self.inner.file.lock().unwrap();
            let file_offset = self.inner.curr_offset();
            file.seek(SeekFrom::Start(file_offset))?;

            let buf_size = 4096;
            let mut out = ScratchVec::new_in(&self.scratch);

            loop {
                let prev_len = out.len();
                out.resize(prev_len + buf_size, 0);

                let nr = file.read(&mut out[prev_len..])?;
                out.truncate(prev_len + nr);

                if let Some(end) = find(&mut out[prev_len..]) {
                    out.truncate(prev_len + end);
                    break out;
                }
            }
        };

        // Remap for future reads after the syscall we are about to do.
        let mmap_offset = self.inner.curr_offset() + mem2file(out.len()) + 1;
        self.inner.mmap = unsafe { self.inner.remap(mmap_offset)? };

        Ok(out.leak())
    }

    #[inline]
    fn clone_and_seek(&self, offset: FileOffset, len: Option<FileSize>) -> io::Result<Box<Self>> {
        let new = unsafe {
            Self::from_cell(
                Arc::clone(&self.inner.file),
                offset,
                len,
                self.inner.file_len,
            )
        };
        new.map(Box::new)
    }

    #[inline]
    fn try_clone(&self) -> io::Result<Box<Self>> {
        let new = unsafe {
            Self::from_cell(
                Arc::clone(&self.inner.file),
                self.inner.curr_offset(),
                None,
                self.inner.file_len,
            )
        };
        new.map(Box::new)
    }

    fn abs_seek(
        mut self: Box<Self>,
        offset: FileOffset,
        len: Option<FileSize>,
    ) -> io::Result<Box<Self>> {
        let file_len = self.inner.file_len;
        let len = len.unwrap_or(file_len - offset);

        // Try to recycle the existing mapping if the new offset/size fits
        // inside it.
        match self.inner.mmap.abs_seek(offset, len) {
            Some(mmap) => {
                self.inner.mmap = mmap;
                self.inner.offset = offset;
                self.inner.len = len;
                Ok(self)
            }
            None => {
                let new = unsafe { Self::from_cell(self.inner.file, offset, Some(len), file_len) };
                new.map(Box::new)
            }
        }
    }
}

/// [BorrowingRead] equivalent to using a [BufReader] on top of a [Read] object.
///
/// This implementation will efficiently buffer reads and seeks so that a single opened file can be
/// used even though multiple readers are trying to read from separate areas.
pub struct BorrowingBufReader<T> {
    inner: BufReader<CursorReader<T>>,
    consume: MemSize,
    len: FileSize,
    max_len: FileSize,
    scratch: ScratchAlloc,
}

impl<T> BorrowingBufReader<T>
where
    T: Read + Seek,
{
    pub fn new(mut reader: T, buf_size: Option<MemSize>) -> io::Result<Self> {
        let len = file_len(&mut reader)?;
        let offset = 0;
        let reader = Arc::new(Mutex::new(reader));

        Ok(Self::new_with(reader, buf_size, offset, len, len))
    }

    fn new_with(
        reader: Arc<Mutex<T>>,
        buf_size: Option<MemSize>,
        offset: FileOffset,
        len: FileSize,
        max_len: FileSize,
    ) -> Self {
        let buf_size = buf_size.unwrap_or(4096);

        let reader = CursorReader::new(reader, offset, len);
        let reader = BufReader::with_capacity(buf_size, reader);

        BorrowingBufReader {
            inner: reader,
            consume: 0,
            len,
            max_len,
            scratch: ScratchAlloc::new(),
        }
    }

    #[inline]
    fn clear_buffer(&mut self) {
        self.scratch.reset();
    }

    #[inline]
    fn consume(&mut self) {
        self.inner.consume(self.consume);
        self.consume = 0;
        self.clear_buffer();
    }
}

impl<T> BorrowingReadCore for BorrowingBufReader<T>
where
    T: Read + Seek,
{
    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
        self.consume();

        let buf = self.inner.fill_buf()?;
        let len = buf.len();

        if len == 0 && count > 0 {
            Err(ErrorKind::UnexpectedEof.into())
        } else if count < len {
            self.consume = count;
            Ok(&self.inner.buffer()[..count])
        } else {
            // Pre-filled buffer not large enough for that read, fallback on
            // read() syscall
            read(&mut self.inner, count, &self.scratch)
        }
    }

    #[inline]
    fn read_null_terminated(&mut self) -> io::Result<&[u8]> {
        self.consume();

        let buf = self.inner.fill_buf()?;
        let end = buf.iter().position(|x| *x == 0);

        match end {
            Some(end) => {
                if end == 0 {
                    let data = OwnedScratchBox::with_capacity_in(0, &self.scratch);
                    Ok(data.leak())
                } else {
                    self.consume = end + 1;
                    // For some reason, the borrow checker is not happy for us
                    // to use buf directly, so we fetch it again with
                    // self.inner.buffer()
                    Ok(&self.inner.buffer()[..end])
                }
            }
            None => {
                // If we could not find the data in the pre-loaded buffer, just read
                // as much as needed
                let mut vec = ScratchVec::new_in(&self.scratch);

                loop {
                    let mut buf = [0];
                    self.inner.read_exact(&mut buf)?;
                    let x = buf[0];
                    if x == 0 {
                        break;
                    } else {
                        vec.push(x)
                    }
                }
                Ok(vec.leak())
            }
        }
    }

    fn try_clone(&self) -> io::Result<Box<Self>> {
        let mut reader = self.inner.get_ref().clone();
        // We need to make sure the CursorReader<T> inside the BufReader is
        // pointing at the current offset, not the offset that we were advanced
        // at to fill the buffer.
        // We could use:
        // self.inner.seek(SeekFrom::Current(0));
        // But this would purge the buffer of our BufReader, so instead we can
        // just fixup the cloned CursorReader current offset.
        reader.offset -= mem2file(self.inner.buffer().len() - self.consume);

        let inner = BufReader::with_capacity(self.inner.capacity(), reader);
        Ok(Box::new(BorrowingBufReader {
            inner,
            consume: 0,
            len: self.len,
            max_len: self.max_len,
            scratch: ScratchAlloc::new(),
        }))
    }

    fn abs_seek(
        mut self: Box<Self>,
        offset: FileOffset,
        len: Option<FileSize>,
    ) -> io::Result<Box<Self>> {
        self.consume();

        let capacity = self.inner.capacity();
        let len = len.unwrap_or(self.max_len - offset);

        // Ensure the underlying reader's cursor is set to the correct position,
        // taking into account the unread part of the BufReader internal buffer.
        // Otherwise, into_inner() will give a reader that is further in the
        // stream compared to what we are currently looking at because BufReader
        // pre-loaded some content.
        self.inner.stream_position()?;
        let reader = self.inner.into_inner().inner;

        Ok(Box::new(BorrowingBufReader::new_with(
            reader,
            Some(capacity),
            offset,
            len,
            self.max_len,
        )))
    }
}

struct CursorReader<T> {
    // This Mutex is necessary for types to be Send/Sync. The cost might seem high, but it's
    // actually quite low as CursorReader is always used behind a BufReader.
    inner: Arc<Mutex<T>>,
    last_offset: FileOffset,
    offset: FileOffset,
}

impl<T> Clone for CursorReader<T> {
    fn clone(&self) -> Self {
        CursorReader {
            inner: self.inner.clone(),
            offset: self.offset,
            last_offset: self.last_offset,
        }
    }
}

impl<T> CursorReader<T> {
    fn new(reader: Arc<Mutex<T>>, offset: FileOffset, len: FileSize) -> Self {
        CursorReader {
            inner: reader.clone(),
            last_offset: offset + len,
            offset,
        }
    }
}

impl<T> Read for CursorReader<T>
where
    T: Read + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<MemSize> {
        let mut reader = self.inner.lock().unwrap();
        reader.seek(SeekFrom::Start(self.offset))?;
        let mut count = reader.read(buf)?;
        self.offset += mem2file(count);

        if self.offset > self.last_offset {
            let rewind = file2mem(self.offset - self.last_offset);
            count = if rewind > count { 0 } else { count - rewind };
            self.offset = self.last_offset;
        }
        Ok(count)
    }
}

impl<T> Seek for CursorReader<T>
where
    T: Seek,
{
    fn seek(&mut self, pos: SeekFrom) -> io::Result<FileOffset> {
        let mut reader = self.inner.lock().unwrap();
        let pos = match pos {
            SeekFrom::Current(x) => {
                let start = if x > 0 {
                    self.offset + x.unsigned_abs()
                } else {
                    self.offset - x.unsigned_abs()
                };
                SeekFrom::Start(start)
            }
            pos => pos,
        };
        self.offset = reader.seek(pos)?;
        Ok(self.offset)
    }
}

fn read<'a, T>(reader: &mut T, count: MemSize, alloc: &'a ScratchAlloc) -> io::Result<&'a [u8]>
where
    T: Read,
{
    let mut out = OwnedScratchBox::with_capacity_in(count, alloc);
    reader.read_exact(out.deref_mut())?;
    Ok(out.leak())
}

#[inline]
fn file_len<T>(stream: &mut T) -> io::Result<FileSize>
where
    T: Seek,
{
    let old_pos = stream.stream_position()?;
    let len = stream.seek(SeekFrom::End(0))?;
    stream.seek(SeekFrom::Start(old_pos))?;
    Ok(len)
}

pub trait DecodeBinary: Sized {
    fn decode(buf: &[u8], endianness: Endianness) -> io::Result<Self>;
}

macro_rules! impl_DecodeBinary {
    ( $($ty:ty),* ) => {
        $(
            impl DecodeBinary for $ty {
                #[inline]
                fn decode(buf: &[u8], endianness: Endianness) -> io::Result<Self> {
                    match buf.try_into() {
                        Ok(buf) => Ok(match endianness {
                            Endianness::Little => Self::from_le_bytes(buf),
                            Endianness::Big => Self::from_be_bytes(buf),
                        }),
                        Err(_) => Err(ErrorKind::UnexpectedEof.into())
                    }
                }
            }
        )*
    }
}

impl_DecodeBinary!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
