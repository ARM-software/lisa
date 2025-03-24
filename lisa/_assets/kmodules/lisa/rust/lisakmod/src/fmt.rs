/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    cmp::{max, min},
    mem::MaybeUninit,
};

use crate::runtime::{alloc::KernelAlloc, kbox::KBox};

// FIXME: when we get config and values from Makefile available somehow, this should be set to
// KBUILD_MODNAME
pub const PRINT_PREFIX: &str = "lisa: ";

// pub struct SliceWriter<W, Text> {
//     inner: W,
//     snip_suffix: Text,
//     idx: usize,
// }
//
// impl<W, Text> SliceWriter<W, Text>
// where
//     W: AsMut<[u8]> + AsRef<[u8]>,
//     Text: AsRef<[u8]>,
// {
//     pub fn new(inner: W, snip_suffix: Text) -> Self {
//         SliceWriter {
//             inner,
//             snip_suffix,
//             idx: 0,
//         }
//     }
//
//     pub fn written(&self) -> &[u8] {
//         &self.inner.as_ref()[..self.idx]
//     }
// }
//
// impl<W, Text> core::fmt::Write for SliceWriter<W, Text>
// where
//     W: AsMut<[u8]>,
//     Text: AsRef<[u8]>,
// {
//     fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error> {
//         let s: &[u8] = s.as_bytes();
//         let s_len = s.len();
//         match self.inner.as_mut().split_at_mut_checked(self.idx) {
//             Some((_, remaining)) => {
//                 let buf_len = remaining.len();
//                 let write = |src: &[u8], dst: &mut [u8], len| {
//                     let dst = &mut dst[..len];
//                     let src = &src[..len];
//                     dst.copy_from_slice(src);
//                     len
//                 };
//
//                 // Write what we can even if there is not room for everything. We then propagate
//                 // the error, and the caller will still be able to use the written buffer to some
//                 // extent.
//                 if buf_len < s_len {
//                     self.idx += write(s, remaining, buf_len);
//                     let snip_suffix = self.snip_suffix.as_ref();
//                     let len = min(snip_suffix.len(), buf_len);
//                     write(snip_suffix, &mut remaining[buf_len - len..], len);
//                     Err(core::fmt::Error)
//                 } else {
//                     self.idx += write(s, remaining, s_len);
//                     Ok(())
//                 }
//             }
//             None => {
//                 if s_len > 0 {
//                     Err(core::fmt::Error)
//                 } else {
//                     Ok(())
//                 }
//             }
//         }
//     }
// }

pub struct InnerKBoxWriter<'ka, KA: KernelAlloc> {
    inner: KBox<[u8], &'ka KA>,
    idx: usize,
    requested_capacity: usize,
}

pub struct KBoxWriter<'ka, KA: KernelAlloc, Text> {
    inner: InnerKBoxWriter<'ka, KA>,

    snip_suffix: Text,
    line_prefix: Text,
}

impl<'ka, KA, Text> KBoxWriter<'ka, KA, Text>
where
    KA: KernelAlloc,
    Text: AsRef<[u8]>,
{
    pub fn new_in(line_prefix: Text, snip_suffix: Text, capacity: usize, alloc: &'ka KA) -> Self {
        KBoxWriter {
            inner: InnerKBoxWriter {
                inner: KBox::<[u8; 0], _>::new_in([], alloc),
                idx: 0,
                requested_capacity: capacity,
            },
            line_prefix,
            snip_suffix,
        }
    }

    pub fn with_writer<T, F>(line_prefix: Text, snip_suffix: Text, capacity: usize, f: F) -> T
    where
        KA: Default,
        F: FnOnce(KBoxWriter<'_, KA, Text>) -> T,
    {
        let alloc = Default::default();
        let writer = KBoxWriter::new_in(line_prefix, snip_suffix, capacity, &alloc);
        f(writer)
    }

    #[inline]
    pub fn written(&self) -> &[u8] {
        self.inner.written()
    }
}

impl<'ka, KA> InnerKBoxWriter<'ka, KA>
where
    KA: KernelAlloc,
{
    fn written(&self) -> &[u8] {
        &self.inner.as_ref()[..self.idx]
    }

    fn is_first_write(&self) -> bool {
        self.idx == 0
    }

    fn extend_by(&mut self, n: usize) -> Result<(), core::fmt::Error> {
        let cur_size = self.inner.len();
        // First allocation, we try to provide the requested capacity to avoid too many subsequent
        // re-allocations.
        let new_size = if cur_size == 0 {
            max(self.requested_capacity, n)
        } else {
            let new_size = cur_size.saturating_add(n);
            let new_size = new_size.checked_next_power_of_two().unwrap_or(new_size);
            new_size.max(64)
        };
        let mut new = KBox::<u8, _>::try_new_uninit_slice_in(new_size, *self.inner.allocator())
            .map_err(|_| core::fmt::Error)?;
        MaybeUninit::copy_from_slice(&mut new[..cur_size], &self.inner);
        MaybeUninit::fill(&mut new[cur_size..], 0);
        // SAFETY: We initialized both the first part of the buffer from the old data and the extra
        // area with 0.
        let new = unsafe { new.assume_init() };

        self.inner = new;
        Ok(())
    }

    fn _write_slice(&mut self, snip_suffix: &[u8], s: &[u8]) -> Result<(), core::fmt::Error> {
        // We need a partial borrow in some places, so we can't use a method.
        macro_rules! remaining {
            ($self:expr) => {{ &mut $self.inner.as_mut()[$self.idx..] }};
        }

        let write = |src: &[u8], dst: &mut [u8], len| {
            let dst = &mut dst[..len];
            let src = &src[..len];
            dst.copy_from_slice(src);
            len
        };

        let s_len = s.len();
        let remaining = remaining!(self);
        let buf_len = remaining.len();

        if buf_len < s_len {
            match self.extend_by(s_len - buf_len) {
                Ok(()) => {
                    // The buffer may have been extended by more (to avoid reallocating too often)
                    // than what was asked, so recompute the length to write.
                    let remaining = remaining!(self);
                    let len = min(remaining.len(), s_len);
                    self.idx += write(s, remaining, len);
                    Ok(())
                }
                Err(err) => {
                    let remaining = remaining!(self);
                    // We write what we can and leave it at that.
                    self.idx += write(s, remaining, remaining.len());

                    // Write the snip marker as a suffix if there is enough room.
                    let inner = &mut self.inner;
                    let inner_len = inner.len();
                    let len = min(snip_suffix.len(), inner_len);
                    write(snip_suffix, &mut inner[inner_len - len..], len);

                    Err(err)
                }
            }
        } else {
            self.idx += write(s, remaining, s_len);
            Ok(())
        }
    }
}

impl<KA, Text> core::fmt::Write for KBoxWriter<'_, KA, Text>
where
    KA: KernelAlloc,
    Text: AsRef<[u8]>,
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error> {
        let snip_suffix = self.snip_suffix.as_ref();
        let line_prefix = self.line_prefix.as_ref();

        let is_first = self.inner.is_first_write();
        let mut write = |s| self.inner._write_slice(snip_suffix, s);

        let mut iter = s.as_bytes().split(|c| *c == b'\n');
        if let Some(first) = iter.next() {
            // The first item is only the beginning of a line if it is the actual first thing we
            // try to write with this writer. Otherwise, it's just a piece of whatever a calling
            // write!() macros is trying to write.
            if is_first {
                write(line_prefix)?;
            }
            write(first)?;

            for _s in iter {
                write(b"\n")?;
                write(line_prefix)?;
                write(_s)?;
            }
        }

        Ok(())
    }
}
