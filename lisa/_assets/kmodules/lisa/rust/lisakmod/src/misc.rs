/* SPDX-License-Identifier: GPL-2.0 */

use core::{
    cmp::{max, min},
    mem::MaybeUninit,
};

use crate::runtime::{alloc::KernelAlloc, kbox::KBox};

macro_rules! container_of {
    ($container:ty, $member:ident, $ptr:expr) => {{
        let ptr = $ptr;
        let ptr: *const _ = (&*ptr);
        let offset = core::mem::offset_of!($container, $member);
        // SAFETY: Our contract is that c_kobj must be the member of a KObjectInner, so we can
        // safely compute the pointer to the parent.
        let container: *const $container = (ptr as *const $container).byte_sub(offset);
        container
    }};
}
pub(crate) use container_of;

#[allow(unused_macros)]
macro_rules! mut_container_of {
    ($container:ty, $member:ident, $ptr:expr) => {{ $crate::misc::container_of!($container, $member, $ptr) as *mut $container }};
}
#[allow(unused_imports)]
pub(crate) use mut_container_of;

pub struct SliceWriter<W, SnipSuffix> {
    inner: W,
    snip_suffix: SnipSuffix,
    idx: usize,
}

impl<W, SnipSuffix> SliceWriter<W, SnipSuffix>
where
    W: AsMut<[u8]> + AsRef<[u8]>,
    SnipSuffix: AsRef<[u8]>,
{
    pub fn new(inner: W, snip_suffix: SnipSuffix) -> Self {
        SliceWriter {
            inner,
            snip_suffix,
            idx: 0,
        }
    }

    pub fn written(&self) -> &[u8] {
        &self.inner.as_ref()[..self.idx]
    }
}

impl<W, SnipSuffix> core::fmt::Write for SliceWriter<W, SnipSuffix>
where
    W: AsMut<[u8]>,
    SnipSuffix: AsRef<[u8]>,
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error> {
        let s: &[u8] = s.as_bytes();
        let s_len = s.len();
        match self.inner.as_mut().split_at_mut_checked(self.idx) {
            Some((_, remaining)) => {
                let buf_len = remaining.len();
                let write = |src: &[u8], dst: &mut [u8], len| {
                    let dst = &mut dst[..len];
                    let src = &src[..len];
                    dst.copy_from_slice(src);
                    len
                };

                // Write what we can even if there is not room for everything. We then propagate
                // the error, and the caller will still be able to use the written buffer to some
                // extent.
                if buf_len < s_len {
                    self.idx += write(s, remaining, buf_len);
                    let snip_suffix = self.snip_suffix.as_ref();
                    let len = min(snip_suffix.len(), buf_len);
                    write(snip_suffix, &mut remaining[buf_len - len..], len);
                    Err(core::fmt::Error)
                } else {
                    self.idx += write(s, remaining, s_len);
                    Ok(())
                }
            }
            None => {
                if s_len > 0 {
                    Err(core::fmt::Error)
                } else {
                    Ok(())
                }
            }
        }
    }
}

pub struct KBoxWriter<'ka, KA: KernelAlloc, SnipSuffix> {
    inner: KBox<[u8], &'ka KA>,
    snip_suffix: SnipSuffix,
    idx: usize,
    requested_capacity: usize,
}

impl<'ka, KA, SnipSuffix> KBoxWriter<'ka, KA, SnipSuffix>
where
    KA: KernelAlloc,
    SnipSuffix: AsRef<[u8]>,
{
    pub fn new_in(snip_suffix: SnipSuffix, capacity: usize, alloc: &'ka KA) -> Self {
        KBoxWriter {
            inner: KBox::<[u8; 0], _>::new_in([], alloc),
            snip_suffix,
            idx: 0,
            requested_capacity: capacity,
        }
    }

    pub fn with_writer<T, F>(snip_suffix: SnipSuffix, capacity: usize, f: F) -> T
    where
        KA: Default,
        F: FnOnce(KBoxWriter<'_, KA, SnipSuffix>) -> T,
    {
        let alloc = Default::default();
        let writer = KBoxWriter::new_in(snip_suffix, capacity, &alloc);
        f(writer)
    }

    #[inline]
    pub fn written(&self) -> &[u8] {
        &self.inner.as_ref()[..self.idx]
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
}

impl<KA, SnipSuffix> core::fmt::Write for KBoxWriter<'_, KA, SnipSuffix>
where
    KA: KernelAlloc,
    SnipSuffix: AsRef<[u8]>,
{
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error> {
        // We need a partial borrow in some places, so we can't use a method.
        macro_rules! remaining {
            ($self:expr) => {{ &mut $self.inner.as_mut()[$self.idx..] }};
        }

        let write_bytes = |src: &[u8], dst: &mut [u8], len| {
            let dst = &mut dst[..len];
            let src = &src[..len];
            dst.copy_from_slice(src);
            len
        };
        let write = |src: &str, dst: &mut [u8], len| write_bytes(src.as_bytes(), dst, len);

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
                    let snip_suffix = self.snip_suffix.as_ref();
                    let inner = &mut self.inner;
                    let inner_len = inner.len();
                    let len = min(snip_suffix.len(), inner_len);
                    write_bytes(snip_suffix, &mut inner[inner_len - len..], len);

                    Err(err)
                }
            }
        } else {
            self.idx += write(s, remaining, s_len);
            Ok(())
        }
    }
}

macro_rules! destructure {
    ($value:expr, $($field:ident),*) => {{
        // Ensure there is no duplicate in the list of fields. If there is any duplicate, the code
        // will not compile as parameter names cannot be duplicated. On top of that we even get a
        // nice error message about duplicated parameters for the user of the macro !
        {
            #[allow(unused)]
            fn check_duplicates($($field: ()),*){}
        }
        let value = $value;
        let value = ::core::mem::MaybeUninit::new(value);
        let value = ::core::mem::MaybeUninit::as_ptr(&value);
        (
            $(
                // SAFETY: Once the value is wrapped in MaybeUninit, no custom Drop implementation
                // will run anymore, so there is no risk of a Drop implementation to read from any
                // of the attributes we moved out of.
                //
                // We also need to ensure that we never move out of the same field twice, which was
                // checked earlier by ensuring there is no duplicated in the fields list.
                unsafe {
                    core::ptr::read(&(*value).$field)
                },
            )*
        )
    }}
}
pub(crate) use destructure;
