use core::{
    cell::UnsafeCell,
    fmt::{Debug, Formatter},
    marker::PhantomData,
    ops::{Deref, DerefMut as _},
};
use std::{
    collections::{btree_map::Entry, BTreeMap},
    io,
    sync::{Arc, Mutex},
};

use bytemuck::cast_slice;
use deref_map::DerefMap;
use genawaiter::{sync::gen, yield_};
use once_cell::unsync::OnceCell;
use smartstring::alias::String;

use crate::{
    array,
    cinterp::{Bitmap, BufferEnv, CompileError, SockAddr, SockAddrKind, Value},
    closure::make_closure_coerce_type,
    compress::Decompressor,
    cparser::{ArrayKind, DynamicKind, Type},
    error::convert_err_impl,
    header::{
        buffer_locations, Abi, BufferId, Cpu, EventDesc, EventId, FieldFmt, FileSize, Header,
        HeaderError, HeaderV6, HeaderV7, LongSize, MemAlign, MemOffset, MemSize, Options,
        Signedness, Timestamp,
    },
    io::{BorrowingRead, BorrowingReadCore},
    iterator::MergedIterator,
    print::{PrintArg, PrintAtom, PrintFmtStr, PrintPrecision, PrintWidth, VBinSpecifier},
    scratch::{ScratchAlloc, ScratchVec},
    str::Str,
};

// Keep a BTreeMap for fast lookup and to avoid huge Vec allocation in case
// some event ids are very large. Since most traces will contain just a few
// types of events, build up the smallest mapping as it goes.
struct EventDescMap<'h, Ctx, MakeCtx> {
    header: &'h Header,
    cold_map: BTreeMap<EventId, &'h EventDesc>,
    hot_map: BTreeMap<EventId, (&'h EventDesc, Ctx)>,
    make_ctx: Arc<Mutex<MakeCtx>>,
}

impl<'h, Ctx: Debug, MakeCtx> Debug for EventDescMap<'h, Ctx, MakeCtx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("EventDescMap")
            .field("cold_map", &self.cold_map)
            .field("hot_map", &self.hot_map)
            .finish_non_exhaustive()
    }
}

impl<'h, Ctx, MakeCtx> EventDescMap<'h, Ctx, MakeCtx>
where
    MakeCtx: FnMut(&'h Header, &'h EventDesc) -> Ctx + 'h,
{
    fn new(header: &'h Header, make_ctx: Arc<Mutex<MakeCtx>>) -> Self {
        EventDescMap {
            header,
            cold_map: header
                .event_descs()
                .into_iter()
                .map(|desc| (desc.id, desc))
                .collect(),
            hot_map: BTreeMap::new(),
            make_ctx,
        }
    }
    #[inline]
    fn lookup<'edm>(&'edm mut self, id: EventId) -> Option<(&'h EventDesc, &'edm Ctx)> {
        match self.hot_map.entry(id) {
            Entry::Occupied(entry) => {
                let (desc, ctx) = entry.into_mut();
                Some((*desc, ctx))
            }
            Entry::Vacant(entry) => match self.cold_map.remove(&id) {
                Some(desc) => {
                    let mut make_ctx = self.make_ctx.lock().unwrap();
                    let (desc, ctx) = entry.insert((desc, make_ctx(self.header, desc)));
                    Some((*desc, ctx))
                }
                None => None,
            },
        }
    }
}

pub struct EventVisitor<'i, 'h, 'edm, MakeCtx, Ctx = ()> {
    pub data: &'i [u8],
    pub header: &'h Header,

    pub timestamp: Timestamp,
    pub buffer_id: &'h BufferId,

    // Using UnsafeCell ensures that the compiler understands that anything derived from what we
    // stored in it can change at any time, even if the EventVisitor is only manipulated via shared
    // ref.
    _desc_map: UnsafeCell<
        // Using *mut here means EventVisitor is invariant in any lifetime contained in Ctx.
        // However, the only values we store in the EventDescMap are either owned by it or have a
        // longer lifetime ('h outlives 'edm), so it's sound to be covariant in 'edm.  So in
        // practice we use 'static but then we cast back to 'h.
        *mut EventDescMap<'static, Ctx, MakeCtx>,
    >,
    // What we really store is:
    // &'edm mut EventDescMap<'h, Ctx, MakeCtx>,
    // But because of variance limitation, we use *mut instead of &mut and we use 'static instead
    // of 'h
    #[allow(clippy::type_complexity)]
    _phantom_desc_map: PhantomData<(
        &'edm mut EventDescMap<'static, Ctx, MakeCtx>,
        &'edm EventDescMap<'h, Ctx, MakeCtx>,
    )>,

    scratch: &'i ScratchAlloc,

    // Unfortunately for now OnceCell<'h> makes EventVisitor invariant in 'h:
    // https://github.com/matklad/once_cell/issues/167
    // The tracking issue for LazyCell also lists the variance issue:
    // https://github.com/rust-lang/rust/issues/109736
    // So to stay covariant in 'h, we use *const instead of &'h. This is fine as we only initialize
    // the OnceCell with a value that does live for 'h, as all the inputs of that computation are
    // stored when the EventVisitor is created.
    event_desc: OnceCell<(*const EventDesc, *const Ctx)>,
}

impl<'i, 'h, 'edm, MakeCtx, Ctx> EventVisitor<'i, 'h, 'edm, MakeCtx, Ctx> {
    fn new(
        header: &'h Header,
        buffer_id: &'h BufferId,
        timestamp: Timestamp,
        data: &'i [u8],
        scratch: &'i ScratchAlloc,
        desc_map: &'edm mut EventDescMap<'h, Ctx, MakeCtx>,
    ) -> Self {
        // SAFETY: Erase the lifetime 'h and replace by 'static so that we stay covariant in 'h. We
        // won't be using the desc_map reference past 'h since:
        // * 'h outlives 'edm
        // * we don't leak self.desc_map anywhere without attaching the 'edm lifetime to what was
        //   leaked
        let desc_map: &'edm mut EventDescMap<'static, Ctx, MakeCtx> = {
            let desc_map: &'edm mut EventDescMap<'h, Ctx, MakeCtx> = desc_map;
            unsafe { core::mem::transmute(desc_map) }
        };

        EventVisitor {
            data,
            header,
            timestamp,
            buffer_id,
            scratch,
            _desc_map: UnsafeCell::new(desc_map),
            event_desc: OnceCell::new(),
            _phantom_desc_map: PhantomData,
        }
    }

    fn __check_covariance_i<'i1>(self) -> EventVisitor<'i1, 'h, 'edm, MakeCtx, Ctx>
    where
        'i: 'i1,
    {
        self
    }
    fn __check_covariance_h<'h1>(self) -> EventVisitor<'i, 'h1, 'edm, MakeCtx, Ctx>
    where
        'h: 'h1,
    {
        self
    }
    fn __check_covariance_edm<'edm1>(self) -> EventVisitor<'i, 'h, 'edm1, MakeCtx, Ctx>
    where
        'edm: 'edm1,
    {
        self
    }
}

// Capture a lifetime syntactically to avoid E0700 when using impl in return position
pub trait CaptureLifetime<'a> {}
impl<T: ?Sized> CaptureLifetime<'_> for T {}

impl<'i, 'h, 'edm, MakeCtx, Ctx> EventVisitor<'i, 'h, 'edm, MakeCtx, Ctx>
where
    MakeCtx: 'h + FnMut(&'h Header, &'h EventDesc) -> Ctx,
{
    pub fn fields<'a>(
        &'a self,
    ) -> Result<
        impl Iterator<Item = (&'a FieldFmt, Result<Value<'a>, BufferError>)>
            + CaptureLifetime<'h>
            + CaptureLifetime<'edm>
            + CaptureLifetime<'i>,
        BufferError,
    > {
        let event_desc = self.event_desc()?;
        let struct_fmt = &event_desc.event_fmt()?.struct_fmt()?;
        let mut fields = struct_fmt.fields.iter();

        Ok(std::iter::from_fn(move || {
            let desc = fields.next()?;
            let offset = desc.offset;
            let size = desc.size;
            let field_data = &self.data[offset..(offset + size)];

            Some((
                desc,
                desc.decoder
                    .decode(self.data, field_data, self.header, self.scratch),
            ))
        }))
    }

    pub fn field_by_name<'a>(
        &'a self,
        name: &str,
    ) -> Result<(&'a FieldFmt, Value<'a>), BufferError> {
        let event_desc = self.event_desc()?;
        let struct_fmt = &event_desc.event_fmt()?.struct_fmt()?;
        let field_fmt = struct_fmt
            .field_by_name(name)
            .ok_or_else(|| CompileError::UnknownField(name.into()))?;

        let val = self.field_by_fmt(field_fmt)?;
        Ok((field_fmt, val))
    }
    pub fn field_by_fmt<'a>(&'a self, field_fmt: &FieldFmt) -> Result<Value<'a>, BufferError> {
        let offset = field_fmt.offset;
        let size = field_fmt.size;
        let field_data = &self.data[offset..(offset + size)];

        field_fmt
            .decoder
            .decode(self.data, field_data, self.header, self.scratch)
    }

    pub fn event_id(&self) -> Result<EventId, BufferError> {
        let parse_u16 = |input| self.header.kernel_abi().parse_u16(input);
        let (_, event_id) = parse_u16(self.data)?;
        Ok(event_id)
    }

    pub fn event_name(&self) -> Result<&str, BufferError> {
        let desc = self.event_desc()?;
        Ok(&desc.name)
    }

    fn event_entry(&self) -> Result<(&'h EventDesc, &'edm Ctx), BufferError> {
        self.event_desc
            .get_or_try_init(|| {
                let event_id = self.event_id()?;
                let not_found = || BufferError::EventDescriptorNotFound(event_id);
                let desc_map = {
                    let desc_map: *mut *mut EventDescMap<'static, Ctx, MakeCtx> =
                        self._desc_map.get();
                    // SAFETY: This comes from an &'edm mut reference in the
                    // first place. Also that mutable reference is short-lived
                    // and does not leak outside of this function. The data
                    // behind it is inside an UnsafeCell so it's ok to modify it
                    // from a shared &self.
                    let desc_map: &mut EventDescMap<'static, Ctx, MakeCtx> =
                        unsafe { &mut **desc_map };
                    let desc_map: &mut EventDescMap<'h, Ctx, MakeCtx> =
                        unsafe { core::mem::transmute(desc_map) };
                    desc_map
                };
                let (desc, ctx) = desc_map.lookup(event_id).ok_or_else(not_found)?;
                Ok((desc, ctx))
            })
            .map(|(desc, ctx)| {
                let ctx: *const Ctx = *ctx;
                let desc: *const EventDesc = *desc;
                // SAFETY: EventDescMap::lookup() returns (&'h EventDesc, &'edm Ctx), which we store
                // as (*const EventDesc, *const Ctx) to avoid variance issues. It's therefore
                // completely safe to just cast it back to &'h EventDesc.
                let desc: &'h EventDesc = unsafe { &*desc };
                let ctx: &'edm Ctx = unsafe { &*ctx };
                (desc, ctx)
            })
    }

    pub fn event_desc(&self) -> Result<&'h EventDesc, BufferError> {
        Ok(self.event_entry()?.0)
    }

    pub fn event_ctx(&self) -> Result<&'edm Ctx, BufferError> {
        Ok(self.event_entry()?.1)
    }

    #[inline]
    pub fn buffer_env(&self) -> BufferEnv {
        BufferEnv::new(self.scratch, self.header, self.data)
    }

    #[inline]
    pub fn vbin_fields<'a>(
        &self,
        print_fmt: &'a PrintFmtStr,
        data: &'a [u32],
    ) -> impl IntoIterator<Item = Result<PrintArg<'a>, BufferError>>
    where
        'h: 'a,
        'i: 'a,
    {
        print_fmt.vbin_fields(self.header, self.scratch, data)
    }
}

pub trait FieldDecoder: Send + Sync {
    fn decode<'d>(
        &self,
        event_data: &'d [u8],
        field_data: &'d [u8],
        header: &'d Header,
        bump: &'d ScratchAlloc,
    ) -> Result<Value<'d>, BufferError>;
}

impl<T: ?Sized> FieldDecoder for T
where
    T: for<'d> Fn(
            &'d [u8],
            &'d [u8],
            &'d Header,
            &'d ScratchAlloc,
        ) -> Result<Value<'d>, BufferError>
        + Send
        + Sync,
{
    fn decode<'d>(
        &self,
        event_data: &'d [u8],
        field_data: &'d [u8],
        header: &'d Header,
        bump: &'d ScratchAlloc,
    ) -> Result<Value<'d>, BufferError> {
        self(event_data, field_data, header, bump)
    }
}

impl FieldDecoder for () {
    fn decode<'d>(
        &self,
        _event_data: &'d [u8],
        _field_data: &'d [u8],
        _header: &'d Header,
        _bump: &'d ScratchAlloc,
    ) -> Result<Value<'d>, BufferError> {
        Err(BufferError::NoDecoder)
    }
}

impl Type {
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn make_decoder(
        &self,
        header: &Header,
    ) -> Result<
        Box<
            dyn for<'d> Fn(
                    &'d [u8],
                    &'d [u8],
                    &'d Header,
                    &'d ScratchAlloc,
                ) -> Result<Value<'d>, BufferError>
                + Send
                + Sync,
        >,
        CompileError,
    > {
        use Type::*;

        let dynamic_decoder = |kind: &DynamicKind| -> Box<
            dyn for<'d> Fn(
                    &'d [u8],
                    &'d [u8],
                    &'d Header,
                    &'d ScratchAlloc,
                ) -> Result<&'d [u8], BufferError>
                + Send
                + Sync,
        > {
            match kind {
                DynamicKind::Dynamic => Box::new(
                    move |data: &[u8], field_data: &[u8], header: &Header, _scratch| {
                        let offset_and_size = header.kernel_abi().parse_u32(field_data)?.1;
                        let offset: usize = (offset_and_size & 0xffff).try_into().unwrap();
                        let size: usize = (offset_and_size >> 16).try_into().unwrap();
                        Ok(&data[offset..(offset + size)])
                    },
                ),
                DynamicKind::DynamicRel => Box::new(
                    move |data: &[u8], field_data: &[u8], header: &Header, _scratch| {
                        let (remainder, offset_and_size) =
                            header.kernel_abi().parse_u32(field_data)?;
                        let next_field_offset =
                            remainder.as_ptr() as usize - data.as_ptr() as usize;

                        let offset: usize = (offset_and_size & 0xffff).try_into().unwrap();
                        let size: usize = (offset_and_size >> 16).try_into().unwrap();

                        let offset = next_field_offset + offset;
                        Ok(&data[offset..(offset + size)])
                    },
                ),
            }
        };

        match self {
            Void => Ok(Box::new(|_, _, _, _| Ok(Value::Unknown))),
            Bool => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::U64Scalar(
                    header.kernel_abi().parse_u8(field_data)?.1.into(),
                ))
            })),
            U8 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::U64Scalar(
                    header.kernel_abi().parse_u8(field_data)?.1.into(),
                ))
            })),
            I8 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::I64Scalar(
                    (header.kernel_abi().parse_u8(field_data)?.1 as i8).into(),
                ))
            })),

            U16 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::U64Scalar(
                    header.kernel_abi().parse_u16(field_data)?.1.into(),
                ))
            })),
            I16 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::I64Scalar(
                    (header.kernel_abi().parse_u16(field_data)?.1 as i16).into(),
                ))
            })),

            U32 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::U64Scalar(
                    header.kernel_abi().parse_u32(field_data)?.1.into(),
                ))
            })),
            I32 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::I64Scalar(
                    (header.kernel_abi().parse_u32(field_data)?.1 as i32).into(),
                ))
            })),

            U64 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::U64Scalar(
                    header.kernel_abi().parse_u64(field_data)?.1,
                ))
            })),
            I64 => Ok(Box::new(move |_data, field_data, header, _| {
                Ok(Value::I64Scalar(
                    header.kernel_abi().parse_u64(field_data)?.1 as i64,
                ))
            })),

            Pointer(_) => match header.kernel_abi().long_size {
                LongSize::Bits32 => U32.make_decoder(header),
                LongSize::Bits64 => U64.make_decoder(header),
            },
            Typedef(typ, _) | Enum(typ, _) => typ.make_decoder(header),

            DynamicScalar(typ, kind) => {
                let decoder = dynamic_decoder(kind);
                match typ.deref() {
                    // Bitmaps created using DECLARE_BITMAP() macro in include/linux/types.h
                    Type::Typedef(_, id)
                        if matches!(
                            id.deref(),
                            "cpumask_t" | "dma_cap_mask_t" | "nodemask_t" | "pnp_irq_mask_t"
                        ) =>
                    {
                        Ok(Box::new(
                            move |data, field_data: &[u8], header: &Header, scratch| {
                                let field_data = decoder(data, field_data, header, scratch)?;
                                Ok(Value::Bitmap(Bitmap::from_bytes(
                                    field_data,
                                    header.kernel_abi(),
                                )))
                            },
                        ))
                    }

                    // As described in:
                    // https://bugzilla.kernel.org/show_bug.cgi?id=217532
                    Type::Typedef(_, id) if id.deref() == "sockaddr_t" => Ok(Box::new(
                        move |data, field_data: &[u8], header: &Header, scratch| {
                            let field_data = decoder(data, field_data, header, scratch)?;
                            Ok(Value::SockAddr(SockAddr::from_bytes(
                                field_data,
                                header.kernel_abi().endianness,
                                SockAddrKind::Full,
                            )?))
                        },
                    )),

                    // Any other dynamic scalar type is unknown, so just provide
                    // the raw buffer to consumers.
                    _ => {
                        let typ = Arc::from(typ.clone());
                        Ok(Box::new(move |data, field_data, header, scratch| {
                            let field_data = decoder(data, field_data, header, scratch)?;
                            Ok(Value::Raw(
                                Arc::clone(&typ),
                                array::Array::Borrowed(field_data),
                            ))
                        }))
                    }
                }
            }

            Array(typ, ArrayKind::Dynamic(kind)) => {
                let data_decoder = dynamic_decoder(kind);
                let array_decoder =
                    Type::Array(typ.clone(), ArrayKind::Fixed(Ok(0))).make_decoder(header)?;

                Ok(Box::new(move |data, field_data, header, scratch| {
                    let array_data = data_decoder(data, field_data, header, scratch)?;
                    array_decoder.decode(data, array_data, header, scratch)
                }))
            }

            Array(typ, ArrayKind::ZeroLength) => {
                let decoder =
                    Type::Array(typ.clone(), ArrayKind::Fixed(Ok(0))).make_decoder(header)?;

                Ok(Box::new(move |data, field_data, header, scratch| {
                    let offset: usize = field_data.as_ptr() as usize - data.as_ptr() as usize;
                    // Currently, ZLA fields are buggy as we cannot know the
                    // true data size. Instead, we get this aligned size,
                    // which can includes padding junk at the end of the event:
                    // https://bugzilla.kernel.org/show_bug.cgi?id=210173
                    let array_data = &data[offset..];
                    decoder.decode(data, array_data, header, scratch)
                }))
            }

            Array(typ, ArrayKind::Fixed(_)) => {
                let item_size = typ.size(header.kernel_abi())?;
                let item_size: usize = item_size.try_into().unwrap();

                macro_rules! parse_scalar {
                    ($ctor:tt, $item_ty:ty, $parse_item:ident) => {{
                        if header.kernel_abi().endianness.is_native() {
                            Box::new(move |_data, field_data: &[u8], header, scratch| {
                                match bytemuck::try_cast_slice(field_data) {
                                    Ok(slice) => Ok(Value::$ctor(array::Array::Borrowed(slice))),
                                    // Data is either misaligned or the array
                                    // size is not a multiple of the item size.
                                    Err(_) => {
                                        let mut vec = ScratchVec::with_capacity_in(
                                            field_data.len() / item_size,
                                            scratch,
                                        );
                                        let item_parser =
                                            |item| header.kernel_abi().$parse_item(item);
                                        for item in field_data.chunks_exact(item_size) {
                                            let item = item_parser(item)?.1 as $item_ty;
                                            vec.push(item)
                                        }
                                        Ok(Value::$ctor(array::Array::Borrowed(vec.leak())))
                                    }
                                }
                            })
                        } else {
                            Box::new(move |_data, field_data: &[u8], header, scratch| {
                                let mut vec = ScratchVec::with_capacity_in(
                                    field_data.len() / item_size,
                                    scratch,
                                );
                                match bytemuck::try_cast_slice::<_, $item_ty>(field_data) {
                                    Ok(slice) => {
                                        vec.extend(slice.into_iter().map(|x| x.swap_bytes()));

                                        // Leak the bumpalo's Vec, which is fine because
                                        // we will deallocate it later by calling
                                        // ScratchAlloc::reset(). Note that Drop impl for items
                                        // will _not_ run.
                                        //
                                        // In order for them to run, we would need to
                                        // return an Vec<> instead of a slice, which
                                        // will be possible one day when the unstable
                                        // allocator_api becomes stable so we can
                                        // allocate a real Vec<> in the ScratchAlloc and simply
                                        // return it.
                                        Ok(Value::$ctor(array::Array::Borrowed(vec.leak())))
                                    }
                                    // Data is either misaligned or the array
                                    // size is not a multiple of the item size.
                                    Err(_) => {
                                        let item_parser =
                                            |item| header.kernel_abi().$parse_item(item);
                                        for item in field_data.chunks_exact(item_size) {
                                            let item = item_parser(item)?.1 as $item_ty;
                                            let item = item.swap_bytes();
                                            vec.push(item)
                                        }
                                        Ok(Value::$ctor(array::Array::Borrowed(vec.leak())))
                                    }
                                }
                            })
                        }
                    }};
                }

                match typ.resolve_wrapper() {
                    Type::Bool => Ok(parse_scalar!(U8Array, u8, parse_u8)),

                    Type::U8 => Ok(parse_scalar!(U8Array, u8, parse_u8)),
                    Type::I8 => Ok(parse_scalar!(I8Array, i8, parse_u8)),

                    Type::U16 => Ok(parse_scalar!(U16Array, u16, parse_u16)),
                    Type::I16 => Ok(parse_scalar!(I16Array, i16, parse_u16)),

                    Type::U32 => Ok(parse_scalar!(U32Array, u32, parse_u32)),
                    Type::I32 => Ok(parse_scalar!(I32Array, i32, parse_u32)),

                    Type::U64 => Ok(parse_scalar!(U64Array, u64, parse_u64)),
                    Type::I64 => Ok(parse_scalar!(I64Array, i64, parse_u64)),

                    Type::Pointer(_) => match header.kernel_abi().long_size {
                        LongSize::Bits32 => Ok(parse_scalar!(U32Array, u32, parse_u32)),
                        LongSize::Bits64 => Ok(parse_scalar!(U64Array, u64, parse_u64)),
                    },

                    _ => Err(CompileError::InvalidArrayItem(typ.deref().clone())),
                }
            }
            typ => {
                let typ = Arc::new(typ.clone());
                Ok(Box::new(move |_data, field_data, _, _| {
                    Ok(Value::Raw(
                        Arc::clone(&typ),
                        array::Array::Borrowed(field_data),
                    ))
                }))
            }
        }
    }
}

use core::cmp::Ordering;
#[derive(Debug)]
struct BufferItem<'a, Ctx, MakeCtx>(
    #[allow(clippy::type_complexity)]
    Result<
        (
            &'a Header,
            &'a mut EventDescMap<'a, Ctx, MakeCtx>,
            &'a BufferId,
            Timestamp,
            &'a [u8],
        ),
        BufferError,
    >,
);

impl<'a, Ctx, MakeCtx> PartialEq for BufferItem<'a, Ctx, MakeCtx> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Ok(x), Ok(y)) => x.3 == y.3,
            _ => std::ptr::eq(self, other),
        }
    }
}

impl<'a, Ctx, MakeCtx> Eq for BufferItem<'a, Ctx, MakeCtx> {}

impl<'a, Ctx, MakeCtx> PartialOrd for BufferItem<'a, Ctx, MakeCtx> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, Ctx, MakeCtx> Ord for BufferItem<'a, Ctx, MakeCtx> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self.0, &other.0) {
            // Compare based on timestamp, then on CPU ID to match the same order as trace-cmd
            // report
            (Ok(x), Ok(y)) => Ord::cmp(&(x.3, x.2.cpu), &(y.3, y.2.cpu)),
            // Consider an error to be like the smallest timestamp possible. This ensures that
            // errors are propagated as soon as they are encountered in the buffer
            (Err(_), Ok(_)) => Ordering::Less,
            (Ok(_), Err(_)) => Ordering::Greater,
            _ => Ordering::Equal,
        }
    }
}

pub struct Buffer<'i, 'h> {
    header: &'h Header,
    pub id: BufferId,
    page_size: MemSize,
    reader: Box<dyn BufferBorrowingRead<'i> + Send>,
}

impl<'i, 'h> Buffer<'i, 'h> {
    // Keep BufferBorrowingRead an implementation detail for now in case we
    // need something more powerful than BufferBorrowingRead in the future.
    pub fn new<I: BorrowingRead + Send + 'i>(
        id: BufferId,
        reader: I,
        page_size: MemSize,
        header: &'h Header,
    ) -> Self {
        Buffer {
            id,
            reader: Box::new(reader),
            page_size,
            header,
        }
    }
}

impl HeaderV7 {
    pub(crate) fn buffers<'i, 'h, 'a, I>(
        &'a self,
        header: &'h Header,
        input: Box<I>,
    ) -> Result<Vec<Buffer<'i, 'h>>, BufferError>
    where
        'a: 'i + 'h,
        I: BorrowingRead + Send + 'i,
    {
        self.options
            .iter()
            .filter_map(|option| match option {
                Options::Buffer {
                    cpu,
                    name,
                    offset,
                    size,
                    page_size,
                    decomp,
                } => {
                    let make_buffer = || -> Result<Buffer<'i, 'h>, BufferError> {
                        // At some point, trace-cmd was creating files with a
                        // broken size for compressed section: the real size was
                        // <size recorded in file> + 4. Since this has been
                        // fixed and there is no way to distinguish if the file
                        // is affected, we simply ignore the size when
                        // compression is used. This is not a major problem as
                        // the compression header contains a chunk count that
                        // will be used to know when to stop reading anyway.
                        //
                        // However, non-compressed buffers still rely on the
                        // recorded size to known when EOF is reached, so we
                        // preserve the value.
                        // https://bugzilla.kernel.org/show_bug.cgi?id=217367
                        let size = if decomp.is_some() { None } else { Some(*size) };

                        let reader = input.clone_and_seek(*offset, size)?;
                        let reader: Box<dyn BufferBorrowingRead + Send> = match decomp {
                            Some(decomp) => Box::new(DecompBorrowingReader::new(
                                &self.kernel_abi,
                                decomp,
                                reader,
                            )?),
                            None => reader,
                        };
                        Ok(Buffer {
                            id: BufferId {
                                cpu: *cpu,
                                name: name.clone(),
                            },
                            reader,
                            page_size: *page_size,
                            header,
                        })
                    };
                    Some(make_buffer())
                }
                _ => None,
            })
            .collect()
    }
}

impl HeaderV6 {
    pub(crate) fn buffers<'i, 'h, 'a: 'i + 'h, I: BorrowingRead + Send + 'i>(
        &'a self,
        header: &'h Header,
        input: Box<I>,
    ) -> Result<Vec<Buffer<'i, 'h>>, BufferError> {
        let nr_cpus = self.nr_cpus;
        let abi = &self.kernel_abi;
        let instances = self.options.iter().filter_map(|opt| match opt {
            Options::Instance { name, offset } => {
                eprintln!("INSTANCE BUFFER OPTION {name:?} {offset}");
                Some((name.clone(), *offset))
            }
            _ => None,
        });

        enum LocId {
            TopLevelInstanceCpu(Cpu),
            Instance(String),
        }

        let locs = self
            .top_level_buffer_locations
            .iter()
            .enumerate()
            .map(|(cpu, loc)| {
                (
                    LocId::TopLevelInstanceCpu(cpu.try_into().unwrap()),
                    loc.offset,
                    Some(loc.size),
                )
            })
            .chain(instances.map(|(name, offset)| (LocId::Instance(name), offset, None)));

        let buffers = locs.map(|(loc_id, offset, size)| {
            let mut reader = input.clone_and_seek(offset, size)?;
            let page_size = self.page_size.try_into().unwrap();
            match loc_id {
                LocId::TopLevelInstanceCpu(cpu) => Ok(vec![Buffer {
                    id: BufferId {
                        cpu,
                        name: "".into(),
                    },
                    page_size,
                    reader,
                    header,
                }]),
                LocId::Instance(name) => {
                    let data_kind = reader.read_null_terminated()?.to_owned();
                    buffer_locations(&data_kind, nr_cpus, abi, &name, reader.deref_mut())?
                        .into_iter()
                        .map(|loc| {
                            Ok(Buffer {
                                id: loc.id,
                                reader: input.clone_and_seek(loc.offset, Some(loc.size))?,
                                page_size,
                                header,
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                }
            }
        });
        let buffers = buffers.collect::<Result<Vec<_>, BufferError>>()?;
        Ok(buffers.into_iter().flatten().collect())
    }
}

#[inline]
unsafe fn transmute_lifetime<'b, T: ?Sized>(x: &T) -> &'b T {
    core::mem::transmute(x)
}

#[inline]
unsafe fn transmute_lifetime_mut<'b, T: ?Sized>(x: &mut T) -> &'b mut T {
    core::mem::transmute(x)
}

pub fn flyrecord<'i, 'h, R, F, IntoIter, MakeCtx, Ctx>(
    buffers: IntoIter,
    mut f: F,
    make_ctx: MakeCtx,
) -> Result<impl IntoIterator<Item = R> + 'h + CaptureLifetime<'i>, BufferError>
where
    IntoIter: IntoIterator<Item = Buffer<'i, 'h>>,
    F: 'h
        + for<'i1, 'edm> FnMut(Result<EventVisitor<'i1, 'h, 'edm, MakeCtx, Ctx>, BufferError>) -> R,
    MakeCtx: 'h + FnMut(&'h Header, &'h EventDesc) -> Ctx,
    Ctx: 'h,
    'i: 'h,
{
    let make_ctx = Arc::new(Mutex::new(make_ctx));

    macro_rules! make_record_iter {
        ($buffer:expr) => {{
            let mut buffer = $buffer;
            let buf_id = buffer.id;
            let header = buffer.header;
            let timestamp_fixer = header.timestamp_fixer();
            let make_ctx = Arc::clone(&make_ctx);

            // Each buffer will have its own hot map which is not ideal, but the
            // maps contain &EventDesc so the descriptor itself actually lives
            // in the header and is shared. This ensures we will not parse event
            // format more than once, which is the main cost here.
            let mut desc_map = EventDescMap::new(header, make_ctx);
            gen!({
                loop {
                    match extract_page(header, &buf_id, &mut buffer.reader, buffer.page_size) {
                        Ok(Some((data, mut timestamp, recoverable_err))) => {
                            if let Some(err) = recoverable_err {
                                yield_!(BufferItem(Err(err)))
                            }

                            let mut data = &*data;
                            while data.len() != 0 {
                                match parse_record(header, data, timestamp) {
                                    Ok((remaining, timestamp_, record)) => {
                                        timestamp = timestamp_;
                                        data = remaining;
                                        match record {
                                            Ok(BufferRecord::Event(data)) => {
                                                // SAFETY: That yielded &[u8] will
                                                // only stay valid until the next
                                                // time next() is called on the
                                                // iterator. MergedIterator
                                                // specifically guarantees to not
                                                // call next() on inner iterators
                                                // before its own next() is called.
                                                //
                                                // Note that this is not the case
                                                // with e.g. itertools kmerge_by()
                                                // method.
                                                let data = unsafe { transmute_lifetime(data) };
                                                let buf_id_ref =
                                                    unsafe { transmute_lifetime(&buf_id) };
                                                let desc_map_ref = unsafe {
                                                    transmute_lifetime_mut(&mut desc_map)
                                                };
                                                yield_!(BufferItem(Ok((
                                                    header,
                                                    desc_map_ref,
                                                    buf_id_ref,
                                                    timestamp_fixer(timestamp),
                                                    data
                                                ))));
                                            }
                                            _ => (),
                                        }
                                    }
                                    Err(err) => {
                                        yield_!(BufferItem(Err(err.into())));
                                        break;
                                    }
                                }
                            }
                        }
                        Ok(None) => break,
                        Err(err) => {
                            yield_!(BufferItem(Err(err)));
                            break;
                        }
                    }
                }
            })
        }};
    }

    let iterators = buffers.into_iter().map(|buffer| make_record_iter!(buffer));
    // Buffer used to reorder array data in case the trace does not have native
    // endianness.
    let mut visitor_scratch = ScratchAlloc::new();

    match MergedIterator::new(iterators) {
        Some(merged) => {
            Ok(merged.enumerate().map(move |(i, x)| match x {
                BufferItem(Ok((header, desc_map, buffer_id, timestamp, data))) => {
                    let visitor = EventVisitor::new(
                        header,
                        buffer_id,
                        timestamp,
                        data,
                        &visitor_scratch,
                        desc_map,
                    );
                    let x = f(Ok(visitor));
                    // Reset the scratch allocator, thereby freeing any value allocated in
                    // it (without running their Drop implementation).
                    //
                    // Reduce the overhead of reseting the scratch allocator.
                    if (i % 16) == 0 {
                        visitor_scratch.reset();
                    }
                    x
                }
                BufferItem(Err(err)) => f(Err(err)),
            }))
        }
        None => Err(BufferError::NoRingBuffer),
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferError {
    #[error("Header contains not ring buffer reference")]
    NoRingBuffer,

    #[error("Some events were lost in buffer {0:?}: {1:?}")]
    LostEvents(BufferId, Option<u64>),

    #[error("Page data too large to be parsed: {0}")]
    PageDataTooLarge(u64),

    #[error("Event descriptor for event ID {0} was not found")]
    EventDescriptorNotFound(EventId),

    #[error("Too many CPUs in the system, CPU ID cannot be represented")]
    TooManyCpus,

    #[error("Could not compute the array size")]
    UnknownArraySize,

    #[error("struct sockaddr buffer was too small to decode")]
    SockAddrTooSmall,

    #[error("Unknown socket family code: {0}")]
    UnknownSockAddrFamily(u16),

    #[error("No decoder for that field")]
    NoDecoder,

    #[error("I/O error while loading data: {0}")]
    IoError(Box<io::ErrorKind>),

    #[error("Error while parsing header: {0}")]
    HeaderError(Box<HeaderError>),

    #[error("Compilation error while loading data: {0}")]
    CompileError(Box<CompileError>),
}

convert_err_impl!(HeaderError, HeaderError, BufferError);
convert_err_impl!(CompileError, CompileError, BufferError);
convert_err_impl!(io::ErrorKind, IoError, BufferError);

impl From<io::Error> for BufferError {
    fn from(err: io::Error) -> BufferError {
        err.kind().into()
    }
}

trait BufferBorrowingRead<'a>
where
    Self: 'a + Send,
{
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]>;
}

impl<'a> BufferBorrowingRead<'a> for Box<dyn BufferBorrowingRead<'a> + Send> {
    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
        self.deref_mut().read(count)
    }
}

impl<'a, R> BufferBorrowingRead<'a> for R
where
    R: BorrowingRead + Send + 'a,
{
    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
        <Self as BorrowingReadCore>::read(self, count)
    }
}

struct DecompBorrowingReader<'a, I, D> {
    abi: &'a Abi,
    inner: Box<I>,
    decomp: &'a D,
    remaining_chunks: u32,

    // Buffer used to decompress data into. It will not incur lots of
    // allocations in the hot path since it will be reused once it reaches the
    // appropriate size.
    buffer: Vec<u8>,

    // current offset in the data.
    offset: MemOffset,
}

impl<'a, I, D> DecompBorrowingReader<'a, I, D>
where
    D: Decompressor + 'a,
    I: BorrowingRead,
{
    fn new(abi: &'a Abi, decomp: &'a D, mut reader: Box<I>) -> io::Result<Self> {
        let nr_chunks: u32 = reader.read_int(abi.endianness)?;

        Ok(DecompBorrowingReader {
            abi,
            decomp,
            inner: reader,
            remaining_chunks: nr_chunks,
            buffer: Vec::new(),
            offset: 0,
        })
    }
}

impl<'a, I, D> BufferBorrowingRead<'a> for DecompBorrowingReader<'a, I, D>
where
    I: BorrowingRead + Send + 'a,
    D: Decompressor + Send + 'a,
{
    fn read(&mut self, count: MemSize) -> io::Result<&[u8]> {
        let len = self.buffer.len();
        let offset = self.offset;

        if offset + count > len {
            if self.remaining_chunks == 0 {
                self.offset = len;
                Err(io::ErrorKind::UnexpectedEof.into())
            } else {
                // Move the non-read data at the beginning of the vec, so we can
                // just reuse that allocation inplace.
                let new_len = len - offset;
                self.buffer.copy_within(offset..len, 0);
                self.buffer.truncate(new_len);
                self.offset = 0;

                while self.buffer.len() < count {
                    if self.remaining_chunks == 0 {
                        return Err(io::ErrorKind::UnexpectedEof.into());
                    } else {
                        self.remaining_chunks -= 1;

                        let compressed_count: u32 = self.inner.read_int(self.abi.endianness)?;
                        let compressed_count: usize = compressed_count.try_into().unwrap();

                        let decompressed_count: u32 = self.inner.read_int(self.abi.endianness)?;
                        let decompressed_count: usize = decompressed_count.try_into().unwrap();

                        let compressed = self.inner.read(compressed_count)?;

                        let len = self.buffer.len();
                        self.buffer.resize(len + decompressed_count, 0);

                        self.decomp.decompress_into(
                            compressed,
                            &mut self.buffer[len..len + decompressed_count],
                        )?;
                    }
                }

                self.offset = count;
                Ok(&self.buffer[..count])
            }
        } else {
            self.offset += count;
            Ok(&self.buffer[offset..offset + count])
        }
    }
}

fn extract_page<'a, 'b: 'a, 'h, I>(
    header: &'h Header,
    buf_id: &'a BufferId,
    input: &'a mut I,
    page_size: MemSize,
) -> Result<
    Option<(
        impl Deref<Target = [u8]> + 'a,
        Timestamp,
        Option<BufferError>,
    )>,
    BufferError,
>
where
    I: BufferBorrowingRead<'b>,
{
    let parse_u32 = |input| header.kernel_abi().parse_u32(input);
    let parse_u64 = |input| header.kernel_abi().parse_u64(input);
    assert!(page_size % 2 == 0);
    let data_size_mask = (1u64 << 27) - 1;
    let missing_events_mask = 1u64 << 31;
    let missing_events_stored_mask = 1u64 << 30;

    let page_data = match input.read(page_size) {
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        x => x,
    }?;
    let data = &page_data;
    let remaining = data.len();

    let (data, timestamp) = parse_u64(data)?;

    let (data, commit) = match header.kernel_abi().long_size {
        LongSize::Bits64 => parse_u64(data),
        LongSize::Bits32 => parse_u32(data).map(|(data, x)| (data, x.into())),
    }?;

    let data_size = data_size_mask & commit;
    let data_size: usize = data_size
        .try_into()
        .map_err(|_| BufferError::PageDataTooLarge(data_size))?;

    let consumed = remaining - data.len();
    let has_missing_events = (commit & missing_events_mask) != 0;
    let recoverable_err = if has_missing_events {
        let has_missing_events_stored = (commit & missing_events_stored_mask) != 0;
        let nr_missing = if has_missing_events_stored {
            let data = &data[data_size..];
            let nr = match header.kernel_abi().long_size {
                LongSize::Bits32 => parse_u32(data)?.1.into(),
                LongSize::Bits64 => parse_u64(data)?.1,
            };
            Some(nr)
        } else {
            None
        };
        Some(BufferError::LostEvents(buf_id.clone(), nr_missing))
    } else {
        None
    };
    let data = DerefMap::new(page_data, move |data| {
        &data[consumed..(data_size + consumed)]
    });
    Ok(Some((data, timestamp, recoverable_err)))
}

#[derive(Debug)]
enum BufferRecord<'a> {
    Event(&'a [u8]),
    #[allow(dead_code)]
    Timestamp(Timestamp),
    #[allow(dead_code)]
    TimeExtend(Timestamp),
    #[allow(dead_code)]
    Padding(FileSize),
}

#[inline]
fn take(input: &[u8], count: usize) -> io::Result<(&[u8], &[u8])> {
    let data = input
        .get(..count)
        .ok_or(io::Error::from(io::ErrorKind::UnexpectedEof))?;
    Ok((&input[count..], data))
}

fn parse_record<'a>(
    header: &Header,
    input: &'a [u8],
    timestamp: Timestamp,
) -> io::Result<(&'a [u8], Timestamp, Result<BufferRecord<'a>, BufferError>)> {
    let parse_u32 = |input| header.kernel_abi().parse_u32(input);

    let (input, record_header) = parse_u32(input)?;
    let record_header: u64 = record_header.into();

    let typ = record_header & ((1 << 5) - 1);
    let delta = record_header >> 5;

    match typ {
        // Padding
        29 => {
            let (input, len) = parse_u32(input)?;
            let len = len.saturating_sub(4);
            let len_usize: usize = len.try_into().unwrap();
            // For some reason the len is sometimes incorrect and larger than the remaining input.
            let input = input.get(len_usize..).unwrap_or(&[]);
            Ok((input, timestamp, Ok(BufferRecord::Padding(len.into()))))
        }
        // Time extend
        30 => {
            let (input, x) = parse_u32(input)?;
            let x: u64 = x.into();

            let extend = delta + (x << 27);
            Ok((
                input,
                timestamp + extend,
                Ok(BufferRecord::TimeExtend(extend)),
            ))
        }
        // Timestamp
        31 => {
            let msb = timestamp & (0xf8u64 << 56);
            let (input, x) = parse_u32(input)?;
            let x: u64 = x.into();
            let timestamp: Timestamp = delta + (x << 27);
            let timestamp = timestamp | msb;
            Ok((input, timestamp, Ok(BufferRecord::Timestamp(timestamp))))
        }
        // Event
        _ => {
            let alignment = 4;
            let (input, size, _padding) = match typ {
                0 => {
                    let (input, size) = parse_u32(input)?;
                    // The size includes the size itself
                    let size = size - 4;
                    // Align the size on the event array item alignment. Since
                    // it's a array of 32bit ints, we align on 4.
                    let aligned = size & !(alignment - 1);
                    let padding = aligned - size;
                    (input, size.into(), padding)
                }
                // Currently, ZLA fields are buggy as we cannot know the
                // true data size. Instead, we get this aligned size, which
                // can includes padding junk:
                // https://bugzilla.kernel.org/show_bug.cgi?id=210173
                _ => {
                    let alignment: u64 = alignment.into();
                    (input, typ * alignment, 0)
                }
            };

            let (input, data) = take(input, size.try_into().unwrap())?;

            Ok((input, timestamp + delta, Ok(BufferRecord::Event(data))))
        }
    }
}

impl PrintFmtStr {
    fn vbin_decoders<'a>(&'a self, header: &'a Header) -> &Vec<VBinDecoder> {
        let abi = header.kernel_abi();
        let char_signedness = abi.char_signedness;
        #[allow(clippy::type_complexity)]
        self.vbin_decoders.get_or_init(|| {
            make_closure_coerce_type!(
                decoder_hrtb,
                Arc<
                    dyn for<'a> Fn(
                            &'a [u8],
                            &'a Header,
                        )
                            -> Result<(&'a [u8], Value<'a>), BufferError>
                        + Send
                        + Sync,
                >
            );

            macro_rules! scalar_parser {
                ($decoder:ident, $typ:ty, $ctor:ident, $align:expr) => {
                    (
                        $align,
                        decoder_hrtb(Arc::new(
                            move |data: &[u8],
                                  header: &Header|
                                  -> Result<(&[u8], Value<'_>), BufferError> {
                                let (remaining, x) = header.kernel_abi().$decoder(data)?;
                                Ok((remaining, Value::$ctor((x as $typ).into())))
                            },
                        )),
                    )
                };
            }
            let atom_decoder = |vbin_spec: &_| match vbin_spec {
                VBinSpecifier::U8 => scalar_parser!(parse_u8, u8, U64Scalar, 1),
                VBinSpecifier::I8 => scalar_parser!(parse_u8, i8, I64Scalar, 1),

                VBinSpecifier::U16 => scalar_parser!(parse_u16, u16, U64Scalar, 2),
                VBinSpecifier::I16 => scalar_parser!(parse_u16, i16, I64Scalar, 2),

                VBinSpecifier::U32 => scalar_parser!(parse_u32, u32, U64Scalar, 4),
                VBinSpecifier::I32 => scalar_parser!(parse_u32, i32, I64Scalar, 4),

                VBinSpecifier::U64 => scalar_parser!(parse_u64, u64, U64Scalar, 4),
                VBinSpecifier::I64 => scalar_parser!(parse_u64, i64, I64Scalar, 4),

                VBinSpecifier::Str => (
                    1,
                    decoder_hrtb(Arc::new(move |data: &[u8], _header| {
                        match data.iter().position(|x| *x == 0) {
                            None => Err(io::ErrorKind::UnexpectedEof.into()),
                            Some(pos) => Ok((
                                &data[pos + 1..],
                                match core::str::from_utf8(&data[..pos]) {
                                    Ok(s) => Value::Str(Str::new_borrowed(s)),
                                    Err(_) => match char_signedness {
                                        Signedness::Unsigned => {
                                            Value::U8Array(array::Array::Borrowed(&data[..pos + 1]))
                                        }
                                        Signedness::Signed => {
                                            let slice: &[i8] = cast_slice(&data[..pos + 1]);
                                            Value::I8Array(array::Array::Borrowed(slice))
                                        }
                                    },
                                },
                            )),
                        }
                    })),
                ),
            };

            self.atoms
                .iter()
                .filter_map(|atom| {
                    let (alignment, decode) = match atom {
                        PrintAtom::Variable { vbin_spec, .. } => Some(atom_decoder(vbin_spec)),
                        _ => None,
                    }?;
                    Some(VBinDecoder {
                        atom: atom.clone(),
                        alignment,
                        decode,
                    })
                })
                .collect()
        })
    }

    #[inline]
    pub fn vbin_fields<'a>(
        &'a self,
        header: &'a Header,
        scratch: &'a ScratchAlloc,
        input: &'a [u32],
    ) -> impl IntoIterator<Item = Result<PrintArg<'a>, BufferError>> {
        let mut i = 0;
        let mut decoders = self.vbin_decoders(header).iter();
        let mut failed = false;
        let align = |x: usize, align: usize| (x + (align - 1)) & !(align - 1);

        let input = if header.kernel_abi().endianness.is_native() {
            input
        } else {
            // The decoding of the [u32] will have swapped bytes to be in our native order, so we
            // need to put it back in the kernel's order before trying to decode. Then within that
            // reconstructed [u8] we can parse stuff as usual, following kernel endianness.  This
            // is because despite the buffer being advertised as a [u32] by the kernel, it is
            // actually manipulated as a [u8] (see vbin_printf() implementation)
            let mut swapped_input = ScratchVec::with_capacity_in(input.len(), scratch);
            for x in input {
                swapped_input.push((*x).swap_bytes())
            }
            swapped_input.leak()
        };
        let input = bytemuck::cast_slice(input);

        std::iter::from_fn(move || {
            if failed {
                return None;
            }

            let decoder = decoders.next()?;

            macro_rules! handle_err {
                ($res:expr) => {
                    match $res {
                        Ok(x) => x,
                        Err(err) => {
                            failed = true;
                            return Some(Err(err.into()));
                        }
                    }
                };
            }

            macro_rules! update_i {
                ($remaining:expr) => {
                    i = $remaining.as_ptr() as usize - input.as_ptr() as usize;
                };
            }

            let (width, precision) = match &decoder.atom {
                PrintAtom::Variable {
                    width: width_kind,
                    precision: precision_kind,
                    ..
                } => {
                    let abi = &header.kernel_abi();
                    let mut decode_u32 = |data: &[u8]| -> io::Result<u32> {
                        let (remaining, x) = abi.parse_u32(&data[align(i, 4)..])?;
                        update_i!(remaining);
                        Ok(x)
                    };
                    (
                        if width_kind == &PrintWidth::Dynamic {
                            Some(handle_err!(decode_u32(input)).try_into().unwrap())
                        } else {
                            None
                        },
                        if precision_kind == &PrintPrecision::Dynamic {
                            Some(handle_err!(decode_u32(input)).try_into().unwrap())
                        } else {
                            None
                        },
                    )
                }
                _ => (None, None),
            };

            let j = align(i, decoder.alignment);
            let (remaining, value) = handle_err!((decoder.decode)(&input[j..], header));
            update_i!(remaining);

            Some(Ok(PrintArg {
                value,
                width,
                precision,
            }))
        })
    }
}

#[derive(Clone)]
pub struct VBinDecoder {
    atom: PrintAtom,
    alignment: MemAlign,
    #[allow(clippy::type_complexity)]
    decode: Arc<
        dyn for<'a> Fn(&'a [u8], &'a Header) -> Result<(&'a [u8], Value<'a>), BufferError>
            + Send
            + Sync,
    >,
}

impl Debug for VBinDecoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.debug_struct("VBinDecoder")
            .field("atom", &self.atom)
            .field("alignment", &self.alignment)
            .finish_non_exhaustive()
    }
}

impl PartialEq<Self> for VBinDecoder {
    fn eq(&self, other: &Self) -> bool {
        self.atom == other.atom && self.alignment == other.alignment
    }
}

impl Eq for VBinDecoder {}

impl PartialOrd<Self> for VBinDecoder {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VBinDecoder {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.atom, &self.alignment).cmp(&(&other.atom, &other.alignment))
    }
}
