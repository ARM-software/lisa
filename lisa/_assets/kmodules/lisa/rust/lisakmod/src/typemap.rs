/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{boxed::Box, collections::BTreeMap, format};
use core::{
    any::{Any, TypeId},
    fmt,
    marker::PhantomData,
};

trait Value: Any + fmt::Debug {
    fn as_any(&self) -> &dyn Any;
    // We need to know the key name from the trait object directly for
    // the Debug implementation. This way we can debug-print key/value without
    // knowing what is the concrete type of any of the values.
    fn key_name(&self) -> &str;
}

/// Wraps the value of a key in a struct that knows its associated `Key` type.
/// This way, when abstracted behind a `&dyn Value`, it can still report
/// the name of its key
struct ValueOf<Idx, Key>
where
    Key: ?Sized + KeyOf<Idx>,
{
    value: <Key as KeyOf<Idx>>::Value,
}

impl<Idx, Key> ValueOf<Idx, Key>
where
    Key: ?Sized + KeyOf<Idx>,
{
    fn new(value: <Key as KeyOf<Idx>>::Value) -> Self {
        ValueOf { value }
    }
}

impl<Idx, Key> Value for ValueOf<Idx, Key>
where
    Idx: 'static,
    Key: ?Sized + KeyOf<Idx>,
    <Key as KeyOf<Idx>>::Value: Any + fmt::Debug,
{
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn key_name(&self) -> &str {
        core::any::type_name::<Key>()
    }
}

impl<Idx, Key> fmt::Debug for ValueOf<Idx, Key>
where
    Key: ?Sized + KeyOf<Idx>,
    <Key as KeyOf<Idx>>::Value: fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

/// Declare `Self` as being a valid type-level key for index `Idx`
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an valid key for index {Idx}, add it using: add_index_key!({Idx}, {Self}, <type of value>)"
)]
pub trait KeyOf<Idx>: Any {
    /// Type of the values that will bound found for key `Self`
    type Value: AllowedValueOf<Idx>;
}

/// # Safety
/// An impl is sound iff Self satisfies all the bounds passed to make_index!(). This is relied upon
/// to ensure all values satisfy certain bounds so we can provide auto-trait implementations for
/// TypeMap.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an valid Value for index {Idx} as it does not satisfy the bounds mentioned in make_index!({Idx}, <bounds>). DO NOT IMPLEMENT THIS TRAIT YOURSELF"
)]
pub unsafe trait AllowedValueOf<Idx> {}

pub struct TypeMap<Idx> {
    _marker: PhantomData<Idx>,
    inner: BTreeMap<TypeId, Box<dyn Value>>,
}

impl<Idx> fmt::Debug for TypeMap<Idx> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.inner
                    .values()
                    .map(|v| (v.key_name(), format!("{v:?}"))),
            )
            .finish()
    }
}

impl<Idx> Default for TypeMap<Idx>
where
    Idx: 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Idx> TypeMap<Idx>
where
    Idx: 'static,
{
    #[inline]
    pub fn new() -> Self {
        TypeMap {
            _marker: PhantomData,
            inner: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn insert<Key>(&mut self, value: <Key as KeyOf<Idx>>::Value)
    where
        Key: ?Sized + KeyOf<Idx>,
        <Key as KeyOf<Idx>>::Value: 'static + fmt::Debug,
    {
        let type_id = TypeId::of::<Key>();
        let value = ValueOf::<Idx, Key>::new(value);
        self.insert_any(type_id, Box::new(value));
    }

    #[inline]
    pub fn get<Key>(&self) -> Option<&<Key as KeyOf<Idx>>::Value>
    where
        Key: ?Sized + KeyOf<Idx>,
        <Key as KeyOf<Idx>>::Value: 'static,
    {
        let key = TypeId::of::<Key>();
        self.get_any(&key).map(|v| {
            &v.as_any()
                .downcast_ref::<ValueOf<Idx, Key>>()
                .expect("An Any value of the wrong concrete type was inserted for that key.")
                .value
        })
    }

    #[inline]
    fn insert_any(&mut self, type_id: TypeId, value: Box<dyn Value>) {
        self.inner.insert(type_id, value);
    }

    #[inline]
    fn get_any(&self, type_id: &TypeId) -> Option<&dyn Value> {
        self.inner.get(type_id).map(|x| &**x)
    }
}

macro_rules! make_index {
    ($vis:vis $index:ident $(, $($value_bound:ident),*)?) => {
        $vis struct $index {
            // Make the index an opaque struct
            _dummy: (),
        }

        // SAFETY: We ensure all types implementing that trait satisfy the bounds they are supposed
        // to.
        unsafe impl<T> $crate::typemap::AllowedValueOf<$index> for T
        where
            $($(
                T: $value_bound,

            )*)?
        {}

        $($(
            $crate::typemap::make_index!(@impl $value_bound, $crate::typemap::TypeMap<$index>);
        )?)*

    };
    (@impl Send, $ty:ty) => {
        unsafe impl Send for $ty {}
    };
    (@impl Sync, $ty:ty) => {
        unsafe impl Sync for $ty {}
    };
    (@impl $trait:ident, $ty:ty) => {
        impl $trait for $ty {}
    };
}
pub(crate) use make_index;

macro_rules! add_index_key {
    ($index:ty, $key:ty, $value_ty:ty) => {
        impl $crate::typemap::KeyOf<$index> for $key {
            type Value = $value_ty;
        }
    };
}
