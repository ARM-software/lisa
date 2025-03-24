/* SPDX-License-Identifier: GPL-2.0 */

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
