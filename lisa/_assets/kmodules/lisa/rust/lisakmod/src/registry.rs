/* SPDX-License-Identifier: GPL-2.0 */

// FIXME: remove this module since we have linkme working

/// # Safety
///
/// The start() and stop() methods must return pointers to boundaries of the section in which the
/// values of type T are stored.
pub unsafe trait Registry {
    type T: 'static;
    const NAME: &'static str;

    fn start() -> *const Self::T;
    fn stop() -> *const Self::T;

    fn iter() -> impl Iterator<Item = &'static Self::T> {
        RegistryIter {
            start: Self::start(),
            stop: Self::stop(),
            curr: None,
        }
    }
}

struct RegistryIter<T> {
    start: *const T,
    stop: *const T,
    curr: Option<*const T>,
}

impl<T> Iterator for RegistryIter<T>
where
    T: 'static,
{
    type Item = &'static T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = |ptr: *const T| {
            if ptr >= self.stop {
                (ptr, None)
            } else {
                let value: &'static T = unsafe { &*ptr };
                (unsafe { ptr.offset(1) }, Some(value))
            }
        };

        match self.curr {
            None => {
                let (curr, value) = next(self.start);
                self.curr = Some(curr);
                value
            }
            Some(curr) => {
                let (curr, value) = next(curr);
                self.curr = Some(curr);
                value
            }
        }
    }
}

macro_rules! define_registry {
    ($name:ident, $ty:ty) => {
        pub struct $name;
        const _: () = {
            unsafe extern "C" {
                #[link_name = concat!("__start_", "rust_registry_", stringify!($name))]
                static START: ();

                #[link_name = concat!("__stop_", "rust_registry_", stringify!($name))]
                static STOP: ();
            }

            unsafe impl $crate::registry::Registry for $name {
                type T = $ty;
                const NAME: &'static str = stringify!($name);

                fn start() -> *const Self::T {
                    unsafe { &START as *const () as *const Self::T }
                }
                fn stop() -> *const Self::T {
                    unsafe { &STOP as *const () as *const Self::T }
                }
            }

            unsafe impl Send for $name where $ty: Send {}
            unsafe impl Sync for $name where $ty: Sync {}
        };
    };
}

macro_rules! add_to_registry {
    ($name:ident, $value:expr) => {
        const _: () = {
            const NAME: &[u8] = stringify!($name).as_bytes();
            match <$name as $crate::registry::Registry>::NAME.as_bytes() {
                NAME => {},
                _ => panic!("The original name as passed to define_registry!(<name>, ...) of the registry must be used",
                )
            }

            #[unsafe(link_section = concat!("rust_registry_", stringify!($name)))]
            #[used]
            static ENTRY: <$name as $crate::registry::Registry>::T = $value;
        };
    };
}
