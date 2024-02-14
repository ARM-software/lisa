macro_rules! convert_err_impl {
    ($src:path, $variant:ident, $dst:ident) => {
        impl From<$src> for $dst {
            fn from(err: $src) -> Self {
                $dst::$variant(Box::new(err.into()))
            }
        }
    };
}
pub(crate) use convert_err_impl;
