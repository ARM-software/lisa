macro_rules! make_closure_coerce {
    ($name:ident, $bound1:tt $(+ $bounds2:tt)*) => {
        #[inline]
        fn $name<F>(f: F) -> F
            where
            F: $bound1 $(+ $bounds2)*
        {
            f
        }
    }
}
pub(crate) use make_closure_coerce;

macro_rules! make_closure_coerce_type {
    ($name:ident, $ty:ty) => {
        #[inline]
        fn $name(f: $ty) -> $ty {
            f
        }
    };
}
pub(crate) use make_closure_coerce_type;

// This is a workaround for the broken HRTB inference:
// https://github.com/rust-lang/rust/issues/41078
macro_rules! closure {
    ($bound1:tt $(+ $bounds2:tt)*, $closure:expr) => {
        {
            $crate::closure::make_closure_coerce!(coerce, $bound1 $(+ $bounds2)*);
            coerce($closure)
        }
    }
}
pub(crate) use closure;
