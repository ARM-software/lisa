/* Workaround this issue: https://github.com/rust-lang/rust/issues/125619 */
PROVIDE(__builtin_copysignq = 0);
