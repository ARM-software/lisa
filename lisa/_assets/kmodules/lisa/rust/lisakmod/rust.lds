/* Workaround this issue: https://github.com/rust-lang/rust/issues/125619 */
PROVIDE(__builtin_copysignq = 0);

SECTIONS {
	/* Remove binstore sections that we created in lisakmodprocmacro::inlinec Rust module */
	/DISCARD/ : { *(.binstore.*) }
}
