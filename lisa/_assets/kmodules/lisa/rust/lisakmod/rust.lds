/* Workaround this issue: https://github.com/rust-lang/rust/issues/125619 */
PROVIDE(__builtin_copysignq = 0);

SECTIONS {
	__trace_printk_fmt : {
		KEEP(*(__trace_printk_fmt))
	}

	/* Remove binstore sections that we created in lisakmod_macros::inlinec Rust module */
	/DISCARD/ : { *(.binstore.*) }
}
