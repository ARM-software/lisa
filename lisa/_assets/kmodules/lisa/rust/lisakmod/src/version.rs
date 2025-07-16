/* SPDX-License-Identifier: GPL-2.0 */

use lisakmod_macros::inlinec::cfunc;

#[cfunc]
pub fn module_version() -> &'static str {
    r#"
    #include <linux/string.h>
    #include "generated/module_version.h"
    "#;

    r#"
    static const char *version = LISA_MODULE_VERSION;
    return (struct const_rust_str){
        .data = version,
        .len = strlen(version)
    };
    "#
}

#[cfunc]
pub fn module_name() -> &'static str {
    r#"
    #include <linux/string.h>
    "#;

    r#"
    static const char *s = KBUILD_MODNAME;
    return (struct const_rust_str){
        .data = s,
        .len = strlen(s)
    };
    "#
}

#[cfunc]
pub fn print_prefix() -> &'static str {
    r#"
    #include <linux/string.h>
    "#;

    r#"
    static const char *s = KBUILD_MODNAME ": ";
    return (struct const_rust_str){
        .data = s,
        .len = strlen(s)
    };
    "#
}
