/* SPDX-License-Identifier: Apache-2.0 */

use lisakmod_macros::inlinec::cconstant;

// #[cfunc]
// fn myfunc(x: u64, y: u64) -> u8 {
// "#include <myheader.h>";

// r#"
// #ifdef FOOBAR
// if (x == 3) {
// return 1;
// } else {
// return y;
// }
// #endif
// "#
// }

// #[cfunc]
// fn myfunc2(x: u64, y: u64) {
// "#include <myheader.h>";

// r#"
// return;
// "#
// }

// use core::ffi::CStr;
// #[cfunc]
// unsafe fn myfunc3<'a, 'b>(x: &'b CStr) -> &'a str
// where
// 'a: 'b,
// {
// "#include <myheader.h>";

// r#"
// return;
// "#
// }

// #[cfunc]
// fn myfunc4(x: *const *const u8) {
// // fn myfunc4(x: ConstPtr<ConstPtr<u8>>) {
// "#include <myheader.h>";

// r#"
// return;
// "#
// }

#[test]
fn test_cfunc() {
    // assert_eq!(myfunc(1, 2), 2);

    macro_rules! mymac {
        ($header:literal, $ty:literal) => {
            cconstant!(("#include <", $header, ">"), ("sizeof(", $ty, ")")).unwrap_or(8)
        };
    }
    assert_eq!(mymac!("linux/kobject.h", "int"), 8);
}
