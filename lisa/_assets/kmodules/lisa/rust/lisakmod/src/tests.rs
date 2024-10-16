/* SPDX-License-Identifier: GPL-2.0 */

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::CStr;

use crate::prelude::*;

extern "C" {
    fn myc_callback(x: u64) -> u64;
}

#[no_mangle]
pub extern "C" fn do_rust_tests() -> u64 {
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();

    pr_info!("Rust: tests finished");
    0
}

fn test_2() {
    pr_info!("Rust: test_2");
    let x = unsafe { myc_callback(42) };
    assert_eq!(x, 43);
}

fn test_3() {
    let left = 1;
    let right = 3;

    pr_info!("Rust: test_3");
    let v: Vec<u64> = vec![left, right];
    let mut mymap = BTreeMap::new();
    mymap.insert(left, right);
    let val = mymap.get(&left).unwrap();
    let b = Arc::new(v);
    let x = val + b[1];
    assert_eq!(x, 6);
}

fn test_4() {
    pr_info!("Rust: test_4");

    let minalign = get_c_macro!("linux/slab.h", ARCH_KMALLOC_MINALIGN, usize);
    // Check we don't get any C compilation error with duplicated code.
    let minalign2 = get_c_macro!("linux/slab.h", ARCH_KMALLOC_MINALIGN, usize);
    assert_eq!(minalign, minalign2);
    assert!(minalign >= 8);
}

fn test_5() {
    pr_info!("Rust: test_5");

    #[cfunc]
    fn my_cfunc_1() {
        "return;"
    }
    my_cfunc_1();

    #[cfunc]
    fn my_cfunc_2(x: u32) -> u64 {
        "return x * 2;"
    }
    assert_eq!(my_cfunc_2(42u32), 84u64);

    #[cfunc]
    unsafe fn my_cfunc_3(x: &CStr) -> &str {
        "return x;"
    }
    assert_eq!(unsafe { my_cfunc_3(c"hello") }, "hello");

    #[cfunc]
    fn my_cfunc_4() -> &'static str {
        r#"
        static const char *mystring = "hello world";
        return mystring;
        "#
    }
    assert_eq!(my_cfunc_4(), "hello world");

    #[cfunc]
    unsafe fn my_cfunc_5(x: &CStr) -> bool {
        "#include <linux/string.h>";

        r#"return strcmp(x, "hello") == 0;"#
    }
    assert!(unsafe { my_cfunc_5(c"hello") });

    #[cfunc]
    unsafe fn my_cfunc_6(x: Option<&CStr>) -> bool {
        "#include <linux/string.h>";

        r#"return x == NULL;"#
    }
    assert!(unsafe { my_cfunc_6(None) });
    assert!(!unsafe { my_cfunc_6(Some(c"hello")) });

    #[cfunc]
    fn my_cfunc_7() -> Option<&'static CStr> {
        r#"
        static const char *mystring = "hello world";
        return mystring;
        "#
    }
    assert_eq!(my_cfunc_7(), Some(c"hello world"));

    #[cfunc]
    fn my_cfunc_8() -> Option<&'static str> {
        r#"
        static const char *mystring = "hello world";
        return mystring;
        "#
    }
    assert_eq!(my_cfunc_8(), Some("hello world"));

    #[cfunc]
    fn my_cfunc_9() -> Option<&'static str> {
        r#"
        return NULL;
        "#
    }
    assert_eq!(my_cfunc_9(), None);

    #[cfunc]
    unsafe fn my_cfunc_10<'a>() -> Option<&'a str> {
        r#"
        return NULL;
        "#
    }
    assert_eq!(unsafe { my_cfunc_10() }, None);
}

fn test_6() {
    pr_info!("Rust: test_6");

    {
        let b = KBox::new(42u8);
        assert_eq!(*b, 42);
        drop(b);
    }

    {
        let zst_addr = get_c_macro!("linux/slab.h", ZERO_SIZE_PTR, *const u8);
        let b = KBox::new(());
        assert_eq!(b.as_ptr().as_ptr() as usize, zst_addr as usize);
        drop(b);
    }
}
