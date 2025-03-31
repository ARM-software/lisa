/* SPDX-License-Identifier: GPL-2.0 */

use alloc::{collections::BTreeMap, sync::Arc, vec, vec::Vec};
use core::ffi::CStr;

use lisakmod_macros::inlinec::{c_eval, cconstant, cfunc};

use crate::{
    error::Error,
    features::define_feature,
    lifecycle::new_lifecycle,
    runtime::{
        kbox::KernelKBox,
        printk::{pr_debug, pr_info},
    },
};

unsafe extern "C" {
    fn myc_callback(x: u64) -> u64;
}

macro_rules! test {
    ($name:ident, $block:block) => {
        #[inline(never)]
        fn $name() {
            pr_info!("Running Rust test {}", stringify!($name));
            $block
            pr_debug!("Finished Rust test {}", stringify!($name));
        }
    }
}

test! {
    test1,
    {
        let x = unsafe { myc_callback(42) };
        assert_eq!(x, 43);
    }
}

test! {
    test2,
    {
        let left = 1;
        let right = 3;

        let v: Vec<u64> = vec![left, right];
        let mut mymap = BTreeMap::new();
        mymap.insert(left, right);
        let val = mymap.get(&left).unwrap();
        let b = Arc::new(v);
        let x = val + b[1];
        assert_eq!(x, 6);
    }
}

test! {
    test3,
    {
        let minalign = c_eval!("linux/slab.h", "ARCH_KMALLOC_MINALIGN", usize);
        // Check we don't get any C compilation error with duplicated code.
        let minalign2: usize = cconstant!("#include <linux/slab.h>", "ARCH_KMALLOC_MINALIGN").unwrap();
        assert_eq!(minalign, minalign2);
        assert!(minalign >= 8);
    }
}

test! {
    test4,
    {
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

        #[cfunc]
        unsafe fn my_cfunc_11(buf: &'static [u8]) -> bool {
            r#"

            return (buf.len == 11) && (buf.data[0] == 'h') && (buf.data[10] == 'd');
            "#
        }
        assert!(unsafe { my_cfunc_11(b"hello world") });

        #[cfunc]
        fn my_cfunc_12() -> &'static [u8] {
            r#"
            unsigned char *s = "hello world";
            struct slice_const_u8 x = {.data = s, .len = strlen(s)};
            return x;
            "#
        }
        assert_eq!(my_cfunc_12(), b"hello world");

        #[cfunc]
        unsafe fn my_cfunc_13<'a>(x: &'a mut [u8]) -> &'a mut [u8] {
            r#"
            x.data[0] = 'b';
            return x;
            "#
        }
        let mut buf: [u8; 1] = [b'a'];
        assert_eq!(unsafe { my_cfunc_13(&mut buf) }, b"b");
    }
}

test! {
    test5,
    {
        {
            let b = KernelKBox::new(42u8);
            assert_eq!(*b, 42);
            drop(b);
        }

        {
            let zst_addr = c_eval!("linux/slab.h", "ZERO_SIZE_PTR", *const u8);
            let b = KernelKBox::new(());
            assert_eq!(b.as_ptr() as usize, zst_addr as usize);
            drop(b);
        }
    }
}

use crate::runtime::sync::new_static_mutex;

test! {
    test6,
    {
        use core::ops::{Deref, DerefMut};

        use crate::runtime::sync::{Lock, LockdepClass, Mutex, SpinLock};
        {
            new_static_mutex!(STATIC_MUTEX, u32, 42);
            macro_rules! test_lock {
                ($lock:expr) => {{
                    let lock = $lock;
                    let mut guard = lock.lock();

                    assert!(lock.is_locked());
                    assert_eq!(*guard.deref(), 42);
                    assert_eq!(*guard.deref_mut(), 42);

                    let myref = guard.deref_mut();

                    assert!(lock.is_locked());

                    let _ = *myref;

                    assert!(lock.is_locked());
                    assert_eq!(*guard.deref(), 42);
                    assert_eq!(*guard.deref_mut(), 42);

                    drop(guard);

                    assert!(!lock.is_locked());

                    #[allow(dropping_references)]
                    drop(lock);
                }};
            }

            test_lock!(SpinLock::new(42, LockdepClass::new()));
            test_lock!(Mutex::new(42, LockdepClass::new()));
            test_lock!(&STATIC_MUTEX);
        }

        {
            use crate::runtime::sync::Rcu;
            let rcu = Rcu::new(42, LockdepClass::new());
            assert_eq!(*rcu.lock(), 42);
            rcu.update(43);
            assert_eq!(*rcu.lock(), 43);
        }
    }
}

test! {
    test7,
    {
        use crate::runtime::sysfs::{KObjType, KObject};

        let root = KObject::sysfs_module_root();

        let kobj_type = Arc::new(KObjType::new());
        let mut kobject = KObject::new(kobj_type.clone());
        let mut kobject2 = KObject::new(kobj_type.clone());

        kobject.add(Some(&root), "foo");
        let kobject = kobject.finalize().expect("Could not finalize kobject");

        kobject2.add(Some(&kobject), "bar");
        let kobject2 = kobject2.finalize().expect("Could not finalize kobject");

        drop(kobject2);
    }
}

pub fn init_tests() -> Result<(), Error> {
    // All of those functions are #[inline(never)] to ensure their name appear in backtraces if any
    // issue happens.
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();

    pr_info!("Rust tests finished");
    Ok(())
}

define_feature! {
    pub struct TestFeature,
    name: "rust_self_tests",
    visibility: Public,
    Service: (),
    Config: (),
    dependencies: [],
    init: |configs| {
        Ok(new_lifecycle!(|_| {
            init_tests()?;
            yield_!(Ok(Arc::new(())));
            Ok(())
        }))
    },
}
