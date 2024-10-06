/* SPDX-License-Identifier: GPL-2.0 */

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::runtime::pr_info;

extern "C" {
    fn myc_callback(x: u64) -> u64;
}

#[no_mangle]
pub extern "C" fn test_1(left: u64, right: u64) -> u64 {
    pr_info!("Rust: test_1");
    left + right
}

#[no_mangle]
pub extern "C" fn test_2(left: u64, right: u64) -> u64 {
    pr_info!("Rust: test_2");
    left + unsafe { myc_callback(right) }
}

#[no_mangle]
pub extern "C" fn test_3(left: u64, right: u64) -> u64 {
    pr_info!("Rust: test_3");
    let v: Vec<u64> = vec![left, right];
    let mut mymap = BTreeMap::new();
    mymap.insert(left, right);
    let val = mymap.get(&left).unwrap();
    let b = Arc::new(v);
    val + b[1]
}
