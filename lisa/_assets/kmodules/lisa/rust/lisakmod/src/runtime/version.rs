/* SPDX-License-Identifier: GPL-2.0 */

use lisakmod_macros::inlinec::cconstant;

pub const fn kernel_version() -> (u32, u32, u32) {
    const CODE: u32 = match cconstant!("#include <linux/version.h>", "LINUX_VERSION_CODE") {
        Some(x) => x,
        None => 0,
    };
    const MAJOR: u32 = (CODE >> 16) & 0xff;
    const SUBLEVEL: u32 = (CODE >> 8) & 0xff;
    const PATCH: u32 = CODE & 0xff;
    (MAJOR, SUBLEVEL, PATCH)
}
