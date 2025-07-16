/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include "main.h"
#include "rust/lisakmod/bindings.h"
/* Import all the symbol namespaces that appear to be defined in the kernel
 * sources so that we won't trigger any warning
 */
#include "generated/symbol_namespaces.h"

static void modexit(void) {
	rust_mod_exit();
}

static int __init modinit(void) {
	return rust_mod_init();
}

module_init(modinit);
module_exit(modexit);

MODULE_LICENSE("GPL");
