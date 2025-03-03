/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include "main.h"
#include "generated/module_version.h"
#include "rust/lisakmod/bindings.h"
/* Import all the symbol namespaces that appear to be defined in the kernel
 * sources so that we won't trigger any warning
 */
#include "generated/symbol_namespaces.h"

static char* version = LISA_MODULE_VERSION;
module_param(version, charp, 0);
MODULE_PARM_DESC(version, "Module version defined as sha1sum of the module sources");

static void modexit(void) {
	rust_mod_exit();
}

static int __init modinit(void) {
	pr_info("Loading Lisa module version %s\n", LISA_MODULE_VERSION);
	if (strcmp(version, LISA_MODULE_VERSION)) {
		pr_err("Lisa module version check failed. Got %s, expected %s\n", version, LISA_MODULE_VERSION);
		return -EPROTO;
	}

	int ret = rust_mod_init();
	if (ret) {
		pr_err("Lisa module failed to initialize properly: %i\n", ret);
	}
	return ret;
}

module_init(modinit);
module_exit(modexit);

MODULE_LICENSE("GPL");
