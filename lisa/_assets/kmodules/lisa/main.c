/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include "main.h"
#include "debugfs.h"
#include "features.h"
#include "introspection.h"
#include "generated/module_version.h"
/* Import all the symbol namespaces that appear to be defined in the kernel
 * sources so that we won't trigger any warning
 */
#include "generated/symbol_namespaces.h"

static char* version = LISA_MODULE_VERSION;
module_param(version, charp, 0);
MODULE_PARM_DESC(version, "Module version defined as sha1sum of the module sources");

static char *features[MAX_FEATURES];
static unsigned int features_len = 0;
module_param_array(features, charp, &features_len, 0600);
MODULE_PARM_DESC(features, "Comma-separated list of features to enable. Available features are printed when loading the module");


static int exit(void) {
	int ret = deinit_features();
	if (ret)
		pr_err("Some errors happened while unloading LISA kernel module: %d\n", ret);
	return ret;
}

static void modexit(void) {
	debugfs_exit();
	exit();
}

static int init(void) {
	int ret;

	pr_info("Loading Lisa module version %s\n", LISA_MODULE_VERSION);
	if (strcmp(version, LISA_MODULE_VERSION)) {
		pr_err("Lisa module version check failed. Got %s, expected %s\n", version, LISA_MODULE_VERSION);
		return -EPROTO;
	}

	pr_info("Kernel features detected. This will impact the module features that are available:\n");
	const char *kernel_feature_names[] = {__KERNEL_FEATURE_NAMES};
	const bool kernel_feature_values[] = {__KERNEL_FEATURE_VALUES};
	for (size_t i=0; i < ARRAY_SIZE(kernel_feature_names); i++) {
		pr_info("  %s: %s\n", kernel_feature_names[i], kernel_feature_values[i] ? "enabled" : "disabled");
	}

	ret = init_features(features, features_len);
	if (ret)
		pr_err("Some errors happened while loading LISA kernel module: %d\n", ret);
	return ret;
}

int reload(void) {
	int ret = 0;
	ret |= exit();
	ret |= init();

	if (ret)
		pr_err("Some errors happened while reloading LISA module: %d\n", ret);
	return ret;
}

static int __init modinit(void) {
	 * reload from debugfs at the same time.
	 */
	if (init()) {
		/* If the user selected features manually, make module loading fail so
		* that they are aware that things went wrong. Otherwise, just
		* keep going as the user just wanted to enable as many features
		* as possible.
		*/
		if (features_len) {
			/* Call modexit() explicitly, since it will not be called when ret != 0.
			* Not calling modexit() can (and will) result in kernel panic handlers
			* installed by the module are not deregistered before the module code
			* vanishes.
			*/
			modexit();

			/* Use one of the standard error code */
			return -EINVAL;

		} else {
			return 0;
		}
	}

	int ret = debugfs_init();
	if (ret) {
		pr_err("Some errors happened while setting up debugfs for LISA kernel module: %d\n", ret);
		return -EINVAL;
	}
	return 0;
}



module_init(modinit);
module_exit(modexit);

MODULE_LICENSE("GPL");
