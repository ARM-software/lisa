/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include "main.h"
#include "debugfs.h"
#include "features.h"
/* Import all the symbol namespaces that appear to be defined in the kernel
 * sources so that we won't trigger any warning
 */
#include "symbol_namespaces.h"

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
	int ret = init_features(features, features_len);
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

static int modinit(void) {
	/* First load the features, so there is no race with someone trying to
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
