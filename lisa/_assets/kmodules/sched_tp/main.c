/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include "main.h"
#include "features.h"

static char *features[MAX_FEATURES];
unsigned int features_len = 0;

module_param_array(features, charp, &features_len, 0);
MODULE_PARM_DESC(features, "Comma-separated list of features to enable. Available features are printed when loading the module");

static void modexit(void) {
	deinit_features();
}

static int __init modinit(void) {
	int ret = 0;
	ret |= init_features(features, features_len);

	/* Use one of the standard error code */
	if (ret)
		ret = -EINVAL;

	/* Call modexit() explicitly, since it will not be called when ret != 0.
	 * Not calling modexit() can (and will) result in kernel panic handlers
	 * installed by the module are not deregistered before the module code
	 * vanishes.
	 */
	if (ret) {
		pr_err("Some errors happened while loading LISA kernel module\n");
		modexit();
	}
	return ret;
}

module_init(modinit);
module_exit(modexit);

MODULE_LICENSE("GPL");
