/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include "main.h"
#include "features.h"

static char *features[MAX_FEATURES];
unsigned int features_len = 0;

module_param_array(features, charp, &features_len, 0);
MODULE_PARM_DESC(features, "Comma-separated list of features to enable. Available features are printed when loading the module");

static void modexit(void) {
	if (deinit_features())
		pr_err("Some errors happened while unloading LISA kernel module\n");
}

static int __init modinit(void) {
	int ret = init_features(features, features_len);

	if (ret) {
		pr_err("Some errors happened while loading LISA kernel module\n");

		/* Use one of the standard error code */
		ret = -EINVAL;

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
			return ret;

		}
	}
	return 0;
}

module_init(modinit);
module_exit(modexit);

MODULE_LICENSE("GPL");
