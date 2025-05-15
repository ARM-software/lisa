/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/string.h>
#include <linux/types.h>

#include "main.h"
#include "features.h"

static int __reset_feature(struct feature* feature) {
	mutex_lock(feature->lock);

	/* All features should have been deinitialized at this point, so this
	 * should be 0
	 */
	BUG_ON(feature->__explicitly_enabled);

	feature->__enable_ret = 0;
	feature->data = NULL;

	mutex_unlock(feature->lock);
	return 0;
}

int __enable_feature(struct feature* feature) {
	int ret;

	if (!feature)
		return 1;

	mutex_lock(feature->lock);
	if (feature->enabled) {
		ret = feature->__enable_ret;
	} else {
		pr_info("Starting legacy feature %s\n", feature->name);
		if (feature->enable)
			ret = feature->enable(feature);
		else
			ret = 0;
		feature->__enable_ret = ret;

		if (ret)
			pr_err("Failed to start legacy feature %s: %i", feature->name, ret);
	}
	feature->enabled++;
	mutex_unlock(feature->lock);
	return ret;
}

int __disable_feature(struct feature* feature) {
	int ret;
	if (!feature)
		return 0;

	mutex_lock(feature->lock);
	if (!feature->enabled) {
		ret = 0;
	} else {
		feature->enabled--;
		if (!feature->enabled) {
			pr_info("Stopping legacy feature %s\n", feature->name);
			if (feature->disable)
				ret = feature->disable(feature);
			else
				ret = 0;
			if (ret)
				pr_err("Failed to stop legacy feature %s: %i\n", feature->name, ret);
		} else {
			ret = 0;
		}
	}
	mutex_unlock(feature->lock);
	return ret;
}

typedef int (*feature_process_t)(struct feature*);

static int __process_features(const char *const *selected, size_t selected_len, feature_process_t process) {
	int ret = 0;

	if (selected) {
		// User asked for a specific set of features
		for (size_t i=0; i < selected_len; i++) {
			bool found = false;

			for (struct feature *feature=__lisa_features_start; feature < __lisa_features_stop; feature++) {
				if (!strcmp(feature->name, selected[i])) {
					found = true;
					ret |= process(feature);
					break;
				}
			}
			if (!found) {
				pr_err("Unknown or compile-time disabled legacy feature: %s", selected[i]);
				ret |= 1;
			}
		}
	} else {
		// User did not ask for any particular feature, so try to enable all non-internal features.
		for (struct feature* feature=__lisa_features_start; feature < __lisa_features_stop; feature++) {
			if (!feature->__internal) {
				ret |= process(feature);
			}
		}
	}

	return ret;
}


static int __list_feature(struct feature* feature) {
	if (!feature->__internal)
		pr_info("  %s", feature->name);
	return 0;
}

static int __enable_feature_explicitly(struct feature* feature) {
	mutex_lock(feature->lock);
	feature->__explicitly_enabled++;
	mutex_unlock(feature->lock);
	return __enable_feature(feature);
}

int init_features(const char *const *selected, size_t selected_len) {
	BUG_ON(MAX_FEATURES < ((__lisa_features_stop - __lisa_features_start) / sizeof(struct feature)));
	__process_features(NULL, 0, __reset_feature);

	pr_info("Available legacy features:");
	__process_features(NULL, 0, __list_feature);
	return __process_features(selected, selected_len, __enable_feature_explicitly);
}

static int __disable_explicitly_enabled_feature(struct feature* feature) {
	int ret = 0;

	mutex_lock(feature->lock);
	while (feature->__explicitly_enabled) {
		mutex_unlock(feature->lock);
		ret |= __disable_feature(feature);
		mutex_lock(feature->lock);
		feature->__explicitly_enabled--;
	}
	mutex_unlock(feature->lock);
	return ret;
}

int deinit_features(void) {
	return __process_features(NULL, 0, __disable_explicitly_enabled_feature);
}

int __placeholder_init(struct feature *feature) {
	pr_err("Legacy feature not available: %s\n", feature->name);
	return 1;
}

int __placeholder_deinit(struct feature *feature) {
	return 0;
}
