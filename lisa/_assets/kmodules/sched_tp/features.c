/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/string.h>
#include <linux/types.h>

#include "main.h"
#include "features.h"

int __enable_feature(struct feature* feature) {
	int ret;

	if (!feature)
		return 1;

	mutex_lock(feature->lock);
	if (feature->enabled) {
		ret = feature->__enable_ret;
	} else {
		pr_info("Enabling lisa feature %s\n", feature->name);
		if (feature->enable)
			ret = feature->enable(feature);
		else
			ret = 0;
		feature->__enable_ret = ret;

		if (ret)
			pr_err("Failed to enable feature %s: %i", feature->name, ret);
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
			pr_info("Disabling lisa feature %s\n", feature->name);
			if (feature->disable)
				ret = feature->disable(feature);
			else
				ret = 0;
			if (ret)
				pr_err("Failed to disable feature %s: %i\n", feature->name, ret);
		} else {
			ret = 0;
		}
	}
	mutex_unlock(feature->lock);
	return ret;
}

typedef int (*feature_process_t)(struct feature*);

static int __select_feature(struct feature* feature, char **selected, size_t selected_len, feature_process_t process) {
	size_t i;
	if (selected_len) {
		for (i=0; i < selected_len; i++) {
			if (!strcmp(selected[i], feature->name))
				return process(feature);
		}
		return 0;
	} else if (!feature->__internal) {
		return process(feature);
	} else {
		return 0;
	}
}

static int __process_features(char **selected, size_t selected_len, feature_process_t process) {
	struct feature *feature;
	int ret = 0;

	for (feature=__lisa_features_start; feature < __lisa_features_stop; feature++) {
		ret |= __select_feature(feature, selected, selected_len, process);
	}
	return ret;
}


static int __list_feature(struct feature* feature) {
	if (!feature->__internal)
		printk(KERN_CONT "%s, ", feature->name);
	return 0;
}

static int __enable_feature_explicitly(struct feature* feature) {
	mutex_lock(feature->lock);
	feature->__explicitly_enabled = true;
	mutex_unlock(feature->lock);
	return __enable_feature(feature);
}

int init_features(char **selected, size_t selected_len) {
	BUG_ON(MAX_FEATURES < ((__lisa_features_stop - __lisa_features_start) / sizeof(struct feature)));

	pr_info("Available features: ");
	__process_features(NULL, 0, __list_feature);
	pr_info("\n");
	return __process_features(selected, selected_len, __enable_feature_explicitly);
}

static int __disable_explicitly_enabled_feature(struct feature* feature) {
	bool selected;
	int ret = 0;

	mutex_lock(feature->lock);
	selected = feature->__explicitly_enabled;
	mutex_unlock(feature->lock);
	if (selected)
		ret |= __disable_feature(feature);
	return ret;
}

int deinit_features(void) {
	return __process_features(NULL, 0, __disable_explicitly_enabled_feature);
}

int __placeholder_init(struct feature *feature) {
	pr_err("Feature not available: %s\n", feature->name);
	return 1;
}

int __placeholder_deinit(struct feature *feature) {
	return 0;
}
