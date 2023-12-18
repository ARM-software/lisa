/* SPDX-License-Identifier: GPL-2.0 */

#include <linux/fs.h>

#include "configs.h"
#include "feature_params.h"

void lisa_fs_remove(struct dentry *dentry);

struct lisa_cfg *allocate_lisa_cfg(const char *name)
{
	struct lisa_cfg *cfg;

	cfg = kzalloc(sizeof(*cfg), GFP_KERNEL);
	if (!cfg)
		return NULL;

	cfg->name = kstrdup(name, GFP_KERNEL);
	if (!cfg->name)
		goto error;

	return cfg;

error:
	kfree(cfg);
	return NULL;
}

void init_lisa_cfg(struct lisa_cfg *cfg, struct hlist_head *cfg_list,
		  struct dentry *dentry)
{
	cfg->dentry = dentry;
	hlist_add_head(&cfg->node, cfg_list);
}

void free_lisa_cfg(struct lisa_cfg *cfg)
{
	/* De-activate the config. */
	activate_lisa_cfg(cfg, false);
	drain_feature_param_entry_cfg(&cfg->list_param);

	/* Remove its dentries. */
	if (cfg->dentry)
		lisa_fs_remove(cfg->dentry);

	hlist_del(&cfg->node);
	kfree(cfg->name);
	kfree(cfg);
}

void drain_lisa_cfg(struct hlist_head *head)
{
	struct hlist_node *tmp;
	struct lisa_cfg *cfg;

	hlist_for_each_entry_safe(cfg, tmp, head, node)
		free_lisa_cfg(cfg);
}

struct lisa_cfg *find_lisa_cfg(struct hlist_head *cfg_list, const char *name)
{
	struct lisa_cfg *cfg;

	hlist_for_each_entry(cfg, cfg_list, node) {
		if (!strcmp(cfg->name, name))
			return cfg;
	}
	return NULL;
}

static int update_global_value(struct lisa_cfg *cfg, int new_value)
{
	struct feature_param_entry *entry, *rollback_entry;
	int ret = 0;

	/* For each parameter of the config. */
	hlist_for_each_entry(entry, &cfg->list_param, node_cfg) {
		if (!list_empty(&entry->list_values)) {
			/* For each value of this entry. */
			if (new_value)
				ret = feature_param_merge_common(entry);
			else
				ret = feature_param_remove_config_common(entry);
			if (ret) {
				rollback_entry = entry;
				goto rollback;
			}
		}
	}

	return ret;

rollback:
	hlist_for_each_entry(entry, &cfg->list_param, node_cfg) {
		if (entry == rollback_entry)
			break;

		if (!list_empty(&entry->list_values)) {
			/* For each value of this entry. */
			if (new_value)
				ret = feature_param_remove_config_common(entry);
			else
				ret = feature_param_merge_common(entry);
			if (ret) {
				pr_err("Could not rollback config values\n");
				return ret;
			}
		}
	}

	return ret;
}

static bool is_feature_set(char *name)
{
	struct feature_param_entry_value *val;

	/* Check whether the feature is in the global set_features list. */
	list_for_each_entry(val, &lisa_features_param.global_value, node)
		if (lisa_features_param.ops->is_equal(&name, val))
			return true;
	return false;
}

int activate_lisa_cfg(struct lisa_cfg *cfg, bool value)
{
	struct feature *feature;
	int ret;

	if (cfg->activated == value)
		return 0;

	/* All the global values have now been updated. Time to enable them. */

	ret = update_global_value(cfg, value);
	if (ret)
		return ret;

	cfg->activated = value;

	for_each_feature(feature) {
		if (!is_feature_set(feature->name)) {
			/*
			 * Feature was enabled, and de-activating this config
			 * disabled the feature.
			 */
			if (feature->__explicitly_enabled && !cfg->activated)
				deinit_single_features(feature->name);
			continue;
		}

		if (cfg->activated) {
			/*
			* Feature was enabled. By default, de-init before re-init the feature
			* to catch potential modifications.
			*/
			if (feature->__explicitly_enabled)
				deinit_single_features(feature->name);
			init_single_feature(feature->name);
			continue;
		}
	}

	return 0;
}