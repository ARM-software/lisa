/* SPDX-License-Identifier: GPL-2.0 */

#include <linux/fs.h>

#include "configs.h"
#include "feature_params.h"

/* List of configs. */
struct hlist_head cfg_list;

void lisa_fs_remove(struct dentry *dentry);
void lisa_activate_config(bool value, struct lisa_cfg *cfg);

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

int init_lisa_cfg(struct lisa_cfg *cfg, struct hlist_head *cfg_list,
		  struct dentry *dentry)
{
	cfg->dentry = dentry;
	hlist_add_head(&cfg->node, cfg_list);
	return 0;
}

void free_lisa_cfg(struct lisa_cfg *cfg)
{
	/* De-activate the config. */
	lisa_activate_config(false, cfg);
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

struct lisa_cfg *find_lisa_cfg(const char *name)
{
	struct lisa_cfg *cfg;
	hlist_for_each_entry(cfg, &cfg_list, node) {
		if (!strcmp(cfg->name, name))
			return cfg;
	}
	return NULL;
}