/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _CONFIGS_H
#define _CONFIGS_H

#include "main.h"

struct lisa_cfg {
	struct dentry *dentry;

	/* Member of cfg_list. */
	struct hlist_node node;

	/* List of (struct feature_param_entry)->node_cfg. */
	struct hlist_head list_param;

	/* This config is currently activated. */
	bool activated;
	char *name;
};

extern struct hlist_head cfg_list;

struct lisa_cfg *allocate_lisa_cfg(const char *name);
int init_lisa_cfg(struct lisa_cfg *cfg, struct hlist_head *cfg_list,
		  struct dentry *dentry);
void free_lisa_cfg(struct lisa_cfg *cfg);
void drain_lisa_cfg(struct hlist_head *head);
struct lisa_cfg *find_lisa_cfg(const char *name);

#endif // _CONFIGS_H
