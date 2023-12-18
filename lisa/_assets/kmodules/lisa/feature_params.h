/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _FEATURE__PARAM_H
#define _FEATURE__PARAM_H

#include <linux/types.h>

#include "configs.h"
#include "features.h"
#include "main.h"
#include "tp.h"

/*
 * Struct containing one single data value.
 * E.g., if the [0, 1, 2] pmu_raw_counters are set,
 * each value is stored in a (struct feature_param_entry_value).
 */
struct feature_param_entry_value {
	/*
	 * Member of a either:
	 * - (struct feature_param_entry)->list_values
	 * - (struct feature_param)->global_value
	 */
	struct list_head node;

	/* Parent entry. */
	struct feature_param_entry *entry;

	/*
	 * Refcount of the struct.
	 * Only meaningful for the (struct feature_param)->global_value, when values
	 * of multiple configs are merged.
	 */
	refcount_t refcnt;

	union {
		unsigned int value;
		void *data;
	};
};

/*
 * Struct containing a list of values.
 * E.g., if the [0, 1, 2] pmu_raw_counters are set for a 'config1',
 * and [2, 3, 4] are set for a 'config2', each set of values will be
 * referenced by a (struct feature_param_entry).
 */
struct feature_param_entry {
	/* Member of (struct feature_param)->param_args */
	struct hlist_node node;

	/* Member of (struct lisa_cfg)->list_param */
	struct hlist_node node_cfg;

	/* List of (struct feature_param_entry_value)->node. */
	struct list_head list_values;

	/* Parent param. */
	struct feature_param *param;

	/* Parent cfg. */
	struct lisa_cfg *cfg;
};

enum feature_param_mode {
	/*
	 * Among all configs, at most one value is allowed.
	 * I.e. for all the configs where a value is set,
	 * this value must be the same.
	 */
	FT_PARAM_MODE_SINGLE = 0,
	/*
	 * Merge values of all configs by creating a set.
	 * E.g. pmu_raw_counters can have different counters enabled in 
	 * different configs. The resulting value is a set of all the
	 * values of the different configs.
	 */
	FT_PARAM_MODE_SET = 1,
};

enum feature_param_type {
	/* Standard parameter. */
	FT_PARAM_TYPE_STD = 0,
	/* Specific to the 'lisa_features_param' parameter handling. */
	FT_PARAM_TYPE_AVAILABLE_FT,
};

struct feature_param {
	const char *name;
	enum feature_param_mode mode;
	enum feature_param_type type;
	struct dentry *dentry;
	umode_t perms;
	const struct feature_param_ops *ops;
	int (*validate)(struct feature_param_entry_value *);

	/* List of (struct feature_param_entry)->node. */
	struct hlist_head param_args;

	/* List of (struct feature_param_entry_value)->node. */
	struct list_head global_value;

	/* Parent feature. */
	struct feature *feature;
};

struct feature_param_ops {
	struct feature_param_entry_value *(*set) (const char *, struct feature_param_entry *);
	size_t (*stringify) (const struct feature_param_entry_value *, char *);
	int (*is_equal) (const void *, const struct feature_param_entry_value *);
	int (*copy) (const struct feature_param_entry_value *, struct feature_param_entry_value *);
};

extern struct feature_param lisa_features_param;
extern const struct feature_param_ops feature_param_ops_uint;
extern const struct feature_param_ops feature_param_ops_string;

#define GET_PARAM_HANDLER(type)						\
	__builtin_choose_expr(						\
		__builtin_types_compatible_p(type, char *),		\
		&feature_param_ops_string,				\
		__builtin_choose_expr(					\
		__builtin_types_compatible_p(type, unsigned int),	\
		&feature_param_ops_uint, NULL))

#define __PARAM(__name, __mode, __type, __perms, __param_type, __feature)	\
	(&(struct feature_param) {						\
		.name = __name,							\
		.mode = __mode,							\
		.type = __type,							\
		.perms = __perms,		 				\
		.ops = GET_PARAM_HANDLER(__param_type),				\
		.param_args = HLIST_HEAD_INIT,					\
		.feature = &__FEATURE_NAME(__feature),				\
	})

#define PARAM_SINGLE(name, perms, param_type, feature)	\
	__PARAM(name, FT_PARAM_MODE_SINGLE, FT_PARAM_TYPE_STD, perms, param_type, __EVENT_FEATURE(feature))
#define PARAM_SET(name, perms, param_type, feature) \
	__PARAM(name, FT_PARAM_MODE_SET, FT_PARAM_TYPE_STD, perms, param_type, __EVENT_FEATURE(feature))

#define FEATURE_PARAMS(...)				\
		.params = (struct feature_param* []){__VA_ARGS__, NULL}	\

#define EXPAND(...)	__VA_ARGS__
#define DEFINE_FEATURE_PARAMS(...) EXPAND(__VA_ARGS__)

#define for_each_feature_param(param, pparam, feature)	\
	if (feature->params)				\
		for (pparam = feature->params, param = *pparam; param != NULL; pparam++, param = *pparam)

#define feature_param_entry_print(param, val) {				\
	bool success = false;						\
	if (param->ops->stringify) {					\
		size_t size = param->ops->stringify(val, NULL);		\
		char *buf = kmalloc(size +1, GFP_KERNEL);		\
		if (buf) {						\
			buf[size] = '\0';				\
			size = param->ops->stringify(val, buf);		\
			pr_err("Value: %s\n", buf);			\
			kfree(buf);					\
			success = true;					\
		}							\
	}								\
	if (!success)							\
		pr_err("Value: failed to print\n");			\
}

struct feature_param_entry_value *allocate_feature_param_entry_value(void);
void init_feature_param_entry_value(struct feature_param_entry_value *val, struct feature_param_entry *entry);
void free_feature_param_entry_value(struct feature_param_entry_value *val);
void drain_feature_param_entry_value(struct list_head *head);

struct feature_param_entry *allocate_feature_param_entry(void);
void init_feature_param_entry(struct feature_param_entry *entry, struct lisa_cfg *cfg, struct feature_param *param);
void free_feature_param_entry(struct feature_param_entry *entry);
void drain_feature_param_entry_cfg(struct hlist_head *head);

int feature_param_add_new(struct feature_param_entry *entry, const char *v);
int feature_param_merge_common(struct feature_param_entry *added_entry);
int feature_param_remove_config_common(struct feature_param_entry *removed_entry);

#endif
