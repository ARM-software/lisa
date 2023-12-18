/* SPDX-License-Identifier: GPL-2.0 */

#include <linux/slab.h>
#include "features.h"

struct feature_param_entry_value *allocate_feature_param_entry_value(void)
{
	struct feature_param_entry_value *val;

	val = kmalloc(sizeof(*val), GFP_KERNEL);
	if (!val)
		return NULL;

	INIT_LIST_HEAD(&val->node);
	return val;
}

void init_feature_param_entry_value(struct feature_param_entry_value *val,
				    struct feature_param_entry *entry)
{
	/* Don't init the refcount for non-global values. */
	list_add_tail(&val->node, &entry->list_values);
	val->entry = entry;
}

void init_feature_param_entry_value_global(struct feature_param_entry_value *val,
					   struct feature_param_entry *entry,
					   struct list_head *head)
{
	refcount_set(&val->refcnt, 1);
	list_add_tail(&val->node, head);
	val->entry = entry;
}

void free_feature_param_entry_value(struct feature_param_entry_value *val)
{
	list_del(&val->node);
	kfree(val);
}

void drain_feature_param_entry_value(struct list_head *head)
{
	struct feature_param_entry_value *val, *tmp;

	list_for_each_entry_safe(val, tmp, head, node)
		free_feature_param_entry_value(val);
}

struct feature_param_entry *allocate_feature_param_entry(void)
{
	struct feature_param_entry *entry;

	entry = kmalloc(sizeof(*entry), GFP_KERNEL);
	return entry;
}

void init_feature_param_entry(struct feature_param_entry *entry,
			      struct lisa_cfg *cfg, struct feature_param *param)
{
	entry->param = param;
	entry->cfg = cfg;

	INIT_LIST_HEAD(&entry->list_values);
	hlist_add_head(&entry->node, &param->param_args);
	hlist_add_head(&entry->node_cfg, &cfg->list_param);
}

void free_feature_param_entry(struct feature_param_entry *entry)
{
	drain_feature_param_entry_value(&entry->list_values);
	hlist_del(&entry->node);
	hlist_del(&entry->node_cfg);
	kfree(entry);
}

void drain_feature_param_entry_cfg(struct hlist_head *head)
{
	struct feature_param_entry *entry;
	struct hlist_node *tmp;

	hlist_for_each_entry_safe(entry, tmp, head, node_cfg)
		free_feature_param_entry(entry);
}

int feature_param_add_new(struct feature_param_entry *entry, const char *v)
{
	struct feature_param *param = entry->param;
	struct feature_param_entry_value *val;
	int ret = 0;

	val = param->ops->set(v, entry);
	if (IS_ERR_OR_NULL(val))
		return IS_ERR(val) ? PTR_ERR(val) : -EINVAL;

	if (param->validate) {
		ret = param->validate(val);
		if (ret)
			goto error;
	}

	init_feature_param_entry_value(val, entry);

	return ret;

error:
	free_feature_param_entry_value(val);
	return ret;
}

int feature_param_merge_common(struct feature_param_entry *added_entry)
{
	struct feature_param_entry_value *added_val, *merged_val, *new_val;
	struct feature_param *param = added_entry->param;
	struct list_head *head;
	int ret = 0;

	/* Should have been checked already. */
	if (list_empty(&added_entry->list_values))
		return -EINVAL;

	head = &param->global_value;

	switch (param->mode) {
	case FT_PARAM_MODE_SINGLE:
		added_val = list_first_entry(&added_entry->list_values,
					     struct feature_param_entry_value, node);

		if (list_empty(head)) {
			/* No global value set yet. Allocate the single value. */

			new_val = allocate_feature_param_entry_value();
			if (!new_val) {
				ret = -ENOMEM;
				break;
			}

			init_feature_param_entry_value_global(
				new_val, added_val->entry, head);
			ret = param->ops->copy(added_val, new_val);
			if (ret)
				goto free;

			break;
		}

		/* Otherwise check added_val has the same value as the global config. */

		merged_val = list_first_entry(
			head, struct feature_param_entry_value, node);
		if (!param->ops->is_equal(&added_val->data, merged_val)) {
			pr_err("Single value must be set across configs for %s\n",
			       added_entry->param->name);
			feature_param_entry_print(param, added_val);
			feature_param_entry_print(param, merged_val);
			ret = -EEXIST;
			goto error;
		}

		break;

	case FT_PARAM_MODE_SET:
		list_for_each_entry(added_val, &added_entry->list_values, node) {
			bool found = false;

			/* Check the value doesn't already exist. */
			list_for_each_entry(merged_val, head, node) {
				if (param->ops->is_equal(&added_val->data, merged_val)) {
					/* If the value exists, increase the refcnt. */
					refcount_inc(&merged_val->refcnt);
					found = true;
					break;
				}
			}
			if (found)
				continue;

			/* Else allocate a new value */
			new_val = allocate_feature_param_entry_value();
			if (!new_val) {
				ret = -ENOMEM;
				break;
			}

			init_feature_param_entry_value_global(
				new_val, added_val->entry, head);
			ret = param->ops->copy(added_val, new_val);
			if (ret)
				goto free;
		}

		break;

	default:
		ret = -EINVAL;
		break;
	}

	return ret;

free:
	free_feature_param_entry_value(new_val);
error:
	return ret;
}

int feature_param_remove_config_common(struct feature_param_entry *removed_entry)
{
	struct feature_param_entry_value *removed_val, *merged_val;
	struct feature_param *param = removed_entry->param;
	struct list_head *head;
	int ret = 0;

	/* Should have been checked already. */
	if (list_empty(&removed_entry->list_values))
		return -EINVAL;

	head = &param->global_value;

	list_for_each_entry(removed_val, &removed_entry->list_values, node) {
		bool found = false;

		/* Check for an existing value. */
		list_for_each_entry(merged_val, head, node) {
			if (!param->ops->is_equal(&removed_val->data, merged_val))
				continue;

			found = true;

			/* This was the last reference. Free. */
			if (refcount_dec_and_test(&merged_val->refcnt)) {
				free_feature_param_entry_value(merged_val);
				break;
			}
		}

		if (!found) {
			pr_err("Value not found while deactivating config.\n");
			feature_param_entry_print(param, removed_val);
			ret = -EINVAL;
			break;
		}
	}

	return ret;
}

/////////////////////////////////////
//    lisa_features_param features
/////////////////////////////////////

int feature_param_lisa_validate(struct feature_param_entry_value *val)
{
	struct feature *feature;

	for_each_feature(feature) {
		if (!strcmp(feature->name, val->data))
			return 0;
	}
	return -EINVAL;
}

/* Handle feature names using the (struct feature_param) logic. */
struct feature_param lisa_features_param = {
	.name = "lisa_features_param",
	.mode = FT_PARAM_MODE_SET,
	.type = FT_PARAM_TYPE_AVAILABLE_FT,
	.perms = S_IFREG | S_IRUGO | S_IWUGO,
	.ops = &feature_param_ops_string,
	.validate = feature_param_lisa_validate,
	.param_args = HLIST_HEAD_INIT,
	.global_value = LIST_HEAD_INIT(lisa_features_param.global_value),
};

/////////////////////////////////////
//    feature_param type handlers
/////////////////////////////////////

static int
feature_param_set_common(struct feature_param_entry *entry, void *data)
{
	struct feature_param_entry_value *val;
	int ret = 0;

	switch (entry->param->mode) {
	case FT_PARAM_MODE_SINGLE:
		/* Single parameter, replace the pre-existing value. */
		/*
		 * TODO This might not be a good idea. The value is replaced
		 * even when the user thinks the value is appended.
		 * I.e. 'echo 1 >> file' will replace the pre-existing value.
		 */
		val = list_first_entry(&entry->list_values,
				       struct feature_param_entry_value, node);
		free_feature_param_entry_value(val);
		break;
	case FT_PARAM_MODE_SET:
		/* Don't allow duplicated values. */
		list_for_each_entry(val, &entry->list_values, node)
			if (entry->param->ops->is_equal(data, val)) {
				pr_err("Value already set.\n");
				ret = -EEXIST;
				break;
			}
		break;
	default:
		ret = -EINVAL;
		break;
	}

	return 0;
}

struct feature_param_entry_value *
feature_param_set_uint(const char *buf, struct feature_param_entry *entry)
{
	struct feature_param_entry_value *val;
	unsigned int input_val;
	int ret;

	if (!buf)
		return ERR_PTR(-EINVAL);

	ret = kstrtouint(buf, 0, &input_val);
	if (ret)
		return ERR_PTR(ret);

	if (list_empty(&entry->list_values))
		goto new_val;

	ret = feature_param_set_common(entry, &input_val);
	if (ret)
		return ERR_PTR(ret);

new_val:
	val = allocate_feature_param_entry_value();
	if (!val)
		return ERR_PTR(-ENOMEM);

	val->value = input_val;
	return val;
}

static size_t
feature_param_stringify_uint(const struct feature_param_entry_value *val,
			     char *buffer)
{
	return buffer ? sprintf(buffer, "%u", val->value) :
			snprintf(NULL, 0, "%u", val->value);
}

static int
feature_param_is_equal_uint(const void *data,
			    const struct feature_param_entry_value *val)
{
	return *(unsigned int *)data == val->value;
}

static int
feature_param_copy_uint(const struct feature_param_entry_value *src_val,
			struct feature_param_entry_value *val)
{
	val->value = src_val->value;
	return 0;
}

static struct feature_param_entry_value *
feature_param_set_string(const char *buf, struct feature_param_entry *entry)
{
	struct feature_param_entry_value *val;
	int ret;

	if (!buf)
		return ERR_PTR(-EINVAL);

	if (list_empty(&entry->list_values))
		goto new_val;

	ret = feature_param_set_common(entry, &buf);
	if (ret)
		return ERR_PTR(ret);

new_val:
	val = allocate_feature_param_entry_value();
	if (!val)
		return ERR_PTR(-ENOMEM);

	val->data = kstrdup(buf, GFP_KERNEL);
	return val;
}

static size_t
feature_param_stringify_string(const struct feature_param_entry_value *val,
			       char *buf)
{
	size_t size = strlen(val->data);
	if (buf)
		memcpy(buf, val->data, size);
	return size;
}

static int
feature_param_is_equal_string(const void *data,
			      const struct feature_param_entry_value *val)
{
	return !strcmp(*(char **)data, val->data);
}

static int
feature_param_copy_string(const struct feature_param_entry_value *src_val,
			  struct feature_param_entry_value *val)
{
	val->data = kstrdup(src_val->data, GFP_KERNEL);
	return 0;
}

const struct feature_param_ops feature_param_ops_uint = {
	.set = feature_param_set_uint,
	.stringify = feature_param_stringify_uint,
	.is_equal = feature_param_is_equal_uint,
	.copy = feature_param_copy_uint,
};

const struct feature_param_ops feature_param_ops_string = {
	.set = feature_param_set_string,
	.stringify = feature_param_stringify_string,
	.is_equal = feature_param_is_equal_string,
	.copy = feature_param_copy_string,
};
