/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/uaccess.h>

#include <linux/fs.h>
#include <linux/debugfs.h>

#include "main.h"
#include "debugfs.h"
#include "features.h"

static struct dentry *reload_file;
static struct dentry *lisa_debugfs;


static ssize_t reload_file_write(struct file *f, const char __user *user_buf, size_t n, loff_t *offset) {
	char buf[16] = {0};
	ssize_t ret = simple_write_to_buffer(buf, sizeof(buf), offset, user_buf, n);
	if (!strncmp(buf, "all\n", sizeof(buf))) {
		if (reload()) {
			return -ENOTRECOVERABLE;
		} else {
			return ret;
		}

	}
	return ret;
}

const struct file_operations reload_file_fops = {
	.owner = THIS_MODULE,
	.write = reload_file_write,
};

#ifdef CONFIG_DEBUG_FS
int debugfs_init(void)
{
	lisa_debugfs = debugfs_create_dir("lisa", NULL);
	if (!lisa_debugfs) {
		pr_err("Could not create lisa debugfs folder\n");
		return -ENOENT;
	}

	reload_file = debugfs_create_file("reload", 0644, lisa_debugfs, NULL,
					  &reload_file_fops);

	if (lisa_debugfs) {
		return 0;
	} else {
		pr_err("Could not create lisa debugfs folder\n");
		debugfs_exit();
		return -ENOENT;
	}
}

void debugfs_exit(void)
{
	if (lisa_debugfs)
		debugfs_remove_recursive(lisa_debugfs);
}

#else /* CONFIG_DEBUG_FS */

int debugfs_init(void)
{
	return 0;
}

void debugfs_exit(void)
{
}
#endif /* CONFIG_DEBUG_FS */
