/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/cpumask.h>
#include <linux/energy_model.h>
#include <linux/jiffies.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/types.h>
#include <linux/sysfs.h>
#include <linux/kobject.h>

#include "main.h"
#include "features.h"
#include "wq.h"
#include "ftrace_events.h"

struct em_cpus_info {
	struct cpumask *cpus;
	struct kobj_attribute attr;
};

struct em_ps_info {
	struct em_perf_state *ps;
	struct attribute attr;
};

struct em_ps_group {
	struct kobject kobj;
	struct em_ps_info psi[4];
};

struct em_info {
	struct kobject *kobj;
	struct em_perf_domain *em;
	struct em_ps_group *psg;
	struct em_cpus_info *cpui;
	int num_psg;
};

static ssize_t em_ps_show(struct kobject *kobj, struct attribute *attr,
			char *buffer);
static void em_sysfs_kobj_release(struct kobject *kobj);

static struct kobject *em_main_kobj;
static struct em_info em_info[5];
static int em_id;
static DEFINE_MUTEX(em_lock);

static const struct sysfs_ops em_sysfs_ops = {
	.show = em_ps_show,
};

static struct kobj_type em_sysfs_kobj_type = {
	.sysfs_ops = &em_sysfs_ops,
	.release = em_sysfs_kobj_release,
};

static void em_sysfs_kobj_release(struct kobject *kobj)
{
	/* Nothing to do here, but print for checks */
	pr_info("EM_INFO: kobj put %s\n", kobj->name);
}

static ssize_t em_ps_show(struct kobject *kobj, struct attribute *attr, char *buffer)
{
	struct em_ps_info *psi = container_of(attr, struct em_ps_info, attr);
	struct em_perf_state *ps = psi->ps;
	int ret = -EINVAL;

	mutex_lock(&em_lock);
	if (!strcmp(attr->name, "frequency"))
		ret = snprintf(buffer, PAGE_SIZE, "%lu\n", ps->frequency);
	else if (!strcmp(attr->name, "power"))
		ret = snprintf(buffer, PAGE_SIZE, "%lu\n", ps->power);
	else if (!strcmp(attr->name, "cost"))
		ret = snprintf(buffer, PAGE_SIZE, "%lu\n", ps->cost);
	else if (!strcmp(attr->name, "flags"))
		ret = snprintf(buffer, PAGE_SIZE, "%lu\n", ps->flags);
	else
		pr_err("EMI_INFO: Invalid attribute");

	mutex_unlock(&em_lock);
	return ret;
}

static void em_sysfs_create_file(struct kobject *kobj, struct attribute *attr,
				 char *name)
{
	int ret;

	if (!kobj || !attr || !name)
		return;

	attr->name = name;
	attr->mode = 0644;
	sysfs_attr_init(attr);

	ret = sysfs_create_file(kobj, attr);
	if (ret)
		pr_warn("EM_INFO: Creating %s/%s failed %d\n",
			kobj->name, name, ret);
	else
		pr_info("EM_INFO: Created %s/%s\n", kobj->name, name);
}

static ssize_t em_cpus_show(struct kobject *kobj, struct kobj_attribute *attr,
			char *buf)
{
	struct em_cpus_info *cpui = container_of(attr, struct em_cpus_info, attr);

	int ret = -EINVAL;
	ret = snprintf(buf, PAGE_SIZE, "%*pbl\n", cpumask_pr_args(cpui->cpus));
	return ret;
}

static int em_create_cpus(struct cpumask *cpus, struct kobject *kobj) {
	int ret;
	struct em_info *emi;
	mutex_lock(&em_lock);

	emi = &em_info[em_id];
	emi->cpui = kmalloc(sizeof(*emi->cpui), GFP_KERNEL);
	if (!emi->cpui) {
		mutex_unlock(&em_lock);
		return -ENOMEM;
	}

	emi->cpui->cpus = cpus;
	emi->cpui->attr.attr.name = "cpus";
	emi->cpui->attr.attr.mode = 0644;
	sysfs_attr_init(emi->cpui->attr.attr);
	emi->cpui->attr.show = em_cpus_show;

	ret = sysfs_create_file(kobj, &emi->cpui->attr.attr);
	if (ret)
		pr_warn("EM_INFO: Creating %s/%s failed %d\n",
			kobj->name, "cpus", ret);
	else
		pr_info("EM_INFO: Created %s/%s\n", kobj->name, "cpus");

	mutex_unlock(&em_lock);

	return 0;
}

static int em_create_ps(struct em_ps_group *psg,
			struct em_perf_state *ps, struct kobject *kobj)
{
	struct kobject *kobj_ps;
	char name[24];
	int ret;

	snprintf(name, sizeof(name), "ps:%lu", ps->frequency);

	kobj_ps = &psg->kobj;

	ret = kobject_init_and_add(kobj_ps, &em_sysfs_kobj_type,
				   kobj, name);
	if (ret) {
		pr_warn("EM_INFO: Creating %s/%s failed %d\n",
			kobj->name, name, ret);
		return ret;
	} else {
		pr_info("EM_INFO: Created %s/%s\n", kobj->name, kobj_ps->name);
	}

	psg->psi[0].ps = ps;
	psg->psi[1].ps = ps;
	psg->psi[2].ps = ps;
	psg->psi[3].ps = ps;

	em_sysfs_create_file(kobj_ps, &psg->psi[0].attr, "frequency");
	em_sysfs_create_file(kobj_ps, &psg->psi[1].attr, "power");
	em_sysfs_create_file(kobj_ps, &psg->psi[2].attr, "cost");
	em_sysfs_create_file(kobj_ps, &psg->psi[3].attr, "flags");

	return 0;
}

static int em_create_ps_files(struct em_perf_domain *em,
				struct kobject *kobj)
{
	struct em_info *emi;
	int i, ret;

	mutex_lock(&em_lock);

	emi = &em_info[em_id++];
	emi->em = em;
	emi->num_psg = em->nr_perf_states;
	emi->kobj = kobj;

	emi->psg = kcalloc(emi->num_psg, sizeof(*emi->psg), GFP_KERNEL);
	if (!emi->psg) {
		mutex_unlock(&em_lock);
		return -ENOMEM;
	}

	for (i = 0; i < em->nr_perf_states; i++) {
		ret = em_create_ps(&emi->psg[i], &em->table[i], kobj);
		if (ret) {
			mutex_unlock(&em_lock);
			return ret;
		}
	}

	mutex_unlock(&em_lock);

	return 0;
}

static int init_em_sysfs(struct feature *_)
{
	struct em_perf_domain *em;
	struct cpumask *cpus;
	struct kobject *kobj;
	struct device *dev;
	int cpu;

	pr_info("EM sysfs init\n");

	em_main_kobj = kobject_create_and_add("energy_model", kernel_kobj);
	if (!em_main_kobj)
		return -ENOMEM;

	for_each_possible_cpu(cpu) {
		em = em_cpu_get(cpu);
		cpus = em_span_cpus(em);

		if (cpumask_first(cpus) != cpu)
			continue;

		dev = get_cpu_device(cpu);
		kobj = kobject_create_and_add(dev_name(dev), em_main_kobj);
		if (!kobj)
			return -ENOMEM;

		pr_info("EM_INFO: creating for cpu%d\n", cpu);
		em_create_cpus(cpus, kobj);
		em_create_ps_files(em, kobj);
	}

	return 0;
}

static int deinit_em_sysfs(struct feature *_)
{
	struct em_info *emi;
	int i, j;

	pr_info("EM sysfs exit\n");

	mutex_lock(&em_lock);

	for (i = 0; i < em_id; i++) {
		emi = &em_info[i];

		for (j = 0; j < emi->num_psg; j++) {
			struct em_ps_group *psg = &emi->psg[j];

			kobject_put(&psg->kobj);
		}
		kobject_put(emi->kobj);
	}

	kobject_put(em_main_kobj);

	for (i = 0; i < em_id; i++) {
		kfree(em_info[i].psg);
		kfree(em_info[i].cpui);
	}

	mutex_unlock(&em_lock);
	return 0;
}

DEFINE_FEATURE(__em_sysfs, init_em_sysfs, deinit_em_sysfs);
