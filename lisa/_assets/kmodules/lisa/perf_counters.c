// SPDX-License-Identifier: GPL-2.0
// Copyright (C) 2023 ARM Ltd.
#include <linux/perf_event.h>
#if defined(CONFIG_HW_PERF_EVENTS) && defined(CONFIG_ARM_PMU)
#include <linux/perf/arm_pmu.h>
#endif
#include "main.h"
#include "ftrace_events.h"
#include "tp.h"

#define MAX_PERF_COUNTERS	6

#define __PERFCTR_PARAM(name, param_name, type, param_type, desc)	\
	static type param_name[MAX_PERF_COUNTERS];			\
	static unsigned int param_name##_count;				\
	module_param_array_named(name, param_name, param_type,		\
				 &param_name##_count, 0644);		\
	MODULE_PARM_DESC(name, desc);

#define PERFCTR_PARAM(name, type, param_type, desc)	\
	__PERFCTR_PARAM(perf_counter_##name, name##_param, type, param_type, desc)

/* Set of perf counters to enable - comma-separated names of events */
PERFCTR_PARAM(generic_perf_events, char *, charp,
	      "Comma-separated list of symbolic names for generic perf events");
/* Set of perf counters to enable - comma-separated PMU raw counter ids */
PERFCTR_PARAM(pmu_raw_counters, unsigned int , uint,
	      "Comma-separated list of raw PMU event counter ids");

/* Initial set of supported counters to be enabled through module params */
struct perfctr_desc {
	/* unique name to identify the counter */
	const char 		*name;
	/* counter id (may be generic or raw) */
	u64			id;
	enum perf_type_id	type;
	/* enable by default if no counters requested */
	bool			default_on;
};

#define PERFCTR_DESC(__name, __id, __type, __en)				\
	((struct perfctr_desc) {						\
		.name = __name, .id = __id, .type = __type, .default_on = __en,	\
	})

#define PERFCTR_DESC_COUNT_HW(__name, __id, __en)	\
	PERFCTR_DESC(__name, __id, PERF_TYPE_HARDWARE, __en)

/* Initial set of supported counters to be enabled based on provided event names */
static const struct perfctr_desc perfctr_generic_lt [] = {
	PERFCTR_DESC_COUNT_HW("cpu_cycles", PERF_COUNT_HW_CPU_CYCLES, 1),
	PERFCTR_DESC_COUNT_HW("inst_retired", PERF_COUNT_HW_INSTRUCTIONS, 0),
	PERFCTR_DESC_COUNT_HW("l1d_cache", PERF_COUNT_HW_CACHE_REFERENCES, 0),
	PERFCTR_DESC_COUNT_HW("l1d_cache_refill", PERF_COUNT_HW_CACHE_MISSES, 0),
	PERFCTR_DESC_COUNT_HW("pc_write_retired", PERF_COUNT_HW_BRANCH_INSTRUCTIONS, 0),
	PERFCTR_DESC_COUNT_HW("br_mis_pred", PERF_COUNT_HW_BRANCH_MISSES, 0),
	PERFCTR_DESC_COUNT_HW("bus_cycles", PERF_COUNT_HW_BUS_CYCLES, 0),
	PERFCTR_DESC_COUNT_HW("stall_frontend", PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, 0),
	PERFCTR_DESC_COUNT_HW("stall_backend", PERF_COUNT_HW_STALLED_CYCLES_BACKEND, 0),
};

struct perfctr_event_entry {
	struct hlist_node		node;
	struct hlist_node		group_link;
	struct perf_event		*event;
	struct perfctr_event_group	*group;
	struct rcu_head			rcu_head;
};

struct perfctr_event_group {
	struct list_head	node;
	struct hlist_head	entries;
	u64			raw_id;
};

struct perfctr_pcpu_data {
	struct hlist_head	events;
};

struct perfctr_core {
	struct list_head			events;
	struct perfctr_pcpu_data __percpu	*pcpu_data;
	unsigned int				nr_events;
	unsigned int				max_nr_events;
};

static inline void perfctr_show_supported_generic_events(void)
{
	int i;

	pr_info("Possible (subject to actual support) generic perf events: ");
	for (i = 0; i < ARRAY_SIZE(perfctr_generic_lt); ++i)
		printk(KERN_CONT "%s, ", perfctr_generic_lt[i].name);
}

static void perfctr_event_release_entry(struct perfctr_event_entry *entry);

static int perfctr_event_activate_single(struct perfctr_core *perf_data,
					 struct perf_event_attr *attr)
{
	struct perfctr_event_entry *entry= NULL;
	struct perfctr_event_group *group;
	struct hlist_node *next;
	int cpu;

	group = kzalloc(sizeof(*group), GFP_KERNEL);
	if (!group)
		return -ENOMEM;

	group->raw_id = PERF_COUNT_HW_MAX;

	for_each_online_cpu(cpu) {
		entry = kzalloc(sizeof(*entry), GFP_KERNEL);
		if (!entry)
			goto activate_failed;

		entry->event =
			/* No overflow handler, at least not at this point */
			perf_event_create_kernel_counter(attr, cpu, NULL,
							 NULL, NULL);
		if (IS_ERR(entry->event)) {
			pr_err("Failed to create counter id=%llu on cpu%d\n",
			       attr->config, cpu);
			goto activate_failed;
		}

		perf_event_enable(entry->event);
		/*
		 * the PMU driver might still fail to assign a slot for a given
		 * counter (@see armpmu_add) which leaves the event ineffective
		 */
		if (entry->event->state != PERF_EVENT_STATE_ACTIVE) {
			pr_err("Failed to enable counter id=%llu on cpu%d\n",
			       attr->config, cpu);
			perf_event_disable(entry->event);
			perf_event_release_kernel(entry->event);
			goto activate_failed;
		}

		hlist_add_head_rcu(&entry->node,
				   &per_cpu_ptr(perf_data->pcpu_data, cpu)->events);

		hlist_add_head(&entry->group_link, &group->entries);
		entry->group = group;

		/* One-time only */
		if (group->raw_id != PERF_COUNT_HW_MAX)
			continue;
		if (attr->type == PERF_TYPE_RAW || !IS_ENABLED(CONFIG_ARM_PMU)) {
			group->raw_id = attr->config;
		} else {
			struct arm_pmu *arm_pmu;

			arm_pmu = to_arm_pmu(entry->event->pmu);
			/* There needs to be a better way to do this !!*/
			group->raw_id = arm_pmu->map_event(entry->event);
		}
	}
	list_add_tail(&group->node, &perf_data->events);
	++perf_data->nr_events;

	pr_info("%s event counter id=%llu activated on cpus=%*pbl",
		 attr->type == PERF_TYPE_RAW ? "PMU raw" : "Generic perf",
		 attr->config, cpumask_pr_args(cpu_online_mask));

	return 0;

activate_failed:
	if (entry)
		kfree(entry);

	hlist_for_each_entry(entry, &group->entries, group_link) {
		hlist_del_rcu(&entry->node);
	}
	synchronize_rcu();
	hlist_for_each_entry_safe(entry, next, &group->entries, group_link) {
		hlist_del(&entry->group_link);
		perfctr_event_release_entry(entry);
	}
	kfree(group);
	return -ENOMEM;

}

/* Lookup match type */
enum perfctr_match_type {
	PERFCTR_MATCH_NAME,
	PERFCTR_MATCH_STATUS
};

struct perfctr_match {
	union {
		char *name;  /* generic perf hw event name */
		bool status; /* enable by default */
	};
	enum perfctr_match_type type;
};

static int perfctr_event_activate(struct perfctr_core *perf_data,
				  const struct perfctr_match *match)
{
	int result = -EINVAL;
	int i;

	struct perf_event_attr attr = {
		.size		= sizeof(struct perf_event_attr),
		.pinned		= 1,
		.disabled	= 1,
	};

	for (i = 0; i < ARRAY_SIZE(perfctr_generic_lt); ++i) {
		switch (match->type) {
		case PERFCTR_MATCH_NAME:
			if (strcmp(match->name, perfctr_generic_lt[i].name))
				continue;
			break;
		case PERFCTR_MATCH_STATUS:
			if (match->status != perfctr_generic_lt[i].default_on)
				continue;
			else
				break;
		default:
			unreachable();
		}
		attr.config = perfctr_generic_lt[i].id;
		attr.type   = perfctr_generic_lt[i].type;

		result = perfctr_event_activate_single(perf_data, &attr);
		if (!result || match->type == PERFCTR_MATCH_NAME)
			break;
	}
	return result;
}

static void perfctr_event_release_entry(struct perfctr_event_entry *entry)
{
	perf_event_disable(entry->event);
	perf_event_release_kernel(entry->event);
	kfree(entry);
}

static void perfctr_events_release_group(struct perfctr_core *perf_data,
					  struct perfctr_event_group *group)
{
	struct perfctr_event_entry *entry;
	struct hlist_node *next;

	hlist_for_each_entry(entry, &group->entries, group_link) {
		hlist_del_rcu(&entry->node);
	}
	synchronize_rcu();
	hlist_for_each_entry_safe(entry, next, &group->entries, group_link) {
		hlist_del(&entry->group_link);
		perfctr_event_release_entry(entry);
	}
	list_del(&group->node);
	kfree(group);
	--perf_data->nr_events;
}

static void perfctr_events_release(struct perfctr_core *perf_data)
{
	struct perfctr_event_group *group, *next;

	list_for_each_entry_safe(group, next, &perf_data->events, node) {
		perfctr_events_release_group(perf_data, group);
	}
}

static void perfctr_sched_switch_probe(struct feature *feature, bool preempt,
				       struct task_struct *prev,
				       struct task_struct *next,
				       unsigned int prev_state)
{
	struct perfctr_core *perf_data = feature->data;

	if (trace_lisa__perf_counter_enabled()) {
		struct perfctr_event_entry *entry;
		struct hlist_head *entry_list;
		int cpu = smp_processor_id();
		u64 value = 0;

		entry_list = &per_cpu_ptr(perf_data->pcpu_data, cpu)->events;

		rcu_read_lock();
		hlist_for_each_entry_rcu(entry, entry_list, node) {
			/*
			 * The approach taken is a *semi*-safe one as:
			 * - the execution context is one as of the caller
			 *   (__schedule) with preemption and interrupts being
			 *   disabled
			 * - the events being traced are per-cpu ones only
			 * - kernel counter so no inheritance (no child events)
			 * - counter is being read on/for a local cpu
			 */
			struct perf_event *event = entry->event;

			event->pmu->read(event);
			value = local64_read(&event->count);
			trace_lisa__perf_counter(cpu, entry->group->raw_id, value);
		}
		rcu_read_unlock();
	}
}

static int perfctr_register_events(struct perfctr_core *perf_data)
{
	struct perfctr_match match;
	unsigned int count;
	int result = 0;

	count = generic_perf_events_param_count + pmu_raw_counters_param_count;
	if (count > perf_data->max_nr_events) {
		pr_err("Requested more than max %d counters\n",
		       perf_data->max_nr_events);
		return -EINVAL;
	}

	count = generic_perf_events_param_count;
	if (count) {
		match.type = PERFCTR_MATCH_NAME;
		for (; count > 0; --count) {
			match.name  = generic_perf_events_param[count - 1];
			result = perfctr_event_activate(perf_data, &match);
			if (result) {
				pr_err("Failed to activate event counter: %s\n",
				       match.name);
				perfctr_show_supported_generic_events();
				goto done;
			}
		}
	}

	count = pmu_raw_counters_param_count;
	if (count) {
		struct perf_event_attr attr = {
			.size		= sizeof(struct perf_event_attr),
			.type		= PERF_TYPE_RAW,
			.pinned		= 1,
			.disabled	= 1,
		};

		for (; count > 0; --count) {
			struct perfctr_event_group *group;
			bool duplicate = false;

			attr.config = pmu_raw_counters_param[count -1];
			/* Skip duplicates */
			list_for_each_entry(group, &perf_data->events, node) {
				if (group->raw_id == attr.config) {
					duplicate = true;
					break;
				}
			}

			result = duplicate ? 0 : perfctr_event_activate_single(perf_data, &attr);
			if (result) {
				pr_err("Failed to activate event counter: %llu\n",
				       attr.config);
				goto done;
			};

		}
	}
	if (!perf_data->nr_events) {
		match.type = PERFCTR_MATCH_STATUS;
		match.status = true;
		result = perfctr_event_activate(perf_data, &match);
	}
done:
	/* All or nothing ..... */
	if (result)
		perfctr_events_release(perf_data);
	return result;
}

static void perfctr_pmu_discover(struct perfctr_core *perf_data)
{
	struct perf_event *event;
	cpumask_var_t active_mask;
	int cpu;

	/*
	 * This is absolutely loathsome but there seems to be no other way
	 * to poke relevant pmu driver for details so, there it is ....
	 */
	struct perf_event_attr attr = {
		.type		= PERF_TYPE_HARDWARE,
		.size		= sizeof(struct perf_event_attr),
		.pinned		= 1,
		.disabled	= 1,
		.config		= PERF_COUNT_HW_CPU_CYCLES,
	};

	perf_data->max_nr_events = MAX_PERF_COUNTERS;

	if (!IS_ENABLED(CONFIG_ARM_PMU))
		return;

	if (!zalloc_cpumask_var(&active_mask, GFP_KERNEL))
		return;

	for_each_possible_cpu(cpu) {

		if (cpumask_test_cpu(cpu, active_mask))
		    continue;

		event = perf_event_create_kernel_counter(&attr, cpu, NULL ,
							 NULL, NULL);

		if (IS_ERR(event)) {
			pr_err("Failed to create an event (cpu%d) while discovery\n",
				cpu);
			break;
		}

		if (event->pmu) {
			struct arm_pmu *pmu = to_arm_pmu(event->pmu);

			perf_data->max_nr_events = min_t(unsigned int,
							 perf_data->max_nr_events,
							 pmu->num_events);

			cpumask_or(active_mask, active_mask, &pmu->supported_cpus);

		}
		perf_event_release_kernel(event);

		if (cpumask_equal(active_mask, cpu_possible_mask))
				break;
	}
	free_cpumask_var(active_mask);
	pr_info("Max of %d PMU counters available on cpus=%*pbl\n",
		perf_data->max_nr_events, cpumask_pr_args(cpu_possible_mask));
	return;
}

static int perfctr_disable(struct feature *feature);

static int perfctr_enable(struct feature *feature)
{
	struct perfctr_core *perf_data;

	if (!IS_ENABLED(CONFIG_HW_PERF_EVENTS)) {
		pr_err("Missing support for HW performance event counters\n");
		return 1;
	}

	perf_data = kzalloc(sizeof(*perf_data), GFP_KERNEL);
	if (!perf_data)
		return 1;

	INIT_LIST_HEAD(&perf_data->events);

	feature->data = perf_data;

	perf_data->pcpu_data = alloc_percpu(struct perfctr_pcpu_data);
	if (!perf_data->pcpu_data) {
		return 1;
	}

	perfctr_pmu_discover(perf_data);

	if (perfctr_register_events(perf_data))
		return 1;

	if (!perf_data->nr_events)
		pr_warn("No counters have been activated\n");

	return 0;

}

static int perfctr_disable(struct feature *feature)
{
	struct perfctr_core *perf_data = feature->data;

	if (!perf_data)
		return 0;

	if (perf_data->pcpu_data) {
		perfctr_events_release(perf_data);
		free_percpu(perf_data->pcpu_data);
	}
	kfree(perf_data);
	feature->data = NULL;
	return 0;
}
DEFINE_EXTENDED_TP_EVENT_FEATURE(lisa__perf_counter,
				 sched_switch, perfctr_sched_switch_probe,
				 perfctr_enable, perfctr_disable);
