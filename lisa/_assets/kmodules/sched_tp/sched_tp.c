/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/module.h>

#include <linux/sched.h>
#include <trace/events/sched.h>

#define CREATE_TRACE_POINTS
#include "sched_events.h"

static inline struct cfs_rq *get_group_cfs_rq(struct sched_entity *se)
{
#ifdef CONFIG_FAIR_GROUP_SCHED
	return se->my_q;
#else
	return NULL;
#endif
}

static inline struct cfs_rq *get_se_cfs_rq(struct sched_entity *se)
{
#ifdef CONFIG_FAIR_GROUP_SCHED
	return se->cfs_rq;
#else
	return NULL;
#endif
}

static inline void _trace_cfs(struct cfs_rq *cfs_rq,
			      void (*trace_event)(int, char*,
						  const struct sched_avg*))
{
	const struct sched_avg *avg;
	char path[PATH_SIZE];
	int cpu;

	avg = sched_tp_cfs_rq_avg(cfs_rq);
	sched_tp_cfs_rq_path(cfs_rq, path, PATH_SIZE);
	cpu = sched_tp_cfs_rq_cpu(cfs_rq);

	trace_event(cpu, path, avg);
 }

static inline void _trace_se(struct sched_entity *se,
			     void (*trace_event)(int, char*, char*, int,
						 const struct sched_avg*))
{
	void *gcfs_rq = get_group_cfs_rq(se);
	void *cfs_rq = get_se_cfs_rq(se);
	struct task_struct *p;
	char path[PATH_SIZE];
	char *comm;
	pid_t pid;
	int cpu;

	sched_tp_cfs_rq_path(gcfs_rq, path, PATH_SIZE);
	cpu = sched_tp_cfs_rq_cpu(cfs_rq);

	p = gcfs_rq ? NULL : container_of(se, struct task_struct, se);
	comm = p ? p->comm : "(null)";
	pid = p ? p->pid : -1;

	trace_event(cpu, path, comm, pid, &se->avg);
}

static void sched_pelt_cfs(void *data, struct cfs_rq *cfs_rq)
{
	if (trace_sched_pelt_cfs_enabled())
		_trace_cfs(cfs_rq, trace_sched_pelt_cfs);

	if (trace_uclamp_util_cfs_enabled()) {
		bool __maybe_unused is_root_rq = (&rq_of(cfs_rq)->cfs == cfs_rq);

		trace_uclamp_util_cfs(is_root_rq, rq_of(cfs_rq), cfs_rq);
	}
}

static void sched_pelt_rt(void *data, struct rq *rq)
{
	if (trace_sched_pelt_rt_enabled()) {
		const struct sched_avg *avg = sched_tp_rq_avg_rt(rq);
		int cpu = sched_tp_rq_cpu(rq);

		if (!avg)
			return;

		trace_sched_pelt_rt(cpu, avg);
	}
}

static void sched_pelt_dl(void *data, struct rq *rq)
{
	if (trace_sched_pelt_dl_enabled()) {
		const struct sched_avg *avg = sched_tp_rq_avg_dl(rq);
		int cpu = sched_tp_rq_cpu(rq);

		if (!avg)
			return;

		trace_sched_pelt_dl(cpu, avg);
	}
}

static void sched_pelt_irq(void *data, struct rq *rq)
{
	if (trace_sched_pelt_irq_enabled()){
		const struct sched_avg *avg = sched_tp_rq_avg_irq(rq);
		int cpu = sched_tp_rq_cpu(rq);

		if (!avg)
			return;

		trace_sched_pelt_irq(cpu, avg);
	}
}

static void sched_pelt_se(void *data, struct sched_entity *se)
{
	if (trace_sched_pelt_se_enabled())
		_trace_se(se, trace_sched_pelt_se);

	if (trace_uclamp_util_se_enabled()) {
		struct cfs_rq __maybe_unused *cfs_rq = get_se_cfs_rq(se);

		trace_uclamp_util_se(entity_is_task(se),
				     container_of(se, struct task_struct, se),
				     rq_of(cfs_rq));
	}
}

static void sched_overutilized(void *data, struct root_domain *rd, bool overutilized)
{
	if (trace_sched_overutilized_enabled()) {
		char span[SPAN_SIZE];

		cpumap_print_to_pagebuf(false, span, sched_tp_rd_span(rd));

		trace_sched_overutilized(overutilized, span);
	}
}

static void sched_update_nr_running(void *data, struct rq *rq, int change)
{
	if (trace_sched_update_nr_running_enabled()) {
		  int cpu = sched_tp_rq_cpu(rq);
		  int nr_running = sched_tp_rq_nr_running(rq);

		trace_sched_update_nr_running(cpu, change, nr_running);
	}
}

static void sched_util_est_cfs(void *data, struct cfs_rq *cfs_rq)
{
	if (trace_sched_util_est_cfs_enabled())
		_trace_cfs(cfs_rq, trace_sched_util_est_cfs);
}

static void sched_util_est_se(void *data, struct sched_entity *se)
{
	if (trace_sched_util_est_se_enabled())
		_trace_se(se, trace_sched_util_est_se);
}

static void sched_cpu_capacity(void *data, struct rq *rq)
{
	if (trace_sched_cpu_capacity_enabled())
		trace_sched_cpu_capacity(rq);
}

static int sched_tp_init(void)
{
	register_trace_pelt_cfs_tp(sched_pelt_cfs, NULL);
	register_trace_pelt_rt_tp(sched_pelt_rt, NULL);
	register_trace_pelt_dl_tp(sched_pelt_dl, NULL);
	register_trace_pelt_irq_tp(sched_pelt_irq, NULL);
	register_trace_pelt_se_tp(sched_pelt_se, NULL);
	register_trace_sched_overutilized_tp(sched_overutilized, NULL);
	register_trace_sched_update_nr_running_tp(sched_update_nr_running, NULL);
	register_trace_sched_util_est_cfs_tp(sched_util_est_cfs, NULL);
	register_trace_sched_util_est_se_tp(sched_util_est_se, NULL);
	register_trace_sched_cpu_capacity_tp(sched_cpu_capacity, NULL);

	return 0;
}

static void sched_tp_finish(void)
{
	unregister_trace_pelt_cfs_tp(sched_pelt_cfs, NULL);
	unregister_trace_pelt_rt_tp(sched_pelt_rt, NULL);
	unregister_trace_pelt_dl_tp(sched_pelt_dl, NULL);
	unregister_trace_pelt_irq_tp(sched_pelt_irq, NULL);
	unregister_trace_pelt_se_tp(sched_pelt_se, NULL);
	unregister_trace_sched_overutilized_tp(sched_overutilized, NULL);
	unregister_trace_sched_update_nr_running_tp(sched_update_nr_running, NULL);
	unregister_trace_sched_util_est_cfs_tp(sched_util_est_cfs, NULL);
	unregister_trace_sched_util_est_se_tp(sched_util_est_se, NULL);
	unregister_trace_sched_cpu_capacity_tp(sched_cpu_capacity, NULL);
}


module_init(sched_tp_init);
module_exit(sched_tp_finish);

MODULE_LICENSE("GPL");
