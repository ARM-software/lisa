/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/slab.h>

#ifndef _IN_TREE_BUILD
#include <linux/sched.h>
#else
#include <linux/sched/cputime.h>
#include <kernel/sched/sched.h>
#endif
#include <trace/events/sched.h>

#define CREATE_TRACE_POINTS

#include "main.h"
#include "ftrace_events.h"
#include "wq.h"
#include "tp.h"

static inline void _trace_cfs(struct cfs_rq *cfs_rq,
			      void (*trace_event)(int, char*,
						  const struct sched_avg*))
{
	const struct sched_avg *avg;
	char path[PATH_SIZE];
	int cpu;

	avg = lisa_cfs_rq_avg(cfs_rq);
	lisa_cfs_rq_path(cfs_rq, path, PATH_SIZE);
	cpu = lisa_cfs_rq_cpu(cfs_rq);

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

	lisa_cfs_rq_path(gcfs_rq, path, PATH_SIZE);
	cpu = lisa_cfs_rq_cpu(cfs_rq);

	p = gcfs_rq ? NULL : container_of(se, struct task_struct, se);
	comm = p ? p->comm : "(null)";
	pid = p ? p->pid : -1;

	trace_event(cpu, path, comm, pid, &se->avg);
}

static void sched_pelt_cfs_probe(void *feature, struct cfs_rq *cfs_rq)
{
	_trace_cfs(cfs_rq, trace_lisa__sched_pelt_cfs);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_cfs, pelt_cfs_tp, sched_pelt_cfs_probe);

static void uclamp_util_cfs_probe(void *feature, struct cfs_rq *cfs_rq) {
	bool __maybe_unused is_root_rq = ((struct cfs_rq *)&rq_of(cfs_rq)->cfs == cfs_rq);
	trace_lisa__uclamp_util_cfs(is_root_rq, rq_of(cfs_rq), cfs_rq);
}
DEFINE_TP_EVENT_FEATURE(lisa__uclamp_util_cfs, pelt_cfs_tp, uclamp_util_cfs_probe);

static void sched_pelt_rt_probe(void *feature, struct rq *rq)
{
	const struct sched_avg *avg = lisa_rq_avg_rt(rq);
	int cpu = lisa_rq_cpu(rq);

	if (!avg)
		return;

	trace_lisa__sched_pelt_rt(cpu, avg);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_rt, pelt_rt_tp, sched_pelt_rt_probe);

static void sched_pelt_dl_probe(void *feature, struct rq *rq)
{
	const struct sched_avg *avg = lisa_rq_avg_dl(rq);
	int cpu = lisa_rq_cpu(rq);

	if (!avg)
		return;

	trace_lisa__sched_pelt_dl(cpu, avg);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_dl, pelt_dl_tp, sched_pelt_dl_probe);

static void sched_pelt_irq_probe(void *feature, struct rq *rq)
{
	const struct sched_avg *avg = lisa_rq_avg_irq(rq);
	int cpu = lisa_rq_cpu(rq);

	if (!avg)
		return;

	trace_lisa__sched_pelt_irq(cpu, avg);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_irq, pelt_irq_tp, sched_pelt_irq_probe);

static void sched_pelt_se_probe(void *feature, struct sched_entity *se)
{
	_trace_se(se, trace_lisa__sched_pelt_se);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_se, pelt_se_tp, sched_pelt_se_probe);

static void uclamp_util_se_probe(void *feature, struct sched_entity *se)
{

	struct cfs_rq __maybe_unused *cfs_rq = get_se_cfs_rq(se);

	trace_lisa__uclamp_util_se(entity_is_task(se),
				container_of(se, struct task_struct, se),
				rq_of(cfs_rq));
}
DEFINE_TP_EVENT_FEATURE(lisa__uclamp_util_se, pelt_se_tp, uclamp_util_se_probe);

static void sched_overutilized_probe(void *feature, struct root_domain *rd, bool overutilized)
{
	if (trace_lisa__sched_overutilized_enabled()) {
		char span[SPAN_SIZE];

		cpumap_print_to_pagebuf(false, span, lisa_rd_span(rd));

		trace_lisa__sched_overutilized(overutilized, span);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_overutilized, sched_overutilized_tp, sched_overutilized_probe);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,9,0)
static void sched_update_nr_running_probe(void *feature, struct rq *rq, int change)
{
	if (trace_lisa__sched_update_nr_running_enabled()) {
		  int cpu = lisa_rq_cpu(rq);
		  int nr_running = lisa_rq_nr_running(rq);

		trace_lisa__sched_update_nr_running(cpu, change, nr_running);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_update_nr_running, sched_update_nr_running_tp, sched_update_nr_running_probe);

static void sched_util_est_cfs_probe(void *feature, struct cfs_rq *cfs_rq)
{
	_trace_cfs(cfs_rq, trace_lisa__sched_util_est_cfs);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_util_est_cfs, sched_util_est_cfs_tp, sched_util_est_cfs_probe);

static void sched_util_est_se_probe(void *feature, struct sched_entity *se)
{
	_trace_se(se, trace_lisa__sched_util_est_se);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_util_est_se, sched_util_est_se_tp, sched_util_est_se_probe);
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,10,0) && (defined(CONFIG_ARM64) || defined(CONFIG_ARM))
static void sched_cpu_capacity_probe(void *feature, struct rq *rq)
{
	trace_lisa__sched_cpu_capacity(rq);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_cpu_capacity, sched_cpu_capacity_tp, sched_cpu_capacity_probe);
#endif

static int init_tp(struct feature *_)
{
	return 0;
}

static int deinit_tp(struct feature *_)
{
	tracepoint_synchronize_unregister();
	return 0;
}

DEFINE_INTERNAL_FEATURE(__tp, init_tp, deinit_tp);
