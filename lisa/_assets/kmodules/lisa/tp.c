/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/slab.h>
#include <linux/sched.h>
#include <trace/events/sched.h>

#define CREATE_TRACE_POINTS

#include "main.h"
#include "ftrace_events.h"
#include "sched_helpers.h"
#include "wq.h"
#include "tp.h"

#if HAS_KERNEL_FEATURE(CFS_PELT)
static inline void _trace_cfs(struct cfs_rq *cfs_rq,
			      void (*trace_event)(int, char*,
						  const struct sched_avg*))
{
	if (cfs_rq) {
		const struct sched_avg *avg = cfs_rq_avg(cfs_rq);
		char path[PATH_SIZE];
		int cpu = cfs_rq_cpu(cfs_rq);

		cfs_rq_path(cfs_rq, path, PATH_SIZE);
		trace_event(cpu, path, avg);
	}
}
#endif


#if HAS_KERNEL_FEATURE(SE_PELT)
static inline void _trace_se(struct sched_entity *se,
			     void (*trace_event)(int, const char*, const char*, int,
						 const struct sched_avg*))
{
	const struct cfs_rq *gcfs_rq = get_group_cfs_rq(se);
	const struct cfs_rq *cfs_rq = get_se_cfs_rq(se);
	char path[PATH_SIZE];

	cfs_rq_path(gcfs_rq, path, PATH_SIZE);
	int cpu = cfs_rq ? cfs_rq_cpu(cfs_rq) : -1;

	struct task_struct *p = gcfs_rq ? NULL : container_of(se, struct task_struct, se);
	char *comm = p ? p->comm : "(null)";
	pid_t pid = p ? p->pid : -1;

	trace_event(cpu, path, comm, pid, &se->avg);
}
#endif

#if HAS_KERNEL_FEATURE(CFS_PELT)
static void sched_pelt_cfs_probe(void *feature, struct cfs_rq *cfs_rq)
{
	_trace_cfs(cfs_rq, trace_lisa__sched_pelt_cfs);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_cfs, TP_PROBES(TP_PROBE("pelt_cfs_tp", sched_pelt_cfs_probe)));
#endif

#if HAS_KERNEL_FEATURE(RQ_UCLAMP)
static void uclamp_util_cfs_probe(void *feature, struct cfs_rq *cfs_rq) {
	bool __maybe_unused is_root_rq = ((struct cfs_rq *)&rq_of(cfs_rq)->cfs == cfs_rq);
	trace_lisa__uclamp_util_cfs(is_root_rq, rq_of(cfs_rq), cfs_rq);
}
DEFINE_TP_EVENT_FEATURE(lisa__uclamp_util_cfs, TP_PROBES(TP_PROBE("pelt_cfs_tp", uclamp_util_cfs_probe)));
#endif

#if HAS_KERNEL_FEATURE(RT_PELT)
static void sched_pelt_rt_probe(void *feature, struct rq *rq)
{
	if (rq) {
		const struct sched_avg *avg = rq_avg_rt(rq);
		int cpu = rq_cpu(rq);

		if (!avg)
			return;

		trace_lisa__sched_pelt_rt(cpu, avg);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_rt, TP_PROBES(TP_PROBE("pelt_rt_tp", sched_pelt_rt_probe)));
#endif

#if HAS_KERNEL_FEATURE(DL_PELT)
static void sched_pelt_dl_probe(void *feature, struct rq *rq)
{
	if (rq) {
		const struct sched_avg *avg = rq_avg_dl(rq);
		int cpu = rq_cpu(rq);

		if (!avg)
			return;

		trace_lisa__sched_pelt_dl(cpu, avg);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_dl, TP_PROBES(TP_PROBE("pelt_dl_tp", sched_pelt_dl_probe)));
#endif

#if HAS_KERNEL_FEATURE(IRQ_PELT)
static void sched_pelt_irq_probe(void *feature, struct rq *rq)
{
	if (rq) {
		const struct sched_avg *avg = rq_avg_irq(rq);
		int cpu = rq_cpu(rq);

		if (!avg)
			return;

		trace_lisa__sched_pelt_irq(cpu, avg);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_irq, TP_PROBES(TP_PROBE("pelt_irq_tp", sched_pelt_irq_probe)));
#endif

#if HAS_KERNEL_FEATURE(SE_PELT)
static void sched_pelt_se_probe(void *feature, struct sched_entity *se)
{
	_trace_se(se, trace_lisa__sched_pelt_se);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_pelt_se, TP_PROBES(TP_PROBE("pelt_se_tp", sched_pelt_se_probe)));
#endif

#if HAS_KERNEL_FEATURE(SE_UCLAMP)
static void uclamp_util_se_probe(void *feature, struct sched_entity *se)
{

	const struct cfs_rq __maybe_unused *cfs_rq = get_se_cfs_rq(se);

	trace_lisa__uclamp_util_se(entity_is_task(se),
				container_of(se, struct task_struct, se),
				rq_of(cfs_rq));
}
DEFINE_TP_EVENT_FEATURE(lisa__uclamp_util_se, TP_PROBES(TP_PROBE("pelt_se_tp", uclamp_util_se_probe)));
#endif

#if HAS_KERNEL_FEATURE(SCHED_OVERUTILIZED)
static void sched_overutilized_probe(void *feature, struct root_domain *rd, bool overutilized)
{
	if (trace_lisa__sched_overutilized_enabled()) {
		char span[SPAN_SIZE];

		cpumap_print_to_pagebuf(false, span, rd_span(rd));

		trace_lisa__sched_overutilized(overutilized, span);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_overutilized, TP_PROBES(TP_PROBE("sched_overutilized_tp", sched_overutilized_probe)));
#endif

#if HAS_KERNEL_FEATURE(RQ_NR_RUNNING)
static void sched_update_nr_running_probe(void *feature, struct rq *rq, int change)
{
	if (rq) {
		int cpu = rq_cpu(rq);
		int nr_running = rq_nr_running(rq);

		trace_lisa__sched_update_nr_running(cpu, change, nr_running);
	}
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_update_nr_running, TP_PROBES(TP_PROBE("sched_update_nr_running_tp", sched_update_nr_running_probe)));
#endif

#if HAS_KERNEL_FEATURE(CFS_UTIL_EST)
static void sched_util_est_cfs_probe(void *feature, struct cfs_rq *cfs_rq)
{
	_trace_cfs(cfs_rq, trace_lisa__sched_util_est_cfs);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_util_est_cfs, TP_PROBES(TP_PROBE("sched_util_est_cfs_tp", sched_util_est_cfs_probe)));
#endif

#if HAS_KERNEL_FEATURE(CFS_UTIL_EST_UNIFIED)
static void sched_util_est_cfs_unified_probe(void *feature, struct cfs_rq *cfs_rq)
{
	_trace_cfs(cfs_rq, trace_lisa__sched_util_est_cfs_unified);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_util_est_cfs_unified, TP_PROBES(TP_PROBE("sched_util_est_cfs_tp", sched_util_est_cfs_unified_probe)));
#endif

#if HAS_KERNEL_FEATURE(SE_UTIL_EST)
static void sched_util_est_se_probe(void *feature, struct sched_entity *se)
{
	_trace_se(se, trace_lisa__sched_util_est_se);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_util_est_se, TP_PROBES(TP_PROBE("sched_util_est_se_tp", sched_util_est_se_probe)));
#endif

#if HAS_KERNEL_FEATURE(SE_UTIL_EST_UNIFIED)
static void sched_util_est_se_unified_probe(void *feature, struct sched_entity *se)
{
	_trace_se(se, trace_lisa__sched_util_est_se_unified);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_util_est_se_unified, TP_PROBES(TP_PROBE("sched_util_est_se_tp", sched_util_est_se_unified_probe)));
#endif

#if HAS_KERNEL_FEATURE(RQ_CAPACITY)
static void sched_cpu_capacity_probe(void *feature, struct rq *rq)
{
	trace_lisa__sched_cpu_capacity(rq);
}
DEFINE_TP_EVENT_FEATURE(lisa__sched_cpu_capacity, TP_PROBES(TP_PROBE("sched_cpu_capacity_tp", sched_cpu_capacity_probe)));
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
