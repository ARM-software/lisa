/* SPDX-License-Identifier: GPL-2.0 */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM lisa

#define MAX_SPAN_SIZE		128

#if !defined(_FTRACE_EVENTS_H) || defined(TRACE_HEADER_MULTI_READ)
#define _FTRACE_EVENTS_H

#define PATH_SIZE		64
#define __SPAN_SIZE		(round_up(NR_CPUS, 4)/4)
#define SPAN_SIZE		(__SPAN_SIZE > MAX_SPAN_SIZE ? MAX_SPAN_SIZE : __SPAN_SIZE)

#include <linux/version.h>
#include <linux/tracepoint.h>
#include <linux/version.h>

#include "sched_helpers.h"

#if HAS_MEMBER(struct, sched_avg, runnable_load_avg)
#define RBL_LOAD_ENTRY		rbl_load
#define RBL_LOAD_MEMBER		runnable_load_avg
#define RBL_LOAD_STR		"rbl_load"
#elif HAS_MEMBER(struct, sched_avg, runnable_avg)
#define RBL_LOAD_ENTRY		runnable
#define RBL_LOAD_MEMBER		runnable_avg
#define RBL_LOAD_STR		"runnable"
#endif

#if HAS_KERNEL_FEATURE(CFS_PELT)
TRACE_EVENT(lisa__sched_pelt_cfs,

	TP_PROTO(int cpu, char *path, const struct sched_avg *avg),

	TP_ARGS(cpu, path, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__array(	char,		path,	PATH_SIZE	)
		__field(	unsigned long,	load			)
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		__field(	unsigned long,	RBL_LOAD_ENTRY		)
#endif
		__field(	unsigned long,	util			)
		__field(	unsigned long long, update_time	        )
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		strlcpy(__entry->path, path, PATH_SIZE);
		__entry->load		= avg->load_avg;
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		__entry->RBL_LOAD_ENTRY	= avg->RBL_LOAD_MEMBER;
#endif
		__entry->util		= avg->util_avg;
		__entry->update_time    = avg->last_update_time;
	),

	TP_printk(
		"cpu=%d path=%s load=%lu util=%lu update_time=%llu"
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		" " RBL_LOAD_STR "=%lu"
#endif
		,
		__entry->cpu, __entry->path, __entry->load,
		__entry->util, __entry->update_time
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		,__entry->RBL_LOAD_ENTRY
#endif
	)
);
#endif

#if HAS_TYPE(struct, sched_avg)
DECLARE_EVENT_CLASS(lisa__sched_pelt_rq_template,

	TP_PROTO(int cpu, const struct sched_avg *avg),

	TP_ARGS(cpu, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__field(	unsigned long,	load			)
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		__field(	unsigned long,	RBL_LOAD_ENTRY		)
#endif
		__field(	unsigned long,	util			)
		__field(	unsigned long long, update_time	        )
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		__entry->load		= avg->load_avg;
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		__entry->RBL_LOAD_ENTRY	= avg->RBL_LOAD_MEMBER;
#endif
		__entry->util		= avg->util_avg;
		__entry->update_time    = avg->last_update_time;
	),

	TP_printk(
		"cpu=%d load=%lu util=%lu update_time=%llu"
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		" " RBL_LOAD_STR "=%lu"
#endif
		,
		__entry->cpu, __entry->load,
		__entry->util, __entry->update_time
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		,__entry->RBL_LOAD_ENTRY
#endif
	)
);
#endif

#if HAS_KERNEL_FEATURE(RT_PELT)
DEFINE_EVENT(lisa__sched_pelt_rq_template, lisa__sched_pelt_rt,
	TP_PROTO(int cpu, const struct sched_avg *avg),
	TP_ARGS(cpu, avg));
#endif

#if HAS_KERNEL_FEATURE(DL_PELT)
DEFINE_EVENT(lisa__sched_pelt_rq_template, lisa__sched_pelt_dl,
	TP_PROTO(int cpu, const struct sched_avg *avg),
	TP_ARGS(cpu, avg));
#endif

#if HAS_KERNEL_FEATURE(IRQ_PELT)
DEFINE_EVENT(lisa__sched_pelt_rq_template, lisa__sched_pelt_irq,
	TP_PROTO(int cpu, const struct sched_avg *avg),
	TP_ARGS(cpu, avg));
#endif

#if HAS_KERNEL_FEATURE(SE_PELT)
TRACE_EVENT(lisa__sched_pelt_se,

	TP_PROTO(int cpu, char *path, char *comm, int pid, const struct sched_avg *avg),

	TP_ARGS(cpu, path, comm, pid, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__array(	char,		path,	PATH_SIZE	)
		__array(	char,		comm,	TASK_COMM_LEN	)
		__field(	int,		pid			)
		__field(	unsigned long,	load			)
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		__field(	unsigned long,	RBL_LOAD_ENTRY		)
#endif
		__field(	unsigned long,	util			)
		__field(	unsigned long long, update_time	        )
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		strlcpy(__entry->path, path, PATH_SIZE);
		strlcpy(__entry->comm, comm, TASK_COMM_LEN);
		__entry->pid		= pid;
		__entry->load		= avg->load_avg;
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		__entry->RBL_LOAD_ENTRY	= avg->RBL_LOAD_MEMBER;
#endif
		__entry->util		= avg->util_avg;
		__entry->update_time    = avg->last_update_time;
	),

	TP_printk(
		"cpu=%d path=%s comm=%s pid=%d load=%lu util=%lu update_time=%llu"
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		" " RBL_LOAD_STR "=%lu"
#endif
		,
		__entry->cpu, __entry->path, __entry->comm, __entry->pid,
		__entry->load, __entry->util, __entry->update_time
#if HAS_KERNEL_FEATURE(SCHED_AVG_RBL)
		,__entry->RBL_LOAD_ENTRY
#endif
	)
);
#endif

#if HAS_KERNEL_FEATURE(SCHED_OVERUTILIZED)
TRACE_EVENT(lisa__sched_overutilized,

	TP_PROTO(int overutilized, char *span),

	TP_ARGS(overutilized, span),

	TP_STRUCT__entry(
		__field(	int,		overutilized		)
		__array(	char,		span,	SPAN_SIZE	)
	),

	TP_fast_assign(
		__entry->overutilized	= overutilized;
		strlcpy(__entry->span, span, SPAN_SIZE);
	),

	TP_printk("overutilized=%d span=0x%s",
		  __entry->overutilized, __entry->span)
);
#endif

#if HAS_KERNEL_FEATURE(RQ_NR_RUNNING)
TRACE_EVENT(lisa__sched_update_nr_running,

	    TP_PROTO(int cpu, int change, unsigned int nr_running),

	    TP_ARGS(cpu, change, nr_running),

	    TP_STRUCT__entry(
			     __field(         int,        cpu           )
			     __field(         int,        change        )
			     __field(unsigned int,        nr_running    )
			     ),

	    TP_fast_assign(
			   __entry->cpu        = cpu;
			   __entry->change     = change;
			   __entry->nr_running = nr_running;
			   ),

	    TP_printk("cpu=%d change=%d nr_running=%d", __entry->cpu, __entry->change, __entry->nr_running)
	    );
#endif

#if HAS_KERNEL_FEATURE(SE_UTIL_EST)
TRACE_EVENT(lisa__sched_util_est_se,

	TP_PROTO(int cpu, char *path, char *comm, int pid,
		 const struct sched_avg *avg),

	TP_ARGS(cpu, path, comm, pid, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__array(	char,		path,	PATH_SIZE	)
		__array(	char,		comm,	TASK_COMM_LEN	)
		__field(	int,		pid			)
		__field( 	unsigned int,	enqueued		)
		__field( 	unsigned int,	ewma			)
		__field(	unsigned long,	util			)
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		strlcpy(__entry->path, path, PATH_SIZE);
		strlcpy(__entry->comm, comm, TASK_COMM_LEN);
		__entry->pid		= pid;
		__entry->enqueued	= avg->util_est.enqueued & ~UTIL_AVG_UNCHANGED;
		__entry->ewma		= avg->util_est.ewma;
		__entry->util		= avg->util_avg;
	),

	TP_printk("cpu=%d path=%s comm=%s pid=%d enqueued=%u ewma=%u util=%lu",
		  __entry->cpu, __entry->path, __entry->comm, __entry->pid,
		  __entry->enqueued, __entry->ewma, __entry->util)
);
#endif

#if HAS_KERNEL_FEATURE(CFS_UTIL_EST)
TRACE_EVENT(lisa__sched_util_est_cfs,

	TP_PROTO(int cpu, char *path, const struct sched_avg *avg),

	TP_ARGS(cpu, path, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__array(	char,		path,	PATH_SIZE	)
		__field( 	unsigned int,	enqueued		)
		__field( 	unsigned int,	ewma			)
		__field(	unsigned long,	util			)
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		strlcpy(__entry->path, path, PATH_SIZE);
		__entry->enqueued	= avg->util_est.enqueued;
		__entry->ewma		= avg->util_est.ewma;
		__entry->util		= avg->util_avg;
	),

	TP_printk("cpu=%d path=%s enqueued=%u ewma=%u util=%lu",
		  __entry->cpu, __entry->path, __entry->enqueued,
		 __entry->ewma, __entry->util)
);
#endif

#if HAS_KERNEL_FEATURE(SE_UCLAMP)
TRACE_EVENT_CONDITION(lisa__uclamp_util_se,

	TP_PROTO(bool is_task, struct task_struct *p, struct rq *rq),

	TP_ARGS(is_task, p, rq),

	TP_CONDITION(is_task),

	TP_STRUCT__entry(
		__field(	pid_t,	pid			)
		__array(	char,	comm,   TASK_COMM_LEN	)
		__field(	 int,	cpu			)
		__field(unsigned long,	util_avg		)
		__field(unsigned long,	uclamp_avg		)
		__field(unsigned long,	uclamp_min		)
		__field(unsigned long,	uclamp_max		)
	),

	TP_fast_assign(
		__entry->pid            = p->pid;
		memcpy(__entry->comm, p->comm, TASK_COMM_LEN);
		__entry->cpu            = rq_cpu(rq);
		__entry->util_avg       = p->se.avg.util_avg;
		__entry->uclamp_avg     = uclamp_rq_util_with(rq, p->se.avg.util_avg);

#    if HAS_KERNEL_FEATURE(CFS_UCLAMP)
		__entry->uclamp_min     = rq->uclamp[UCLAMP_MIN].value;
		__entry->uclamp_max     = rq->uclamp[UCLAMP_MAX].value;
#    endif
		),

	TP_printk("pid=%d comm=%s cpu=%d util_avg=%lu uclamp_avg=%lu"
#    if HAS_KERNEL_FEATURE(CFS_UCLAMP)
		  " uclamp_min=%lu uclamp_max=%lu"
#    endif
		  ,
		  __entry->pid, __entry->comm, __entry->cpu,
		  __entry->util_avg, __entry->uclamp_avg
#    if HAS_KERNEL_FEATURE(CFS_UCLAMP)
		  ,__entry->uclamp_min, __entry->uclamp_max
#    endif
		)
);
#else
#define trace_lisa__uclamp_util_se(is_task, p, rq) while(false) {}
#define trace_lisa__uclamp_util_se_enabled() (false)
#endif

#if HAS_KERNEL_FEATURE(CFS_UCLAMP)
TRACE_EVENT_CONDITION(lisa__uclamp_util_cfs,

	TP_PROTO(bool is_root, struct rq *rq, struct cfs_rq *cfs_rq),

	TP_ARGS(is_root, rq, cfs_rq),

	TP_CONDITION(is_root),

	TP_STRUCT__entry(
		__field(	 int,	cpu			)
		__field(unsigned long,	util_avg		)
		__field(unsigned long,	uclamp_avg		)
		__field(unsigned long,	uclamp_min		)
		__field(unsigned long,	uclamp_max		)
	),

	TP_fast_assign(
		__entry->cpu            = rq_cpu(rq);
		__entry->util_avg       = cfs_rq->avg.util_avg;
		__entry->uclamp_avg     = uclamp_rq_util_with(rq, cfs_rq->avg.util_avg);
		__entry->uclamp_min     = rq->uclamp[UCLAMP_MIN].value;
		__entry->uclamp_max     = rq->uclamp[UCLAMP_MAX].value;
		),

	TP_printk("cpu=%d util_avg=%lu uclamp_avg=%lu "
		  "uclamp_min=%lu uclamp_max=%lu",
		  __entry->cpu, __entry->util_avg, __entry->uclamp_avg,
		  __entry->uclamp_min, __entry->uclamp_max)
);
#else
#define trace_lisa__uclamp_util_cfs(is_root, cpu, cfs_rq) while(false) {}
#define trace_lisa__uclamp_util_cfs_enabled() (false)
#endif

#if HAS_KERNEL_FEATURE(RQ_CAPACITY)
TRACE_EVENT(lisa__sched_cpu_capacity,

	TP_PROTO(struct rq *rq),

	TP_ARGS(rq),

	TP_STRUCT__entry(
		__field(	int,		cpu		)
		__field(	unsigned long,	capacity	)
#if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
		__field(	unsigned long,	capacity_orig	)
		__field(	unsigned long,	capacity_curr	)
#endif
	),

	TP_fast_assign(
		__entry->cpu		= rq->cpu;
		__entry->capacity	= rq->cpu_capacity;
#if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
		__entry->capacity_orig	= rq_cpu_orig_capacity(rq);
		__entry->capacity_curr	= rq_cpu_current_capacity(rq);
#endif
	),

	TP_printk(
		"cpu=%d"
		" capacity=%lu"
#if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
		" capacity_orig=%lu"
		" capacity_curr=%lu"
#endif
		,__entry->cpu
		,__entry->capacity
#if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
		,__entry->capacity_orig
		,__entry->capacity_curr
#endif
	)
);
#endif


#define PIXEL6_EMETER_CHAN_NAME_MAX_SIZE 64

TRACE_EVENT(lisa__pixel6_emeter,
	TP_PROTO(unsigned long ts, unsigned int device, unsigned int chan, char *chan_name, unsigned long value),
	TP_ARGS(ts, device, chan, chan_name, value),

	TP_STRUCT__entry(
		__field(unsigned long,		ts			)
		__field(unsigned long,		value			)
		__field(unsigned int,		device			)
		__field(unsigned int,		chan			)
		__array(char,			chan_name,	PIXEL6_EMETER_CHAN_NAME_MAX_SIZE	)
	),

	TP_fast_assign(
		__entry->ts		    = ts;
		__entry->device		= device;
		__entry->chan		= chan;
		__entry->value		= value;
		strlcpy(__entry->chan_name, chan_name, PIXEL6_EMETER_CHAN_NAME_MAX_SIZE);
	),

	TP_printk("ts=%lu device=%u chan=%u chan_name=%s value=%lu",
		  __entry->ts, __entry->device, __entry->chan, __entry->chan_name, __entry->value)
);

#endif /* _FTRACE_EVENTS_H */

/* This part must be outside protection */
#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH .
#define TRACE_INCLUDE_FILE ftrace_events
#include <trace/define_trace.h>
