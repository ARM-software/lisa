/* SPDX-License-Identifier: GPL-2.0 */
#undef TRACE_SYSTEM
#define TRACE_SYSTEM sched

#define MAX_SPAN_SIZE		128

#if !defined(_SCHED_EVENTS_H) || defined(TRACE_HEADER_MULTI_READ)
#define _SCHED_EVENTS_H

#define PATH_SIZE		64
#define __SPAN_SIZE		(round_up(NR_CPUS, 4)/4)
#define SPAN_SIZE		(__SPAN_SIZE > MAX_SPAN_SIZE ? MAX_SPAN_SIZE : __SPAN_SIZE)

#include <linux/version.h>
#include <linux/tracepoint.h>
#include <linux/version.h>

#include "sched_tp_helpers.h"

#if LINUX_VERSION_CODE <= KERNEL_VERSION(5,6,0)
#define RBL_LOAD_ENTRY		rbl_load
#define RBL_LOAD_MEMBER		runnable_load_avg
#define RBL_LOAD_STR		"rbl_load"
#else
#define RBL_LOAD_ENTRY		runnable
#define RBL_LOAD_MEMBER		runnable_avg
#define RBL_LOAD_STR		"runnable"
#endif

TRACE_EVENT(sched_pelt_cfs,

	TP_PROTO(int cpu, char *path, const struct sched_avg *avg),

	TP_ARGS(cpu, path, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__array(	char,		path,	PATH_SIZE	)
		__field(	unsigned long,	load			)
		__field(	unsigned long,	RBL_LOAD_ENTRY		)
		__field(	unsigned long,	util			)
		__field(	unsigned long long, update_time	        )
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		strlcpy(__entry->path, path, PATH_SIZE);
		__entry->load		= avg->load_avg;
		__entry->RBL_LOAD_ENTRY	= avg->RBL_LOAD_MEMBER;
		__entry->util		= avg->util_avg;
		__entry->update_time    = avg->last_update_time;
	),

	TP_printk("cpu=%d path=%s load=%lu " RBL_LOAD_STR "=%lu util=%lu update_time=%llu",
		  __entry->cpu, __entry->path, __entry->load,
		  __entry->RBL_LOAD_ENTRY,__entry->util, __entry->update_time)
);

DECLARE_EVENT_CLASS(sched_pelt_rq_template,

	TP_PROTO(int cpu, const struct sched_avg *avg),

	TP_ARGS(cpu, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__field(	unsigned long,	load			)
		__field(	unsigned long,	RBL_LOAD_ENTRY		)
		__field(	unsigned long,	util			)
		__field(	unsigned long long, update_time	        )
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		__entry->load		= avg->load_avg;
		__entry->RBL_LOAD_ENTRY	= avg->RBL_LOAD_MEMBER;
		__entry->util		= avg->util_avg;
		__entry->update_time    = avg->last_update_time;
	),

	TP_printk("cpu=%d load=%lu " RBL_LOAD_STR "=%lu util=%lu update_time=%llu",
		  __entry->cpu, __entry->load,
		  __entry->RBL_LOAD_ENTRY,__entry->util, __entry->update_time)
);

DEFINE_EVENT(sched_pelt_rq_template, sched_pelt_rt,
	TP_PROTO(int cpu, const struct sched_avg *avg),
	TP_ARGS(cpu, avg));

DEFINE_EVENT(sched_pelt_rq_template, sched_pelt_dl,
	TP_PROTO(int cpu, const struct sched_avg *avg),
	TP_ARGS(cpu, avg));

DEFINE_EVENT(sched_pelt_rq_template, sched_pelt_irq,
	TP_PROTO(int cpu, const struct sched_avg *avg),
	TP_ARGS(cpu, avg));

TRACE_EVENT(sched_pelt_se,

	TP_PROTO(int cpu, char *path, char *comm, int pid, const struct sched_avg *avg),

	TP_ARGS(cpu, path, comm, pid, avg),

	TP_STRUCT__entry(
		__field(	int,		cpu			)
		__array(	char,		path,	PATH_SIZE	)
		__array(	char,		comm,	TASK_COMM_LEN	)
		__field(	int,		pid			)
		__field(	unsigned long,	load			)
		__field(	unsigned long,	RBL_LOAD_ENTRY		)
		__field(	unsigned long,	util			)
		__field(	unsigned long long, update_time	        )
	),

	TP_fast_assign(
		__entry->cpu		= cpu;
		strlcpy(__entry->path, path, PATH_SIZE);
		strlcpy(__entry->comm, comm, TASK_COMM_LEN);
		__entry->pid		= pid;
		__entry->load		= avg->load_avg;
		__entry->RBL_LOAD_ENTRY	= avg->RBL_LOAD_MEMBER;
		__entry->util		= avg->util_avg;
		__entry->update_time    = avg->last_update_time;
	),

	TP_printk("cpu=%d path=%s comm=%s pid=%d load=%lu " RBL_LOAD_STR "=%lu util=%lu update_time=%llu",
		  __entry->cpu, __entry->path, __entry->comm, __entry->pid,
		  __entry->load, __entry->RBL_LOAD_ENTRY,__entry->util, __entry->update_time)
);

TRACE_EVENT(sched_overutilized,

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

TRACE_EVENT(sched_update_nr_running,

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

TRACE_EVENT(sched_util_est_se,

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
		__entry->enqueued	= avg->util_est.enqueued;
		__entry->ewma		= avg->util_est.ewma;
		__entry->util		= avg->util_avg;
	),

	TP_printk("cpu=%d path=%s comm=%s pid=%d enqueued=%u ewma=%u util=%lu",
		  __entry->cpu, __entry->path, __entry->comm, __entry->pid,
		  __entry->enqueued, __entry->ewma, __entry->util)
);

TRACE_EVENT(sched_util_est_cfs,

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

#ifdef CONFIG_UCLAMP_TASK

TRACE_EVENT_CONDITION(uclamp_util_se,

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
		__entry->cpu            = sched_tp_rq_cpu(rq);
		__entry->util_avg       = p->se.avg.util_avg;
		__entry->uclamp_avg     = uclamp_rq_util_with(rq, p->se.avg.util_avg, NULL);
		__entry->uclamp_min     = rq->uclamp[UCLAMP_MIN].value;
		__entry->uclamp_max     = rq->uclamp[UCLAMP_MAX].value;
		),

	TP_printk("pid=%d comm=%s cpu=%d util_avg=%lu uclamp_avg=%lu "
		  "uclamp_min=%lu uclamp_max=%lu",
		  __entry->pid, __entry->comm, __entry->cpu,
		  __entry->util_avg, __entry->uclamp_avg,
		  __entry->uclamp_min, __entry->uclamp_max)
);

TRACE_EVENT_CONDITION(uclamp_util_cfs,

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
		__entry->cpu            = sched_tp_rq_cpu(rq);
		__entry->util_avg       = cfs_rq->avg.util_avg;
		__entry->uclamp_avg     = uclamp_rq_util_with(rq, cfs_rq->avg.util_avg, NULL);
		__entry->uclamp_min     = rq->uclamp[UCLAMP_MIN].value;
		__entry->uclamp_max     = rq->uclamp[UCLAMP_MAX].value;
		),

	TP_printk("cpu=%d util_avg=%lu uclamp_avg=%lu "
		  "uclamp_min=%lu uclamp_max=%lu",
		  __entry->cpu, __entry->util_avg, __entry->uclamp_avg,
		  __entry->uclamp_min, __entry->uclamp_max)
);
#else
#define trace_uclamp_util_se(is_task, p, rq) while(false) {}
#define trace_uclamp_util_se_enabled() false
#define trace_uclamp_util_cfs(is_root, cpu, cfs_rq) while(false) {}
#define trace_uclamp_util_cfs_enabled() false
#endif /* CONFIG_UCLAMP_TASK */

#endif /* _SCHED_EVENTS_H */

/* This part must be outside protection */
#undef TRACE_INCLUDE_PATH
#define TRACE_INCLUDE_PATH .
#define TRACE_INCLUDE_FILE sched_events
#include <trace/define_trace.h>
