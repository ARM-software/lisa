/* SPDX-License-Identifier: GPL-2.0 */
#ifndef SCHED_HELPERS_H
#define SCHED_HELPERS_H

#include <linux/kconfig.h>

/* Required for some structs */
#include <linux/irq_work.h>
#include <linux/cgroup.h>

#include "introspection.h"
#include "generated/introspection_data.h"


#if HAS_TYPE(struct, cfs_rq)
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, cfs_rq, rq)
static inline struct rq *rq_of(struct cfs_rq *cfs_rq)
{
	return cfs_rq->rq;
}
#    else
static inline struct rq *rq_of(struct cfs_rq *cfs_rq)
{
	return container_of(cfs_rq, struct rq, cfs);
}
#    endif
#endif


#if HAS_MEMBER(struct, sched_entity, my_q)
#    define entity_is_task(se)	(!(se)->my_q)
#else
#    define entity_is_task(se)	(1)
#endif


#if HAS_TYPE(struct, rq)
static inline int cpu_of(struct rq *rq)
{
#    if defined(CONFIG_SMP) && HAS_MEMBER(struct, rq, cpu)
	return rq->cpu;
#    else
	return 0;
#    endif
}
#endif

#define cap_scale(v, s) ((v)*(s) >> SCHED_CAPACITY_SHIFT)


#if HAS_TYPE(struct, task_group)
static inline bool task_group_is_autogroup(struct task_group *tg)
{
#    if HAS_KERNEL_FEATURE(SCHED_AUTOGROUP)
	return !!tg->autogroup;
#    else
	return false;
#    endif
}
#endif

#if HAS_TYPE(struct, task_group)
static int autogroup_path(struct task_group *tg, char *buf, int buflen)
{
#    if HAS_KERNEL_FEATURE(SCHED_AUTOGROUP) && HAS_MEMBER(struct, autogroup, id)
	if (!task_group_is_autogroup(tg))
		return 0;

	return snprintf(buf, buflen, "%s-%ld", "/autogroup", tg->autogroup->id);
#    else
	return 0;
#    endif
}
#endif


#if HAS_TYPE(struct, rq)
/* A cut down version of the original. @p MUST be NULL */
static __always_inline
unsigned long uclamp_rq_util_with(struct rq *rq, unsigned long util)
{
#    if HAS_KERNEL_FEATURE(SE_UCLAMP)
	unsigned long min_util;
	unsigned long max_util;

	min_util = READ_ONCE(rq->uclamp[UCLAMP_MIN].value);
	max_util = READ_ONCE(rq->uclamp[UCLAMP_MAX].value);

	if (unlikely(min_util >= max_util))
		return min_util;

	return clamp(util, min_util, max_util);
#    else
	return util;
#    endif
}
#endif


#if HAS_TYPE(struct, cfs_rq)
static inline void cfs_rq_tg_path(struct cfs_rq *cfs_rq, char *path, int len)
{
	if (!path)
		return;

#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, cfs_rq, tg) && HAS_MEMBER(struct, task_group, css) && HAS_MEMBER(struct, cgroup_subsys_state, cgroup)
	if (cfs_rq && task_group_is_autogroup(cfs_rq->tg))
		autogroup_path(cfs_rq->tg, path, len);
	else if (cfs_rq && cfs_rq->tg->css.cgroup)
		cgroup_path((struct cgroup *)cfs_rq->tg->css.cgroup, path, len);
	else
#    endif
		strlcpy(path, "(null)", len);
}
#endif

#if HAS_TYPE(struct, sched_entity)
static inline struct cfs_rq *get_group_cfs_rq(struct sched_entity *se)
{
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, sched_entity, my_q)
	return se->my_q;
#    else
	return NULL;
#    endif
}

static inline struct cfs_rq *get_se_cfs_rq(struct sched_entity *se)
{
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, sched_entity, cfs_rq)
	return se->cfs_rq;
#    else
	return NULL;
#    endif
}
#endif


#if HAS_TYPE(struct, cfs_rq)
static inline const struct sched_avg *lisa_cfs_rq_avg(struct cfs_rq *cfs_rq)
{
#    if HAS_KERNEL_FEATURE(CFS_PELT)
	return cfs_rq ? (struct sched_avg *)&cfs_rq->avg : NULL;
#    else
	return NULL;
#    endif
}

static inline char *lisa_cfs_rq_path(struct cfs_rq *cfs_rq, char *str, int len)
{
	if (!cfs_rq) {
		if (str)
			strlcpy(str, "(null)", len);
		else
			return NULL;
	}

	cfs_rq_tg_path(cfs_rq, str, len);
	return str;
}

static inline int lisa_cfs_rq_cpu(struct cfs_rq *cfs_rq)
{
	return cfs_rq ? cpu_of(rq_of(cfs_rq)) : -1;
}

#endif

#if HAS_TYPE(struct, rq)
static inline const struct sched_avg *lisa_rq_avg_rt(struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(RT_PELT)
	return rq ? (struct sched_avg *)&rq->avg_rt : NULL;
#    else
	return NULL;
#    endif
}

static inline const struct sched_avg *lisa_rq_avg_dl(struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(DL_PELT)
	return rq ? (struct sched_avg *)&rq->avg_dl : NULL;
#    else
	return NULL;
#    endif
}

static inline const struct sched_avg *lisa_rq_avg_irq(struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(IRQ_PELT)
	return rq ? (struct sched_avg *)&rq->avg_irq : NULL;
#    else
	return NULL;
#    endif
}

static inline int lisa_rq_cpu(struct rq *rq)
{
	return rq ? cpu_of(rq) : -1;
}

static inline int lisa_rq_cpu_capacity(struct rq *rq)
{
	return rq ?
#if    HAS_KERNEL_FEATURE(RQ_CAPACITY)
		rq->cpu_capacity
#    else
		SCHED_CAPACITY_SCALE
#    endif
		: -1;
}

static inline int lisa_rq_nr_running(struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(RQ_NR_RUNNING)
	if (rq->nr_running)
		return rq->nr_running;
#    endif
	return -1;
}
#endif

#if HAS_TYPE(struct, root_domain)
static inline const struct cpumask *lisa_rd_span(struct root_domain *rd)
{
#    if defined(CONFIG_SMP) && HAS_MEMBER(struct, root_domain, span)
	return rd ? (struct cpumask *)rd->span : NULL;
#    else
	return NULL;
#    endif
}
#endif

#endif /* SCHED_HELPERS_H */
