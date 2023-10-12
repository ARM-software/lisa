/* SPDX-License-Identifier: GPL-2.0 */
#ifndef SCHED_HELPERS_H
#define SCHED_HELPERS_H

#include <linux/kconfig.h>

/* Required for some structs */
#include <linux/irq_work.h>
#include <linux/cgroup.h>

#include "introspection.h"


#if HAS_TYPE(struct, cfs_rq)
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, cfs_rq, rq)
static inline const struct rq *rq_of(const struct cfs_rq *cfs_rq)
{
	return cfs_rq->rq;
}
#    elif HAS_MEMBER(struct, rq, cfs)
static inline struct rq *rq_of(struct cfs_rq *cfs_rq)
{
	return container_of(cfs_rq, struct rq, cfs);
}
#    else
#        warning "Cannot get the parent struct rq of a struct cfs_rq"
#    endif
#endif


static inline bool entity_is_task(const struct sched_entity *se)
{
	return
#if HAS_MEMBER(struct, sched_entity, my_q)
		!se->my_q
#else
		true
#endif
	;
}


#if HAS_TYPE(struct, rq)
static inline int cpu_of(const struct rq *rq)
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
static inline bool task_group_is_autogroup(const struct task_group *tg)
{
#    if HAS_KERNEL_FEATURE(SCHED_AUTOGROUP)
	return !!tg->autogroup;
#    else
	return false;
#    endif
}
#endif

#if HAS_TYPE(struct, task_group)
static int autogroup_path(const struct task_group *tg, char *buf, int buflen)
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
static inline unsigned long uclamp_rq_util_with(const struct rq *rq, unsigned long util)
{
#    if HAS_KERNEL_FEATURE(RQ_UCLAMP)
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
static inline void cfs_rq_tg_path(const struct cfs_rq *cfs_rq, char *path, int len)
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
static inline const struct cfs_rq *get_group_cfs_rq(const struct sched_entity *se)
{
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, sched_entity, my_q)
	return se->my_q;
#    else
	return NULL;
#    endif
}

static inline const struct cfs_rq *get_se_cfs_rq(const struct sched_entity *se)
{
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, sched_entity, cfs_rq)
	return se->cfs_rq;
#    else
	return NULL;
#    endif
}
#endif


#if HAS_TYPE(struct, cfs_rq)
static inline int cfs_rq_cpu(const struct cfs_rq *cfs_rq)
{
	return cpu_of(rq_of(cfs_rq));
}

static inline const struct sched_avg *cfs_rq_avg(const struct cfs_rq *cfs_rq)
{
#    if HAS_KERNEL_FEATURE(CFS_PELT)
	return cfs_rq ? (struct sched_avg *)&cfs_rq->avg : NULL;
#    else
	return NULL;
#    endif
}


#    if HAS_MEMBER(struct, cfs_rq, rq) && HAS_MEMBER(struct, rq, cfs)
static inline bool cfs_rq_is_root(const struct cfs_rq *cfs_rq)
{
	return (&cfs_rq->rq->cfs == cfs_rq);
}
#    endif

static inline const struct sched_avg *cfs_tg_avg(const struct cfs_rq *cfs_rq)
{
#    if defined(CONFIG_FAIR_GROUP_SCHED) && HAS_MEMBER(struct, cfs_rq, tg) && HAS_MEMBER(struct, task_group, cfs_rq) && HAS_MEMBER(struct, task_group, se)
	if (!cfs_rq || cfs_rq_is_root(cfs_rq)) {
		// For the root cfs_rq, we simply do not have any associated
		// taskgroup
		return NULL;
	} else {
		const struct task_group *tg = cfs_rq->tg;
		if (tg) {
			int cpu = cfs_rq_cpu(cfs_rq);

			// Check the cfs_rq for that CPU on the task_group is
			// the one we expect
			BUG_ON(tg->cfs_rq[cpu] != cfs_rq);
			return (const struct sched_avg *)&tg->se[cpu]->avg;
		}
	}
#    endif
	return NULL;
}

static inline char *cfs_rq_path(const struct cfs_rq *cfs_rq, char *str, int len)
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

#endif

#if HAS_TYPE(struct, rq)
static inline const struct sched_avg *rq_avg_rt(struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(RT_PELT)
	return rq ? (struct sched_avg *)&rq->avg_rt : NULL;
#    else
	return NULL;
#    endif
}

static inline const struct sched_avg *rq_avg_dl(const struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(DL_PELT)
	return rq ? (struct sched_avg *)&rq->avg_dl : NULL;
#    else
	return NULL;
#    endif
}

static inline const struct sched_avg *rq_avg_irq(const struct rq *rq)
{
#    if HAS_KERNEL_FEATURE(IRQ_PELT)
	return rq ? (struct sched_avg *)&rq->avg_irq : NULL;
#    else
	return NULL;
#    endif
}

static inline int rq_cpu(const struct rq *rq)
{
	return cpu_of(rq);
}


static inline int se_cpu(const struct sched_entity *se)
{
	const struct cfs_rq *cfs_rq = get_se_cfs_rq(se);
	int cpu = cfs_rq ? cfs_rq_cpu(cfs_rq) : -1;
	return cpu;
}

static inline int rq_cpu_capacity(const struct rq *rq)
{
	return
#    if HAS_KERNEL_FEATURE(RQ_CAPACITY)
		rq->cpu_capacity
#    else
		SCHED_CAPACITY_SCALE
#    endif
	;
}

static inline int rq_cpu_orig_capacity(const struct rq *rq)
{
	return
#    if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
		rq->cpu_capacity_orig;
#    else
		rq_cpu_capacity(rq)
#    endif
	;
}

#    if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
DECLARE_PER_CPU(unsigned long, arch_freq_scale);
#    endif

static inline int rq_cpu_current_capacity(const struct rq *rq)
{
	return
#    if HAS_KERNEL_FEATURE(FREQ_INVARIANCE)
		({
		    unsigned long capacity_orig = rq_cpu_orig_capacity(rq);
		    unsigned long scale_freq = per_cpu(arch_freq_scale, rq->cpu);
		    cap_scale(capacity_orig, scale_freq);
		})
#    else
		rq_cpu_orig_capacity(rq)
#    endif
    ;
}

#    if HAS_KERNEL_FEATURE(RQ_NR_RUNNING)
static inline int rq_nr_running(const struct rq *rq)
{
	return rq->nr_running;
}
#    endif

#endif

#if HAS_TYPE(struct, root_domain)
static inline const struct cpumask *rd_span(const struct root_domain *rd)
{
#    if defined(CONFIG_SMP) && HAS_MEMBER(struct, root_domain, span)
	return rd ? (struct cpumask *)rd->span : NULL;
#    else
	return NULL;
#    endif
}
#endif

#endif /* SCHED_HELPERS_H */
