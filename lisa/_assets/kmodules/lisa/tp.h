/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _TP_H
#define _TP_H

#include "utils.h"
#include "features.h"

struct __find_tracepoint_params {
	const char *name;
	struct tracepoint *found;
};

static void __do_find_tracepoint(struct tracepoint *tp, void *__finder) {
	struct __find_tracepoint_params *finder = __finder;
	if (!strcmp(tp->name, finder->name))
		finder->found = tp;
}

__attribute__((unused)) static struct tracepoint *__find_tracepoint(const char *name) {
	struct __find_tracepoint_params res = {.name = name, .found = NULL};
	for_each_kernel_tracepoint(__do_find_tracepoint, &res);
	return res.found;
}

struct __tp_probe {
	void *probe;
	const char *tp_name;
};

#define TP_PROBE(_tp_name, _probe) (&(const struct __tp_probe){.tp_name=_tp_name, .probe=_probe})
#define TP_PROBES(...) ((const struct __tp_probe **)(const struct __tp_probe *[]){__VA_ARGS__, NULL})

#define DEFINE_TP_ENABLE_DISABLE(feature_name, tp_probes, enable_name, disable_name) \
	static u64 CONCATENATE(__feature_tp_registered_, feature_name) = 0; \
	static int enable_name(struct feature* feature) {		\
		int ret = 0;						\
		int __ret;						\
		struct tracepoint *tp;					\
		__ret = ENABLE_FEATURE(__tp);				\
		ret |= __ret;						\
		if (ret) {						\
			pr_err(#feature_name ": could not enable tracepoint support: %i\n", __ret); \
		} else { 						\
			if (!ret) {					\
				const struct __tp_probe **__tp_probes = tp_probes;	\
				for (size_t i=0; __tp_probes[i]; i++) {			\
					BUG_ON(i > (sizeof(CONCATENATE(__feature_tp_registered_, feature_name)) * 8 - 1));\
					const struct __tp_probe *probe = __tp_probes[i];\
					tp = __find_tracepoint(probe->tp_name);		\
					if (tp) {					\
						__ret = tracepoint_probe_register(tp, probe->probe, feature); \
						ret |= __ret;		\
						if (__ret)		\
							pr_err(#feature_name ": could not attach probe to tracepoint %s\n", probe->tp_name);	\
						CONCATENATE(__feature_tp_registered_, feature_name) |= (!ret) << i;			\
					} else {				\
						pr_err(#feature_name ": could not attach probe to undefined tracepoint %s\n", probe->tp_name);	\
						ret |= 1;			\
					}					\
				}            					\
			}						\
		}							\
		return ret;						\
	}								\
	static int disable_name(struct feature* feature) {		\
		int ret = 0;						\
		int __ret;						\
		const struct __tp_probe **__tp_probes = tp_probes;	\
		for (size_t i=0; __tp_probes[i]; i++) {			\
			BUG_ON(i > (sizeof(CONCATENATE(__feature_tp_registered_, feature_name)) * 8 - 1));\
			const struct __tp_probe *probe = __tp_probes[i];		\
			struct tracepoint *tp = __find_tracepoint(probe->tp_name);	\
			if (tp) {							\
				if(CONCATENATE(__feature_tp_registered_, feature_name) | (1ull << i)) {	\
					__ret = tracepoint_probe_unregister(tp, probe->probe, feature); \
					ret |= __ret;				\
					if (__ret)				\
						pr_err(#feature_name ": failed to unregister function probe on tracepoint %s\n", probe->tp_name); \
				}					\
			}						\
		}							\
		ret |= DISABLE_FEATURE(__tp);				\
		return ret;						\
	}								\

/**
 * DEFINE_TP_FEATURE() - Define a feature linked to a tracepoint.
 * @feature_name: Name of the feature.
 * @probes: List of tracepoint probes built using TP_PROBES(TP_PROBE("my_tp", my_probe), ...)
 *
 * Define a feature with a probe attached to a tracepoint. If the tracepoint is
 * not found, the user functions will not be called.
 */
#define DEFINE_TP_FEATURE(feature_name, probes) \
	DEFINE_TP_ENABLE_DISABLE(feature_name, probes, CONCATENATE(__tp_feature_enable_, feature_name), CONCATENATE(__tp_feature_disable_, feature_name)); \
	DEFINE_FEATURE(feature_name, CONCATENATE(__tp_feature_enable_, feature_name), CONCATENATE(__tp_feature_disable_, feature_name));

#define __EVENT_FEATURE(event_name) CONCATENATE(event__, event_name)

/**
 * DEFINE_TP_EVENT_FEATURE() - Same as DEFINE_TP_FEATURE() with automatic
 * "event__" prefixing of the feature name.
 */
#define DEFINE_TP_EVENT_FEATURE(event_name, probes) DEFINE_TP_FEATURE(__EVENT_FEATURE(event_name), probes)

#endif /* _TP_H */
