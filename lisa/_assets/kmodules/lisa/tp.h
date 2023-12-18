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

#define DEFINE_TP_ENABLE_DISABLE(feature_name, tp_probes, enable_name, enable_f, disable_name, disable_f) \
	static u64 CONCATENATE(__feature_tp_registered_, feature_name) = 0; \
	static int enable_name(struct feature* feature) {		\
		int ret = 0;						\
		int __ret;						\
		struct tracepoint *tp;					\
		int (*_enable_f)(struct feature*) = enable_f;		\
		__ret = ENABLE_FEATURE(__tp);				\
		ret |= __ret;						\
		if (ret) {						\
			pr_err(#feature_name ": could not enable tracepoint support: %i\n", __ret); \
		} else { 						\
			if (_enable_f) {				\
				__ret = _enable_f(feature);		\
				ret |= __ret;				\
				if (__ret)				\
					pr_err(#feature_name ": init function " #enable_f "() failed with error: %i\n", __ret); \
			}						\
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
		int (*_disable_f)(struct feature*) = disable_f;		\
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
		if (_disable_f)						\
			ret |= _disable_f(feature);			\
		ret |= DISABLE_FEATURE(__tp);				\
		return ret;						\
	}								\

/**
 * DEFINE_EXTENDED_TP_FEATURE() - Define a feature linked to a tracepoint.
 * @feature_name: Name of the feature.
 * @probes: List of tracepoint probes built using TP_PROBES(TP_PROBE("my_tp", my_probe), ...)
 * @enable_f: Additional enable function for the feature. It must take a struct
 * feature * and return a non-zero int in case of failure.
 * @disable_f: Additional disable function for the feature. Same signature as enable_f().
 *
 * Define a feature with a probe attached to a tracepoint, with additional
 * user-defined enable/disable functions. If the tracepoint is not found, the
 * user functions will not be called.
 */
#define DEFINE_EXTENDED_TP_FEATURE(feature_name, probes, enable_f, disable_f, ...) \
	DEFINE_TP_ENABLE_DISABLE(feature_name, probes, CONCATENATE(__tp_feature_enable_, feature_name), enable_f, CONCATENATE(__tp_feature_disable_, feature_name), disable_f); \
	DEFINE_FEATURE(feature_name, CONCATENATE(__tp_feature_enable_, feature_name), CONCATENATE(__tp_feature_disable_, feature_name), ##__VA_ARGS__);

/**
 * DEFINE_TP_FEATURE() - Same as DEFINE_EXTENDED_TP_FEATURE() without custom
 * enable/disable functions.
 */
#define DEFINE_TP_FEATURE(feature_name, probes) DEFINE_EXTENDED_TP_FEATURE(feature_name, probes, NULL, NULL)

#define __EVENT_FEATURE(event_name) CONCATENATE(event__, event_name)

/**
 * DEFINE_TP_EVENT_FEATURE() - Same as DEFINE_TP_FEATURE() with automatic
 * "event__" prefixing of the feature name.
 */
#define DEFINE_TP_EVENT_FEATURE(event_name, probes) DEFINE_TP_FEATURE(__EVENT_FEATURE(event_name), probes)

/**
 * DEFINE_EXTENDED_TP_EVENT_FEATURE() - Same as DEFINE_EXTENDED_TP_FEATURE()
 * with automatic "event__" prefixing of the feature name.
 */
#define DEFINE_EXTENDED_TP_EVENT_FEATURE(event_name, probes, enable_f, disable_f, ...)	\
	DEFINE_EXTENDED_TP_FEATURE(__EVENT_FEATURE(event_name), probes, enable_f, disable_f, ##__VA_ARGS__)

#define __DEPRECATED_EVENT_ENABLE(event_name) CONCATENATE(__enable_deprecated_feature_, __EVENT_FEATURE(event_name))
/**
 * DEFINE_TP_DEPRECATED_EVENT_FEATURE() - Same as DEFINE_TP_EVENT_FEATURE()
 * with extra deprecation warnings upon init.
 */
#define DEFINE_TP_DEPRECATED_EVENT_FEATURE(msg, event_name, probes)	\
static int __DEPRECATED_EVENT_ENABLE(event_name)(struct feature *feature)	\
{										\
	pr_warn("The feature %s is deprecated: " msg, feature->name);		\
	return 0;								\
}										\
DEFINE_EXTENDED_TP_EVENT_FEATURE(event_name, probes, __DEPRECATED_EVENT_ENABLE(event_name), NULL)
#endif
