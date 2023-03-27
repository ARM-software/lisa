/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _TP_H
#define _TP_H

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

#define DEFINE_TP_ENABLE_DISABLE(feature_name, tp_name, probe, enable_name, enable_f, disable_name, disable_f) \
	static bool __feature_tp_registered_##feature_name = false;	\
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
			tp = __find_tracepoint(#tp_name);		\
			if (tp) {					\
				if (_enable_f) {			\
					__ret = _enable_f(feature);	\
					ret |= __ret;			\
					if (__ret)			\
						pr_err(#feature_name ": init function " #enable_f "() failed with error: %i\n", __ret); \
				}					\
				if (!ret) {				\
					__ret = tracepoint_probe_register(tp, (void *)probe, feature); \
					ret |= __ret;			\
					if (__ret)			\
						pr_err(#feature_name ": could not attach " #probe "() to tracepoint " #tp_name "\n"); \
				}					\
				__feature_tp_registered_##feature_name = !ret; \
				return ret;				\
			} else {					\
				pr_err(#feature_name ": could not attach " #probe "() to undefined tracepoint " #tp_name "\n"); \
				ret |= 1;				\
			}						\
		}							\
		return ret;						\
	}								\
	static int disable_name(struct feature* feature) {		\
		int ret = 0;						\
		int __ret;						\
		int (*_disable_f)(struct feature*) = disable_f;		\
		struct tracepoint *tp = __find_tracepoint(#tp_name);	\
		if (tp) {						\
			if(__feature_tp_registered_##feature_name)	{ \
				__ret = tracepoint_probe_unregister(tp, (void *)probe, feature); \
				ret |= __ret;				\
				if (__ret)				\
					pr_err(#feature_name ": failed to unregister function " #probe "() on tracepoint " #tp_name "\n"); \
			}						\
			if (_disable_f)					\
				ret |= _disable_f(feature);		\
		}							\
		ret |= DISABLE_FEATURE(__tp);				\
		return ret;						\
	}								\

/**
 * DEFINE_EXTENDED_TP_FEATURE() - Define a feature linked to a tracepoint.
 * @feature_name: Name of the feature.
 * @tp_name: Name of the tracepoint to attach to.
 * @probe: Probe function passed to the relevant tracepoint registering function register_trace_*().
 * @enable_f: Additional enable function for the feature. It must take a struct
 * feature * and return a non-zero int in case of failure.
 * @disable_f: Additional disable function for the feature. Same signature as enable_f().
 *
 * Define a feature with a probe attached to a tracepoint, with additional
 * user-defined enable/disable functions. If the tracepoint is not found, the
 * user functions will not be called.
 */
#define DEFINE_EXTENDED_TP_FEATURE(feature_name, tp_name, probe, enable_f, disable_f) \
	DEFINE_TP_ENABLE_DISABLE(feature_name, tp_name, probe, __tp_feature_enable_##feature_name, enable_f, __tp_feature_disable_##feature_name, disable_f); \
	DEFINE_FEATURE(feature_name, __tp_feature_enable_##feature_name, __tp_feature_disable_##feature_name);

/**
 * DEFINE_TP_FEATURE() - Same as DEFINE_EXTENDED_TP_FEATURE() without custom
 * enable/disable functions.
 */
#define DEFINE_TP_FEATURE(feature_name, tp_name, probe) DEFINE_EXTENDED_TP_FEATURE(feature_name, tp_name, probe, NULL, NULL)

#define __EVENT_FEATURE(event_name) event__##event_name
/**
 * DEFINE_TP_EVENT_FEATURE() - Same as DEFINE_TP_FEATURE() with automatic
 * "event__" prefixing of the feature name.
 */
#define DEFINE_TP_EVENT_FEATURE(event_name, tp_name, probe) DEFINE_TP_FEATURE(__EVENT_FEATURE(event_name), tp_name, probe)

/**
 * __DEFINE_EXTENDED_TP_EVENT_FEATURE - Wrapper for
 * DEFINE_EXTENDED_TP_EVENT_FEATURE to allow safe macro-expansion for
 * __EVENT_FEATURE
 */
#define __DEFINE_EXTENDED_TP_EVENT_FEATURE(feature_name, ...) \
	DEFINE_EXTENDED_TP_FEATURE(feature_name, ##__VA_ARGS__)
/**
 * DEFINE_EXTENDED_TP_EVENT_FEATURE() - Same as DEFINE_EXTENDED_TP_FEATURE()
 * with automatic "event__" prefixing of the feature name.
 */
#define DEFINE_EXTENDED_TP_EVENT_FEATURE(event_name, tp_name, probe, enable_f, disable_f) \
	__DEFINE_EXTENDED_TP_EVENT_FEATURE(__EVENT_FEATURE(event_name), tp_name, probe, enable_f, disable_f)
#endif
