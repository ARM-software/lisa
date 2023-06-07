/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/mutex.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/module.h>

#include <linux/tracepoint.h>

#ifndef _FEATURE_H
#define _FEATURE_H

/**
 * struct feature - LISA kernel module feature
 *
 * @name: name of the feature. Must be unique.
 * @data: private void * that will be available to the feature's implementation.
 * @enabled: Reference counter for the feature.
 * @lock: Lock taken when enabling and disabling the feature.
 * @enable: Function pointer to the enable function. Return non-zero value in case of error.
 * @disable: Function pointer to the disable function. Return non-zero value in case of error.
 *
 * A struct feature represent an independent feature of the kernel module that
 * can be enabled and disabled dynamically. Features are ref-counted so that
 * they can depend on each other easily.
 */
struct feature {
	char *name;
	void *data;
	u64 enabled;
	struct mutex *lock;
	int (*enable)(struct feature*);
	int (*disable)(struct feature*);

	/* Return code of the enable() function */
	int __enable_ret;

	/* true if the feature has been explicitly enabled by the user */
	bool __explicitly_enabled;
	/* true if the feature is internal, i.e. not exposed to the user.
	 * Internal features are used to share some code between feature, taking
	 * advantage of reference counting to ensure safe setup/teardown.
	 */
	bool __internal;
};

/* Start and stop address of the ELF section containing the struct feature
 *instances
 */
extern struct feature __lisa_features_start[];
extern struct feature __lisa_features_stop[];

/**
 * MAX_FEATURES - Maximum number of features allowed in this module.
 */
#define MAX_FEATURES 1024

int __placeholder_init(struct feature *feature);
int __placeholder_deinit(struct feature *feature);

#define __FEATURE_NAME(name) __lisa_feature_##name

/* Weak definition, can be useful to deal with compiled-out features */
#define __DEFINE_FEATURE_WEAK(feature_name)				\
	__attribute__((weak)) DEFINE_MUTEX(__lisa_mutex_feature_##feature_name); \
	__attribute__((weak)) struct feature __FEATURE_NAME(feature_name) = { \
		.name = #feature_name,					\
		.data = NULL,						\
		.enabled = 0,						\
		.__explicitly_enabled = false,				\
		.enable = __placeholder_init,				\
		.disable = __placeholder_deinit,			\
		.lock = &__lisa_mutex_feature_##feature_name,		\
		.__internal = true,					\
		.__enable_ret = 0,					\
	};

#define __DEFINE_FEATURE_STRONG(feature_name, enable_f, disable_f, internal)	\
	DEFINE_MUTEX(__lisa_mutex_feature_##feature_name);		\
	struct feature __FEATURE_NAME(feature_name) __attribute__((unused,section(".__lisa_features"))) = { \
		.name = #feature_name,					\
		.data = NULL,						\
		.enabled = 0,						\
		.__explicitly_enabled = false,				\
		.enable = enable_f,					\
		.disable = disable_f,					\
		.lock = &__lisa_mutex_feature_##feature_name,		\
		.__internal = internal,					\
		.__enable_ret = 0,					\
	};

/**
 * DEFINE_FEATURE() - Define a feature
 * @feature_name: Name of the feature, as a C identifier.
 * @enable_f: Function to enable the feature. Takes a struct feature * as
 * parameter and must return a non-zero int in case of failure.
 * @disable_f: Function to disable the feature. Takes a struct feature * as
 * parameter and must return a non-zero int in case of failure. Note that this
 * function must be able to deal with a partially enabled feature. It must call
 * DISABLE_FEATURE() on all the features that were enabled by ENABLE_FEATURE()
 * in enable_f() in order to keep accurate reference-counting.
 */
#define DEFINE_FEATURE(feature_name, enable_f, disable_f) __DEFINE_FEATURE_STRONG(feature_name, enable_f, disable_f, false)

/**
 * DEFINE_INTERNAL_FEATURE() - Same as DEFINE_FEATURE() but for internal features.
 *
 * Internal features are identical to normal features, except they will not be
 * displayed in user-visible listings. They can be used to share code between
 * multiple other features, e.g. to initialize and teardown the use of a kernel
 * API (workqueues, tracepoints etc).
 */
#define DEFINE_INTERNAL_FEATURE(feature_name, enable_f, disable_f) __DEFINE_FEATURE_STRONG(feature_name, enable_f, disable_f, true)

/**
 * DECLARE_FEATURE() - Declare a feature to test for its presence dynamically.
 * @feature_name: Name of the feature to declare.
 *
 * Very similar to DEFINE_FEATURE() but for user code that wants to deal with
 * features that might be entirely compiled-out. If the feature is compiled-in,
 * DECLARE_FEATURE() will essentially be a no-op. If the feature is
 * compiled-out, DECLARE_FEATURE() will still allow making use of it so that its
 * initialization fails with an error message, and its presence can be tested
 * with FEATURE_IS_AVAILABLE().
 *
 * Note that because of weak symbols limitations, a given compilation unit
 * cannot contain both DECLARE_FEATURE() and DEFINE_FEATURE().
 */
#define DECLARE_FEATURE(feature_name) __DEFINE_FEATURE_WEAK(feature_name)

/**
 * FEATURE() - Pointer the the struct feature
 * @name: name of the feature as a C identifier.
 *
 * Evaluates to a struct feature * of the given feature.
 */
#define FEATURE(name) ({				\
	extern struct feature __FEATURE_NAME(name);	\
	&__FEATURE_NAME(name);				\
})

/**
 * FEATURE_IS_AVAILABLE() - Runtime check if a feature is available.
 * @name: name of the feature
 *
 * Useful in conjunction with DECLARE_FEATURE() to test if a feature is available.
 * Note that it's not necessary to use it before calling
 * ENABLE_FEATURE()/DISABLE_FEATURE() in simple cases as they will fail with an
 * appropriate error message if the feature is missing.
 */
#define FEATURE_IS_AVAILABLE(name) (FEATURE(name)->enable != &__placeholder_init)

/**
 * ENABLE_FEATURE() - Enable a feature
 * @feature_name: Name of the feature, as a C identifier.
 *
 * Enable a given feature. Since features are reference-counted, each
 * ENABLE_FEATURE() **must** be paired with a DISABLE_FEATURE() call, even if
 * ENABLE_FEATURE() failed.
 */
#define ENABLE_FEATURE(feature_name) ({					\
		int __enable_feature(struct feature* feature);		\
		__enable_feature(FEATURE(feature_name));		\
	})
/**
 * DISABLE_FEATURE() - Disable a feature
 * @feature_name: Name of the feature.
 *
 * Symmetrical to ENABLE_FEATURE(). DISABLE_FEATURE() must be called for each
 * ENABLE_FEATURE() to maintain a correct feature reference count.
 */
#define DISABLE_FEATURE(feature_name) ({				\
		int __disable_feature(struct feature* feature);		\
		__disable_feature(FEATURE(feature_name));		\
	})

/**
 * init_features() - Initialize features
 * @selected: Array of char * containing feature names to initialize.
 * @selected_len: Length of @selected.
 *
 * Initialize features listed by name in the provided array. The list of actual
 * struct features * is built automatically by DEFINE_FEATURE() and does not
 * need to be passed.
 */
int init_features(char **selected, size_t selected_len);

/**
 * deinit_features() - De-initialize features
 *
 * De-initialize features initialized with init_features().
 * Return: non-zero in case of errors.
 */
int deinit_features(void);
#endif
