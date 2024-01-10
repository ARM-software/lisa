/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/stringify.h>
#include <linux/thermal.h>
#include <linux/types.h>

#include "main.h"

#include "features.h"
#include "ftrace_events.h"
#include "utils.h"
#include "wq.h"

/* Millisecond interval for polling temperatures. Clamped to [1,5000]. */
#ifdef CONFIG_POLL_INTERVAL_MS
# define POLL_INTERVAL_MS		CLAMP((CONFIG_POLL_INTERVAL_MS), 1, 5000)
#else
# define POLL_INTERVAL_MS		100
#endif /* CONFIG_POLL_INTERVAL_MS */

/** Comma-separted list of names of thermal zones to monitor. */
#ifdef CONFIG_THERMAL_ZONE_LIST
static const char *thermal_zone_names[] = {STRINGIFY_ALL(CONFIG_THERMAL_ZONE_LIST)};
#else
static const char *thermal_zone_names[] = {"LITTLE","MID","BIG"};
#endif /* CONFIG_THERMAL_ZONE_LIST */

/* Number of items in THERMAL_ZONE_LIST. */
#define NUM_THERMAL_ZONES		(ARRAY_SIZE(thermal_zone_names))

#define FEATURE_NAME			lisa__thermal
#define TEMPERATURE_MIN			-273100
#define TEMPERATURE_MAX			999999

/* Shorthands for allocators. */
#define kzalloc_k(N)			(kzalloc((N), GFP_KERNEL))
#define kcalloc_k(N, SIZE)		(kcalloc((N), (SIZE), GFP_KERNEL))

struct thermal_config {
	int				temps[NUM_THERMAL_ZONES];
	struct thermal_zone_device	*devs[NUM_THERMAL_ZONES];
	struct work_item 		*work;
};

/* struct thermal_config accessors */
#define thermal_config_temp(CFG, I)	((CFG)->temps[(I)])
#define thermal_config_device(CFG, I)	((CFG)->devs[(I)])
#define trace_ith_tz(CFG, I, TS)				\
	trace_lisa__thermal(					\
		(TS),						\
		thermal_config_device((CFG), (I))->id,		\
		thermal_config_temp((CFG), (I)),		\
		thermal_config_device((CFG), (I))->type)

#define NANOS_TO_MILLIS 1000000

static int thermal_worker(void *data)
{
	struct thermal_config *config = data;
	/* Use same timestamp for each TZ to make parsing easier.
	 * Convert to millis for consistency with lisa__pixel6_emeter.
	 */
	u64 ts = ktime_get_real_ns()/NANOS_TO_MILLIS;
	for (u32 i = 0; i < NUM_THERMAL_ZONES; i++) {
		struct thermal_zone_device *dev;
		int temp, ret;

		dev = thermal_config_device(config, i);
		ret = thermal_zone_get_temp(dev, &temp);
		if (ret != 0) {
			pr_err("Could not get temperature for %s: %d\n",
			       dev->type, ret);
			continue;
		}

		temp = CLAMP(temp, TEMPERATURE_MIN, TEMPERATURE_MAX);
		if (temp != thermal_config_temp(config, i)) {
			thermal_config_temp(config, i) = temp;
			trace_ith_tz(config, i, ts);
		}
	}

	return WORKER_SAME_DELAY;
}

static void thermal_config_deinit(struct thermal_config *config)
{
	if (!config)
		return;
	/* Destroy work first in case the worker is running */
	destroy_work(config->work);
	/* Now the worker is stopped */
	kfree(config);
}

static int thermal_config_init(
	struct feature *feature, struct thermal_config **pconfig)
{
	int ret = 0;
	struct thermal_config *config;

	config = kzalloc_k(sizeof(*config));
	if (!config) {
		ret = -ENOMEM;
		goto out;
	}

	for (int i = 0; i < NUM_THERMAL_ZONES; i++) {
		config->devs[i] = thermal_zone_get_zone_by_name(thermal_zone_names[i]);
		if (config->devs[i] == NULL) {
			pr_err("No such device: %s\n", thermal_zone_names[i]);
			return -ENOENT;
		}

		/* Initialise to an invalid value so the first trace always
		 * detects the temperature as having changed. */
		config->temps[i] = THERMAL_TEMP_INVALID;
	}

out:
	if (ret != 0) {
		thermal_config_deinit(config);
		pr_warn("Initialise thermal zone list failed (err=%d)\n", ret);
		return ret;
	}

	*pconfig = config;
	pr_info("Thermal zone list initialised\n");
	return ret;
}

static int thermal_disable(struct feature *feature)
{

	thermal_config_deinit(feature->data);
	feature->data = NULL;
	return DISABLE_FEATURE(__worqueue);
}

static int thermal_enable(struct feature *feature)
{
	struct thermal_config *config;
	int ret = 0;

	ret = ENABLE_FEATURE(__worqueue);
	if (ret != 0)
		return ret;

	ret = thermal_config_init(feature, &config);
	if (ret != 0)
		goto out;

	feature->data = config;

	int delay_jiffies = (int)msecs_to_jiffies(POLL_INTERVAL_MS);
	config->work = start_work(thermal_worker, delay_jiffies, config);
	if (!config->work) {
		ret = -ENOMEM;
		goto out;
	}

out:
	if (ret != 0)
		thermal_disable(feature);
	return ret;
}

DEFINE_FEATURE(FEATURE_NAME, thermal_enable, thermal_disable);
