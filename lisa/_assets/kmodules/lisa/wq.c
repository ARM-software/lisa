/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/mutex.h>
#include <linux/slab.h>

#include "features.h"
#include "main.h"
#include "wq.h"

/* This worker function is not expected to be re-entrant. */
static void worker(struct work_struct* work) {
	int delay;
	struct work_item *item = container_of(to_delayed_work(work), struct work_item, __dwork);

	delay = item->f(item->data);

	if (delay == WORKER_SAME_DELAY)
		delay = item->__delay;
	else
		item->__delay = delay;

	if (delay >= 0 && delay != WORKER_DISABLE)
		queue_delayed_work(item->__wq, &item->__dwork, delay);
}

static __always_inline void __start_work(struct work_item *item)
{
	if (item->__cpu < 0)
		/* cpu-unbound work - try to use local */
		queue_delayed_work(item->__wq, &item->__dwork, item->__delay);
	else
		queue_delayed_work_on(item->__cpu, item->__wq, &item->__dwork,
				      item->__delay);
}

struct work_item *start_work_on(worker_t f, int delay, int cpu, void *data) {
	struct work_item *item;
	struct workqueue_struct *wq = FEATURE(__worqueue)->data;
	if (!wq)
		return NULL;

	item = kmalloc(sizeof(*item), GFP_KERNEL);
	if (item) {
		item->f = f;
		item->data = data;

		item->__cpu = cpu;
		item->__delay = delay;
		item->__wq = wq;
		INIT_DELAYED_WORK(&item->__dwork, worker);

		__start_work(item);
	}
	return item;
}

void restart_work(struct work_item *item, int delay)
{
	struct workqueue_struct *wq = FEATURE(__worqueue)->data;

	if (!wq || !item)
		return;

	item->__delay = delay;
	__start_work(item);
}

int destroy_work(struct work_item *item) {
	if (item) {
		cancel_delayed_work_sync(&item->__dwork);
		kfree(item);
	}
	return 0;
}


static int init_wq(struct feature *feature) {
	struct workqueue_struct *wq;
	wq = alloc_workqueue("lisa", WQ_FREEZABLE, 0);
	feature->data = wq;
	if (!wq) {
		pr_err("Could not allocate workqueue\n");
		return 1;
	}
	return 0;
}

static int deinit_wq(struct feature *feature) {
	struct workqueue_struct *wq = feature->data;
	if (wq)
		destroy_workqueue(wq);
	return 0;
}
DEFINE_INTERNAL_FEATURE(__worqueue, init_wq, deinit_wq);


/*
 * Example of a feature using workqueues.
 */

struct example_data {
	struct work_item *work;
};

static int example_worker(void *data) {
	struct feature *feature = data;
	pr_info("executing a wq item of feature %s\n", feature->name);

	/* Schedule the next run in 3 seconds */
	/* return HZ * 3; */

	/* Disable the worker. */
	/* return WORKER_DISABLE; */

	/* Schedule the next run using the same delay as previously */
	return WORKER_SAME_DELAY;
}

__maybe_unused static int example_init(struct feature *feature) {
	struct example_data *data;
	pr_info("Starting worker for feature %s\n", feature->name);

	if (ENABLE_FEATURE(__worqueue))
		return 1;

	data = kmalloc(sizeof(*data), GFP_KERNEL);
	feature->data = data;
	if (!data)
		return 1;

	data->work = start_work(example_worker, 2 * HZ, feature);
	if (!data->work)
		return 1;

	return 0;
}

__maybe_unused static int example_deinit(struct feature *feature) {
	int ret = 0;
	struct example_data *data = feature->data;

	pr_info("Stopping worker for feature %s\n", feature->name);

	if (data) {
		ret |= destroy_work(data->work);
		kfree(data);
	}

	ret |= DISABLE_FEATURE(__worqueue);

	return ret;
}
/* DEFINE_FEATURE(example, example_init, example_deinit); */
