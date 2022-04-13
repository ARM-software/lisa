/* SPDX-License-Identifier: GPL-2.0 */

#ifndef _WQ_H
#define _WQ_H

#include <linux/workqueue.h>
#include <linux/list.h>

/**
 * enum worker_ret_t - Return value of typedef worker_t functions.
 * @WORKER_DISABLE: Disable the worker, so that it will not execute anymore.
 * @WORKER_SAME_DELAY: Schedule the worker to execute again in the future using
 * the same delay as for the past executions.
 *
 * Any other value will be interpreted as a delay in jiffies.
 */
enum worker_ret_t {
	WORKER_DISABLE = -1,
	WORKER_SAME_DELAY = -2,
};

/**
 * typedef worker_t - Worker function to be ran by a workqueue.
 * @data: Custom void * passed when starting the work with start_work()
 * Return: A value from enum worker_ret_t or a positive value interpreted as a
 * delay in jiffies for the next execution of that worker.
 */
typedef int (*worker_t)(void *data);

/**
 * struct work_item - Work item to be ran on a workqueue.
 * @f: Worker function.
 * @data: Custom void * set by the user and passed to f()
 */
struct work_item {
	worker_t f;
	void *data;

	/* Workqueue the item got scheduled on */
	struct workqueue_struct *__wq;
	/* Delayed work from kernel workqueue API */
	struct delayed_work __dwork;
	/* Delay of the last enqueue, used to implement WORKER_SAME_DELAY */
	int __delay;
};

/**
 * start_work() - Start a worker on a workqueue
 * @f: User function of the worker.
 * @data: void * passed to f()
 *
 * Context: The __workqueue feature must be enabled using
 * ENABLE_FEATURE(__workqueue) before starting any work.
 *
 * Return struct work_item* to be passed to destroy_work().
 */
struct work_item *start_work(worker_t f, int delay, void *data);

/**
 * destroy_work() - Stop a work item and deallocate it.
 * @item: struct work_item * to destroy.
 * Return: 0 if destruction was successful.
 */
int destroy_work(struct work_item *item);

#endif
