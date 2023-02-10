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

	/* CPU to queue the work on (-1 for cpu-unbound) */
	int __cpu;
	/* Workqueue the item got scheduled on */
	struct workqueue_struct *__wq;
	/* Delayed work from kernel workqueue API */
	struct delayed_work __dwork;
	/* Delay of the last enqueue, used to implement WORKER_SAME_DELAY */
	int __delay;
};

/**
 * start_work_on() - Start a worker on a workqueue
 * @f: User function of the worker.
 * @delay: An amount of time (in jiffies) to wait before queueing the work
 * @cpu: cpu id to queue the work on
 * @data: void * passed to f()
 *
 * Context: The __workqueue feature must be enabled using
 * ENABLE_FEATURE(__workqueue) before starting any work.
 *
 * Return struct work_item* to be passed to destroy_work().
 */
struct work_item *start_work_on(worker_t f, int delay, int cpu, void *data);

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
static __always_inline
struct work_item *start_work(worker_t f, int delay, void *data)
{
	return start_work_on(f, delay, -1, data);
}

/**
 * restart_work() - Queue existing worker
 * @wi - An existing struct work_item instance to queue
 * @delay - An amount of time (in jiffies) to wait before queueing the work
 *
 * Context: The struct work_item should be properly initialised prior to
 * re-queueing on a dedicated workqueue.
 */
void restart_work(struct work_item *wi, int delay);

/**
 * destroy_work() - Stop a work item and deallocate it.
 * @item: struct work_item * to destroy.
 * Return: 0 if destruction was successful.
 */
int destroy_work(struct work_item *item);

#endif
