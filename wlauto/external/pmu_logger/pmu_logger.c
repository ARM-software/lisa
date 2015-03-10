/*    Copyright 2013-2015 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/


/*
 * pmu_logger.c - Kernel module to log the CCI PMU counters
 */

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/debugfs.h>
#include <linux/timer.h>
#include <asm/io.h>

#define MODULE_NAME "cci_pmu_logger"

// CCI_BASE needs to be modified to point to the mapped location of CCI in
// memory on your device.
#define CCI_BASE 0x2C090000  // TC2
//#define CCI_BASE 0x10D20000 
#define CCI_SIZE 0x00010000

#define PMCR 0x100

#define PMCR_CEN (1 << 0)
#define PMCR_RST (1 << 1)
#define PMCR_CCR (1 << 2)
#define PMCR_CCD (1 << 3)
#define PMCR_EX  (1 << 4)
#define PMCR_DP  (1 << 5)

#define CC_BASE  0x9000
#define PC0_BASE 0xA000
#define PC1_BASE 0xB000
#define PC2_BASE 0xC000
#define PC3_BASE 0xD000

#define PC_ESR      0x0
#define CNT_VALUE   0x4
#define CNT_CONTROL 0x8

#define CNT_ENABLE (1 << 0)

u32 counter0_event = 0x6A;
u32 counter1_event = 0x63;
u32 counter2_event = 0x8A;
u32 counter3_event = 0x83;

u32 enable_console = 0;
u32 enable_ftrace = 1;

void *cci_base = 0;

static struct dentry *module_debugfs_root;
static int enabled = false;

u32 delay = 10; //jiffies. This translates to 1 sample every 100 ms
struct timer_list timer;

static void call_after_delay(void)
{
	timer.expires = jiffies + delay;
	add_timer(&timer);
}

	
static void setup_and_call_after_delay(void (*fn)(unsigned long))
{
	init_timer(&timer);
	timer.data = (unsigned long)&timer;
	timer.function = fn;

	call_after_delay();
}

static void print_counter_configuration(void)
{
	if (enable_ftrace)
		trace_printk("Counter_0: %02x Counter_1: %02x Counter_2: %02x Counter_3: %02x\n", \
			     counter0_event, counter1_event, counter2_event, counter3_event);

	if (enable_console)
		printk("Counter_0: %02x Counter_1: %02x Counter_2: %02x Counter_3: %02x\n", \
		       counter0_event, counter1_event, counter2_event, counter3_event);
}

static void initialize_cci_pmu(void)
{
	u32 val;

	// Select the events counted
	iowrite32(counter0_event, cci_base + PC0_BASE + PC_ESR);
	iowrite32(counter1_event, cci_base + PC1_BASE + PC_ESR);
	iowrite32(counter2_event, cci_base + PC2_BASE + PC_ESR);
	iowrite32(counter3_event, cci_base + PC3_BASE + PC_ESR);

	// Enable the individual PMU counters
	iowrite32(CNT_ENABLE, cci_base + PC0_BASE + CNT_CONTROL);
	iowrite32(CNT_ENABLE, cci_base + PC1_BASE + CNT_CONTROL);
	iowrite32(CNT_ENABLE, cci_base + PC2_BASE + CNT_CONTROL);
	iowrite32(CNT_ENABLE, cci_base + PC3_BASE + CNT_CONTROL);
	iowrite32(CNT_ENABLE, cci_base + CC_BASE + CNT_CONTROL);

	// Reset the counters and configure the Cycle Count Divider
	val = ioread32(cci_base + PMCR);
	iowrite32(val | PMCR_RST | PMCR_CCR | PMCR_CCD, cci_base + PMCR);
}

static void enable_cci_pmu_counters(void)
{
	u32 val = ioread32(cci_base + PMCR);
	iowrite32(val | PMCR_CEN, cci_base + PMCR);
}

static void disable_cci_pmu_counters(void)
{
	u32 val = ioread32(cci_base + PMCR);
	iowrite32(val & ~PMCR_CEN, cci_base + PMCR);
}

static void trace_values(unsigned long arg)
{
	u32 cycles;
	u32 counter[4];

	cycles = ioread32(cci_base + CC_BASE + CNT_VALUE);
	counter[0] = ioread32(cci_base + PC0_BASE + CNT_VALUE);
	counter[1] = ioread32(cci_base + PC1_BASE + CNT_VALUE);
	counter[2] = ioread32(cci_base + PC2_BASE + CNT_VALUE);
	counter[3] = ioread32(cci_base + PC3_BASE + CNT_VALUE);

	if (enable_ftrace)
		trace_printk("Cycles: %08x Counter_0: %08x"
			     " Counter_1: %08x Counter_2: %08x Counter_3: %08x\n", \
			     cycles, counter[0], counter[1], counter[2], counter[3]);

	if (enable_console)
		printk("Cycles: %08x Counter_0: %08x"
		       " Counter_1: %08x Counter_2: %08x Counter_3: %08x\n", \
		       cycles, counter[0], counter[1], counter[2], counter[3]);

	if (enabled) {
		u32 val;
		// Reset the counters
		val = ioread32(cci_base + PMCR);
		iowrite32(val | PMCR_RST | PMCR_CCR, cci_base + PMCR);

		call_after_delay();
	}
}

static ssize_t read_control(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
	char status[16];
	/* printk(KERN_DEBUG "%s\n", __func__); */

	if (enabled)
		snprintf(status, 16, "enabled\n");
	else
		snprintf(status, 16, "disabled\n");

	return simple_read_from_buffer(buf, count, ppos, status, strlen(status));
}

static ssize_t write_control(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
	if (enabled) {
		disable_cci_pmu_counters();
		enabled = false;
	} else {
		initialize_cci_pmu();
		enable_cci_pmu_counters();
		enabled = true;

		print_counter_configuration();
		setup_and_call_after_delay(trace_values);
	}

	return count;
}

static ssize_t read_values(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
	char values[256];
	/* u32 val; */

	snprintf(values, 256, "Cycles: %08x Counter_0: %08x"
		 " Counter_1: %08x Counter_2: %08x Counter_3: %08x\n", \
		 ioread32(cci_base + CC_BASE + CNT_VALUE),  \
		 ioread32(cci_base + PC0_BASE + CNT_VALUE), \
		 ioread32(cci_base + PC1_BASE + CNT_VALUE), \
		 ioread32(cci_base + PC2_BASE + CNT_VALUE), \
		 ioread32(cci_base + PC3_BASE + CNT_VALUE));

	return simple_read_from_buffer(buf, count, ppos, values, strlen(values));
}

static const struct file_operations control_fops = {
	.owner = THIS_MODULE,
	.read = read_control,
	.write = write_control,
};

static const struct file_operations value_fops = {
	.owner = THIS_MODULE,
	.read = read_values,
};

static int __init pmu_logger_init(void)
{
	struct dentry *retval;
	
	module_debugfs_root = debugfs_create_dir(MODULE_NAME, NULL);
	if (!module_debugfs_root || IS_ERR(module_debugfs_root)) {
		printk(KERN_ERR "error creating debugfs dir.\n");
		goto out;
	}
	
	retval = debugfs_create_file("control", S_IRUGO | S_IWUGO, module_debugfs_root, NULL, &control_fops);
	if (!retval)
		goto out;

	retval = debugfs_create_file("values", S_IRUGO, module_debugfs_root, NULL, &value_fops);
	if (!retval)
		goto out;

	retval = debugfs_create_bool("enable_console", S_IRUGO | S_IWUGO, module_debugfs_root, &enable_console);
	if (!retval)
		goto out;

	retval = debugfs_create_bool("enable_ftrace", S_IRUGO | S_IWUGO, module_debugfs_root, &enable_ftrace);
	if (!retval)
		goto out;

	retval = debugfs_create_u32("period_jiffies", S_IRUGO | S_IWUGO, module_debugfs_root, &delay);
	if (!retval)
		goto out;

	retval = debugfs_create_x32("counter0", S_IRUGO | S_IWUGO, module_debugfs_root, &counter0_event);
	if (!retval)
		goto out;
	retval = debugfs_create_x32("counter1", S_IRUGO | S_IWUGO, module_debugfs_root, &counter1_event);
	if (!retval)
		goto out;
	retval = debugfs_create_x32("counter2", S_IRUGO | S_IWUGO, module_debugfs_root, &counter2_event);
	if (!retval)
		goto out;
	retval = debugfs_create_x32("counter3", S_IRUGO | S_IWUGO, module_debugfs_root, &counter3_event);
	if (!retval)
		goto out;

	cci_base = ioremap(CCI_BASE, CCI_SIZE);
	if (!cci_base)
		goto out;

	printk(KERN_INFO "CCI PMU Logger loaded.\n");
	return 0;
	
out:
	debugfs_remove_recursive(module_debugfs_root);
	return 1;
}

static void __exit pmu_logger_exit(void)
{
	if (module_debugfs_root) {
		debugfs_remove_recursive(module_debugfs_root);
		module_debugfs_root = NULL;
	}
	if (cci_base)
		iounmap(cci_base);

	printk(KERN_INFO "CCI PMU Logger removed.\n");
}

module_init(pmu_logger_init);
module_exit(pmu_logger_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Punit Agrawal");
MODULE_DESCRIPTION("logger for CCI PMU counters");
