/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/jiffies.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/types.h>

#include "main.h"
#include "features.h"
#include "wq.h"
#include "ftrace_events.h"
#include "parsec.h"

/* There is no point in setting this value to less than 8 times what is written
 * in usec to POWER_METER_RATE_FILE
 */
#define POWER_METER_SAMPLING_RATE_MS 50

#define POWER_METER_SAMPLE_FILE_MAX_SIZE 1024
#define POWER_METER_SAMPLE_FILE "/sys/bus/iio/devices/iio:device0/energy_value"
#define POWER_METER_RATE_FILE "/sys/bus/iio/devices/iio:device0/sampling_rate"

static PARSE_RESULT(int) parse_content(parse_buffer *);

typedef struct sample {
	unsigned long ts;
	unsigned long value;
	unsigned int chan;
	char chan_name[PIXEL6_EMETER_CHAN_NAME_MAX_SIZE];
} sample_t;
DEFINE_PARSE_RESULT_TYPE(sample_t);


static struct file *open_file(int *error, const char *path, umode_t mode)
{
	struct file* file;
	file = filp_open(path, mode, 0);
	if (IS_ERR_OR_NULL(file)) {
		pr_err("Could not open %s: %li\n", path, file ? PTR_ERR(file) : 0);
		*error |= 1;
		file = NULL;
	}
	return file;
}

static void close_file(int *error, struct file *file)
{
	if (file) {
		int close_ret = filp_close(file, 0);
		if (close_ret) {
			pr_err("Could not close file: %i\n", close_ret);
			*error |= 1;
		}
	}
}

static void write_str(struct file *file, char *str)
{
	if (file) {
		kernel_write(file, str, strlen(str) + 1, 0);
	}
}


struct p6_emeter_data {
	struct work_item *work;
	struct file *sample_file;
};

static int free_p6_emeter_data(struct p6_emeter_data *data) {
	int ret = 0;
	if (data) {
		ret |= destroy_work(data->work);

		close_file(&ret, data->sample_file);

		kfree(data);
	}
	return ret;
}

static void process_content(unsigned char *content, size_t content_capacity)
{

	size_t size = strlen(content) + 1;
	parse_buffer input = {
		.data = (u8 *)content,
		.size = size,
		.capacity = content_capacity,
	};
	PARSE_RESULT(int) res = parse_content(&input);
	if (!IS_SUCCESS(res))
		pr_err("Failed to parse content\n");
}

static int p6_emeter_worker(void *data) {
	struct feature *feature = data;
	struct p6_emeter_data *p6_emeter_data = feature->data;
	char content[POWER_METER_SAMPLE_FILE_MAX_SIZE];
	ssize_t count = 0;

	count = kernel_read(p6_emeter_data->sample_file, content, POWER_METER_SAMPLE_FILE_MAX_SIZE - 1, 0);
	if (count < 0 || count >= POWER_METER_SAMPLE_FILE_MAX_SIZE) {
		pr_err("Could not read " POWER_METER_SAMPLE_FILE ": %ld\n", count);
	} else {
		content[count] = '\0';
		process_content(content, ARRAY_SIZE(content));
	}

	/* Schedule the next run using the same delay as previously */
	return WORKER_SAME_DELAY;
}


static int enable_p6_emeter(struct feature* feature) {
	struct p6_emeter_data *data = NULL;
	struct file* sample_file = NULL;
	struct file *rate_file = NULL;
	int ret = 0;

#define HANDLE_ERR(code) if (code) {ret |= code; goto finish;}

	HANDLE_ERR(ENABLE_FEATURE(__worqueue))

	data = kzalloc(sizeof(*data), GFP_KERNEL);
	feature->data = data;
	if (!data)
		HANDLE_ERR(1);

	/* Note that this is the hardware sampling rate. Software will only see
	 *an updated value every 8 hardware periods
	 */
	rate_file = open_file(&ret, POWER_METER_RATE_FILE, O_WRONLY);
	write_str(rate_file, "500\n");
	close_file(&ret, rate_file);
	HANDLE_ERR(ret);

	sample_file = open_file(&ret, POWER_METER_SAMPLE_FILE, O_RDONLY);
	data->sample_file = sample_file;
	HANDLE_ERR(ret);

	data->work = start_work(p6_emeter_worker, msecs_to_jiffies(POWER_METER_SAMPLING_RATE_MS), feature);
	if (!data->work)
		ret |= 1;
finish:
	return ret;
#undef HANDLE_ERR
};

static int disable_p6_emeter(struct feature* feature) {
	int ret = 0;
	struct p6_emeter_data *data = feature->data;

	if (data)
		free_p6_emeter_data(data);

	ret |= DISABLE_FEATURE(__worqueue);

	return ret;
};

DEFINE_FEATURE(event__pixel6_emeter, enable_p6_emeter, disable_p6_emeter);



/***********************************************
 * Parser for the energy_value sysfs file format
 ***********************************************/

APPLY(u8, parse_name_char, parse_not_char, ']');
TAKEWHILE(u8, parse_name, parse_name_char);

SEQUENCE(sample_t, parse_sample, ({
	sample_t value;

	/* CH42 */
	PARSE(parse_string, "CH");
	value.chan = PARSE(parse_ulong);

	/* (T=42) */
	PARSE(parse_string, "(T=");
	value.ts = PARSE(parse_ulong);
	PARSE(parse_char, ')');

	/* [CHAN_NAME] */
	PARSE(parse_char, '[');
	parse_buffer _name = PARSE(parse_name);
	parse_buffer2charp(&_name, value.chan_name,
			PIXEL6_EMETER_CHAN_NAME_MAX_SIZE);
	PARSE(parse_char, ']');

	/* , */
	PARSE(parse_string, ", ");

	/* 12345 */
	value.value = PARSE(parse_ulong);

	value;
}))

LEFT(sample_t, int, parse_sample_line, parse_sample, count_whitespaces)

int process_sample(int nr, sample_t sample)
{
	/* pr_info("parsed: chan=%u, ts=%lu chan_name=%s value=%lu\n", sample.chan, */
	/*        sample.ts, sample.chan_name, sample.value); */
	trace_pixel6_emeter(sample.ts, sample.chan, sample.chan_name, sample.value);
	return nr + 1;
}

MANY(int, parse_all_samples, parse_sample_line, process_sample);

/* This parser is able to parse strings formatted like that:
   "t=473848\nCH42(T=473848)[S10M_VDD_TPU], 3161249\nCH1(T=473848)[VSYS_PWR_MODEM], 48480309\nCH2(T=473848)[VSYS_PWR_RFFE], 9594393\nCH3(T=473848)[S2M_VDD_CPUCL2], 28071872\nCH4(T=473848)[S3M_VDD_CPUCL1], 17477139\nCH5(T=473848)[S4M_VDD_CPUCL0], 113447446\nCH6(T=473848)[S5M_VDD_INT], 12543588\nCH7(T=473848)[S1M_VDD_MIF], 25901660\n"
*/
SEQUENCE(int, parse_content, ({
	/* t=12345 */
	PARSE(parse_string, "t=");
	PARSE(parse_number_string);
	PARSE(count_whitespaces);

	/* Parse all the following sample lines */
	PARSE(parse_all_samples, 0);
}))
