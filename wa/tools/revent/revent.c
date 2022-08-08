/*    Copyright 2012-2017 ARM Limited
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

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/input.h>
#include <linux/uinput.h>

#define die(args...) do { \
	fprintf(stderr, "ERROR: "); \
	fprintf(stderr, args);   \
	fprintf(stderr, "\n");  \
	exit(EXIT_FAILURE); \
} while(0)

#define dprintf(args...) if (verbose) printf(args)

#define INPDEV_MAX_DEVICES  16
#define INPDEV_MAX_PATH     30
#define MAX_NAME_LEN 255
#define EV_BITS_SIZE (EV_MAX / 8 + 1)
#define KEY_BITS_SIZE (KEY_MAX / 8 + 1)


#define HEADER_PADDING_SIZE 6
#define EVENT_PADDING_SIZE 4

const char MAGIC[] = "REVENT";

// NOTE: This should be incremented if any changes are made to the file format.
//       Should that be the case, also make sure to update the format description
//       in doc/source/revent.rst and the Python parser in wa/utils/revent.py.
uint16_t FORMAT_VERSION = 3;

typedef enum {
	FALSE=0,
	TRUE
} bool_t;

typedef enum {
	GENERAL_MODE=0,
	GAMEPAD_MODE,
	INVALID_MODE  // should be last
} recording_mode_t;

typedef enum {
	RECORD_COMMAND=0,
	REPLAY_COMMAND,
	DUMP_COMMAND,
	INFO_COMMAND,
	INVALID_COMMAND
} revent_command_t;

typedef struct {
	struct input_absinfo absinfo;
	int ev_code;
} absinfo_t;

typedef struct {
	struct input_id id;
	char name[MAX_NAME_LEN];
	char ev_bits[EV_BITS_SIZE];
	char abs_bits[KEY_BITS_SIZE];
	char rel_bits[KEY_BITS_SIZE];
	char key_bits[KEY_BITS_SIZE];
	uint32_t num_absinfo;
	absinfo_t absinfo[ABS_CNT];
} device_info_t;

typedef struct {
	revent_command_t command;
	recording_mode_t mode;
	int32_t record_time;
	int32_t device_number;
	char *file;
} revent_args_t;

typedef struct {
	int32_t num;
	char **paths;
	int *fds;
	int max_fd;
} input_devices_t;

typedef struct {
	int16_t dev_idx;
	struct input_event event;
} replay_event_t;

typedef struct {
	uint16_t version;
	recording_mode_t mode;
} revent_record_desc_t;

typedef struct {
	revent_record_desc_t desc;
	input_devices_t devices;
	device_info_t *gamepad_info;
	uint64_t num_events;
	struct timeval start_time;
	struct timeval end_time;
	replay_event_t *events;
} revent_recording_t;

bool_t verbose = FALSE;
bool_t wait_for_stdin = TRUE;

bool_t is_numeric(char *string)
{
	int len = strlen(string);

	int i = 0;
	while(i < len)
	{
		if(!isdigit(string[i]))
			return FALSE;
		i++;
	}

	return TRUE;
}

int test_bit(const char *mask, int bit) {
	return mask[bit / 8] & (1 << (bit % 8));
}

int count_bits(const char *mask) {
	int count = 0, i;
	static const uint8_t nybble_lookup[16] = {
		0, 1, 1, 2, 1, 2, 2, 3,
		1, 2, 2, 3, 2, 3, 3, 4
	};

	for (i = 0; i < KEY_MAX/8 + 1; i++) {
		char byte = mask[i];
		count +=  nybble_lookup[byte & 0x0F] + nybble_lookup[byte >> 4];
	}

	return count;
}

/*
 * An input device is considered to be a gamepad if it supports
 * ABS x and Y axes and the four gamepad buttons (variously known as
 * square/triangle/circle/X, A/B/X/Y, or north/south/east/west).
 */
bool_t is_gamepad(device_info_t *dev)
{
	if (!test_bit(dev->abs_bits, ABS_X))
		return FALSE;
	if (!test_bit(dev->abs_bits, ABS_Y))
		return FALSE;
	if (!test_bit(dev->key_bits, BTN_GAMEPAD))
		return FALSE;
	return TRUE;
}

off_t get_file_size(const char *filename) {
	struct stat st;

	if (stat(filename, &st) == 0)
		return st.st_size;

	die("Cannot determine size of %s: %s", filename, strerror(errno));
}

int get_device_info(int fd, device_info_t *info) {
	bzero(info, sizeof(device_info_t));

	if (ioctl(fd, EVIOCGID, &info->id) < 0)
		return errno;

	if (ioctl(fd, EVIOCGNAME(MAX_NAME_LEN * sizeof(char)), &info->name) < 0)
		return errno;

	if (ioctl(fd, EVIOCGBIT(0, sizeof(info->ev_bits)), &info->ev_bits) < 0)
		return errno;

	int ev_type;
	for (ev_type = 0 ; ev_type < EV_MAX; ev_type++) {
		if (test_bit(info->ev_bits, ev_type)) {

			if (ev_type == EV_ABS) {
				ioctl(fd, EVIOCGBIT(ev_type, sizeof(info->abs_bits)), &info->abs_bits);

				int ev_code;
				for (ev_code = 0; ev_code < KEY_MAX; ev_code++) {
					if (test_bit(info->abs_bits, ev_code)) {
						absinfo_t *inf = &info->absinfo[info->num_absinfo++];
						inf->ev_code = ev_code;
						ioctl(fd, EVIOCGABS(ev_code), &inf->absinfo);
					}
				}
			} else if (ev_type == EV_REL) {
				ioctl(fd, EVIOCGBIT(ev_type, sizeof(info->rel_bits)), &info->rel_bits);
			} else if (ev_type == EV_KEY) {
				ioctl(fd, EVIOCGBIT(ev_type, sizeof(info->key_bits)), &info->key_bits);
			}
		}
	}

	return 0;
}

void destroy_replay_device(int fd)
{
	if(ioctl(fd, UI_DEV_DESTROY) < 0)
		die("Could not destroy replay device");
}

inline void set_evbit(int fd, int bit)
{
	if(ioctl(fd, UI_SET_EVBIT, bit) < 0)
		die("Could not set EVBIT %i", bit);
}

inline void set_keybit(int fd, int bit)
{
	if(ioctl(fd, UI_SET_KEYBIT, bit) < 0)
		die("Could not set KEYBIT %i", bit);
}

inline void set_absbit(int fd, int bit)
{
	if(ioctl(fd, UI_SET_ABSBIT, bit) < 0)
		die("Could not set ABSBIT %i", bit);
}

inline void set_relbit(int fd, int bit)
{
	if(ioctl(fd, UI_SET_RELBIT, bit) < 0)
		die("Could not set RELBIT %i", bit);
}

inline void block_sigterm(sigset_t *oldset)
{
	sigset_t sigset;
	sigemptyset(&sigset);
	sigaddset(&sigset, SIGTERM);
	sigprocmask(SIG_BLOCK, &sigset, oldset);
}

// Events are recorded with their original timestamps, but for playback, we
// want to treat timestamps as deltas from event zero.
void adjust_timestamps(revent_recording_t *recording)
{
	uint64_t i;
	struct timeval time_zero, time_delta;

	time_zero.tv_sec = recording->start_time.tv_sec;
	time_zero.tv_usec = recording->start_time.tv_usec;

	for(i = 0; i < recording->num_events; i++) {
		timersub(&recording->events[i].event.time, &time_zero, &time_delta);
		recording->events[i].event.time.tv_sec = time_delta.tv_sec;
		recording->events[i].event.time.tv_usec = time_delta.tv_usec;
	}
	timersub(&recording->end_time, &time_zero, &time_delta);
	recording->end_time.tv_sec = time_delta.tv_sec;
	recording->end_time.tv_usec = time_delta.tv_usec;
}

int write_record_header(int fd, const revent_record_desc_t *desc)
{
	ssize_t ret;
	char padding[HEADER_PADDING_SIZE];

	ret = write(fd, MAGIC, 6);
	if (ret < 6)
		return errno;

	ret = write(fd, &desc->version, sizeof(desc->version));
	if (ret < sizeof(desc->version))
		return errno;

	ret = write(fd, (uint16_t *)&desc->mode, sizeof(uint16_t));
	if (ret < sizeof(uint16_t))
		return errno;

	bzero(padding, HEADER_PADDING_SIZE);
	ret = write(fd, padding, HEADER_PADDING_SIZE);
	if (ret < HEADER_PADDING_SIZE)
		return errno;

	return 0;
}

int read_record_header(int fd, revent_record_desc_t *desc)
{
	char start[7], padding[HEADER_PADDING_SIZE];
	ssize_t ret;

	ret = read(fd, start, 6);
	if (ret < 6)
		return errno;

	start[6] = '\0';
	if (strcmp(start, MAGIC))
		return EINVAL;

	ret = read(fd, &desc->version, sizeof(desc->version));
	if (ret < sizeof(desc->version))
		return errno;

	if (desc->version >= 2) {
		ret = read(fd, &desc->mode, sizeof(uint16_t));
		if (ret < sizeof(uint16_t))
			return errno;

		ret = read(fd, padding, HEADER_PADDING_SIZE);
		if (ret < HEADER_PADDING_SIZE)
			return errno;
	} else {
		/* Version 1 supports only general recordings (mode 0) and
		 * does not have padding
		 */
		desc->mode = GENERAL_MODE;
	}

	return 0;
}

int write_general_input_devices(const input_devices_t *devices, FILE *fout)
{
	size_t ret;
	uint32_t path_len;
	int i;

	ret = fwrite(&devices->num, sizeof(uint32_t), 1, fout);
	if (ret < 1) {
		return errno;
	}

	for (i = 0; i < devices->num; i++) {
		path_len = (uint32_t)strlen(devices->paths[i]);
		ret = fwrite(&path_len, sizeof(uint32_t), 1, fout);
		if (ret < 1) {
			return errno;
		}

		ret = fwrite(devices->paths[i], sizeof(char), path_len, fout);
		if (ret < path_len) {
			return errno;
		}
	}

	return 0;
}

int read_general_input_devices(input_devices_t *devices, FILE *fin)
{
	size_t ret;
	uint32_t path_len;
	int i;

	ret = fread(&devices->num, sizeof(uint32_t), 1, fin);
	if (ret < 1) {
		return EIO;
	}

	devices->paths = malloc(sizeof(char *) * devices->num);
	if (devices->paths == NULL) {
		return ENOMEM;
	}

	for (i = 0; i < devices->num; i++) {
		ret = fread(&path_len, sizeof(uint32_t), 1, fin);
		if (ret < 1) {
			return EIO;
		}

		devices->paths[i] = malloc(sizeof(char) * path_len + 1);
		if (devices->paths[i] == NULL) {
			return ENOMEM;
		}

		ret = fread(devices->paths[i], sizeof(char), path_len, fin);
		if (ret < path_len) {
			return EIO;
		}
		devices->paths[i][path_len] = '\0';
	}

	return 0;
}

int write_input_id(FILE *fout, const struct input_id *id)
{
	int ret = 0;
	ret += fwrite(&id->bustype, sizeof(uint16_t), 1, fout);
	ret += fwrite(&id->vendor, sizeof(uint16_t), 1, fout);
	ret += fwrite(&id->product, sizeof(uint16_t), 1, fout);
	ret += fwrite(&id->version, sizeof(uint16_t), 1, fout);
	if (ret < 4)
		return errno;
	return 0;
}

int read_input_id(FILE *fin, struct input_id *id)
{
	int ret = 0;
	ret += fread(&id->bustype, sizeof(uint16_t), 1, fin);
	ret += fread(&id->vendor, sizeof(uint16_t), 1, fin);
	ret += fread(&id->product, sizeof(uint16_t), 1, fin);
	ret += fread(&id->version, sizeof(uint16_t), 1, fin);
	if (ret < 4)
		return errno;
	return 0;
}

int write_absinfo(FILE *fout, const absinfo_t *info)
{
	int ret = 0;
	ret += fwrite(&info->ev_code, sizeof(int32_t), 1, fout);
	ret += fwrite(&info->absinfo.value, sizeof(int32_t), 1, fout);
	ret += fwrite(&info->absinfo.minimum, sizeof(int32_t), 1, fout);
	ret += fwrite(&info->absinfo.maximum, sizeof(int32_t), 1, fout);
	ret += fwrite(&info->absinfo.fuzz, sizeof(int32_t), 1, fout);
	ret += fwrite(&info->absinfo.flat, sizeof(int32_t), 1, fout);
	ret += fwrite(&info->absinfo.resolution, sizeof(int32_t), 1, fout);
	if (ret < 7)
		return errno;
	return 0;
}

int read_absinfo(FILE *fin, absinfo_t *info)
{
	int ret = 0;
	ret += fread(&info->ev_code, sizeof(int32_t), 1, fin);
	ret += fread(&info->absinfo.value, sizeof(int32_t), 1, fin);
	ret += fread(&info->absinfo.minimum, sizeof(int32_t), 1, fin);
	ret += fread(&info->absinfo.maximum, sizeof(int32_t), 1, fin);
	ret += fread(&info->absinfo.fuzz, sizeof(int32_t), 1, fin);
	ret += fread(&info->absinfo.flat, sizeof(int32_t), 1, fin);
	ret += fread(&info->absinfo.resolution, sizeof(int32_t), 1, fin);
	if (ret < 7)
		return errno;
	return 0;
}

int write_device_info(FILE *fout, const device_info_t *info)
{
	int ret = write_input_id(fout, &info->id);
	if (ret)
		return ret;

	uint32_t name_len = (uint32_t)strlen(info->name);
	ret = fwrite(&name_len, sizeof(uint32_t), 1, fout);
	ret += fwrite(info->name, sizeof(char), name_len, fout);
	if (ret < (name_len + 1))
		return EIO;

	ret = fwrite(info->ev_bits, sizeof(char), EV_BITS_SIZE, fout);
	ret += fwrite(info->abs_bits, sizeof(char), KEY_BITS_SIZE, fout);
	ret += fwrite(info->rel_bits, sizeof(char), KEY_BITS_SIZE, fout);
	ret += fwrite(info->key_bits, sizeof(char), KEY_BITS_SIZE, fout);
	if (ret < (EV_BITS_SIZE + KEY_BITS_SIZE * 3))
		return EIO;
        printf("EV_BITS_SIZE: %d\n", EV_BITS_SIZE);
        printf("KEY_BITS_SIZE: %d\n", KEY_BITS_SIZE);

	ret = fwrite(&info->num_absinfo, sizeof(uint32_t), 1, fout);
	if (ret < 1)
		return errno;

	int i;
	for (i = 0; i < info->num_absinfo; i++) {
		ret = write_absinfo(fout, &info->absinfo[i]);
		if (ret)
			return ret;
	}

	return 0;
}

int read_device_info(FILE *fin, device_info_t *info)
{
	int ret = read_input_id(fin, &info->id);
	if (ret)
		return ret;

	uint32_t name_len = 0;
	fread(&name_len, sizeof(uint32_t), 1, fin);
	if (!name_len)
		return EIO;

	ret += fread(info->name, sizeof(char), name_len, fin);
	if (ret < name_len)
		return EIO;
	info->name[name_len] = '\0';

	ret = fread(info->ev_bits, sizeof(char), EV_BITS_SIZE, fin);
	ret += fread(info->abs_bits, sizeof(char), KEY_BITS_SIZE, fin);
	ret += fread(info->rel_bits, sizeof(char), KEY_BITS_SIZE, fin);
	ret += fread(info->key_bits, sizeof(char), KEY_BITS_SIZE, fin);
	if (ret < (EV_BITS_SIZE + KEY_BITS_SIZE * 3))
		return EIO;

	ret = fread(&info->num_absinfo, sizeof(uint32_t), 1, fin);
	if (ret < 1)
		return errno;

	int i;
	for (i = 0; i < info->num_absinfo; i++) {
		ret = read_absinfo(fin, &info->absinfo[i]);
		if (ret)
			return ret;
	}

	return 0;
}

void print_device_info(device_info_t *info)
{
	printf("device name: %s\n", info->name);
	printf("bustype: 0x%x vendor: 0x%x product: 0x%x version: 0x%x\n",
                info->id.bustype, info->id.vendor, info->id.product, info->id.version);
	printf("abs_bits: %d\n", count_bits(info->abs_bits));
	printf("rel_bits: %d\n", count_bits(info->rel_bits));
	printf("key_bits: %d\n", count_bits(info->key_bits));
	printf("num_absinfo: %ld\n", info->num_absinfo);

	int i;
	printf("KEY: ");
	for (i = 0; i < KEY_MAX; i++) {
		if (test_bit(info->key_bits, i)) {
			printf("%04x ", i);
		}
	}
	printf("\n");

	struct input_absinfo *inf;
	int ev_code;
	printf("ABS:\n");
	for (i = 0; i < info->num_absinfo; i++) {
		ev_code = info->absinfo[i].ev_code;
		inf = &info->absinfo[i].absinfo;
		printf("%04x  : min %i, max %i, fuzz %0i, flat %i, res %i\n", ev_code,
				inf->minimum, inf->maximum, inf->fuzz, inf->flat,
				inf->resolution);
	}
}

int read_record_timestamps(FILE *fin, revent_recording_t *recording)
{
	int ret;
	ret = fread(&recording->start_time.tv_sec, sizeof(uint64_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&recording->start_time.tv_usec, sizeof(uint64_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&recording->end_time.tv_sec, sizeof(uint64_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&recording->end_time.tv_usec, sizeof(uint64_t), 1, fin);
	if (ret < 1)
		return errno;

	return 0;
}

int write_replay_event(FILE *fout, const replay_event_t *ev)
{
	size_t ret;
	uint64_t time;

	ret = fwrite(&ev->dev_idx, sizeof(uint16_t), 1, fout);
	if (ret < 1)
		return errno;
	
	time = (uint64_t)ev->event.time.tv_sec;
	ret = fwrite(&time, sizeof(uint64_t), 1, fout);
	if (ret < 1)
		return errno;

	time = (uint64_t)ev->event.time.tv_usec;
	ret = fwrite(&time, sizeof(uint64_t), 1, fout);
	if (ret < 1)
		return errno;

	ret = fwrite(&ev->event.type, sizeof(uint16_t), 1, fout);
	if (ret < 1)
		return errno;

	ret = fwrite(&ev->event.code, sizeof(uint16_t), 1, fout);
	if (ret < 1)
		return errno;

	ret = fwrite(&ev->event.value, sizeof(uint32_t), 1, fout);
	if (ret < 1)
		return errno;

	return 0;
}

int read_replay_event(FILE *fin, replay_event_t *ev)
{
	size_t ret;

	ret = fread(&ev->dev_idx, sizeof(uint16_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&ev->event.time.tv_sec, sizeof(uint64_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&ev->event.time.tv_usec, sizeof(uint64_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&ev->event.type, sizeof(uint16_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&ev->event.code, sizeof(uint16_t), 1, fin);
	if (ret < 1)
		return errno;

	ret = fread(&ev->event.value, sizeof(uint32_t), 1, fin);
	if (ret < 1)
		return errno;

	return 0;
}

int read_legacy_replay_event(int fdin, replay_event_t* ev)
{
	size_t rb;
	char padding[EVENT_PADDING_SIZE];

	rb = read(fdin, &(ev->dev_idx), sizeof(int32_t));
	if (rb < (int)sizeof(int32_t)){
		//Allow for abrupt ending of legacy recordings.
		if (!errno)
			return EOF;
		return errno;
	}
	rb = read(fdin, &padding, EVENT_PADDING_SIZE);
	if (rb < (int)sizeof(int32_t))
		return errno;

	struct timeval time;
	uint64_t temp_time;
	rb = read(fdin, &temp_time, sizeof(uint64_t));
	if (rb < (int)sizeof(uint64_t))
		return errno;
	time.tv_sec = (time_t)temp_time;

	rb = read(fdin, &temp_time, sizeof(uint64_t));
	if (rb < (int)sizeof(uint64_t))
		return errno;
	time.tv_usec = (suseconds_t)temp_time;

	ev->event.time = time;

	rb = read(fdin, &(ev->event.type), sizeof(uint16_t));
	if (rb < (int)sizeof(uint16_t))
		return errno;

	rb = read(fdin, &(ev->event.code), sizeof(uint16_t));
	if (rb < (int)sizeof(uint16_t))
		return errno;

	rb = read(fdin, &(ev->event.value), sizeof(int32_t));
	if (rb < (int)sizeof(int32_t))
		return errno;

	return 0;
}

int open_revent_recording(const char *filepath, revent_record_desc_t *desc, FILE **fin)
{
	*fin = fopen(filepath, "r");
	if (*fin == NULL)
		return errno;

	int ret = read_record_header(fileno(*fin), desc);
	if (ret)
		return ret;

	if (desc->version < 0 || desc->version > FORMAT_VERSION)
		return EPROTO;

	return 0;
}

FILE *init_recording(const char *pathname, recording_mode_t mode)
{
	revent_record_desc_t desc = { .mode = mode, .version = FORMAT_VERSION };

	FILE *fh = fopen(pathname, "w");
	if (fh == NULL)
		return fh;

	write_record_header(fileno(fh), &desc);

	return fh;
}

void init_input_devices(input_devices_t *devices)
{
	devices->num = 0;
	devices->max_fd = -1;
	devices->paths = NULL;
	devices->fds = NULL;
}

int init_general_input_devices(input_devices_t *devices)
{
	uint32_t num, i, path_len;
	char paths[INPDEV_MAX_DEVICES][INPDEV_MAX_PATH];
	int fds[INPDEV_MAX_DEVICES];
	int max_fd = 0;
	int ret;
	int clk_id = CLOCK_MONOTONIC;

	num = 0;
	for(i = 0; i < INPDEV_MAX_DEVICES; ++i) {
		sprintf(paths[num], "/dev/input/event%d", i);
		fds[num] = open(paths[num], O_RDONLY);
		if(fds[num] > 0) {
			if (fds[num] > max_fd)
				max_fd = fds[num];
			if (ret = ioctl(fds[num], EVIOCSCLOCKID, &clk_id)) {
				dprintf("Failed to set monotonic clock for %s.\n", paths[num]);
				return -ret;
			}
			dprintf("opened %s\n", paths[num]);
			num++;
		}
		else {
			dprintf("could not open %s\n", paths[num]);
		}
	}

	if (num == 0)
		return EACCES;

	devices->num = num;
	devices->max_fd = max_fd;

	devices->paths = malloc(sizeof(char *) * num);
	if (devices->paths == NULL) {
		return ENOMEM;
	}
	for (i = 0; i < num; i ++) {
		path_len = strlen(paths[i]);
		devices->paths[i] = malloc(sizeof(char) * (path_len + 1));
		if (devices->paths[i] == NULL)
			return ENOMEM;
		strncpy(devices->paths[i], paths[i],  path_len + 1);
	}

	devices->fds = malloc(sizeof(int) * num);
	if (devices->fds == NULL) {
		return ENOMEM;
	}
	for (i = 0; i < num; i ++)
		devices->fds[i] = fds[i];

	return 0;
}

void fini_general_input_devices(input_devices_t *devices)
{
	int i;
	for (i = 0; i < devices->num; i++) {
		if (devices->fds != NULL)
			close(devices->fds[i]);
		if (devices->paths != NULL)
			free(devices->paths[i]);
	}
	free(devices->fds);
	devices->num = 0;
}


int init_gamepad_input_devices(input_devices_t *devices, device_info_t *gamepad_info)
{
	int i;
	char *gamepad_path = NULL;
	input_devices_t all_devices;
	device_info_t info;

	int ret = init_general_input_devices(&all_devices);
	if (ret) {
		return ret;
	}

	for (i = 0; i < all_devices.num; i++) {
		ret = get_device_info(all_devices.fds[i], &info);
		if (ret) {
			dprintf("Could not get info for %s: %s\n", all_devices.paths[i], strerror(errno));
			continue;
		}

		if (!is_gamepad(&info)) {
			dprintf("not a gamepad: %s\n", all_devices.paths[i]);
			continue;
		}

		if (gamepad_path != NULL) {
			die("More than one device identified as a gamepad (run \"reven info\" to see which)");
		}

		gamepad_path = malloc(sizeof(char) * INPDEV_MAX_PATH);
		if (gamepad_path == NULL)
			die("Could not create replay device: %s", strerror(ENOMEM));
		strncpy(gamepad_path, all_devices.paths[i], INPDEV_MAX_PATH);
		memcpy(gamepad_info, &info, sizeof(device_info_t));
	}

	fini_general_input_devices(&all_devices);

	if (gamepad_path == NULL) {
		return ENOMEDIUM;
	}

	dprintf("Found gamepad: %s\n", gamepad_path);
	devices->num = 1;

	devices->paths = malloc(sizeof(char *));
	devices->paths[0] = gamepad_path;

	devices->fds = malloc(sizeof(int *));
	if (devices->fds == NULL)
		return ENOMEM;
	devices->fds[0] = open(gamepad_path, O_RDONLY);
	if (devices->fds[0] < 0) {
		return errno;
	}

	int clk_id = CLOCK_MONOTONIC;
	if (ret = ioctl(devices->fds[0], EVIOCSCLOCKID, &clk_id)) {
		dprintf("Could not set monotonic clock for the gamepad.\n");
		return -ret;
	}

	devices->max_fd = devices->fds[0];

	return 0;
}

void fini_gamepad_input_devices(input_devices_t *devices)
{
	fini_general_input_devices(devices);
}

void init_revent_recording(revent_recording_t *recording)
{
	recording->num_events = 0;
	recording->desc.version = 0;
	recording->desc.mode = INVALID_MODE;
	recording->events = NULL;
	recording->gamepad_info = NULL;
	init_input_devices(&recording->devices);
}

void fini_revent_recording(revent_recording_t *recording)
{
	if (recording->desc.mode == GENERAL_MODE) {
		fini_general_input_devices(&recording->devices);
	} else if (recording->desc.mode == GAMEPAD_MODE) {
		fini_gamepad_input_devices(&recording->devices);
		free(recording->gamepad_info);
	} else {
		// We're finalizing the recording so at this point,
		// we don't care.
	}
	if (recording->num_events) {
		free(recording->events);
	}
	recording->num_events = 0;
	recording->desc.version = 0;
	recording->desc.mode = INVALID_MODE;
}

void open_general_input_devices_for_playback_or_die(input_devices_t *devices)
{
	int i, ret;
	devices->fds = malloc(sizeof(int) * devices->num);
	if (devices->fds == NULL)
		die("Could not allocate file descriptor array: %s", strerror(ENOMEM));

	for (i = 0; i < devices->num; i++)
	{
		ret = open(devices->paths[i], O_WRONLY | O_NDELAY);
		if (ret < 0) {
			die("Could not open \"%s\" for writing: %s",
					devices->paths[i], strerror(errno));
		}
		devices->fds[i] = ret;
		if (devices->fds[i] > devices->max_fd)
			devices->max_fd =  devices->fds[i];
		dprintf("Opened %s\n", devices->paths[i]);
	}
}

int create_replay_device_or_die(const device_info_t *info)
{
	int i;

	int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
	if (fd < 0) {
		if (errno == ENOENT) {
			die("uinput not supported by the kernel (is the module installed?)");
		} else if (errno == EACCES) {
			die("Cannot access \"/dev/uinput\" (try re-running as root)");
		} else {
			die("Could not open \"/dev/uinput\" for writing: %s", strerror(errno));
		}
	}

	struct uinput_user_dev uidev;
	memset(&uidev, 0, sizeof(uidev));
	snprintf(uidev.name, UINPUT_MAX_NAME_SIZE, "revent-replay %s", info->name);
	uidev.id.bustype = BUS_USB;
	uidev.id.vendor  = info->id.vendor;
	uidev.id.product = info->id.product;
	uidev.id.version = info->id.version;

	set_evbit(fd, EV_SYN);

	set_evbit(fd, EV_KEY);
	for (i = 0; i < KEY_MAX; i++) {
		if (test_bit(info->key_bits, i))
			set_keybit(fd, i);
	}

	set_evbit(fd, EV_REL);
	for (i = 0; i < REL_MAX; i++) {
		if (test_bit(info->rel_bits, i))
			set_relbit(fd, i);
	}

	set_evbit(fd, EV_ABS);
	for (i = 0; i < info->num_absinfo; i++) {
		int ev_code = info->absinfo[i].ev_code;
		set_absbit(fd, ev_code);
		uidev.absmin[ev_code] = info->absinfo[i].absinfo.minimum;
		uidev.absmax[ev_code] = info->absinfo[i].absinfo.maximum;
		uidev.absfuzz[ev_code] = info->absinfo[i].absinfo.fuzz;
		uidev.absflat[ev_code] = info->absinfo[i].absinfo.flat;
	}
	if (write(fd, &uidev, sizeof(uidev)) < sizeof(uidev)) {
		die("Could not write absinfo:", strerror(errno));
	}

	if(ioctl(fd, UI_DEV_CREATE) < 0)
		die("Could not create replay device:", strerror(errno));

        // wait for the new device to be recognised by the system
        sleep(3);

	return fd;
}

inline void read_revent_recording_or_die(const char *filepath, revent_recording_t *recording)
{
	int ret;
	FILE *fin;
	uint64_t i;
	off_t fsize;

	ret = open_revent_recording(filepath, &recording->desc, &fin);
	if (ret) {
		if (ret == EINVAL) {
			die("%s does not appear to be an revent recording", filepath);
		} else if (ret == EPROTO) {
			die("%s contains recording for unsupported version \"%u\"; max supported version is \"%u\"",
					filepath, recording->desc.version, FORMAT_VERSION);
		} else  {
			die("%s revent recording appears to be corrupted", filepath);
		}
	}

	if (recording->desc.mode == GENERAL_MODE) {
		ret = read_general_input_devices(&recording->devices, fin);
		if (ret) {
			die("Could not read devices: %s", strerror(ret));
		}
		recording->gamepad_info = NULL;
	} else if (recording->desc.mode == GAMEPAD_MODE) {
		recording->gamepad_info = malloc(sizeof(device_info_t));
		if (recording->gamepad_info == NULL)
			die("Could not allocate gamepad info buffer: %s", strerror(ENOMEM));
		ret = read_device_info(fin, recording->gamepad_info);
		if (ret)
			die("Could not read gamepad info: %s", strerror(ret));
	} else {
		die("Unexpected recording mode: %d", recording->desc.mode);
	}

	if (recording->desc.version > 1) {
		ret = fread(&recording->num_events, sizeof(uint64_t), 1, fin);
		if (ret < 1)
			die("Could not read the number of recorded events");

		if (recording->desc.version > 2) {
			ret = read_record_timestamps(fin, recording);
			if (ret)
				die("Could not read recroding timestamps.");
		}

		recording->events = malloc(sizeof(replay_event_t) * recording->num_events);
		if (recording->events == NULL)
			die("Not enough memory to allocate replay buffer");

		// start/end times tracking for recording as a whole was added in version 3
		// of recording format; for earlier recordings, use timestamps of the first and
		// last events.
		read_replay_event(fin, &recording->events[0]);
		if (recording->desc.version <= 2) {
			recording->start_time.tv_sec  = recording->events[0].event.time.tv_sec;
			recording->start_time.tv_usec  = recording->events[0].event.time.tv_usec;
		}

		for(i=1; i < recording->num_events; i++) {
			read_replay_event(fin, &recording->events[i]);
		}

		if (recording->desc.version <= 2) {
			recording->end_time.tv_sec  = recording->events[i].event.time.tv_sec;
			recording->end_time.tv_usec  = recording->events[i].event.time.tv_usec;
		}
	} else {   // backwards compatibility
		/* Prior to verion 2, the total number of recorded events was not being
		 * written as part of the recording. We will use the size of the file on
		 * disk to estimate the recording buffer size and keep reading the events
		 * untils EOF, keeping track of how many we read so that the total can
		 * then be updated. The format of the events is also different -- it
		 * featured larger device ID an unnecessary padding.
		 */
		 fsize  = get_file_size(filepath);
		 recording->events = malloc((size_t)fsize);
		 i = 0;

		// Safely get file descriptor for fin, by flushing first.
		fflush(fin);

		 while (1) {
			ret = read_legacy_replay_event(fileno(fin), &recording->events[i]);
			if (ret == EOF) {
				break;
			} else if (ret) {
				die("error reading events: %s", strerror(ret));
			}
			i++;
		 }
		 recording->num_events = i;
	}

	fclose(fin);
}

void open_gamepad_input_devices_for_playback_or_die(input_devices_t *devices, const device_info_t *info)
{
	int fd = create_replay_device_or_die(info);
	devices->num = 1;
	devices->fds = malloc(sizeof(int));
	if (devices->fds == NULL)
		die("Could not create replay devices: %s", strerror(ENOMEM));
	devices->fds[0] = fd;
	devices->max_fd = fd;
}

//Used to exit program properly on termination
static volatile int EXIT = 0;
void exitHandler(int z) {
    EXIT = 1;
}

void record(const char *filepath, int delay, recording_mode_t mode)
{
	int ret;
	struct timespec start_time, end_time;
	FILE *fout = init_recording(filepath, mode);
	if (fout == NULL)
		die("Could not create recording \"%s\": %s", filepath, strerror(errno));

	input_devices_t devices;
	init_input_devices(&devices);

	if (mode == GENERAL_MODE) {
		ret = init_general_input_devices(&devices);
		if (ret)
			die("Could not initialize input devices: %s", strerror(ret));
		ret = write_general_input_devices(&devices, fout);
		if (ret)
			die("Could not record input devices: %s", strerror(ret));
	} else if (mode == GAMEPAD_MODE) {
		device_info_t info;
		ret = init_gamepad_input_devices(&devices, &info);
		if (ret == ENOMEDIUM) {
			die("There does not appear to be a gamepad connected");
		} else if (ret) {
			die("Problem initializing gamepad device: %s", strerror(ret));
		}
		ret = write_device_info(fout, &info);
		if (ret)
			die("Problem writing gamepad info: %s", strerror(ret));
	} else {
		fclose(fout);
		die("Invalid recording mode specified");
	}

	sigset_t old_sigset;
	sigemptyset(&old_sigset);
	block_sigterm(&old_sigset);

	// Write the zero size as a place holder and remember the position in the
	// file stream, so that it may be updated at the end with the actual event
	// count. Reserving space for five uint64_t's -- the number of events and
	// end time stamps.
	uint64_t event_count = 0;
	long size_pos = ftell(fout);
	ret = fwrite(&event_count, sizeof(uint64_t), 5, fout);
	if (ret < 1)
		die("Could not initialise event count: %s", strerror(errno));

	char padding[EVENT_PADDING_SIZE];
	bzero(padding, EVENT_PADDING_SIZE);

	fd_set readfds;
	struct timespec tout;
	replay_event_t rev;
	int32_t maxfd = 0;
	int32_t keydev = 0;
	int i;
	printf("recording...\n");

	errno = 0;
	signal(SIGINT, exitHandler);
	
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	while(1)
	{
		FD_ZERO(&readfds);
		FD_SET(STDIN_FILENO, &readfds);
		for (i=0; i < devices.num; i++)
			FD_SET(devices.fds[i], &readfds);

		/* wait for input */
		tout.tv_sec = delay;
		tout.tv_nsec = 0;

		ret = pselect(devices.max_fd + 1, &readfds, NULL, NULL, &tout, &old_sigset);

		if (EXIT){
			break;
		}
		if (errno == EINTR){
			break;
		}
		if (!ret){
			break;
		}

		if (wait_for_stdin && FD_ISSET(STDIN_FILENO, &readfds)) {
			// in this case the key down for the return key will be recorded
			// so we need to up the key up
			memset(&rev, 0, sizeof(rev));
			rev.dev_idx = keydev;
			rev.event.type = EV_KEY;
			rev.event.code = KEY_ENTER;
			rev.event.value = 0;
			gettimeofday(&rev.event.time, NULL);
			write_replay_event(fout, &rev);

			// syn
			memset(&rev, 0, sizeof(rev));
			rev.dev_idx = keydev;
			rev.event.type = EV_SYN;
			rev.event.code = 0;
			rev.event.value = 0;
			gettimeofday(&rev.event.time, NULL);
			write_replay_event(fout, &rev);

			dprintf("added fake return exiting...\n");
			break;
		}

		for (i = 0; i < devices.num; i++)
		{
			if (FD_ISSET(devices.fds[i], &readfds))
			{
				dprintf("got event from %s\n", devices.paths[i]);
				memset(&rev, 0, sizeof(rev));
				rev.dev_idx = i;
				ret = read(devices.fds[i], (void *)&rev.event, sizeof(rev.event));
				dprintf("%d event: type %d code %d value %d\n",
						(unsigned int)ret, rev.event.type, rev.event.code, rev.event.value);
				if (rev.event.type == EV_KEY && rev.event.code == KEY_ENTER && rev.event.value == 1)
					keydev = i;
				write_replay_event(fout, &rev);
				event_count++;
			}
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end_time);

	dprintf("Writing event count...\n");
	if ((ret = fseek(fout, size_pos, SEEK_SET)) == -1)
		die("Could not write event count: %s", strerror(errno));
	ret = fwrite(&event_count, sizeof(uint64_t), 1, fout);
	if (ret < 1)
		die("Could not write event count: %s", strerror(errno));
	dprintf("Writing recording timestamps...\n");
	uint64_t secs, usecs;
	secs = start_time.tv_sec;
	fwrite(&secs, sizeof(uint64_t), 1, fout);
	usecs = start_time.tv_nsec / 1000;
	fwrite(&usecs, sizeof(uint64_t), 1, fout);
	secs = end_time.tv_sec;
	fwrite(&secs, sizeof(uint64_t), 1, fout);
	usecs = end_time.tv_nsec / 1000;
	ret = fwrite(&usecs, sizeof(uint64_t), 1, fout);
	if (ret < 1)
		die("Could not write recording timestamps: %s\n", strerror(errno));

	fclose(fout);
	dprintf("Recording complete.\n");

	if (mode == GENERAL_MODE) {
		fini_general_input_devices(&devices);
	} else if (mode == GAMEPAD_MODE) {
		fini_gamepad_input_devices(&devices);
	} else {
		// Should never get here, as would have failed at the beginning
		die("Unexpected mode on finish");
	}
}

void dump(const char *filepath)
{
	int i, ret = 0;
	revent_recording_t recording;
	init_revent_recording(&recording);

	read_revent_recording_or_die(filepath, &recording);
	printf("recording version: %u\n", recording.desc.version);
	printf("recording type: %i\n", recording.desc.mode);
	printf("number of recorded events: %lu\n", recording.num_events);
	printf("start time: %ld.%06ld \n", recording.start_time.tv_sec, recording.start_time.tv_usec);
	printf("end time:   %ld.%06ld \n", recording.end_time.tv_sec, recording.end_time.tv_usec);

	printf("\n");
	if (recording.desc.mode == GENERAL_MODE) {
		printf("devices:\n");
		for (i = 0; i < recording.devices.num; i++) {
			printf("%2i: %s\n", i, recording.devices.paths[i]);
		}
	} else if (recording.desc.mode == GAMEPAD_MODE) {
		print_device_info(recording.gamepad_info);
	} else {
		die("Unexpected recording type: %d", recording.desc.mode);
	}

	printf("\nevents:\n");
	for (i =0; i < recording.num_events; i++) {
		printf("%ld.%06ld dev: %d type: %d code: %d value %d\n",
				recording.events[i].event.time.tv_sec,
				recording.events[i].event.time.tv_usec,
				recording.events[i].dev_idx,
				recording.events[i].event.type,
				recording.events[i].event.code,
				recording.events[i].event.value
		      );
	}

	fini_revent_recording(&recording);
}

void replay(const char *filepath)
{
	revent_recording_t recording;
	init_revent_recording(&recording);

	read_revent_recording_or_die(filepath, &recording);
	switch (recording.desc.mode) {
	case GENERAL_MODE:
		dprintf("Opening input devices for playback\n");
		open_general_input_devices_for_playback_or_die(&recording.devices);
		break;
	case GAMEPAD_MODE:
		dprintf("Creating gamepad playback device\n");
		open_gamepad_input_devices_for_playback_or_die(&recording.devices, recording.gamepad_info);
		break;
	default:
		die("Unexpected recording mod: %d", recording.desc.mode);
	}
	dprintf("Adjusting timestamps\n");
	adjust_timestamps(&recording);

	struct timeval start_time, now, desired_time, last_event_delta, delta;
	bzero(&last_event_delta, sizeof(struct timeval));
	gettimeofday(&start_time, NULL);

	int ret;
	uint64_t i = 0;
	dprintf("Starting payback\n");
	while (i < recording.num_events) {
		gettimeofday(&now, NULL);
		timeradd(&start_time, &last_event_delta, &desired_time);

		if (timercmp(&desired_time, &now, >)) {
			timersub(&desired_time, &now, &delta);
			useconds_t d = (useconds_t)delta.tv_sec * 1000000 + delta.tv_usec;
			dprintf("now %u.%u desiredtime %u.%u sleeping %u uS\n",
					(unsigned int)now.tv_sec,
					(unsigned int)now.tv_usec,
					(unsigned int)desired_time.tv_sec,
					(unsigned int)desired_time.tv_usec,
					d);
			usleep(d);
		}

		int32_t idx = (recording.events[i]).dev_idx;
		struct input_event ev = (recording.events[i]).event;
		while(!timercmp(&ev.time, &last_event_delta, !=)) {
			ret = write(recording.devices.fds[idx], &ev, sizeof(ev));
			if (ret != sizeof(ev))
				die("Could not replay event");
			dprintf("replayed event: type %d code %d value %d\n", ev.type, ev.code, ev.value);

			i++;
			if (i >= recording.num_events) {
				break;
			}
			idx = recording.events[i].dev_idx;
			ev = recording.events[i].event;
		}
		last_event_delta = ev.time;
	}

	timeradd(&start_time, &recording.end_time, &desired_time);
	gettimeofday(&now, NULL);
	if (timercmp(&desired_time, &now, >)) {
		timersub(&desired_time, &now, &delta);
		useconds_t d = (useconds_t)delta.tv_sec * 1000000 + delta.tv_usec;
		dprintf("now %u.%u recording end time %u.%u; sleeping %u uS\n",
				(unsigned int)now.tv_sec,
				(unsigned int)now.tv_usec,
				(unsigned int)desired_time.tv_sec,
				(unsigned int)desired_time.tv_usec,
				d);
		usleep(d);
	}
	else {
		dprintf("now %u.%u recording end time %u.%u; no need to sleep\n",
				(unsigned int)now.tv_sec,
				(unsigned int)now.tv_usec,
				(unsigned int)desired_time.tv_sec,
				(unsigned int)desired_time.tv_usec);
	}
	dprintf("Playback complete\n");

        if (recording.desc.mode == GAMEPAD_MODE)
		destroy_replay_device(recording.devices.fds[0]);
	fini_revent_recording(&recording);
}

void info(void)
{
	input_devices_t devices;
	init_input_devices(&devices);

	int ret = init_general_input_devices(&devices);
	if (ret) {
		die("Could not read input devices: %s", strerror(errno));
	}

	int i;
	device_info_t info;
	for (i = 0; i < devices.num; i++) {
		ret = get_device_info(devices.fds[i], &info);
		if (ret) {
			printf("Could not get info for %s: %s\n", devices.paths[i], strerror(errno));
			continue;
		}

		printf("DEVICE %d\n", i);
		printf("device path: %s\n", devices.paths[i]);
		printf("is gamepad: %s\n", is_gamepad(&info) ? "yes" : "no");
		print_device_info(&info);
		printf("\n");
	}

	fini_general_input_devices(&devices);
}

void usage()
{
	printf("usage:\n    revent [-h] [-v] COMMAND [OPTIONS] \n"
			"\n"
			"    Options:\n"
			"        -h  print this help message and quit.\n"
			"        -v  enable verbose output.\n"
			"\n"
			"    Commands:\n"
			"        record [-t SECONDS] [-d DEVICE] FILE\n"
			"            Record input event. stops after return on STDIN (or, optionally, \n"
			"            a fixed delay)\n"
			"\n"
			"                FILE       file into which events will be recorded.\n"
			"                -t SECONDS time, in seconds, for which to record events.\n"
			"                           if not specified, recording will continue until\n"
			"                           return key is pressed.\n"
			"                -d DEVICE  the number of the input device form which\n"
			"                           events will be recorded. If not specified, \n"
			"                           all available inputs will be used.\n"
			"                -s         Recording will not be stopped if there is \n"
			"                           input on STDIN.\n"
			"                -g         Record in \"gamepad\" mode. A gamepad must be \n"
			"                           connected to the device. The recording will only\n"
			"                           be done for the gamepad and other input devices\n"
			"                           will not be recorded. In addition to the input\n"
			"                           events, the information about the gamepad will\n"
			"                           also be stored in the recording. When this\n"
			"                           recording is played back, revent will first\n"
			"                           create a virtual gamepad device based on the\n"
			"                           stored info and the event will be played back\n"
			"                           into it. This type of recording should be more\n"
			"                           portable across different devices.\n"
			"\n"
			"        replay FILE\n"
			"            replays previously recorded events from the specified file.\n"
			"\n"
			"                FILE       file into which events will be recorded.\n"
			"\n"
			"        dump FILE\n"
			"            dumps the contents of the specified event log to STDOUT in\n"
			"            human-readable form.\n"
			"\n"
			"                FILE       event log which will be dumped.\n"
			"\n"
			"        info\n"
			"             shows info about each event char device\n"
			"\n"
			);
}

void revent_args_init(revent_args_t **rargs, int argc, char** argv)
{
	*rargs = malloc(sizeof(revent_args_t));
	revent_args_t *revent_args = *rargs;
	revent_args->command = INVALID_COMMAND;
	revent_args->mode = GENERAL_MODE;
	revent_args->record_time = INT_MAX;
	revent_args->device_number = -1;
	revent_args->file = NULL;

	int opt;
	while ((opt = getopt(argc, argv, "hgt:d:vs")) != -1)
	{
		switch (opt) {
			case 'h':
				usage();
				exit(0);
				break;
			case 'g':
				revent_args->mode = GAMEPAD_MODE;
				break;
			case 't':
				if (is_numeric(optarg)) {
					revent_args->record_time = atoi(optarg);
					dprintf("timeout: %d\n", revent_args->record_time);
				} else {
					die("-t parameter must be numeric; got %s.", optarg);
				}
				break;
			case 'd':
				if (is_numeric(optarg)) {
					revent_args->device_number = atoi(optarg);
					dprintf("device: %d\n", revent_args->device_number);
				} else {
					die("-d parameter must be numeric; got %s.", optarg);
				}
				break;
			case 'v':
				verbose = TRUE;
				break;
			case 's':
				wait_for_stdin = FALSE;
				break;

			default:
				die("Unexpected option: %c", opt);
		}
	}

	int next_arg = optind;
	if (next_arg == argc) {
		usage();
		die("Must specify a command.");
	}
	if (!strcmp(argv[next_arg], "record"))
		revent_args->command = RECORD_COMMAND;
	else if (!strcmp(argv[next_arg], "replay"))
		revent_args->command = REPLAY_COMMAND;
	else if (!strcmp(argv[next_arg], "dump"))
		revent_args->command = DUMP_COMMAND;
	else if (!strcmp(argv[next_arg], "info"))
		revent_args->command = INFO_COMMAND;
	else {
		usage();
		die("Unknown command -- %s", argv[next_arg]);
	}
	next_arg++;

	if (next_arg != argc) {
		revent_args->file = argv[next_arg];
		dprintf("file: %s\n", revent_args->file);
		next_arg++;
		if (next_arg != argc) {
			die("Trailling arguments (use -h for help).");
		}
	}

	if ((revent_args->command != RECORD_COMMAND) && (revent_args->record_time != INT_MAX)) {
		die("-t parameter is only valid for \"record\" command.");
	}
	if ((revent_args->command != RECORD_COMMAND) && (revent_args->device_number != -1)) {
		die("-d parameter is only valid for \"record\" command.");
	}
	if ((revent_args->command == INFO_COMMAND) && (revent_args->file != NULL)) {
		die("File path cannot be specified for \"info\" command.");
	}
	if (((revent_args->command == RECORD_COMMAND) || (revent_args->command == REPLAY_COMMAND))
			&& (revent_args->file == NULL)) {
		die("Must specify a file for recording/replaying (use -h for help).");
	}
}

int revent_args_close(revent_args_t *rargs)
{
	free(rargs);
	return 0;
}

int main(int argc, char** argv)
{
	int i;
	char *logfile = NULL;
	revent_args_t *rargs = NULL;

	revent_args_init(&rargs, argc, argv);

	switch(rargs->command) {
		case RECORD_COMMAND:
			record(rargs->file, rargs->record_time, rargs->mode);
			break;
		case REPLAY_COMMAND:
			replay(rargs->file);
			break;
		case DUMP_COMMAND:
			dump(rargs->file);
			break;
		case INFO_COMMAND:
			info();
			break;
		defaut:
			die("Unexpected revent command: %d", rargs->command);
	};

	revent_args_close(rargs);
	return 0;
}
