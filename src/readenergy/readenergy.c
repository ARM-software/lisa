/*    Copyright 2014-2015 ARM Limited
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
 * readenergy.c 
 *
 * Reads APB energy registers in Juno and outputs the measurements (converted to appropriate units).
 *
*/
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// The following values obtained from Juno TRM 2014/03/04 section 4.5

// Location of APB registers in memory
#define APB_BASE_MEMORY 0x1C010000
// APB energy counters start at offset 0xD0 from the base APB address.
#define BASE_INDEX 0xD0 / 4
// the one-past last APB counter
#define APB_SIZE 0x120

// Masks specifying the bits that contain the actual counter values
#define CMASK 0xFFF
#define VMASK 0xFFF
#define PMASK 0xFFFFFF

// Sclaing factor (divisor) or getting measured values from counters
#define SYS_ADC_CH0_PM1_SYS_SCALE 761
#define SYS_ADC_CH1_PM2_A57_SCALE 381
#define SYS_ADC_CH2_PM3_A53_SCALE 761
#define SYS_ADC_CH3_PM4_GPU_SCALE 381
#define SYS_ADC_CH4_VSYS_SCALE 1622
#define SYS_ADC_CH5_VA57_SCALE 1622
#define SYS_ADC_CH6_VA53_SCALE 1622
#define SYS_ADC_CH7_VGPU_SCALE 1622
#define SYS_POW_CH04_SYS_SCALE (SYS_ADC_CH0_PM1_SYS_SCALE * SYS_ADC_CH4_VSYS_SCALE)
#define SYS_POW_CH15_A57_SCALE (SYS_ADC_CH1_PM2_A57_SCALE * SYS_ADC_CH5_VA57_SCALE)
#define SYS_POW_CH26_A53_SCALE (SYS_ADC_CH2_PM3_A53_SCALE * SYS_ADC_CH6_VA53_SCALE)
#define SYS_POW_CH37_GPU_SCALE (SYS_ADC_CH3_PM4_GPU_SCALE * SYS_ADC_CH7_VGPU_SCALE)
#define SYS_ENM_CH0_SYS_SCALE 12348030000
#define SYS_ENM_CH1_A57_SCALE 6174020000
#define SYS_ENM_CH0_A53_SCALE 12348030000
#define SYS_ENM_CH0_GPU_SCALE 6174020000

// Original values prior to re-callibrations.
/*#define SYS_ADC_CH0_PM1_SYS_SCALE 819.2*/
/*#define SYS_ADC_CH1_PM2_A57_SCALE 409.6*/
/*#define SYS_ADC_CH2_PM3_A53_SCALE 819.2*/
/*#define SYS_ADC_CH3_PM4_GPU_SCALE 409.6*/
/*#define SYS_ADC_CH4_VSYS_SCALE 1638.4*/
/*#define SYS_ADC_CH5_VA57_SCALE 1638.4*/
/*#define SYS_ADC_CH6_VA53_SCALE 1638.4*/
/*#define SYS_ADC_CH7_VGPU_SCALE 1638.4*/
/*#define SYS_POW_CH04_SYS_SCALE (SYS_ADC_CH0_PM1_SYS_SCALE * SYS_ADC_CH4_VSYS_SCALE)*/
/*#define SYS_POW_CH15_A57_SCALE (SYS_ADC_CH1_PM2_A57_SCALE * SYS_ADC_CH5_VA57_SCALE)*/
/*#define SYS_POW_CH26_A53_SCALE (SYS_ADC_CH2_PM3_A53_SCALE * SYS_ADC_CH6_VA53_SCALE)*/
/*#define SYS_POW_CH37_GPU_SCALE (SYS_ADC_CH3_PM4_GPU_SCALE * SYS_ADC_CH7_VGPU_SCALE)*/
/*#define SYS_ENM_CH0_SYS_SCALE 13421772800.0*/
/*#define SYS_ENM_CH1_A57_SCALE 6710886400.0*/
/*#define SYS_ENM_CH0_A53_SCALE 13421772800.0*/
/*#define SYS_ENM_CH0_GPU_SCALE 6710886400.0*/

// Ignore individual errors but if see too many, abort.
#define ERROR_THRESHOLD 10

// Default counter poll period (in milliseconds).
#define DEFAULT_PERIOD 100

// Default duration for the instrument execution (in seconds); 0 means 'forever'
#define DEFAULT_DURATION 0

// A single reading from the energy meter. The values are the proper readings converted
// to appropriate units (e.g. Watts for power); they are *not* raw counter values.
struct reading
{
	double sys_adc_ch0_pm1_sys;
	double sys_adc_ch1_pm2_a57;
	double sys_adc_ch2_pm3_a53;
	double sys_adc_ch3_pm4_gpu;
	double sys_adc_ch4_vsys;
	double sys_adc_ch5_va57;
	double sys_adc_ch6_va53;
	double sys_adc_ch7_vgpu;
	double sys_pow_ch04_sys;
	double sys_pow_ch15_a57;
	double sys_pow_ch26_a53;
	double sys_pow_ch37_gpu;
	double sys_enm_ch0_sys;
	double sys_enm_ch1_a57;
	double sys_enm_ch0_a53;
	double sys_enm_ch0_gpu;
};

static inline uint64_t join_64bit_register(uint32_t *buffer, int index)
{
	uint64_t result = 0;
	result |= buffer[index];
	result |= (uint64_t)(buffer[index+1]) << 32;
	return result;
}

int nsleep(const struct timespec *req, struct timespec *rem)
{
	struct timespec temp_rem;
	if (nanosleep(req, rem) == -1)
	{
		if (errno == EINTR)
		{
			nsleep(rem, &temp_rem);
		}
		else
		{
			return errno;
		}
	}
	else
	{
		return 0;
	}
}
 
void print_help()
{
	fprintf(stderr, "Usage: readenergy [-t PERIOD] [-o OUTFILE]\n\n"
			"Read Juno energy counters every PERIOD milliseconds, writing them\n"
			"to OUTFILE in CSV format either until SIGTERM is received OR\n"
			"till the specified duration elapsed.\n"
			"If OUTFILE is not specified, stdout will be used.\n\n"
			"Parameters:\n"
			"	PERIOD is the counter poll period in milliseconds.\n"
			"	       (Defaults to 100 milliseconds.)\n"
			"	DURATION is the duration before execution terminates.\n"
			"		(Defaults to 0 seconds, meaning run till user\n"
			"		terminates execution.\n"
			"	OUTFILE is the output file path\n");
}

// debugging only...
inline void dprint(char *msg)
{
	fprintf(stderr, "%s\n", msg);
	sync();
}

// -------------------------------------- config ----------------------------------------------------

struct config
{
	struct timespec period;
	char *output_file;
	long duration_in_sec;
};

void config_init_period_from_millis(struct config *this, long millis)
{
	this->period.tv_sec = (time_t)(millis / 1000);
	this->period.tv_nsec = (millis % 1000) * 1000000;
}

void config_init(struct config *this, int argc, char *argv[])
{
	this->output_file = NULL;
	config_init_period_from_millis(this, DEFAULT_PERIOD);
	this->duration_in_sec = DEFAULT_DURATION;

	int opt;
	while ((opt = getopt(argc, argv, "ht:o:d:")) != -1)
	{
		switch(opt)
		{
			case 't':
				config_init_period_from_millis(this, atol(optarg));
				break;
			case 'o':
				this->output_file = optarg;
				break;
			case 'd':
				this->duration_in_sec = atol(optarg);
				break;
			case 'h':
				print_help();
				exit(EXIT_SUCCESS);
				break;
			default:
				fprintf(stderr, "ERROR: Unexpected option %s\n\n", opt);
				print_help();
				exit(EXIT_FAILURE);
		}
	}
}

// -------------------------------------- /config ---------------------------------------------------

// -------------------------------------- emeter ----------------------------------------------------

struct emeter
{
	int fd;
	FILE *out;
	void *mmap_base;
};

void emeter_init(struct emeter *this, char *outfile)
{
	if(outfile)
	{
		this->out = fopen(outfile, "w");
		if (this->out == NULL)
		{
			fprintf(stderr, "ERROR: Could not open output file %s; got %s\n", outfile, strerror(errno));
			exit(EXIT_FAILURE);
		}
	} else {
		this->out = stdout;
	}
        this->fd = open("/dev/mem", O_RDONLY);
        if(this->fd < 0)
        {
                fprintf(stderr, "ERROR: Can't open /dev/mem; got %s\n", strerror(errno));
		fclose(this->out);
		exit(EXIT_FAILURE);
        }

	this->mmap_base = mmap(NULL, APB_SIZE, PROT_READ, MAP_SHARED, this->fd, APB_BASE_MEMORY);
	if (this->mmap_base == MAP_FAILED)
	{
		fprintf(stderr, "ERROR: mmap failed; got %s\n", strerror(errno));
		close(this->fd);
		fclose(this->out);
		exit(EXIT_FAILURE);
	}

	if(this->out) {
		fprintf(this->out, "sys_current,a57_current,a53_current,gpu_current,"
				   "sys_voltage,a57_voltage,a53_voltage,gpu_voltage,"
				   "sys_power,a57_power,a53_power,gpu_power,"
				   "sys_energy,a57_energy,a53_energy,gpu_energy\n");
	}
}

void emeter_read_measurements(struct emeter *this, struct reading *reading)
{
	uint32_t *buffer = (uint32_t *)this->mmap_base;
	reading->sys_adc_ch0_pm1_sys = (double)(CMASK & buffer[BASE_INDEX+0]) / SYS_ADC_CH0_PM1_SYS_SCALE;
	reading->sys_adc_ch1_pm2_a57 = (double)(CMASK & buffer[BASE_INDEX+1]) / SYS_ADC_CH1_PM2_A57_SCALE;
	reading->sys_adc_ch2_pm3_a53 = (double)(CMASK & buffer[BASE_INDEX+2]) / SYS_ADC_CH2_PM3_A53_SCALE;
	reading->sys_adc_ch3_pm4_gpu = (double)(CMASK & buffer[BASE_INDEX+3]) / SYS_ADC_CH3_PM4_GPU_SCALE;
	reading->sys_adc_ch4_vsys = (double)(VMASK & buffer[BASE_INDEX+4]) / SYS_ADC_CH4_VSYS_SCALE;
	reading->sys_adc_ch5_va57 = (double)(VMASK & buffer[BASE_INDEX+5]) / SYS_ADC_CH5_VA57_SCALE;
	reading->sys_adc_ch6_va53 = (double)(VMASK & buffer[BASE_INDEX+6]) / SYS_ADC_CH6_VA53_SCALE;
	reading->sys_adc_ch7_vgpu = (double)(VMASK & buffer[BASE_INDEX+7]) / SYS_ADC_CH7_VGPU_SCALE;
	reading->sys_pow_ch04_sys = (double)(PMASK & buffer[BASE_INDEX+8]) / SYS_POW_CH04_SYS_SCALE;
	reading->sys_pow_ch15_a57 = (double)(PMASK & buffer[BASE_INDEX+9]) / SYS_POW_CH15_A57_SCALE;
	reading->sys_pow_ch26_a53 = (double)(PMASK & buffer[BASE_INDEX+10]) / SYS_POW_CH26_A53_SCALE;
	reading->sys_pow_ch37_gpu = (double)(PMASK & buffer[BASE_INDEX+11]) / SYS_POW_CH37_GPU_SCALE;
	reading->sys_enm_ch0_sys = (double)join_64bit_register(buffer, BASE_INDEX+12) / SYS_ENM_CH0_SYS_SCALE;
	reading->sys_enm_ch1_a57 = (double)join_64bit_register(buffer, BASE_INDEX+14) / SYS_ENM_CH1_A57_SCALE;
	reading->sys_enm_ch0_a53 = (double)join_64bit_register(buffer, BASE_INDEX+16) / SYS_ENM_CH0_A53_SCALE;
	reading->sys_enm_ch0_gpu = (double)join_64bit_register(buffer, BASE_INDEX+18) / SYS_ENM_CH0_GPU_SCALE;
}

void emeter_take_reading(struct emeter *this)
{
	static struct reading reading;
	int error_count = 0;
	emeter_read_measurements(this, &reading);
	int ret = fprintf(this->out, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
			reading.sys_adc_ch0_pm1_sys,
			reading.sys_adc_ch1_pm2_a57,
			reading.sys_adc_ch2_pm3_a53,
			reading.sys_adc_ch3_pm4_gpu,
			reading.sys_adc_ch4_vsys,
			reading.sys_adc_ch5_va57,
			reading.sys_adc_ch6_va53,
			reading.sys_adc_ch7_vgpu,
			reading.sys_pow_ch04_sys,
			reading.sys_pow_ch15_a57,
			reading.sys_pow_ch26_a53,
			reading.sys_pow_ch37_gpu,
			reading.sys_enm_ch0_sys,
			reading.sys_enm_ch1_a57,
			reading.sys_enm_ch0_a53,
			reading.sys_enm_ch0_gpu);
	if (ret < 0)
	{
		fprintf(stderr, "ERROR: while writing a meter reading: %s\n", strerror(errno));
		if (++error_count > ERROR_THRESHOLD)
			exit(EXIT_FAILURE);
	}
}

void emeter_finalize(struct emeter *this)
{
	if (munmap(this->mmap_base, APB_SIZE) == -1) 
	{
		// Report the error but don't bother doing anything else, as we're not gonna do 
		// anything with emeter after this point anyway.
		fprintf(stderr, "ERROR: munmap failed; got %s\n", strerror(errno));
	}
	close(this->fd);
	fclose(this->out);
}

// -------------------------------------- /emeter ----------------------------------------------------

volatile int done = 0;

void term_handler(int signum)
{
	done = 1;
}

void sigalrm_handler(int signum)
{
	done = 1;
}


int main(int argc, char *argv[])
{
	struct sigaction action;
	memset(&action, 0, sizeof(struct sigaction));
	action.sa_handler = term_handler;
	sigaction(SIGTERM, &action, NULL);

	struct config config;
	struct emeter emeter;
	config_init(&config, argc, argv);
	emeter_init(&emeter, config.output_file);

	if (0 != config.duration_in_sec)
	{
		/*Set the alarm with the duration from use only if a non-zero value is specified
		  else it will run forever until SIGTERM signal received from user*/
		/*Set the signal handler first*/
		signal(SIGALRM, sigalrm_handler);
		/*Now set the alarm for the duration specified by the user*/
		alarm(config.duration_in_sec);

	}

	if(config.output_file)
	{
		struct timespec remaining;
		while (!done)
		{
			emeter_take_reading(&emeter);
			nsleep(&config.period, &remaining);
		}
	} else 	{
		emeter_take_reading(&emeter);
	}

	emeter_finalize(&emeter);
	return EXIT_SUCCESS;
}
