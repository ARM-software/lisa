/*    Copyright 2013-2017 ARM Limited
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


#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <time.h>

const int MAX_CPUS = 8;
const int DEFAULT_ITERATIONS = 1000;
const int DEFAULT_BUFFER_SIZE = 1024 * 1024 * 5;

int set_affinity(size_t cpus_size, int* cpus)
{
	int i;
	int mask = 0;

	for(i = 0; i < cpus_size; ++i)
	{
		mask |= 1 << cpus[i];
	}
	
	return syscall(__NR_sched_setaffinity, 0, sizeof(mask), &mask);
}

int main(int argc, char** argv)
{
	int cpus[MAX_CPUS];
	int next_cpu = 0;
	int iterations = DEFAULT_ITERATIONS;
	int buffer_size = DEFAULT_BUFFER_SIZE;
	
	int c;
	while ((c = getopt(argc, argv, "i:c:s:")) != -1)
		switch (c)
		{
		case 'c':
			cpus[next_cpu++] = atoi(optarg);
			if (next_cpu == MAX_CPUS)
			{
				fprintf(stderr, "Max CPUs exceeded.");
				abort();
			}
			break;
		case 'i':
			iterations = atoi(optarg);
			break;
		case 's':
			buffer_size = atoi(optarg);
			break;
		default:
			abort();
			break;
		}

	int ret;
	if (next_cpu != 0)
		if (ret = set_affinity(next_cpu, cpus))
		{
			fprintf(stderr, "sched_setaffinity returnred %i.", ret);
			abort();
		}
	
	char* source  = malloc(buffer_size);
	char* dest = malloc(buffer_size);

	struct timespec before, after;
	if (clock_gettime(CLOCK_MONOTONIC, &before))
	{
	 	fprintf(stderr, "Could not get start time.");
		abort();
	}

	int i;
	for (i = 0; i < iterations; ++i)
	{
		memcpy(dest, source, buffer_size);
	}

	if (clock_gettime(CLOCK_MONOTONIC, &after))
	{
	 	fprintf(stderr, "Could not get end time.");
		abort();
	}

	free(dest);
	free(source);

	long delta_sec =  (long)(after.tv_sec - before.tv_sec);
	long delta_nsec = after.tv_nsec - before.tv_nsec;
	double delta = (double)delta_sec + delta_nsec / 1e9;
	printf("Total time: %f s\n", delta);
	printf("Bandwidth: %f MB/s\n", buffer_size / delta * iterations / 1e6);

	return 0;
}
