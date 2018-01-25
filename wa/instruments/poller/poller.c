#include <fcntl.h>
#include <stdio.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>

volatile sig_atomic_t done = 0;
void term(int signum)
{
    done = 1;
}

void strip(char *s) {
    char *stripped_s = s;
    while(*s != '\0') {
        if(*s != ',' && *s != '\n') {
            *stripped_s++ = *s++;
        } else {
            ++s;
        }
    }
    *stripped_s = '\0';
}

typedef struct {
        int fd;
        char *path;
} poll_source_t;


int write_trace_marker(char *marker, int size)
{
        int ret;
        FILE *file;

        file = fopen("/sys/kernel/debug/tracing/trace_marker", "w");
        if (file == NULL) {
                return -errno;
        }

        ret = fwrite(marker, sizeof(char), size, file);

        fclose(file);
        return ret;
}

int main(int argc, char ** argv) {

    extern char *optarg;
    extern int optind;
    int c = 0;
    int show_help = 0;
    useconds_t interval = 1000000;
    char buf[1024];
    memset(buf, 0, sizeof(buf));
    struct timespec current_time;
    double time_float;
    char *labels;
    int labelCount = 0;
    int should_write_marker = 0;
    int ret;

    static char usage[] = "usage: %s [-h] [-m] [-t INTERVAL] FILE [FILE ...]\n"
                          "polls FILE(s) every INTERVAL microseconds and outputs\n"
                          "the results in CSV format including a timestamp to STDOUT\n"
                          "\n"
                          "    -h     Display this message\n"
                          "    -m     Insert a marker into ftrace at the time of the first\n"
                          "           sample. This marker may be used to align the timestamps\n"
                          "           produced by the poller with those of ftrace events.\n"
                          "    -t     The polling sample interval in microseconds\n"
                          "           Defaults to 1000000 (1 second)\n"
                          "    -l     Comma separated list of labels to use in the CSV\n"
                          "           output. This should match the number of files\n";


    //Handling command line arguments
    while ((c = getopt(argc, argv, "hmt:l:")) != -1)
    {
        switch(c) {
            case 'h':
            case '?':
            default:
                show_help = 1;
                break;
            case 'm':
                should_write_marker = 1;
		break;
            case 't':
                interval = (useconds_t)atoi(optarg);
                break;
            case 'l':
                labels = optarg;
                labelCount = 1;
                int i;
                for (i=0; labels[i]; i++)
                    labelCount += (labels[i] == ',');
        }
    }

    if (show_help) {
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    if (optind >= argc) {
        fprintf(stderr, "ERROR: %s: missing file path(s)\n", argv[0]);
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    int num_files = argc - optind;
    poll_source_t files_to_poll[num_files];

    if (labelCount && labelCount != num_files)
    {
        fprintf(stderr, "ERROR: %s: %d labels specified but %d files specified\n",
                argv[0], labelCount, num_files);
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    //Print headers and open files to poll
    printf("time");
    if(labelCount)
    {
        printf(",%s", labels);
    }
    int i;
    for (i = 0; i < num_files; i++)
    {
        files_to_poll[i].path = argv[optind + i];
        files_to_poll[i].fd = open(files_to_poll[i].path, O_RDONLY);
        if (files_to_poll[i].fd == -1) {
            fprintf(stderr, "ERROR: Could not open \"%s\", got: %s\n",
                    files_to_poll[i].path, strerror(errno));
            exit(2);
        }

        if(!labelCount) {
            printf(",%s", argv[optind + i]);
        }
    }
    printf("\n");

    //Setup SIGTERM handler
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = term;
    sigaction(SIGTERM, &action, NULL);

    //Poll files 
    int bytes_read = 0;
    while (!done) {
        clock_gettime(CLOCK_BOOTTIME, &current_time);
        if (should_write_marker) {
            ret = write_trace_marker("POLLER_START", 12);
            if (ret < 0) {
                fprintf(stderr, "ERROR writing trace marker: %s\n", strerror(ret));
                exit(ret);
            }
        }

        time_float = (double)current_time.tv_sec;
        time_float += ((double)current_time.tv_nsec)/1000/1000/1000;
        printf("%f", time_float);
        for (i = 0; i < num_files; i++) {
            lseek(files_to_poll[i].fd, 0, SEEK_SET);
            bytes_read = read(files_to_poll[i].fd, buf, 1024);

            if (bytes_read < 0) {
                fprintf(stderr, "WARNING: Read nothing from \"%s\"\n",
                        files_to_poll[i].path);
                printf(",");
                continue;
            }

            strip(buf);
            printf(",%s", buf);
            buf[0] = '\0'; // "Empty" buffer
        }
        printf("\n");
        usleep(interval);
    }

    //Close files
    for (i = 0; i < num_files; i++)
    {
        close(files_to_poll[i].fd);
    }
    exit(0);
}
