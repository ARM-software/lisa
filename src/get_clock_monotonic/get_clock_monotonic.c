#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    int ret;
    struct timespec tp;

    ret = clock_gettime(CLOCK_MONOTONIC, &tp);
    if (ret) {
        perror("clock_gettime()");
        return EXIT_FAILURE;
    }

    printf("%ld.%ld\n", tp.tv_sec, tp.tv_nsec);

    return EXIT_SUCCESS;
}
