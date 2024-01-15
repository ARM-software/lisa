CFLAGS=-Wall --pedantic-errors -O2 -static

all: get_clock_boottime

get_clock_boottime: get_clock_boottime.c
	$(CC) $(CFLAGS) $^ -o $@
