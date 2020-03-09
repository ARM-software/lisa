CFLAGS=-Wall --pedantic-errors -O2 -static

all: get_clock_monotonic

get_clock_monotonic: get_clock_monotonic.c
	$(CC) $(CFLAGS) $^ -o $@
