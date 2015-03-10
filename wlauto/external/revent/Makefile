# CROSS_COMPILE=aarch64-linux-gnu- make
#
CC=gcc
CFLAGS=-static -lc

revent: revent.c
	$(CROSS_COMPILE)$(CC) $(CFLAGS) revent.c -o revent

clean:
	rm -rf revent

.PHONY: clean
