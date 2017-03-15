# To build:
#
# CROSS_COMPILE=aarch64-linux-gnu- make
#
CROSS_COMPILE?=aarch64-linux-gnu-
CC=$(CROSS_COMPILE)gcc
CFLAGS=-static -lc

readenergy: readenergy.c
	$(CC) $(CFLAGS) readenergy.c -o readenergy
	mv readenergy ../../devlib/bin/arm64/readenergy
