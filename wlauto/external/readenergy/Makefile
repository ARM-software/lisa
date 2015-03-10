# To build:
#
# CROSS_COMPILE=aarch64-linux-gnu- make
#
CROSS_COMPILE?=aarch64-linux-gnu-
CC=$(CROSS_COMPILE)gcc
CFLAGS='-Wl,-static -Wl,-lc'

readenergy: readenergy.c
	$(CC) $(CFLAGS) readenergy.c -o readenergy
	cp readenergy ../../instrumentation/juno_energy/readenergy
