# CROSS_COMPILE=aarch64-linux-gnu- make
#
CC=gcc

ifdef DEBUG
	CFLAGS=-static -lc -g
else
	CFLAGS=-static -lc -O2
endif

poller: poller.c
	$(CROSS_COMPILE)$(CC) $(CFLAGS) poller.c -o poller

clean:
	rm -rf poller

.PHONY: clean
