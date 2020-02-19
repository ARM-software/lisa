#!/bin/sh
BUSYBOX=$1
OUTFILE=$2
PERIOD=$3
STOP_SIGNAL_FILE=$4

if [ "$#" != "4" ]; then
    echo "USAGE: gather-load.sh BUSYBOX OUTFILE PERIOD STOP_SIGNAL_FILE"
    exit 1
fi

echo "timestamp,user,nice,system,idle,iowait,irq,softirq,steal,guest,guest_nice" > $OUTFILE
while true; do
    echo -n $(${BUSYBOX} date -Iseconds) >> $OUTFILE
    ${BUSYBOX} cat /proc/stat | ${BUSYBOX} head -n 1 | \
        ${BUSYBOX} cut -d ' ' -f 2- | ${BUSYBOX} sed 's/ /,/g' >> $OUTFILE
    if [ -f $STOP_SIGNAL_FILE ]; then
        rm $STOP_SIGNAL_FILE
        break
    else
        sleep $PERIOD
    fi
done
