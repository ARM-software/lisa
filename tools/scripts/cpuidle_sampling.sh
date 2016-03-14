#!/bin/bash

# CPU to monitor
CPU=${1:-0}
# Sampling time
SLEEP=${2:-1}
# Samples to collect
COUNT=${3:-3}

# Enter CPU's sysfs
cd /sys/devices/system/cpu

# Initial C-State residencies counter
ISC=$(find cpu0/cpuidle -name "state*" | wc -l)
for I in $(seq 0 $((ISC-1))); do
	LCS[$I]=`cat cpu$CPU/cpuidle/state$I/usage`
done

# Dump header
printf "#%13s " "Time"
for I in $(seq 0 $((ISC-1))); do
  printf "%14s " "idle$I"
done
echo

# Sampling loop
for I in $(seq $COUNT); do

	sleep $SLEEP

	# Dump CPU C-State residencies
	now=$(date +%s)
	printf "%14d " $now
	for I in $(seq 0 $((ISC-1))); do
		U=`cat cpu$CPU/cpuidle/state$I/usage`
		CCS=$(($U - ${LCS[$I]}))
		printf "%14d " $CCS
		LCS[$I]=$U
	done
	echo


done

# vim: ts=2
