#!/bin/bash

FREQS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq)
CPU=0; for F in $FREQS; do
	echo "cpu_frequency:        state=$F cpu_id=$CPU" > /sys/kernel/debug/tracing/trace_marker
	let CPU++
done

