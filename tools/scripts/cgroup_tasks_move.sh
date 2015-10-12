#!/bin/sh

SRC_GRP=${1}
DST_GRP=${2}
GREP_EXCLUSE=${3:-''}

cat $SRC_GRP/tasks | while read TID; do
  echo $TID > $DST_GRP/cgroup.procs
done

[ "$GREP_EXCLUSE" = "" ] && exit 0

PIDS=`ps | grep $GREP_EXCLUSE | awk '{print $2}'`
PIDS=`echo $PIDS`
echo "PIDs to save: [$PIDS]"
for TID in $PIDS; do
  CMDLINE=`cat /proc/$TID/cmdline`
  echo "$TID : $CMDLINE"
  echo $TID > $SRC_GRP/cgroup.procs
done


