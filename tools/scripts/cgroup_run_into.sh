#!/bin/sh

DST_GRP=${1}
CMD=${2}

if [ -f $DST_GRP/cgroup.procs ]
then
  echo $$ > $DST_GRP/cgroup.procs
  $CMD
else
  echo "WARNING: couldn't find $DST_GRP cgroup!"
fi
