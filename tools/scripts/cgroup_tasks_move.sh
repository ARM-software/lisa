#!/bin/sh
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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


