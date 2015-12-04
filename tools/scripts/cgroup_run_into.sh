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

# Control groups mount point
CGMOUNT=${CGMOUNT:-/sys/fs/cgroup}
# The control group we want to run into
CGP=${1}
# The command to run
CMD=${2}

# Check if the required CGroup exists
find $CGMOUNT -type d | grep $CGP &>/dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: could not find any $CGP cgroup under $CGMOUNT"
  exit 1
fi

find $CGMOUNT -type d | grep $CGP | \
while read CGPATH; do
    # Move this shell into that control group
    echo $$ > $CGPATH/cgroup.procs
    echo "Moving task into $CGPATH"
done

# Execute the command
$CMD
