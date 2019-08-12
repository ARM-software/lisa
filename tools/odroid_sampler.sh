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

SYSFS_BASE="/sys/devices/12c60000.i2c/i2c-4/i2c-dev/i2c-4/device/"
SYSFS_ARM=$SYSFS_BASE"/4-0040"
SYSFS_KFC=$SYSFS_BASE"/4-0045"

if [ $# -lt 2 ]; then
    echo "Usage: $0 samples period_s [arm|kfc]"
    exit 1
fi

SAMPLES=$1
PERIOD=$2
DEVICE=${3:-"arm"}

case $DEVICE in
"arm")
    SYSFS_ENABLE=$SYSFS_ARM"/enable"
    SYSFS_W=$SYSFS_ARM"/sensor_W"
    ;;
"kfc")
    SYSFS_ENABLE=$SYSFS_KFC"/enable"
    SYSFS_W=$SYSFS_KFC"/sensor_W"
    ;;
esac

echo "Samping $SAMPLES time, every $PERIOD [s]:"
echo "   $SYSFS_W"

rm samples_w.txt 2>/dev/null
echo 1 > $SYSFS_ENABLE
sleep 1

while [ 1 ]; do
    sleep $PERIOD
    cat $SYSFS_W >> samples_w.txt
    SAMPLES=$((SAMPLES-1))
    [ $SAMPLES -eq 0 ] && break
done

echo 0 > $SYSFS_ENABLE
