#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, ARM Limited and contributors.
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
# Prepare the groundwork and run tests/test_target.py on the Docker image.
#

set -eu

ANDROID_HOME="/devlib/tools/android/android-sdk-linux"
export ANDROID_HOME
export ANDROID_USER_HOME="${ANDROID_HOME}/.android"
export ANDROID_EMULATOR_HOME="${ANDROID_HOME}/.android"
export PATH=${ANDROID_HOME}/platform-tools/:${PATH}

EMULATOR="${ANDROID_HOME}/emulator/emulator"
EMULATOR_ARGS="-no-window -no-snapshot -memory 2048"
${EMULATOR} -avd devlib-p6-12 ${EMULATOR_ARGS} &
${EMULATOR} -avd devlib-p6-14 ${EMULATOR_ARGS} &
${EMULATOR} -avd devlib-chromeos ${EMULATOR_ARGS} &

echo "Waiting 30 seconds for Android virtual devices to finish boot up..."
sleep 30

cd /devlib
cp -f tools/docker/test_config.yml tests/
python3 -m pytest --log-cli-level DEBUG ./tests/test_target.py
