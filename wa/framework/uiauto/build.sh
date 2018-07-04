#!/bin/bash
#    Copyright 2013-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set -e

# Ensure gradelw exists before starting
if [[ ! -f gradlew ]]; then
    echo 'gradlew file not found! Check that you are in the right directory.'
    exit 9
fi

# Build and return appropriate exit code if failed
./gradlew clean :app:assembleDebug
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "ERROR: 'gradle build' exited with code $exit_code"
    exit $exit_code
fi

cp app/build/outputs/aar/app-debug.aar ./uiauto.aar
