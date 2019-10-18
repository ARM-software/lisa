#!/bin/bash
#    Copyright 2018 ARM Limited
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


# CD into build dir if possible - allows building from any directory
script_path='.'
if `readlink -f $0 &>/dev/null`; then
    script_path=`readlink -f $0 2>/dev/null`
fi
script_dir=`dirname $script_path`
cd $script_dir

# Ensure gradelw exists before starting
if [[ ! -f gradlew ]]; then
    echo 'gradlew file not found! Check that you are in the right directory.'
    exit 9
fi

# Copy base class library from wa dist
libs_dir=app/libs
base_class=`python3 -c "import os, wa; print(os.path.join(os.path.dirname(wa.__file__), 'framework', 'uiauto', 'uiauto.aar'))"`
mkdir -p $libs_dir
cp $base_class $libs_dir

# Build and return appropriate exit code if failed
# gradle build
./gradlew clean :app:assembleDebug
exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    echo "ERROR: 'gradle build' exited with code $exit_code"
    exit $exit_code
fi

# If successful move APK file to workload folder (overwrite previous)
package=com.arm.wa.uiauto.pcmark
rm -f ../$package
if [[ -f app/build/apk/$package.apk ]]; then
    cp app/build/apk/$package.apk ../$package.apk
else
    echo 'ERROR: UiAutomator apk could not be found!'
    exit 9
fi
