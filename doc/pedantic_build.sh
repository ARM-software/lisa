# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

#!/bin/bash

# The documentation of those modules causes some issues,
# be permissive for warnings related to them
IGNORE_PATTERN="devlib|bart|wa|exekall.engine"

make SPHINXOPTS='-n' html 2>doc_build.log
all_warns=$(cat doc_build.log)
warns=$(cat doc_build.log | grep WARNING | grep -E -v "$IGNORE_PATTERN")

echo

if [ ! -z "$warns" ]; then
    echo "Documentation build warnings:"
    echo
    echo "Warnings to fix:"
    echo
    echo "$warns"
    echo
    echo "Sphinx log:"
    echo
    echo "$all_warns"
    exit 1
elif [ ! -z "$all_warns" ]; then
    echo "Ignored warnings:"
    echo
    echo "$all_warns"
fi
