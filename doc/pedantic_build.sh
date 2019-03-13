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

make SPHINXOPTS='-n --no-color' html 2>doc_build.log
log=$(cat doc_build.log)
warns=$(cat doc_build.log | grep WARNING --color=always | grep -E -w -v "$IGNORE_PATTERN")
ignored_warns=$(cat doc_build.log | grep WARNING | grep --color=always -E -w "$IGNORE_PATTERN")

echo

echo "Documentation build warnings:"
echo
if [ -n "$warns" ]; then
    echo "Warnings to fix:"
    echo
    echo "$warns"
    echo
    ret=1
else
    echo "Ignored warnings:"
    echo
    echo "$ignored_warns"
    ret=0
fi

echo
echo "Sphinx log:"
echo
echo "$log"
exit $ret
