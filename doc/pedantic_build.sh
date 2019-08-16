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

ret=0

##############################
# Build the main documentation
##############################

# The documentation of those modules causes some issues,
# be permissive for warnings related to them
IGNORE_PATTERN="devlib|bart|wa|docutils.parsers|gi.repository"

make SPHINXOPTS='-n --no-color' html 2>doc_build.log
log=$(cat doc_build.log)
warns=$(cat doc_build.log | grep -E '(WARNING|Exception occurred)' --color=always | grep -E -w -v "$IGNORE_PATTERN")
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
fi

echo
echo "Sphinx log:"
echo
echo "$log"

echo
echo "Building man pages"
echo

#################
# Build man pages
#################

docs=(
    "tools/bisector/doc/man/"
    "tools/exekall/doc/man/"
    "doc/lisa_shell/man"
)

for doc in "${docs[@]}"; do
    sphinx-build "$LISA_HOME/$doc" "$LISA_HOME/doc/man1" -Enab man || ret=2
done

exit $ret
