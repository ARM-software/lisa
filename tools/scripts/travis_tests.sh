#!/bin/bash
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

# Script run by Travis. It is mostly a workaround for Travis inability to
# correctly handle environment variable set in sourced scripts.

# Failing commands will make the script return with an error code
set -e

illegal_location="external/"
illegal_commits=$(find "$illegal_location" -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 git log --no-merges --oneline)

if [[ -n "$illegal_commits" ]]; then
    echo -e "The following commits are touching $illegal_location, which is not allowed apart from updates:\n$illegal_commits"
    exit 1
fi;

source init_env

echo "Starting nosetests ..."
python3 -m nose

echo "Starting documentation pedantic build ..."
cd doc/ && ./pedantic_build.sh
