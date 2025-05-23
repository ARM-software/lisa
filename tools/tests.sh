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

# Script run by github CI, VM setup check etc.


# Some commands are allowed to fail in init_env, e.g. to probe for installed
# tools. However, the overall script has to succeed.
source init_env || exit 1

# Failing commands will make the script return with an error code
set -e

echo "Starting self tests ..."
# Use timeout to send SIGINT if the test hangs, so we get a proper backtrace
# before github action kills the whole thing without any useful log
timeout -s INT 1h python3 -m pytest -vv

echo "Starting exekall self tests"
exekall run "$LISA_HOME/tools/exekall/src/exekall/_tests"

echo "Starting documentation pedantic build ..."
lisa-doc-build

MANPAGE_PYTHON_VERSION="3.11"

echo "Checking that the man pages are up to date ..."
if python3 --version | grep -F "$MANPAGE_PYTHON_VERSION" ; then
    if ! git diff --exit-code doc/man1/; then
        echo "Please regenerate man pages in doc/man1 and commit them"
        exit 1
    fi
else
    echo "Skipped man page check as it only runs for $MANPAGE_PYTHON_VERSION"
fi

# FIXME: cbindgen does not work on our source anymore due to that issue:
# https://github.com/mozilla/cbindgen/issues/993
#
# echo "Checking that the kernel module Rust bindings are up to date ..."
# (
	# cd "$LISA_HOME/lisa/_assets/kmodules/lisa/" || exit 1
	# make_rust_bindings="make refresh-rust-bindings"

	# $make_rust_bindings
	# if ! git diff --exit-code .; then
		# echo "Please regenerate Rust bindings: $make_rust_bindings"
		# exit 1
	# fi
# )

