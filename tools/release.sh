#!/bin/bash
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, ARM Limited and contributors.
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

TAG=master
LISA_HOME=./release

export LISA_HOME=$(readlink -f "$LISA_HOME")

git clone https://github.com/ARM-Software/lisa.git "$LISA_HOME"
git -C "$LISA_HOME" checkout "$TAG"

vagrant box update
vagrant destroy -f


VAGRANT_FORWARD_JUPYTER=0 vagrant up &&
vagrant ssh -c 'source start-lisa && pip freeze > release_requirements.txt && $LISA_HOME/tools/tests.sh'
