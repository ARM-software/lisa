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

"""Initialization for Android module"""

from lisa.android.screen import Screen
from lisa.android.system import System
from lisa.android.workload import Workload
from lisa.android.viewer import ViewerWorkload
from lisa.android.benchmark import LisaBenchmark

# Initialization of Android Workloads
import os
import sys

from glob import glob
from importlib import import_module

# Add workloads dir to system path
workloads_dir = os.path.dirname(os.path.abspath(__file__))
workloads_dir = os.path.join(workloads_dir, 'workloads')
sys.path.insert(0, workloads_dir)

for filepath in glob(os.path.join(workloads_dir, '*.py')):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # Ignore __init__ files
    if filename.startswith('__'):
        continue
    # Import workload module
    import_module(filename)
