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

"""Initialization for wlgen"""

import pkg_resources
from wlgen.workload import Workload
from wlgen.rta import RTA, Ramp, Step, Pulse, Periodic, RunAndSync
from wlgen.perf_bench import PerfMessaging, PerfPipe

try:
    __version__ = pkg_resources.get_distribution("wlgen").version
except pkg_resources.DistributionNotFound:
    __version__ = "local"
