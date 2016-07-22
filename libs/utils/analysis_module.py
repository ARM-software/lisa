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

class AnalysisModule(object):

    def __init__(self, trace):
        """
        Support for CPUs Signals Analysis
        """
        self._trace = trace
        self._platform = trace.platform
        self._tasks = trace.tasks
        self._data_dir = trace.data_dir

        self._dfg_trace_event = trace._dfg_trace_event

        trace._registerDataFrameGetters(self)

