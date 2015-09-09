#    Copyright 2015-2015 ARM Limited
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
"""A helper module to extract data from WorkloadAutomation
output directories. WorkloadAutomation is a tool for
automating the execution of workloads. For more information
please visit https://github.com/ARM-software/workload-automation

.. note::

    TRAPpy does not have a dependency on workload automation
"""

from trappy.wa.results import Result, get_results, combine_results
from trappy.wa.sysfs_extractor import SysfsExtractor
