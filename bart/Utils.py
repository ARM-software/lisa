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

"""Utility functions for sheye"""

import cr2

def init_run(trace):
    """Initialize the Run Object"""

    if isinstance(trace, basestring):
        return cr2.Run(trace)

    elif isinstance(trace, cr2.Run):
        return trace

    raise ValueError("Invalid trace Object")
