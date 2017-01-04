#    Copyright 2015-2017 ARM Limited
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

"""Config Parameters for the Statistics Framework"""

# Default interval between a uniform time series
DELTA_DEFAULT = 0.000025
"""The default delta for uniformly resampled data"""
GRAMMAR_DEFAULT_PIVOT = "NO_PIVOT"
"""Default pivot value for :mod:`trappy.stats.grammar`"""
REINDEX_METHOD_DEFAULT = "pad"
"""Default method for reindexing and filling up NaNs"""
REINDEX_LIMIT_DEFAULT = None
"""Number or indices a value will be propagated forward when reindexing"""
NAN_FILL_DEFAULT = True
"""Fill NaN values by default"""
