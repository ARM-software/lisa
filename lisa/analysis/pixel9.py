# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, ARM Limited and contributors.
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
"""
Pixel 9-specific analysis.
"""

from lisa.analysis._pixel import PixelAnalysis


class Pixel9Analysis(PixelAnalysis):
    """
    Support for Pixel 9-specific data analysis
    """

    name = 'pixel9'

    """
    Power rail vs CPU/GPU cluster mapping

    Power rail information is extracted from
    https://cs.android.com/android/platform/superproject/main/+/main:device/google/caimito/thermal/thermal_info_config_tokay.json;l=628
    """
    EMETER_CHAN_NAMES = {
        'S4M_VDD_CPUCL0': 'CPU-Little',
        'S2M_VDD_CPUCL1': 'CPU-Mid',
        'S3M_VDD_CPUCL2': 'CPU-Big',
        'S2S_VDD_G3D': 'GPU',
    }
