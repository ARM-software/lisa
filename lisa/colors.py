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

import sys

class TestColors:

    level = {
        'failed'    : '\033[0;31m', # Red
        'good'      : '\033[0;32m', # Green
        'warning'   : '\033[0;33m', # Yellow
        'passed'    : '\033[0;34m', # Blue
        'purple'    : '\033[0;35m', # Purple
        'endc'      : '\033[0m'     # End color
    }

    @staticmethod
    def rate(val, positive_is_good=True):
        str_val = "{:9.2f}%".format(val)

        if not sys.stdout.isatty():
            return str_val

        if not positive_is_good:
            val = -val

        if val < -10:
            str_color = TestColors.level['failed']
        elif val < 0:
            str_color = TestColors.level['warning']
        elif val < 10:
            str_color = TestColors.level['passed']
        else:
            str_color = TestColors.level['good']

        return str_color + str_val + TestColors.level['endc']

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
