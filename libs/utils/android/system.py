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

import logging

class System(object):
    """
    Collection of Android related services
    """

    @staticmethod
    def set_airplane_mode(target, on=True):
        """
        Set airplane mode
        """
        ap_mode = 1 if on else 0
        ap_state = 'true' if on else 'false'

        target.execute('settings put global airplane_mode_on {}'\
                       .format(ap_mode))
        target.execute('am broadcast '\
                       '-a android.intent.action.AIRPLANE_MODE '\
                       '--ez state {}'\
                       .format(ap_state))

# vim :set tabstop=4 shiftwidth=4 expandtab
