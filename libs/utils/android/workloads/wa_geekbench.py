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

from android import Wlauto
from android import Workload

import glob
import logging

class WaGeekbench(Wlauto):
    """
    Geekbench workload
    """

    # Package required by this workload
    package = 'com.primatelabs.geekbench'

    # WA name of the workload
    workload_name = 'geekbench'

    def __init__(self, test_env):
        super(WaGeekbench, self).__init__(test_env)
        self._log = logging.getLogger('Geekbench')
        self._log.debug('Workload created')

    def run(self, exp_dir, agenda, collect=''):
        """
        :param exp_dir: Path to experiment directory where to store results.
        :type exp_dir: str

        :param agenda: Agenda to be passed to workload-automation. Can be a
            YAML file or a dictionary.
        :type agenda: str or dict

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
        :type collect: list(str)
        """
        # Setting agenda_yaml properly
        _agenda = agenda
        if isinstance(agenda, dict):
            # Convert dictionary to YAML file
            _agenda = self.generate_yaml_agenda(agenda)

        self._log.debug('Running')

        nrg_report = Wlauto.wa_run(exp_dir, agenda,
                                   WaGeekbench.workload_name, collect=collect)

        return None, nrg_report

    @staticmethod
    def _is_available(target):
        if Wlauto._wload_is_available(WaGeekbench.workload_name):
            return True
        return False

# vim :set tabstop=4 shiftwidth=4 expandtab
