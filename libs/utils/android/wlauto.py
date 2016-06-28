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

import os
import subprocess as sp
import tempfile
import yaml
import glob

from . import Workload
from distutils.spawn import find_executable
from shutil import copyfile, rmtree

import logging

WA_RUN_CMD = 'wa run -f {} -d {}'
WA_REPO = 'https://github.com/ARM-software/workload-automation.git'

# Workloads directory path
workloads_dir = os.path.dirname(os.path.abspath(__file__))
workloads_dir = os.path.join(workloads_dir, 'workloads')
agendas_dir = os.path.join(workloads_dir, 'agendas')

class Wlauto(Workload):
    """
    Wrapper class for workload-automation

    Workload-automation (WA) already provides many workloads for Android
    targets.

    In order to use one of the WA workloads (`wa list workloads` tells you what
    is available) you need to implement a <workload__class_name> class in a
    <workload_name>.py file stored under libs/utils/android/workloads.

    <workload_class_name> inherits Wlauto and has to implement the methods from
    Workload and (eventually) Wlauto.

    This way, the workload is visible by the Workload class that will list it
    as available and will eventually check that the related package is
    installed on the Android target.

    :param test_env: Test Environment object
    :type test_env: TestEnv object
    """

    te = None

    def __init__(self, test_env):
        if not find_executable('wa'):
            raise RuntimeError(
                'wa not installed. Please, install it from %s', WA_REPO
            )

        Wlauto.te = test_env
        super(Wlauto, self).__init__(test_env)
        self._log = logging.getLogger('Wlauto')
        self._log.debug('WA wrapper created')


    @staticmethod
    def _wload_is_available(workload_name):
        """
        Check if the specified workload is available in workload-automation.

        :param workload_name: Name of the workload
        :type workload_name: str
        """
        _log = logging.getLogger('Wlauto')
        _log.debug('Checking for %s', workload_name)

        if sp.check_output(['wa', 'list', '-n', workload_name, 'workloads']):
            return True

        _log.error('%s not available in WA', workload_name)
        return False

    @staticmethod
    def wa_run(exp_dir, agenda, workload_name, collect=''):
        """
        Run workload automation using the specified agenda.

        :param exp_dir: Path to experiment directory where to store results.
        :type exp_dir: str

        :param agenda: Path to the YAML file to be passed to
            workload-automation. You can pass either an absolute path or a
            filename. In the latter case the file must be under
            android/workloads.
        :type agenda: str

        :param workload_name: Name of the workload
        :type workload_name: str

        :param collect: Specifies what to collect. Possible values:
            - 'energy'
        :type collect: list(str)
        """
        _log = logging.getLogger('Wlauto')

        # Check if the specified path is absolute
        if agenda.startswith('/'):
            _agenda = agenda
        else:
            _agenda = os.path.join(agendas_dir, agenda)

        _log.debug('Using agenda %s', _agenda)

        # Create temprary output directory for WA
        res_dir = tempfile.mkdtemp(prefix='lisa_wa_')
        _log.debug('WA output dir is %s', res_dir)

        # Initialize energy meter results
        nrg_report = None
        # Start energy collection
        if 'energy' in collect and Wlauto.te.emeter:
            Wlauto.te.emeter.reset()

        # Execute workload-automation
        p = sp.Popen(WA_RUN_CMD.format(_agenda, res_dir).split(),
                     stdout=sp.PIPE,
                     stderr=sp.PIPE)
        out, err = p.communicate()

        # Stop energy collection
        if 'energy' in collect and Wlauto.te.emeter:
            nrg_report = Wlauto.te.emeter.report(exp_dir)

        _log.debug(out)
        _log.debug(err)

        # Copy results to exp_dir
        # Take into account results from multiple iterations
        wload_dirs = glob.glob('{}/{}*'.format(res_dir, workload_name))

        # Store relevant results under `exp_dir`
        for res_dir in wload_dirs:
            dst_res_dir = os.path.join(exp_dir, os.path.basename(res_dir))
            if os.path.exists(dst_res_dir):
                # Remove possible outdated results
                rmtree(dst_res_dir)
            os.mkdir(dst_res_dir)

            for f in os.listdir(res_dir):
                src_path = os.path.join(res_dir, f)
                if os.path.isfile(src_path):
                    dst_path = os.path.join(dst_res_dir, f)
                    copyfile(src_path, dst_path)

        return nrg_report

    def generate_yaml_agenda(self, agenda):
        """
        Converts a dictionary into a YAML file which should be used as the
        agenda for running workload-automation benchmarks.

        The YAML file is put under the results directory specified in the test
        environment.

        :param agenda: Dictionary to be converted into YAML file
        :type agenda: dict
        """
        if not isinstance(agenda, dict):
            raise ValueError('Specified agenda must be a dictionary')

        _agenda = os.path.join(self.te.res_dir, 'agenda.yaml')

        with open(_agenda, 'w') as f:
            f.write(yaml.dump(agenda, default_flow_style=True))

        return _agenda

# vim :set tabstop=4 shiftwidth=4 expandtab
