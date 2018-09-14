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

SCRIPT_NAME = 'remote_script.sh'

class TargetScript(object):
    """
    This class provides utility to create and run a script
    directly on a devlib target.

    The execute() method is made to look like Devlib's, so a Target instance can
    be swapped with an instance of this TargetScript class, and the commands
    will be accumulated for later use instead of being executed straight away.

    :param env: Reference TestEnv instance. Will be used for some commands
        that must really be executed instead of accumulated.
    :type env: TestEnv

    :param script_name: Name of the script that will be pushed on the target
    :type script_name: str

    :param local_dir: Local directory to use to prepare the script
    :type local_dir: str
    """

    _target_attrs = ['screen_resolution', 'android_id', 'abi', 'os_version', 'model']

    def __init__(self, env, script_name=SCRIPT_NAME, local_dir='./'):
        self._env = env
        self._target = env.target
        self._script_name = script_name
        self.local_dir = local_dir
        self.commands = []

    # This is made to look like the devlib Target execute()
    def execute(self, cmd):
        """
        Accumulate command for later execution.

        :param cmd: Command that would be run on the target
        :type cmd: str
        """
        self.append(cmd)

    def append(self, cmd):
        """
        Append a command to the script.

        :param cmd: Command string to append
        :type cmd: str
        """
        self.commands.append(cmd)

    # Some commands may require some info about the real target.
    # For instance, System.{h,v}swipe needs to get the value of
    # screen_resolution to generate a swipe command at a given
    # screen coordinate percentage.
    # Thus, if such a property is called on this object,
    # it will be fetched from the 'real' target object.
    def __getattr__(self, name):
        if name in self._target_attrs:
            return getattr(self._target, name)

        return getattr(super, name)

    def push(self):
        """
        Push a script to the target

        The script is created and stored on the host, and is then sent
        to the target.
        """

        actions = ['set -e'] + self.commands + ['set +e']
        actions = ['#!{} sh'.format(self._target.busybox)] + actions
        actions = str.join('\n', actions)

        # Create script locally
        self._local_path = os.path.join(self.local_dir, self._script_name)
        with open(self._local_path, 'w') as script:
            script.write(actions)

        # Push it on target
        self._remote_path = self._target.install(self._local_path)

    def run(self, as_root=False, background=False, timeout=None):
        """
        Run the previously pushed script
        """
        if self._target.file_exists(self._remote_path):
            self._run_as_root = as_root
            self._bg_shell = None
            if background:
                self._bg_shell = self._target.background(self._remote_path,
                                                    as_root=self._run_as_root)
            else:
                self._target.execute(self._remote_path,
                                     as_root=self._run_as_root,
                                     timeout=timeout)
        else:
            raise IOError('Remote script was not found on target device')

    def kill(self):
        """
        Kill a running script
        """
        cmd_pid = '$(pgrep {})'.format(self._script_name)
        self._target.kill(cmd_pid, as_root=self._run_as_root)
        if self._bg_shell:
            self._bg_shell.kill()

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
