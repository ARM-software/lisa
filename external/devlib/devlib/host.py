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
import glob
import os
import signal
import shutil
import subprocess
import logging
from distutils.dir_util import copy_tree
from getpass import getpass
from pipes import quote

from devlib.exception import (
    TargetTransientError, TargetStableError,
    TargetTransientCalledProcessError, TargetStableCalledProcessError
)
from devlib.utils.misc import check_output
from devlib.connection import ConnectionBase, PopenBackgroundCommand


PACKAGE_BIN_DIRECTORY = os.path.join(os.path.dirname(__file__), 'bin')


# pylint: disable=redefined-outer-name
def kill_children(pid, signal=signal.SIGKILL):
    with open('/proc/{0}/task/{0}/children'.format(pid), 'r') as fd:
        for cpid in map(int, fd.read().strip().split()):
            kill_children(cpid, signal)
            os.kill(cpid, signal)


class LocalConnection(ConnectionBase):

    name = 'local'
    host = 'localhost'

    @property
    def connected_as_root(self):
        if self._connected_as_root is None:
            result = self.execute('id', as_root=False)
            self._connected_as_root = 'uid=0(' in result
        return self._connected_as_root

    @connected_as_root.setter
    def connected_as_root(self, state):
        self._connected_as_root = state

    # pylint: disable=unused-argument
    def __init__(self, platform=None, keep_password=True, unrooted=False,
                 password=None, timeout=None):
        super().__init__()
        self._connected_as_root = None
        self.logger = logging.getLogger('local_connection')
        self.keep_password = keep_password
        self.unrooted = unrooted
        self.password = password


    def _copy_path(self, source, dest):
        self.logger.debug('copying {} to {}'.format(source, dest))
        if os.path.isdir(source):
            # Use distutils copy_tree since it behaves the same as
            # shutils.copytree except that it won't fail if some folders
            # already exist.
            #
            # Mirror the behavior of all other targets which only copy the
            # content without metadata
            copy_tree(source, dest, preserve_mode=False, preserve_times=False)
        else:
            shutil.copy(source, dest)

    def _copy_paths(self, sources, dest):
        for source in sources:
            self._copy_path(source, dest)

    def push(self, sources, dest, timeout=None, as_root=False):  # pylint: disable=unused-argument
        self._copy_paths(sources, dest)

    def pull(self, sources, dest, timeout=None, as_root=False): # pylint: disable=unused-argument
        self._copy_paths(sources, dest)

    # pylint: disable=unused-argument
    def execute(self, command, timeout=None, check_exit_code=True,
                as_root=False, strip_colors=True, will_succeed=False):
        self.logger.debug(command)
        use_sudo = as_root and not self.connected_as_root
        if use_sudo:
            if self.unrooted:
                raise TargetStableError('unrooted')
            password = self._get_password()
            command = "echo {} | sudo -k -p ' ' -S -- sh -c {}".format(quote(password), quote(command))
        ignore = None if check_exit_code else 'all'
        try:
            stdout, stderr = check_output(command, shell=True, timeout=timeout, ignore=ignore)
        except subprocess.CalledProcessError as e:
            cls = TargetTransientCalledProcessError if will_succeed else TargetStableCalledProcessError
            raise cls(
                e.returncode,
                command,
                e.output,
                e.stderr,
            )

        # Remove the one-character prompt of sudo -S -p
        if use_sudo and stderr:
            stderr = stderr[1:]

        return stdout + stderr

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        if as_root and not self.connected_as_root:
            if self.unrooted:
                raise TargetStableError('unrooted')
            password = self._get_password()
            # The sudo prompt will add a space on stderr, but we cannot filter
            # it out here
            command = "echo {} | sudo -k -p ' ' -S -- sh -c {}".format(quote(password), quote(command))

        # Make sure to get a new PGID so PopenBackgroundCommand() can kill
        # all sub processes that could be started without troubles.
        def preexec_fn():
            os.setpgrp()

        popen = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.PIPE,
            shell=True,
            preexec_fn=preexec_fn,
        )
        bg_cmd = PopenBackgroundCommand(popen)
        self._current_bg_cmds.add(bg_cmd)
        return bg_cmd

    def _close(self):
        pass

    def cancel_running_command(self):
        pass

    def wait_for_device(self, timeout=30):
        return

    def reboot_bootloader(self, timeout=30):
        raise NotImplementedError()

    def _get_password(self):
        if self.password:
            return self.password
        password = getpass('sudo password:')
        if self.keep_password:
            self.password = password
        return password
