#    Copyright 2014-2015 ARM Limited
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

import os
import sys


from wa import Command, settings
from wa.framework import pluginloader
from wa.framework.agenda import Agenda
from wa.framework.resource import Executable, NO_ONE, ResourceResolver
from wa.framework.configuration import RunConfiguration
from wa.framework.workload import ApkUiautoWorkload


class RecordCommand(Command):

    name = 'record'
    description = '''Performs a revent recording

    This command helps making revent recordings. It will automatically
    deploy revent and even has the option of automatically opening apps.

    Revent allows you to record raw inputs such as screen swipes or button presses.
    This can be useful for recording inputs for workloads such as games that don't
    have XML UI layouts that can be used with UIAutomator. As a drawback from this,
    revent recordings are specific to the device type they were recorded on.

    WA uses two parts to the names of revent recordings in the format,
    {device_name}.{suffix}.revent.

     - device_name can either be specified manually with the ``-d`` argument or
       it can be automatically determined. On Android device it will be obtained
       from ``build.prop``, on Linux devices it is obtained from ``/proc/device-tree/model``.
     - suffix is used by WA to determine which part of the app execution the
       recording is for, currently these are either ``setup`` or ``run``. This
       should be specified with the ``-s`` argument.
    '''

    def initialize(self, context):
        self.parser.add_argument('-d', '--device', metavar='DEVICE',
                                 help='''
                                 Specify the device on which to run. This will
                                 take precedence over the device (if any)
                                 specified in configuration.
                                 ''')

    def execute(state, args):
        if args.device:
            device =  args.device
            device_config = {}
        else:
            device = state.run_config.device
            device_config = state.run_config.device_config
        target_manager = TargetManager(device, device_config)


def get_revent_binary(abi):
    resolver = ResourceResolver()
    resource = Executable(NO_ONE, abi, 'revent')
    return resolver.get(resource)


class ReventRecorder(object):

    def __init__(self, target):
        self.target = target
        self.executable = None
        self.deploy()

    def deploy(self):
        host_executable = get_revent_binary(self.target.abi)
        self.executable = self.target.install(host_executable)

    def record(self, path):
        name = os.path.basename(path)
        target_path = self.target.get_workpath(name)
        command = '{} record {}'

    def remove(self):
        if self.executable:
            self.target.uninstall('revent')
