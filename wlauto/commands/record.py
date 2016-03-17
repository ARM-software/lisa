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

from wlauto.common.resources import Executable
from wlauto.core.resource import NO_ONE
from wlauto.core.resolver import ResourceResolver
from wlauto.core.configuration import RunConfiguration
from wlauto.core.agenda import Agenda


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
        self.context = context
        self.parser.add_argument('-d', '--device', help='The name of the device')
        self.parser.add_argument('-s', '--suffix', help='The suffix of the revent file, e.g. ``setup``')
        self.parser.add_argument('-o', '--output', help='Directory to save the recording in')
        self.parser.add_argument('-p', '--package', help='Package to launch before recording')
        self.parser.add_argument('-C', '--clear', help='Clear app cache before launching it',
                                 action="store_true")

    # Validate command options
    def validate_args(self, args):
        if args.clear and not args.package:
            print "Package must be specified if you want to clear cache\n"
            self.parser.print_help()
            sys.exit()

    # pylint: disable=W0201
    def execute(self, args):
        self.validate_args(args)
        self.logger.info("Connecting to device...")

        ext_loader = PluginLoader(packages=settings.plugin_packages,
                                     paths=settings.plugin_paths)

        # Setup config
        self.config = RunConfiguration(ext_loader)
        for filepath in settings.get_config_paths():
            self.config.load_config(filepath)
        self.config.set_agenda(Agenda())
        self.config.finalize()

        context = LightContext(self.config)

        # Setup device
        self.device = ext_loader.get_device(settings.device, **settings.device_config)
        self.device.validate()
        self.device.connect()
        self.device.initialize(context)

        host_binary = context.resolver.get(Executable(NO_ONE, self.device.abi, 'revent'))
        self.target_binary = self.device.install_if_needed(host_binary)

        self.run(args)

    def run(self, args):
        if args.device:
            self.device_name = args.device
        else:
            self.device_name = self.device.get_device_model()

        if args.suffix:
            args.suffix += "."

        revent_file = self.device.path.join(self.device.working_directory,
                                            '{}.{}revent'.format(self.device_name, args.suffix or ""))

        if args.clear:
            self.device.execute("pm clear {}".format(args.package))

        if args.package:
            self.logger.info("Starting {}".format(args.package))
            self.device.execute('monkey -p {} -c android.intent.category.LAUNCHER 1'.format(args.package))

        self.logger.info("Press Enter when you are ready to record...")
        raw_input("")
        command = "{} record -t 100000 -s {}".format(self.target_binary, revent_file)
        self.device.kick_off(command)

        self.logger.info("Press Enter when you have finished recording...")
        raw_input("")
        self.device.killall("revent")

        self.logger.info("Pulling files from device")
        self.device.pull(revent_file, args.output or os.getcwdu())


class ReplayCommand(RecordCommand):

    name = 'replay'
    description = '''Replay a revent recording

    Revent allows you to record raw inputs such as screen swipes or button presses.
    See ``wa show record`` to see how to make an revent recording.
    '''

    def initialize(self, context):
        self.context = context
        self.parser.add_argument('revent', help='The name of the file to replay')
        self.parser.add_argument('-p', '--package', help='Package to launch before recording')
        self.parser.add_argument('-C', '--clear', help='Clear app cache before launching it',
                                 action="store_true")

    # pylint: disable=W0201
    def run(self, args):
        self.logger.info("Pushing file to device")
        self.device.push(args.revent, self.device.working_directory)
        revent_file = self.device.path.join(self.device.working_directory, os.path.split(args.revent)[1])

        if args.clear:
            self.device.execute("pm clear {}".format(args.package))

        if args.package:
            self.logger.info("Starting {}".format(args.package))
            self.device.execute('monkey -p {} -c android.intent.category.LAUNCHER 1'.format(args.package))

        command = "{} replay {}".format(self.target_binary, revent_file)
        self.device.execute(command)
        self.logger.info("Finished replay")


# Used to satisfy the API
class LightContext(object):
    def __init__(self, config):
        self.resolver = ResourceResolver(config)
        self.resolver.load()
