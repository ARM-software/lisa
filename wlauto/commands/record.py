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


from wlauto import Command, settings
from wlauto.core import pluginloader
from wlauto.common.resources import Executable
from wlauto.core.resource import NO_ONE
from wlauto.core.resolver import ResourceResolver
from wlauto.core.configuration import RunConfiguration
from wlauto.core.agenda import Agenda
from wlauto.common.android.workload import ApkWorkload


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
        self.parser.add_argument('-o', '--output', help='Directory to save the recording in')

        # Need validation
        self.parser.add_argument('-s', '--suffix', help='The suffix of the revent file, e.g. ``setup``')
        self.parser.add_argument('-C', '--clear', help='Clear app cache before launching it',
                                 action="store_true")

        group = self.parser.add_mutually_exclusive_group(required=False)
        group.add_argument('-p', '--package', help='Package to launch before recording')
        group.add_argument('-w', '--workload', help='Name of a revent workload (mostly games)')

    # Validate command options
    def validate_args(self, args):
        if args.clear and not (args.package or args.workload):
            self.logger.error("Package/Workload must be specified if you want to clear cache")
            self.parser.print_help()
            sys.exit()
        if args.workload and args.suffix:
            self.logger.error("cannot specify manual suffixes for workloads")
            self.parser.print_help()
            sys.exit()
        if args.suffix:
            args.suffix += "."

    # pylint: disable=W0201
    def execute(self, args):
        self.validate_args(args)
        self.logger.info("Connecting to device...")

        # Setup config
        self.config = RunConfiguration(pluginloader)
        for filepath in settings.config_paths:
            self.config.load_config(filepath)
        self.config.set_agenda(Agenda())
        self.config.finalize()

        # Setup device
        self.device_manager = pluginloader.get_manager(self.config.device)
        self.device_manager.validate()
        self.device_manager.connect()
        context = LightContext(self.config, self.device_manager)
        self.device_manager.initialize(context)
        self.device = self.device_manager.target
        if args.device:
            self.device_name = args.device
        else:
            self.device_name = self.device.model

        # Install Revent
        host_binary = context.resolver.get(Executable(NO_ONE, self.device.abi, 'revent'))
        self.target_binary = self.device.install_if_needed(host_binary)

        if args.workload:
            self.workload_record(args, context)
        elif args.package:
            self.package_record(args, context)
        else:
            self.manual_record(args, context)

    def manual_record(self, args, context):
        revent_file = self.device.get_workpath('{}.{}revent'.format(self.device_name, args.suffix or ""))
        self._record(revent_file, "", args.output)

    def package_record(self, args, context):
        revent_file = self.device.get_workpath('{}.{}revent'.format(self.device_name, args.suffix or ""))
        if args.clear:
            self.device.execute("pm clear {}".format(args.package))

        self.logger.info("Starting {}".format(args.package))
        self.device.execute('monkey -p {} -c android.intent.category.LAUNCHER 1'.format(args.package))

        self._record(revent_file, "", args.output)

    def workload_record(self, args, context):
        setup_file = self.device.get_workpath('{}.setup.revent'.format(self.device_name))
        run_file = self.device.get_workpath('{}.run.revent'.format(self.device_name))

        self.logger.info("Deploying {}".format(args.workload))
        workload = pluginloader.get_workload(args.workload, self.device)
        workload.apk_init_resources(context)
        workload.initialize_package(context)
        workload.do_post_install(context)
        workload.start_activity()

        if args.clear:
            workload.reset(context)

        self._record(setup_file, " SETUP",
                     args.output or os.path.join(workload.dependencies_directory, 'revent_files'))
        self._record(run_file, " RUN",
                     args.output or os.path.join(workload.dependencies_directory, 'revent_files'))

        self.logger.info("Tearing down {}".format(args.workload))
        workload.apk_teardown(context)

    def _record(self, revent_file, name, output_path):
        self.logger.info("Press Enter when you are ready to record{}...".format(name))
        raw_input("")
        command = "{} record -t 100000 -s {}".format(self.target_binary, revent_file)
        self.device.kick_off(command)

        self.logger.info("Press Enter when you have finished recording {}...".format(name))
        raw_input("")
        self.device.killall("revent")

        output_path = output_path or os.getcwdu()
        if not os.path.isdir(output_path):
            os.mkdirs(output_path)

        revent_file_name = self.device.path.basename(revent_file)
        host_path = os.path.join(output_path, revent_file_name)
        if os.path.exists(host_path):
            self.logger.info("Revent file '{}' already exists, overwrite? [y/n]".format(revent_file_name))
            if raw_input("") == "y":
                os.remove(host_path)
            else:
                self.logger.warning("Did not pull and overwrite '{}'".format(revent_file_name))
                return
        self.logger.info("Pulling '{}' from device".format(self.device.path.basename(revent_file)))
        self.device.pull(revent_file, output_path)

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
    def __init__(self, config, device_manager):
        self.resolver = ResourceResolver(config)
        self.resolver.load()
        self.device_manager = device_manager
