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
from time import sleep

from wa import Command
from wa.framework import pluginloader
from wa.framework.resource import ResourceResolver
from wa.framework.target.manager import TargetManager
from wa.utils.revent import ReventRecorder


class RecordCommand(Command):

    name = 'record'
    description = '''
    Performs a revent recording

    This command helps making revent recordings. It will automatically
    deploy revent and has options to automatically open apps and record
    specified stages of a workload.

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
       recording is for, currently these are either ``setup``, ``run``, ``extract_results``
       or ``teardown``. All stages except ``run`` are optional and these should
       be specified with the ``-s``, ``-e`` or ``-t`` arguments respectively,
       or optionally ``-a`` to indicate all stages should be recorded.
    '''

    def __init__(self, **kwargs):
        super(RecordCommand, self).__init__(**kwargs)
        self.tm = None
        self.target = None
        self.revent_recorder = None

    def initialize(self, context):
        self.parser.add_argument('-d', '--device', metavar='DEVICE',
                                 help='''
                                 Specify the device on which to run. This will
                                 take precedence over the device (if any)
                                 specified in configuration.
                                 ''')
        self.parser.add_argument('-o', '--output', help='Specify the output file', metavar='FILE')
        self.parser.add_argument('-s', '--setup', help='Record a recording for setup stage',
                                 action='store_true')
        self.parser.add_argument('-e', '--extract_results', help='Record a recording for extract_results stage',
                                 action='store_true')
        self.parser.add_argument('-t', '--teardown', help='Record a recording for teardown stage',
                                 action='store_true')
        self.parser.add_argument('-a', '--all', help='Record recordings for available stages',
                                 action='store_true')

        # Need validation
        self.parser.add_argument('-C', '--clear', help='Clear app cache before launching it',
                                 action='store_true')
        group = self.parser.add_mutually_exclusive_group(required=False)
        group.add_argument('-p', '--package', help='Package to launch before recording')
        group.add_argument('-w', '--workload', help='Name of a revent workload (mostly games)')

    def validate_args(self, args):
        if args.clear and not (args.package or args.workload):
            self.logger.error("Package/Workload must be specified if you want to clear cache")
            sys.exit()
        if args.workload and args.output:
            self.logger.error("Output file cannot be specified with Workload")
            sys.exit()
        if not args.workload and (args.setup or args.extract_results or
                                  args.teardown or args.all):
            self.logger.error("Cannot specify a recording stage without a Workload")
            sys.exit()

    def execute(self, state, args):
        self.validate_args(args)
        state.run_config.merge_device_config(state.plugin_cache)
        if args.device:
            device = args.device
            device_config = {}
        else:
            device = state.run_config.device
            device_config = state.run_config.device_config or {}

        if args.output:
            outdir = os.path.basename(args.output)
        else:
            outdir = os.getcwd()

        self.tm = TargetManager(device, device_config, outdir)
        self.target = self.tm.target
        self.revent_recorder = ReventRecorder(self.target)
        self.revent_recorder.deploy()

        if args.workload:
            self.workload_record(args)
        elif args.package:
            self.package_record(args)
        else:
            self.manual_record(args)

        self.revent_recorder.remove()

    def record(self, revent_file, name, output_path):
        msg = 'Press Enter when you are ready to record {}...'
        self.logger.info(msg.format(name))
        raw_input('')
        self.revent_recorder.start_record(revent_file)
        msg = 'Press Enter when you have finished recording {}...'
        self.logger.info(msg.format(name))
        raw_input('')
        self.revent_recorder.stop_record()

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        revent_file_name = self.target.path.basename(revent_file)
        host_path = os.path.join(output_path, revent_file_name)
        if os.path.exists(host_path):
            msg = 'Revent file \'{}\' already exists, overwrite? [y/n]'
            self.logger.info(msg.format(revent_file_name))
            if raw_input('') == 'y':
                os.remove(host_path)
            else:
                msg = 'Did not pull and overwrite \'{}\''
                self.logger.warning(msg.format(revent_file_name))
                return
        msg = 'Pulling \'{}\' from device'
        self.logger.info(msg.format(self.target.path.basename(revent_file)))
        self.target.pull(revent_file, output_path, as_root=self.target.is_rooted)

    def manual_record(self, args):
        output_path, file_name = self._split_revent_location(args.output)
        revent_file = self.target.get_workpath(file_name)
        self.record(revent_file, '', output_path)
        msg = 'Recording is available at: \'{}\''
        self.logger.info(msg.format(os.path.join(output_path, file_name)))

    def package_record(self, args):
        if args.clear:
            self.target.execute('pm clear {}'.format(args.package))
        self.logger.info('Starting {}'.format(args.package))
        cmd = 'monkey -p {} -c android.intent.category.LAUNCHER 1'
        self.target.execute(cmd.format(args.package))

        output_path, file_name = self._split_revent_location(args.output)
        revent_file = self.target.get_workpath(file_name)
        self.record(revent_file, '', output_path)
        msg = 'Recording is available at: \'{}\''
        self.logger.info(msg.format(os.path.join(output_path, file_name)))

    def workload_record(self, args):
        context = LightContext(self.tm)
        setup_revent = '{}.setup.revent'.format(self.target.model)
        run_revent = '{}.run.revent'.format(self.target.model)
        extract_results_revent = '{}.extract_results.revent'.format(self.target.model)
        teardown_file_revent = '{}.teardown.revent'.format(self.target.model)
        setup_file = self.target.get_workpath(setup_revent)
        run_file = self.target.get_workpath(run_revent)
        extract_results_file = self.target.get_workpath(extract_results_revent)
        teardown_file = self.target.get_workpath(teardown_file_revent)

        self.logger.info('Deploying {}'.format(args.workload))
        workload = pluginloader.get_workload(args.workload, self.target)
        # Setup apk if android workload
        if hasattr(workload, 'apk'):
            workload.apk.initialize(context)
            workload.apk.setup(context)
            sleep(workload.loading_time)

        output_path = os.path.join(workload.dependencies_directory,
                                   'revent_files')
        if args.setup or args.all:
            self.record(setup_file, 'SETUP', output_path)
        self.record(run_file, 'RUN', output_path)
        if args.extract_results or args.all:
            self.record(extract_results_file, 'EXTRACT_RESULTS', output_path)
        if args.teardown or args.all:
            self.record(teardown_file, 'TEARDOWN', output_path)
        self.logger.info('Tearing down {}'.format(args.workload))
        workload.teardown(context)
        self.logger.info('Recording(s) are available at: \'{}\''.format(output_path))

    def _split_revent_location(self, output):
        output_path = None
        file_name = None
        if output:
            output_path, file_name, = os.path.split(output)

        if not file_name:
            file_name = '{}.revent'.format(self.target.model)
        if not output_path:
            output_path = os.getcwdu()

        return output_path, file_name

class ReplayCommand(Command):

    name = 'replay'
    description = '''
    Replay a revent recording

    Revent allows you to record raw inputs such as screen swipes or button presses.
    See ``wa show record`` to see how to make an revent recording.
    '''

    def initialize(self, context):
        self.parser.add_argument('recording', help='The name of the file to replay',
                                 metavar='FILE')
        self.parser.add_argument('-d', '--device', help='The name of the device')
        self.parser.add_argument('-p', '--package', help='Package to launch before recording')
        self.parser.add_argument('-C', '--clear', help='Clear app cache before launching it',
                                 action="store_true")

    # pylint: disable=W0201
    def execute(self, state, args):
        state.run_config.merge_device_config(state.plugin_cache)
        if args.device:
            device = args.device
            device_config = {}
        else:
            device = state.run_config.device
            device_config = state.run_config.device_config or {}

        target_manager = TargetManager(device, device_config, None)
        self.target = target_manager.target
        revent_file = self.target.path.join(self.target.working_directory,
                                            os.path.split(args.recording)[1])

        self.logger.info("Pushing file to target")
        self.target.push(args.recording, self.target.working_directory)

        revent_recorder = ReventRecorder(target_manager.target)
        revent_recorder.deploy()

        if args.clear:
            self.target.execute('pm clear {}'.format(args.package))

        if args.package:
            self.logger.info('Starting {}'.format(args.package))
            cmd = 'monkey -p {} -c android.intent.category.LAUNCHER 1'
            self.target.execute(cmd.format(args.package))

        self.logger.info("Starting replay")
        revent_recorder.replay(revent_file)
        self.logger.info("Finished replay")
        revent_recorder.remove()


# Used to satisfy the workload API
class LightContext(object):
    def __init__(self, tm):
        self.tm = tm
        self.resolver = ResourceResolver()
        self.resolver.load()
