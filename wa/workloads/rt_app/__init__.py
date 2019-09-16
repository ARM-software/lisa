#    Copyright 2015-2018 ARM Limited
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
import re
import json
import tarfile
from collections import OrderedDict
from subprocess import CalledProcessError

from wa import Workload, Parameter, Executable, File
from wa.framework.exception import WorkloadError, ResourceError, ConfigError
from wa.utils.misc import check_output
from wa.utils.exec_control import once
from wa.utils.types import cpu_mask


RAW_OUTPUT_FILENAME = 'raw-output.txt'
TARBALL_FILENAME = 'rtapp-logs.tar.gz'
BINARY_NAME = 'rt-app'
PACKAGED_USE_CASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), 'use_cases'))

PLOAD_REGEX = re.compile(r'pLoad = (\d+)(\w+) : calib_cpu (\d+)')
ERROR_REGEX = re.compile(r'error')
CRIT_REGEX = re.compile(r'crit')


class RtApp(Workload):
    # pylint: disable=no-member,attribute-defined-outside-init

    name = 'rt-app'
    description = """
    A test application that simulates configurable real-time periodic load.

    rt-app is a test application that starts multiple periodic threads in order to
    simulate a real-time periodic load. It supports SCHED_OTHER, SCHED_FIFO,
    SCHED_RR as well as the AQuoSA framework and SCHED_DEADLINE.

    The load is described using JSON-like config files. Below are a couple of simple
    examples.


    Simple use case which creates a thread that run 1ms then sleep 9ms
    until the use case is stopped with Ctrl+C:

    .. code-block:: json

        {
            "tasks" : {
                "thread0" : {
                    "loop" : -1,
                    "run" :   20000,
                    "sleep" : 80000
                }
            },
            "global" : {
                "duration" : 2,
                "calibration" : "CPU0",
                "default_policy" : "SCHED_OTHER",
                "pi_enabled" : false,
                "lock_pages" : false,
                "logdir" : "./",
                "log_basename" : "rt-app1",
                "ftrace" : false,
                "gnuplot" : true,
            }
        }


    Simple use case with 2 threads that runs for 10 ms and wake up each
    other until the use case is stopped with Ctrl+C

    .. code-block:: json

        {
            "tasks" : {
                "thread0" : {
                    "loop" : -1,
                    "run" :     10000,
                    "resume" : "thread1",
                    "suspend" : "thread0"
                },
                "thread1" : {
                    "loop" : -1,
                    "run" :     10000,
                    "resume" : "thread0",
                    "suspend" : "thread1"
                }
            }
        }

    Please refer to the existing configs in ``$WA_ROOT/wa/workloads/rt_app/use_case``
    for more examples.

    The upstream version of rt-app is hosted here:

    https://github.com/scheduler-tools/rt-app

    """

    parameters = [
        Parameter('config', kind=str, default='taskset',
                  description='''
                  Use case configuration file to run with rt-app. This may be
                  either the name of one of the "standard" configurations included
                  with the workload. or a path to a custom JSON file provided by
                  the user. Either way, the ".json" extension is implied and will
                  be added automatically if not specified in the argument.

                  The following is the list of standard configurations currently
                  included with the workload: {}

                  '''.format(', '.join(os.listdir(PACKAGED_USE_CASE_DIRECTORY)))),
        Parameter('duration', kind=int,
                  description='''
                  Duration of the workload execution in Seconds. If specified, this
                  will override the corresponding parameter in the JSON config.
                  '''),
        Parameter('cpus', kind=cpu_mask, default=0, aliases=['taskset_mask'],
                  description='Constrain execution to specific CPUs.'),
        Parameter('uninstall', aliases=['uninstall_on_exit'], kind=bool, default=False,
                  override=True, description="""
                  If set to ``True``, rt-app binary will be uninstalled from the device
                  at the end of the run.
                  """),
        Parameter('force_install', kind=bool, default=False,
                  description="""
                  If set to ``True``, rt-app binary will always be deployed to the
                  target device at the beginning of the run, regardless of whether it
                  was already installed there.
                  """),
    ]

    @once
    def initialize(self, context):
        # initialize() runs once per run. setting a class variable to make it
        # available to other instances of the workload
        RtApp.target_working_directory = self.target.path.join(self.target.working_directory,
                                                               'rt-app-working')
        RtApp.host_binary = context.get_resource(Executable(self,
                                                            self.target.abi,
                                                            BINARY_NAME), strict=False)
        RtApp.workgen_script = context.get_resource(File(self, 'workgen'))
        self.target.execute('mkdir -p {}'.format(self.target_working_directory))
        self._deploy_rt_app_binary_if_necessary()

    def setup(self, context):
        self.output = None
        self.log_basename = context.current_job.label
        self.host_json_config = self._load_json_config(context)
        self.config_file_on_target = self.target.path.join(self.target_working_directory,
                                                           os.path.basename(self.host_json_config))
        self.target.push(self.host_json_config, self.config_file_on_target, timeout=60)
        self.command = '{} {}'.format(self.target_binary, self.config_file_on_target)

        time_buffer = 30
        self.timeout = self.duration + time_buffer

    def run(self, context):
        self.output = self.target.invoke(self.command,
                                         in_directory=self.target_working_directory,
                                         on_cpus=self.cpus and self.cpus.list() or None,
                                         redirect_stderr=True,
                                         timeout=self.timeout,
                                         as_root=self.target.is_rooted)

    def update_output(self, context):
        self._pull_rt_app_logs(context)
        context.output.classifiers.update(dict(
            duration=self.duration,
            task_count=self.task_count,
        ))

        if not self.output:
            return
        outfile = os.path.join(context.output_directory, RAW_OUTPUT_FILENAME)
        with open(outfile, 'w') as wfh:
            wfh.write(self.output)

        error_count = 0
        crit_count = 0
        for line in self.output.split('\n'):
            match = PLOAD_REGEX.search(line)
            if match:
                pload_value = match.group(1)
                pload_unit = match.group(2)
                calib_cpu_value = match.group(3)
                context.add_metric('pLoad', float(pload_value), pload_unit)
                context.add_metric('calib_cpu', float(calib_cpu_value))

            error_match = ERROR_REGEX.search(line)
            if error_match:
                error_count += 1

            crit_match = CRIT_REGEX.search(line)
            if crit_match:
                crit_count += 1

        context.add_metric('error_count', error_count, 'count')
        context.add_metric('crit_count', crit_count, 'count')

    @once
    def finalize(self, context):
        if self.uninstall:
            self.target.uninstall(self.target_binary)
        if self.cleanup_assets:
            self.target.execute('rm -rf {}'.format(self.target_working_directory))

    def _deploy_rt_app_binary_if_necessary(self):
        # called from initialize() so gets invoked once per run
        RtApp.target_binary = self.target.get_installed("rt-app")
        if self.force_install or not RtApp.target_binary:
            if not self.host_binary:
                message = '''rt-app is not installed on the target and could not be
                             found in workload resources'''
                raise ResourceError(message)
            RtApp.target_binary = self.target.install(self.host_binary)

    def _load_json_config(self, context):
        user_config_file = self._get_raw_json_config(context)
        config_file = self._generate_workgen_config(user_config_file,
                                                    context.output_directory)
        with open(config_file) as fh:
            try:
                config_data = json.load(fh, object_pairs_hook=OrderedDict)
            except ValueError:
                # We were not able to parse the JSON file. Raise an informative error.
                msg = "Failed to parse {}. Please make sure it is valid JSON."
                raise ConfigError(msg.format(user_config_file))

        self._update_rt_app_config(config_data)
        self.duration = config_data['global'].get('duration', 0)
        self.task_count = len(config_data.get('tasks', []))
        with open(config_file, 'w') as wfh:
            json.dump(config_data, wfh, indent=4)
        return config_file

    def _get_raw_json_config(self, resolver):
        if os.path.splitext(self.config)[1] != '.json':
            self.config += '.json'
        if os.path.isfile(self.config):
            return os.path.abspath(self.config)
        partial_path = os.path.join('use_cases', self.config)
        return resolver.get(File(self, partial_path))

    def _generate_workgen_config(self, user_file, output_directory):
        output_file = os.path.join(output_directory, 'unkind.json')
        # use workgen dry run option to generate a use case
        # file with proper JSON grammar on host first
        try:
            check_output('python3 {} -d -o {} {}'.format(self.workgen_script,
                                                         output_file,
                                                         user_file),
                         shell=True)
        except CalledProcessError as e:
            message = 'Could not generate config using workgen, got "{}"'
            raise WorkloadError(message.format(e))
        return output_file

    def _update_rt_app_config(self, config_data):
        config_data['global'] = config_data.get('global', {})
        config_data['global']['logdir'] = self.target_working_directory
        config_data['global']['log_basename'] = self.log_basename
        if self.duration is not None:
            config_data['global']['duration'] = self.duration

    def _pull_rt_app_logs(self, context):
        tar_command = '{} tar czf {}/{} -C {} .'.format(self.target.busybox,
                                                        self.target_working_directory,
                                                        TARBALL_FILENAME,
                                                        self.target_working_directory)
        self.target.execute(tar_command, timeout=300)
        target_path = self.target.path.join(self.target_working_directory, TARBALL_FILENAME)
        host_path = os.path.join(context.output_directory, TARBALL_FILENAME)
        self.target.pull(target_path, host_path, timeout=120)
        with tarfile.open(host_path, 'r:gz') as tf:
            tf.extractall(context.output_directory)
        os.remove(host_path)
        self.target.execute('rm -rf {}/*'.format(self.target_working_directory))
