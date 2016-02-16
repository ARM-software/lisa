#    Copyright 2013-2015 ARM Limited
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

# pylint: disable=E1101

from __future__ import division
import os

try:
    import jinja2
except ImportError:
    jinja2 = None

from wlauto import Workload, settings, Parameter
from wlauto.exceptions import WorkloadError
from wlauto.utils.hwmon import discover_sensors
from wlauto.utils.misc import get_meansd
from wlauto.utils.types import boolean, identifier, list_of_strs


THIS_DIR = os.path.dirname(__file__)
TEMPLATE_NAME = 'device_script.template'
SCRIPT_TEMPLATE = os.path.join(THIS_DIR, TEMPLATE_NAME)

APP_CONFIG = {
    'browser': {
        'package': 'com.android.browser',
        'activity': '.BrowserActivity',
        'options': '-d about:blank',
    },
    'calculator': {
        'package': 'com.android.calculator2',
        'activity': '.Calculator',
        'options': '',
    },
    'calendar': {
        'package': 'com.android.calendar',
        'activity': '.LaunchActivity',
        'options': '',
    },
}


class ApplaunchWorkload(Workload):

    name = 'applaunch'
    description = """
    Measures the time and energy used in launching an application.

    """
    supported_platforms = ['android']

    parameters = [
        Parameter('app', default='browser', allowed_values=['calculator', 'browser', 'calendar'],
                  description='The name of the application to measure.'),
        Parameter('set_launcher_affinity', kind=bool, default=True,
                  description=('If ``True``, this will explicitly set the affinity of the launcher '
                               'process to the A15 cluster.')),
        Parameter('times', kind=int, default=8,
                  description='Number of app launches to do on the device.'),
        Parameter('measure_energy', kind=boolean, default=False,
                  description="""
                  Specfies wether energy measurments should be taken during the run.

                  .. note:: This depends on appropriate sensors to be exposed through HWMON.

                  """),
        Parameter('io_stress', kind=boolean, default=False,
                  description='Specifies whether to stress IO during App launch.'),
        Parameter('io_scheduler', allowed_values=['noop', 'deadline', 'row', 'cfq', 'bfq'],
                  description='Set the IO scheduler to test on the device.'),
        Parameter('cleanup', kind=boolean, default=True,
                  description='Specifies whether to clean up temporary files on the device.'),
    ]

    def __init__(self, device, **kwargs):
        super(ApplaunchWorkload, self).__init__(device, **kwargs)
        if not jinja2:
            raise WorkloadError('Please install jinja2 Python package: "sudo pip install jinja2"')
        filename = '{}-{}.sh'.format(self.name, self.app)
        self.host_script_file = os.path.join(settings.meta_directory, filename)
        self.device_script_file = os.path.join(self.device.working_directory, filename)
        self._launcher_pid = None
        self._old_launcher_affinity = None
        self.sensors = []

    def on_run_init(self, context):  # pylint: disable=W0613
        if self.measure_energy:
            self.sensors = discover_sensors(self.device, ['energy'])
            for sensor in self.sensors:
                sensor.label = identifier(sensor.label).upper()

    def setup(self, context):
        self.logger.debug('Creating script {}'.format(self.host_script_file))
        with open(self.host_script_file, 'w') as wfh:
            env = jinja2.Environment(loader=jinja2.FileSystemLoader(THIS_DIR))
            template = env.get_template(TEMPLATE_NAME)
            wfh.write(template.render(device=self.device,  # pylint: disable=maybe-no-member
                                      sensors=self.sensors,
                                      iterations=self.times,
                                      io_stress=self.io_stress,
                                      io_scheduler=self.io_scheduler,
                                      cleanup=self.cleanup,
                                      package=APP_CONFIG[self.app]['package'],
                                      activity=APP_CONFIG[self.app]['activity'],
                                      options=APP_CONFIG[self.app]['options'],
                                      ))
        self.device_script_file = self.device.install(self.host_script_file)
        if self.set_launcher_affinity:
            self._set_launcher_affinity()
        self.device.clear_logcat()

    def run(self, context):
        self.device.execute('sh {}'.format(self.device_script_file), timeout=300, as_root=self.io_stress)

    def update_result(self, context):  # pylint: disable=too-many-locals
        result_files = ['time.result']
        result_files += ['{}.result'.format(sensor.label) for sensor in self.sensors]
        metric_suffix = ''
        if self.io_stress:
            host_scheduler_file = os.path.join(context.output_directory, 'scheduler')
            device_scheduler_file = '/sys/block/mmcblk0/queue/scheduler'
            self.device.pull(device_scheduler_file, host_scheduler_file)
            with open(host_scheduler_file) as fh:
                scheduler = fh.read()
                scheduler_used = scheduler[scheduler.index("[") + 1:scheduler.index("]")]
                metric_suffix = '_' + scheduler_used
        for filename in result_files:
            self._extract_results_from_file(context, filename, metric_suffix)

    def teardown(self, context):
        if self.set_launcher_affinity:
            self._reset_launcher_affinity()
        if self.cleanup:
            self.device.remove(self.device_script_file)

    def _set_launcher_affinity(self):
        try:
            self._launcher_pid = self.device.get_pids_of('com.android.launcher')[0]
            result = self.device.execute('taskset -p {}'.format(self._launcher_pid), busybox=True, as_root=True)
            self._old_launcher_affinity = int(result.split(':')[1].strip(), 16)

            cpu_ids = [i for i, x in enumerate(self.device.core_names) if x == 'a15']
            if not cpu_ids or len(cpu_ids) == len(self.device.core_names):
                self.logger.debug('Cannot set affinity.')
                return

            new_mask = reduce(lambda x, y: x | y, cpu_ids, 0x0)
            self.device.execute('taskset -p 0x{:X} {}'.format(new_mask, self._launcher_pid), busybox=True, as_root=True)
        except IndexError:
            raise WorkloadError('Could not set affinity of launcher: PID not found.')

    def _reset_launcher_affinity(self):
        command = 'taskset -p 0x{:X} {}'.format(self._old_launcher_affinity, self._launcher_pid)
        self.device.execute(command, busybox=True, as_root=True)

    def _extract_results_from_file(self, context, filename, metric_suffix):
        host_result_file = os.path.join(context.output_directory, filename)
        device_result_file = self.device.path.join(self.device.working_directory, filename)
        self.device.pull(device_result_file, host_result_file)

        with open(host_result_file) as fh:
            if filename == 'time.result':
                values = [v / 1000 for v in map(int, fh.read().split())]
                _add_metric(context, 'time' + metric_suffix, values, 'Seconds')
            else:
                metric = filename.replace('.result', '').lower()
                numbers = iter(map(int, fh.read().split()))
                deltas = [(after - before) / 1000000 for before, after in zip(numbers, numbers)]
                _add_metric(context, metric, deltas, 'Joules')


def _add_metric(context, metric, values, units):
    mean, sd = get_meansd(values)
    context.result.add_metric(metric, mean, units)
    context.result.add_metric(metric + ' sd', sd, units, lower_is_better=True)
