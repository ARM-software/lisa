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

# pylint: disable=E1101,W0201,E0203

from __future__ import division
import os
import re
import time
import select
import json
import threading
import subprocess

from wlauto import ApkWorkload, Parameter, Alias
from wlauto.exceptions import WorkloadError


DELAY = 2


class GlbCorp(ApkWorkload):

    name = 'glb_corporate'
    description = """
    GFXBench GL (a.k.a. GLBench) v3.0 Corporate version.

    This is a version of GLBench available through a corporate license (distinct
    from the version available in Google Play store).

    """
    package = 'net.kishonti.gfxbench'
    activity = 'net.kishonti.benchui.TestActivity'

    result_start_regex = re.compile(r'I/TfwActivity\s*\(\s*\d+\):\s+\S+\s+result: {')
    preamble_regex = re.compile(r'I/TfwActivity\s*\(\s*\d+\):\s+')

    valid_test_ids = [
        'gl_alu',
        'gl_alu_off',
        'gl_blending',
        'gl_blending_off',
        'gl_driver',
        'gl_driver_off',
        'gl_fill',
        'gl_fill_off',
        'gl_manhattan',
        'gl_manhattan_off',
        'gl_trex',
        'gl_trex_battery',
        'gl_trex_off',
        'gl_trex_qmatch',
        'gl_trex_qmatch_highp',
    ]

    supported_resolutions = {
        '720p': {
            '-ei -w': 1280,
            '-ei -h': 720,
        },
        '1080p': {
            '-ei -w': 1920,
            '-ei -h': 1080,
        }
    }

    parameters = [
        Parameter('times', kind=int, default=1, constraint=lambda x: x > 0,
                  description=('Specifies the number of times the benchmark will be run in a "tight '
                               'loop", i.e. without performaing setup/teardown inbetween.')),
        Parameter('resolution', default=None, allowed_values=['720p', '1080p', '720', '1080'],
                  description=('Explicitly specifies the resultion under which the benchmark will '
                               'be run. If not specfied, device\'s native resoution will used.')),
        Parameter('test_id', default='gl_manhattan_off', allowed_values=valid_test_ids,
                  description='ID of the GFXBench test to be run.')
        Parameter('run_timeout', kind=int, default=10 * 60,
                  description="""
                  Time out for workload execution. The workload will be killed if it hasn't completed
                  withint this period.
                  """),
    ]

    aliases = [
        Alias('manhattan', test_id='gl_manhattan'),
        Alias('manhattan_off', test_id='gl_manhattan_off'),
        Alias('manhattan_offscreen', test_id='gl_manhattan_off'),
    ]

    def setup(self, context):
        super(GlbCorp, self).setup(context)
        self.command = self._build_command()
        self.monitor = GlbRunMonitor(self.device)
        self.monitor.start()

    def start_activity(self):
        # Unlike with most other APK workloads, we're invoking the use case
        # directly by starting the activity with appropriate parameters on the
        # command line during execution, so we dont' need to start activity
        # during setup.
        pass

    def run(self, context):
        for _ in xrange(self.times):
            result = self.device.execute(self.command, timeout=self.run_timeout)
            if 'FAILURE' in result:
                raise WorkloadError(result)
            else:
                self.logger.debug(result)
            time.sleep(DELAY)
            self.monitor.wait_for_run_end(self.run_timeout)

    def update_result(self, context):  # NOQA
        super(GlbCorp, self).update_result(context)
        self.monitor.stop()
        iteration = 0
        results = []
        with open(self.logcat_log) as fh:
            try:
                line = fh.next()
                result_lines = []
                while True:
                    if self.result_start_regex.search(line):
                        result_lines.append('{')
                        line = fh.next()
                        while self.preamble_regex.search(line):
                            result_lines.append(self.preamble_regex.sub('', line))
                            line = fh.next()
                        try:
                            result = json.loads(''.join(result_lines))
                            results.append(result)
                            if iteration:
                                suffix = '_{}'.format(iteration)
                            else:
                                suffix = ''
                            for sub_result in result['results']:
                                frames = sub_result['score']
                                elapsed_time = sub_result['elapsed_time'] / 1000
                                fps = frames / elapsed_time
                                context.result.add_metric('score' + suffix, frames, 'frames')
                                context.result.add_metric('fps' + suffix, fps)
                        except ValueError:
                            self.logger.warning('Could not parse result for iteration {}'.format(iteration))
                        result_lines = []
                        iteration += 1
                    line = fh.next()
            except StopIteration:
                pass  # EOF
        if results:
            outfile = os.path.join(context.output_directory, 'glb-results.json')
            with open(outfile, 'wb') as wfh:
                json.dump(results, wfh, indent=4)

    def _build_command(self):
        command_params = []
        command_params.append('-e test_ids "{}"'.format(self.test_id))
        if self.resolution:
            if not self.resolution.endswith('p'):
                self.resolution += 'p'
            for k, v in self.supported_resolutions[self.resolution].iteritems():
                command_params.append('{} {}'.format(k, v))
        return 'am start -W -S -n {}/{} {}'.format(self.package,
                                                   self.activity,
                                                   ' '.join(command_params))


class GlbRunMonitor(threading.Thread):

    regex = re.compile(r'I/Runner\s+\(\s*\d+\): finished:')

    def __init__(self, device):
        super(GlbRunMonitor, self).__init__()
        self.device = device
        self.daemon = True
        self.run_ended = threading.Event()
        self.stop_event = threading.Event()
        # Not using clear_logcat() because command collects directly, i.e. will
        # ignore poller.
        self.device.execute('logcat -c')
        if self.device.adb_name:
            self.command = ['adb', '-s', self.device.adb_name, 'logcat']
        else:
            self.command = ['adb', 'logcat']

    def run(self):
        proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while not self.stop_event.is_set():
            if self.run_ended.is_set():
                time.sleep(DELAY)
            else:
                ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 2)
                if ready:
                    line = ready[0].readline()
                    if self.regex.search(line):
                        self.run_ended.set()

    def stop(self):
        self.stop_event.set()
        self.join()

    def wait_for_run_end(self, timeout):
        self.run_ended.wait(timeout)
        self.run_ended.clear()

