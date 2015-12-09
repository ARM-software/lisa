#    Copyright 2015 ARM Limited
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

# pylint: disable=attribute-defined-outside-init
import os
import re
import csv
import shutil
import json
import urllib
import stat
from zipfile import is_zipfile, ZipFile

try:
    import pandas as pd
except ImportError:
    pd = None

from wlauto import Workload, Parameter
from wlauto.exceptions import WorkloadError, ConfigError
from wlauto.utils.misc import check_output, get_null, get_meansd
from wlauto.utils.types import numeric


RESULT_REGEX = re.compile(r'RESULT ([^:]+): ([^=]+)\s*=\s*'  # preamble and test/metric name
                          r'(\[([^\]]+)\]|(\S+))'  # value
                          r'\s*(\S+)')  # units
TRACE_REGEX = re.compile(r'Trace saved as ([^\n]+)')

# Trace event that signifies rendition of a Frame
FRAME_EVENT = 'SwapBuffersLatency'

TELEMETRY_ARCHIVE_URL = 'http://storage.googleapis.com/chromium-telemetry/snapshots/telemetry.zip'


class Telemetry(Workload):

    name = 'telemetry'
    description = """
    Executes Google's Telemetery benchmarking framework

    Url: https://www.chromium.org/developers/telemetry

    From the web site:

    Telemetry is Chrome's performance testing framework. It allows you to
    perform arbitrary actions on a set of web pages and report metrics about
    it. The framework abstracts:

      - Launching a browser with arbitrary flags on any platform.
      - Opening a tab and navigating to the page under test.
      - Fetching data via the Inspector timeline and traces.
      - Using Web Page Replay to cache real-world websites so they don't
        change when used in benchmarks.

    Design Principles

      - Write one performance test that runs on all platforms - Windows, Mac,
        Linux, Chrome OS, and Android for both Chrome and ContentShell.
      - Runs on browser binaries, without a full Chromium checkout, and without
        having to build the browser yourself.
      - Use WebPageReplay to get repeatable test results.
      - Clean architecture for writing benchmarks that keeps measurements and
        use cases separate.
      - Run on non-Chrome browsers for comparative studies.

    This instrument runs  telemetry via its ``run_benchmark`` script (which
    must be in PATH or specified using ``run_benchmark_path`` parameter) and
    parses metrics from the resulting output.

    **device setup**

    The device setup will depend on whether you're running a test image (in
    which case little or no setup should be necessary)


    """

    supported_platforms = ['android', 'chromeos']

    parameters = [
        Parameter('run_benchmark_path', default=None,
                  description="""
                  This is the path to run_benchmark script which runs a
                  Telemetry benchmark. If not specified, WA will look for Telemetry in its
                  dependencies; if not found there, Telemetry will be downloaded.
                  """),
        Parameter('test', default='page_cycler.top_10_mobile',
                  description="""
                  Specifies the telemetry test to run.
                  """),
        Parameter('run_benchmark_params', default='',
                  description="""
                  Additional paramters to be passed to ``run_benchmark``.
                  """),
        Parameter('run_timeout', kind=int, default=900,
                  description="""
                  Timeout for execution of the test.
                  """),
        Parameter('extract_fps', kind=bool, default=False,
                  description="""
                  if ``True``, FPS for the run will be computed from the trace (must be enabled).
                  """),
        Parameter('target_config', kind=str, default=None,
                  description="""
                  Manually specify target configuration for telemetry. This must contain
                  --browser option plus any addition options Telemetry requires for a particular
                  target (e.g. --device or --remote)
                  """),
    ]

    def validate(self):
        ret = os.system('{} > {} 2>&1'.format(self.run_benchmark_path, get_null()))
        if ret > 255:
            pass  # telemetry found and appears to be installed properly.
        elif ret == 127:
            raise WorkloadError('run_benchmark not found (did you specify correct run_benchmark_path?)')
        else:
            raise WorkloadError('Unexected error from run_benchmark: {}'.format(ret))
        if self.extract_fps and 'trace' not in self.run_benchmark_params:
            raise ConfigError('"trace" profiler must be enabled in order to extract FPS for Telemetry')
        self._resolve_run_benchmark_path()

    def setup(self, context):
        self.raw_output = None
        self.error_output = None
        self.command = self.build_command()

    def run(self, context):
        self.logger.debug(self.command)
        self.raw_output, self.error_output = check_output(self.command, shell=True, timeout=self.run_timeout, ignore='all')

    def update_result(self, context):  # pylint: disable=too-many-locals
        if self.error_output:
            self.logger.error('run_benchmarks output contained errors:\n' + self.error_output)
        elif not self.raw_output:
            self.logger.warning('Did not get run_benchmark output.')
            return
        raw_outfile = os.path.join(context.output_directory, 'telemetry_raw.out')
        with open(raw_outfile, 'w') as wfh:
            wfh.write(self.raw_output)
        context.add_artifact('telemetry-raw', raw_outfile, kind='raw')

        results, artifacts = parse_telemetry_results(raw_outfile)
        csv_outfile = os.path.join(context.output_directory, 'telemetry.csv')
        with open(csv_outfile, 'wb') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(['kind', 'url', 'iteration', 'value', 'units'])
            for result in results:
                writer.writerows(result.rows)

                for i, value in enumerate(result.values, 1):
                    context.add_metric(result.kind, value, units=result.units,
                                       classifiers={'url': result.url, 'time': i})

            context.add_artifact('telemetry', csv_outfile, kind='data')

        for idx, artifact in enumerate(artifacts):
            if is_zipfile(artifact):
                zf = ZipFile(artifact)
                for item in zf.infolist():
                    zf.extract(item, context.output_directory)
                    zf.close()
                    context.add_artifact('telemetry_trace_{}'.format(idx), path=item.filename, kind='data')
            else:  # not a zip archive
                wa_path = os.path.join(context.output_directory,
                                       os.path.basename(artifact))
                shutil.copy(artifact, wa_path)
                context.add_artifact('telemetry_artifact_{}'.format(idx), path=wa_path, kind='data')

        if self.extract_fps:
            self.logger.debug('Extracting FPS...')
            _extract_fps(context)

    def build_command(self):
        device_opts = ''
        if self.target_config:
            device_opts = self.target_config
        else:
            if self.device.platform == 'chromeos':
                if '--remote' not in self.run_benchmark_params:
                    device_opts += '--remote={} '.format(self.device.host)
                if '--browser' not in self.run_benchmark_params:
                    device_opts += '--browser=cros-chrome '
            elif self.device.platform == 'android':
                if '--device' not in self.run_benchmark_params and self.device.adb_name:
                    device_opts += '--device={} '.format(self.device.adb_name)
                if '--browser' not in self.run_benchmark_params:
                    device_opts += '--browser=android-webview-shell '
            else:
                raise WorkloadError('Unless you\'re running Telemetry on a ChromeOS or Android device, '
                                    'you mast specify target_config option')
        return '{} {} {} {}'.format(self.run_benchmark_path,
                                    self.test,
                                    device_opts,
                                    self.run_benchmark_params)

    def _resolve_run_benchmark_path(self):
        # pylint: disable=access-member-before-definition
        if self.run_benchmark_path:
            if not os.path.exists(self.run_benchmark_path):
                raise ConfigError('run_benchmark path "{}" does not exist'.format(self.run_benchmark_path))
        else:
            self.run_benchmark_path = os.path.join(self.dependencies_directory, 'telemetry', 'run_benchmark')
            self.logger.debug('run_benchmark_path not specified using {}'.format(self.run_benchmark_path))
            if not os.path.exists(self.run_benchmark_path):
                self.logger.debug('Telemetry not found locally; downloading...')
                local_archive = os.path.join(self.dependencies_directory, 'telemetry.zip')
                urllib.urlretrieve(TELEMETRY_ARCHIVE_URL, local_archive)
                zf = ZipFile(local_archive)
                zf.extractall(self.dependencies_directory)
            if not os.path.exists(self.run_benchmark_path):
                raise WorkloadError('Could not download and extract Telemetry')
            old_mode = os.stat(self.run_benchmark_path).st_mode
            os.chmod(self.run_benchmark_path, old_mode | stat.S_IXUSR)


def _extract_fps(context):
    trace_files = [a.path for a in context.iteration_artifacts
                   if a.name.startswith('telemetry_trace_')]
    for tf in trace_files:
        name = os.path.splitext(os.path.basename(tf))[0]
        fps_file = os.path.join(context.output_directory, name + '-fps.csv')
        with open(tf) as fh:
            data = json.load(fh)
            events = pd.Series([e['ts'] for e in data['traceEvents'] if
                                FRAME_EVENT == e['name']])
            fps = (1000000 / (events - events.shift(1)))
            fps.index = events
            df = fps.dropna().reset_index()
            df.columns = ['timestamp', 'fps']
            with open(fps_file, 'w') as wfh:
                df.to_csv(wfh, index=False)
            context.add_artifact('{}_fps'.format(name), fps_file, kind='data')
            context.result.add_metric('{} FPS'.format(name), df.fps.mean(),
                                      units='fps')
            context.result.add_metric('{} FPS (std)'.format(name), df.fps.std(),
                                      units='fps', lower_is_better=True)


class TelemetryResult(object):

    @property
    def average(self):
        return get_meansd(self.values)[0]

    @property
    def std(self):
        return get_meansd(self.values)[1]

    @property
    def rows(self):
        for i, v in enumerate(self.values):
            yield [self.kind, self.url, i, v, self.units]

    def __init__(self, kind=None, url=None, values=None, units=None):
        self.kind = kind
        self.url = url
        self.values = values or []
        self.units = units

    def __str__(self):
        return 'TR({kind},{url},{values},{units})'.format(**self.__dict__)

    __repr__ = __str__


def parse_telemetry_results(filepath):
    results = []
    artifacts = []
    with open(filepath) as fh:
        for line in fh:
            match = RESULT_REGEX.search(line)
            if match:
                result = TelemetryResult()
                result.kind = match.group(1)
                result.url = match.group(2)
                if match.group(4):
                    result.values = map(numeric, match.group(4).split(','))
                else:
                    result.values = [numeric(match.group(5))]
                result.units = match.group(6)
                results.append(result)
            match = TRACE_REGEX.search(line)
            if match:
                artifacts.append(match.group(1))
    return results, artifacts


if __name__ == '__main__':
    import sys  # pylint: disable=wrong-import-order,wrong-import-position
    from pprint import pprint  # pylint: disable=wrong-import-order,wrong-import-position
    path = sys.argv[1]
    pprint(parse_telemetry_results(path))
