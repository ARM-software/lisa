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
import logging
from HTMLParser import HTMLParser
from collections import defaultdict, OrderedDict

from wlauto import AndroidUiAutoBenchmark, Parameter
from wlauto.utils.types import list_of_strs, numeric
from wlauto.exceptions import WorkloadError


#pylint: disable=no-member
class Vellamo(AndroidUiAutoBenchmark):

    name = 'vellamo'
    description = """
    Android benchmark designed by Qualcomm.

    Vellamo began as a mobile web benchmarking tool that today has expanded
    to include three primary chapters. The Browser Chapter evaluates mobile
    web browser performance, the Multicore chapter measures the synergy of
    multiple CPU cores, and the Metal Chapter measures the CPU subsystem
    performance of mobile processors. Through click-and-go test suites,
    organized by chapter, Vellamo is designed to evaluate: UX, 3D graphics,
    and memory read/write and peak bandwidth performance, and much more!

    Note: Vellamo v3.0 fails to run on Juno

    """
    package = 'com.quicinc.vellamo'
    run_timeout = 15 * 60
    benchmark_types = {
        '2.0.3': ['html5', 'metal'],
        '3.0': ['Browser', 'Metal', 'Multi'],
    }
    valid_versions = benchmark_types.keys()
    summary_metrics = None

    parameters = [
        Parameter('version', kind=str, allowed_values=valid_versions, default=sorted(benchmark_types, reverse=True)[0],
                  description=('Specify the version of Vellamo to be run. '
                               'If not specified, the latest available version will be used.')),
        Parameter('benchmarks', kind=list_of_strs, allowed_values=benchmark_types['3.0'], default=benchmark_types['3.0'],
                  description=('Specify which benchmark sections of Vellamo to be run. Only valid on version 3.0 and newer.'
                               '\nNOTE: Browser benchmark can be problematic and seem to hang,'
                               'just wait and it will progress after ~5 minutes')),
        Parameter('browser', kind=int, default=1,
                  description=('Specify which of the installed browsers will be used for the tests. The number refers to '
                               'the order in which browsers are listed by Vellamo. E.g. ``1`` will select the first browser '
                               'listed, ``2`` -- the second, etc. Only valid for version ``3.0``.'))
    ]

    def __init__(self, device, **kwargs):
        super(Vellamo, self).__init__(device, **kwargs)
        if self.version == '2.0.3':
            self.activity = 'com.quicinc.vellamo.VellamoActivity'
        if self.version == '3.0':
            self.activity = 'com.quicinc.vellamo.main.MainActivity'
        self.summary_metrics = self.benchmark_types[self.version]

    def setup(self, context):
        self.uiauto_params['version'] = self.version
        self.uiauto_params['browserToUse'] = self.browser
        self.uiauto_params['metal'] = 'Metal' in self.benchmarks
        self.uiauto_params['browser'] = 'Browser' in self.benchmarks
        self.uiauto_params['multicore'] = 'Multi' in self.benchmarks
        super(Vellamo, self).setup(context)

    def validate(self):
        super(Vellamo, self).validate()
        if self.version == '2.0.3' or not self.benchmarks or self.benchmarks == []:  # pylint: disable=access-member-before-definition
            self.benchmarks = self.benchmark_types[self.version]  # pylint: disable=attribute-defined-outside-init
        else:
            for benchmark in self.benchmarks:
                if benchmark not in self.benchmark_types[self.version]:
                    raise WorkloadError('Version {} does not support {} benchmarks'.format(self.version, benchmark))

    def update_result(self, context):
        super(Vellamo, self).update_result(context)

        # Get total scores from logcat
        self.non_root_update_result(context)

        if not self.device.is_rooted:
            return

        for test in self.benchmarks:  # Get all scores from HTML files
            filename = None
            if test == "Browser":
                result_folder = self.device.path.join(self.device.package_data_directory, self.package, 'files')
                for result_file in self.device.listdir(result_folder, as_root=True):
                    if result_file.startswith("Browser"):
                        filename = result_file
            else:
                filename = '{}_results.html'.format(test)

            device_file = self.device.path.join(self.device.package_data_directory, self.package, 'files', filename)
            host_file = os.path.join(context.output_directory, filename)
            self.device.pull_file(device_file, host_file, as_root=True)
            with open(host_file) as fh:
                parser = VellamoResultParser()
                parser.feed(fh.read())
                for benchmark in parser.benchmarks:
                    benchmark.name = benchmark.name.replace(' ', '_')
                    context.result.add_metric('{}_Total'.format(benchmark.name), benchmark.score)
                    for name, score in benchmark.metrics.items():
                        name = name.replace(' ', '_')
                        context.result.add_metric('{}_{}'.format(benchmark.name, name), score)
            context.add_iteration_artifact('vellamo_output', kind='raw', path=filename)

    def non_root_update_result(self, context):
        failed = []
        with open(self.logcat_log) as logcat:
            metrics = OrderedDict()
            for line in logcat:
                if 'VELLAMO RESULT:' in line:
                    info = line.split(':')
                    parts = info[2].split(" ")
                    metric = parts[1].strip()
                    value = int(parts[2].strip())
                    metrics[metric] = value
                if 'VELLAMO ERROR:' in line:
                    self.logger.warning("Browser crashed during benchmark, results may not be accurate")
            for key, value in metrics.iteritems():
                key = key.replace(' ', '_')
                context.result.add_metric(key, value)
                if value == 0:
                    failed.append(key)
        if failed:
            raise WorkloadError("The following benchmark groups failed: {}".format(", ".join(failed)))


class VellamoResult(object):

    def __init__(self, name):
        self.name = name
        self.score = None
        self.metrics = {}

    def add_metric(self, data):
        split_data = data.split(":")
        name = split_data[0].strip()
        score = split_data[1].strip()

        if name in self.metrics:
            raise KeyError("A metric of that name is already present")
        self.metrics[name] = float(score)


class VellamoResultParser(HTMLParser):

    class StopParsingException(Exception):
        pass

    def __init__(self):
        HTMLParser.__init__(self)
        self.inside_div = False
        self.inside_span = 0
        self.inside_li = False
        self.got_data = False
        self.failed = False
        self.benchmarks = []

    def feed(self, text):
        try:
            HTMLParser.feed(self, text)
        except self.StopParsingException:
            pass

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            self.inside_div = True
        if tag == 'span':
            self.inside_span += 1
        if tag == 'li':
            self.inside_li = True

    def handle_endtag(self, tag):
        if tag == 'div':
            self.inside_div = False
            self.inside_span = 0
            self.got_data = False
            self.failed = False
        if tag == 'li':
            self.inside_li = False

    def handle_data(self, data):
        if self.inside_div and not self.failed:
            if "Problem" in data:
                self.failed = True
            elif self.inside_span == 1:
                self.benchmarks.append(VellamoResult(data))
            elif self.inside_span == 3 and not self.got_data:
                self.benchmarks[-1].score = int(data)
                self.got_data = True
            elif self.inside_li and self.got_data:
                if 'failed' not in data:
                    self.benchmarks[-1].add_metric(data)
                else:
                    self.failed = True

