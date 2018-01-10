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
import json
import re

from HTMLParser import HTMLParser

from wa import ApkUiautoWorkload, Parameter
from wa.utils.types import list_of_strs
from wa.framework.exception import WorkloadError


class Vellamo(ApkUiautoWorkload):

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
    package_names = ['com.quicinc.vellamo']
    run_timeout = 15 * 60
    benchmark_types = {
        '2.0.3': ['html5', 'metal'],
        '3.0': ['Browser', 'Metal', 'Multi'],
        '3.2.4': ['Browser', 'Metal', 'Multi'],
    }
    valid_versions = benchmark_types.keys()
    summary_metrics = None

    parameters = [
        Parameter('version', kind=str, allowed_values=valid_versions, default=sorted(benchmark_types, reverse=True)[0], override=True,
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


    def setup(self, context):
        self.gui.uiauto_params['version'] = self.version
        self.gui.uiauto_params['browserToUse'] = self.browser
        self.gui.uiauto_params['metal'] = 'Metal' in self.benchmarks
        self.gui.uiauto_params['browser'] = 'Browser' in self.benchmarks
        self.gui.uiauto_params['multicore'] = 'Multi' in self.benchmarks
        super(Vellamo, self).setup(context)

    def validate(self):
        super(Vellamo, self).validate()
        if self.version == '2.0.3' or not self.benchmarks:  # pylint: disable=access-member-before-definition
            self.benchmarks = self.benchmark_types[self.version]  # pylint: disable=attribute-defined-outside-init
        else:
            for benchmark in self.benchmarks:
                if benchmark not in self.benchmark_types[self.version]:
                    raise WorkloadError('Version {} does not support {} benchmarks'.format(self.version, benchmark))

    def update_output(self, context):
        super(Vellamo, self).update_output(context)

        # Get total scores from logcat
        self.non_root_update_output(context)

        if not self.target.is_rooted:
            return
        elif self.version == '3.0.0':
            self.update_output_v3(context)
        elif self.version == '3.2.4':
            self.update_output_v3_2(context)

    def update_output_v3(self, context):
        for test in self.benchmarks:  # Get all scores from HTML files
            filename = None
            if test == "Browser":
                result_folder = self.target.path.join(self.target.package_data_directory,
                                                      self.apk.apk_info.package, 'files')
                for result_file in self.target.listdir(result_folder, as_root=True):
                    if result_file.startswith("Browser"):
                        filename = result_file
            else:
                filename = '{}_results.html'.format(test)

            device_file = self.target.path.join(self.target.package_data_directory,
                                                self.apk.apk_info.package, 'files', filename)
            host_file = os.path.join(context.output_directory, filename)
            self.target.pull(device_file, host_file, as_root=True)
            with open(host_file) as fh:
                parser = VellamoResultParser()
                parser.feed(fh.read())
                for benchmark in parser.benchmarks:
                    benchmark.name = benchmark.name.replace(' ', '_')
                    context.add_metric('{}_Total'.format(benchmark.name),
                                                                benchmark.score)
                    for name, score in benchmark.metrics.items():
                        name = name.replace(' ', '_')
                        context.add_metric('{}_{}'.format(benchmark.name,
                                                                 name), score)
            context.add_artifact('vellamo_output', kind='raw',
                                           path=filename)

    def update_output_v3_2(self, context):
        device_file = self.target.path.join(self.target.package_data_directory,
                                            self.apk.apk_info.package,
                                            'files',
                                            'chapterscores.json')
        host_file = os.path.join(context.output_directory, 'vellamo.json')
        self.target.pull(device_file, host_file, as_root=True)
        context.add_artifact('vellamo_output', kind='raw', path=host_file)
        # context.add_iteration_artifact('vellamo_output', kind='raw', path=host_file)
        with open(host_file) as results_file:
            data = json.load(results_file)
            for chapter in data:
                for result in chapter['benchmark_results']:
                    name = result['id']
                    score = result['score']
                    context.add_metric(name, score)

    def non_root_update_output(self, context):
        failed = []
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file) as fh:
            iteration_result_regex = re.compile("VELLAMO RESULT: (Browser|Metal|Multicore) (\d+)")
            for line in fh:
                if 'VELLAMO ERROR:' in line:
                    msg = "Browser crashed during benchmark, results may not be accurate"
                    self.logger.warning(msg)
                result = iteration_result_regex.findall(line)
                if result:
                    for (metric, score) in result:
                        if not score:
                            failed.append(metric)
                        else:
                            context.add_metric(metric, score)
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
