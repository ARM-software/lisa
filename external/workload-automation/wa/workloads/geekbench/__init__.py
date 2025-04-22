#    Copyright 2013-2025 ARM Limited
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
import os
import re
import tempfile
import json
from collections import defaultdict

from wa import Workload, ApkUiautoWorkload, Parameter
from wa.framework.exception import ConfigError, WorkloadError
from wa.utils.misc import capitalize
from wa.utils.types import version_tuple, list_or_integer
from wa.utils.exec_control import once


class Geekbench(ApkUiautoWorkload):

    name = 'geekbench'
    description = """
    Geekbench provides a comprehensive set of benchmarks engineered to quickly
    and accurately measure processor and memory performance.

    http://www.primatelabs.com/geekbench/
    From the website:
    Designed to make benchmarks easy to run and easy to understand, Geekbench
    takes the guesswork out of producing robust and reliable benchmark results.
    Geekbench scores are calibrated against a baseline score of 1,000 (which is
    the score of a single-processor Power Mac G5 @ 1.6GHz). Higher scores are
    better, with double the score indicating double the performance.

    The benchmarks fall into one of four categories:
        - integer performance.
        - floating point performance.
        - memory performance.
        - stream performance.

    Geekbench benchmarks: http://www.primatelabs.com/geekbench/doc/benchmarks.html
    Geekbench scoring methedology:
    http://support.primatelabs.com/kb/geekbench/interpreting-geekbench-scores
    """
    summary_metrics = ['score', 'multicore_score']

    supported_versions = ['6', '5', '4.4.2', '4.4.0', '4.3.4', '4.3.2', '4.3.1', '4.2.0', '4.0.1', '3.4.1', '3.0.0', '2']
    package_names = ['com.primatelabs.geekbench6', 'com.primatelabs.geekbench5', 'com.primatelabs.geekbench', 'com.primatelabs.geekbench3', 'ca.primatelabs.geekbench2']

    begin_regex = re.compile(r'^\s*D/WebViewClassic.loadDataWithBaseURL\(\s*\d+\s*\)'
                             r'\s*:\s*(?P<content>\<.*)\s*$')
    replace_regex = re.compile(r'<[^>]*>')

    parameters = [
        Parameter('version', allowed_values=supported_versions,
                  description='Specifies which version of the workload should be run.',
                  override=True),
        Parameter('loops', kind=int, default=1, aliases=['times'],
                  description=('Specfies the number of times the benchmark will be run in a "tight '
                               'loop", i.e. without performaing setup/teardown inbetween.')),
        Parameter('timeout', kind=int, default=3600,
                  description=('Timeout for a single iteration of the benchmark. This value is '
                               'multiplied by ``times`` to calculate the overall run timeout. ')),
        Parameter('disable_update_result', kind=bool, default=False,
                  description=('If ``True`` the results file will not be pulled from the targets '
                               '``/data/data/com.primatelabs.geekbench`` folder.  This allows the '
                               'workload to be run on unrooted targets and the results extracted '
                               'manually later.')),
    ]

    is_corporate = False

    phones_home = True

    requires_network = True

    def initialize(self, context):
        super(Geekbench, self).initialize(context)
        self.gui.uiauto_params['version'] = self.version
        self.gui.uiauto_params['loops'] = self.loops
        self.gui.uiauto_params['is_corporate'] = self.is_corporate
        self.gui.timeout = self.timeout
        if not self.disable_update_result and not self.target.is_rooted:
            raise WorkloadError(
                'Geekbench workload requires root to collect results. '
                'You can set disable_update_result=True in the workload params '
                'to run without collecting results.')

    def setup(self, context):
        super(Geekbench, self).setup(context)
        self.run_timeout = self.timeout * self.loops

    def update_output(self, context):
        super(Geekbench, self).update_output(context)
        if not self.disable_update_result:
            major_version = version_tuple(self.version)[0]
            update_method = getattr(self, 'update_result_{}'.format(major_version))
            update_method(context)

    def validate(self):
        if (self.loops > 1) and (self.version == '2'):
            raise ConfigError('loops parameter is not supported for version 2 of Geekbench.')

    def update_result_2(self, context):
        score_calculator = GBScoreCalculator()
        score_calculator.parse(self.logcat_log)
        score_calculator.update_results(context)

    def update_result_3(self, context):
        outfile_glob = self.target.path.join(self.target.package_data_directory, self.apk.package, 'files', '*gb3')
        on_target_output_files = [f.strip() for f in self.target.execute('ls {}'.format(outfile_glob),
                                                                         as_root=True).split('\n') if f]
        for i, on_target_output_file in enumerate(on_target_output_files):
            host_temp_file = tempfile.mktemp()
            self.target.pull(on_target_output_file, host_temp_file, as_root=True)
            host_output_file = os.path.join(context.output_directory, os.path.basename(on_target_output_file))
            with open(host_temp_file) as fh:
                data = json.load(fh)
            os.remove(host_temp_file)
            with open(host_output_file, 'w') as wfh:
                json.dump(data, wfh, indent=4)
            context.add_artifact('geekout', host_output_file, kind='data',
                                 description='Geekbench 3 output from target.')
            context.add_metric(namemify('score', i), data['score'])
            context.add_metric(namemify('multicore_score', i), data['multicore_score'])
            for section in data['sections']:
                context.add_metric(namemify(section['name'] + '_score', i), section['score'])
                context.add_metric(namemify(section['name'] + '_multicore_score', i),
                                   section['multicore_score'])

    def update_result(self, context):
        outfile_glob = self.target.path.join(self.target.package_data_directory, self.apk.package, 'files', '*gb*')
        on_target_output_files = [f.strip() for f in self.target.execute('ls {}'.format(outfile_glob),
                                                                         as_root=True).split('\n') if f]
        for i, on_target_output_file in enumerate(on_target_output_files):
            host_temp_file = tempfile.mktemp()
            self.target.pull(on_target_output_file, host_temp_file, as_root=True)
            host_output_file = os.path.join(context.output_directory, os.path.basename(on_target_output_file))
            with open(host_temp_file) as fh:
                data = json.load(fh)
            os.remove(host_temp_file)
            with open(host_output_file, 'w') as wfh:
                json.dump(data, wfh, indent=4)
            context.add_artifact('geekout', host_output_file, kind='data',
                                 description='Geekbench output from target.')
            context.add_metric(namemify('score', i), data['score'])
            context.add_metric(namemify('multicore_score', i), data['multicore_score'])
            for section in data['sections']:
                context.add_metric(namemify(section['name'] + '_score', i), section['score'])
                for workloads in section['workloads']:
                    workload_name = workloads['name'].replace(" ", "-")
                    context.add_metric(namemify(section['name'] + '_' + workload_name + '_score', i),
                                       workloads['score'])

    update_result_4 = update_result
    update_result_5 = update_result
    update_result_6 = update_result


class GBWorkload(object):
    """
    Geekbench workload (not to be confused with WA's workloads). This is a single test run by
    geek bench, such as preforming compression or generating Madelbrot.
    """

    # Index maps onto the hundreds digit of the ID.
    categories = [None, 'integer', 'float', 'memory', 'stream']

    # 2003 entry-level Power Mac G5 is considered to have a baseline score of
    # 1000 for every category.
    pmac_g5_base_score = 1000

    units_conversion_map = {
        'K': 1,
        'M': 1000,
        'G': 1000000,
    }

    def __init__(self, wlid, name, pmac_g5_st_score, pmac_g5_mt_score):
        """
        :param wlid: A three-digit workload ID. Uniquely identifies a workload and also
                     determines the category a workload belongs to.
        :param name: The name of the workload.
        :param pmac_g5_st_score: Score achieved for this workload on 2003 entry-level
                                 Power Mac G5 running in a single thread.
        :param pmac_g5_mt_score: Score achieved for this workload on 2003 entry-level
                                 Power Mac G5 running in multiple threads.
        """
        self.wlid = wlid
        self.name = name
        self.pmac_g5_st_score = pmac_g5_st_score
        self.pmac_g5_mt_score = pmac_g5_mt_score
        self.category = self.categories[int(wlid) // 100]
        self.collected_results = []

    def add_result(self, value, units):
        self.collected_results.append(self.convert_to_kilo(value, units))

    def convert_to_kilo(self, value, units):
        return value * self.units_conversion_map[units[0]]

    def clear(self):
        self.collected_results = []

    def get_scores(self):
        """
        Returns a tuple (single-thraded score, multi-threaded score) for this workload.
        Some workloads only have a single-threaded score, in which case multi-threaded
        score will be ``None``.
        Geekbench will perform four iterations of each workload in single-threaded and,
        for some workloads, multi-threaded configurations. Thus there should always be
        either four or eight scores collected for each workload. Single-threaded iterations
        are always done before multi-threaded, so the ordering of the scores can be used
        to determine which configuration they belong to.
        This method should not be called before score collection has finished.
        """
        no_of_results = len(self.collected_results)
        if no_of_results == 4:
            return (self._calculate(self.collected_results[:4], self.pmac_g5_st_score), None)
        if no_of_results == 8:
            return (self._calculate(self.collected_results[:4], self.pmac_g5_st_score),
                    self._calculate(self.collected_results[4:], self.pmac_g5_mt_score))
        else:
            msg = 'Collected {} results for Geekbench {} workload;'.format(no_of_results, self.name)
            msg += ' expecting either 4 or 8.'
            raise WorkloadError(msg)

    def _calculate(self, values, scale_factor):
        return max(values) * self.pmac_g5_base_score / scale_factor

    def __str__(self):
        return self.name

    __repr__ = __str__


class GBScoreCalculator(object):
    """
    Parses logcat output to extract raw Geekbench workload values and converts them into
    category and overall scores.
    """

    result_regex = re.compile(r'workload (?P<id>\d+) (?P<value>[0-9.]+) '
                              r'(?P<units>[a-zA-Z/]+) (?P<time>[0-9.]+)s')

    # Indicates contribution to the overall score.
    category_weights = {
        'integer': 0.3357231,
        'float': 0.3594,
        'memory': 0.1926489,
        'stream': 0.1054738,
    }

    workloads = [
        #          ID    Name        Power Mac ST  Power Mac MT
        GBWorkload(101, 'Blowfish',         43971,   40979),  # NOQA
        GBWorkload(102, 'Text Compress',    3202,    3280),  # NOQA
        GBWorkload(103, 'Text Decompress',  4112,    3986),  # NOQA
        GBWorkload(104, 'Image Compress',   8272,    8412),  # NOQA
        GBWorkload(105, 'Image Decompress', 16800,   16330),  # NOQA
        GBWorkload(107, 'Lua',              385,     385),  # NOQA

        GBWorkload(201, 'Mandelbrot',       665589,  653746),  # NOQA),
        GBWorkload(202, 'Dot Product',      481449,  455422),  # NOQA,
        GBWorkload(203, 'LU Decomposition', 889933,  877657),  # NOQA
        GBWorkload(204, 'Primality Test',   149394,  185502),  # NOQA
        GBWorkload(205, 'Sharpen Image',    2340,    2304),  # NOQA
        GBWorkload(206, 'Blur Image',       791,     787),  # NOQA

        GBWorkload(302, 'Read Sequential',  1226708, None),  # NOQA
        GBWorkload(304, 'Write Sequential', 683782,  None),  # NOQA
        GBWorkload(306, 'Stdlib Allocate',  3739,    None),  # NOQA
        GBWorkload(307, 'Stdlib Write',     2070681, None),  # NOQA
        GBWorkload(401, 'Stream Copy',      1367892, None),  # NOQA
        GBWorkload(402, 'Stream Scale',     1296053, None),  # NOQA
        GBWorkload(403, 'Stream Add',       1507115, None),  # NOQA
        GBWorkload(404, 'Stream Triad',     1384526, None),  # NOQA
    ]

    def __init__(self):
        self.workload_map = {wl.wlid: wl for wl in self.workloads}

    def parse(self, filepath):
        """
        Extract results from the specified file. The file should contain a logcat log of Geekbench execution.
        Iteration results in the log appear as 'I/geekbench' category entries in the following format::
         |                     worklod ID          value      units   timing
         |                         \-------------    |     ----/     ---/
         |                                      |    |     |         |
         |  I/geekbench(29026): [....] workload 101 132.9 MB/sec 0.0300939s
         |      |               |
         |      |               -----\
         |      label    random crap we don't care about
        """
        for wl in self.workloads:
            wl.clear()
        with open(filepath) as fh:
            for line in fh:
                match = self.result_regex.search(line)
                if match:
                    wkload = self.workload_map[int(match.group('id'))]
                    wkload.add_result(float(match.group('value')), match.group('units'))

    def update_results(self, context):
        """
        http://support.primatelabs.com/kb/geekbench/interpreting-geekbench-2-scores
        From the website:
        Each workload's performance is compared against a baseline to determine a score. These
        scores are averaged together to determine an overall, or Geekbench, score for the system.
        Geekbench uses the 2003 entry-level Power Mac G5 as the baseline with a score of 1,000
        points. Higher scores are better, with double the score indicating double the performance.
        Geekbench provides three different kinds of scores:
            :Workload Scores: Each time a workload is executed Geekbench calculates a score based
                              on the computer's performance compared to the baseline
                              performance. There can be multiple workload scores for the
                              same workload as Geekbench can execute each workload multiple
                              times with different settings. For example, the "Dot Product"
                              workload is executed four times (single-threaded scalar code,
                              multi-threaded scalar code, single-threaded vector code, and
                              multi-threaded vector code) producing four "Dot Product" scores.
            :Section Scores: A section score is the average of all the workload scores for
                             workloads that are part of the section. These scores are useful
                             for determining the performance of the computer in a particular
                             area. See the section descriptions above for a summary on what
                             each section measures.
            :Geekbench Score: The Geekbench score is the weighted average of the four section
                              scores. The Geekbench score provides a way to quickly compare
                              performance across different computers and different platforms
                              without getting bogged down in details.
        """
        scores_by_category = defaultdict(list)
        for wkload in self.workloads:
            st_score, mt_score = wkload.get_scores()
            scores_by_category[wkload.category].append(st_score)
            context.add_metric(wkload.name + ' (single-threaded)', int(st_score))
            if mt_score is not None:
                scores_by_category[wkload.category].append(mt_score)
                context.add_metric(wkload.name + ' (multi-threaded)', int(mt_score))

        overall_score = 0
        for category in scores_by_category:
            scores = scores_by_category[category]
            category_score = sum(scores) / len(scores)
            overall_score += category_score * self.category_weights[category]
            context.add_metric(capitalize(category) + ' Score', int(category_score))
        context.add_metric('Geekbench Score', int(overall_score))


class GeekbenchCorproate(Geekbench):  # pylint: disable=too-many-ancestors
    name = "geekbench-corporate"
    is_corporate = True
    requires_network = False
    supported_versions = ['5.0.3', '5.0.1', '4.1.0', '4.3.4', '5.0.0']
    package_names = ['com.primatelabs.geekbench4.corporate', 'com.primatelabs.geekbench5.corporate']
    activity = 'com.primatelabs.geekbench.HomeActivity'

    parameters = [
        Parameter('version', allowed_values=supported_versions, override=True)
    ]


def namemify(basename, i):
    return basename + (' {}'.format(i) if i else '')


class GeekbenchCmdline(Workload):

    name = "geekbench_cli"
    description = "Workload for running command line version Geekbench"

    gb6_workloads = {
        # Single-Core and Multi-Core
        101: 'File Compression',
        102: 'Navigation',
        103: 'HTML5 Browser',
        104: 'PDF Renderer',
        105: 'Photo Library',
        201: 'Clang',
        202: 'Text Processing',
        203: 'Asset Compression',
        301: 'Object Detection',
        402: 'Object Remover',
        403: 'HDR',
        404: 'Photo Filter',
        501: 'Ray Tracer',
        502: 'Structure from Motion',
        # OpenCL and Vulkan
        303: 'Face Detection',
        406: 'Edge Detection',
        407: 'Gaussian Blur',
        503: 'Feature Matching',
        504: 'Stereo Matching',
        601: 'Particle Physics',
        # Single-Core, Multi-Core, OpenCL, and Vulkan
        302: 'Background Blur',
        401: 'Horizon Detection',
    }

    gb5_workloads = {
        # Single-Core and Multi-Core
        101: 'AES-XTS',
        201: 'Text Compression',
        202: 'Image Compression',
        203: 'Navigation',
        204: 'HTML5',
        205: 'SQLite',
        206: 'PDF Rendering',
        207: 'Text Rendering',
        208: 'Clang',
        209: 'Camera',
        301: 'N-Body Physics',
        302: 'Rigid Body Physics',
        307: 'Image Inpainting',
        308: 'HDR',
        309: 'Ray Tracing',
        310: 'Structure from Motion',
        312: 'Speech Recognition',
        313: 'Machine Learning',
        # OpenCL and Vulkan
        220: 'Sobel',
        221: 'Canny',
        222: 'Stereo Matching',
        230: 'Histogram Equalization',
        304: 'Depth of Field',
        311: 'Feature Matching',
        320: 'Particle Physics',
        321: 'SFFT',
        # Single-Core, Multi-Core, OpenCL, and Vulkan
        303: 'Gaussian Blur',
        305: 'Face Detection',
        306: 'Horizon Detection',
    }

    binary_name = 'geekbench_aarch64'

    allowed_extensions = ['json', 'csv', 'xml', 'html', 'text']

    parameters = [
        Parameter('cpumask', kind=str, default='',
                  description='CPU mask used by taskset.'),
        Parameter('section', kind=int, default=1, allowed_values=[1, 4, 9],
                  description="""Run the specified sections. It should be 1 for CPU benchmarks,
                  4 for OpenCL benchmarks and 9 for Vulkan benchmarks."""),
        Parameter('upload', kind=bool, default=False,
                  description='Upload results to Geekbench Browser'),
        Parameter('is_single_core', kind=bool, default=True,
                  description='Run workload in single-core or multi-core mode.'),
        Parameter('workload', kind=list_or_integer, default=301,
                  description='Specify workload to run'),
        Parameter('iterations', kind=int, default=5,
                  description='Number of iterations'),
        Parameter('workload_gap', kind=int, default=2000,
                  description='N milliseconds gap between workloads'),
        Parameter('output_file', kind=str, default='gb_cli.json',
                  description=f"""Specify the name of the output results file.
                  If it is not specified, the output file will be generated as a JSON file.
                  It can be {', '.join(allowed_extensions)} files."""),
        Parameter('timeout', kind=int, default=2000,
                  description='The test timeout in ms. It should be long for 1000 iterations.'),
        Parameter('version', kind=str, default='6.3.0',
                  description='Specifies which version of the Geekbench should run.'),
    ]

    def __init__(self, target, **kwargs):
        super(GeekbenchCmdline, self).__init__(target, **kwargs)
        self.target_result_json = None
        self.host_result_json = None
        self.workloads = self.gb6_workloads
        self.params = ''
        self.output = ''
        self.target_exec_directory = ''
        self.tar_file_src = ''
        self.tar_file_dst = ''
        self.file_exists = False

    def init_resources(self, context):
        """
        Retrieves necessary files to run the benchmark in TAR format.
        WA will look for `gb_cli_artifacts_<version>.tar` file to deploy them to the
        working directory. If there is no specified version, it will look for version
        6.3.0 by default.
        """
        self.deployable_assets = [''.join(['gb_cli_artifacts', '_', self.version, '.tar'])]

        # Create an executables directory
        self.target_exec_directory = self.target.path.join(self.target.executables_directory, f'gb_cli-{self.version}')
        self.target.execute("mkdir -p {}".format(self.target_exec_directory))

        # Source and Destination paths for the artifacts tar file
        self.tar_file_src = self.target.path.join(self.target.working_directory, self.deployable_assets[0])
        self.tar_file_dst = self.target.path.join(self.target_exec_directory, self.deployable_assets[0])
        # Check the tar file if it already exists
        if self.target.file_exists(self.tar_file_dst):
            self.file_exists = True
        else:
            # Get the assets file
            super(GeekbenchCmdline, self).init_resources(context)

    @once
    def initialize(self, context):
        if self.version[0] == '5':
            self.workloads = self.gb5_workloads
        # If the tar file does not exist in the target, deploy the assets
        if not self.file_exists:
            super(GeekbenchCmdline, self).initialize(context)
            # Move the tar file to the executables directory
            self.target.execute(
                '{} mv {} {}'.format(
                    self.target.busybox, self.tar_file_src, self.tar_file_dst))
            # Extract the tar file
            self.target.execute(
                '{} tar -xf {} -C {}'.format(
                    self.target.busybox, self.tar_file_dst, self.target_exec_directory))

    def setup(self, context):
        super(GeekbenchCmdline, self).setup(context)

        self.params = ''

        self.params += '--section {} '.format(self.section)
        if self.section == 1:
            self.params += '--single-core ' if self.is_single_core else '--multi-core '

        self.params += '--upload ' if self.upload else '--no-upload '

        known_workloads = '\n'.join("{}: {}".format(k, v) for k, v in self.workloads.items())
        if any([t not in self.workloads.keys() for t in self.workload]):
            msg = 'Unknown workload(s) specified. Known workloads: {}'
            raise ValueError(msg.format(known_workloads))

        self.params += '--workload {} '.format(''.join("{},".format(i) for i in self.workload))

        if self.iterations:
            self.params += '--iterations {} '.format(self.iterations)

        if self.workload_gap:
            self.params += '--workload-gap {} '.format(self.workload_gap)

        extension = os.path.splitext(self.output_file)[1][1:]
        if self.output_file and extension not in self.allowed_extensions:
            msg = f"No allowed extension specified. Allowed extensions: {', '.join(self.allowed_extensions)}"
            raise ValueError(msg)
        elif self.output_file:
            # Output results file with the given name and extension
            self.target_result_json = os.path.join(self.target_exec_directory, self.output_file)
            self.params += '--export-{} {}'.format(extension, self.target_result_json)
            self.host_result_json = os.path.join(context.output_directory, self.output_file)
        else:
            # The output file is not specified
            self.target_result_json = os.path.join(self.target_exec_directory, self.output_file)
            self.params += '--save {}'.format(self.target_result_json)
            self.host_result_json = os.path.join(context.output_directory, self.output_file)

    def run(self, context):
        super(GeekbenchCmdline, self).run(context)
        taskset = f"taskset {self.cpumask}" if self.cpumask else ""
        binary = self.target.path.join(self.target_exec_directory, self.binary_name)
        cmd = '{} {} {}'.format(taskset, binary, self.params)

        try:
            self.output = self.target.execute(cmd, timeout=self.timeout, as_root=True)
        except KeyboardInterrupt:
            self.target.killall(self.binary_name)
            raise

    def update_output(self, context):
        super(GeekbenchCmdline, self).update_output(context)
        if not self.output:
            return
        for workload in self.workload:
            scores = []
            matches = re.findall(self.workloads[workload] + '(.+\d)', self.output)
            for match in matches:
                scores.append(int(re.search(r'\d+', match).group(0)))
            if self.section == 4:
                context.add_metric("OpenCL Score " + self.workloads[workload], scores[0])
            elif self.section == 9:
                context.add_metric("Vulkan Score " + self.workloads[workload], scores[0])
            else:
                context.add_metric("Single-Core Score " + self.workloads[workload], scores[0])
                if not self.is_single_core:
                    context.add_metric("Multi-Core Score " + self.workloads[workload], scores[1])

    def extract_results(self, context):
        # Extract results on the target
        super(GeekbenchCmdline, self).extract_results(context)
        self.target.pull(self.target_result_json, self.host_result_json)
        context.add_artifact('GeekbenchCmdline_results', self.host_result_json, kind='raw')

    @once
    def finalize(self, context):
        if self.cleanup_assets:
            self.target.remove(self.target_exec_directory)
