#    Copyright 2013-2018 ARM Limited
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

from wa import ApkUiautoWorkload, Parameter
from wa.framework.exception import ConfigError, WorkloadError
from wa.utils.misc import capitalize
from wa.utils.types import version_tuple


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

    supported_versions = ['4.3.2', '4.3.1', '4.2.0', '4.0.1', '3.4.1', '3.0.0', '2']
    package_names = ['com.primatelabs.geekbench', 'com.primatelabs.geekbench3', 'ca.primatelabs.geekbench2']

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

    def update_result_4(self, context):
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
                                 description='Geekbench 4 output from target.')
            context.add_metric(namemify('score', i), data['score'])
            context.add_metric(namemify('multicore_score', i), data['multicore_score'])
            for section in data['sections']:
                context.add_metric(namemify(section['name'] + '_score', i), section['score'])
                for workloads in section['workloads']:
                    workload_name = workloads['name'].replace(" ", "-")
                    context.add_metric(namemify(section['name'] + '_' + workload_name + '_score', i),
                                       workloads['score'])

    update_result_5 = update_result_4


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
    # pylint: disable=C0326
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
    supported_versions = ['4.1.0', '4.3.4', '5.0.0']
    package_names = ['com.primatelabs.geekbench4.corporate', 'com.primatelabs.geekbench5.corporate']
    activity = 'com.primatelabs.geekbench.HomeActivity'

    parameters = [
        Parameter('version', allowed_values=supported_versions, override=True)
    ]


def namemify(basename, i):
    return basename + (' {}'.format(i) if i else '')
