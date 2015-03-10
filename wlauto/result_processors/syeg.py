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


#pylint: disable=E1101,W0201
import os
import csv
import math
import re

from wlauto import ResultProcessor, Parameter, File
from wlauto.utils.misc import get_meansd


class SyegResultProcessor(ResultProcessor):

    name = 'syeg_csv'
    description = """
    Generates a CSV results file in the format expected by SYEG toolchain.

    Multiple iterations get parsed into columns, adds additional columns for mean
    and standard deviation, append number of threads to metric names (where
    applicable) and add some metadata based on external mapping files.

    """

    parameters = [
        Parameter('outfile', kind=str, default='syeg_out.csv',
                  description='The name of the output CSV file.'),
    ]

    def initialize(self, context):
        self.levelmap = self._read_map(context, 'final_sub.csv',
                                       'Could not find metrics level mapping.')
        self.typemap = self._read_map(context, 'types.csv',
                                      'Could not find benchmark suite types mapping.')

    def process_run_result(self, result, context):
        syeg_results = {}
        max_iterations = max(ir.iteration for ir in result.iteration_results)
        for ir in result.iteration_results:
            for metric in ir.metrics:
                key = ir.spec.label + metric.name
                if key not in syeg_results:
                    syeg_result = SyegResult(max_iterations)
                    syeg_result.suite = ir.spec.label
                    syeg_result.version = getattr(ir.workload, 'apk_version', None)
                    syeg_result.test = metric.name
                    if hasattr(ir.workload, 'number_of_threads'):
                        syeg_result.test += ' NT {} (Iterations/sec)'.format(ir.workload.number_of_threads)
                    syeg_result.final_sub = self.levelmap.get(metric.name)
                    syeg_result.lower_is_better = metric.lower_is_better
                    syeg_result.device = context.device.name
                    syeg_result.type = self._get_type(ir.workload.name, metric.name)
                    syeg_results[key] = syeg_result
                syeg_results[key].runs[ir.iteration - 1] = metric.value

        columns = ['device', 'suite', 'test', 'version', 'final_sub', 'best', 'average', 'deviation']
        columns += ['run{}'.format(i + 1) for i in xrange(max_iterations)]
        columns += ['type', 'suite_version']

        outfile = os.path.join(context.output_directory, self.outfile)
        with open(outfile, 'wb') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(columns)
            for syeg_result in syeg_results.values():
                writer.writerow([getattr(syeg_result, c) for c in columns])
        context.add_artifact('syeg_csv', outfile, 'export')

    def _get_type(self, workload, metric):
        metric = metric.lower()
        type_ = self.typemap.get(workload)
        if type_ == 'mixed':
            if 'native' in metric:
                type_ = 'native'
            if ('java' in metric) or ('dalvik' in metric):
                type_ = 'dalvik'
        return type_

    def _read_map(self, context, filename, errormsg):
        mapfile = context.resolver.get(File(self, filename))
        if mapfile:
            with open(mapfile) as fh:
                reader = csv.reader(fh)
                return dict([c.strip() for c in r] for r in reader)
        else:
            self.logger.warning(errormsg)
            return {}


class SyegResult(object):

    @property
    def average(self):
        if not self._mean:
            self._mean, self._sd = get_meansd(self.run_values)
        return self._mean

    @property
    def deviation(self):
        if not self._sd:
            self._mean, self._sd = get_meansd(self.run_values)
        return self._sd

    @property
    def run_values(self):
        return [r for r in self.runs if not math.isnan(r)]

    @property
    def best(self):
        if self.lower_is_better:
            return min(self.run_values)
        else:
            return max(self.run_values)

    @property
    def suite_version(self):
        return ' '.join(map(str, [self.suite, self.version]))

    def __init__(self, max_iter):
        self.runs = [float('nan') for _ in xrange(max_iter)]
        self.device = None
        self.suite = None
        self.test = None
        self.version = None
        self.final_sub = None
        self.lower_is_better = None
        self.type = None
        self._mean = None
        self._sd = None

    def __getattr__(self, name):
        match = re.search(r'run(\d+)', name)
        if not match:
            raise AttributeError(name)
        return self.runs[int(match.group(1)) - 1]


