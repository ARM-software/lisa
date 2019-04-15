#    Copyright 2018 ARM Limited
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

# pylint: disable=E1101,W0201

import os
import re

import pandas as pd

from wa import Workload, Parameter, Alias, Executable
from wa.utils.types import numeric


class Deepbench(Workload):

    name = 'deepbench'
    description = """
    Benchmarks operations that are important to deep learning. Including GEMM
    and convolution.

    The benchmark and its documentation are available here:

        https://github.com/baidu-research/DeepBench


    .. note:: parameters of matrices used in each sub-test are added as
              classifiers to the metrics. See the benchmark documentation
              for the explanation of the various parameters

    .. note:: at the moment only the "Arm Benchmarks" subset of DeepBench
              is supported.

    """

    parameters = [
        Parameter('test', default='gemm',
                  allowed_values=['gemm', 'conv', 'sparse'],
                  description='''
                  Specifies which of the available benchmarks will be run.

                  gemm
                    Performs GEneral Matrix Multiplication of dense matrices
                    of varying sizes.

                  conv
                    Performs convolutions on inputs in NCHW format.

                  sparse
                    Performs GEneral Matrix Multiplication of sparse matrices
                    of varying sizes, and compares them to corresponding dense
                    operations.

                  '''),
    ]

    aliases = [
        Alias('deep-gemm', test='gemm'),
        Alias('deep-conv', test='conv'),
        Alias('deep-sparse', test='sparse'),
    ]

    test_metrics = {
        'gemm': ['time (msec)', 'GOPS'],
        'conv': ['fwd_time (usec)'],
        'sparse': ['sparse time (usec)', 'dense time (usec)', 'speedup'],
    }

    lower_is_better = {
        'time (msec)': True,
        'GOPS': False,
        'fwd_time (usec)': True,
        'sparse time (usec)': True,
        'dense time (usec)': True,
        'speedup': False,
    }

    installed = {}

    def initialize(self, context):
        self.exe_name = '{}_bench'.format(self.test)
        if self.exe_name not in self.installed:
            resource = Executable(self, self.target.abi, self.exe_name)
            host_exe = context.get_resource(resource)
            self.target.killall(self.exe_name)
            self.installed[self.exe_name] = self.target.install(host_exe)
        self.target_exe = self.installed[self.exe_name]

    def setup(self, context):
        self.target.killall(self.exe_name)

    def run(self, context):
        self.output = None
        try:
            timeout = 10800
            self.output = self.target.execute(self.target_exe, timeout=timeout)
        except KeyboardInterrupt:
            self.target.killall(self.exe_name)
            raise

    def extract_results(self, context):
        if self.output:
            outfile = os.path.join(context.output_directory, '{}.output'.format(self.test))
            with open(outfile, 'w') as wfh:
                wfh.write(self.output)
            context.add_artifact('deepbench-output', outfile, 'raw', "deepbench's stdout")

    def update_output(self, context):
        raw_file = context.get_artifact_path('deepbench-output')
        if not raw_file:
            return
        table = read_result_table(raw_file)
        for _, row in table.iterrows():
            items = dict(row)

            metrics = []
            for metric_name in self.test_metrics[self.test]:
                metrics.append((metric_name, items.pop(metric_name)))

            for name, value in metrics:
                context.add_metric(name, value,
                                   lower_is_better=self.lower_is_better[name],
                                   classifiers=items)

    def finalize(self, context):
        if self.cleanup_assets:
            if self.exe_name in self.installed:
                self.target.uninstall(self.exe_name)
                del self.installed[self.exe_name]


def numeric_best_effort(value):
    try:
        return numeric(value)
    except ValueError:
        return value


def read_result_table(filepath):
    columns = []
    entries = []
    with open(filepath) as fh:
        try:
            # fast-forward to the header
            line = next(fh)
            while not line.startswith('----'):
                line = next(fh)
            header_line = next(fh)
            haader_sep = re.compile(r'(?<=[) ]) ')
            # Since headers can contain spaces, use two spaces as column separator
            parts = [p.strip() for p in haader_sep.split(header_line)]
            columns = [p for p in parts if p]

            line = next(fh)
            while line.strip():
                if line.startswith('----'):
                    line = next(fh)
                row = [numeric_best_effort(i) for i in line.strip().split()]
                entries.append(row)
                line = next(fh)
        except StopIteration:
            pass

    return pd.DataFrame(entries, columns=columns)
