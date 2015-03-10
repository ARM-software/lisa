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


# pylint: disable=R0201
"""
This module contains a few "standard" result processors that write results to
text files in various formats.

"""
import os
import csv
import json

from wlauto import ResultProcessor, settings


class StandardProcessor(ResultProcessor):

    name = 'standard'
    description = """
    Creates a ``result.txt`` file for every iteration that contains metrics
    for that iteration.

    The metrics are written in ::

        metric = value [units]

    format.

    """

    def process_iteration_result(self, result, context):
        outfile = os.path.join(context.output_directory, 'result.txt')
        with open(outfile, 'w') as wfh:
            for metric in result.metrics:
                line = '{} = {}'.format(metric.name, metric.value)
                if metric.units:
                    line = ' '.join([line, metric.units])
                line += '\n'
                wfh.write(line)
        context.add_artifact('iteration_result', 'result.txt', 'export')


class CsvReportProcessor(ResultProcessor):
    """
    Creates a ``results.csv`` in the output directory containing results for
    all iterations in CSV format, each line containing a single metric.

    """

    name = 'csv'

    def process_run_result(self, result, context):
        outfile = os.path.join(settings.output_directory, 'results.csv')
        with open(outfile, 'wb') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(['id', 'workload', 'iteration', 'metric', 'value', 'units'])
            for result in result.iteration_results:
                for metric in result.metrics:
                    row = [result.id, result.spec.label, result.iteration,
                           metric.name, str(metric.value), metric.units or '']
                    writer.writerow(row)
        context.add_artifact('run_result_csv', 'results.csv', 'export')


class JsonReportProcessor(ResultProcessor):
    """
    Creates a ``results.json`` in the output directory containing results for
    all iterations in JSON format.

    """

    name = 'json'

    def process_run_result(self, result, context):
        outfile = os.path.join(settings.output_directory, 'results.json')
        with open(outfile, 'wb') as wfh:
            output = []
            for result in result.iteration_results:
                output.append({
                    'id': result.id,
                    'workload': result.workload.name,
                    'iteration': result.iteration,
                    'metrics': [dict([(k, v) for k, v in m.__dict__.iteritems()
                                      if not k.startswith('_')])
                                for m in result.metrics],
                })
            json.dump(output, wfh, indent=4)
        context.add_artifact('run_result_json', 'results.json', 'export')


class SummaryCsvProcessor(ResultProcessor):
    """
    Similar to csv result processor, but only contains workloads' summary metrics.

    """

    name = 'summary_csv'

    def process_run_result(self, result, context):
        outfile = os.path.join(settings.output_directory, 'summary.csv')
        with open(outfile, 'wb') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(['id', 'workload', 'iteration', 'metric', 'value', 'units'])
            for result in result.iteration_results:
                for metric in result.metrics:
                    if metric.name in result.workload.summary_metrics:
                        row = [result.id, result.workload.name, result.iteration,
                               metric.name, str(metric.value), metric.units or '']
                        writer.writerow(row)
        context.add_artifact('run_result_summary', 'summary.csv', 'export')
