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

from devlib.utils.csvutil import csvwriter

from wa import OutputProcessor, Parameter
from wa.framework.exception import ConfigError
from wa.utils.types import list_of_strings


class CsvReportProcessor(OutputProcessor):

    name = 'csv'
    description = """
    Creates a ``results.csv`` in the output directory containing results for
    all iterations in CSV format, each line containing a single metric.

    """

    parameters = [
        Parameter('use_all_classifiers', kind=bool, default=False,
                  global_alias='use_all_classifiers',
                  description="""
                  If set to ``True``, this will add a column for every classifier
                  that features in at least one collected metric.

                  .. note:: This cannot be ``True`` if ``extra_columns`` is set.

                  """),
        Parameter('extra_columns', kind=list_of_strings,
                  description="""
                  List of classifiers to use as columns.

                   .. note:: This cannot be set if ``use_all_classifiers`` is
                             ``True``.

                  """),
    ]

    def __init__(self, *args, **kwargs):
        super(CsvReportProcessor, self).__init__(*args, **kwargs)
        self.outputs_so_far = []
        self.artifact_added = False

    def validate(self):
        super(CsvReportProcessor, self).validate()
        if self.use_all_classifiers and self.extra_columns:
            msg = 'extra_columns cannot be specified when '\
                  'use_all_classifiers is True'
            raise ConfigError(msg)

    # pylint: disable=unused-argument
    def process_job_output(self, output, target_info, run_output):
        self.outputs_so_far.append(output)
        self._write_outputs(self.outputs_so_far, run_output)
        if not self.artifact_added:
            run_output.add_artifact('run_result_csv', 'results.csv', 'export')
            self.artifact_added = True  # pylint: disable=attribute-defined-outside-init

    def process_run_output(self, output, target_info):  # pylint: disable=unused-argument
        self.outputs_so_far.append(output)
        self._write_outputs(self.outputs_so_far, output)
        if not self.artifact_added:
            output.add_artifact('run_result_csv', 'results.csv', 'export')
            self.artifact_added = True  # pylint: disable=attribute-defined-outside-init

    def _write_outputs(self, outputs, output):
        if self.use_all_classifiers:
            classifiers = set([])
            for out in outputs:
                for metric in out.metrics:
                    classifiers.update(list(metric.classifiers.keys()))
            extra_columns = list(classifiers)
        elif self.extra_columns:
            extra_columns = self.extra_columns
        else:
            extra_columns = []

        outfile = output.get_path('results.csv')
        with csvwriter(outfile) as writer:
            writer.writerow(['id', 'workload', 'iteration', 'metric', ]
                            + extra_columns + ['value', 'units'])

            for o in outputs:
                if o.kind == 'job':
                    header = [o.id, o.label, o.iteration]
                elif o.kind == 'run':
                    # Should be a RunOutput. Run-level metrics aren't attached
                    # to any job so we leave 'id' and 'iteration' blank, and use
                    # the run name for the 'label' field.
                    header = [None, o.info.run_name, None]
                else:
                    raise RuntimeError(
                        'Output of kind "{}" unrecognised by csvproc'.format(o.kind))

                for metric in o.result.metrics:
                    row = (header + [metric.name]
                           + [str(metric.classifiers.get(c, ''))
                           for c in extra_columns]
                           + [str(metric.value), metric.units or ''])
                    writer.writerow(row)
