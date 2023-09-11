#    Copyright 2023 ARM Limited
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

from devlib import PerfettoCollector

from wa import Instrument, Parameter
from wa.framework.instrument import very_slow, is_installed
from wa.framework.exception import InstrumentError

OUTPUT_PERFETTO_TRACE = 'devlib-trace.perfetto-trace'
PERFETTO_CONFIG_FILE = 'config.pbtx'


class PerfettoInstrument(Instrument):
    name = 'perfetto'
    description = """
        perfetto is an instrument that interacts with Google's Perfetto tracing
        infrastructure.

        From Perfetto's website:
        Perfetto is a production-grade open-source stack for performance instrumentation and trace analysis.
        It offers services and libraries for recording system-level and app-level traces, native + java heap profiling,
        a library for analyzing traces using SQL and a web-based UI to visualize and explore multi-GB traces.

        The instrument either requires Perfetto to be present on the target device or the standalone tracebox binary
        to be built from source and included in devlib's Package Bin directory.
        For more information, consult the PerfettoCollector documentation in devlib.

        More information can be found on https://perfetto.dev/
    """

    parameters = [
        Parameter('config', kind=str, mandatory=True,
                  description="""
                  Path to the Perfetto trace config file.

                  All the Perfetto-specific tracing configuration should be done inside
                  that file. This config option should just take a full
                  filesystem path to where the config can be found.
                  """),
        Parameter('force_tracebox', kind=bool, default=False,
                  description="""
                  Install tracebox even if traced is already running on the target device.
                  If set to true, the tracebox binary needs to be placed in devlib's Package Bin directory.
                  """)
    ]

    def __init__(self, target, **kwargs):
        super(PerfettoInstrument, self).__init__(target, **kwargs)
        self.collector = None

    def initialize(self, context):  # pylint: disable=unused-argument
        self.target_config = self.target.path.join(self.target.working_directory, PERFETTO_CONFIG_FILE)
        # push the config file to target
        self.target.push(self.config, self.target_config)
        collector_params = dict(
            config=self.target_config,
            force_tracebox=self.force_tracebox
        )
        self.collector = PerfettoCollector(self.target, **collector_params)

    @very_slow
    def start(self, context):  # pylint: disable=unused-argument
        self.collector.start()

    @very_slow
    def stop(self, context):  # pylint: disable=unused-argument
        self.collector.stop()

    def update_output(self, context):
        self.logger.info('Extracting Perfetto trace from target...')
        outfile = os.path.join(context.output_directory, OUTPUT_PERFETTO_TRACE)
        self.collector.set_output(outfile)
        self.collector.get_data()
        context.add_artifact('perfetto-bin', outfile, 'data')

    def teardown(self, context):  # pylint: disable=unused-argument
        self.target.remove(self.collector.target_output_file)

    def finalize(self, context):  # pylint: disable=unused-argument
        self.target.remove(self.target_config)

    def validate(self):
        if is_installed('trace-cmd'):
            raise InstrumentError('perfetto cannot be used at the same time as trace-cmd')
        if not os.path.isfile(self.config):
            raise InstrumentError('perfetto config file not found at "{}"'.format(self.config))
