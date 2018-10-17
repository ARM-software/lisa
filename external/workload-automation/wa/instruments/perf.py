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


# pylint: disable=unused-argument
import os
import re

from devlib.trace.perf import PerfCollector

from wa import Instrument, Parameter
from wa.utils.types import list_or_string, list_of_strs

PERF_COUNT_REGEX = re.compile(r'^(CPU\d+)?\s*(\d+)\s*(.*?)\s*(\[\s*\d+\.\d+%\s*\])?\s*$')


class PerfInstrument(Instrument):

    name = 'perf'
    description = """
    Perf is a Linux profiling with performance counters.

    Performance counters are CPU hardware registers that count hardware events
    such as instructions executed, cache-misses suffered, or branches
    mispredicted. They form a basis for profiling applications to trace dynamic
    control flow and identify hotspots.

    pref accepts options and events. If no option is given the default '-a' is
    used. For events, the default events are migrations and cs. They both can
    be specified in the config file.

    Events must be provided as a list that contains them and they will look like
    this ::

        perf_events = ['migrations', 'cs']

    Events can be obtained by typing the following in the command line on the
    device ::

        perf list

    Whereas options, they can be provided as a single string as following ::

        perf_options = '-a -i'

    Options can be obtained by running the following in the command line ::

        man perf-stat
    """

    parameters = [
        Parameter('events', kind=list_of_strs, default=['migrations', 'cs'],
                  global_alias='perf_events',
                  constraint=(lambda x: x, 'must not be empty.'),
                  description="""Specifies the events to be counted."""),
        Parameter('optionstring', kind=list_or_string, default='-a',
                  global_alias='perf_options',
                  description="""Specifies options to be used for the perf command. This
                  may be a list of option strings, in which case, multiple instances of perf
                  will be kicked off -- one for each option string. This may be used to e.g.
                  collected different events from different big.LITTLE clusters.
                  """),
        Parameter('labels', kind=list_of_strs, default=None,
                  global_alias='perf_labels',
                  description="""Provides labels for pref output. If specified, the number of
                  labels must match the number of ``optionstring``\ s.
                  """),
        Parameter('force_install', kind=bool, default=False,
                  description="""
                  always install perf binary even if perf is already present on the device.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(PerfInstrument, self).__init__(target, **kwargs)
        self.collector = None

    def initialize(self, context):
        self.collector = PerfCollector(self.target,
                                       self.events,
                                       self.optionstring,
                                       self.labels,
                                       self.force_install)

    def setup(self, context):
        self.collector.reset()

    def start(self, context):
        self.collector.start()

    def stop(self, context):
        self.collector.stop()

    def update_output(self, context):
        self.logger.info('Extracting reports from target...')
        outdir = os.path.join(context.output_directory, 'perf')
        self.collector.get_trace(outdir)

        for host_file in os.listdir(outdir):
            label = host_file.split('.out')[0]
            host_file_path = os.path.join(outdir, host_file)
            context.add_artifact(label, host_file_path, 'raw')
            with open(host_file_path) as fh:
                in_results_section = False
                for line in fh:
                    if 'Performance counter stats' in line:
                        in_results_section = True
                        next(fh)  # skip the following blank line
                    if in_results_section:
                        if not line.strip():  # blank line
                            in_results_section = False
                            break
                        else:
                            line = line.split('#')[0]  # comment
                            match = PERF_COUNT_REGEX.search(line)
                            if match:
                                classifiers = {}
                                cpu = match.group(1)
                                if cpu is not None:
                                    classifiers['cpu'] = int(cpu.replace('CPU', ''))
                                count = int(match.group(2))
                                metric = '{}_{}'.format(label, match.group(3))
                                context.add_metric(metric, count, classifiers=classifiers)

    def teardown(self, context):
        self.collector.reset()
