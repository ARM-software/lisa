#    Copyright 2016-2018 ARM Limited
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

import os
import re
import sys

from wa import Workload, Parameter, Executable, ConfigError, WorkloadError
from wa.utils.exec_control import once
from wa.utils.types import list_of_ints


phase_start_regex = re.compile(r"Starting phase\s+(?P<phase>\d+)")
counter_value_regex = re.compile(r"Thread\s+(?P<thread>\d+)\s+(?P<name>\w+)\svalue\s+\=\s+(?P<value>\d+)")
duration_regex = re.compile(r"Phase\s+(?P<phase>\d+)[\s\w\(\)]+\:\s+(?P<duration>\d+)")


class Meabo(Workload):

    name = 'meabo'
    description = '''
    A multi-phased multi-purpose micro-benchmark. The micro-benchmark is
    composed of 10 phases that perform various generic calculations (from
    memory to compute intensive).

    It is a highly configurable tool which can be used for energy efficiency
    studies, ARM big.LITTLE Linux scheduler analysis and DVFS studies. It can
    be used for other benchmarking as well.

    All floating-point calculations are double-precision.

    |   Phase 1: Floating-point & integer computations with good data locality
    |   Phase 2: Vector multiplication & addition, 1 level of indirection in 1
    |            source vector
    |   Phase 3: Vector scalar addition and reductions
    |   Phase 4: Vector addition
    |   Phase 5: Vector addition, 1 level of indirection in both source vectors
    |   Phase 6: Sparse matrix-vector multiplication
    |   Phase 7: Linked-list traversal
    |   Phase 8: Electrostatic force calculations
    |   Phase 9: Palindrome calculations
    |   Phase 10: Random memory accesses

    For more details and benchmark source, see:

        https://github.com/ARM-software/meabo

    .. note:: current implementation of automation relies on the executable to
              be either statically linked or for all necessary depencies to be
              installed on the target.

    '''

    parameters = [
        Parameter(
            'array_size',
            kind=int,
            description='''
            Size of arrays used in Phases 1, 2, 3, 4 and 5.
            ''',
            constraint=lambda x: x > 0,
            default=1048576,
        ),
        Parameter(
            'num_rows',
            kind=int,
            aliases=['nrow'],
            description='''
            Number of rows for the sparse matrix used in Phase 6.
            ''',
            constraint=lambda x: x > 0,
            default=16384,
        ),
        Parameter(
            'num_cols',
            kind=int,
            aliases=['ncol'],
            description='''
            Number of columns for the sparse matrix used in Phase 6.
            ''',
            constraint=lambda x: x > 0,
            default=16384,
        ),
        Parameter(
            'loops',
            kind=int,
            aliases=['num_iterations'],
            description='''
            Number of iterations that core loop is executed.
            ''',
            constraint=lambda x: x > 0,
            default=1000,
        ),
        Parameter(
            'block_size',
            kind=int,
            description='''
            Block size used in Phase 1.
            ''',
            constraint=lambda x: x > 0,
            default=8,
        ),
        Parameter(
            'num_cpus',
            kind=int,
            description='''
            Number of total CPUs that the application can bind threads to.
            ''',
            constraint=lambda x: x > 0,
            default=6,
        ),
        Parameter(
            'per_phase_cpu_ids',
            kind=list_of_ints,
            description='''
            Sets which cores each phase is run on.
            ''',
            constraint=lambda x: all(v >= -1 for v in x),
            default=[-1] * 10,
        ),
        Parameter(
            'num_hwcntrs',
            kind=int,
            description='''
            Only available when using PAPI. Controls how many hardware counters
            PAPI will get access to.
            ''',
            constraint=lambda x: x >= 0,
            default=7,
        ),
        Parameter(
            'run_phases',
            kind=list_of_ints,
            description='''
            Controls which phases to run.
            ''',
            constraint=lambda x: all(0 < v <= 10 for v in x),
            default=list(range(1, 11)),
        ),
        Parameter(
            'threads',
            kind=int,
            aliases=['num_threads'],
            description='''
            Controls how many threads the application will be using.
            ''',
            constraint=lambda x: x >= 0,
            default=0,
        ),
        Parameter(
            'bind_to_cpu_set',
            kind=int,
            description='''
            Controls whether threads will be bound to a core set, or each
            individual thread will be bound to a specific core within the core
            set.
            ''',
            constraint=lambda x: 0 <= x <= 1,
            default=1,
        ),
        Parameter(
            'llist_size',
            kind=int,
            description='''
            Size of the linked list available for each thread.
            ''',
            constraint=lambda x: x > 0,
            default=16777216,
        ),
        Parameter(
            'num_particles',
            kind=int,
            description='''
            Number of particles used in Phase 8.
            ''',
            constraint=lambda x: x > 0,
            default=1048576,
        ),
        Parameter(
            'num_palindromes',
            kind=int,
            description='''
            Number of palindromes used in Phase 9.
            ''',
            constraint=lambda x: x > 0,
            default=1024,
        ),
        Parameter(
            'num_randomloc',
            kind=int,
            description='''
            Number of random memory locations accessed in Phase 10.
            ''',
            constraint=lambda x: x > 0,
            default=2097152,
        ),
        Parameter(
            'timeout',
            kind=int,
            description="""
            Timeout for execution of the test.
            """,
            aliases=['run_timeout'],
            constraint=lambda x: x > 0,
            default=60 * 45,
        ),
    ]

    options = [
        ('-s', 'array_size'),
        ('-B', 'bind_to_cpu_set'),
        ('-b', 'block_size'),
        ('-l', 'llist_size'),
        ('-c', 'num_col'),
        ('-r', 'num_row'),
        ('-C', 'num_cpus'),
        ('-H', 'num_hwcntrs'),
        ('-i', 'loops'),
        ('-x', 'num_palindromes'),
        ('-p', 'num_particles'),
        ('-R', 'num_randomloc'),
        ('-T', 'threads'),
    ]

    def validate(self):
        if len(self.run_phases) != len(self.per_phase_cpu_ids):
            msg = "Number of phases doesn't match the number of CPU mappings"
            raise ConfigError(msg)

    def initialize(self, context):
        self._install_executable(context)
        self._build_command()

    def setup(self, context):
        self.output = None

    def run(self, context):
        self.output = self.target.execute(self.command,
                                          timeout=self.timeout)

    def update_output(self, context):
        if self.output is None:
            self.logger.warning('Did not collect output')
            return

        outfile = os.path.join(context.output_directory, 'meabo-output.txt')
        with open(outfile, 'wb') as wfh:
            if sys.version_info[0] == 3:
                wfh.write(self.output.encode('utf-8'))
            else:
                wfh.write(self.output)
        context.add_artifact('meabo-output', outfile, kind='raw')

        cur_phase = 0
        for line in self.output.split('\n'):
            line = line.strip()

            match = phase_start_regex.search(line)
            if match:
                cur_phase = match.group('phase')

            match = counter_value_regex.search(line)
            if match:
                if cur_phase == 0:
                    msg = 'Matched thread performance counters outside of phase!'
                    raise WorkloadError(msg)
                name = 'phase_{}_thread_{}_{}'.format(cur_phase,
                                                      match.group('thread'),
                                                      match.group('name'))
                context.add_metric(name, int(match.group('value')))

            match = duration_regex.search(line)
            if match:
                context.add_metric("phase_{}_duration".format(match.group('phase')),
                                   int(match.group('duration')), units="ns")

    def finalize(self, context):
        if self.uninstall:
            self._uninstall_executable()

    def _build_command(self):
        self.command = self.target_exe

        # We need to calculate the phase mask
        phase_mask = 0
        for phase in self.run_phases:
            phase_mask |= 1 << (phase - 1)

        self.command += ' -P {:d}'.format(phase_mask)

        # Set the CPU ids for each phase we are running
        for phase, cpu_id in zip(self.run_phases, self.per_phase_cpu_ids):
            self.command += ' -{0:1d} {1:d}'.format(phase, cpu_id)

        # We need to append extra arguments to the command based on the
        # parameters passed in from the agenda.
        for option, param_name in self.options:
            param_value = getattr(self, param_name, None)
            if param_value is not None:
                self.command += ' {} {}'.format(option, param_value)

    @once
    def _install_executable(self, context):
        resource = Executable(self, self.target.abi, 'meabo')
        host_exe = context.get_resource(resource)
        Meabo.target_exe = self.target.install(host_exe)

    @once
    def _uninstall_executable(self):
        self.target.uninstall(self.target_exe)
