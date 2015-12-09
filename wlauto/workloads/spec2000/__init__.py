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
import re
import string
import tarfile
from collections import defaultdict

from wlauto import Workload, Parameter, Alias
from wlauto.exceptions import ConfigError, WorkloadError
from wlauto.common.resources import ExtensionAsset
from wlauto.utils.misc import get_cpu_mask
from wlauto.utils.types import boolean, list_or_string


class Spec2000(Workload):

    name = 'spec2000'
    description = """
    SPEC2000 benchmarks measuring processor, memory and compiler.

    http://www.spec.org/cpu2000/

    From the web site:

    SPEC CPU2000 is the next-generation industry-standardized CPU-intensive benchmark suite. SPEC
    designed CPU2000 to provide a comparative measure of compute intensive performance across the
    widest practical range of hardware. The implementation resulted in source code benchmarks
    developed from real user applications. These benchmarks measure the performance of the
    processor, memory and compiler on the tested system.

    .. note:: At the moment, this workload relies on pre-built SPEC binaries (included in an
              asset bundle). These binaries *must* be built according to rules outlined here::

                  http://www.spec.org/cpu2000/docs/runrules.html#toc_2.0

              in order for the results to be valid SPEC2000 results.

    .. note:: This workload does not attempt to generate results in an admissible SPEC format. No
              metadata is provided (though some, but not all, of the required metdata is colleted
              by WA elsewhere). It is upto the user to post-process results to generated
              SPEC-admissible results file, if that is their intention.

    *base vs peak*

    SPEC2000 defines two build/test configuration: base and peak. Base is supposed to use basic
    configuration (e.g. default compiler flags) with no tuning, and peak is specifically optimized for
    a system. Since this workload uses externally-built binaries, there is no way for WA to be sure
    what configuration is used -- the user is expected to keep track of that. Be aware that
    base/peak also come with specfic requirements for the way workloads are run (e.g. how many instances
    on multi-core systems)::

        http://www.spec.org/cpu2000/docs/runrules.html#toc_3

    These are not enforced by WA, so it is again up to the user to ensure that correct workload
    parameters are specfied inthe agenda, if they intend to collect "official" SPEC results. (Those
    interested in collecting official SPEC results should also note that setting runtime parameters
    would violate SPEC runs rules that state that no configuration must be done to the platform
    after boot).

    *bundle structure*

    This workload expects the actual benchmark binaries to be provided in a tarball "bundle" that has
    a very specific structure. At the top level of the tarball, there should be two directories: "fp"
    and "int" -- for each of the SPEC2000 categories. Under those, there is a sub-directory per benchmark.
    Each benchmark sub-directory contains three sub-sub-directorie:

    - "cpus" contains a subdirector for each supported cpu (e.g. a15) with a single executable binary
      for that cpu, in addition to a "generic" subdirectory that has not been optimized for a specific
      cpu and should run on any ARM system.
    - "data" contains all additional files (input, configuration, etc) that  the benchmark executable
      relies on.
    - "scripts" contains one or more one-liner shell scripts that invoke the benchmark binary with
      appropriate command line parameters. The name of the script must be in the format
      <benchmark name>[.<variant name>].sh, i.e. name of benchmark, optionally followed by variant
      name, followed by ".sh" extension. If there is more than one script, then all of them must
      have  a variant; if there is only one script the it should not cotain a variant.

    A typical bundle may look like this::

        |- fp
        |  |-- ammp
        |  |   |-- cpus
        |  |   |   |-- generic
        |  |   |   |   |-- ammp
        |  |   |   |-- a15
        |  |   |   |   |-- ammp
        |  |   |   |-- a7
        |  |   |   |   |-- ammp
        |  |   |-- data
        |  |   |   |-- ammp.in
        |  |   |-- scripts
        |  |   |   |-- ammp.sh
        |  |-- applu
        .  .   .
        .  .   .
        .  .   .
        |- int
        .

    """

    # TODO: This is a bit of a hack. Need to re-think summary metric indication
    #      (also more than just summary/non-summary classification?)
    class _SPECSummaryMetrics(object):
        def __contains__(self, item):
            return item.endswith('_real')

    asset_file = 'spec2000-assets.tar.gz'

    aliases = [
        Alias('spec2k'),
    ]

    summary_metrics = _SPECSummaryMetrics()

    parameters = [
        Parameter('benchmarks', kind=list_or_string,
                  description='Specfiles the SPEC benchmarks to run.'),
        Parameter('mode', kind=str, allowed_values=['speed', 'rate'], default='speed',
                  description='SPEC benchmarks can report either speed to execute or throughput/rate. '
                              'In the latter case, several "threads" will be spawned.'),
        Parameter('number_of_threads', kind=int, default=None,
                  description='Specify the number of "threads" to be used in \'rate\' mode. (Note: '
                              'on big.LITTLE systems this is the number of threads, for *each cluster*). '),

        Parameter('force_extract_assets', kind=boolean, default=False,
                  description='if set to ``True``, will extract assets from the bundle, even if they are '
                              'already extracted. Note: this option implies ``force_push_assets``.'),
        Parameter('force_push_assets', kind=boolean, default=False,
                  description='If set to ``True``, assets will be pushed to device even if they\'re already '
                              'present.'),
        Parameter('timeout', kind=int, default=20 * 60,
                  description='Timemout, in seconds, for the execution of single spec test.'),
    ]

    speed_run_template = 'cd {datadir}; time ({launch_command})'
    rate_run_template = 'cd {datadir}; time ({loop}; wait)'
    loop_template = 'for i in $(busybox seq 1 {threads}); do {launch_command} 1>/dev/null 2>&1 & done'
    launch_template = 'busybox taskset {cpumask} {command} 1>/dev/null 2>&1'

    timing_regex = re.compile(r'(?P<minutes>\d+)m(?P<seconds>[\d.]+)s\s+(?P<category>\w+)')

    def init_resources(self, context):
        self._load_spec_benchmarks(context)

    def setup(self, context):
        cpus = self.device.core_names
        if not cpus:
            raise WorkloadError('Device has not specifed CPU cores configruation.')
        cpumap = defaultdict(list)
        for i, cpu in enumerate(cpus):
            cpumap[cpu.lower()].append(i)
        for benchspec in self.benchmarks:
            commandspecs = self._verify_and_deploy_benchmark(benchspec, cpumap)
            self._build_command(benchspec, commandspecs)

    def run(self, context):
        for name, command in self.commands:
            self.timings[name] = self.device.execute(command, timeout=self.timeout)

    def update_result(self, context):
        for benchmark, output in self.timings.iteritems():
            matches = self.timing_regex.finditer(output)
            found = False
            for match in matches:
                category = match.group('category')
                mins = float(match.group('minutes'))
                secs = float(match.group('seconds'))
                total = secs + 60 * mins
                context.result.add_metric('_'.join([benchmark, category]),
                                          total, 'seconds',
                                          lower_is_better=True)
                found = True
            if not found:
                self.logger.error('Could not get timings for {}'.format(benchmark))

    def validate(self):
        if self.force_extract_assets:
            self.force_push_assets = True
        if self.benchmarks is None:  # pylint: disable=access-member-before-definition
            self.benchmarks = ['all']
        for benchname in self.benchmarks:
            if benchname == 'all':
                self.benchmarks = self.loaded_benchmarks.keys()
                break
            if benchname not in self.loaded_benchmarks:
                raise ConfigError('Unknown SPEC benchmark: {}'.format(benchname))
        if self.mode == 'speed':
            if self.number_of_threads is not None:
                raise ConfigError('number_of_threads cannot be specified in speed mode.')
        else:
            raise ValueError('Unexpected SPEC2000 mode: {}'.format(self.mode))  # Should never get here
        self.commands = []
        self.timings = {}

    def _load_spec_benchmarks(self, context):
        self.loaded_benchmarks = {}
        self.categories = set()
        if self.force_extract_assets or len(os.listdir(self.dependencies_directory)) < 2:
            bundle = context.resolver.get(ExtensionAsset(self, self.asset_file))
            with tarfile.open(bundle, 'r:gz') as tf:
                tf.extractall(self.dependencies_directory)
        for entry in os.listdir(self.dependencies_directory):
            entrypath = os.path.join(self.dependencies_directory, entry)
            if os.path.isdir(entrypath):
                for bench in os.listdir(entrypath):
                    self.categories.add(entry)
                    benchpath = os.path.join(entrypath, bench)
                    self._load_benchmark(benchpath, entry)

    def _load_benchmark(self, path, category):
        datafiles = []
        cpus = []
        for df in os.listdir(os.path.join(path, 'data')):
            datafiles.append(os.path.join(path, 'data', df))
        for cpu in os.listdir(os.path.join(path, 'cpus')):
            cpus.append(cpu)
        commandsdir = os.path.join(path, 'commands')
        for command in os.listdir(commandsdir):
            bench = SpecBenchmark()
            bench.name = os.path.splitext(command)[0]
            bench.path = path
            bench.category = category
            bench.datafiles = datafiles
            bench.cpus = cpus
            with open(os.path.join(commandsdir, command)) as fh:
                bench.command_template = string.Template(fh.read().strip())
            self.loaded_benchmarks[bench.name] = bench

    def _verify_and_deploy_benchmark(self, benchspec, cpumap):  # pylint: disable=R0914
        """Verifies that the supplied benchmark spec is valid and deploys the required assets
        to the device (if necessary). Returns a list of command specs (one for each CPU cluster)
        that can then be used to construct the final command."""
        bench = self.loaded_benchmarks[benchspec]
        basename = benchspec.split('.')[0]
        datadir = self.device.path.join(self.device.working_directory, self.name, basename)
        if self.force_push_assets or not self.device.file_exists(datadir):
            self.device.execute('mkdir -p {}'.format(datadir))
            for datafile in bench.datafiles:
                self.device.push_file(datafile, self.device.path.join(datadir, os.path.basename(datafile)))

        if self.mode == 'speed':
            cpus = [self._get_fastest_cpu().lower()]
        else:
            cpus = cpumap.keys()

        cmdspecs = []
        for cpu in cpus:
            try:
                host_bin_file = bench.get_binary(cpu)
            except ValueError, e:
                try:
                    msg = e.message
                    msg += ' Attempting to use generic binary instead.'
                    self.logger.debug(msg)
                    host_bin_file = bench.get_binary('generic')
                    cpu = 'generic'
                except ValueError, e:
                    raise ConfigError(e.message)  # re-raising as user error
            binname = os.path.basename(host_bin_file)
            binary = self.device.install(host_bin_file, with_name='.'.join([binname, cpu]))
            commandspec = CommandSpec()
            commandspec.command = bench.command_template.substitute({'binary': binary})
            commandspec.datadir = datadir
            commandspec.cpumask = get_cpu_mask(cpumap[cpu])
            cmdspecs.append(commandspec)
        return cmdspecs

    def _build_command(self, name, commandspecs):
        if self.mode == 'speed':
            if len(commandspecs) != 1:
                raise AssertionError('Must be exactly one command spec specifed in speed mode.')
            spec = commandspecs[0]
            launch_command = self.launch_template.format(command=spec.command, cpumask=spec.cpumask)
            self.commands.append((name,
                                  self.speed_run_template.format(datadir=spec.datadir,
                                                                 launch_command=launch_command)))
        elif self.mode == 'rate':
            loops = []
            for spec in commandspecs:
                launch_command = self.launch_template.format(command=spec.command, cpumask=spec.cpumask)
                loops.append(self.loop_template.format(launch_command=launch_command, threads=spec.threads))
                self.commands.append((name,
                                      self.rate_run_template.format(datadir=spec.datadir,
                                                                    loop='; '.join(loops))))
        else:
            raise ValueError('Unexpected SPEC2000 mode: {}'.format(self.mode))  # Should never get here

    def _get_fastest_cpu(self):
        cpu_types = set(self.device.core_names)
        if len(cpu_types) == 1:
            return cpu_types.pop()
        fastest_cpu = None
        fastest_freq = 0
        for cpu_type in cpu_types:
            try:
                idx = self.device.get_core_online_cpu(cpu_type)
                freq = self.device.get_cpu_max_frequency(idx)
                if freq > fastest_freq:
                    fastest_freq = freq
                    fastest_cpu = cpu_type
            except ValueError:
                pass
        if not fastest_cpu:
            raise WorkloadError('No active CPUs found on device. Something is very wrong...')
        return fastest_cpu


class SpecBenchmark(object):

    def __init__(self):
        self.name = None
        self.path = None
        self.category = None
        self.command_template = None
        self.cpus = []
        self.datafiles = []

    def get_binary(self, cpu):
        if cpu not in self.cpus:
            raise ValueError('CPU {} is not supported by {}.'.format(cpu, self.name))
        binpath = os.path.join(self.path, 'cpus', cpu, self.name.split('.')[0])
        if not os.path.isfile(binpath):
            raise ValueError('CPU {} is not supported by {}.'.format(cpu, self.name))
        return binpath


class CommandSpec(object):

    def __init__(self):
        self.cpumask = None
        self.datadir = None
        self.command = None
        self.threads = None
