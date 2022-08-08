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

import random
from itertools import groupby, chain

from future.moves.itertools import zip_longest

from devlib.utils.types import identifier

from wa.framework.configuration.core import (MetaConfiguration, RunConfiguration,
                                             JobGenerator, settings)
from wa.framework.configuration.parsers import ConfigParser
from wa.framework.configuration.plugin_cache import PluginCache
from wa.framework.exception import NotFoundError, ConfigError
from wa.framework.job import Job
from wa.utils import log
from wa.utils.serializer import Podable


class CombinedConfig(Podable):

    _pod_serialization_version = 1

    @staticmethod
    def from_pod(pod):
        instance = super(CombinedConfig, CombinedConfig).from_pod(pod)
        instance.settings = MetaConfiguration.from_pod(pod.get('settings', {}))
        instance.run_config = RunConfiguration.from_pod(pod.get('run_config', {}))
        return instance

    def __init__(self, settings=None, run_config=None):  # pylint: disable=redefined-outer-name
        super(CombinedConfig, self).__init__()
        self.settings = settings
        self.run_config = run_config

    def to_pod(self):
        pod = super(CombinedConfig, self).to_pod()
        pod['settings'] = self.settings.to_pod()
        pod['run_config'] = self.run_config.to_pod()
        return pod

    @staticmethod
    def _pod_upgrade_v1(pod):
        pod['_pod_version'] = pod.get('_pod_version', 1)
        return pod


class ConfigManager(object):
    """
    Represents run-time state of WA. Mostly used as a container for loaded
    configuration and discovered plugins.

    This exists outside of any command or run and is associated with the running
    instance of wA itself.
    """

    @property
    def enabled_instruments(self):
        return self.jobs_config.enabled_instruments

    @property
    def enabled_processors(self):
        return self.jobs_config.enabled_processors

    @property
    def job_specs(self):
        if not self._jobs_generated:
            msg = 'Attempting to access job specs before '\
                  'jobs have been generated'
            raise RuntimeError(msg)
        return [j.spec for j in self._jobs]

    @property
    def jobs(self):
        if not self._jobs_generated:
            msg = 'Attempting to access jobs before '\
                  'they have been generated'
            raise RuntimeError(msg)
        return self._jobs

    def __init__(self, settings=settings):  # pylint: disable=redefined-outer-name
        self.settings = settings
        self.run_config = RunConfiguration()
        self.plugin_cache = PluginCache()
        self.jobs_config = JobGenerator(self.plugin_cache)
        self.loaded_config_sources = []
        self._config_parser = ConfigParser()
        self._jobs = []
        self._jobs_generated = False
        self.agenda = None

    def load_config_file(self, filepath):
        includes = self._config_parser.load_from_path(self, filepath)
        self.loaded_config_sources.append(filepath)
        self.loaded_config_sources.extend(includes)

    def load_config(self, values, source):
        self._config_parser.load(self, values, source)
        self.loaded_config_sources.append(source)

    def get_plugin(self, name=None, kind=None, *args, **kwargs):
        return self.plugin_cache.get_plugin(identifier(name), kind, *args, **kwargs)

    def get_instruments(self, target):
        instruments = []
        for name in self.enabled_instruments:
            try:
                instruments.append(self.get_plugin(name, kind='instrument',
                                                   target=target))
            except NotFoundError:
                msg = 'Instrument "{}" not found'
                raise NotFoundError(msg.format(name))
        return instruments

    def get_processors(self):
        processors = []
        for name in self.enabled_processors:
            try:
                proc = self.plugin_cache.get_plugin(name, kind='output_processor')
            except NotFoundError:
                msg = 'Output Processor "{}" not found'
                raise NotFoundError(msg.format(name))
            processors.append(proc)
        return processors

    def get_config(self):
        return CombinedConfig(self.settings, self.run_config)

    def finalize(self):
        if not self.agenda:
            msg = 'Attempting to finalize config before agenda has been set'
            raise RuntimeError(msg)
        self.run_config.merge_device_config(self.plugin_cache)
        return self.get_config()

    def generate_jobs(self, context):
        job_specs = self.jobs_config.generate_job_specs(context.tm)
        if not job_specs:
            msg = 'No jobs available for running.'
            raise ConfigError(msg)
        exec_order = self.run_config.execution_order
        log.indent()
        for spec, i in permute_iterations(job_specs, exec_order):
            job = Job(spec, i, context)
            job.load(context.tm.target)
            self._jobs.append(job)
            context.run_state.add_job(job)
        log.dedent()
        self._jobs_generated = True


def permute_by_workload(specs):
    """
    This is that "classic" implementation that executes all iterations of a
    workload spec before proceeding onto the next spec.

    """
    for spec in specs:
        for i in range(1, spec.iterations + 1):
            yield (spec, i)


def permute_by_iteration(specs):
    """
    Runs the first iteration for all benchmarks first, before proceeding to the
    next iteration, i.e. A1, B1, C1, A2, B2, C2...  instead of  A1, A1, B1, B2,
    C1, C2...

    If multiple sections where specified in the agenda, this will run all
    sections for the first global spec first, followed by all sections for the
    second spec, etc.

    e.g. given sections X and Y, and global specs A and B, with 2 iterations,
    this will run

    X.A1, Y.A1, X.B1, Y.B1, X.A2, Y.A2, X.B2, Y.B2

    """
    groups = [list(g) for _, g in groupby(specs, lambda s: s.workload_id)]

    all_tuples = []
    for spec in chain(*groups):
        all_tuples.append([(spec, i + 1)
                           for i in range(spec.iterations)])
    for t in chain(*list(map(list, zip_longest(*all_tuples)))):
        if t is not None:
            yield t


def permute_by_section(specs):
    """
    Runs the first iteration for all benchmarks first, before proceeding to the
    next iteration, i.e. A1, B1, C1, A2, B2, C2...  instead of  A1, A1, B1, B2,
    C1, C2...

    If multiple sections where specified in the agenda, this will run all specs
    for the first section followed by all specs for the seciod section, etc.

    e.g. given sections X and Y, and global specs A and B, with 2 iterations,
    this will run

    X.A1, X.B1, Y.A1, Y.B1, X.A2, X.B2, Y.A2, Y.B2

    """
    groups = [list(g) for _, g in groupby(specs, lambda s: s.section_id)]

    all_tuples = []
    for spec in chain(*groups):
        all_tuples.append([(spec, i + 1)
                           for i in range(spec.iterations)])
    for t in chain(*list(map(list, zip_longest(*all_tuples)))):
        if t is not None:
            yield t


def permute_randomly(specs):
    """
    This will generate a random permutation of specs/iteration tuples.

    """
    result = []
    for spec in specs:
        for i in range(1, spec.iterations + 1):
            result.append((spec, i))
    random.shuffle(result)
    for t in result:
        yield t


permute_map = {
    'by_iteration': permute_by_iteration,
    'by_workload': permute_by_workload,
    'by_section': permute_by_section,
    'random': permute_randomly,
}


def permute_iterations(specs, exec_order):
    if exec_order not in permute_map:
        msg = 'Unknown execution order "{}"; must be in: {}'
        raise ValueError(msg.format(exec_order, list(permute_map.keys())))
    return permute_map[exec_order](specs)
