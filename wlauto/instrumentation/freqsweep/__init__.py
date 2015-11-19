#    Copyright 2015 ARM Limited
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
# pylint: disable=access-member-before-definition,attribute-defined-outside-init

import os
from collections import OrderedDict
from wlauto import Instrument, Parameter
from wlauto.exceptions import ConfigError, InstrumentError
from wlauto.utils.misc import merge_dicts
from wlauto.utils.types import caseless_string


class FreqSweep(Instrument):
    name = 'freq_sweep'
    description = """
    Sweeps workloads through all available frequencies on a device.

    When enabled this instrument will take all workloads specified in an agenda
    and run them at all available frequencies for all clusters.

    Recommendations:
        - Setting the runner to 'by_spec' increases the chance of successfully
          completing an agenda without encountering hotplug issues
        - If possible disable dynamic hotplug on the target device
    """

    parameters = [
        Parameter('sweeps', kind=list,
                  description="""
                  By default this instrument will sweep across all available
                  frequencies for all available clusters. If you wish to only
                  sweep across certain frequencies on particular clusters you
                  can do so by specifying this parameter.

                  Sweeps should be a lists of dictionaries that can contain:
                    - Cluster (mandatory): The name of the cluster this sweep will be
                                           performed on. E.g A7
                    - Frequencies: A list of frequencies (in KHz) to use. If this is
                                   not provided all frequencies available for this
                                   cluster will be used.
                                   E.g: [800000, 900000, 100000]
                    - label: Workload specs will be named '{spec id}_{label}_{frequency}'.
                             If a label is not provided it will be named 'sweep{sweep No.}'

                 Example sweep specification:

                     freq_sweep:
                         sweeps:
                             - cluster: A53
                               label: littles
                               frequencies: [800000, 900000, 100000]
                             - cluster: A57
                               label: bigs
                  """),
    ]

    def validate(self):
        if not self.device.core_names:
            raise ConfigError('The Device does not appear to have core_names configured.')

    def initialize(self, context):  # pylint: disable=r0912
        if not self.device.is_rooted:
            raise InstrumentError('The device must be rooted to sweep frequencies')

        if 'userspace' not in self.device.list_available_cluster_governors(0):
            raise InstrumentError("'userspace' cpufreq governor must be enabled")

        # Create sweeps for each core type using num_cpus cores
        if not self.sweeps:
            self.sweeps = []
            for core in set(self.device.core_names):
                sweep_spec = {}
                sweep_spec['cluster'] = core
                sweep_spec['label'] = core
                self.sweeps.append(sweep_spec)

        new_specs = []
        old_specs = []
        for job in context.runner.job_queue:
            if job.spec not in old_specs:
                old_specs.append(job.spec)

        # Validate sweeps, add missing sections and create workload specs
        for i, sweep_spec in enumerate(self.sweeps):
            if 'cluster' not in sweep_spec:
                raise ConfigError('cluster must be define for all sweeps')
            # Check if cluster exists on device
            if caseless_string(sweep_spec['cluster']) not in self.device.core_names:
                raise ConfigError('Only {} cores are present on this device, you specified {}'
                                  .format(", ".join(set(self.device.core_names)), sweep_spec['cluster']))

            # Default to all available frequencies
            if 'frequencies' not in sweep_spec:
                self.device.enable_cpu(self.device.core_names.index(sweep_spec['cluster']))
                sweep_spec['frequencies'] = self.device.list_available_core_frequencies(sweep_spec['cluster'])

            # Check that given frequencies are valid of the core cluster
            else:
                self.device.enable_cpu(self.device.core_names.index(sweep_spec['cluster']))
                available_freqs = self.device.list_available_core_frequencies(sweep_spec['cluster'])
                for freq in sweep_spec['frequencies']:
                    if freq not in available_freqs:
                        raise ConfigError('Frequency {} is not supported by {} cores'.format(freq, sweep_spec['cluster']))

            # Add default labels
            if 'label' not in sweep_spec:
                sweep_spec['label'] = "sweep{}".format(i + 1)

            new_specs.extend(self.get_sweep_workload_specs(old_specs, sweep_spec, context))

        # Update config to refect jobs that will actually run.
        context.config.workload_specs = new_specs
        config_file = os.path.join(context.host_working_directory, 'run_config.json')
        with open(config_file, 'wb') as wfh:
            context.config.serialize(wfh)
        context.runner.init_queue(new_specs)

    def get_sweep_workload_specs(self, old_specs, sweep_spec, context):
        new_specs = []
        for old_spec in old_specs:
            for freq in sweep_spec['frequencies']:
                spec = old_spec.copy()
                if 'runtime_params' in sweep_spec:
                    spec.runtime_parameters = merge_dicts(spec.runtime_parameters,
                                                          sweep_spec['runtime_params'],
                                                          dict_type=OrderedDict)
                if 'workload_params' in sweep_spec:
                    spec.workload_parameters = merge_dicts(spec.workload_parameters,
                                                           sweep_spec['workload_params'],
                                                           dict_type=OrderedDict)
                spec.runtime_parameters['{}_governor'.format(sweep_spec['cluster'])] = "userspace"
                spec.runtime_parameters['{}_frequency'.format(sweep_spec['cluster'])] = freq
                spec.id = '{}_{}_{}'.format(spec.id, sweep_spec['label'], freq)
                spec.classifiers['core'] = sweep_spec['cluster']
                spec.classifiers['freq'] = freq
                spec.load(self.device, context.config.ext_loader)
                spec.workload.init_resources(context)
                spec.workload.validate()
                new_specs.append(spec)
        return new_specs
