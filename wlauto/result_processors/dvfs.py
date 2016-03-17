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

import os
import csv
import re

from wlauto import ResultProcessor, settings, instrumentation
from wlauto.exceptions import ConfigError, ResultProcessorError


class DVFS(ResultProcessor):
    name = 'dvfs'
    description = """
    Reports DVFS state residency data form ftrace power events.

    This generates a ``dvfs.csv`` in the top-level results directory that,
    for each workload iteration, reports the percentage of time each CPU core
    spent in each of the DVFS frequency states (P-states), as well as percentage
    of the time spent in idle, during the execution of the workload.

    .. note:: ``trace-cmd`` instrument *MUST* be enabled in the instrumentation,
              and at least ``'power*'`` events must be enabled.


    """

    def __init__(self, **kwargs):
        super(DVFS, self).__init__(**kwargs)
        self.device = None
        self.infile = None
        self.outfile = None
        self.current_cluster = None
        self.currentstates_of_clusters = []
        self.current_frequency_of_clusters = []
        self.timestamp = []
        self.state_time_map = {}  # hold state at timestamp
        self.cpuid_time_map = {}  # hold cpuid at timestamp
        self.cpu_freq_time_spent = {}
        self.cpuids_of_clusters = []
        self.power_state = [0, 1, 2, 3]
        self.UNKNOWNSTATE = 4294967295
        self.multiply_factor = None
        self.corename_of_clusters = []
        self.numberofcores_in_cluster = []
        self.minimum_frequency_cluster = []
        self.idlestate_description = {}

    def validate(self):
        if not instrumentation.instrument_is_installed('trace-cmd'):
            raise ConfigError('"dvfs" works only if "trace_cmd" in enabled in instrumentation')

    def initialize(self, context):  # pylint: disable=R0912
        self.device = context.device
        if not self.device.has('cpuidle'):
            raise ConfigError('Device does not appear to have cpuidle capability; is the right module installed?')
        if not self.device.core_names:
            message = 'Device does not specify its core types (core_names/core_clusters not set in device_config).'
            raise ResultProcessorError(message)
        number_of_clusters = max(self.device.core_clusters) + 1
        # In IKS devices, actual number of cores is double
        # from what we get from device.number_of_cores
        if self.device.scheduler == 'iks':
            self.multiply_factor = 2
        elif self.device.scheduler == 'unknown':
            # Device doesn't specify its scheduler type. It could be IKS, in
            # which case reporeted values would be wrong, so error out.
            message = ('The Device doesn not specify it\'s scheduler type. If you are '
                       'using a generic device interface, please make sure to set the '
                       '"scheduler" parameter in the device config.')
            raise ResultProcessorError(message)
        else:
            self.multiply_factor = 1
        # separate out the cores in each cluster
        # It is list of list of cores in cluster
        listof_cores_clusters = []
        for cluster in range(number_of_clusters):
            listof_cores_clusters.append([core for core in self.device.core_clusters if core == cluster])
        # Extract minimum frequency of each cluster and
        # the idle power state with its descriptive name
        #
        total_cores = 0
        current_cores = 0
        for cluster, cores_list in enumerate(listof_cores_clusters):
            self.corename_of_clusters.append(self.device.core_names[total_cores])
            if self.device.scheduler != 'iks':
                self.idlestate_description.update({s.id: s.desc for s in self.device.get_cpuidle_states(total_cores)})
            else:
                self.idlestate_description.update({s.id: s.desc for s in self.device.get_cpuidle_states()})
            total_cores += len(cores_list)
            self.numberofcores_in_cluster.append(len(cores_list))
            for i in range(current_cores, total_cores):
                if i in self.device.online_cpus:
                    self.minimum_frequency_cluster.append(int(self.device.get_cpu_min_frequency("cpu{}".format(i))))
                    break
            current_cores = total_cores
        length_frequency_cluster = len(self.minimum_frequency_cluster)
        if length_frequency_cluster != number_of_clusters:
            diff = number_of_clusters - length_frequency_cluster
            offline_value = -1
            for i in range(diff):
                if self.device.scheduler != 'iks':
                    self.minimum_frequency_cluster.append(offline_value)
                else:
                    self.minimum_frequency_cluster.append(self.device.iks_switch_frequency)

    def process_iteration_result(self, result, context):
        """
        Parse the trace.txt for each iteration,  calculate DVFS residency state/frequencies
        and dump the result in csv and flush the data for next iteration.
        """
        self.infile = os.path.join(context.output_directory, 'trace.txt')
        if os.path.isfile(self.infile):
            self.logger.debug('Running result_processor "dvfs"')
            self.outfile = os.path.join(settings.output_directory, 'dvfs.csv')
            self.flush_parse_initialize()
            self.calculate()
            self.percentage()
            self.generate_csv(context)
            self.logger.debug('Completed result_processor "dvfs"')
        else:
            self.logger.debug('trace.txt not found.')

    def flush_parse_initialize(self):
        """
        Store state, cpu_id for each timestamp from trace.txt and flush all the values for
        next iterations.
        """
        self.current_cluster = 0
        self.current_frequency_of_clusters = []
        self.timestamp = []
        self.currentstates_of_clusters = []
        self.state_time_map = {}
        self.cpuid_time_map = {}
        self.cpu_freq_time_spent = {}
        self.cpuids_of_clusters = []
        self.parse()  # Parse trace.txt generated from trace-cmd instrumentation
        # Initialize the states of each core of clusters and frequency of
        # each clusters with its minimum freq
        # cpu_id is assigned for each of clusters.
        # For IKS devices cpuid remains same in other clusters
        # and for other it will increment by 1
        count = 0
        for cluster, cores_number in enumerate(self.numberofcores_in_cluster):
            self.currentstates_of_clusters.append([-1 for dummy in range(cores_number)])
            self.current_frequency_of_clusters.append(self.minimum_frequency_cluster[cluster])
            if self.device.scheduler == 'iks':
                self.cpuids_of_clusters.append([j for j in range(cores_number)])
            else:
                self.cpuids_of_clusters.append(range(count, count + cores_number))
                count += cores_number

        # Initialize the time spent in each state/frequency for each core.
        for i in range(self.device.number_of_cores * self.multiply_factor):
            self.cpu_freq_time_spent["cpu{}".format(i)] = {}
            for j in self.unique_freq():
                self.cpu_freq_time_spent["cpu{}".format(i)][j] = 0
            # To determine offline -1 state is added
            offline_value = -1
            self.cpu_freq_time_spent["cpu{}".format(i)][offline_value] = 0
            if 0 not in self.unique_freq():
                self.cpu_freq_time_spent["cpu{}".format(i)][0] = 0

    def update_cluster_freq(self, state, cpu_id):
        """ Update the cluster frequency and current cluster"""
        # For IKS devices cluster changes only possible when
        # freq changes, for other it is determine by cpu_id.
        if self.device.scheduler != 'iks':
            self.current_cluster = self.get_cluster(cpu_id, state)
        if self.get_state_name(state) == "freqstate":
            self.current_cluster = self.get_cluster(cpu_id, state)
            self.current_frequency_of_clusters[self.current_cluster] = state

    def get_cluster(self, cpu_id, state):
        # For IKS if current state is greater than switch
        # freq then it is in cluster2 else cluster1
        # For other, Look the current cpu_id and check this id
        # belong to which cluster.
        if self.device.scheduler == 'iks':
            return 1 if state >= self.device.iks_switch_frequency else 0
        else:
            for cluster, cpuids_list in enumerate(self.cpuids_of_clusters):
                if cpu_id in cpuids_list:
                    return cluster

    def get_cluster_freq(self):
        return self.current_frequency_of_clusters[self.current_cluster]

    def update_state(self, state, cpu_id):  # pylint: disable=R0912
        """
        Update state of each cores in every cluster.
        This is done for each timestamp.
        """
        POWERDOWN = 2
        offline_value = -1
        # if state is in unknowstate, then change state of current cpu_id
        # with cluster freq of current cluster.
        # if state is in powerstate then change state with that power state.
        if self.get_state_name(state) in ["unknownstate", "powerstate"]:
            for i in range(len(self.cpuids_of_clusters[self.current_cluster])):
                if cpu_id == self.cpuids_of_clusters[self.current_cluster][i]:
                    if self.get_state_name(state) == "unknownstate":
                        self.currentstates_of_clusters[self.current_cluster][i] = self.current_frequency_of_clusters[self.current_cluster]
                    elif self.get_state_name(state) == "powerstate":
                        self.currentstates_of_clusters[self.current_cluster][i] = state
        # If state is in freqstate then update the state with current state.
        # For IKS, if all cores is in power down and current state is freqstate
        # then update the all the cores in current cluster to current state
        # and other state cluster changed to Power down.
        if self.get_state_name(state) == "freqstate":
            for i, j in enumerate(self.currentstates_of_clusters[self.current_cluster]):
                if j != offline_value:
                    self.currentstates_of_clusters[self.current_cluster][i] = state
                if cpu_id == self.cpuids_of_clusters[self.current_cluster][i]:
                    self.currentstates_of_clusters[self.current_cluster][i] = state
            if self.device.scheduler == 'iks':
                check = False  # All core in cluster is power down
                for i in range(len(self.currentstates_of_clusters[self.current_cluster])):
                    if self.currentstates_of_clusters[self.current_cluster][i] != POWERDOWN:
                        check = True
                        break
                if not check:
                    for i in range(len(self.currentstates_of_clusters[self.current_cluster])):
                        self.currentstates_of_clusters[self.current_cluster][i] = self.current_frequency_of_clusters[self.current_cluster]
                for cluster, state_list in enumerate(self.currentstates_of_clusters):
                    if cluster != self.current_cluster:
                        for j in range(len(state_list)):
                            self.currentstates_of_clusters[i][j] = POWERDOWN

    def unique_freq(self):
        """ Determine the unique Frequency and state"""
        unique_freq = []
        for i in self.timestamp:
            if self.state_time_map[i] not in unique_freq and self.state_time_map[i] != self.UNKNOWNSTATE:
                unique_freq.append(self.state_time_map[i])
        for i in self.minimum_frequency_cluster:
            if i not in unique_freq:
                unique_freq.append(i)
        return unique_freq

    def parse(self):
        """
        Parse the trace.txt ::

            store timestamp, state, cpu_id
            ---------------------------------------------------------------------------------
                                |timestamp|                       |state|        |cpu_id|
            <idle>-0     [001]   294.554380: cpu_idle:             state=4294967295 cpu_id=1
            <idle>-0     [001]   294.554454: power_start:          type=1 state=0 cpu_id=1
            <idle>-0     [001]   294.554458: cpu_idle:             state=0 cpu_id=1
            <idle>-0     [001]   294.554464: power_end:            cpu_id=1
            <idle>-0     [001]   294.554471: cpu_idle:             state=4294967295 cpu_id=1
            <idle>-0     [001]   294.554590: power_start:          type=1 state=0 cpu_id=1
            <idle>-0     [001]   294.554593: cpu_idle:             state=0 cpu_id=1
            <idle>-0     [001]   294.554636: power_end:            cpu_id=1
            <idle>-0     [001]   294.554639: cpu_idle:             state=4294967295 cpu_id=1
            <idle>-0     [001]   294.554669: power_start:          type=1 state=0 cpu_id=1


        """
        pattern = re.compile(r'\s+(?P<time>\S+)\S+\s*(?P<desc>(cpu_idle:|cpu_frequency:))\s*state=(?P<state>\d+)\s*cpu_id=(?P<cpu_id>\d+)')
        start_trace = False
        stop_trace = False
        with open(self.infile, 'r') as f:
            for line in f:
                #Start collecting data from label "TRACE_MARKER_START" and
                #stop with label "TRACE_MARKER_STOP"
                if line.find("TRACE_MARKER_START") != -1:
                    start_trace = True
                if line.find("TRACE_MARKER_STOP") != -1:
                    stop_trace = True
                if start_trace and not stop_trace:
                    match = pattern.search(line)
                    if match:
                        self.timestamp.append(float(match.group('time')))
                        self.state_time_map[float(match.group('time'))] = int(match.group('state'))
                        self.cpuid_time_map[float(match.group('time'))] = int(match.group('cpu_id'))

    def get_state_name(self, state):
        if state in self.power_state:
            return "powerstate"
        elif state == self.UNKNOWNSTATE:
            return "unknownstate"
        else:
            return "freqstate"

    def populate(self, time1, time2):
        diff = time2 - time1
        for cluster, states_list in enumerate(self.currentstates_of_clusters):
            for k, j in enumerate(states_list):
                if self.device.scheduler == 'iks' and cluster == 1:
                    self.cpu_freq_time_spent["cpu{}".format(self.cpuids_of_clusters[cluster][k] + len(self.currentstates_of_clusters[0]))][j] += diff
                else:
                    self.cpu_freq_time_spent["cpu{}".format(self.cpuids_of_clusters[cluster][k])][j] += diff

    def calculate(self):
        for i in range(len(self.timestamp) - 1):
            self.update_cluster_freq(self.state_time_map[self.timestamp[i]], self.cpuid_time_map[self.timestamp[i]])
            self.update_state(self.state_time_map[self.timestamp[i]], self.cpuid_time_map[self.timestamp[i]])
            self.populate(self.timestamp[i], self.timestamp[i + 1])

    def percentage(self):
        """Normalize the result with total execution time."""
        temp = self.cpu_freq_time_spent.copy()
        for i in self.cpu_freq_time_spent:
            total = 0
            for j in self.cpu_freq_time_spent[i]:
                total += self.cpu_freq_time_spent[i][j]
            for j in self.cpu_freq_time_spent[i]:
                if total != 0:
                    temp[i][j] = self.cpu_freq_time_spent[i][j] * 100 / total
                else:
                    temp[i][j] = 0
        return temp

    def generate_csv(self, context):  # pylint: disable=R0912,R0914
        """ generate the '''dvfs.csv''' with the state, frequency and cores """
        temp = self.percentage()
        total_state = self.unique_freq()
        offline_value = -1
        ghz_conversion = 1000000
        mhz_conversion = 1000
        with open(self.outfile, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            reader = csv.reader(f)
            # Create the header in the format below
            # workload name, iteration, state, A7 CPU0,A7 CPU1,A7 CPU2,A7 CPU3,A15 CPU4,A15 CPU5
            if sum(1 for row in reader) == 0:
                header_row = ['workload', 'iteration', 'state']
                count = 0
                for cluster, states_list in enumerate(self.currentstates_of_clusters):
                    for dummy_index in range(len(states_list)):
                        header_row.append("{} CPU{}".format(self.corename_of_clusters[cluster], count))
                        count += 1
                writer.writerow(header_row)
            if offline_value in total_state:
                total_state.remove(offline_value)  # remove the offline state
            for i in sorted(total_state):
                temprow = []
                temprow.extend([context.result.spec.label, context.result.iteration])
                if "state{}".format(i) in self.idlestate_description:
                    temprow.append(self.idlestate_description["state{}".format(i)])
                else:
                    state_value = float(i)
                    if state_value / ghz_conversion >= 1:
                        temprow.append("{} Ghz".format(state_value / ghz_conversion))
                    else:
                        temprow.append("{} Mhz".format(state_value / mhz_conversion))
                for j in range(self.device.number_of_cores * self.multiply_factor):
                    temprow.append("{0:.3f}".format(temp["cpu{}".format(j)][i]))
                writer.writerow(temprow)
            check_off = True  # Checking whether core is OFFLINE
            for i in range(self.device.number_of_cores * self.multiply_factor):
                temp_val = "{0:.3f}".format(temp["cpu{}".format(i)][offline_value])
                if float(temp_val) > 1:
                    check_off = False
                    break
            if check_off is False:
                temprow = []
                temprow.extend([context.result.spec.label, context.result.iteration])
                temprow.append("OFFLINE")
                for i in range(self.device.number_of_cores * self.multiply_factor):
                    temprow.append("{0:.3f}".format(temp["cpu{}".format(i)][offline_value]))
                writer.writerow(temprow)
