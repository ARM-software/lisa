# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Idle Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl

from analysis_module import AnalysisModule
from trappy.utils import listify

# Configure logging
import logging


class IdleAnalysis(AnalysisModule):
    """
    Support for plotting Idle Analysis data

    :param trace: input Trace object
    :type trace: :mod:`libs.utils.Trace`
    """

    def __init__(self, trace):
        super(IdleAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    def _dfg_cpu_idle_state_residency(self, cpu):
        """
        Compute time spent by a given CPU in each idle state.

        :param entity: CPU ID
        :type entity: int

        :returns: :mod:`pandas.DataFrame` - idle state residency dataframe
        """
        if not self._trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'idle state residency computation not possible!')
            return None

        idle_df = self._dfg_trace_event('cpu_idle')
        cpu_idle = idle_df[idle_df.cpu_id == cpu]

        cpu_is_idle = self._trace.getCPUActiveSignal(cpu) ^ 1

        # In order to compute the time spent in each idle state we
        # multiply 2 square waves:
        # - cpu_idle
        # - idle_state, square wave of the form:
        #     idle_state[t] == 1 if at time t CPU is in idle state i
        #     idle_state[t] == 0 otherwise
        available_idles = sorted(idle_df.state.unique())
        # Remove non-idle state from availables
        available_idles.pop()
        cpu_idle = cpu_idle.join(cpu_is_idle.to_frame(name='is_idle'),
                                 how='outer')
        cpu_idle.fillna(method='ffill', inplace=True)
        idle_time = []
        for i in available_idles:
            idle_state = cpu_idle.state.apply(
                lambda x: 1 if x == i else 0
            )
            idle_t = cpu_idle.is_idle * idle_state
            # Compute total time by integrating the square wave
            idle_time.append(self._trace.integrate_square_wave(idle_t))

        idle_time_df = pd.DataFrame({'time' : idle_time}, index=available_idles)
        idle_time_df.index.name = 'idle_state'
        return idle_time_df

    def _dfg_cluster_idle_state_residency(self, cluster):
        """
        Compute time spent by a given cluster in each idle state.

        :param cluster: cluster name or list of CPU IDs
        :type cluster: str or list(int)

        :returns: :mod:`pandas.DataFrame` - idle state residency dataframe
        """
        if not self._trace.hasEvents('cpu_idle'):
            logging.warn('Events [cpu_idle] not found, '\
                         'idle state residency computation not possible!')
            return None

        _cluster = cluster
        if isinstance(cluster, str) or isinstance(cluster, unicode):
            try:
                _cluster = self._platform['clusters'][cluster.lower()]
            except KeyError:
                logging.warn('%s cluster not found!', cluster)
                return None

        idle_df = self._dfg_trace_event('cpu_idle')
        # Each core in a cluster can be in a different idle state, but the
        # cluster lies in the idle state with lowest ID, that is the shallowest
        # idle state among the idle states of its CPUs
        cl_idle = idle_df[idle_df.cpu_id == _cluster[0]].state.to_frame(
            name=_cluster[0])
        for cpu in _cluster[1:]:
            cl_idle = cl_idle.join(
                idle_df[idle_df.cpu_id == cpu].state.to_frame(name=cpu),
                how='outer'
            )
        cl_idle.fillna(method='ffill', inplace=True)
        cl_idle = pd.DataFrame(cl_idle.min(axis=1), columns=['state'])

        # Build a square wave of the form:
        #     cl_is_idle[t] == 1 if all CPUs in the cluster are reported
        #                      to be idle by cpufreq at time t
        #     cl_is_idle[t] == 0 otherwise
        cl_is_idle = self._trace.getClusterActiveSignal(_cluster) ^ 1

        # In order to compute the time spent in each idle statefrequency we
        # multiply 2 square waves:
        # - cluster_is_idle
        # - idle_state, square wave of the form:
        #     idle_state[t] == 1 if at time t cluster is in idle state i
        #     idle_state[t] == 0 otherwise
        available_idles = sorted(idle_df.state.unique())
        # Remove non-idle state from availables
        available_idles.pop()
        cl_idle = cl_idle.join(cl_is_idle.to_frame(name='is_idle'),
                               how='outer')
        cl_idle.fillna(method='ffill', inplace=True)
        idle_time = []
        for i in available_idles:
            idle_state = cl_idle.state.apply(
                lambda x: 1 if x == i else 0
            )
            idle_t = cl_idle.is_idle * idle_state
            # Compute total time by integrating the square wave
            idle_time.append(self._trace.integrate_square_wave(idle_t))

        idle_time_df = pd.DataFrame({'time' : idle_time}, index=available_idles)
        idle_time_df.index.name = 'idle_state'
        return idle_time_df


###############################################################################
# Plotting Methods
###############################################################################


###############################################################################
# Utility Methods
###############################################################################


# vim :set tabstop=4 shiftwidth=4 expandtab
