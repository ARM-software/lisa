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

from lisa.analysis.base import AnalysisBase, ResidencyTime, ResidencyData
from trappy.utils import listify


class IdleAnalysis(AnalysisBase):
    """
    Support for plotting Idle Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'idle'

    def __init__(self, trace):
        super(IdleAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    def df_cpu_idle_state_residency(self, cpu):
        """
        Compute time spent by a given CPU in each idle state.

        :param entity: CPU ID
        :type entity: int

        :returns: :mod:`pandas.DataFrame` - idle state residency dataframe
        """
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, '
                              'idle state residency computation not possible!')
            return None

        idle_df = self.df_events('cpu_idle')
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
        available_idles = available_idles[1:]
        cpu_idle = cpu_idle.join(cpu_is_idle.to_frame(name='is_idle'),
                                 how='outer')
        cpu_idle.fillna(method='ffill', inplace=True)

        # Extend the last cpu_idle event to the end of the time window under
        # consideration
        final_entry = pd.DataFrame([cpu_idle.iloc[-1]], index=[self._trace.x_max])
        cpu_idle = cpu_idle.append(final_entry)

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

    def df_cluster_idle_state_residency(self, cluster):
        """
        Compute time spent by a given cluster in each idle state.

        :param cluster: cluster name or list of CPU IDs
        :type cluster: str or list(int)

        :returns: :mod:`pandas.DataFrame` - idle state residency dataframe
        """
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, '
                              'idle state residency computation not possible!')
            return None

        _cluster = cluster
        if isinstance(cluster, str) or isinstance(cluster, str):
            try:
                _cluster = self._platform['clusters'][cluster.lower()]
            except KeyError:
                self._log.warning('%s cluster not found!', cluster)
                return None

        idle_df = self.df_events('cpu_idle')
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
        available_idles = available_idles[1:]
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

    def plot_cpu_idle_state_residency(self, cpus=None, pct=False):
        """
        Plot per-CPU idle state residency. big CPUs are plotted first and then
        LITTLEs.

        Requires cpu_idle trace events.

        :param cpus: list of CPU IDs. By default plot all CPUs
        :type cpus: list(int) or int

        :param pct: plot residencies in percentage
        :type pct: bool
        """
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, '
                              'plot DISABLED!')
            return

        if cpus is None:
            # Generate plots only for available CPUs
            cpuidle_data = self.df_events('cpu_idle')
            _cpus = list(range(cpuidle_data.cpu_id.max() + 1))
        else:
            _cpus = listify(cpus)

        # Split between big and LITTLE CPUs ordered from higher to lower ID
        _cpus.reverse()
        big_cpus = [c for c in _cpus if c in self._big_cpus]
        little_cpus = [c for c in _cpus if c in self._little_cpus]
        _cpus = big_cpus + little_cpus

        residencies = []
        xmax = 0.0
        for cpu in _cpus:
            r = self.df_cpu_idle_state_residency(cpu)
            residencies.append(ResidencyData('CPU{}'.format(cpu), r))

            max_time = r.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plot_idle_state_residency(residencies, 'cpu', xmax, pct=pct)

    def plot_cluster_idle_state_residency(self, clusters=None, pct=False):
        """
        Plot per-cluster idle state residency in a given cluster, i.e. the
        amount of time cluster `cluster` spent in idle state `i`. By default,
        both 'big' and 'LITTLE' clusters data are plotted.

        Requires cpu_idle following trace events.
        :param clusters: name of the clusters to be plotted (all of them by
            default)
        :type clusters: str ot list(str)
        """
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, plot DISABLED!')
            return
        if 'clusters' not in self._platform:
            self._log.warning('No platform cluster info. Plot DISABLED!')
            return

        # Sanitize clusters
        if clusters is None:
            _clusters = list(self._platform['clusters'].keys())
        else:
            _clusters = listify(clusters)

        # Precompute residencies for each cluster
        residencies = []
        xmax = 0.0
        for c in _clusters:
            r = self.df_cluster_idle_state_residency(c.lower())
            residencies.append(ResidencyData('{} Cluster'.format(c), r))

            max_time = r.max().values[0]
            if xmax < max_time:
                xmax = max_time

        self._plot_idle_state_residency(residencies, 'cluster', xmax, pct=pct)

###############################################################################
# Utility Methods
###############################################################################

    def _plot_idle_state_residency(self, residencies, entity_name, xmax,
                                pct=False):
        """
        Generate Idle state residency plots for the given entities.

        :param residencies: list of residencies to be plot
        :type residencies: list(namedtuple(ResidencyData)) - each tuple
            contains:
            - a label to be used as subplot title
            - a dataframe with residency for each idle state

        :param entity_name: name of the entity ('cpu' or 'cluster') used in the
            figure name
        :type entity_name: str

        :param xmax: upper bound of x-axes
        :type xmax: double

        :param pct: plot residencies in percentage
        :type pct: bool
        """
        n_plots = len(residencies)
        gs = gridspec.GridSpec(n_plots, 1)
        fig = plt.figure()

        for idx, data in enumerate(residencies):
            r = data.residency
            if r is None:
                plt.close(fig)
                return

            axes = fig.add_subplot(gs[idx])
            is_first = idx == 0
            is_last = idx+1 == n_plots
            yrange = 0.4 * max(6, len(r)) * n_plots
            if pct:
                duration = r.time.sum()
                r_pct = r.apply(lambda x: x*100/duration)
                r_pct.columns = [data.label]
                r_pct.T.plot.barh(ax=axes, stacked=True, figsize=(16, yrange))

                axes.legend(loc='lower center', ncol=7)
                axes.set_xlim(0, 100)
            else:
                r.plot.barh(ax=axes, color='g',
                            legend=False, figsize=(16, yrange))

                axes.set_xlim(0, 1.05*xmax)
                axes.set_ylabel('Idle State')
                axes.set_title(data.label)

            axes.grid(True)
            if is_last:
                if pct:
                    axes.set_xlabel('Residency [%]')
                else:
                    axes.set_xlabel('Time [s]')
            else:
                axes.set_xticklabels([])

            if is_first:
                legend_y = axes.get_ylim()[1]
                axes.annotate('Idle State Residency Time', xy=(0, legend_y),
                              xytext=(-50, 45), textcoords='offset points',
                              fontsize=18)

        figname = '{}/{}{}_idle_state_residency.png'\
                  .format(self._trace.plots_dir,
                          self._trace.plots_prefix,
                          entity_name)

        pl.savefig(figname, bbox_inches='tight')

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
