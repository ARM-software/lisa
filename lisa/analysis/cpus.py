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

""" CPUs Analysis Module """

import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd

from lisa.analysis.base import AnalysisBase


class CpusAnalysis(AnalysisBase):
    """
    Support for CPUs Signals Analysis

    :param trace: input Trace object
    :type trace: :class:`Trace`
    """

    name = 'cpus'

    def __init__(self, trace):
        super(CpusAnalysis, self).__init__(trace)


###############################################################################
# DataFrame Getter Methods
###############################################################################

    def df_context_switches(self):
        """
        Compute number of context switches on each CPU.

        :returns: :mod:`pandas.DataFrame`
        """
        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Events [sched_switch] not found, context switch '
                              'computation not possible!')
            return None

        sched_df = self.df_events('sched_switch')
        cpus = list(range(self._platform['cpus_count']))
        ctx_sw_df = pd.DataFrame(
            [len(sched_df[sched_df['__cpu'] == cpu]) for cpu in cpus],
            index=cpus,
            columns=['context_switch_cnt']
        )
        ctx_sw_df.index.name = 'cpu'
        return ctx_sw_df

    def df_cpu_wakeups(self, cpus=None):
        """"
        Get a DataFrame showing when a CPU was woken from idle

        :param cpus: List of CPUs to find wakeups for. If None, all CPUs.
        :type cpus: list(int) or None

        :returns: :mod:`pandas.DataFrame` with one column ``cpu``, where each
                  row shows a time when the given ``cpu`` was woken up from
                  idle.
        """
        if not self._trace.hasEvents('cpu_idle'):
            self._log.warning('Events [cpu_idle] not found, cannot '
                              'get CPU wakeup events.')
            return None

        cpus = cpus or list(range(self._trace.platform['cpus_count']))

        sr = pd.Series()
        for cpu in cpus:
            cpu_sr = self._trace.getCPUActiveSignal(cpu)
            cpu_sr = cpu_sr[cpu_sr == 1]
            cpu_sr = cpu_sr.replace(1, cpu)
            sr = sr.append(cpu_sr)

        return pd.DataFrame({'cpu': sr}).sort_index()

###############################################################################
# Plotting Methods
###############################################################################

    def plot_cpu(self, cpus=None):
        """
        Plot CPU-related signals for both big and LITTLE clusters.

        :param cpus: list of CPUs to be plotted
        :type cpus: list(int)
        """
        if not self._trace.hasEvents('sched_load_avg_cpu'):
            self._log.warning('Events [sched_load_avg_cpu] not found, '
                              'plot DISABLED!')
            return

        # Filter on specified cpus
        if cpus is None:
            cpus = sorted(self._big_cpus + self._little_cpus)

        # Plot: big CPUs
        bcpus = set(cpus).intersection(self._big_cpus)
        if bcpus:
            self._plot_cpu(bcpus, "big")

        # Plot: LITTLE CPUs
        lcpus = set(cpus).intersection(self._little_cpus)
        if lcpus:
            self._plot_cpu(lcpus, "LITTLE")


###############################################################################
# Utility Methods
###############################################################################

    def _plot_cpu(self, cpus, label=''):
        """
        Internal method that generates plots for all input CPUs.

        :param cpus: list of CPUs to be plotted
        :type cpus: list(int)
        """
        if label != '':
            label1 = '{} '.format(label)
            label2 = '_{}s'.format(label.lower())

        # Plot required CPUs
        _, pltaxes = plt.subplots(len(cpus), 1, figsize=(16, 3*(len(cpus))))

        idx = 0
        for cpu in cpus:

            # Reference axes to be used
            axes = pltaxes
            if len(cpus) > 1:
                axes = pltaxes[idx]

            # Add CPU utilization
            axes.set_title('{0:s}CPU [{1:d}]'.format(label1, cpu))
            df = self.df_events('sched_load_avg_cpu')
            df = df[df.cpu == cpu]
            if len(df):
                df[['util_avg']].plot(ax=axes, drawstyle='steps-post',
                                      alpha=0.4)

            # if self._trace.hasEvents('sched_boost_cpu'):
            #     df = self.df_events('sched_boost_cpu')
            #     df = df[df.cpu == cpu]
            #     if len(df):
            #         df[['usage', 'boosted_usage']].plot(
            #             ax=axes,
            #             style=['m-', 'r-'],
            #             drawstyle='steps-post');

            # Add Capacities data if avilable
            if self._trace.hasEvents('cpu_capacity'):
                df = self.df_events('cpu_capacity')
                df = df[df.cpu == cpu]
                if len(df):
                    # data = df[['capacity', 'tip_capacity', 'max_capacity']]
                    # data.plot(ax=axes, style=['m', 'y', 'r'],
                    data = df[['capacity', 'tip_capacity']]
                    data.plot(ax=axes, style=['m', '--y'],
                              drawstyle='steps-post')

            # Add overutilized signal to the plot
            self._trace.analysis.status.plot_overutilized(axes)

            axes.set_ylim(0, 1100)
            axes.set_xlim(self._trace.x_min, self._trace.x_max)

            if idx == 0:
                axes.annotate("{}CPUs Signals".format(label1),
                              xy=(0, axes.get_ylim()[1]),
                              xytext=(-50, 25),
                              textcoords='offset points', fontsize=16)
            # Disable x-axis timestamp for top-most cpus
            if len(cpus) > 1 and idx < len(cpus)-1:
                axes.set_xticklabels([])
                axes.set_xlabel('')
            axes.grid(True)

            idx += 1

        # Save generated plots into datadir
        figname = '{}/{}cpus{}.png'.format(self._trace.plots_dir,
                                           self._trace.plots_prefix, label2)
        pl.savefig(figname, bbox_inches='tight')

    def plot_context_switch(self):
        """
        Plot histogram of context switches on each CPU.
        """
        if not self._trace.hasEvents('sched_switch'):
            self._log.warning('Events [sched_switch] not found, plot DISABLED!')
            return

        ctx_sw_df = self.df_context_switches()
        ax = ctx_sw_df.plot.bar(title="Per-CPU Task Context Switches",
                                legend=False,
                                figsize=(16, 8))
        ax.grid()

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
