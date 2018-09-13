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

""" EAS-specific Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pylab as pl

from base import AnalysisBase


class EasAnalysis(AnalysisBase):
    """
    Support for EAS signals anaysis

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'eas'

    def __init__(self, trace):
        super(EasAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################


###############################################################################
# Plotting Methods
###############################################################################

    def plotEDiffTime(self, tasks=None,
                      min_usage_delta=None, max_usage_delta=None,
                      min_cap_delta=None, max_cap_delta=None,
                      min_nrg_delta=None, max_nrg_delta=None,
                      min_nrg_diff=None, max_nrg_diff=None):
        """
        Plot energy_diff()-related signals on time axes.
        """
        if not self._trace.hasEvents('sched_energy_diff'):
            self._log.warning('Event [sched_energy_diff] not found, plot DISABLED!')
            return
        df = self._dfg_trace_event('sched_energy_diff')

        # Filter on 'tasks'
        if tasks is not None:
            self._log.info('Plotting EDiff data just for task(s) [%s]', tasks)
            df = df[df['comm'].isin(tasks)]

        # Filter on 'usage_delta'
        if min_usage_delta is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'usage_delta of [%d]', min_usage_delta)
            df = df[abs(df['usage_delta']) >= min_usage_delta]
        if max_usage_delta is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'usage_delta of [%d]', max_usage_delta)
            df = df[abs(df['usage_delta']) <= max_usage_delta]

        # Filter on 'cap_delta'
        if min_cap_delta is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'cap_delta of [%d]', min_cap_delta)
            df = df[abs(df['cap_delta']) >= min_cap_delta]
        if max_cap_delta is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'cap_delta of [%d]', max_cap_delta)
            df = df[abs(df['cap_delta']) <= max_cap_delta]

        # Filter on 'nrg_delta'
        if min_nrg_delta is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'nrg_delta of [%d]', min_nrg_delta)
            df = df[abs(df['nrg_delta']) >= min_nrg_delta]
        if max_nrg_delta is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'nrg_delta of [%d]', max_nrg_delta)
            df = df[abs(df['nrg_delta']) <= max_nrg_delta]

        # Filter on 'nrg_diff'
        if min_nrg_diff is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'nrg_diff of [%d]', min_nrg_diff)
            df = df[abs(df['nrg_diff']) >= min_nrg_diff]
        if max_nrg_diff is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'nrg_diff of [%d]', max_nrg_diff)
            df = df[abs(df['nrg_diff']) <= max_nrg_diff]

        # Grid: setup stats for gris
        gs = gridspec.GridSpec(4, 3, height_ratios=[2, 4, 2, 4])
        gs.update(wspace=0.1, hspace=0.1)

        # Configure plot
        fig = plt.figure(figsize=(16, 8*2+4*2+2))
        plt.suptitle("EnergyDiff Data",
                     y=.92, fontsize=16, horizontalalignment='center')

        # Plot1: src and dst CPUs
        axes = plt.subplot(gs[0, :])
        axes.set_title('Source and Destination CPUs')
        df[['src_cpu', 'dst_cpu']].plot(ax=axes, style=['bo', 'r+'])
        axes.set_ylim(-1, self._platform['cpus_count']+1)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.grid(True)
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self._trace.analysis.status.plotOverutilized(axes)

        # Plot2: energy and capacity variations
        axes = plt.subplot(gs[1, :])
        axes.set_title('Energy vs Capacity Variations')

        colors_labels = zip('gbyr', ['Optimal Accept', 'SchedTune Accept',
                                     'SchedTune Reject', 'Suboptimal Reject'])
        for color, label in colors_labels:
            subset = df[df.nrg_payoff_group == label]
            if len(subset) == 0:
                continue
            subset[['nrg_diff_pct']].plot(ax=axes, style=[color+'o'])
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.set_yscale('symlog')
        axes.grid(True)
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self._trace.analysis.status.plotOverutilized(axes)

        # Plot3: energy payoff
        axes = plt.subplot(gs[2, :])
        axes.set_title('Energy Payoff Values')
        for color, label in colors_labels:
            subset = df[df.nrg_payoff_group == label]
            if len(subset) == 0:
                continue
            subset[['nrg_payoff']].plot(ax=axes, style=[color+'o'])
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.set_yscale('symlog')
        axes.grid(True)
        axes.set_xticklabels([])
        axes.set_xlabel('')
        self._trace.analysis.status.plotOverutilized(axes)

        # Plot4: energy deltas (kernel and host computed values)
        axes = plt.subplot(gs[3, :])
        axes.set_title('Energy Deltas Values')
        df[['nrg_delta', 'nrg_diff_pct']].plot(ax=axes, style=['ro', 'b+'])
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.grid(True)
        self._trace.analysis.status.plotOverutilized(axes)

        # Save generated plots into datadir
        figname = '{}/{}ediff_time.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

        # Grid: setup stats for gris
        gs = gridspec.GridSpec(1, 3, height_ratios=[2])
        gs.update(wspace=0.1, hspace=0.1)

        fig = plt.figure(figsize=(16, 4))

        # Plot: usage, capacity and energy distributuions
        axes = plt.subplot(gs[0, 0])
        df[['usage_delta']].hist(ax=axes, bins=60)
        axes = plt.subplot(gs[0, 1])
        df[['cap_delta']].hist(ax=axes, bins=60)
        axes = plt.subplot(gs[0, 2])
        df[['nrg_delta']].hist(ax=axes, bins=60)

        # Save generated plots into datadir
        figname = '{}/{}ediff_stats.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

    def plotEDiffSpace(self, tasks=None,
                       min_usage_delta=None, max_usage_delta=None,
                       min_cap_delta=None, max_cap_delta=None,
                       min_nrg_delta=None, max_nrg_delta=None,
                       min_nrg_diff=None, max_nrg_diff=None):
        """
        Plot energy_diff()-related signals on the Performance-Energy space
        (PxE).
        """
        if not self._trace.hasEvents('sched_energy_diff'):
            self._log.warning('Event [sched_energy_diff] not found, plot DISABLED!')
            return
        df = self._dfg_trace_event('sched_energy_diff')

        # Filter on 'tasks'
        if tasks is not None:
            self._log.info('Plotting EDiff data just for task(s) [%s]', tasks)
            df = df[df['comm'].isin(tasks)]

        # Filter on 'usage_delta'
        if min_usage_delta is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'usage_delta of [%d]', min_usage_delta)
            df = df[abs(df['usage_delta']) >= min_usage_delta]
        if max_usage_delta is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'usage_delta of [%d]', max_usage_delta)
            df = df[abs(df['usage_delta']) <= max_usage_delta]

        # Filter on 'cap_delta'
        if min_cap_delta is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'cap_delta of [%d]', min_cap_delta)
            df = df[abs(df['cap_delta']) >= min_cap_delta]
        if max_cap_delta is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'cap_delta of [%d]', max_cap_delta)
            df = df[abs(df['cap_delta']) <= max_cap_delta]

        # Filter on 'nrg_delta'
        if min_nrg_delta is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'nrg_delta of [%d]', min_nrg_delta)
            df = df[abs(df['nrg_delta']) >= min_nrg_delta]
        if max_nrg_delta is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'nrg_delta of [%d]', max_nrg_delta)
            df = df[abs(df['nrg_delta']) <= max_nrg_delta]

        # Filter on 'nrg_diff'
        if min_nrg_diff is not None:
            self._log.info('Plotting EDiff data just with minimum '
                           'nrg_diff of [%d]', min_nrg_diff)
            df = df[abs(df['nrg_diff']) >= min_nrg_diff]
        if max_nrg_diff is not None:
            self._log.info('Plotting EDiff data just with maximum '
                           'nrg_diff of [%d]', max_nrg_diff)
            df = df[abs(df['nrg_diff']) <= max_nrg_diff]

        # Grid: setup grid for P-E space
        gs = gridspec.GridSpec(1, 2, height_ratios=[2])
        gs.update(wspace=0.1, hspace=0.1)

        fig = plt.figure(figsize=(16, 8))

        # Get min-max of each axes
        x_min = df.nrg_diff_pct.min()
        x_max = df.nrg_diff_pct.max()
        y_min = df.cap_delta.min()
        y_max = df.cap_delta.max()
        axes_min = min(x_min, y_min)
        axes_max = max(x_max, y_max)

        # # Tag columns by usage_delta
        # ccol = df.usage_delta
        # df['usage_delta_group'] = np.select(
        #     [ccol < 150, ccol < 400, ccol < 600],
        #     ['< 150', '< 400', '< 600'], '>= 600')
        #
        # # Tag columns by nrg_payoff
        # ccol = df.nrg_payoff
        # df['nrg_payoff_group'] = np.select(
        #     [ccol > 2e9, ccol > 0, ccol > -2e9],
        #     ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject'],
        #     'Suboptimal Reject')

        # Plot: per usage_delta values
        axes = plt.subplot(gs[0, 0])

        for color, label in zip('bgyr', ['< 150', '< 400', '< 600', '>= 600']):
            subset = df[df.usage_delta_group == label]
            if len(subset) == 0:
                continue
            plt.scatter(subset.nrg_diff_pct, subset.cap_delta,
                        s=subset.usage_delta,
                        c=color, label='task_usage ' + str(label),
                        axes=axes)

        # Plot space axes
        plt.plot((0, 0), (-1025, 1025), 'y--', axes=axes)
        plt.plot((-1025, 1025), (0, 0), 'y--', axes=axes)

        # # Perf cuts
        # plt.plot((0, 100), (0, 100*delta_pb), 'b--',
        #          label='PB (Perf Boost)')
        # plt.plot((0, -100), (0, -100*delta_pc), 'r--',
        #          label='PC (Perf Constraint)')
        #
        # # Perf boost setups
        # for y in range(0,6):
        #     plt.plot((0, 500), (0,y*100), 'g:')
        # for x in range(0,5):
        #     plt.plot((0, x*100), (0,500), 'g:')

        axes.legend(loc=4, borderpad=1)

        plt.xlim(1.1*axes_min, 1.1*axes_max)
        plt.ylim(1.1*axes_min, 1.1*axes_max)

        # axes.title('Performance-Energy Space')
        axes.set_xlabel('Energy diff [%]')
        axes.set_ylabel('Capacity diff [%]')

        # Plot: per usage_delta values
        axes = plt.subplot(gs[0, 1])

        colors_labels = zip('gbyr', ['Optimal Accept', 'SchedTune Accept',
                                     'SchedTune Reject', 'Suboptimal Reject'])
        for color, label in colors_labels:
            subset = df[df.nrg_payoff_group == label]
            if len(subset) == 0:
                continue
            plt.scatter(subset.nrg_diff_pct, subset.cap_delta,
                        s=60,
                        c=color,
                        marker='+',
                        label='{} Region'.format(label),
                        axes=axes)
                        # s=subset.usage_delta,

        # Plot space axes
        plt.plot((0, 0), (-1025, 1025), 'y--', axes=axes)
        plt.plot((-1025, 1025), (0, 0), 'y--', axes=axes)

        # # Perf cuts
        # plt.plot((0, 100), (0, 100*delta_pb), 'b--',
        #          label='PB (Perf Boost)')
        # plt.plot((0, -100), (0, -100*delta_pc), 'r--',
        #          label='PC (Perf Constraint)')
        #
        # # Perf boost setups
        # for y in range(0,6):
        #     plt.plot((0, 500), (0,y*100), 'g:')
        # for x in range(0,5):
        #     plt.plot((0, x*100), (0,500), 'g:')

        axes.legend(loc=4, borderpad=1)

        plt.xlim(1.1*axes_min, 1.1*axes_max)
        plt.ylim(1.1*axes_min, 1.1*axes_max)

        # axes.title('Performance-Energy Space')
        axes.set_xlabel('Energy diff [%]')
        axes.set_ylabel('Capacity diff [%]')

        plt.title('Performance-Energy Space')

        # Save generated plots into datadir
        figname = '{}/{}ediff_space.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

    def plotSchedTuneConf(self):
        """
        Plot the configuration of SchedTune.
        """
        if not self._trace.hasEvents('sched_tune_config'):
            self._log.warning('Event [sched_tune_config] not found, plot DISABLED!')
            return
        # Grid
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        gs.update(wspace=0.1, hspace=0.1)

        # Figure
        plt.figure(figsize=(16, 2*6))
        plt.suptitle("SchedTune Configuration",
                     y=.97, fontsize=16, horizontalalignment='center')

        # Plot: Margin
        axes = plt.subplot(gs[0, 0])
        axes.set_title('Margin')
        data = self._dfg_trace_event('sched_tune_config')[['margin']]
        data.plot(ax=axes, drawstyle='steps-post', style=['b'])
        axes.set_ylim(0, 110)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.xaxis.set_visible(False)

        # Plot: Boost mode
        axes = plt.subplot(gs[1, 0])
        axes.set_title('Boost mode')
        data = self._dfg_trace_event('sched_tune_config')[['boostmode']]
        data.plot(ax=axes, drawstyle='steps-post')
        axes.set_ylim(0, 4)
        axes.set_xlim(self._trace.x_min, self._trace.x_max)
        axes.xaxis.set_visible(True)

        # Save generated plots into datadir
        figname = '{}/{}schedtune_conf.png'\
                  .format(self._trace.plots_dir, self._trace.plots_prefix)
        pl.savefig(figname, bbox_inches='tight')

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
