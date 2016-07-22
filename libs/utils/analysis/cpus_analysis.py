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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pylab as pl

from trappy.utils import listify

from analysis_module import AnalysisModule

# Configure logging
import logging

class CpusAnalysis(AnalysisModule):
    """
    Support for CPUs Signals Analysis

    :param trace: input Trace object
    :type trace: :mod:`libs.utils.Trace`
    """

    def __init__(self, trace):
        super(CpusAnalysis, self).__init__(trace)

################################################################################
# Plotting Methods
################################################################################

    def plotCPU(self, cpus=None):
        """
        Plot CPU-related signals for both big and LITTLE clusters.
        """
        if not self._trace.hasEvents('sched_load_avg_cpu'):
            logging.warn('Events [sched_load_avg_cpu] not found, '\
                    'plot DISABLED!')
            return

        # Filter on specified cpus
        if cpus is None:
            cpus = sorted(self._platform['clusters']['little'] + self._platform['clusters']['big'])
        cpus = listify(cpus)

        # Plot: big CPUs
        bcpus = set(cpus) & set(self._platform['clusters']['big'])
        self._plotCPU(bcpus, "big")

        # Plot: LITTLE CPUs
        lcpus = set(cpus) & set(self._platform['clusters']['little'])
        self._plotCPU(lcpus, "LITTLE")


################################################################################
# Utility Methods
################################################################################

    def _plotCPU(self, cpus, label=''):
        """
        Internal method that generates plots for all input CPUs.

        :param cpus: list of CPUs to be plotted
        :type cpus: list(int)
        """
        if label != '':
            label1 = '{} '.format(label)
            label2 = '_{}s'.format(label.lower())

        # Plot required CPUs
        fig, pltaxes = plt.subplots(len(cpus), 1, figsize=(16, 3*(len(cpus))));
        plt.suptitle("{}CPUs Signals".format(label1),
                     y=.99, fontsize=16, horizontalalignment='center');

        idx = 0
        for cpu in cpus:

            # Reference axes to be used
            axes = pltaxes
            if (len(cpus) > 1):
                axes = pltaxes[idx]

            # Add CPU utilization
            axes.set_title('{0:s}CPU [{1:d}]'.format(label1, cpu));
            df = self._dfg_trace_event('sched_load_avg_cpu')
            df = df[df.cpu == cpu]
            if len(df):
                df[['util_avg']].plot(ax=axes, drawstyle='steps-post', alpha=0.4);

            # if self._trace.hasEvents('sched_boost_cpu'):
            #     df = self._dfg_trace_event('sched_boost_cpu')
            #     df = df[df.cpu == cpu]
            #     if len(df):
            #         df[['usage', 'boosted_usage']].plot(
            #             ax=axes,
            #             style=['m-', 'r-'],
            #             drawstyle='steps-post');

            # Add Capacities data if avilable
            if self._trace.hasEvents('cpu_capacity'):
                df = self._dfg_trace_event('cpu_capacity')
                df = df[df.cpu == cpu]
                if len(df):
                    # data = df[['capacity', 'tip_capacity', 'max_capacity']]
                    # data.plot(ax=axes, style=['m', 'y', 'r'],
                    data = df[['capacity', 'tip_capacity' ]]
                    data.plot(ax=axes, style=['m', '--y' ],
                              drawstyle='steps-post')

            axes.set_ylim(0, 1100);
            axes.set_xlim(self._trace.x_min, self._trace.x_max);

            # Disable x-axis timestamp for top-most cpus
            if (len(cpus) > 1 and idx < len(cpus)-1):
                axes.set_xticklabels([])
                axes.set_xlabel('')
            axes.grid(True);

            idx+=1

        # Save generated plots into datadir
        figname = '{}/{}cpus{}.png'.format(self._trace.plots_dir,
                                           self._trace.plots_prefix, label2)
        pl.savefig(figname, bbox_inches='tight')

