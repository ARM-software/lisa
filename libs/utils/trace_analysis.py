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

import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pylab as pl
import re
import sys
import trappy
import operator
from trappy.utils import listify
from devlib.utils.misc import memoized
from collections import namedtuple

# Configure logging
import logging

NON_IDLE_STATE = 4294967295

ResidencyTime = namedtuple('ResidencyTime', ['total', 'active'])
ResidencyData = namedtuple('ResidencyData', ['label', 'residency'])

class TraceAnalysis(object):

    def __init__(self, trace, tasks=None, plotsdir=None, prefix=''):
        """
        Support for plotting a standard set of trace singals and events
        """

        self.trace = trace
        self.tasks = tasks
        self.plotsdir = plotsdir
        self.prefix = prefix

        # Keep track of the Trace::platform
        self.platform = trace.platform

        # Plotsdir is byb default the trace dir
        if self.plotsdir is None:
            self.plotsdir = self.trace.data_dir

        # Minimum and Maximum x_time to use for all plots
        self.x_min = 0
        self.x_max = self.trace.time_range

        # Reset x axis time range to full scale
        t_min = self.trace.window[0]
        t_max = self.trace.window[1]
        self.setXTimeRange(t_min, t_max)

    def setXTimeRange(self, t_min=None, t_max=None):
        if t_min is None:
            self.x_min = 0
        else:
            self.x_min = t_min
        if t_max is None:
            self.x_max = self.trace.time_range
        else:
            self.x_max = t_max
        logging.info('Set plots time range to (%.6f, %.6f)[s]',
                self.x_min, self.x_max)

    def plotFunctionStats(self, functions=None, metrics='avg'):
        """
        Plot functions profiling metrics for the specified kernel functions.

        For each speficied metric a barplot is generated which report the value
        of the metric when the kernel function has been executed on each CPU.
        By default all the kernel functions are plotted.

        :param functions: the name of list of name of kernel functions to plot
        :type functions: str or list

        :param metrics: the metrics to plot
                        avg   - average execution time
                        time  - total execution time
        :type metrics: srt or list
        """
        if not hasattr(self.trace, '_functions_stats_df'):
            logging.warning('Functions stats data not available')
            return

        metrics = listify(metrics)
        df = self.trace.functions_stats_df(functions)

        # Check that all the required metrics are acutally availabe
        available_metrics = df.columns.tolist()
        if not set(metrics).issubset(set(available_metrics)):
            msg = 'Metrics {} not supported, available metrics are {}'\
                    .format(set(metrics) - set(available_metrics),
                            available_metrics)
            raise ValueError(msg)

        for _m in metrics:
            if _m.upper() == 'AVG':
                title = 'Average Completion Time per CPUs'
                ylabel = 'Completion Time [us]'
            if _m.upper() == 'TIME':
                title = 'Total Execution Time per CPUs'
                ylabel = 'Execution Time [us]'
            data = df[_m.lower()].unstack()
            axes = data.plot(kind='bar',
                             figsize=(16,8), legend=True,
                             title=title, table=True)
            axes.set_ylabel(ylabel)
            axes.get_xaxis().set_visible(False)


    def __addCapacityColum(self):
        df = self.trace.df('cpu_capacity')
        # Rename CPU and Capacity columns
        df.rename(columns={'cpu_id':'cpu'}, inplace=True)
        # Add column with LITTLE and big CPUs max capacities
        nrg_model = self.platform['nrg_model']
        max_lcap = nrg_model['little']['cpu']['cap_max']
        max_bcap = nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [tip_lcap], tip_bcap)

    def __plotCPU(self, cpus=None, label=''):
        if cpus is None or len(cpus) == 0:
            return
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
            df = self.trace.df('sched_load_avg_cpu')
            df = df[df.cpu == cpu]
            if len(df):
                df[['util_avg']].plot(ax=axes, drawstyle='steps-post', alpha=0.4);

            # if self.trace.hasEvents('sched_boost_cpu'):
            #     df = self.trace.df('sched_boost_cpu')
            #     df = df[df.cpu == cpu]
            #     if len(df):
            #         df[['usage', 'boosted_usage']].plot(
            #             ax=axes,
            #             style=['m-', 'r-'],
            #             drawstyle='steps-post');

            # Add Capacities data if avilable
            if self.trace.hasEvents('cpu_capacity'):
                df = self.trace.df('cpu_capacity')
                df = df[df.cpu == cpu]
                if len(df):
                    # data = df[['capacity', 'tip_capacity', 'max_capacity']]
                    # data.plot(ax=axes, style=['m', 'y', 'r'],
                    data = df[['capacity', 'tip_capacity' ]]
                    data.plot(ax=axes, style=['m', '--y' ],
                              drawstyle='steps-post')

            axes.set_ylim(0, 1100);
            axes.set_xlim(self.x_min, self.x_max);

            # Disable x-axis timestamp for top-most cpus
            if (len(cpus) > 1 and idx < len(cpus)-1):
                axes.set_xticklabels([])
                axes.set_xlabel('')
            axes.grid(True);

            idx+=1

        # Save generated plots into datadir
        figname = '{}/{}cpus{}.png'.format(self.plotsdir, self.prefix, label2)
        pl.savefig(figname, bbox_inches='tight')

    def plotCPU(self, cpus=None):
        if not self.trace.hasEvents('sched_load_avg_cpu'):
            logging.warn('Events [sched_load_avg_cpu] not found, '\
                    'plot DISABLED!')
            return

        # Filter on specified cpus
        if cpus is None:
            cpus = sorted(self.platform['clusters']['little'] + self.platform['clusters']['big'])

        # Plot: big CPUs
        bcpus = set(cpus) & set(self.platform['clusters']['big'])
        self.__plotCPU(bcpus, "big")

        # Plot: LITTLE CPUs
        lcpus = set(cpus) & set(self.platform['clusters']['little'])
        self.__plotCPU(lcpus, "LITTLE")

