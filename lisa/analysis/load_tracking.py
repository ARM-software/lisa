# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

""" Scheduler load tracking analysis module """

import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd

from lisa.analysis.base import AnalysisBase


class LoadTrackingAnalysis(AnalysisBase):
    """
    Support for scheduler load tracking analysis

    :param trace: input Trace object
    :type trace: :class:`Trace`
    """

    name = 'load_tracking'

    def __init__(self, trace):
        super().__init__(trace)

    def df_cpus(self):
        """
        A DataFrame containing per-CPU load tracking signals
        """
        try:
            event = 'sched_load_cfs_rq'
            self.check_events([event])
        except RuntimeError:
            event = 'sched_load_avg_cpu'
            self.check_events([event])

        df = self._trace.df_events(event)
        if event == 'sched_cfs_rq':
            df = df[df.path == '/']
            df.drop('rbl_load')
        else:
            pass

        return df

    def df_tasks(self):
        """
        A DataFrame containing per-task load tracking signals
        """
        try:
            event = 'sched_load_se'
            self.check_events([event])
        except RuntimeError:
            event = 'sched_load_avg_task'
            self.check_events([event])

        df = self._trace.df_events(event)
        if event == 'sched_load_se':
            df = df[df.path == '(null)']
        else:
            pass

        return df


    def plot_cpus(self, filepath=None, cpus=None):
        """
        Plot CPU-related signals

        :param cpus: list of CPUs to be plotted
        :type cpus: list(int)
        """
        cpus = cpus or list(range(self._trace.cpus_count))
        fig, axes = self.setup_plot(nrows=len(cpus), sharex=True)

        cpus_df = self.df_cpus()

        for idx, cpu in enumerate(cpus):
            axis = axes[cpu] if len(cpus) > 1 else axes
            self.cycle_colors(axis, 2 * idx)

            # Add CPU utilization
            axis.set_title('CPU{}'.format(cpu))
            df = cpus_df[cpus_df.cpu == cpu]

            if len(df):
                df[['util']].plot(ax=axis, drawstyle='steps-post', alpha=0.4)
                df[['load']].plot(ax=axis, drawstyle='steps-post', alpha=0.4)

            # Add capacities data if available
            if self._trace.hasEvents('cpu_capacity'):
                df = self._trace.df_events('cpu_capacity')
                df = df[df.cpu == cpu]
                if len(df):
                    data = df[['capacity', 'tip_capacity']]
                    data.plot(ax=axis, style=['m', '--y'],
                              drawstyle='steps-post')

            # Add overutilized signal to the plot
            self._trace.analysis.status.plot_overutilized(axis)

            axis.set_ylim(0, 1100)
            axis.set_xlim(self._trace.x_min, self._trace.x_max)

        self.save_plot(fig, filepath)
        return axes
