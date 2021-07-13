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

import pandas as pd
import holoviews as hv

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events, CPU
from lisa.datautils import df_window


class CpusAnalysis(TraceAnalysisBase):
    """
    Support for CPUs signals analysis
    """

    name = 'cpus'

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.cache
    @requires_events('sched_switch')
    def df_context_switches(self):
        """
        Compute number of context switches on each CPU.

        :returns: A :class:`pandas.DataFrame` with:

          * A ``context_switch_cnt`` column (the number of context switch per CPU)
        """
        # Since we want to count the number of context switches, we don't want
        # all tasks to appear
        sched_df = self.trace.df_event('sched_switch', signals_init=False)
        # Make sure to only get the switches inside the window
        sched_df = df_window(
            sched_df,
            method='exclusive',
            window=self.trace.window,
            clip_window=False,
        )
        cpus = list(range(self.trace.cpus_count))
        ctx_sw_df = pd.DataFrame(
            [len(sched_df[sched_df['__cpu'] == cpu]) for cpu in cpus],
            index=cpus,
            columns=['context_switch_cnt']
        )
        ctx_sw_df.index.name = 'cpu'

        return ctx_sw_df

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method
    @df_context_switches.used_events
    def plot_context_switches(self):
        """
        Plot histogram of context switches on each CPU.
        """
        ctx_sw_df = self.df_context_switches()
        return hv.Bars(
            ctx_sw_df["context_switch_cnt"]
        ).options(
            title='Per-CPU Task Context Switches',
            xlabel='CPU',
            ylabel='Number of context switches',
            invert_axes=True,
        )

    @TraceAnalysisBase.plot_method
    def plot_orig_capacity(self, cpu: CPU):
        """
        Plot the orig capacity of a CPU onto a given axis

        :param cpu: The CPU
        :type cpu: int
        """
        orig_capacities = self.trace.plat_info['cpu-capacities']['orig']
        return hv.HLine(
            orig_capacities[cpu],
            label='orig capacity'
        ).options(
            backend='matplotlib',
            linestyle='--',
        ).options(
            backend='bokeh',
            line_dash='dashed',
        )

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
