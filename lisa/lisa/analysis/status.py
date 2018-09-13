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

# pylint: disable=E1101

""" System Status Analaysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lisa.analysis.base import AnalysisBase


class StatusAnalysis(AnalysisBase):
    """
    Support for System Status analysis

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'status'

    def __init__(self, trace):
        super(StatusAnalysis, self).__init__(trace)


###############################################################################
# DataFrame Getter Methods
###############################################################################

    def _dfg_overutilized(self):
        """
        Get data frame with sched_overutilized data.
        """
        if not self._trace.hasEvents('sched_overutilized'):
            return None

        # Build sequence of overutilization "bands"
        df = self._dfg_trace_event('sched_overutilized')

        # Remove duplicated index events, keep only last event which is the
        # only one with a non null length
        df = df[df.len != 0]
        # This filtering can also be achieved by removing events happening at
        # the same time, but perhaps this filtering is more complex
        # df = df.reset_index()\
        #         .drop_duplicates(subset='Time', keep='last')\
        #         .set_index('Time')

        return df[['len', 'overutilized']]


###############################################################################
# Plotting Methods
###############################################################################

    def plotOverutilized(self, axes=None):
        """
        Draw a plot that shows intervals of time where the system was reported
        as overutilized.

        The optional axes parameter allows to plot the signal on an existing
        graph.

        :param axes: axes on which to plot the signal
        :type axes: :mod:`matplotlib.axes.Axes`
        """
        if not self._trace.hasEvents('sched_overutilized'):
            self._log.warning('Event [sched_overutilized] not found, '
                              'plot DISABLED!')
            return

        df = self._dfg_overutilized()

        # Compute intervals in which the system is reported to be overutilized
        bands = [(t, df['len'][t], df['overutilized'][t]) for t in df.index]

        # If not axis provided: generate a standalone plot
        if not axes:
            gs = gridspec.GridSpec(1, 1)
            plt.figure(figsize=(16, 1))
            axes = plt.subplot(gs[0, 0])
            axes.set_title('System Status {white: EAS mode, '
                           'red: Non EAS mode}')
            axes.set_xlim(self._trace.x_min, self._trace.x_max)
            axes.set_yticklabels([])
            axes.set_xlabel('Time [s]')
            axes.grid(True)

        # Otherwise: draw overutilized bands on top of the specified plot
        for (start, delta, overutilized) in bands:
            if not overutilized:
                continue
            end = start + delta
            axes.axvspan(start, end, facecolor='r', alpha=0.1)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
