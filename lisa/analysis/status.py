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

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_refit_index


class StatusAnalysis(TraceAnalysisBase):
    """
    Support for System Status analysis

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'status'

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @requires_events("sched_overutilized")
    def df_overutilized(self):
        """
        Get overutilized events

        :returns: A :class:`pandas.DataFrame` with:

          * A ``overutilized`` column (the overutilized status at a given time)
          * A ``len`` column (the time spent in that overutilized status)
        """
        # Build sequence of overutilization "bands"
        df = self.trace.df_events('sched_overutilized')

        # Remove duplicated index events, keep only last event which is the
        # only one with a non null length
        df = df[df.len != 0]
        # This filtering can also be achieved by removing events happening at
        # the same time, but perhaps this filtering is more complex
        # df = df.reset_index()\
        #         .drop_duplicates(subset='Time', keep='last')\
        #         .set_index('Time')
        return df[['len', 'overutilized']]

    def get_overutilized_time(self):
        """
        Return the time spent in overutilized state.
        """
        df = self.df_overutilized()
        return df[df['overutilized'] == 1]['len'].sum()

    def get_overutilized_pct(self):
        """
        The percentage of the time spent in overutilized state.
        """
        ou_time = self.get_overutilized_time()
        return 100 * ou_time / self.trace.time_range

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method()
    @df_overutilized.used_events
    def plot_overutilized(self, axis, local_fig):
        """
        Draw the system's overutilized status as colored bands
        """

        df = self.df_overutilized()
        df = df_refit_index(df, self.trace.start, self.trace.end)

        # Compute intervals in which the system is reported to be overutilized
        bands = [(t, df['len'][t], df['overutilized'][t]) for t in df.index]

        color = self.get_next_color(axis)
        label = "Overutilized"
        for (start, delta, overutilized) in bands:
            if not overutilized:
                continue

            end = start + delta
            axis.axvspan(start, end, alpha=0.2, facecolor=color, label=label)

            if label:
                label = None

        axis.legend()

        if local_fig:
            axis.set_title("System-wide overutilized status")

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
