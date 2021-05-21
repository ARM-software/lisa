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

import holoviews as hv

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_refit_index, df_add_delta, df_deduplicate
from lisa.notebook import _hv_neutral


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
        df = self.trace.df_event('sched_overutilized')
        # There might be a race between multiple CPUs to emit the
        # sched_overutilized event, so get rid of duplicated events
        df = df_deduplicate(df, cols=['overutilized'], keep='first', consecutives=True)
        df = df_add_delta(df, col='len', window=self.trace.window)
        # Ignore the last line added by df_refit_index() with a NaN len
        df = df.iloc[:-1]
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

    @TraceAnalysisBase.plot_method
    @df_overutilized.used_events
    def plot_overutilized(self):
        """
        Draw the system's overutilized status as colored bands
        """
        df = self.df_overutilized()
        if not df.empty:
            df = df_refit_index(df, window=self.trace.window)

            # Compute intervals in which the system is reported to be overutilized
            return hv.Overlay(
                [
                    hv.VSpan(
                        start,
                        start + delta,
                        label='Overutilized'
                    ).options(
                        color='red',
                        alpha=0.05,
                    )
                    for start, delta, overutilized in df[['len', 'overutilized']].itertuples()
                    if overutilized
                ]
            ).options(
                title='System-wide overutilized status'
            )
        else:
            return _hv_neutral()

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
