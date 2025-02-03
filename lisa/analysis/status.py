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
import polars as pl

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_refit_index, df_add_delta, df_deduplicate, _df_to
from lisa.notebook import _hv_neutral


class StatusAnalysis(TraceAnalysisBase):
    """
    Support for System Status analysis

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'status'

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.df_method
    @requires_events("sched_overutilized")
    def df_overutilized(self):
        """
        Get overutilized events

        :returns: A :class:`pandas.DataFrame` with:

          * A ``overutilized`` column (the overutilized status at a given time)
          * A ``len`` column (the time spent in that overutilized status)
        """
        trace = self.trace.get_view(df_fmt='polars-lazyframe')
        # Build sequence of overutilization "bands"
        df = trace.df_event('sched_overutilized')
        # Deduplicate before calling df_refit_index() since it will likely add
        # a row with duplicated state to have the expected window end
        # timestamp.
        df = df.filter(
            pl.col('overutilized') !=
            pl.col('overutilized').shift(
                1,
                # We want to select the first row, so make sure the filter
                # evaluates to true at that index.
                fill_value=pl.col('overutilized').not_(),
            )
        )
        df = df_refit_index(df, window=trace.window)

        # There might be a race between multiple CPUs to emit the
        # sched_overutilized event, so get rid of duplicated events
        df = df.with_columns(
            overutilized=pl.col('overutilized').cast(pl.Boolean),
            len=pl.col('Time').diff().shift(-1),
        )
        return df.select(('Time', 'overutilized', 'len'))

    def get_overutilized_time(self):
        """
        Return the time spent in overutilized state.
        """
        df = self.df_overutilized(df_fmt='polars-lazyframe')
        df = df.filter(pl.col('overutilized'))
        duration = df.select(
            pl.col('len').dt.total_nanoseconds().sum() / 1e9
        ).collect().item()
        return float(duration)

    def get_overutilized_pct(self):
        """
        The percentage of the time spent in overutilized state.
        """
        ou_time = self.get_overutilized_time()
        return float(100 * ou_time / self.trace.time_range)

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method
    @df_overutilized.used_events
    def plot_overutilized(self):
        """
        Draw the system's overutilized status as colored bands
        """
        df = self.df_overutilized(df_fmt='polars-lazyframe')

        df = df.filter(pl.col('overutilized'))
        df = df.select(
            pl.col('Time'),
            (pl.col('Time') + pl.col('len')).alias('width'),
        )
        df = _df_to(df, fmt='pandas')
        df.reset_index(inplace=True)

        # Compute intervals in which the system is reported to be overutilized
        return hv.VSpans(
            (df['Time'], df['width']),
            label='Overutilized'
        ).options(
            color='red',
            alpha=0.05,
            title='System-wide overutilized status',
        )

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
