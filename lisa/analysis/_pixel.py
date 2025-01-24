# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, ARM Limited and contributors.
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
"""
Pixel phones specific analysis.
"""

import pandas as pd
import polars as pl
import holoviews as hv
from holoviews import opts

from lisa.datautils import SignalDesc
from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.notebook import plot_signal


class PixelAnalysis(TraceAnalysisBase):
    """
    Support for Pixel phones specific data analysis

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    ###############################################################################
    # DataFrame Getter Methods
    ###############################################################################
    @TraceAnalysisBase.df_method
    @requires_events('pixel6_emeter')
    def df_power_meter(self):
        """
        Get the power meter readings across the trace.

        :returns: A :class:`pandas.DataFrame` with:

            * A ``channel`` column (name of the power meter channel)
            * A ``power`` column (average power usage in mW since the last measurement)
            * A ``energy`` column (energy samples in mJ provided by the PMIC)
        """
        name_map = self.EMETER_CHAN_NAMES
        trace = self.trace.get_view(df_fmt='polars-lazyframe')

        signals = [
            SignalDesc('pixel6_emeter', ['chan_name']),
        ]
        df = trace.df_event('pixel6_emeter', signals=signals)
        df = df.rename({'value': 'energy'})
        df = df.filter(pl.col('chan_name').is_in(name_map.keys()))

        nrg_diff = pl.col('energy').diff()
        ts_diff = pl.col('ts').diff()
        chan = pl.col('chan_name')
        df = df.with_columns(
            power=(nrg_diff / ts_diff).over('chan_name'),
            channel=chan.replace_strict(name_map, default=chan),
        )

        return df.select(('Time', 'channel', 'energy', 'power'))

    ###############################################################################
    # Plotting methods
    ###############################################################################
    @TraceAnalysisBase.plot_method
    @df_power_meter.used_events
    def plot_power_meter(self, channels=None):
        """
        Plot the power meter readings from various channels.

        :param channels: List of channels to plot
        :type channels: list(str)

        The channels needs to correspond to values in the ``channel`` column of df_power_meter().
        """
        df = self.df_power_meter()

        channels = list(channels or df['channel'].unique())
        if any(channel not in df['channel'].cat.categories for channel in channels):
            raise ValueError('Specified channel not found')

        channel_data = dict(iter(df[df['channel'].isin(channels)].groupby('channel', group_keys=False, observed=True)))
        return hv.Overlay([
            plot_signal(channel_data[channel]['power'], name=channel, vdim=hv.Dimension('power', label='Power', unit='mW'))
            for channel in channels
        ]).opts(title='Power usage per channel over time')
