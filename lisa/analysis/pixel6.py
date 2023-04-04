# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2023, ARM Limited and contributors.
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

import pandas as pd
import holoviews as hv
from holoviews import opts

from lisa.datautils import df_add_delta
from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.notebook import plot_signal


class Pixel6Analysis(TraceAnalysisBase):
    """
    Support for Pixel 6-specific data analysis

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    name = 'pixel6'

    EMETER_CHAN_NAMES = {
        'S4M_VDD_CPUCL0': 'little',
        'S3M_VDD_CPUCL1': 'mid',
        'S2M_VDD_CPUCL2': 'big',
    }

###############################################################################
# DataFrame Getter Methods
###############################################################################
    @TraceAnalysisBase.cache
    @requires_events('pixel6_emeter')
    def df_power_meter(self):
        """
        Get the power meter readings across the trace.

        :retuns: A :class:`pandas.DataFrame` with:

            * A ``channel`` column (description of the power usage channel)
            * A ``value`` column (average power usage in mW since the last measurement)
        """
        df = self.trace.df_event('pixel6_emeter')
        df = df[df['chan_name'].isin(Pixel6Analysis.EMETER_CHAN_NAMES)]
        df = df.groupby(['chan_name'], observed=True, group_keys=False).apply(
            lambda x: df_add_delta(x, col='value_diff', src_col='value', window=self.trace.window)['value_diff'] / df_add_delta(x, col='ts_diff', src_col='ts', window=self.trace.window)['ts_diff']
        ).reset_index().rename(columns={0:'value', 'chan_name':'channel'}).dropna().set_index('Time')
        df['channel'] = df['channel'].astype('category').cat.rename_categories(Pixel6Analysis.EMETER_CHAN_NAMES)

        return df

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

        channels = channels if channels is not None else list(df['channel'].cat.categories)
        if any(channel not in df['channel'].cat.categories for channel in channels):
            raise ValueError('Specified channel not found')

        channel_data = dict(iter(df[df['channel'].isin(channels)].groupby(['channel'], group_keys=False, observed=True)))
        return hv.Overlay([
            plot_signal(channel_data[channel]['value'], name=channel, vdim=hv.Dimension('value', label='Power', unit='mW'))
            for channel in channels
        ]).opts(title='Power usage per channel over time')
