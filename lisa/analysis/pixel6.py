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
        'S4M_VDD_CPUCL0': 'CPU-Little',
        'S3M_VDD_CPUCL1': 'CPU-Mid',
        'S2M_VDD_CPUCL2': 'CPU-Big',
        'S2S_VDD_G3D': 'GPU',
    }

###############################################################################
# DataFrame Getter Methods
###############################################################################
    @TraceAnalysisBase.df_method
    @requires_events('pixel6_emeter')
    def df_power_meter(self):
        """
        Get the power meter readings across the trace.

        :retuns: A :class:`pandas.DataFrame` with:

            * A ``channel`` column (name of the power meter channel)
            * A ``power`` column (average power usage in mW since the last measurement)
        """
        df = self.trace.df_event('pixel6_emeter')
        df = df[df['chan_name'].isin(Pixel6Analysis.EMETER_CHAN_NAMES)]
        grouped = df.groupby('chan_name', observed=True, group_keys=False)

        def make_chan_df(df):
            energy_diff = df_add_delta(df, col='energy_diff', src_col='value', window=self.trace.window)['energy_diff']
            ts_diff = df_add_delta(df, col='ts_diff', src_col='ts', window=self.trace.window)['ts_diff']
            power = energy_diff / ts_diff
            df = pd.DataFrame(dict(power=power, channel=df['chan_name']))
            return df.dropna()

        df = grouped[df.columns].apply(make_chan_df)
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

        channels = list(channels or df['channel'].unique())
        if any(channel not in df['channel'].cat.categories for channel in channels):
            raise ValueError('Specified channel not found')

        channel_data = dict(iter(df[df['channel'].isin(channels)].groupby('channel', group_keys=False, observed=True)))
        return hv.Overlay([
            plot_signal(channel_data[channel]['power'], name=channel, vdim=hv.Dimension('power', label='Power', unit='mW'))
            for channel in channels
        ]).opts(title='Power usage per channel over time')
