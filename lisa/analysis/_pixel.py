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
from typing import List, Optional

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
    def plot_power_meter(self, channels: Optional[List[str]] = None, metrics: Optional[List[str]] = None):
        """
        Plot the power meter readings from various channels.

        :param channels: List of channels to plot
        :type channels: list(str)

        :param metrics: List of metrics to plot. Can be:
            * ``"power"``: plot the power (mW)
            * ``"energy"``: plot the energy (mJ)
        :type metrics: list(str)

        The channels needs to correspond to values in the ``channel`` column of df_power_meter().
        """
        all_metrics = {
            'power': 'mW',
            'energy': 'mJ',
        }
        metrics = sorted(['power'] if metrics is None else metrics)
        channels = channels or sorted(self.EMETER_CHAN_NAMES.values())

        def check_allowed(kind, values, allowed):
            forbidden = set(values) - set(allowed)
            if forbidden:
                forbidden = ', '.join(sorted(forbidden))
                raise ValueError(f'{kind} names not recognized: {forbidden}')

        check_allowed('Channel', channels, self.EMETER_CHAN_NAMES.values())
        check_allowed('Metrics', metrics, all_metrics.keys())

        df = self.df_power_meter(df_fmt='polars-lazyframe')
        df = df.filter(pl.col('channel').is_in(channels))
        df = df.select(('Time', *metrics, 'channel'))
        df = df.collect()
        per_channel = df.partition_by('channel', include_key=False, as_dict=True)

        return hv.Overlay([
            plot_signal(
                df.select(('Time', metric)),
                name=f'{channel} {metric}',
                vdim=hv.Dimension(metric, label=metric.title(), unit=all_metrics[metric]),
                window=self.trace.window,
            )
            for (channel,), df in sorted(per_channel.items())
            for metric in sorted(metrics)
        ]).opts(
            title='Power usage per channel over time',
            multi_y=True,
        )
