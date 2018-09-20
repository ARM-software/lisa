# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, Arm Limited and contributors.
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

""" Thermal Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import operator
import os

from trappy.utils import listify
from devlib.utils.misc import memoized, list_to_mask, mask_to_list

from lisa.analysis.base import AnalysisBase, ResidencyTime, ResidencyData
from bart.common.Utils import area_under_curve
from matplotlib.ticker import MaxNLocator


class ThermalAnalysis(AnalysisBase):
    """
    Support for plotting Thermal Analysis data

    :param trace: input Trace object
    :type trace: lisa.Trace
    """

    name = 'thermal'

###############################################################################
# Analysis properties
###############################################################################

    @property
    @memoized
    def thermal_zones(self):
        """
        Get thermal zone ids that appear in the trace
        """
        df = self._trace.df_events('thermal_temperature')
        return df["thermal_zone"].unique().tolist()

    @property
    @memoized
    def cpufreq_cdevs(self):
        """
        Get cpufreq cooling devices that appear in the trace
        """
        df = self._trace.df_events('thermal_power_cpu_limit')
        res = df['cpus'].unique().tolist()
        return [mask_to_list(mask) for mask in res]

    @property
    @memoized
    def devfreq_cdevs(self):
        """
        Get devfreq cooling devices that appear in the trace
        """
        df = self._trace.df_events('thermal_power_devfreq_limit')
        return df['type'].unique().tolist()

###############################################################################
# DataFrame Getter Methods
###############################################################################

    def df_thermal_zone_temperature(self, ids=None):
        """
        Get the temperature readings of one or more thermal zone(s)
        (all by default)

        :param ids: The thermal zones to consider
        :type ids: list(int)
        """
        df = self._trace.df_events('thermal_temperature')
        df = df[['id', 'thermal_zone', 'temp']]

        if ids is not None:
            df = df[df.id.isin(ids)]

        return df

    def df_cpufreq_cooling_state(self, cpus=None):
        """
        Get the cooling states of one or more cpufreq cooling device(s)
        (all by default)

        :param cpus: The CPUs to consider
        :type cpus: list(int)
        """
        df = self._trace.df_events('thermal_power_cpu_limit')
        df = df[['cpus', 'freq', 'cdev_state']]

        if cpus is not None:
            # Find masks that match the requested CPUs
            # This can include other CPUs
            masks = self._matching_masks(cpus)
            df = df[df.cpus.isin(masks)]

        return df

    def df_devfreq_cooling_state(self, devices=None):
        """
        Get the cooling states of one or more devfreq cooling device(s)
        (all by default)

        :param devices: The devfreq devices to consider
        :type device: list(str)
        """
        df = self._trace.df_events('thermal_power_devfreq_limit')
        df = df[['type', 'freq', 'cdev_state']]

        if devices is not None:
            df = df[df.type.isin(devices)]

        return df


###############################################################################
# Plotting Methods
###############################################################################

    def plot_temperature(self, thermal_zones=None, ax=None):
        """
        Plot temperature of thermal zones (all by default)

        Requires the following trace event:
            - thermal_temperature

        :param thermal_zones: ID(s) of the zones to be plotted.
            All the zones are plotted by default.
            IDs can be found in syfs:  /sys/class/thermal/thermal_zone<ID>
        :type thermal_zones: list(int)
        """
        if not self._trace.hasEvents('thermal_temperature'):
            self._log.warning('Event [{}] not found, plot DISABLED!'
                              .format('thermal_temperature'))
            return

        plot_df = self.df_thermal_zone_temperature(thermal_zones)

        def stringify_tz(id):
            return plot_df[plot_df.id == id]['thermal_zone'].unique()[0]

        filters = None if thermal_zones is None else {'thermal_zone' : thermal_zones}
        self._plot_generic(plot_df, 'id', filters=filters, columns=['temp'],
                          prettify_name=stringify_tz,
                          drawstyle='steps-post', ax=ax
        )

        if thermal_zones is None:
            suffix = ''
        else:
            suffix = '_' + '_'.join(map(str, thermal_zones))

        # Save generated plots into datadir
        figname = os.path.join(
            self._trace.plots_dir,
            '{}thermal_temperature{}.png'.format(
                self._trace.plots_dir, self._trace.plots_prefix, suffix
            )
        )

        pl.savefig(figname, bbox_inches='tight')

    def plot_cpu_cooling_states(self, cpus=None, ax=None):
        """
        Plot the state evolution of cpufreq cooling devices (all by default)

        Requires the following trace event:
            - thermal_power_cpu_limit

        :param cpus: list of CPUs to plot. Whole clusters can be controlled as
            a single cooling device, they will be plotted as long as one of their
            CPUs is in the list.
        :type cpus: list(int)
        """
        if not self._trace.hasEvents('thermal_power_cpu_limit'):
            self._log.warning('Event [{}] not found, plot DISABLED!'
                              .format('thermal_power_cpu_limit'))
            return

        plot_df = self._trace.df_events('thermal_power_cpu_limit')

        def stringify_mask(mask):
            return 'CPUs {}'.format(mask_to_list(mask))

        # Find masks that match the requested CPUs
        # This can include other CPUs
        masks = None
        if cpus is not None:
            masks = self._matching_masks(cpus)

        filters = None if masks is None else {'cpus' : masks}
        _ax = self._plot_generic(plot_df, 'cpus', filters=filters, columns=['cdev_state'],
                          prettify_name=stringify_mask,
                          drawstyle='steps-post', ax=ax
        )

        if ax is None:
            ax = _ax

        # Cdev status is an integer series
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y')

        if cpus is None:
            suffix = ''
        else:
            suffix = '_' + '_'.join(map(str, cpus))

        # Save generated plots into datadir
        figname = os.path.join(
            self._trace.plots_dir,
            '{}thermal_cpufreq_cdev_state{}.png'.format(
                self._trace.plots_dir, self._trace.plots_prefix, suffix
            )
        )
        pl.savefig(figname, bbox_inches='tight')

    def plot_dev_freq_cooling_states(self, devices=None, ax=None):
        """
        Plot the state evolution of devfreq cooling devices (all by default)

        Requires the following trace event:
            - thermal_power_devfreq_limit

        :param devices: list of devfreq devices to plot.
        :type cpus: list(int)
        """
        if not self._trace.hasEvents('thermal_power_devfreq_limit'):
            self._log.warning('Event [{}] not found, plot DISABLED!'
                              .format('thermal_power_devfreq_limit'))
            return

        plot_df = self._trace.df_events('thermal_power_devfreq_limit')

        # Might have more than one device selected by 'type', but that's
        # the best we can do
        filters = None if devices is None else {'type' : devices}
        _ax = self._plot_generic(plot_df, 'type', filters=filters, columns=['cdev_state'],
                          drawstyle='steps-post', ax=ax
        )

        if ax is None:
            ax = _ax

        # Cdev status is an integer series
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y')

        if devices is None:
            suffix = ''
        else:
            suffix = '_' + '_'.join(map(str, devices))

        # Save generated plots into datadir
        figname = os.path.join(
            self._trace.plots_dir,
            '{}thermal_devfreq_cdev_state{}.png'.format(
                self._trace.plots_dir, self._trace.plots_prefix, suffix
            )
        )
        pl.savefig(figname, bbox_inches='tight')

###############################################################################
# Utility Methods
###############################################################################

    def _matching_masks(self, cpus):
        df = self._trace.df_events('thermal_power_cpu_limit')

        global_mask = list_to_mask(cpus)
        cpumasks = df['cpus'].unique().tolist()
        return [m for m in cpumasks if m & global_mask]

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
