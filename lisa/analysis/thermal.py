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

from matplotlib.ticker import MaxNLocator

from devlib.utils.misc import list_to_mask, mask_to_list

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import memoized
from lisa.trace import requires_events, CPU
from lisa.datautils import df_refit_index, series_refit_index


class ThermalAnalysis(TraceAnalysisBase):
    """
    Support for plotting Thermal Analysis data

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    name = 'thermal'

    @requires_events("thermal_temperature")
    def df_thermal_zones_temperature(self):
        """
        Get the temperature of the thermal zones

        :returns: a :class:`pandas.DataFrame` with:

          * An ``id`` column (The thermal zone ID)
          * A ``thermal_zone`` column (The thermal zone name)
          * A ``temp`` column (The reported temperature)
        """
        df = self.trace.df_event("thermal")
        df = df[['id', 'thermal_zone', 'temp']]

        return df

    @TraceAnalysisBase.cache
    @requires_events("thermal_power_cpu_limit")
    def df_cpufreq_cooling_state(self, cpus=None):
        """
        Get cpufreq cooling device states

        :param cpus: The CPUs to consider (all by default)
        :type cpus: list(int)

        :returns: a :class:`pandas.DataFrame` with:

          * An ``cpus`` column (The CPUs affected by the cooling device)
          * A ``freq`` column (The frequency limit)
          * A ``cdev_state`` column (The cooling device state index)

        """
        df = self.trace.df_event("thermal_power_cpu_limit")
        df = df[['cpus', 'freq', 'cdev_state']]

        if cpus is not None:
            # Find masks that match the requested CPUs
            # This can include other CPUs
            masks = self._matching_masks(cpus)
            df = df[df.cpus.isin(masks)]

        return df

    @TraceAnalysisBase.cache
    @requires_events("thermal_power_devfreq_limit")
    def df_devfreq_cooling_state(self, devices=None):
        """
        Get devfreq cooling device states

        :param devices: The devfreq devices to consider (all by default)
        :type device: list(str)

        :returns: a :class:`pandas.DataFrame` with:

          * An ``cpus`` column (The CPUs affected by the cooling device)
          * A ``freq`` column (The frequency limit)
          * A ``cdev_state`` column (The cooling device state index)
        """
        df = self.trace.df_event("devfreq_out_power")
        df = df[['type', 'freq', 'cdev_state']]

        if devices is not None:
            df = df[df.type.isin(devices)]

        return df

    @property
    @memoized
    @df_thermal_zones_temperature.used_events
    def thermal_zones(self):
        """
        Get thermal zone ids that appear in the trace
        """
        df = self.df_thermal_zones_temperature()
        return df["thermal_zone"].unique().tolist()

    @property
    @memoized
    @df_cpufreq_cooling_state.used_events
    def cpufreq_cdevs(self):
        """
        Get cpufreq cooling devices that appear in the trace
        """
        df = self.df_cpufreq_cooling_state()
        res = df['cpus'].unique().tolist()
        return [mask_to_list(mask) for mask in res]

    @property
    @memoized
    @df_devfreq_cooling_state.used_events
    def devfreq_cdevs(self):
        """
        Get devfreq cooling devices that appear in the trace
        """
        df = self.df_devfreq_cooling_state()
        return df['type'].unique().tolist()

###############################################################################
# Plotting Methods
###############################################################################

    @TraceAnalysisBase.plot_method()
    @df_thermal_zones_temperature.used_events
    def plot_thermal_zone_temperature(self, thermal_zone_id: int, axis, local_fig):
        """
        Plot temperature of thermal zones (all by default)

        :param thermal_zone_id: ID of the zone
        :type thermal_zone_id: int
        """
        window = self.trace.window

        df = self.df_thermal_zones_temperature()
        df = df[df.id == thermal_zone_id]
        df = df_refit_index(df, window=window)

        tz_name = df.thermal_zone.unique()[0]

        series = series_refit_index(df['temp'], window=window)
        series.plot(drawstyle="steps-post", ax=axis,
                     label=f"Thermal zone \"{tz_name}\"")

        axis.legend()

        if local_fig:
            axis.grid(True)
            axis.set_title("Temperature evolution")
            axis.set_ylabel("Temperature (°C.10e3)")

    @TraceAnalysisBase.plot_method()
    @df_cpufreq_cooling_state.used_events
    def plot_cpu_cooling_states(self, cpu: CPU, axis, local_fig):
        """
        Plot the state evolution of a cpufreq cooling device

        :param cpu: The CPU. Whole clusters can be controlled as
          a single cooling device, they will be plotted as long this CPU
          belongs to the cluster.
        :type cpu: int
        """
        window = self.trace.window

        df = self.df_cpufreq_cooling_state([cpu])
        df = df_refit_index(df, window=window)
        cdev_name = f"CPUs {mask_to_list(df.cpus.unique()[0])}"

        series = series_refit_index(df['cdev_state'], window=window)
        series.plot(drawstyle="steps-post", ax=axis,
                           label=f"\"{cdev_name}\"")

        axis.legend()

        if local_fig:
            axis.grid(True)
            axis.set_title("cpufreq cooling devices status")
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis='y')

    @TraceAnalysisBase.plot_method()
    def plot_dev_freq_cooling_states(self, device: str, axis, local_fig):
        """
        Plot the state evolution of a devfreq cooling device

        :param device: The devfreq devices to consider
        :type device: str
        """
        df = self.df_devfreq_cooling_state([device])
        df = df_refit_index(df, window=self.trace.window)

        df['cdev_state'].plot(drawstyle="steps-post", ax=axis,
                           label=f"Device \"{device}\"")

        axis.legend()

        if local_fig:
            axis.grid(True)
            axis.set_title("devfreq cooling devices status")
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis='y')

###############################################################################
# Utility Methods
###############################################################################

    def _matching_masks(self, cpus):
        df = self.trace.df_event('thermal_power_cpu_limit')

        global_mask = list_to_mask(cpus)
        cpumasks = df['cpus'].unique().tolist()
        return [m for m in cpumasks if m & global_mask]

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
