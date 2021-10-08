# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
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

from math import isnan

import pandas as pd
import holoviews as hv

from itertools import chain

from lisa.wlgen.rta import RTAPhase, PeriodicWload, DutyCycleSweepPhase
from lisa.analysis.rta import RTAEventsAnalysis
from lisa.analysis.tasks import TasksAnalysis
from lisa.tests.base import ResultBundle, TestBundle, RTATestBundle
from lisa.utils import ArtifactPath, memoized
from lisa.datautils import series_integrate, df_deduplicate
from lisa.energy_model import EnergyModel, EnergyModelCapacityError
from lisa.target import Target
from lisa.pelt import PELT_SCALE, pelt_swing
from lisa.datautils import df_refit_index
from lisa.notebook import plot_signal


class EASBehaviour(RTATestBundle, TestBundle):
    """
    Abstract class for EAS behavioural testing.

    :param nrg_model: The energy model of the platform the synthetic workload
      was run on
    :type nrg_model: EnergyModel

    This class provides :meth:`test_task_placement` to validate the basic
    behaviour of EAS. The implementations of this class have been developed to
    verify patches supporting Arm's big.LITTLE in the Linux scheduler. You can
    see these test results being published
    `here <https://developer.arm.com/open-source/energy-aware-scheduling/eas-mainline-development>`_.
    """

    @property
    def nrg_model(self):
        return self.plat_info['nrg-model']

    @classmethod
    def get_big_duty_cycle(cls, plat_info):
        """
        Returns a duty cycle for :class:`lisa.wlgen.rta.PeriodicWload` that
        will guarantee placement on a big CPU.

        The duty cycle will be chosen so that the task will not fit on the
        second to biggest CPUs in the system, thereby forcing up-migration
        while minimizing the thermal impact.
        """
        capa_classes = plat_info['capacity-classes']
        max_class = len(capa_classes) - 1

        def get_class_util(class_, pct):
            cpus = capa_classes[class_]
            return cls.unscaled_utilization(plat_info, cpus[0], pct)

        def get_pelt_swing(pct):
            return pelt_swing(
                period=cls.TASK_PERIOD,
                duty_cycle=util / 100,
                kind='above',
            ) / PELT_SCALE * 100

        class_ = -2

        # Resolve to an positive index
        class_ %= (max_class + 1)

        capacity_margin_pct = 20
        util = get_class_util(class_, 100)

        if class_ < max_class:
            higher_class_capa = get_class_util(class_ + 1, (100 - capacity_margin_pct))
            # If the CPU class and util we picked is too close to the capacity
            # of the next bigger CPU, we need to take a smaller util
            if (util + get_pelt_swing(util)) >= higher_class_capa:
                # Take a 5% margin for rounding errors
                util = 0.95 * higher_class_capa
                return (
                    util -
                    # And take extra margin to take into account the swing of
                    # the PELT value around the average
                    get_pelt_swing(util)
                )
            else:
                return util
        else:
            return util

    @classmethod
    def get_little_cpu(cls, plat_info):
        """
        Return a little CPU ID.
        """
        littles = plat_info["capacity-classes"][0]
        return littles[0]

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)
        kconfig = target.plat_info['kernel']['config']
        for option in (
            'CONFIG_ENERGY_MODEL',
            'CONFIG_CPU_FREQ_GOV_SCHEDUTIL',
        ):
            if not kconfig.get(option):
                ResultBundle.raise_skip(f"The target's kernel needs {option}=y kconfig enabled")

        for domain in target.plat_info['freq-domains']:
            if "schedutil" not in target.cpufreq.list_governors(domain[0]):
                ResultBundle.raise_skip(
                    f"Can't set schedutil governor for domain {domain}")

        if 'nrg-model' not in target.plat_info:
            ResultBundle.raise_skip("Energy model not available")

    @classmethod
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath = None, collector=None) -> 'EASBehaviour':
        """
        :meta public:

        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in
        :meth:`lisa.tests.base.RTATestBundle.get_rtapp_profile`
        """
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info)

        # EAS doesn't make a lot of sense without schedutil,
        # so make sure this is what's being used
        with target.disable_idle_states():
            with target.cpufreq.use_governor("schedutil"):
                cls.run_rtapp(target, res_dir, rtapp_profile, collector=collector)

        return cls(res_dir, plat_info)

    @RTAEventsAnalysis.df_phases.used_events
    def _get_expected_task_utils_df(self):
        """
        Get a DataFrame with the *expected* utilization of each task over time.

        :param nrg_model: EnergyModel used to computed the expected utilization
        :type nrg_model: EnergyModel

        :returns: A Pandas DataFrame with a column for each task, showing how
                  the utilization of that task varies over time

        .. note:: The timestamps to match the beginning and end of each rtapp
            phase are taken from the trace.
        """
        tasks_map = self.rtapp_tasks_map
        rtapp_profile = self.rtapp_profile

        def task_util(task, wlgen_task):
            task_list = tasks_map[task]
            assert len(task_list) == 1
            task = task_list[0]

            df = self.trace.ana.rta.df_phases(task, wlgen_profile=rtapp_profile)
            df = df[df['properties'].transform(lambda phase: phase['meta']['from_test'])]

            def get_phase_max_util(phase):
                wload = phase['wload']
                # Take into account the duty cycle of the phase
                avg = wload.unscaled_duty_cycle_pct(
                    plat_info=self.plat_info,
                ) * PELT_SCALE / 100
                # Also take into account the period and the swing of PELT
                # around its "average"
                swing = pelt_swing(
                    period=wload.period,
                    duty_cycle=wload.duty_cycle_pct / 100,
                    kind='above',
                )
                return avg + swing

            phases_util = {
                phase.get('name'): get_phase_max_util(phase)
                for phase in wlgen_task.phases
                if phase['meta']['from_test']
            }

            expected_util = df['phase'].map(phases_util)
            return task, expected_util

        cols = dict(
            task_util(task, wlgen_task)
            for task, wlgen_task in rtapp_profile.items()
        )
        df = pd.DataFrame(cols)
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)

        # Ensure the index is refitted so that integrals work as expected
        df = df_refit_index(df, window=self.trace.window)
        return df

    @TasksAnalysis.df_task_activation.used_events
    def _get_task_cpu_df(self):
        """
        Get a DataFrame mapping task names to the CPU they ran on

        Use the sched_switch trace event to find which CPU each task ran
        on. Does not reflect idleness - tasks not running are shown as running
        on the last CPU they woke on.

        :returns: A Pandas DataFrame with a column for each task, showing the
                  CPU that the task was "on" at each moment in time
        """
        def task_cpu(task):
            return task.comm, self.trace.ana.tasks.df_task_activation(task=task)['cpu']

        df = pd.DataFrame(dict(
            task_cpu(task_ids[0])
            for task, task_ids in self.rtapp_task_ids_map.items()
        ))
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        df = df_deduplicate(df, consecutives=True, keep='first')

        # Ensure the index is refitted so that integrals work as expected
        df = df_refit_index(df, window=self.trace.window)
        return df

    def _sort_power_df_columns(self, df, nrg_model):
        """
        Helper method to re-order the columns of a power DataFrame

        This has no significance for code, but when examining DataFrames by hand
        they are easier to understand if the columns are in a logical order.

        :param nrg_model: EnergyModel used to get the CPU from
        :type nrg_model: EnergyModel
        """
        node_cpus = [node.cpus for node in nrg_model.root.iter_nodes()]
        return pd.DataFrame(df, columns=[c for c in node_cpus if c in df])

    def _plot_expected_util(self, util_df, nrg_model):
        """
        Create a plot of the expected per-CPU utilization for the experiment
        The plot is then output to the test results directory.

        :param experiment: The :class:Experiment to examine
        :param util_df: A Pandas Dataframe with a column per CPU giving their
                        (expected) utilization at each timestamp.

        :param nrg_model: EnergyModel used to get the CPU from
        :type nrg_model: EnergyModel
        """
        def plot_cpu(cpu):
            name = f'CPU{cpu} util'
            series = util_df[cpu].copy(deep=False)
            series.index.name = 'Time'
            series.name = name
            fig = plot_signal(series).options(
                'Curve',
                ylabel='Utilization',
            )

            # The "y" dimension has the name of the series that we plotted
            fig = fig.redim.range(**{name: (-10, 1034)})

            times, utils = zip(*series.items())
            fig *= hv.Overlay(
                [
                    hv.VSpan(start, end).options(
                        alpha=0.1,
                        color='grey',
                    )
                    for util, start, end in zip(
                        utils,
                        times,
                        times[1:],
                    )
                    if not util
                ]
            )
            return fig

        cpus = sorted(nrg_model.cpus)
        fig = hv.Layout(
            list(map(plot_cpu, cpus))
        ).cols(1).options(
            title='Per-CPU expected utilization',
        )

        self._save_debug_plot(fig, name='expected_placement')
        return fig

    @_get_expected_task_utils_df.used_events
    def _get_expected_power_df(self, nrg_model, capacity_margin_pct):
        """
        Estimate *optimal* power usage over time

        Examine a trace and use :meth:get_optimal_placements and
        :meth:EnergyModel.estimate_from_cpu_util to get a DataFrame showing the
        estimated power usage over time under ideal EAS behaviour.

        :meth:get_optimal_placements returns several optimal placements. They
        are usually equivalent, but can be drastically different in some cases.
        Currently only one of those placements is used (the first in the list).

        :param nrg_model: EnergyModel used compute the optimal placement
        :type nrg_model: EnergyModel

        :param capacity_margin_pct:

        :returns: A Pandas DataFrame with a column each node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) and a
                  "power" column with the sum of other columns. Shows the
                  estimated *optimal* power over time.
        """
        task_utils_df = self._get_expected_task_utils_df()

        data = []
        index = []

        def exp_power(row):
            task_utils = row.to_dict()
            try:
                expected_utils = nrg_model.get_optimal_placements(task_utils, capacity_margin_pct)[0]
            except EnergyModelCapacityError:
                ResultBundle.raise_skip(
                    'The workload will result in overutilized status for all possible task placement, making it unsuitable to test EAS on this platform'
                )
            power = nrg_model.estimate_from_cpu_util(expected_utils)
            columns = list(power.keys())

            # Assemble a dataframe to plot the expected utilization
            data.append(expected_utils)
            index.append(row.name)

            return pd.Series([power[c] for c in columns], index=columns)

        res_df = self._sort_power_df_columns(
            task_utils_df.apply(exp_power, axis=1), nrg_model)

        self._plot_expected_util(pd.DataFrame(data, index=index), nrg_model)

        return res_df

    @_get_task_cpu_df.used_events
    @_get_expected_task_utils_df.used_events
    def _get_estimated_power_df(self, nrg_model):
        """
        Considering only the task placement, estimate power usage over time

        Examine a trace and use :meth:EnergyModel.estimate_from_cpu_util to get
        a DataFrame showing the estimated power usage over time. This assumes
        perfect cpuidle and cpufreq behaviour. Only the CPU on which the tasks
        are running is extracted from the trace, all other signals are guessed.

        :param nrg_model: EnergyModel used compute the optimal placement and
                          CPUs
        :type nrg_model: EnergyModel

        :returns: A Pandas DataFrame with a column node in the energy model
                  (keyed with a tuple of the CPUs contained by that node) Shows
                  the estimated power over time.
        """
        task_cpu_df = self._get_task_cpu_df()
        task_utils_df = self._get_expected_task_utils_df()
        tasks = self.rtapp_tasks

        # Create a combined DataFrame with the utilization of a task and the CPU
        # it was running on at each moment. Looks like:
        #                       utils                  cpus
        #          task_wmig0 task_wmig1 task_wmig0 task_wmig1
        # 2.375056      102.4      102.4        NaN        NaN
        # 2.375105      102.4      102.4        2.0        NaN

        df = pd.concat([task_utils_df, task_cpu_df],
                       axis=1, keys=['utils', 'cpus'])
        df = df.sort_index().fillna(method='ffill').dropna()

        # Now make a DataFrame with the estimated power at each moment.
        def est_power(row):
            cpu_utils = [0 for cpu in nrg_model.cpus]
            for task in tasks:
                cpu = row['cpus'][task]
                util = row['utils'][task]
                if not isnan(cpu):
                    cpu_utils[int(cpu)] += util
            power = nrg_model.estimate_from_cpu_util(cpu_utils)
            columns = list(power.keys())
            return pd.Series([power[c] for c in columns], index=columns)

        return self._sort_power_df_columns(df.apply(est_power, axis=1), nrg_model)

    @_get_expected_power_df.used_events
    @_get_estimated_power_df.used_events
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    # Memoize so that the result is shared with _check_valid_placement()
    @memoized
    def test_task_placement(self, energy_est_threshold_pct=5,
            nrg_model: EnergyModel = None, capacity_margin_pct=20) -> ResultBundle:
        """
        Test that task placement was energy-efficient

        :param nrg_model: Allow using an alternate EnergyModel instead of
            ``nrg_model```
        :type nrg_model: EnergyModel

        :param energy_est_threshold_pct: Allowed margin for estimated vs
            optimal task placement energy cost
        :type energy_est_threshold_pct: int

        Compute optimal energy consumption (energy-optimal task placement)
        and compare to energy consumption estimated from the trace.
        Check that the estimated energy does not exceed the optimal energy by
        more than ``energy_est_threshold_pct``` percents.
        """
        nrg_model = nrg_model or self.nrg_model

        exp_power = self._get_expected_power_df(nrg_model, capacity_margin_pct)
        est_power = self._get_estimated_power_df(nrg_model)

        exp_energy = series_integrate(exp_power.sum(axis=1), method='rect')
        est_energy = series_integrate(est_power.sum(axis=1), method='rect')

        msg = f'Estimated {est_energy} bogo-Joules to run workload, expected {exp_energy}'
        threshold = exp_energy * (1 + (energy_est_threshold_pct / 100))

        passed = est_energy < threshold
        res = ResultBundle.from_bool(passed)
        res.add_metric("estimated energy", est_energy, 'bogo-joules')
        res.add_metric("energy threshold", threshold, 'bogo-joules')

        return res

    def _check_valid_placement(self):
        """
        Check that a valid placement can be found for the tasks.

        If no placement can be found, :meth:`test_task_placement` will raise
        an :class:`ResultBundle`.
        """
        self.test_task_placement()

    @RTAEventsAnalysis.df_rtapp_stats.used_events
    def test_slack(self, negative_slack_allowed_pct=15) -> ResultBundle:
        """
        Assert that the RTApp workload was given enough performance

        :param negative_slack_allowed_pct: Allowed percentage of RT-app task
            activations with negative slack.
        :type negative_slack_allowed_pct: int

        Use :class:`lisa.analysis.rta.RTAEventsAnalysis` to find instances
        where the RT-App workload wasn't able to complete its activations (i.e.
        its reported "slack" was negative). Assert that this happened less than
        ``negative_slack_allowed_pct`` percent of the time.
        """
        self._check_valid_placement()

        passed = True
        bad_activations = {}
        test_tasks = list(chain.from_iterable(self.rtapp_tasks_map.values()))
        for task in test_tasks:
            slack = self.trace.ana.rta.df_rtapp_stats(task)["slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100 / len(slack)
            if bad_activations_pct > negative_slack_allowed_pct:
                passed = False

            bad_activations[task] = bad_activations_pct

        res = ResultBundle.from_bool(passed)

        for task, bad_activations_pct in bad_activations.items():
            res.add_metric(
                f"{task} delayed activations",
                bad_activations_pct, '%'
            )
        return res


class OneSmallTask(EASBehaviour):
    """
    A single 'small' task
    """

    task_name = "small"

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        return {
            cls.task_name: RTAPhase(
                prop_wload=PeriodicWload(
                    duty_cycle_pct=50,
                    scale_for_cpu=cls.get_little_cpu(plat_info),
                    duration=1,
                    period=cls.TASK_PERIOD,
                )
            )
        }


class ThreeSmallTasks(EASBehaviour):
    """
    Three 'small' tasks
    """
    task_prefix = "small"

    @EASBehaviour.test_task_placement.used_events
    def test_task_placement(self, energy_est_threshold_pct=20, nrg_model: EnergyModel = None,
                            noise_threshold_pct=1, noise_threshold_ms=None,
                            capacity_margin_pct=20) -> ResultBundle:
        """
        Same as :meth:`EASBehaviour.test_task_placement` but with a higher
        default threshold

        The energy estimation for this test is probably not very accurate and this
        isn't a very realistic workload. It doesn't really matter if we pick an
        "ideal" task placement for this workload, we just want to avoid using big
        CPUs in a big.LITTLE system. So use a larger energy threshold that
        hopefully prevents too much use of big CPUs but otherwise is flexible in
        allocation of LITTLEs.
        """
        return super().test_task_placement(
            energy_est_threshold_pct, nrg_model,
            noise_threshold_pct=noise_threshold_pct,
            noise_threshold_ms=noise_threshold_ms,
            capacity_margin_pct=capacity_margin_pct)

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        return {
            f"{cls.task_prefix}_{i}": RTAPhase(
                prop_wload=PeriodicWload(
                    duty_cycle_pct=50,
                    scale_for_cpu=cls.get_little_cpu(plat_info),
                    duration=1,
                    period=cls.TASK_PERIOD,
                )
            )
            for i in range(3)
        }


class TwoBigTasks(EASBehaviour):
    """
    Two 'big' tasks
    """

    task_prefix = "big"

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        duty = cls.get_big_duty_cycle(plat_info)
        return {
            f"{cls.task_prefix}_{i}": RTAPhase(
                prop_wload=PeriodicWload(
                    duty_cycle_pct=duty,
                    duration=1,
                    period=cls.TASK_PERIOD,
                )
            )
            for i in range(2)
        }


class TwoBigThreeSmall(EASBehaviour):
    """
    A mix of 'big' and 'small' tasks
    """

    small_prefix = "small"
    big_prefix = "big"

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        little = cls.get_little_cpu(plat_info)
        big_duty = cls.get_big_duty_cycle(plat_info)

        return {
            **{
                f"{cls.small_prefix}_{i}": RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=50,
                        scale_for_cpu=little,
                        duration=1,
                        period=cls.TASK_PERIOD
                    )
                )
                for i in range(3)
            },
            **{
                f"{cls.big_prefix}_{i}": RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=big_duty,
                        duration=1,
                        period=cls.TASK_PERIOD
                    )
                )
                for i in range(2)
            }
        }


class EnergyModelWakeMigration(EASBehaviour):
    """
    One task per big CPU, alternating between two phases:

    * Low utilization phase (should run on a LITTLE CPU)
    * High utilization phase (should run on a big CPU)
    """
    task_prefix = "emwm"

    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)
        if len(target.plat_info["capacity-classes"]) < 2:
           ResultBundle.raise_skip(
           'Cannot test migration on single capacity group')

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        little = cls.get_little_cpu(plat_info)
        end_pct = cls.get_big_duty_cycle(plat_info)
        bigs = plat_info["capacity-classes"][-1]

        return {
            f"{cls.task_prefix}_{i}": 2 * (
                RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=20,
                        scale_for_cpu=little,
                        duration=2,
                        period=cls.TASK_PERIOD,
                    )
                ) +
                RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=end_pct,
                        duration=2,
                        period=cls.TASK_PERIOD,
                    )
                )
            )
            for i in range(len(bigs))
        }


class RampUp(EASBehaviour):
    """
    A single task whose utilization slowly ramps up
    """
    task_name = "up"

    @EASBehaviour.test_task_placement.used_events
    def test_task_placement(self, energy_est_threshold_pct=15, nrg_model: EnergyModel = None,
                            noise_threshold_pct=1, noise_threshold_ms=None,
                            capacity_margin_pct=20) -> ResultBundle:
        """
        Same as :meth:`EASBehaviour.test_task_placement` but with a higher
        default threshold.

        The main purpose of this test is to ensure that as it grows in load, a
        task is migrated from LITTLE to big CPUs on a big.LITTLE system.
        This migration naturally happens some time _after_ it could possibly be
        done, since there must be some hysteresis to avoid a performance cost.
        Therefore allow a larger energy usage threshold
        """
        return super().test_task_placement(
            energy_est_threshold_pct, nrg_model,
            noise_threshold_pct=noise_threshold_pct,
            noise_threshold_ms=noise_threshold_ms,
            capacity_margin_pct=capacity_margin_pct)

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        little = cls.get_little_cpu(plat_info)
        start_pct = cls.unscaled_utilization(plat_info, little, 10)
        end_pct = cls.get_big_duty_cycle(plat_info)

        return {
            cls.task_name: DutyCycleSweepPhase(
                start=start_pct,
                stop=end_pct,
                step=5,
                duration=0.5,
                duration_of='step',
                period=cls.TASK_PERIOD,
            )
        }


class RampDown(EASBehaviour):
    """
    A single task whose utilization slowly ramps down
    """
    task_name = "down"

    @EASBehaviour.test_task_placement.used_events
    def test_task_placement(self, energy_est_threshold_pct=18, nrg_model: EnergyModel = None,
                            noise_threshold_pct=1, noise_threshold_ms=None,
                            capacity_margin_pct=20) -> ResultBundle:
        """
        Same as :meth:`EASBehaviour.test_task_placement` but with a higher
        default threshold

        The main purpose of this test is to ensure that as it reduces in load, a
        task is migrated from big to LITTLE CPUs on a big.LITTLE system.
        This migration naturally happens some time _after_ it could possibly be
        done, since there must be some hysteresis to avoid a performance cost.
        Therefore allow a larger energy usage threshold

        The number below has been found by trial and error on the platform
        generally used for testing EAS (at the time of writing: Juno r0, Juno r2,
        Hikey960 and TC2). It would be better to estimate the amount of energy
        'wasted' in the hysteresis (the overutilized band) and compute a threshold
        based on that. But implementing this isn't easy because it's very platform
        dependent, so until we have a way to do that easily in test classes, let's
        stick with the arbitrary threshold.
        """
        return super().test_task_placement(
            energy_est_threshold_pct, nrg_model,
            noise_threshold_pct=noise_threshold_pct,
            noise_threshold_ms=noise_threshold_ms,
            capacity_margin_pct=capacity_margin_pct)

    @classmethod
    def _get_rtapp_profile(cls, plat_info):
        little = cls.get_little_cpu(plat_info)
        start_pct = cls.get_big_duty_cycle(plat_info)
        end_pct = cls.unscaled_utilization(plat_info, little, 10)

        return {
            cls.task_name: DutyCycleSweepPhase(
                start=start_pct,
                stop=end_pct,
                step=5,
                duration=0.5,
                duration_of='step',
                period=cls.TASK_PERIOD,
            )
        }

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
