# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from copy import deepcopy
from unittest import SkipTest

from test import LisaTest, experiment_test
from trace import Trace
from executor import Executor
from wlgen.rta import Periodic, RTA

from devlib.utils.misc import memoized
from devlib.module.sched import SchedDomain

WORKLOAD_PERIOD_MS =  16

SD_ASYM_CPUCAPACITY = 0x0040

class _MisfitMigrationBase(LisaTest):
    """
    Base class for shared functionality of misfit migration tests
    """

    test_conf = {
        "ftrace" : {
            "events" : [
                "sched_switch",
                "sched_wakeup",
                "sched_wakeup_new",
                "cpu_idle",
            ],
            "buffsize" : 100 * 1024
        },
        "modules": ["cgroups", "cpufreq"],
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(_MisfitMigrationBase, cls).runExperiments(*args, **kwargs)

    @memoized
    @staticmethod
    def _has_asym_cpucapacity(test_env):
        # Just try to find at least one instance of that flag
        sd_info = test_env.target.sched.get_sd_info()

        for cpu, domain_node in sd_info.cpus.items():
            for domain in domain_node.domains.values():
                if domain.has_flags(SchedDomain.SD_ASYM_CPUCAPACITY):
                    return True

        return False

    @memoized
    @staticmethod
    def _classify_cpus(test_env):
        """
        Classify cpus per capacity.

        :returns: A list of list. CPUs of equal capacities are packed in the
        same list, and those lists of CPUs are ordered by capacity values.
        """
        cpus = {}
        for cpu in xrange(test_env.target.number_of_cpus):
            cap = test_env.target.sched.get_capacity(cpu)
            if cap not in cpus:
                cpus[cap] = []

            cpus[cap].append(cpu)

        capacities = sorted(cpus.keys())
        return [cpus[capacity] for capacity in capacities]

    @memoized
    @staticmethod
    def _get_max_lb_interval(test_env):
        """
        Get the value of maximum_load_balance_interval.

        The kernel computes it so:
            HZ*num_online_cpus()/10;
        (https://elixir.bootlin.com/linux/v4.15/source/kernel/sched/fair.c#L9101)

        Here we don't do any hotplugging so we consider all CPUs to be online.

        :returns: The absolute maximum load-balance interval in seconds
        """
        HZ = test_env.target.sched.get_hz()
        return ((HZ * test_env.target.number_of_cpus) / 10) * (1. / HZ)

    @classmethod
    def _get_wload(cls, test_env):
        raise NotImplementedError()

    @classmethod
    def _getExperimentsConf(cls, test_env):
        if not cls._has_asym_cpucapacity(test_env):
            raise SkipTest(
                'This test requires a target with asymetric CPU capacities. '
                'SD_ASYM_CPUCAPACITY was not found.'
            )

        conf = {
            'tag' : 'misfit',
            'flags' : ['ftrace', 'freeze_userspace'],
        }

        if 'cpufreq' in test_env.target.modules:
            available_govs = test_env.target.cpufreq.list_governors(0)
            conf['cpufreq'] = {'governor' : 'performance'}

        return {
            'wloads' : cls._get_wload(test_env),
            'confs' : [conf],
        }

class StaggeredFinishes(_MisfitMigrationBase):
    """
    Test Misfit task migration happens at idle balance (staggered test case)

    This test spawns nr_cpus 100% tasks. The tasks running on bigger-capacity
    CPUs will finish first, and it is expected of them to instantly pull the
    tasks running on smaller-capacity CPUs via idle-balance.
    """

    # How long the tasks will be pinned to their "starting" CPU. Doesn't have
    # to be long (we just have to ensure they spawn there), so arbitrary value
    pin_delay_s = 0.001

    # Somewhat arbitrary delay - long enough to ensure
    # rq->avg_idle > sysctl_sched_migration_cost
    idling_delay_s = 1

    # How long do we allow the bigs to be idle when there are tasks running on
    # the LITTLEs
    allowed_idle_time_s = 0.001

    # How much % of time do we allow the tasks to be preempted, out of the
    # total test duration
    allowed_preempt_pct = 1

    @classmethod
    def _get_wload(cls, test_env):
        cpus = range(test_env.platform['cpus_count'])

        # We're pinning stuff in the first phase, so give it ample time to
        # clean the pinned logic out of balance_interval
        free_time_s = 1.1 * cls._get_max_lb_interval(test_env)
        stagger_s = free_time_s / (10 * len(cpus))

        params = {}

        for cpu in cpus:
            params["misfit_{}".format(cpu)] = (
                Periodic(
                    duty_cycle_pct=100,
                    duration_s=cls.pin_delay_s,
                    delay_s=cls.idling_delay_s,
                    period_ms=16,
                    cpus=[cpu]
                ) + Periodic(
                    duty_cycle_pct=100,
                    # Introduce staggered task completions
                    duration_s=free_time_s + cpu * stagger_s,
                    period_ms=16,
                    cpus=cpus
                )
            ).get()

        wload = RTA(test_env.target, 'tmp',
            calibration=test_env.calibration())
        wload.conf(kind='profile', params=params,
                   run_dir=Executor.get_run_dir(test_env.target))

        return {
            'staggered' : {
                'type' : 'rt-app',
                'conf' : {
                    'class' : 'custom',
                    'json' : wload.json
                }
            }
        }

    @memoized
    def get_active_df(self, trace, cpu):
        """
        :returns: A dataframe that describes the idle status (on/off) of 'cpu'
        """
        active_df = pd.DataFrame(trace.getCPUActiveSignal(cpu), columns=['state'])
        trace.addEventsDeltas(active_df)

        return active_df

    def max_idle_time(self, trace, start, end, cpus):
        """
        :returns: The maximum idle time of 'cpus' in the [start, end] interval
        """
        idle_df = pd.DataFrame()
        max_time = 0
        max_cpu = 0

        for cpu in cpus:
            busy_df = self.get_active_df(trace, cpu)
            busy_df = Trace.squash_df(busy_df, start, end)
            busy_df = busy_df[busy_df.state == 0]

            if busy_df.empty:
                continue

            local_max = busy_df.delta.max()
            if local_max > max_time:
                max_time = local_max
                max_cpu = cpu

        return max_time, max_cpu

    @memoized
    def start_time(self, experiment):
        """
        :returns: The start time of the test workload, IOW the time at which
            all tasks are up and running on their designated CPUs.
        """
        trace = self.get_trace(experiment)
        sdf = trace.data_frame.trace_event('sched_switch')
        # Get the time where the first rt-app task spawns
        init_start = sdf[sdf.next_comm.str.contains('misfit')].index[0]

        # The tasks don't wake up at the same exact time, find the task that is
        # the last to wake up.
        last_start = 0

        sdf = sdf[init_start + self.idling_delay_s * 0.9 :]

        for cpu in range(self.te.target.number_of_cpus):
            task_name = "misfit_{}".format(cpu)
            task_start = sdf[(sdf.next_comm == task_name) & (sdf["__cpu"] == cpu)].index[0]
            last_start = max(last_start, task_start)

        return last_start

    def trim_lat_df(self, start, lat_df):
        if lat_df.empty:
            return lat_df

        lat_df = Trace.squash_df(lat_df, start, lat_df.index[-1], "t_delta")
        # squash_df only updates t_delta, remove t_start to make sure it's not used
        return lat_df.drop('t_start', 1)

    @experiment_test
    def test_preempt_time(self, experiment, tasks):
        """
        Test that tasks are not being preempted too much
        """
        trace = self.get_trace(experiment)

        cpus = range(self.te.target.number_of_cpus)
        sorted_cpus = self._classify_cpus(self.te)

        sdf = trace.data_frame.trace_event('sched_switch')
        latency_dfs = {
            i : trace.data_frame.latency_df('misfit_{}'.format(i))
            for i in cpus
        }

        start_time = self.start_time(experiment)
        end_time = sdf[sdf.prev_comm.str.contains('misfit')].index[-1]
        test_duration = end_time - start_time

        for task_num in cpus:
            task_name = "misfit_{}".format(task_num)
            lat_df = latency_dfs[task_num]

            # The sched_switch dataframe where the misfit task
            # is replaced by another misfit task
            preempt_sdf = sdf[
                (sdf.prev_comm == task_name) &
                (sdf.next_comm.str.startswith("misfit_"))
            ]

            lat_df = self.trim_lat_df(
                start_time,
                lat_df[
                    (lat_df.index.isin(preempt_sdf.index)) &
                    # Ensure this is a preemption and not just the task ending
                    (lat_df.curr_state == "S")
                ]
            )

            task_name = "misfit_{}".format(task_num)
            preempt_time = lat_df.t_delta.sum()

            preempt_pct = (preempt_time / test_duration) * 100
            self._log.debug("{} was preempted {:.2f}% of the time".format(task_name, preempt_pct))

            if preempt_time > test_duration * self.allowed_preempt_pct/100.:
                err = "{} was preempted for {:.2f}% ({:.2f}s) of the test duration, " \
                      "expected < {}%".format(
                          task_name,
                          preempt_pct,
                          preempt_time,
                          self.allowed_preempt_pct
                      )
                raise AssertionError(err)

    def _test_idle_time(self, trace, latency_dfs, busy_cpus):
        """
        Test that for every event in latency_dfs, busy_cpus are
        not idle for more than self.allowed_idle_time_s

        :param trace: The trace to process
        :type trace: :class:`Trace`:

        :param latency_dfs: The latency dataframes (see :class:`analysis.LatencyAnalysis`),
            arranged in a {task_name : latency_df} shape
        :type latency_dfs: dict

        :param busy_cpus: The CPUs we want to assert are kept busy
        :type busy_cpus: list
        """
        cpus = range(self.te.target.number_of_cpus)
        sdf = trace.data_frame.trace_event('sched_switch')

        for task_name, lat_df in latency_dfs.iteritems():
            # Have a look at every task activation
            for index, row in lat_df.iterrows():
                cpu = int(row["__cpu"])
                end = index + row.t_delta
                # Ensure 'busy_cpus' are not idle for too long
                idle_time, other_cpu = self.max_idle_time(trace, index, end, busy_cpus)

                if idle_time > self.allowed_idle_time_s:
                    err = "{} was on CPU{} @{:.3f} but CPU{} was idle " \
                          "for {:.3f}s, expected < {}s".format(
                              task_name,
                              cpu,
                              index + trace.ftrace.basetime,
                              other_cpu,
                              idle_time,
                              self.allowed_idle_time_s
                          )
                    raise AssertionError(err)

    @experiment_test
    def test_migration_delay(self, experiment, tasks):
        """
        Test that big CPUs pull tasks ASAP
        """

        trace = self.get_trace(experiment)
        cpus = range(self.te.target.number_of_cpus)
        sorted_cpus = self._classify_cpus(self.te)

        littles = sorted_cpus[0]
        bigs = []
        for group in sorted_cpus[1:]:
            bigs += group

        start_time = self.start_time(experiment)

        latency_dfs = {}
        for i in cpus:
            # This test is about the first migration delay.
            # Trim the latency_df to up until the first time the task
            # runs on a big CPU. The test will fail if the task wasn't
            # migrated ASAP
            res = pd.DataFrame([])
            task_name = 'misfit_{}'.format(i)

            df = trace.data_frame.latency_df(task_name)
            df = self.trim_lat_df(start_time, df[df.curr_state == "A"])

            first_big = df[df["__cpu"].isin(bigs)]

            if not first_big.empty:
                res = df[df["__cpu"].isin(littles)][:first_big.index[0]]

            latency_dfs[task_name] = res

        self._test_idle_time(trace, latency_dfs, bigs)

    @experiment_test
    def test_throughput(self, experiment, tasks):
        """
        Test that big CPUs are kept as busy as possible
        """
        trace = self.get_trace(experiment)
        cpus = range(self.te.target.number_of_cpus)
        sorted_cpus = self._classify_cpus(self.te)

        littles = sorted_cpus[0]
        bigs = []
        for group in sorted_cpus[1:]:
            bigs += group

        start_time = self.start_time(experiment)

        latency_dfs = {}
        for i in cpus:
            # This test is all about throughput: check that every time a task
            # runs on a little it's because bigs are busy
            task_name = 'misfit_{}'.format(i)

            df = trace.data_frame.latency_df(task_name)
            latency_dfs[task_name] = self.trim_lat_df(
                start_time,
                df[
                    (df.curr_state == "A") &
                    (df["__cpu"].isin(littles))
                ])

        self._test_idle_time(trace, latency_dfs, bigs)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
