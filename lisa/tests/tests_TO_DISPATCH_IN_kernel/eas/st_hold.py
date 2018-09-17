#    Copyright 2018 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import division

import os
import logging
import unittest

from env import TestEnv
from test import LisaTest
from trappy import FTrace
from wlgen import Periodic, RTA
from bart.common.Utils import area_under_curve

BOOSTED_CGROUP_NAME = '/boosted'
UNBOOSTED_CGROUP_NAME = '/unboosted'
FREQ_MARGIN = 20000
TEST_CPU = 0

class _SchedtuneHoldTest(LisaTest):
    """
    Base class for all Schedtune-Hold feature tests. Contains
    setup code and useful test methods.
    When a derived class is instantiated, this sequence of events
    happens:
      :meth:`setUpClass` called
          -> connect target object
          -> call derived_class:child_check_config() if implemented
          -> call derived_class:populate_params() (must be implemented)
          -> run workload
          -> build Trace object in cls.trace_obj
    Then the actual test function is executed on the collected trace.

    child_check_config is used for the two tests present where only one
    can be run - which one is OK depends upon the status of the
    SCHEDTUNE_BOOST_HOLD_ALL sched_feature. This function should raise a
    :class:`unittest.SkipTest` exception if it wants to prevent a derived
    class test from running without causing a failure.
    """
    boosted_cgroup_name = BOOSTED_CGROUP_NAME
    unboosted_cgroup_name = UNBOOSTED_CGROUP_NAME
    frequency_margin = FREQ_MARGIN
    test_cpu = TEST_CPU

    test_conf = { "modules": ["bl", "cpufreq", "cgroups" ],
                  "tools": ["rt-app"],
                  "ftrace" : {
                      "events" : ["cpu_frequency", "sched_switch", "sched_boost_cpu"],
                      "buffsize" : 10240
                  },
                  "flags" : [ "ftrace" ]
                }

    @classmethod
    def setUpClass(cls):
        cls._log = logging.getLogger('StHoldTest')
        cls.params = {}
        # connect the target
        cls.env = TestEnv(test_conf=cls._getTestConf())
        cls.trace_file = os.path.join(cls.env.res_dir, '{}_trace.dat'.format(cls.task_name))
        cls.target = cls.env.target
        # check that the SCHEDTUNE_HOLD_ALL feature exists in the target
        cls.check_hold_available()
        # check any per-class requirements
        cls.child_check_config()
        cls.rta_name = 'rt-test-{}'.format(cls.task_name.replace('/',''))
        # ask derived class to populate workload config etc.
        cls.populate_params()
        # ensure required cgroups exist
        cls.setup_cgroups()
        # run workload
        with cls.env.freeze_userspace():
            cls.run_workload()
        # collect and parse the trace
        cls.ftrace_obj = FTrace(cls.trace_file)

    @classmethod
    def child_check_config(cls):
        pass

    @classmethod
    def run_workload(cls):
        cls.env.ftrace.start()
        cls.rtapp.run(out_dir=cls.env.res_dir, cgroup=cls.cgroup_name, as_root=True)
        cls.env.ftrace.stop()
        cls.env.ftrace.get_trace(cls.trace_file)
        cls.env.platform_dump(cls.env.res_dir)

    @classmethod
    def setup_cpuset_cgroup(cls, cgroup, cpus, mems):
        cpu_g = cls.cpuset_controller.cgroup(cgroup)
        cpu_g.set(cpus=cpus, mems=mems)

    @classmethod
    def setup_schedtune_cgroup(cls, cgroup, boost, prefer_idle):
        st_g = cls.schedtune_controller.cgroup(cgroup)
        st_g.set(boost=boost, prefer_idle=prefer_idle)

    @classmethod
    def setup_cgroups(cls):
        """
        When we set up the cgroups in a unified hierarchy, we
        need to ensure that the cpu controller is minimally
        configured before we can run tasks inside a group.
        Copy the root group's mems and cpus parameters into the
        groups we create just in case.
        """
        cls.cpuset_controller = cls.target.cgroups.controller('cpuset')
        root_g = cls.cpuset_controller.cgroup('/')
        params = root_g.get()
        cls.schedtune_controller = cls.target.cgroups.controller('schedtune')
        cls.setup_schedtune_cgroup(cls.boosted_cgroup_name, 100, 0)
        cls.setup_cpuset_cgroup(cls.boosted_cgroup_name,
                                params['cpus'], params['mems'])
        cls.setup_schedtune_cgroup(cls.unboosted_cgroup_name, 0, 0)
        cls.setup_cpuset_cgroup(cls.unboosted_cgroup_name,
                                params['cpus'], params['mems'])

    @classmethod
    def get_task_times(cls):
        """
        figure out timestamps where task is running
        defined here as:
          start_time = first time we switch to the correctly named task
          end_time   = last time we switch away from the correctly named task
        """
        df = cls.ftrace_obj.sched_switch.data_frame
        cls.task_start_time = df[df.next_comm == cls.task_name].iloc[0].name
        cls.task_end_time = df[df.prev_comm == cls.task_name].iloc[-1].name

    @classmethod
    def avg_cpu_freq(cls):
        """
        To calculate average frequency, first obtain the area under the
        series of frequency transitions for the period of interest. We do
        this by recomputing the index on the frequency series derived from
        the cpu_frequency trace objects and then using area_under_curve from
        bart. Once we have the area, we can divide it by the time span to
        obtain average frequency.
        """
        df = cls.ftrace_obj.cpu_frequency.data_frame
        cpu_df = df[df.cpu == cls.test_cpu]
        freq_s = cpu_df.frequency
        old_index = list(freq_s[cls.task_start_time:cls.task_end_time].index)
        # the new index should run from the start to the end time, including
        # any events which occurred in between
        new_index = [ cls.task_start_time ] + old_index + [ cls.task_end_time ]
        windowed_freq_s = freq_s.reindex(index=new_index, method='pad')
        # avg_freq is area / time for the window of interest
        area = area_under_curve(windowed_freq_s, method='rect')
        return int((area + 0.5) / (cls.task_end_time - cls.task_start_time))

    @classmethod
    def check_hold_available(cls):
        path = '/sys/kernel/debug/sched_features'
        sf = cls.target.read_value(path)
        if 'SCHEDTUNE_BOOST_HOLD_ALL' not in sf:
            # The feature is not compiled in, we cannot test this.
            raise unittest.SkipTest('SCHEDTUNE_BOOST_HOLD sched feature not present')

    @classmethod
    def check_hold_config(cls, want_default=True):
        msg = ('SCHEDTUNE_BOOST_HOLD_ALL present but {}set '
               'for all tasks. Cannot run this test in this '
               'configuration. Skip')
        path = '/sys/kernel/debug/sched_features'
        sf = cls.target.read_value(path)
        got_default = ('NO_SCHEDTUNE_BOOST_HOLD_ALL' in sf)
        if want_default != got_default:
            raise unittest.SkipTest(msg.format('not ' if got_default else ''))

    @classmethod
    def _test_cpu_frequency(cls, lower_freq_tgt = False):
        cls.get_task_times()
        # check that the max-avg frequency for a given cpu and
        # time is either less than or greater than the stored margin
        max_freq = cls.target.cpufreq.get_max_frequency(cls.test_cpu)
        avg_freq = cls.avg_cpu_freq()
        cls._log.info("Computed avg_freq {}, max_freq {} "
                       "for timestamps {} to {}".format(avg_freq, max_freq,
                       cls.task_start_time, cls.task_end_time))
        if lower_freq_tgt:
            if not max_freq - avg_freq < cls.frequency_margin:
                raise AssertionError(
                    "During test run, avg_freq {} was not less than"
                    " {} lower than max_freq {} on test CPU {}\n".format(
                        avg_freq, cls.frequency_margin, max_freq, cls.test_cpu))
        else:
            if not max_freq - avg_freq > cls.frequency_margin:
                raise AssertionError(
                    "During test run, avg_freq {} was not more than"
                    " {} lower than max_freq {} on test CPU {}\n".format(
                        avg_freq, cls.frequency_margin, max_freq, cls.test_cpu))

    @classmethod
    def test_cpu_frequency(cls):
        return cls._test_cpu_frequency()


class RTBoostedTaskTest(_SchedtuneHoldTest):
    """
    Schedtune Boost Hold keeps the boost level active for at least
    50ms since the last enqueue, for qualifying tasks. Eligibility is
    controlled by a sched_feature. In the default configurations,
    NO_SCHEDTUNE_BOOST_HOLD_ALL is the sched_feature enabled, and
    this means that only rt-tasks in boosted groups trigger the
    boost hold mechanism. If SCHEDTUNE_BOOST_HOLD_ALL is enabled
    instead, any rt or cfs task in a boosted group will trigger
    the boost holding.
    This test uses a 40ms periodic 1% rt task in a 100% boosted
    group to cause schedutil to raise the cpu frequency to max
    for the entire duration of the test.
    """
    task_name = "1pct_rt_100"
    cgroup_name = BOOSTED_CGROUP_NAME

    @classmethod
    def populate_params(cls):
        cls.rtapp = RTA(cls.target,cls.rta_name)
        cls.rtapp.conf(
            kind='profile',
            params={
                cls.task_name: Periodic(
                    period_ms=40, duty_cycle_pct=1,
                    duration_s=2, sched={ 'policy': 'FIFO', },
                    cpus=cls.test_cpu,
                ).get(),
            },
            run_dir='/tmp'
        )

    @classmethod
    def test_cpu_frequency(cls):
        cls._test_cpu_frequency(lower_freq_tgt=True)

class RTUnboostedTaskTest(_SchedtuneHoldTest):
    """
    Schedtune Boost Hold keeps the boost level active for at least
    50ms since the last enqueue, for qualifying tasks. Eligibility is
    controlled by a sched_feature. In the default configurations,
    NO_SCHEDTUNE_BOOST_HOLD_ALL is the sched_feature enabled, and
    this means that only rt-tasks in boosted groups trigger the
    boost hold mechanism. If SCHEDTUNE_BOOST_HOLD_ALL is enabled
    instead, any rt or cfs task in a boosted group will trigger
    the boost holding.
    This test uses a 40ms periodic 1% rt task in a 0% boosted
    group and verifies that schedutil does not raise the cpu
    frequency to max for the entire duration of the test.
    """
    task_name = "1pct_rt_0"
    cgroup_name = UNBOOSTED_CGROUP_NAME

    @classmethod
    def populate_params(cls):
        cls.rtapp = RTA(cls.target,cls.rta_name)
        cls.rtapp.conf(
            kind='profile',
            params={
                cls.task_name: Periodic(
                    period_ms=40, duty_cycle_pct=1,
                    duration_s=2, sched={ 'policy': 'FIFO', },
                    cpus=cls.test_cpu,
                ).get(),
            },
            run_dir='/tmp'
        )

class NormalUnboostedTaskTest(_SchedtuneHoldTest):
    """
    Schedtune Boost Hold keeps the boost level active for at least
    50ms since the last enqueue, for qualifying tasks. Eligibility is
    controlled by a sched_feature. In the default configurations,
    NO_SCHEDTUNE_BOOST_HOLD_ALL is the sched_feature enabled, and
    this means that only rt-tasks in boosted groups trigger the
    boost hold mechanism. If SCHEDTUNE_BOOST_HOLD_ALL is enabled
    instead, any rt or cfs task in a boosted group will trigger
    the boost holding.
    This test uses a 40ms periodic 1% cfs task in a 0% boosted
    group and verifies that schedutil does not raise the cpu
    frequency to max for the entire duration of the test.
    """
    task_name = "1pct_0"
    cgroup_name = UNBOOSTED_CGROUP_NAME

    @classmethod
    def populate_params(cls):
        cls.rtapp = RTA(cls.target,cls.rta_name)
        cls.rtapp.conf(
            kind='profile',
            params={
                cls.task_name: Periodic(
                    period_ms=40, duty_cycle_pct=1,
                    duration_s=2,
                    cpus=cls.test_cpu,
                ).get(),
            },
            run_dir='/tmp'
        )

class NormalBoostedTaskTest(_SchedtuneHoldTest):
    """
    Schedtune Boost Hold keeps the boost level active for at least
    50ms since the last enqueue, for qualifying tasks. Eligibility is
    controlled by a sched_feature. In the default configurations,
    NO_SCHEDTUNE_BOOST_HOLD_ALL is the sched_feature enabled, and
    this means that only rt-tasks in boosted groups trigger the
    boost hold mechanism. If SCHEDTUNE_BOOST_HOLD_ALL is enabled
    instead, any rt or cfs task in a boosted group will trigger
    the boost holding.
    This test uses a 40ms periodic 1% cfs task in a 100% boosted
    group and verifies that schedutil does not raise the cpu
    frequency to max for the entire duration of the test. This
    test is skipped if SCHEDTUNE_BOOST_HOLD_ALL is set.
    """
    task_name = "1pct_100"
    cgroup_name = BOOSTED_CGROUP_NAME

    @classmethod
    def populate_params(cls):
        cls.rtapp = RTA(cls.target,cls.rta_name)
        cls.rtapp.conf(
            kind='profile',
            params={
                cls.task_name: Periodic(
                    period_ms=40, duty_cycle_pct=1,
                    duration_s=2,
                    cpus=cls.test_cpu,
                ).get(),
            },
            run_dir='/tmp'
        )

    @classmethod
    def child_check_config(cls):
        cls.check_hold_config()

class NormalBoostedTaskTestBoostAll(_SchedtuneHoldTest):
    """
    Schedtune Boost Hold keeps the boost level active for at least
    50ms since the last enqueue, for qualifying tasks. Eligibility is
    controlled by a sched_feature. In the default configurations,
    NO_SCHEDTUNE_BOOST_HOLD_ALL is the sched_feature enabled, and
    this means that only rt-tasks in boosted groups trigger the
    boost hold mechanism. If SCHEDTUNE_BOOST_HOLD_ALL is enabled
    instead, any rt or cfs task in a boosted group will trigger
    the boost holding.
    This test uses a 40ms periodic 1% cfs task in a 100% boosted
    group and verifies that schedutil raises the cpu frequency
    to max for the entire duration of the test. This
    test is skipped if NO_SCHEDTUNE_BOOST_HOLD_ALL is set.
    """
    task_name = "1pct_100a"
    cgroup_name = BOOSTED_CGROUP_NAME

    @classmethod
    def populate_params(cls):
        cls.rtapp = RTA(cls.target,cls.rta_name)
        cls.rtapp.conf(
            kind='profile',
            params={
                cls.task_name: Periodic(
                    period_ms=40, duty_cycle_pct=1,
                    duration_s=2,
                    cpus=cls.test_cpu,
                ).get(),
            },
            run_dir='/tmp'
        )

    @classmethod
    def child_check_config(cls):
        cls.check_hold_config(want_default=False)

    @classmethod
    def test_cpu_frequency(cls):
        cls._test_cpu_frequency(lower_freq_tgt=True)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
