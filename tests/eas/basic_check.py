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

import json
import time
import re
import pandas
import StringIO

from env import TestEnv
from test import LisaTest

TEST_CONF = {
    "modules": ["cpufreq"],
    "results_dir": "BasicCheck",
    "tools": [
        "sysbench",
        "taskset",
    ]
}

class BasicCheck_Tests(LisaTest):
    """
    Goal
    ====

    Check that the configuration of a given device is suitable
    for running EAS.

    Detailed Description
    ====================

    This test reads the kernel configuration and digs around in sysfs to
    check the following attributes are true:
      * the minimum set of required config options are enabled
      * all CPUs have access to the 'sched' CPUFreq governor
      * SchedTune CGroup Controller is present and mounted
      * runtime sysctl values are configured as we would expect
      * energy aware scheduling is present and enabled

    Expected Behaviour
    ==================

    All required config options are set, sched governor is present,
    SchedTune is mounted, energy_aware_scheduler sched feature is on

    """
    @classmethod
    def setUpClass(cls):
        cls.params = {}
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.target = cls.env.target

    def test_sched_governor_available(self):
        """
        Check that the "sched" cpufreq governor is available on all CPUs
        """
        fail_list = []
        for cpu in self.target.list_online_cpus():
            if 'sched' not in self.target.cpufreq.list_governors(cpu):
                fail_list.append(cpu)
        msg = "CPUs {} do not support sched cpufreq governor".format(fail_list)
        self.assertTrue(len(fail_list) == 0, msg=msg)

    def test_kernel_config(self):
        """
        Check that the kernel config has the basic requirements for EAS
        """
        kernel_config = self.target.config
        necessary_configs = {
            "CONFIG_CPU_FREQ_GOV_SCHED" : "sched governor present",
            "CONFIG_SCHED_TUNE" : "SchedTune present",
            "CONFIG_CGROUPS" : "CGroups enabled",
            "CONFIG_CGROUP_SCHEDTUNE" : "SchedTune CGroup Controller present",
            "CONFIG_SMP" : "SMP enabled",
            "CONFIG_SCHED_MC" : "MC-level scheduler domains",
            "CONFIG_CPU_FREQ" : "CPUFreq enabled",
            "CONFIG_CPU_IDLE" : "CPUIdle enabled",
            "CONFIG_SCHED_DEBUG" : "Scheduler Debugging enabled",
            #"CONFIG_SCHED_WALT" : "WALT Load Tracking",
        }
        fail_list = []
        message = ""
        for config in necessary_configs:
            if not kernel_config.is_enabled(config):
                fail_list.append(config)
        if len(fail_list):
            message = "Configs are missing: "
            for cfg in fail_list:
                message = message + "[{} : {}] ".format(
                    cfg, necessary_configs[cfg])
        self.assertTrue(len(fail_list) == 0, msg=message)

    def test_schedtune(self):
        """
        Check that SchedTune is present on the target
        """
        mount_output = self.target.execute('mount')
        mount_location = ''
        SCHEDTUNE_REGEXP = re.compile(r'(?P<location>/\S+).*schedtune\)')
        DIRENTRY_REGEXP = re.compile(r'd.*[0-9]{2}:[0-9]{2}.(?P<group_name>.+)$')
        for line in mount_output.splitlines():
            match = SCHEDTUNE_REGEXP.search(line)
            if match:
                mount_location = match.group('location')
        schedtune = { 'groups': {} }
        self.assertTrue(len(mount_location) > 0,
                        "SchedTune CGroup Controller not mounted")
        # populate groups, start with root group
        paths = { '' : 'root' }
        # now find directories
        dir_list = self.target.execute('ls -l {}'.format(mount_location))
        for entry in dir_list.splitlines():
            match = DIRENTRY_REGEXP.search(entry)
            if match:
                group_name = match.group('group_name')
                paths['/{}'.format(group_name)] = group_name
        # now read the values
        files = [ 'schedtune.boost' ]
        for group in paths:
            schedtune['groups'][paths[group]] = {}
            for name in files:
                schedtune['groups'][paths[group]][name] = self.target.read_int(
                    '{}{}/{}'.format(mount_location, group, name))
        self.assertTrue(schedtune['groups']['root']['schedtune.boost'] == 0,
                        "SchedTune Root Group Boost != 0")
        groups_with_nonzero_boost = 0
        for groups in schedtune['groups']:
            if groups == 'root':
                continue
            if schedtune['groups'][groups]['schedtune.boost'] != 0:
                groups_with_nonzero_boost += 1
        self.assertTrue(groups_with_nonzero_boost != 0,
                        "No SchedTune groups have a boost configured")

    def test_runtime_config(self):
        # do we want to check these values are as expected?
        #runtime_files = [
        #                 "sched_cstate_aware",
        #                 "sched_initial_task_util",
        #                 "sched_is_big_little",
        #                 "sched_sync_hint_enable",
        #                 "sched_use_walt_cpu_util",
        #                 "sched_use_walt_task_util",
        #                 ]
        #files = {}
        #for name in runtime_files:
        #    try:
        #        value = self.target.read_value('/proc/sys/kernel/{}'.format(name))
        #    except Exception:
        #        value = ''
        #    files[name] = value
        #
        # at least we can check if ENERGY_AWARE is enabled
        sched_features = ""
        try:
            sched_features = self.target.read_value('/sys/kernel/debug/sched_features')
        except Exception:
            pass
        self.assertFalse('NO_ENERGY_AWARE' in sched_features,
                         "Energy Aware Scheduling is not enabled")

    def get_freq_residencies(self):
        state_file = '/sys/devices/system/cpu/cpufreq/all_time_in_state'
        result = self.target.read_value(state_file)
        buf = StringIO.StringIO(result)
        return pandas.read_table(buf, delim_whitespace=True)

    def check_freq_accounting(self, cpu):
        seconds = 1.0
        margin = 0.2
        frequencies = self.target.cpufreq.list_frequencies(cpu)

        original_governor = self.target.cpufreq.get_governor(cpu)
        original_freq = None
        if original_governor == 'userspace':
            original_freq = self.target.cpufreq.get_frequency(cpu)

        self.target.cpufreq.set_governor(cpu, "userspace")
        # Set max freq
        self.target.cpufreq.set_frequency(cpu, frequencies[-1])
        pre_change_residencies = self.get_freq_residencies()
        self.target.cpufreq.set_frequency(cpu, frequencies[0])
        time.sleep(seconds)
        post_change_residencies = self.get_freq_residencies()
        # Restore governor
        self.target.cpufreq.set_governor(cpu, original_governor)
        if original_freq:
            self.target.cpufreq.set_frequency(cpu, original_freq)

        # Compare frequency residencies
        diff = {}
        for freq in frequencies:
            filter_df = pre_change_residencies['freq'] == freq
            pre = pre_change_residencies.loc[filter_df].iloc[0]

            filter_df = post_change_residencies['freq']==freq
            post = post_change_residencies.loc[filter_df].iloc[0]

            diff[freq] = ((post['cpu{}'.format(cpu)] - pre['cpu{}'.format(cpu)])
                          / 100.0)
        t = diff[frequencies[0]]
        if t < (seconds-margin) or t > (seconds+margin):
            return False
        t = diff[frequencies[-1]]
        if t > margin:
            return False
        return True

    def test_freq_accounting(self):
        """
        Test that cpufreq all_time_in_state accounting gets updated
        """
        failed_cpus = []
        for cpulist in self.env.topology.get_level('cpu'):
            cpu = cpulist[0]
            if not self.check_freq_accounting(cpu):
                failed_cpus.append(cpu)

        msg="Frequency Accounting did not change with CPU Freq on CPUs: {}"\
            .format(failed_cpus)
        self.assertFalse(len(failed_cpus), msg=msg)

    def run_sysbench_work(self, cpu, duration):
        """
        Run benchmark using 1 thread on a given CPU.

        :param cpu: cpu to run the benchmark on
        :type cpu: str
        :param duration: length of time, in seconds to run the benchmark

        :returns: float - performance score
        """
        args = "--test=cpu --num-threads=1 --max-time={} run".format(duration)

        bench_out = self.target.invoke(self.sysbench, args=args, on_cpus=[cpu])

        match = re.search(r'(total number of events:\s*)([\d.]*)', bench_out)
        return float(match.group(2))

    def check_work_throughput(self, cpu):
        seconds = 1.0
        margin = 0.2
        frequencies = self.target.cpufreq.list_frequencies(cpu)
        original_governor = self.target.cpufreq.get_governor(cpu)
        original_freq = None
        if original_governor == 'userspace':
            original_freq = self.target.cpufreq.get_frequency(cpu)
        # set userspace governor
        self.target.cpufreq.set_governor(cpu, "userspace")
        # do each freq in turn
        result = {}
        for freq in frequencies:
            self.target.cpufreq.set_frequency(cpu, freq)
            result[freq] = self.run_sysbench_work(cpu, seconds)
        # restore governor
        self.target.cpufreq.set_governor(cpu, original_governor)
        if original_freq:
            self.target.cpufreq.set_frequency(cpu, original_freq)
        # compare work throughput
        return result[frequencies[0]] < result[frequencies[-1]]

    def test_work_throughput(self):
        host_path = "tools/{}/sysbench".format(self.target.abi)
        self.sysbench = self.target.install_if_needed(host_path)

        failed_cpus = []
        for cpulist in self.env.topology.get_level('cpu'):
            cpu = cpulist[0]
            if not self.check_work_throughput(cpu):
                failed_cpus.append(cpu)
        msg="Work done did not scale with CPU Freq on CPUs: {}"\
            .format(failed_cpus)
        self.assertFalse(len(failed_cpus), msg=msg)
