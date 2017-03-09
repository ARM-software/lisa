#!/usr/bin/python

import os
from time import sleep

# The workload class MUST be loaded before the LisaBenchmark
from android import Workload
from android import LisaBenchmark

from devlib.exception import TargetError

class UiBenchTest(LisaBenchmark):

    bm_conf = {

        # Target platform and board
        "platform"      : 'android',

        # Define devlib modules to load
        "modules"     : [
            'cpufreq',
        ],

        # FTrace events to collect for all the tests configuration which have
        # the "ftrace" flag enabled
        "ftrace"  : {
            "events" : [
                "sched_switch",
                "sched_overutilized",
                "sched_contrib_scale_f",
                "sched_load_avg_cpu",
                "sched_load_avg_task",
                "sched_tune_tasks_update",
                "sched_boost_cpu",
                "sched_boost_task",
                "sched_energy_diff",
                "cpu_frequency",
                "cpu_idle",
                "cpu_capacity",
            ],
            "buffsize" : 10 * 1024,
        },

        # Default EnergyMeter Configuration
        "emeter" : {
            "instrument" : "acme",
            "channel_map" : {
                "Device0" : 0,
            }
        },

        # Tools required by the experiments
        "tools"   : [ 'trace-cmd' ],

		# Default results folder
		"results_dir" : "AndroidUiBench",

    }

    # Android Workload to run
    bm_name = 'UiBench'

    # Default products to be collected
    bm_collect = 'ftrace energy'

    def benchmarkInit(self):
        self.setupWorkload()
        self.setupGovernor()
        if self.reboot:
            self.reboot_target()

    def benchmarkFinalize(self):
        if self.delay_after_s:
            self._log.info("Waiting %d[s] before to continue...",
                           self.delay_after_s)
            sleep(self.delay_after_s)

    def __init__(self, governor, test, duration_s, reboot=False,
                 delay_after_s=0):
        self.reboot = reboot
        self.governor = governor
        self.test = test
        self.duration_s = duration_s
        self.delay_after_s = delay_after_s
        super(UiBenchTest, self).__init__()

    def setupWorkload(self):
        # Create a results folder for each "governor/test"
        self.out_dir = os.path.join(self.te.res_dir, governor, self.test)
        try:
                os.stat(self.out_dir)
        except:
                os.makedirs(self.out_dir)
        # Setup workload parameters
        self.bm_params = {
            'test_name'  : self.test,
            'duration_s' : self.duration_s,
        }

    def setupGovernor(self):
        try:
            self.target.cpufreq.set_all_governors(self.governor);
        except TargetError:
            self._log.warning('Governor [%s] not available on target',
                             self.governor)
            raise

        # Setup schedutil parameters
        if self.governor == 'schedutil':
            rate_limit_us = 2000
            # Different schedutil versions have different tunables
            tunables = self.target.cpufreq.list_governor_tunables(0)
            if 'rate_limit_us' in tunables:
                tunables = {'rate_limit_us' : str(rate_limit_us)}
            else:
                assert ('up_rate_limit_us' in tunables and
                        'down_rate_limit_us' in tunables)
                tunables = {
                    'up_rate_limit_us' : str(rate_limit_us),
                    'down_rate_limit_us' : str(rate_limit_us)
                }

            try:
                for cpu_id in range(self.te.platform['cpus_count']):
                    self.target.cpufreq.set_governor_tunables(
                        cpu_id, 'schedutil', **tunables)
            except TargetError as e:
                self._log.warning('Failed to set schedutils parameters: {}'\
                                 .format(e))
                raise
            self._log.info('Set schedutil.rate_limit_us=%d', rate_limit_us)

        # Setup ondemand parameters
        if self.governor == 'ondemand':
            try:
                for cpu_id in range(self.te.platform['cpus_count']):
                    tunables = self.target.cpufreq.get_governor_tunables(cpu_id)
                    self.target.cpufreq.set_governor_tunables(
                        cpu_id, 'ondemand',
                        **{'sampling_rate' : tunables['sampling_rate_min']})
            except TargetError as e:
                self._log.warning('Failed to set ondemand parameters: {}'\
                                 .format(e))
                raise
            self._log.info('Set ondemand.sampling_rate to minimum supported')

        # Report configured governor
        governors = self.target.cpufreq.get_all_governors()
        self._log.info('Using governors: %s', governors)

# Run the benchmark in each of the supported governors

duration_s = 20

governors = [
    'performance',
    'ondemand',
    'interactive',
    'sched',
    'schedutil',
    'powersave',
]

tests = [
# General
    'DialogListActivity',
    'FullscreenOverdrawActivity',
    'GlTextureViewActivity',
    'InvalidateActivity',
    'TrivialAnimationActivity',
    'TrivialListActivity',
    'TrivialRecyclerViewActivity',
# Inflation
    'InflatingListActivity',
# Rendering
    'BitmapUploadActivity',
    'ShadowGridActivity',
# Text
    'EditTextTypeActivity',
    'TextCacheHighHitrateActivity',
    'TextCacheLowHitrateActivity',
# Transitions
    'ActivityTransition',
    'ActivityTransitionDetails',
]

# Reboot device only the first time
do_reboot = True
tests_remaining = len(governors) * len(tests)
tests_completed = 0
for governor in governors:
    for test in tests:
        tests_remaining -= 1
        delay_after_s = 30 if tests_remaining else 0
        try:
            UiBenchTest(governor, test, duration_s,
                          do_reboot, delay_after_s)
            tests_completed += 1
        except:
            # A test configuraion failed, continue with other tests
            pass
        do_reboot = False

# We want to collect data from at least one governor
assert(tests_completed >= 1)

# vim :set tabstop=4 shiftwidth=4 expandtab
