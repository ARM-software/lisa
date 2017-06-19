#!/usr/bin/env python

import os
from time import sleep

# The workload class MUST be loaded before the LisaBenchmark
from android import Workload
from android import LisaBenchmark

from devlib.exception import TargetError

class GMapsTest(LisaBenchmark):

    # Android Workload to run
    bm_name = 'GMaps'

    # Default products to be collected
    bm_collect = 'ftrace energy'

    def benchmarkInit(self):
        self.setupWorkload()
        self.setupGovernor()

    def __init__(self, governor, location_search, swipe_count):
        self.governor = governor
        self.location_search = location_search
        self.swipe_count = swipe_count
        super(GMapsTest, self).__init__()

    def setupWorkload(self):
        # Create a results folder for each "governor/test"
        self.out_dir = os.path.join(self.te.res_dir, governor,
                       self.location_search.replace(' ', '_'))
        try:
                os.stat(self.out_dir)
        except:
                os.makedirs(self.out_dir)
        # Setup workload parameters
        self.bm_params = {
            'location_search'  : self.location_search,
            'swipe_count'      : self.swipe_count,
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

swipe_count = 5;

governors = [
    'performance',
    'ondemand',
    'interactive',
    'sched',
    'schedutil',
    'powersave',
]

locations = [
    "ARM Cambridge 110 Fulbourn",
    "London British Museum"
]

tests_remaining = len(governors) * len(locations)
tests_completed = 0
for governor in governors:
    for location in locations:
        tests_remaining -= 1
        try:
            GMapsTest(governor, location, swipe_count)
            tests_completed += 1
        except:
            # A test configuration failed, continue with other tests
            pass

# We want to collect data from at least one governor
assert(tests_completed >= 1)

# vim :set tabstop=4 shiftwidth=4 expandtab
