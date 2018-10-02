# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

import argparse
import logging
import os
import select

from subprocess import Popen, PIPE
from time import sleep

from lisa.conf import setup_logging
from lisa.android import System, Workload
from lisa.env import TestEnv
from lisa.utils import memoized

from devlib.utils.android import fastboot_command

class LisaBenchmark(object):
    """
    A base class for LISA custom benchmarks execution

    This class is intended to be subclassed in order to create a custom
    benckmark execution for LISA.
    It sets up the TestEnv and and provides convenience methods for
    test environment setup, execution and post-processing.

    Subclasses should provide a bm_conf to setup the TestEnv and
    a set of optional callback methods to configuere a test environment
    and process collected data.

    Example users of this class can be found under LISA's tests/benchmarks
    directory.
    """

    bm_conf = {

        # Target platform and board
        "platform"      : 'android',

        # Define devlib modules to load
        "modules"     : [
            'cpufreq',
            'cpuidle',
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

    }
    """Override this with a dictionary or JSON path to configure the TestEnv"""

    bm_name = None
    """Override this with the name of the LISA's benchmark to run"""

    bm_params = None
    """Override this with the set of parameters for the LISA's benchmark to run"""

    bm_collect = None
    """Override this with the set of data to collect during test exeution"""

    bm_reboot = False
    """Override this with True if a boot image was passed as command line parameter"""

    bm_iterations = 1
    """Override this with the desired number of iterations of the test"""

    bm_iterations_pause = 30
    """
    Override this with the desired amount of time (in seconds) to pause
    for before each iteration
    """

    bm_iterations_reboot = False
    """
    Override this with the desired behaviour: reboot or not reboot before
    each iteration
    """

    def benchmarkInit(self):
        """
        Code executed before running the benchmark
        """
        pass

    def benchmarkFinalize(self):
        """
        Code executed after running the benchmark
        """
        pass

################################################################################
# Private Interface

    @memoized
    def _parseCommandLine(self):

        parser = argparse.ArgumentParser(
                description='LISA Benchmark Configuration')

        # Bootup settings
        parser.add_argument('--boot-image', type=str,
                default=None,
                help='Path of the Android boot.img to be used')
        parser.add_argument('--boot-timeout', type=int,
                default=60,
                help='Timeout in [s] to wait after a reboot (default 60)')

        # Android settings
        parser.add_argument('--android-device', type=str,
                default=None,
                help='Identifier of the Android target to use')
        parser.add_argument('--android-home', type=str,
                default=None,
                help='Path used to configure ANDROID_HOME')

        # Test customization
        parser.add_argument('--results-dir', type=str,
                default=self.__class__.__name__,
                help='Results folder, '
                     'if specified override test defaults')
        parser.add_argument('--collect', type=str,
                default=None,
                help='Set of metrics to collect, '
                     'e.g. "energy systrace_30" to sample energy and collect a 30s systrace, '
                     'if specified overrides test defaults')
        parser.add_argument('--iterations', type=int,
                default=1,
                help='Number of iterations the same test has to be repeated for (default 1)')
        parser.add_argument('--iterations-pause', type=int,
                default=30,
                help='Amount of time (in seconds) to pause for before each iteration (default 30s)')
        parser.add_argument('--iterations-reboot', action="store_true",
                help='Reboot before each iteration (default False)')

        # Measurements settings
        parser.add_argument('--iio-channel-map', type=str,
                default=None,
                help='List of IIO channels to sample, '
                     'e.g. "ch0:0,ch3:1" to sample CHs 0 and 3, '
                     'if specified overrides test defaults')

        # Parse command line arguments
        return parser.parse_args()


    def _getBmConf(self):
        # Override default configuration with command line parameters
        if self.args.boot_image:
            self.bm_reboot = True
        if self.args.android_device:
            self.bm_conf['device'] = self.args.android_device
        if self.args.android_home:
            self.bm_conf['ANDROID_HOME'] = self.args.android_home
        if self.args.results_dir:
            self.bm_conf['results_dir'] = self.args.results_dir
        if self.args.collect:
            self.bm_collect = self.args.collect
        if self.args.iterations:
            self.bm_iterations = self.args.iterations
        if self.args.iterations_pause:
            self.bm_iterations_pause = self.args.iterations_pause
        if self.args.iterations_reboot:
            self.bm_iterations_reboot = True

        # Override energy meter configuration
        if self.args.iio_channel_map:
            em = {
                'instrument'  : 'acme',
                'channel_map' : {},
            }
            for ch in self.args.iio_channel_map.split(','):
                ch_name, ch_id = ch.split(':')
                em['channel_map'][ch_name] = ch_id
            self.bm_conf['emeter'] = em
            self._log.info('Using ACME energy meter channels: %s', em)

        # Override EM if energy collection not required
        if 'energy' not in self.bm_collect:
            try:
                self.bm_conf.pop('emeter')
            except:
                pass

        return self.bm_conf

    def _getWorkload(self):
        if self.bm_name is None:
            msg = 'Benchmark subclasses must override the `bm_name` attribute'
            raise NotImplementedError(msg)
        # Get a referench to the worload to run
        wl = Workload.getInstance(self.te, self.bm_name)
        if wl is None:
            raise ValueError('Specified benchmark [{}] is not supported'\
                             .format(self.bm_name))
        return wl

    def _getBmParams(self):
        if self.bm_params is None:
            msg = 'Benchmark subclasses must override the `bm_params` attribute'
            raise NotImplementedError(msg)
        return self.bm_params

    def _getBmCollect(self):
        if self.bm_collect is None:
            msg = 'Benchmark subclasses must override the `bm_collect` attribute'
            self._log.warning(msg)
            return ''
        return self.bm_collect

    def _preInit(self):
        """
        Code executed before running the benchmark
        """
        # If iterations_reboot is True we are going to reboot before the
        # first iteration anyway.
        if self.bm_reboot and not self.bm_iterations_reboot:
            self.reboot_target()

        self.iterations_count = 1

    def _preRun(self):
        """
        Code executed before every iteration of the benchmark
        """
        rebooted = False

        if self.bm_reboot and self.bm_iterations_reboot:
            rebooted = self.reboot_target()

        if not rebooted and self.iterations_count > 1:
            self._log.info('Waiting {}[s] before executing iteration {}...'\
                           .format(self.bm_iterations_pause, self.iterations_count))
            sleep(self.bm_iterations_pause)

        self.iterations_count += 1

    def __init__(self):
        """
        Set up logging and trigger running experiments
        """
        setup_logging()
        self._log = logging.getLogger('Benchmark')

        self._log.info('=== CommandLine parsing...')
        self.args = self._parseCommandLine()

        self._log.info('=== TestEnv setup...')
        self.bm_conf = self._getBmConf()
        self.te = TestEnv(self.bm_conf)
        self.target = self.te.target

        self._log.info('=== Initialization...')
        self.wl = self._getWorkload()
        self.out_dir=self.te.res_dir
        try:
            self._preInit()
            self.benchmarkInit()
        except:
            self._log.warning('Benchmark initialization failed: execution aborted')
            raise

        self._log.info('=== Execution...')
        for iter_id in range(1, self.bm_iterations+1):
            self._log.info('=== Iteration {}/{}...'.format(iter_id, self.bm_iterations))
            out_dir = os.path.join(self.out_dir, "{:03d}".format(iter_id))
            try:
                os.makedirs(out_dir)
            except: pass

            self._preRun()

            self.wl.run(out_dir=out_dir,
                        collect=self._getBmCollect(),
                        **self.bm_params)

        self._log.info('=== Finalization...')
        self.benchmarkFinalize()

    def _wait_for_logcat_idle(self, seconds=1):
        lines = 0

        # Clear logcat
        # os.system('{} logcat -s {} -c'.format(adb, DEVICE));
        self.target.clear_logcat()

        # Dump logcat output
        logcat_cmd = 'adb -s {} logcat'.format(self.target.adb_name)
        logcat = Popen(logcat_cmd, shell=True, stdout=PIPE)
        logcat_poll = select.poll()
        logcat_poll.register(logcat.stdout, select.POLLIN)

        # Monitor logcat until it's idle for the specified number of [s]
        self._log.info('Waiting for system to be almost idle')
        self._log.info('   i.e. at least %d[s] of no logcat messages', seconds)
        while True:
            poll_result = logcat_poll.poll(seconds * 1000)
            if not poll_result:
                break
            lines = lines + 1
            line = logcat.stdout.readline(1024)
            if lines % 1000:
                self._log.debug('   still waiting...')
            if lines > 1e6:
                self._log.warning('device logcat seems quite busy, '
                                  'continuing anyway... ')
                break

    def reboot_target(self, disable_charge=True):
        """
        Reboot the target if a "boot-image" has been specified

        If the user specify a boot-image as a command line parameter, this
        method will reboot the target with the specified kernel and wait
        for the target to be up and running.
        """
        rebooted = False

        # Reboot the device, if a boot_image has been specified
        if self.args.boot_image:

            self._log.warning('=== Rebooting...')
            self._log.warning('Rebooting image to use: %s', self.args.boot_image)

            self._log.debug('Waiting 6[s] to enter bootloader...')
            self.target.adb_reboot_bootloader()
            sleep(6)
            # self._fastboot('boot {}'.format(self.args.boot_image))
            cmd = 'boot {}'.format(self.args.boot_image)
            fastboot_command(cmd, device=self.target.adb_name)
            self._log.debug('Waiting {}[s] for boot to start...'\
                            .format(self.args.boot_timeout))
            sleep(self.args.boot_timeout)
            rebooted = True

        else:
            self._log.warning('Device NOT rebooted, using current image')

        # Restart ADB in root mode
        self._log.warning('Restarting ADB in root mode...')
        self.target.adb_root(force=True)

        # TODO add check for kernel SHA1
        self._log.warning('Skipping kernel SHA1 cross-check...')

        # Disable charge via USB
        if disable_charge:
            self._log.debug('Disabling charge over USB...')
            self.target.charging_enabled = False

        # Log current kernel version
        self._log.info('Running with kernel:')
        self._log.info('   %s', self.target.kernel_version)

        # Wait for the system to complete the boot
        self._wait_for_logcat_idle()

        return rebooted

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
