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

import collections
import datetime
import json
import os
import re
import time

# Configure logging
import logging
reload(logging)
logging.basicConfig(
    format='%(asctime)-9s %(levelname)-8s: %(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S')

# Add UnitTest support
import unittest

# Add support for Test Environment configuration
from env import TestEnv

# Add JSON parsing support
from conf import JsonConf

import wlgen

# Target specific paths
TGT_RUN_DIR     = 'run_dir'

################################################################################
# Base RFC class
################################################################################

class TestBase(unittest.TestCase):

    @classmethod
    def setUpTest(cls, tests_config):

        # Initialize globals
        cls.kernel = None
        cls.dtb = None
        cls.governor = None
        cls.cgroup = None

        cls.print_section('Main', 'Experiments configuration')

        # Load test specific configuration
        tests_config = os.path.join('tests/eas', tests_config)
        logging.info('%14s - Loading EAS RFC tests configuration [%s]...',
                'Main', tests_config)
        json_conf = JsonConf(tests_config)
        cls.conf = json_conf.load()


        # Check for mandatory configurations
        if 'confs' not in cls.conf or not cls.conf['confs']:
            raise ValueError(
                    'Configuration error: missing \'conf\' definitions')
        if 'wloads' not in cls.conf or not cls.conf['wloads']:
            raise ValueError(
                    'Configuration error: missing \'wloads\' definitions')

        # Setup devlib to access the configured target
        cls.env = TestEnv(test_conf = cls.conf)

        # Compute total number of experiments
        cls.exp_count = cls.conf['iterations'] \
                * len(cls.conf['wloads']) \
                * len(cls.conf['confs'])

        cls.print_section('Main', 'Experiments execution')

        # Run all the configured experiments
        exp_idx = 1
        for tc in cls.conf['confs']:
            tc_idx = tc['tag']
            # TARGET: configuration
            if not cls.target_configure(tc):
                continue
            for wl_idx in cls.conf['wloads']:
                # TEST: configuration
                wload = cls.wload_init(tc_idx, wl_idx)
                for itr_idx in range(1, cls.conf['iterations']+1):
                    # WORKLOAD: execution
                    cls.wload_run(exp_idx, tc_idx, wl_idx, wload, itr_idx)
                    exp_idx += 1

        cls.print_section('Main', 'Experiments post-processing')


################################################################################
# Test cases
################################################################################

    def test_execution_complete(self):
        """Check that data have been collected from the target"""
        logging.info(r'Check for data being collected')
        return True


################################################################################
# Utility methods
################################################################################

    @classmethod
    def load_conf(cls, filename):
        """ Parse a JSON file
            First remove comments and then use the json module package
            Comments look like :
                // ...
            or
                /*
                ...
                */
        """
        if not os.path.isfile(filename):
            raise RuntimeError(
                'Missing configuration file: {}'.format(filename)
            )
        logging.debug('loading JSON...')

        with open(filename) as f:
            content = ''.join(f.readlines())

            ## Looking for comments
            match = JSON_COMMENTS_RE.search(content)
            while match:
                # single line comment
                content = content[:match.start()] + content[match.end():]
                match = JSON_COMMENTS_RE.search(content)

            # Return json file
            conf = json.loads(content, parse_int=int)
            logging.debug('Target config: %s', conf)

            return conf

    @classmethod
    def print_section(cls, tag, message):
        logging.info('')
        logging.info(FMT_SECTION)
        logging.info(r'%14s - %s', tag, message)
        logging.info(FMT_SECTION)

    @classmethod
    def print_header(cls, tag, message):
        logging.info('')
        logging.info(FMT_HEADER)
        logging.info(r'%14s - %s', tag, message)

    @classmethod
    def print_title(cls, tag, message):
        logging.info(FMT_TITLE)
        logging.info(r'%14s - %s', tag, message)

    @classmethod
    def cgroups_init(cls, tc):
        if 'cgroups' not in tc:
            return True
        logging.info(r'%14s - Initialize CGroups support...', 'CGroups')
        errors = False
        for kind in tc['cgroups']['conf']:
            logging.info(r'%14s - Setup [%s] controller...',
                    'CGroups', kind)
            controller = cls.env.target.cgroups.controller(kind)
            if not controller:
                logging.warning(r'%14s - CGroups controller [%s] NOT available',
                        'CGroups', kind)
                errors = True
        return not errors

    @classmethod
    def setup_kernel(cls, tc):
        # Deploy kernel on the device
        cls.env.install_kernel(tc, reboot=True)
        # Setup the rootfs for the experiments
        cls.setup_rootfs(tc)

    @classmethod
    def setup_sched_features(cls, tc):
        if 'sched_features' not in tc:
            logging.debug('%14s - Configuration not provided', 'SchedFeatures')
            return
        feats = tc['sched_features'].split(",")
        for feat in feats:
            cls.env.target.execute('echo {} > /sys/kernel/debug/sched_features'.format(feat))

    @classmethod
    def setup_rootfs(cls, tc):
        # Initialize CGroups if required
        cls.cgroups_init(tc)
        # Setup target folder for experiments execution
        cls.env.run_dir = os.path.join(
                cls.env.target.working_directory, TGT_RUN_DIR)
        # Create run folder as tmpfs
        logging.debug('%14s - Setup RT-App run folder [%s]...',
                'TargetSetup', cls.env.run_dir)
        cls.env.target.execute('[ -d {0} ] || mkdir {0}'\
                .format(cls.env.run_dir), as_root=True)
        cls.env.target.execute(
                'grep schedtest /proc/mounts || '\
                '  mount -t tmpfs -o size=1024m {} {}'\
                .format('schedtest', cls.env.run_dir),
                as_root=True)

    @classmethod
    def setup_cpufreq(cls, tc):
        if 'cpufreq' not in tc:
            logging.debug('%14s - Configuration not provided', 'CPUFreq')
            return
        if cls.governor == tc['cpufreq']['governor']:
            return
        logging.info(r'%14s - Configuring all CPUs to use [%s] governor',
                'CPUFreq', tc['cpufreq']['governor'])
        try:
            cpufreq = tc['cpufreq']
        except KeyError:
            logging.warning(r'%14s - Using currently configured governor',
                    'CPUFreq')
            return
        if cpufreq['governor'] == 'ondemand':
            try:
                sampling_rate = cpufreq['params']['sampling_rate']
            except KeyError:
                sampling_rate = 20000
            cls.env.target.execute(
                    'for CPU in /sys/devices/system/cpu/cpu[0-9]*; do   '\
                    '   echo {} > $CPU/cpufreq/scaling_governor;  '\
                    '   echo {} > $CPU/cpufreq/ondemand/sampling_rate;  '\
                    'done'\
                    .format('ondemand', sampling_rate))
        else:
            cls.env.target.execute(
                    'for CPU in /sys/devices/system/cpu/cpu[0-9]*; do   '\
                    '   echo {} > $CPU/cpufreq/scaling_governor;  '\
                    'done'\
                    .format(cpufreq['governor']))
        # Keep track of currently configured governor
        cls.governor = cpufreq['governor']

    @classmethod
    def setup_cgroups(cls, tc):
        if 'cgroups' not in tc:
            return True
        # Setup default CGroup to run tasks into
        if 'default' in tc['cgroups']:
            cls.cgroup = tc['cgroups']['default']
        # Configure each required controller
        if 'conf' not in tc['cgroups']:
            return True
        errors = False
        for kind in tc['cgroups']['conf']:
            controller = cls.env.target.cgroups.controller(kind)
            if not controller:
                logging.warning(r'%14s - Configuration error: '\
                        '[%s] contoller NOT supported',
                        'CGroups', kind)
                errors = True
                continue
            cls.setup_controller(tc, controller)
        return not errors

    @classmethod
    def setup_controller(cls, tc, controller):
        kind = controller.kind
        # Configure each required groups for that controller
        errors = False
        for name in tc['cgroups']['conf'][controller.kind]:
            group = controller.cgroup(name)
            if not group:
                logging.warning(r'%14s - Configuration error: '\
                        '[%s/%s] cgroup NOT available',
                        'CGroups', kind, name)
                errors = True
                continue
            cls.setup_group(tc, group)
        return not errors

    @classmethod
    def setup_group(cls, tc, group):
        kind = group.controller.kind
        name = group.name
        # Configure each required attribute
        group.set(**tc['cgroups']['conf'][kind][name])


    @classmethod
    def target_reboot(cls):
        # TODO: actually reboot the target and wait for it to be back online
        cls.governor = None


    @classmethod
    def target_configure(cls, tc):
        cls.print_header('TargetConfig',
                r'configuring target for [{}] experiments'\
                .format(tc['tag']))
        cls.setup_kernel(tc)
        cls.setup_sched_features(tc)
        cls.setup_cpufreq(tc)
        return cls.setup_cgroups(tc)

    # def cleanup(cls):
    #     target.execute('umount ' + wl_logs, as_root=True)
    #     target.execute('rmdir ' + wl_logs, as_root=True)

    @classmethod
    def wload_rtapp_task_idxs(cls, wl_idx, tasks):
        if type(tasks) == int:
            return range(tasks)
        if tasks == 'cpus':
            return range(len(cls.env.target.core_names))
        if tasks == 'little':
            return range(len([t
                for t in cls.env.target.core_names
                if t == cls.env.target.little_core]))
        if tasks == 'big':
            return range(len([t
                for t in cls.env.target.core_names
                if t == cls.env.target.big_core]))
        raise ValueError('Configuration error - '
                'unsupported \'tasks\' value for [{}] '\
                'RT-App workload specification'\
                .format(wl_idx))

    @classmethod
    def wload_cpus(cls, wl_idx, wlspec):
        if not 'cpus' in wlspec['conf']:
            return None
        cpus = wlspec['conf']['cpus']

        if type(cpus) == int:
            return list(cpus)
        if cpus.startswith('littles'):
            if 'first' in cpus:
                return [ cls.env.target.bl.littles_online[0] ]
            if 'last' in cpus:
                return [ cls.env.target.bl.littles_online[-1] ]
            return cls.env.target.bl.littles_online
        if cpus.startswith('bigs'):
            if 'first' in cpus:
                return [ cls.env.target.bl.bigs_online[0] ]
            if 'last' in cpus:
                return [ cls.env.target.bl.bigs_online[-1] ]
            return cls.env.target.bl.bigs_online
        raise ValueError('Configuration error - '
                'unsupported [{}] \'cpus\' value for [{}] '\
                'workload specification'\
                .format(cpus, wl_idx))

    @classmethod
    def wload_rtapp(cls, wl_idx, wlspec, cpus):
        conf = wlspec['conf']
        logging.debug(r'%14s - Configuring [%s] rt-app...',
                'RTApp', conf['class'])

        # Setup a default "empty" task name prefix
        if 'prefix' not in conf:
            conf['prefix'] = 'task_'

        if conf['class'] == 'profile':
            params = {}
            # Load each task specification
            for task_name in conf['params']:
                task = conf['params'][task_name]
                task_name = conf['prefix'] + task_name
                if task['kind'] not in wlgen.RTA.__dict__:
                    logging.error(r'%14s - RTA task of kind [%s] not supported',
                            'RTApp', task['kind'])
                    raise ValueError('Configuration error - '
                        'unsupported \'kind\' value for task [{}] '\
                        'in RT-App workload specification'\
                        .format(task))
                task_ctor = getattr(wlgen.RTA, task['kind'])
                params[task_name] = task_ctor(**task['params'])
            rtapp = wlgen.RTA(cls.env.target,
                        wl_idx, calibration = cls.env.calibration())
            rtapp.conf(kind='profile', params=params,
                    cpus=cpus, run_dir=cls.env.run_dir)
            return rtapp

        if conf['class'] == 'periodic':
            task_idxs = cls.wload_rtapp_task_idxs(wl_idx, conf['tasks'])
            params = {}
            for idx in task_idxs:
                task = conf['prefix'] + str(idx)
                params[task] = wlgen.RTA.periodic(**conf['params'])
            rtapp = wlgen.RTA(cls.env.target,
                        wl_idx, calibration = cls.env.calibration())
            rtapp.conf(kind='profile', params=params,
                    cpus=cpus, run_dir=cls.env.run_dir)
            return rtapp

        if conf['class'] == 'custom':
            rtapp = wlgen.RTA(cls.env.target,
                        wl_idx, calibration = cls.env.calib)
            rtapp.conf(kind='custom',
                    params=conf['json'],
                    duration=conf['duration'],
                    cpus=cpus, run_dir=cls.env.run_dir)
            return rtapp

        raise ValueError('Configuration error - '
                'unsupported \'class\' value for [{}] '\
                'RT-App workload specification'\
                .format(wl_idx))

    @classmethod
    def wload_perf_bench(cls, wl_idx, wlspec, cpus):
        conf = wlspec['conf']
        logging.debug(r'%14s - Configuring perf_message...',
                'PerfMessage')

        if conf['class'] == 'messaging':
            perf_bench = wlgen.PerfMessaging(cls.env.target, wl_idx)
            perf_bench.conf(**conf['params'])
            return perf_bench

        if conf['class'] == 'pipe':
            perf_bench = wlgen.PerfPipe(cls.env.target, wl_idx)
            perf_bench.conf(**conf['params'])
            return perf_bench

        raise ValueError('Configuration error - '\
                'unsupported \'class\' value for [{}] '\
                'perf bench workload specification'\
                .format(wl_idx))

    @classmethod
    def wload_conf(cls, wl_idx, wlspec):

        # CPUS: setup execution on CPUs if required by configuration
        cpus = cls.wload_cpus(wl_idx, wlspec)

        if wlspec['type'] == 'rt-app':
            return cls.wload_rtapp(wl_idx, wlspec, cpus)
        if wlspec['type'] == 'perf_bench':
            return cls.wload_perf_bench(wl_idx, wlspec, cpus)


        raise ValueError('Configuration error - '
                'unsupported \'type\' value for [{}] '\
                'workload specification'\
                .format(wl_idx))

    @classmethod
    def wload_init(cls, tc_idx, wl_idx):

        # Configure the test workload
        wlspec = cls.conf['wloads'][wl_idx]
        wload = cls.wload_conf(wl_idx, wlspec)

        # Keep track of platform configuration
        cls.env.test_dir = '{}/{}:{}:{}'\
            .format(cls.env.res_dir, wload.wtype, tc_idx, wl_idx)
        os.system('mkdir -p ' + cls.env.test_dir)
        cls.env.platform_dump(cls.env.test_dir)

        return wload

    @classmethod
    def wload_run(cls, exp_idx, tc_idx, wl_idx, wload, run_idx):

        cls.print_title('MultiRun', 'Experiment {}/{}, [{}:{}] {}/{}'\
                .format(exp_idx, cls.exp_count,
                        tc_idx, wl_idx,
                        run_idx, cls.conf['iterations']))

        # Setup local results folder
        cls.wload_run_init(run_idx)

        # FTRACE: start (if a configuration has been provided)
        if cls.env.ftrace:
            logging.warning('%14s - Starting FTrace', 'MultiRun')
            cls.env.ftrace.start()

        # ENERGY: start sampling
        cls.env.emeter.reset()

        # WORKLOAD: Run the configured workload
        wload.run(out_dir=cls.env.out_dir, cgroup=cls.cgroup)

        # ENERGY: collect measurements
        cls.env.emeter.report(cls.env.out_dir)

        # FTRACE: stop and collect measurements
        if cls.env.ftrace:
            cls.env.ftrace.stop()
            cls.env.ftrace.get_trace(cls.env.out_dir + '/trace.dat')

    @classmethod
    def wload_run_init(cls, run_idx):
        cls.env.out_dir = '{}/{}'\
                .format(cls.env.test_dir, run_idx)
        logging.debug(r'%14s - out_dir [%s]', 'MultiRun', cls.env.out_dir)
        os.system('mkdir -p ' + cls.env.out_dir)

        logging.debug(r'%14s - cleanup target output folder', 'MultiRun')

        target_dir = cls.env.target.working_directory
        logging.debug('%14s - setup target directory [%s]',
                'MultiRun', target_dir)
        # cls.env.target.execute('rm {}/output.txt'\
        #         .format(target_dir), as_root=True)


################################################################################
# Specific RFC test cases
################################################################################

class EAS(TestBase):

    @classmethod
    def setUpClass(cls):
        super(EAS, cls).setUpTest('rfc_eas.config')


class SFreq(TestBase):

    @classmethod
    def setUpClass(cls):
        super(SFreq, cls).setUpTest('rfc_sfreq.config')


class STune(TestBase):

    @classmethod
    def setUpClass(cls):
        super(STune, cls).setUpTest('rfc_stune.config')


################################################################################
# Globals
################################################################################

# Regular expression for comments
JSON_COMMENTS_RE = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

# Logging formatters
FMT_SECTION = r'{:#<80}'.format('')
FMT_HEADER  = r'{:=<80}'.format('')
FMT_TITLE   = r'{:~<80}'.format('')

if __name__ == '__main__':
    unittest.main()

# vim :set tabstop=4 shiftwidth=4 expandtab
