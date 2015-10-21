#!/usr/bin/python

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
from test_env import TestEnv

import wlgen

# Host specific paths
HST_OUT_PREFIX = './results'
HST_OUT_LAST   = './latest'

# Target specific paths
TGT_RUN_DIR     = 'run_dir'
TGT_CGR_ROOT    = '/sys/fs/cgroup'

class EAS_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Initialize globals
        cls.kernel = None
        cls.dtb = None
        cls.governor = None

        cls.print_section('Main', 'Experiments configuration')

        # Load target specific configuration
        target_config = 'target.config'
        cls.conf = cls.load_conf(target_config)

        # Load test specific configuration
        class_config = 'tests/eas/rfc.config'
        cls.conf.update(cls.load_conf(class_config))

        logging.debug('Complete configuration %s', cls.conf)

        # Check for mandatory configurations
        if 'confs' not in cls.conf.keys() or len(cls.conf['confs']) == 0:
            raise ValueError(
                    'Configuration error: missing \'conf\' definitions')
        if 'wloads' not in cls.conf.keys() or len(cls.conf['wloads']) == 0:
            raise ValueError(
                    'Configuration error: missing \'wloads\' definitions')

        # Setup devlib to access the configured target
        cls.env = TestEnv(cls.conf)

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
                wload = cls.test_conf(tc_idx, wl_idx)
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

    # def test_two(self):
    #     """Test 2"""
    #     logging.info(r'Test2')
    #     return True
    #
    # def test_three(self):
    #     """Test 3"""
    #     logging.info(r'Test3')
    #     return True


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
    def schedtune_supported(cls):
        # TODO: check for STune support available on target system
        return False

    @classmethod
    def schedtune_init(cls):

        cls.env.cgroup = None

        if cls.env.target.file_exists(cls.schedtune_cgpath('/schedtest')):
            return

        if not cls.schedtune_supported():
            logging.warning(r'%14s - SchedTune not supported, '\
                    'related experiments will be skipped',
                    'CGroups')
            return

        logging.info(r'%14s - Initialize CGroups for SchedTune support...',
                'CGroups')

        cls.env.target.execute('mount -t tmpfs cgroup {}'\
                .format(TGT_CGR_ROOT), as_root=True)
        cls.env.target.execute('mkdir -p {}/stune'\
                .format(TGT_CGR_ROOT), as_root=True)
        cls.env.target.execute('mount -t cgroup -o schedtune stune {}/stune'\
                .format(TGT_CGR_ROOT), as_root=True)
        cls.env.target.execute('mkdir {}/stune/schedtest || echo'\
                .format(TGT_CGR_ROOT), as_root=True)

        # Keep track of SchedTune CGroup support being available
        cls.env.cgroup = cls.schedtune_cgpath('/schedtest')

    @classmethod
    def schedtune_cgpath(cls, cgroup='', attr=None):
        if attr is None:
            attr=''
        else:
            attr='/'+attr
        return '{}/stune{}{}'.format(TGT_CGR_ROOT, cgroup, attr)

    @classmethod
    def schedtune_boost(cls, cg, boost):
        if cg != '/':
            logging.debug(r'%14s - reset [/] cgroup boost value', 'CGroups')
            cls.env.target.write_value(
                    cls.schedtune_cgpath('/', 'schedtune.boost'), 0)
        logging.debug(r'%14s - set [%s] cgroup boost value to [%d]',
                    'CGroups', cg, boost)
        cls.env.target.write_value(
                cls.schedtune_cgpath('/', 'schedtune.boost'), boost)

    @classmethod
    def setup_kernel(cls, tc):
        # Deploy kernel on the device
        cls.env.install_kernel(tc, reboot=True)
        # Setup the rootfs for the experiments
        cls.setup_rootfs()

    @classmethod
    def setup_sched_features(cls, tc):
        feats = tc['sched_features'].split(",")
        for feat in feats:
            cls.env.target.execute('echo {} > /sys/kernel/debug/sched_features'.format(feat))

    @classmethod
    def setup_rootfs(cls):
        # Initialize CGroups if required
        cls.schedtune_init()
        # Setup target folder for experiments execution
        cls.env.run_dir = '{}/{}'\
                .format(cls.env.target.working_directory, TGT_RUN_DIR)
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

        if cls.env.cgroup is None:
            if 'cgroup' not in tc.keys():
                return True
            else:
                logging.warning(r'%14s - Configuration error: '\
                        'CGroup support NOT available',
                        'SchedTune')
                return False

        if 'cgroup' not in tc.keys():
            logging.debug(r'%14s - reset root control group boost value', \
                    'SchedTune')
            cls.schedtune_boost('/', 0)
            return True

        logging.info(r'%14s - setup [%s] with boost value [%s]',
                'SchedTune', tc['cgroup'], tc['boost'])
        cls.schedtune_boost(tc['cgroup'], tc['boost'])
        return True

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
    def wload_rtapp(cls, wl_idx, wlspec, cpus, cgroup):
        conf = wlspec['conf']
        logging.debug(r'%14s - Configuring [%s] rt-app...',
                'RTApp', conf['class'])

        if conf['class'] == 'periodic':
            task_idxs = cls.wload_rtapp_task_idxs(wl_idx, conf['tasks'])
            params = {}
            for idx in task_idxs:
                task = conf['prefix'] + str(idx)
                params[task] = wlgen.RTA.periodic(**conf['params'])
            rtapp = wlgen.RTA(cls.env.target,
                        wl_idx, calibration = cls.env.calib)
            rtapp.conf(kind='profile', params=params,
                    cpus=cpus, cgroup=cgroup,
                    run_dir=cls.env.run_dir)
            return rtapp

        if conf['class'] == 'custom':
            rtapp = wlgen.RTA(cls.env.target,
                        wl_idx, calibration = cls.env.calib)
            rtapp.conf(kind='custom',
                    params=conf['json'],
                    duration=conf['duration'],
                    cpus=cpus, cgroup=cgroup,
                    run_dir=cls.env.run_dir)
            return rtapp

        raise ValueError('Configuration error - '
                'unsupported \'class\' value for [{}] '\
                'RT-App workload specification'\
                .format(wl_idx))

    @classmethod
    def wload_perf_bench(cls, wl_idx, wlspec, cpus, cgroup):
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

        # CGROUP: setup execution on cgroup if required by configuration
        # TODO: add cgroup specification support
        cgroup = None

        # CPUS: setup executioon on CPUs if required by configuration
        # TODO: add CPUs specification support
        cpus = None

        if wlspec['type'] == 'rt-app':
            return cls.wload_rtapp(wl_idx, wlspec, cpus, cgroup)
        if wlspec['type'] == 'perf_bench':
            return cls.wload_perf_bench(wl_idx, wlspec, cpus, cgroup)


        raise ValueError('Configuration error - '
                'unsupported \'type\' value for [{}] '\
                'workload specification'\
                .format(wl_idx))

    @classmethod
    def test_conf(cls, tc_idx, wl_idx):

        # Configure the test workload
        wlspec = cls.conf['wloads'][wl_idx]
        wload = cls.wload_conf(wl_idx, wlspec)

        # Keep track of platform configuration
        cls.env.test_dir = '{}/{}:{}:{}'\
            .format(cls.env.res_dir, wload.wtype, tc_idx, wl_idx)
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
        cls.env.energy_reset()

        # WORKLOAD: Run the configured workload
        wload.run(out_dir=cls.env.out_dir)

        # ENERGY: collect measurements
        cls.env.energy_report(cls.env.out_dir)

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
                'SchedTune', target_dir)
        # cls.env.target.execute('rm {}/output.txt'\
        #         .format(target_dir), as_root=True)


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
