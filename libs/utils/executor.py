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

from bart.common.Analyzer import Analyzer
import collections
from collections import namedtuple
import datetime
import gzip
import json
import os
import re
import time
import trappy
from devlib import TargetError

# Configure logging
import logging

# Add JSON parsing support
from conf import JsonConf

import wlgen

from devlib import TargetError

Experiment = namedtuple('Experiment', ['wload_name', 'wload',
                                       'conf', 'iteration', 'out_dir'])

class Executor():
    """
    Abstraction for running sets of experiments and gathering data from targets

    An executor can be configured to run a set of workloads (wloads) in each
    different target configuration of a specified set (confs). These wloads and
    confs can be specified by the "experiments_conf" input dictionary. Each
    (workload, conf, iteration) tuple is called an "experiment".

    After the workloads have been run, the Executor object's `experiments`
    attribute is a list of Experiment objects. The `out_dir` attribute of these
    objects can be used to find the results of the experiment. This output
    directory follows this format:

        results/<test_id>/<wltype>:<conf>:<wload>/<run_id>

    where:

        test_id
            Is the "tid" defined by the experiments_conf, or a timestamp based
            folder in case "tid" is not specified.
        wltype
            Is the class of workload executed, e.g. rtapp or sched_perf.
        conf
            Is the "tag" of one of the specified **confs**.
        wload
            Is the identifier of one of the specified **wloads**.
        run_id
            Is the progressive execution number from 1 up to the specified
            **iterations**.

    :param experiments_conf: Dict with experiment configuration. Keys are:

        **confs**
          Mandatory. Platform configurations to be tested. List of dicts,
          each with keys:

            tag
              String to identify this configuration. Required, may be empty.
            flags
              List of strings describing features required for this
              conf. Available flags are:

              "ftrace"
                  Enable collecting ftrace during the experiment.
              "freeze_userspace"
                  Use the cgroups freezer to freeze as many userspace tasks as
                  possible during the experiment execution, in order to reduce
                  system noise. Some tasks cannot be frozen, such as those
                  required to maintain a connection to LISA.

            sched_features
              Optional list of features to be written to
              /sys/kernel/debug/sched_features. Prepend "NO\_" to a feature to
              actively disable it. Requires ``CONFIG_SCHED_DEBUG`` in target
              kernel.
            cpufreq
              Parameters to configure cpufreq via Devlib's cpufreq
              module. Dictionary with fields:

              .. TODO link to devlib cpufreq module docs (which don't exist)

              governor
                cpufreq governor to set (for all CPUs) before execution. The
                previous governor is not restored when execution is finished.
              governor_tunables
                Dictionary of governor-specific tunables, expanded and passed as
                kwargs to the cpufreq module's ``set_governor_tunables`` method.
              freq
                Requires "governor" to be "userspace". Dictionary mapping CPU
                numbers to frequencies. Exact frequencies should be available on
                those CPUs. It is not necessary to provide a frequency for every
                CPU - the frequency for unspecified CPUs is not affected. Note
                that cpufreq will transparrently set the frequencies of any
                other CPUs sharing a clock domain.

            cgroups
              Optional cgroups configuration. To use this, ensure the 'cgroups'
              devlib module is enabled in your test_conf Contains fields:

              .. TODO reference test_conf
              .. TODO link to devlib cgroup module's docs (which don't exist)

              conf
                Dict specifying the cgroup controllers, cgroups, and cgroup
                parameters to setup. If a controller listed here is not
                enabled in the target kernel, a message is logged and the
                configuration is **ignored**. Of the form:

                ::

                  "<controller>" : {
                      "<group1>" : { "<group_param" : <value> }
                      "<group2>" : { "<group_param" : <value> }
                  }

                These cgroups can then be used in the "cgroup" field of workload
                specifications.

              default
                The default cgroup to run workloads in, if no "cgroup" is
                specified.

              For example, to create a cpuset cgroup named "/big" which
              restricts constituent tasks to CPUs 1 and 2:

              ::

                "cgroups" : {
                    "conf" : {
                        "cpuset" : {
                            "/big" : {"cpus" : "1-2"},
                        }
                    },
                    "default" : "/",
                }

          **wloads**
            .. TODO document wloads field.

            Mandatory. Workloads to run on each platform configuration

          **iterations**
            Number of iterations for each workload/conf combination. Default
            is 1.
    :type experiments_conf: dict

    :ivar experiments: After calling :func:`run`, the list of
                       :class:`Experiment` s that were run

    :ivar iterations: The number of iterations run for each wload/conf pair
                       (i.e. ``experiments_conf['iterations']``.

    """

    def __init__(self, test_env, experiments_conf):
        # Initialize globals
        self._default_cgroup = None
        self._cgroup = None
        self._old_selinux_mode = None

        # Setup logging
        self._log = logging.getLogger('Executor')

        # Setup test configuration
        if isinstance(experiments_conf, dict):
            self._log.info('Loading custom (inline) test configuration')
            self._experiments_conf = experiments_conf
        elif isinstance(experiments_conf, str):
            self._log.info('Loading custom (file) test configuration')
            json_conf = JsonConf(experiments_conf)
            self._experiments_conf = json_conf.load()
        else:
            raise ValueError(
                'experiments_conf must be either a dictionary or a filepath')

        # Check for mandatory configurations
        if not self._experiments_conf.get('confs', None):
            raise ValueError('Configuration error: '
                             'missing "conf" definitions')
        if not self._experiments_conf.get('wloads', None):
            raise ValueError('Configuration error: '
                             'missing "wloads" definitions')

        self.te = test_env
        self.target = self.te.target

        self.iterations = self._experiments_conf.get('iterations', 1)
        # Compute total number of experiments
        self._exp_count = self.iterations \
                * len(self._experiments_conf['wloads']) \
                * len(self._experiments_conf['confs'])

        self._print_section('Experiments configuration')

        self._log.info('Configured to run:')

        self._log.info('   %3d target configurations:',
                       len(self._experiments_conf['confs']))
        target_confs = [conf['tag'] for conf in self._experiments_conf['confs']]
        target_confs = ', '.join(target_confs)
        self._log.info('      %s', target_confs)

        self._log.info('   %3d workloads (%d iterations each)',
                       len(self._experiments_conf['wloads']),
                       self.iterations)
        wload_confs = ', '.join(self._experiments_conf['wloads'])
        self._log.info('      %s', wload_confs)

        self._log.info('Total: %d experiments', self._exp_count)

        self._log.info('Results will be collected under:')
        self._log.info('      %s', self.te.res_dir)

        if any(wl['type'] == 'rt-app'
               for wl in self._experiments_conf['wloads'].values()):
            self._log.info('rt-app workloads found, installing tool on target')
            self.te.install_tools(['rt-app'])

    def run(self):
        self._print_section('Experiments execution')

        self.experiments = []

        # Run all the configured experiments
        exp_idx = 0
        for tc in self._experiments_conf['confs']:
            # TARGET: configuration
            if not self._target_configure(tc):
                continue
            for wl_idx in self._experiments_conf['wloads']:
                # TEST: configuration
                wload, test_dir = self._wload_init(tc, wl_idx)
                for itr_idx in range(1, self.iterations + 1):
                    exp = Experiment(
                        wload_name=wl_idx,
                        wload=wload,
                        conf=tc,
                        iteration=itr_idx,
                        out_dir=os.path.join(test_dir, str(itr_idx)))
                    self.experiments.append(exp)

                    # WORKLOAD: execution
                    if self._target_conf_flag(tc, 'freeze_userspace'):
                        with self.te.freeze_userspace():
                            self._wload_run(exp_idx, exp)
                    else:
                        self._wload_run(exp_idx, exp)
                    exp_idx += 1
            self._target_cleanup(tc)

        self._print_section('Experiments execution completed')
        self._log.info('Results available in:')
        self._log.info('      %s', self.te.res_dir)


################################################################################
# Target Configuration
################################################################################

    def _cgroups_init(self, tc):
        self._default_cgroup = None
        if 'cgroups' not in tc:
            return True
        if 'cgroups' not in self.target.modules:
            raise RuntimeError('CGroups module not available. Please ensure '
                               '"cgroups" is listed in your target/test modules')
        self._log.info('Initialize CGroups support...')
        errors = False
        for kind in tc['cgroups']['conf']:
            self._log.info('Setup [%s] CGroup controller...', kind)
            controller = self.target.cgroups.controller(kind)
            if not controller:
                self._log.warning('CGroups controller [%s] NOT available',
                                  kind)
                errors = True
        return not errors

    def _setup_kernel(self, tc):
        # Deploy kernel on the device
        self.te.install_kernel(tc, reboot=True)
        # Setup the rootfs for the experiments
        self._setup_rootfs(tc)

    def _setup_sched_features(self, tc):
        if 'sched_features' not in tc:
            self._log.debug('Scheduler features configuration not provided')
            return
        feats = tc['sched_features'].split(",")
        for feat in feats:
            self._log.info('Set scheduler feature: %s', feat)
            self.target.execute('echo {} > /sys/kernel/debug/sched_features'.format(feat),
                                as_root=True)

    @staticmethod
    def get_run_dir(target):
        return os.path.join(target.working_directory, TGT_RUN_DIR)

    def _setup_rootfs(self, tc):
        # Initialize CGroups if required
        self._cgroups_init(tc)
        # Setup target folder for experiments execution
        self.te.run_dir = self.get_run_dir(self.target)
        # Create run folder as tmpfs
        self._log.debug('Setup RT-App run folder [%s]...', self.te.run_dir)
        self.target.execute('[ -d {0} ] || mkdir {0}'\
                .format(self.te.run_dir))

        if self.target.is_rooted:
            self.target.execute(
                'grep schedtest /proc/mounts || '\
                '  mount -t tmpfs -o size=1024m {} {}'\
                .format('schedtest', self.te.run_dir),
                as_root=True)

            # tmpfs mounts have an SELinux context with "tmpfs" as the type
            # (while other files we create have "shell_data_file"). That
            # prevents non-root users from creating files in tmpfs mounts. For
            # now, just put SELinux in permissive mode to get around that.
            try:
                # First, save the old SELinux mode
                self._old_selinux_mode = self.target.execute('getenforce')
            except TargetError:
                # Probably the target doesn't have SELinux. No problem.
                pass
            else:

                self._log.warning('Setting target SELinux in permissive mode')
                self.target.execute('setenforce 0', as_root=True)
        else:
            self._log.warning('Not mounting tmpfs because no root')

    def _setup_cpufreq(self, tc):
        if 'cpufreq' not in tc:
            self._log.warning('cpufreq governor not specified, '
                              'using currently configured governor')
            return

        cpufreq = tc['cpufreq']
        self._log.info('Configuring all CPUs to use [%s] cpufreq governor',
                       cpufreq['governor'])

        self.target.cpufreq.set_all_governors(cpufreq['governor'])

        if 'freqs' in cpufreq:
            if cpufreq['governor'] != 'userspace':
                raise ValueError('Must use userspace governor to set CPU freqs')
            self._log.info(r'%14s - CPU frequencies: %s',
                    'CPUFreq', str(cpufreq['freqs']))
            for cpu, freq in cpufreq['freqs'].iteritems():
                self.target.cpufreq.set_frequency(cpu, freq)

        if 'params' in cpufreq:
            self._log.info('governor params: %s', str(cpufreq['params']))
            for cpu in self.target.list_online_cpus():
                self.target.cpufreq.set_governor_tunables(
                        cpu,
                        cpufreq['governor'],
                        **cpufreq['params'])

    def _setup_cgroups(self, tc):
        if 'cgroups' not in tc:
            return True
        # Setup default CGroup to run tasks into
        if 'default' in tc['cgroups']:
            self._default_cgroup = tc['cgroups']['default']
        # Configure each required controller
        if 'conf' not in tc['cgroups']:
            return True
        errors = False
        for kind in tc['cgroups']['conf']:
            controller = self.target.cgroups.controller(kind)
            if not controller:
                self._log.warning('Configuration error: '
                                  '[%s] contoller NOT supported',
                                  kind)
                errors = True
                continue
            self._setup_controller(tc, controller)
        return not errors

    def _setup_controller(self, tc, controller):
        kind = controller.kind
        # Configure each required groups for that controller
        errors = False
        for name in tc['cgroups']['conf'][controller.kind]:
            if name[0] != '/':
                raise ValueError('Wrong CGroup name [{}]. '
                                 'CGroups names must start by "/".'
                                 .format(name))
            group = controller.cgroup(name)
            if not group:
                self._log.warning('Configuration error: '
                                  '[%s/%s] cgroup NOT available',
                                  kind, name)
                errors = True
                continue
            self._setup_group(tc, group)
        return not errors

    def _setup_group(self, tc, group):
        kind = group.controller.kind
        name = group.name
        # Configure each required attribute
        group.set(**tc['cgroups']['conf'][kind][name])

    def _setup_files(self, tc):
        if 'files' not in tc:
            self._log.debug('\'files\' Configuration block not provided')
            return True
        for name, value in tc['files'].iteritems():
            check = False
            if name.startswith('!/'):
                check = True
                name = name[1:]
            self._log.info('File Write(check=%s): \'%s\' -> \'%s\'',
                         check, value, name)
            try:
                self.target.write_value(name, value, True)
            except TargetError:
                self._log.info('File Write Failed: \'%s\' -> \'%s\'',
                         value, name)
                if check:
                    raise
        return False

    def _target_configure(self, tc):
        self._print_header(
                'configuring target for [{}] experiments'\
                .format(tc['tag']))
        self._setup_kernel(tc)
        self._setup_sched_features(tc)
        self._setup_cpufreq(tc)
        self._setup_files(tc)
        return self._setup_cgroups(tc)

    def _target_conf_flag(self, tc, flag):
        if 'flags' not in tc:
            has_flag = False
        else:
            has_flag = flag in tc['flags']
        self._log.debug('Check if target configuration [%s] has flag [%s]: %s',
                        tc['tag'], flag, has_flag)
        return has_flag

    def _target_cleanup(self, tc):
        if self._old_selinux_mode is not None:
            self._log.info('Restoring target SELinux mode: %s',
                           self._old_selinux_mode)
            self.target.execute('setenforce ' + self._old_selinux_mode,
                                as_root=True)

################################################################################
# Workload Setup and Execution
################################################################################

    def _wload_cpus(self, wl_idx, wlspec):
        if not 'cpus' in wlspec['conf']:
            return None
        cpus = wlspec['conf']['cpus']

        if type(cpus) == list:
            return cpus
        if type(cpus) == int:
            return [cpus]

        # SMP target (or not bL module loaded)
        if not hasattr(self.target, 'bl'):
            if 'first' in cpus:
                return [ self.target.list_online_cpus()[0] ]
            if 'last' in cpus:
                return [ self.target.list_online_cpus()[-1] ]
            return self.target.list_online_cpus()

        # big.LITTLE target
        if cpus.startswith('littles'):
            if 'first' in cpus:
                return [ self.target.bl.littles_online[0] ]
            if 'last' in cpus:
                return [ self.target.bl.littles_online[-1] ]
            return self.target.bl.littles_online
        if cpus.startswith('bigs'):
            if 'first' in cpus:
                return [ self.target.bl.bigs_online[0] ]
            if 'last' in cpus:
                return [ self.target.bl.bigs_online[-1] ]
            return self.target.bl.bigs_online
        raise ValueError('unsupported [{}] "cpus" value for [{}] '
                         'workload specification'
                         .format(cpus, wl_idx))

    def _wload_task_idxs(self, wl_idx, tasks):
        if type(tasks) == int:
            return range(tasks)
        if tasks == 'cpus':
            return range(len(self.target.core_names))
        if tasks == 'little':
            return range(len([t
                for t in self.target.core_names
                if t == self.target.little_core]))
        if tasks == 'big':
            return range(len([t
                for t in self.target.core_names
                if t == self.target.big_core]))
        raise ValueError('unsupported "tasks" value for [{}] RT-App '
                         'workload specification'
                         .format(wl_idx))

    def _wload_rtapp(self, wl_idx, wlspec, cpus):
        conf = wlspec['conf']
        self._log.debug('Configuring [%s] rt-app...', conf['class'])

        # Setup a default "empty" task name prefix
        if 'prefix' not in conf:
            conf['prefix'] = 'task_'

        # Setup a default loadref CPU
        loadref = None
        if 'loadref' in wlspec:
            loadref = wlspec['loadref']

        if conf['class'] == 'profile':
            params = {}
            # Load each task specification
            for task_name, task in conf['params'].items():
                if task['kind'] not in wlgen.__dict__:
                    self._log.error('RTA task of kind [%s] not supported',
                                    task['kind'])
                    raise ValueError('unsupported "kind" value for task [{}] '
                                     'in RT-App workload specification'
                                     .format(task))
                task_ctor = getattr(wlgen, task['kind'])
                num_tasks = task.get('tasks', 1)
                task_idxs = self._wload_task_idxs(wl_idx, num_tasks)
                for idx in task_idxs:
                    idx_name = "_{}".format(idx) if len(task_idxs) > 1 else ""
                    task_name_idx = conf['prefix'] + task_name + idx_name
                    params[task_name_idx] = task_ctor(**task['params']).get()

            rtapp = wlgen.RTA(self.target,
                        wl_idx, calibration = self.te.calibration())
            rtapp.conf(kind='profile', params=params, loadref=loadref,
                       cpus=cpus, run_dir=self.te.run_dir,
                       duration=conf.get('duration'))
            return rtapp

        if conf['class'] == 'periodic':
            task_idxs = self._wload_task_idxs(wl_idx, conf['tasks'])
            params = {}
            for idx in task_idxs:
                task = conf['prefix'] + str(idx)
                params[task] = wlgen.Periodic(**conf['params']).get()
            rtapp = wlgen.RTA(self.target,
                        wl_idx, calibration = self.te.calibration())
            rtapp.conf(kind='profile', params=params, loadref=loadref,
                       cpus=cpus, run_dir=self.te.run_dir,
                       duration=conf.get('duration'))
            return rtapp

        if conf['class'] == 'custom':
            rtapp = wlgen.RTA(self.target,
                              wl_idx, calibration = self.te.calibration())
            rtapp.conf(kind='custom',
                    params=conf['json'],
                    duration=conf.get('duration'),
                    loadref=loadref,
                    cpus=cpus, run_dir=self.te.run_dir)
            return rtapp

        raise ValueError('unsupported \'class\' value for [{}] '
                         'RT-App workload specification'
                         .format(wl_idx))

    def _wload_perf_bench(self, wl_idx, wlspec, cpus):
        conf = wlspec['conf']
        self._log.debug('Configuring perf_message...')

        if conf['class'] == 'messaging':
            perf_bench = wlgen.PerfMessaging(self.target, wl_idx)
            perf_bench.conf(**conf['params'])
            return perf_bench

        if conf['class'] == 'pipe':
            perf_bench = wlgen.PerfPipe(self.target, wl_idx)
            perf_bench.conf(**conf['params'])
            return perf_bench

        raise ValueError('unsupported "class" value for [{}] '
                         'perf bench workload specification'
                         .format(wl_idx))

    def _wload_conf(self, wl_idx, wlspec):

        # CPUS: setup execution on CPUs if required by configuration
        cpus = self._wload_cpus(wl_idx, wlspec)

        # CGroup: setup CGroups if requried by configuration
        self._cgroup = self._default_cgroup
        if 'cgroup' in wlspec:
            if 'cgroups' not in self.target.modules:
                raise RuntimeError('Target not supporting CGroups or CGroups '
                                   'not configured for the current test configuration')
            self._cgroup = wlspec['cgroup']

        if wlspec['type'] == 'rt-app':
            return self._wload_rtapp(wl_idx, wlspec, cpus)
        if wlspec['type'] == 'perf_bench':
            return self._wload_perf_bench(wl_idx, wlspec, cpus)


        raise ValueError('unsupported "type" value for [{}] '
                         'workload specification'
                         .format(wl_idx))

    def _wload_init(self, tc, wl_idx):
        tc_idx = tc['tag']

        # Configure the test workload
        wlspec = self._experiments_conf['wloads'][wl_idx]
        wload = self._wload_conf(wl_idx, wlspec)

        # Keep track of platform configuration
        test_dir = '{}/{}:{}:{}'\
            .format(self.te.res_dir, wload.wtype, tc_idx, wl_idx)
        os.makedirs(test_dir)
        self.te.platform_dump(test_dir)

        # Keep track of kernel configuration and version
        config = self.target.config
        with gzip.open(os.path.join(test_dir, 'kernel.config'), 'wb') as fh:
            fh.write(config.text)
        output = self.target.execute('{} uname -a'\
                .format(self.target.busybox))
        with open(os.path.join(test_dir, 'kernel.version'), 'w') as fh:
            fh.write(output)

        return wload, test_dir

    def _wload_run(self, exp_idx, experiment):
        tc = experiment.conf
        wload = experiment.wload
        tc_idx = tc['tag']

        self._print_title('Experiment {}/{}, [{}:{}] {}/{}'\
                .format(exp_idx, self._exp_count,
                        tc_idx, experiment.wload_name,
                        experiment.iteration, self.iterations))

        # Setup local results folder
        self._log.debug('out_dir set to [%s]', experiment.out_dir)
        os.system('mkdir -p ' + experiment.out_dir)

        # FTRACE: start (if a configuration has been provided)
        if self.te.ftrace and self._target_conf_flag(tc, 'ftrace'):
            self._log.info('FTrace events collection enabled')
            self.te.ftrace.start()

        # ENERGY: start sampling
        if self.te.emeter:
            self.te.emeter.reset()

        # WORKLOAD: Run the configured workload
        wload.run(out_dir=experiment.out_dir, cgroup=self._cgroup)

        # ENERGY: collect measurements
        if self.te.emeter:
            self.te.emeter.report(experiment.out_dir)

        # FTRACE: stop and collect measurements
        if self.te.ftrace and self._target_conf_flag(tc, 'ftrace'):
            self.te.ftrace.stop()

            trace_file = experiment.out_dir + '/trace.dat'
            self.te.ftrace.get_trace(trace_file)
            self._log.info('Collected FTrace binary trace:')
            self._log.info('   %s',
                           trace_file.replace(self.te.res_dir, '<res_dir>'))

            stats_file = experiment.out_dir + '/trace_stat.json'
            self.te.ftrace.get_stats(stats_file)
            self._log.info('Collected FTrace function profiling:')
            self._log.info('   %s',
                           stats_file.replace(self.te.res_dir, '<res_dir>'))

        self._print_footer()

################################################################################
# Utility Functions
################################################################################

    def _print_section(self, message):
        self._log.info('')
        self._log.info(FMT_SECTION)
        self._log.info(message)
        self._log.info(FMT_SECTION)

    def _print_header(self, message):
        self._log.info('')
        self._log.info(FMT_HEADER)
        self._log.info(message)

    def _print_title(self, message):
        self._log.info(FMT_TITLE)
        self._log.info(message)

    def _print_footer(self, message=None):
        if message:
            self._log.info(message)
        self._log.info(FMT_FOOTER)


################################################################################
# Globals
################################################################################

# Regular expression for comments
JSON_COMMENTS_RE = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

# Target specific paths
TGT_RUN_DIR = 'run_dir'

# Logging formatters
FMT_SECTION = r'{:#<80}'.format('')
FMT_HEADER  = r'{:=<80}'.format('')
FMT_TITLE   = r'{:~<80}'.format('')
FMT_FOOTER  = r'{:-<80}'.format('')

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
