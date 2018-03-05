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

import fileinput
import json
import os
import re

from collections import OrderedDict
from wlgen import Workload
from devlib.utils.misc import ranges_to_list

import logging

class Phase(object):
    """
    Descriptor for an RT-App load phase

    :param duration_s: the phase duration in [s].
    :type duration_s: int

    :param period_ms: the phase period in [ms].
    :type period_ms: int

    :param duty_cycle_pct: the generated load in [%].
    :type duty_cycle_pct: int

    :param cpus: the list of cpus on which task execution is restricted during
                 this phase.
    :type cpus: [int] or int

    :param barrier_after: if provided, the name of the barrier to sync against
                          when reaching the end of this phase. Currently only
                          supported when duty_cycle_pct=100
    :type barrier_after: str
    """
    def __init__(self, duration_s, period_ms, duty_cycle_pct, cpus=None, barrier_after=None):
        if barrier_after and duty_cycle_pct != 100:
            # This could be implemented but currently don't foresee any use.
            raise ValueError('Barriers only supported when duty_cycle_pct=100')

        self.duration_s = duration_s
        self.period_ms = period_ms
        self.duty_cycle_pct = duty_cycle_pct
        self.cpus = cpus
        self.barrier_after = barrier_after

class RTA(Workload):
    """
    Class for creating RT-App workloads
    """

    def __init__(self,
                 target,
                 name,
                 calibration=None):
        """
        :param target: Devlib target to run workload on.
        :param name: Human-readable name for the workload.
        :param calibration: CPU calibration specification. Can be obtained from
                            :meth:`calibrate`.
        """

        # Setup logging
        self._log = logging.getLogger('RTApp')

        # rt-app calibration
        self.pload = calibration

        # TODO: Assume rt-app is pre-installed on target
        # self.target.setup('rt-app')

        super(RTA, self).__init__(target, name)

        # rt-app executor
        self.wtype = 'rtapp'
        self.executor = 'rt-app'

        # Default initialization
        self.json = None
        self.rta_profile = None
        self.loadref = None
        self.rta_cmd  = None
        self.rta_conf = None
        self.test_label = None

        # Setup RTA callbacks
        self.setCallback('postrun', self.__postrun)

    @staticmethod
    def calibrate(target):
        """
        Calibrate RT-App on each CPU in the system

        :param target: Devlib target to run calibration on.
        :returns: Dict mapping CPU numbers to RT-App calibration values.
        """
        pload_regexp = re.compile(r'pLoad = ([0-9]+)ns')
        pload = {}

        # Setup logging
        log = logging.getLogger('RTApp')

        # Save previous governors
        old_governors = {}
        for domain in target.cpufreq.iter_domains():
            cpu = domain[0]
            governor = target.cpufreq.get_governor(cpu)
            tunables = target.cpufreq.get_governor_tunables(cpu)
            old_governors[cpu] = governor, tunables

        target.cpufreq.set_all_governors('performance')

        # Create calibration task
        max_rtprio = int(target.execute('ulimit -Hr').split('\r')[0])
        log.debug('Max RT prio: %d', max_rtprio)
        if max_rtprio > 10:
            max_rtprio = 10

        calib_task = Periodic(period_ms=100,
                              duty_cycle_pct=50,
                              duration_s=1,
                              sched={
                                  'policy': 'FIFO',
                                  'prio': max_rtprio
                              }).get()
        rta = RTA(target, 'rta_calib')

        for cpu in target.list_online_cpus():

            log.info('CPU%d calibration...', cpu)

            rta.conf(kind='profile', params={'task1': calib_task}, cpus=[cpu])
            rta.run(as_root=True)

            for line in rta.getOutput().split('\n'):
                pload_match = re.search(pload_regexp, line)
                if pload_match is None:
                    continue
                pload[cpu] = int(pload_match.group(1))
                log.debug('>>> cpu%d: %d', cpu, pload[cpu])

        # Restore previous governors
        #   Setting a governor & tunables for a cpu will set them for all cpus
        #   in the same clock domain, so only restoring them for one cpu
        #   per domain is enough to restore them all.
        for cpu, (governor, tunables) in old_governors.iteritems():
            target.cpufreq.set_governor(cpu, governor)
            target.cpufreq.set_governor_tunables(cpu, **tunables)

        log.info('Target RT-App calibration:')
        log.info("{" + ", ".join('"%r": %r' % (key, pload[key])
                                 for key in pload) + "}")

        # Sanity check calibration values for big.LITTLE systems
        if 'bl' in target.modules:
            bcpu = target.bl.bigs_online[0]
            lcpu = target.bl.littles_online[0]
            if pload[bcpu] > pload[lcpu]:
                log.warning('Calibration values reports big cores less '
                            'capable than LITTLE cores')
                raise RuntimeError('Calibration failed: try again or file a bug')
            bigs_speedup = ((float(pload[lcpu]) / pload[bcpu]) - 1) * 100
            log.info('big cores are ~%.0f%% more capable than LITTLE cores',
                     bigs_speedup)

        return pload

    def __postrun(self, params):
        destdir = params['destdir']
        if destdir is None:
            return
        self._log.debug('Pulling logfiles to [%s]...', destdir)
        for task in self.tasks.keys():
            logfile = self.target.path.join(self.run_dir,
                                            '*{}*.log'.format(task))
            self.target.pull(logfile, destdir)
        self._log.debug('Pulling JSON to [%s]...', destdir)
        self.target.pull(self.target.path.join(self.run_dir, self.json),
                         destdir)
        logfile = self.target.path.join(destdir, 'output.log')
        self._log.debug('Saving output on [%s]...', logfile)
        with open(logfile, 'w') as ofile:
            for line in self.output['executor'].split('\n'):
                ofile.write(line+'\n')

    def getCalibrationConf(self):
        # Select CPU for task calibration, which is the first little
        # of big depending on the loadref tag
        if self.pload is not None:
            if self.loadref and self.loadref.upper() == 'LITTLE':
                return max(self.pload.values())
            else:
                return min(self.pload.values())
        else:
            cpus = self.cpus or range(self.target.number_of_cpus)

            target_cpu = cpus[-1]
            if 'bl'in self.target.modules:
                cluster = self.target.bl.bigs
                candidates = sorted(set(self.target.bl.bigs).intersection(cpus))
                if candidates:
                    target_cpu = candidates[0]

            return 'CPU{0:d}'.format(target_cpu)

    def _confCustom(self):

        rtapp_conf = self.params['custom']

        self._log.info('Loading custom configuration:')
        self._log.info('   %s', rtapp_conf)
        self.json = '{0:s}_{1:02d}.json'.format(self.name, self.exc_id)
        ofile = open(self.json, 'w')

        calibration = self.getCalibrationConf()
        # Calibration can either be a string like "CPU1" or an integer, if the
        # former we need to quote it.
        if type(calibration) != int:
            calibration = '"{}"'.format(calibration)

        replacements = {
            '__DURATION__' : str(self.duration),
            '__PVALUE__'   : str(calibration),
            '__LOGDIR__'   : str(self.run_dir),
            '__WORKDIR__'  : '"'+self.target.working_directory+'"',
        }

        # Check for inline config
        if not isinstance(self.params['custom'], basestring):
            if isinstance(self.params['custom'], dict):
                # Inline config present, turn it into a file repr
                tmp_json = json.dumps(self.params['custom'],
                    indent=4, separators=(',', ': '), sort_keys=True)
                ifile = tmp_json.splitlines(True)
            else:
                raise ValueError("Value specified for 'params'  can only be "
                                 "a filename or an embedded dictionary")
        else:
            # We assume we are being passed a filename instead of a dict
            ifile = open(rtapp_conf, 'r')

        for line in ifile:
            if '__DURATION__' in line and self.duration is None:
                raise ValueError('Workload duration not specified')
            for src, target in replacements.iteritems():
                line = line.replace(src, target)
            ofile.write(line)

        if isinstance(ifile, file):
            ifile.close()
        ofile.close()

        with open(self.json) as f:
            conf = json.load(f)
        for tid in conf['tasks']:
            self.tasks[tid] = {'pid': -1}

        return self.json

    def _confProfile(self):

        # Sanity check for task names
        for task in self.params['profile'].keys():
            if len(task) > 15:
                # rt-app uses pthread_setname_np(3) which limits the task name
                # to 16 characters including the terminal '\0'.
                msg = ('Task name "{}" too long, please configure your tasks '
                       'with names shorter than 16 characters').format(task)
                raise ValueError(msg)

        # Task configuration
        self.rta_profile = {
            'tasks': {},
            'global': {}
        }

        # Initialize global configuration
        global_conf = {
                'default_policy': 'SCHED_OTHER',
                'duration': -1,
                'calibration': self.getCalibrationConf(),
                'logdir': self.run_dir,
            }

        if self.duration is not None:
            global_conf['duration'] = self.duration
            self._log.warn('Limiting workload duration to %d [s]',
                           global_conf['duration'])
        else:
            self._log.info('Workload duration defined by longest task')

        # Setup default scheduling class
        if 'policy' in self.sched:
            policy = self.sched['policy'].upper()
            if policy not in ['OTHER', 'FIFO', 'RR', 'DEADLINE']:
                raise ValueError('scheduling class {} not supported'\
                        .format(policy))
            global_conf['default_policy'] = 'SCHED_' + self.sched['policy']

        self._log.info('Default policy: %s', global_conf['default_policy'])

        # Setup global configuration
        self.rta_profile['global'] = global_conf

        # Setup tasks parameters
        for tid in sorted(self.params['profile'].keys()):
            task = self.params['profile'][tid]

            # Initialize task configuration
            task_conf = {}

            if 'sched' not in task:
                policy = 'DEFAULT'
            else:
                policy = task['sched']['policy'].upper()
            if policy == 'DEFAULT':
                task_conf['policy'] = global_conf['default_policy']
                sched_descr = 'sched: using default policy'
            elif policy not in ['OTHER', 'FIFO', 'RR', 'DEADLINE']:
                raise ValueError('scheduling class {} not supported'\
                        .format(task['sclass']))
            else:
                task_conf.update(task['sched'])
                task_conf['policy'] = 'SCHED_' + policy
                sched_descr = 'sched: {0:s}'.format(task['sched'])

            # Initialize task phases
            task_conf['phases'] = OrderedDict()

            self._log.info('------------------------')
            self._log.info('task [%s], %s', tid, sched_descr)

            if 'delay' in task.keys():
                if task['delay'] > 0:
                    task_conf['delay'] = int(task['delay'] * 1e6)
                    self._log.info(' | start delay: %.6f [s]',
                            task['delay'])

            if 'loops' not in task.keys():
                task['loops'] = 1
            task_conf['loop'] = task['loops']
            self._log.info(' | loops count: %d', task['loops'])

            # Setup task configuration
            self.rta_profile['tasks'][tid] = task_conf

            # Getting task phase descriptor
            pid=1
            for phase in task['phases']:

                # Convert time parameters to integer [us] units
                duration = int(phase.duration_s * 1e6)

                task_phase = OrderedDict()

                # A duty-cycle of 0[%] translates on a 'sleep' phase
                if phase.duty_cycle_pct == 0:

                    self._log.info(' + phase_%06d: sleep %.6f [s]',
                                   pid, duration/1e6)

                    task_phase['loop'] = 1
                    task_phase['sleep'] = duration

                # A duty-cycle of 100[%] translates on a 'run-only' phase
                elif phase.duty_cycle_pct == 100:

                    self._log.info(' + phase_%06d: batch %.6f [s]',
                                   pid, duration/1e6)

                    task_phase['loop'] = 1
                    task_phase['run'] = duration
                    if phase.barrier_after:
                        task_phase['barrier'] = phase.barrier_after

                # A certain number of loops is requires to generate the
                # proper load
                else:
                    period = int(phase.period_ms * 1e3)

                    cloops = -1
                    if duration >= 0:
                        cloops = int(duration / period)

                    sleep_time = period * (100 - phase.duty_cycle_pct) / 100
                    running_time = period - sleep_time

                    self._log.info('+ phase_%06d: duration %.6f [s] (%d loops)',
                                   pid, duration/1e6, cloops)
                    self._log.info('|  period   %6d [us], duty_cycle %3d %%',
                                   period, phase.duty_cycle_pct)
                    self._log.info('|  run_time %6d [us], sleep_time %6d [us]',
                                   running_time, sleep_time)

                    task_phase['loop'] = cloops
                    task_phase['run'] = running_time
                    task_phase['timer'] = {'ref': tid, 'period': period}

                self.rta_profile['tasks'][tid]['phases']\
                    ['p'+str(pid).zfill(6)] = task_phase

                if phase.cpus is not None:
                    if isinstance(phase.cpus, str):
                        task_phase['cpus'] = ranges_to_list(phase.cpus)
                    elif isinstance(phase.cpus, list):
                        task_phase['cpus'] = phase.cpus
                    elif isinstance(phase.cpus, int):
                        task_phase['cpus'] = [phase.cpus]
                    else:
                        raise ValueError('phases cpus must be a list or string \
                                          or int')
                    self._log.info('|  CPUs affinity: {}'.format(phase.cpus))

                pid+=1

            # Append task name to the list of this workload tasks
            self.tasks[tid] = {'pid': -1}

        # Generate JSON configuration on local file
        self.json = '{0:s}_{1:02d}.json'.format(self.name, self.exc_id)
        with open(self.json, 'w') as outfile:
            json.dump(self.rta_profile, outfile,
                      indent=4, separators=(',', ': '))

        return self.json

    def conf(self,
             kind,
             params,
             duration=None,
             cpus=None,
             sched=None,
             run_dir=None,
             exc_id=0,
             loadref='big'):
        """
        Configure a workload of a specified kind.

        The rt-app based workload allows to define different classes of
        workloads. The classes supported so far are detailed hereafter.

        Custom workloads
          When 'kind' is 'custom' the tasks generated by this workload can be:
          * Defined in a provided rt-app JSON configuration file. In this case
            the 'params' parameter must be used to specify the complete path of
            the rt-app JSON configuration file to use.
          * Defined in a dictionnary that matches the format of an rt-app JSON
            configuration file. This can be useful when you need to create a
            more complex scenario not covered by the workload generator, as
            you can create it with a dict and "feed" it directly to rtapp.

        Profile based workloads
          When ``kind`` is "profile", ``params`` is a dictionary mapping task
          names to task specifications. The easiest way to create these task
          specifications using :meth:`RTATask.get`.

          For example, the following configures an RTA workload with a single
          task, named 't1', using the default parameters for a Periodic RTATask:

          ::

            wl = RTA(...)
            wl.conf(kind='profile', params={'t1': Periodic().get()})

        :param kind: Either 'custom' or 'profile' - see above.
        :param params: RT-App parameters - see above.
        :param duration: Maximum duration of the workload in seconds. Any
                         remaining tasks are killed by rt-app when this time has
                         elapsed.
        :param cpus: CPUs to restrict this workload to, using ``taskset``.
                    .. note:: if not specified, it can run on all CPUs
        :type cpus: list(int)

        :param sched: Global RT-App scheduler configuration. Dict with fields:

          policy
            The default scheduler policy. Choose from 'OTHER', 'FIFO', 'RR',
            and 'DEADLINE'.

        :param run_dir: Target dir to store output and config files in.

        .. TODO: document or remove loadref
        """

        if not sched:
            sched = {'policy' : 'OTHER'}

        super(RTA, self).conf(kind, params, duration,
                cpus, sched, run_dir, exc_id)

        self.loadref = loadref

        # Setup class-specific configuration
        if kind == 'custom':
            self._confCustom()
        elif kind == 'profile':
            self._confProfile()

        # Move configuration file to target
        self.target.push(self.json, self.run_dir)

        self.rta_cmd  = self.target.executables_directory + '/rt-app'
        self.rta_conf = self.run_dir + '/' + self.json
        self.command = '{0:s} {1:s} 2>&1'.format(self.rta_cmd, self.rta_conf)

        # Set and return the test label
        self.test_label = '{0:s}_{1:02d}'.format(self.name, self.exc_id)
        return self.test_label

class RTATask(object):
    """
    Base class for conveniently constructing params to :meth:`RTA.conf`

    This class represents an RT-App task which may contain multiple phases. It
    implements ``__add__`` so that using ``+`` on two tasks concatenates their
    phases. For example ``Ramp() + Periodic()`` would yield an ``RTATask`` that
    executes the default phases for ``Ramp`` followed by the default phases for
    ``Periodic``.
    """

    def __init__(self, delay_s=0, loops=1, sched=None):
        self._task = {}

        self._task['sched'] = sched or {'policy' : 'DEFAULT'}
        self._task['delay'] = delay_s
        self._task['loops'] = loops

    def get(self):
        """
        Return a dict that can be passed as an element of the ``params`` field
        to :meth:`RTA.conf`.
        """
        return self._task

    def __add__(self, next_phases):
        if next_phases._task.get('delay', 0):
            # This won't work, because rt-app's "delay" field is per-task and
            # not per-phase. We might be able to implement it by adding a
            # "sleep" event here, but let's not bother unless such a need
            # arises.
            raise ValueError("Can't compose rt-app tasks "
                             "when the second has nonzero 'delay_s'")

        self._task['phases'].extend(next_phases._task['phases'])
        return self


class Ramp(RTATask):
    """
    Configure a ramp load.

    This class defines a task which load is a ramp with a configured number
    of steps according to the input parameters.

    :param start_pct: the initial load percentage.
    :param end_pct: the final load percentage.
    :param delta_pct: the load increase/decrease at each step, in percentage
                      points.
    :param time_s: the duration in seconds of each load step.
    :param period_ms: the period used to define the load in [ms].
    :param delay_s: the delay in seconds before ramp start.
    :param loops: number of time to repeat the ramp, with the specified delay in
                  between.

    :param sched: the scheduler configuration for this task.
    :type sched: dict

    :param cpus: the list of CPUs on which task can run.
                .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=0, end_pct=100, delta_pct=10, time_s=1,
                 period_ms=100, delay_s=0, loops=1, sched=None, cpus=None):
        super(Ramp, self).__init__(delay_s, loops, sched)

        if start_pct not in range(0,101) or end_pct not in range(0,101):
            raise ValueError('start_pct and end_pct must be in [0..100] range')

        if start_pct >= end_pct:
            if delta_pct > 0:
                delta_pct = -delta_pct
            delta_adj = -1
        if start_pct <= end_pct:
            if delta_pct < 0:
                delta_pct = -delta_pct
            delta_adj = +1

        phases = []
        steps = range(start_pct, end_pct+delta_adj, delta_pct)
        for load in steps:
            if load == 0:
                phase = Phase(time_s, 0, 0, cpus)
            else:
                phase = Phase(time_s, period_ms, load, cpus)
            phases.append(phase)

        self._task['phases'] = phases

class Step(Ramp):
    """
    Configure a step load.

    This class defines a task which load is a step with a configured initial and
    final load. Using the ``loops`` param, this can be used to create a workload
    that alternates between two load values.

    :param start_pct: the initial load percentage.
    :param end_pct: the final load percentage.
    :param time_s: the duration in seconds of each load step.
    :param period_ms: the period used to define the load in [ms].
    :param delay_s: the delay in seconds before ramp start.
    :param loops: number of time to repeat the step, with the specified delay in
                  between.

    :param sched: the scheduler configuration for this task.
    :type sched: dict

    :param cpus: the list of CPUs on which task can run.
                .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=0, end_pct=100, time_s=1, period_ms=100,
                 delay_s=0, loops=1, sched=None, cpus=None):
        delta_pct = abs(end_pct - start_pct)
        super(Step, self).__init__(start_pct, end_pct, delta_pct, time_s,
                                   period_ms, delay_s, loops, sched, cpus)

class Pulse(RTATask):
    """
    Configure a pulse load.

    This class defines a task which load is a pulse with a configured
    initial and final load.

    The main difference with the 'step' class is that a pulse workload is
    by definition a 'step down', i.e. the workload switch from an finial
    load to a final one which is always lower than the initial one.
    Moreover, a pulse load does not generate a sleep phase in case of 0[%]
    load, i.e. the task ends as soon as the non null initial load has
    completed.

    :param start_pct: the initial load percentage.
    :param end_pct: the final load percentage. Must be lower than ``start_pct``
                    value. If end_pct is 0, the task end after the ``start_pct``
                    period has completed.
    :param time_s: the duration in seconds of each load step.
    :param period_ms: the period used to define the load in [ms].
    :param delay_s: the delay in seconds before ramp start.
    :param loops: number of time to repeat the pulse, with the specified delay
                  in between.

    :param sched: the scheduler configuration for this task.
    :type sched: dict

    :param cpus: the list of CPUs on which task can run
                .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=100, end_pct=0, time_s=1, period_ms=100,
                 delay_s=0, loops=1, sched=None, cpus=None):
        super(Pulse, self).__init__(delay_s, loops, sched)

        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        if end_pct not in range(0,101) or start_pct not in range(0,101):
            raise ValueError('end_pct and start_pct must be in [0..100] range')
        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        phases = []
        for load in [start_pct, end_pct]:
            if load == 0:
                continue
            phase = Phase(time_s, period_ms, load, cpus)
            phases.append(phase)

        self._task['phases'] = phases


class Periodic(Pulse):
    """
    Configure a periodic load. This is the simplest type of RTA task.

    This class defines a task which load is periodic with a configured
    period and duty-cycle.

    :param duty_cycle_pct: the load percentage.
    :param duration_s: the total duration in seconds of the task.
    :param period_ms: the period used to define the load in milliseconds.
    :param delay_s: the delay in seconds before starting the periodic phase.

    :param sched: the scheduler configuration for this task.
    :type sched: dict

    :param cpus: the list of CPUs on which task can run.
                .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, duty_cycle_pct=50, duration_s=1, period_ms=100,
                 delay_s=0, sched=None, cpus=None):
        super(Periodic, self).__init__(duty_cycle_pct, 0, duration_s,
                                       period_ms, delay_s, 1, sched, cpus)

class RunAndSync(RTATask):
    """
    Configure a task that runs 100% then waits on a barrier

    :param barrier: name of barrier to wait for. Sleeps until any other tasks
                    that refer to this barrier have reached the barrier too.
    :type barrier: str

    :param time_s: time to run for

    :param delay_s: the delay in seconds before starting.

    :param sched: the scheduler configuration for this task.
    :type sched: dict

    :param cpus: the list of CPUs on which task can run.
                .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)

    """
    def __init__(self, barrier, time_s=1,
                 delay_s=0, loops=1, sched=None, cpus=None):
        super(RunAndSync, self).__init__(delay_s, loops, sched)

        # This should translate into a phase containing a 'run' event and a
        # 'barrier' event
        self._task['phases'] = [Phase(time_s, None, 100, cpus,
                                      barrier_after=barrier)]
