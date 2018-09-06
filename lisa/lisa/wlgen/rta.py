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

import json
import logging
import os
import re
import sys
from collections import OrderedDict

from workload import Workload

from utilities import Loggable

class RTA(Workload):
    """
    An rt-app workload

    :param json_file: Path to the rt-app json description
    :type json_file: str

    The class constructor only deals with pre-constructed json files.
    For creating rt-app workloads through other means, see :meth:`by_profile`
    and :meth:`by_conf`.

    For more information about rt-app itself, see
    https://github.com/scheduler-tools/rt-app
    """

    required_tools = Workload.required_tools + ['rt-app']

    sched_policies = ['OTHER', 'FIFO', 'RR', 'DEADLINE']

    def __init__(self, te, name, res_dir=None, json_file=None):
        # Don't add code here, use the early/late init methods instead.
        # This lets us factorize some code for the class methods that serve as
        # alternate constructors.
        self._early_init(te, name, res_dir, json_file)
        self._late_init()

    def _early_init(self, te, name, res_dir, json_file):
        """
        Initialize everything that is not related to the contents of the json file
        """
        super(RTA, self).__init__(te, name, res_dir)

        if not json_file:
            json_file = '{}.json'.format(self.name)

        self.local_json = os.path.join(self.res_dir, json_file)
        self.remote_json = self.te.target.path.join(self.run_dir, json_file)

        rta_cmd = self.te.target.which('rt-app')
        if not rta_cmd:
            raise RuntimeError("No rt-app executable found on the target")

        self.command = '{0:s} {1:s} 2>&1'.format(rta_cmd, self.remote_json)

    def _late_init(self, calibration=None, tasks_names=None):
        """
        Complete initialization with a ready json file

        :parameters: Attributes that have been pre-computed and ended up
          in the json file. Passing them can prevent a needless file read.
        """
        if not os.path.exists(self.local_json):
            raise RuntimeError("Could not find {}".format(self.local_json))

        if calibration or not tasks_names:
            with open(self.local_json, "r") as fh:
                desc = json.load(fh)

                if calibration is None:
                    calibration = desc["global"]["calibration"]
                if not tasks_names:
                    tasks_names = desc["tasks"].keys()

        self.calibration = calibration
        self.tasks = tasks_names

        # Move configuration file to target
        self.te.target.push(self.local_json, self.remote_json)

    def run(self, cpus=None, cgroup=None, background=False, as_root=False):
        super(RTA, self).run(cpus, cgroup, background, as_root)

        if background:
            # TODO: handle background case
            return

        self.logger.debug('Pulling logfiles to [%s]...', self.res_dir)
        for task in self.tasks:
            # RT-app appends some number to the logs, so we can't predict the
            # exact filename
            logfile = self.te.target.path.join(self.run_dir, '*{}*.log'.format(task))
            self.te.target.pull(logfile, self.res_dir)

    def _process_calibration(self, calibration):
        """
        Select CPU or pload value for task calibration
        """
        # This is done at init time rather than at run time, because the
        # calibration value lives in the file
        if isinstance(calibration, int):
            pass
        elif isinstance(calibration, basestring):
            calibration = calibration.upper()
        else:
            cpus = range(self.te.target.number_of_cpus)
            target_cpu = cpus[-1]
            if 'bl'in self.te.target.modules:
                candidates = sorted(set(self.te.target.bl.bigs).intersection(cpus))
                if candidates:
                    target_cpu = candidates[0]

            calibration = 'CPU{0:d}'.format(target_cpu)

        return calibration

    @classmethod
    def by_profile(cls, te, name, profile, res_dir=None, default_policy=None,
                   max_duration_s=None, calibration=None):
        """
        Create an rt-app workload using :class:`RTATask` instances

        :param profile: The workload description in a {task_name : :class:`RTATask`}
          shape
        :type profile: dict

        :param default_policy: Default scheduler policy. See :attr:`sched_policies`
        :type default_policy: str

        :param max_duration_s: Maximum duration of the workload. Will be determined
          by the longest running task if not specified.
        :type max_duration_s: int

        :param calibration:
        :type calibration: int or str

        A simple profile workload would be::

            task = Periodic(duty_cycle_pct=5)
            rta = RTA.by_profile(te, "test",  {"foo" : task})
            rta.run()
        """
        self = cls.__new__(cls)
        self._early_init(te, name, res_dir, None)

        # Sanity check for task names
        for task in profile.keys():
            if len(task) > 15:
                # rt-app uses pthread_setname_np(3) which limits the task name
                # to 16 characters including the terminal '\0'.
                raise ValueError(
                    'Task name "{}" too long, please configure your tasks '
                    'with names shorter than 16 characters'.format(task))

        rta_profile = {
            'tasks': {},
            'global': {}
        }

        calibration = self._process_calibration(calibration)

        global_conf = {
            'default_policy': 'SCHED_OTHER',
            'duration': -1 if not max_duration_s else max_duration_s,
            'calibration': calibration,
            'logdir': self.run_dir,
        }

        if max_duration_s:
            self.logger.warn('Limiting workload duration to %d [s]', max_duration_s)

        if default_policy:
            if default_policy in self.sched_policies:
                global_conf['default_policy'] = 'SCHED_{}'.format(default_policy)
            else:
                raise ValueError('scheduling class {} not supported'.format(default_policy))

        self.logger.info('Calibration value: %s', global_conf['calibration'])
        self.logger.info('Default policy: %s', global_conf['default_policy'])

        rta_profile['global'] = global_conf

        # Setup tasks parameters
        for tid, task in profile.items():
            task_conf = {}

            if not task.sched_policy:
                task_conf['policy'] = global_conf['default_policy']
                sched_descr = 'sched: using default policy'
            else:
                task_conf['policy'] = 'SCHED_{}'.format(task.sched_policy)
                if task.priority is not None:
                    task_conf['prio'] = task.priority
                sched_descr = 'sched: {0:s}'.format(task.sched_policy)


            self.logger.info('------------------------')
            self.logger.info('task [%s], %s', tid, sched_descr)

            task_conf['delay'] = int(task.delay_s * 1e6)
            self.logger.info(' | start delay: %.6f [s]', task.delay_s)

            task_conf['loop'] = task.loops
            self.logger.info(' | loops count: %d', task.loops)

            task_conf['phases'] = OrderedDict()
            rta_profile['tasks'][tid] = task_conf

            pid = 1
            for phase in task.phases:
                phase_name = 'p{}'.format(str(pid).zfill(6))

                self.logger.info(' + phase_%06d', pid)
                rta_profile['tasks'][tid]['phases'][phase_name] = phase.get_rtapp_repr(tid)

                pid += 1

        # Generate JSON configuration on local file
        with open(self.local_json, 'w') as outfile:
            json.dump(rta_profile, outfile, indent=4, separators=(',', ': '))

        self._late_init(calibration=calibration, tasks_names=profile.keys())
        return self

    @classmethod
    def process_template(cls, template, output, duration=None, pload=None,
                         log_dir=None, work_dir=None):
        """
        :param output: An open <thing>
        :type output: yolo (IOStream?)
        """
        # pload can either be a string like "CPU1" or an integer, if the
        # former we need to quote it.
        if not isinstance(pload, int):
            pload = '"{}"'.format(pload)

        replacements = {
            '__DURATION__' : duration,
            '__PVALUE__'   : pload,
            '__LOGDIR__'   : log_dir,
            '__WORKDIR__'  : work_dir,
        }

        for line in template:
            for token, replacement in replacements.iteritems():
                if token not in line:
                    continue

                if replacement is None:
                    raise RuntimeError("No replacement value given for {}".format(token))

                line = line.replace(token, str(replacement))

            output.write(line)

    @classmethod
    def by_conf(cls, te, name, conf, res_dir=None, max_duration_s=None,
                calibration=None):

        self = cls.__new__(cls)
        self._early_init(te, name, res_dir, None)

        str_conf = json.dumps(conf, indent=4, separators=(',', ': '),
                              sort_keys=True).splitlines(True)

        calibration = self._process_calibration(calibration)

        with open(self.local_json, 'w') as fh:
            self.process_template(str_conf, fh, max_duration_s, calibration,
                                  self.run_dir, self.run_dir)

        tasks_names = [tid for tid in conf['tasks']]
        self._late_init(calibration=calibration, tasks_names=tasks_names)

        return self

    @classmethod
    def _calibrate(cls, te, res_dir):
        pload_regexp = re.compile(r'pLoad = ([0-9]+)ns')
        pload = {}

        log = logging.getLogger(cls.__name__)

        # Create calibration task
        max_rtprio = int(te.target.execute('ulimit -Hr').split('\r')[0])
        log.debug('Max RT prio: %d', max_rtprio)

        if max_rtprio > 10:
            max_rtprio = 10

        if not res_dir:
            res_dir = te.get_res_dir("rta_calib")

        for cpu in te.target.list_online_cpus():
            log.info('CPU%d calibration...', cpu)

            # RT-app will run a calibration for us, so we just need to
            # run a dummy task and read the output
            calib_task = Periodic(duty_cycle_pct=100, duration_s=0.001, period_ms=1)
            rta = RTA.by_profile(te, name="rta_calib_cpu{}".format(cpu),
                                 res_dir=res_dir, profile={'task1': calib_task},
                                 calibration="CPU{}".format(cpu))
            rta.run(as_root=True)

            for line in rta.output.split('\n'):
                pload_match = re.search(pload_regexp, line)
                if pload_match is None:
                    continue
                pload[cpu] = int(pload_match.group(1))
                log.debug('>>> cpu%d: %d', cpu, pload[cpu])

        log.info('Target RT-App calibration:')
        log.info("{" + ", ".join('"%r": %r' % (key, pload[key])
                                 for key in pload) + "}")

        if 'sched' not in te.target.modules:
            return pload

        # Sanity check calibration values for asymmetric systems
        cpu_capacities = te.target.sched.get_capacities()
        capa_pload = {capacity : sys.maxint for capacity in cpu_capacities.values()}

        # Find the max pload per capacity level
        # for capacity, index in enumerate(sorted_capacities):
        for cpu, capacity in cpu_capacities.items():
            capa_pload[capacity] = max(capa_pload[capacity], pload[cpu])

        sorted_capas = sorted(capa_pload.keys())

        # Make sure pload decreases with capacity
        for index, capa in enumerate(sorted_capas[1:]):
            if capa_pload[capa] > capa_pload[sorted_capas[index-1]]:
                log.warning('Calibration values reports big cores less '
                            'capable than LITTLE cores')
                raise RuntimeError('Calibration failed: try again or file a bug')

        return pload

    @classmethod
    def get_cpu_calibrations(cls, te, res_dir=None):
        """
        Get the rt-ap calibration value for all CPUs.

        :param target: Devlib target to run calibration on.
        :returns: Dict mapping CPU numbers to rt-app calibration values.
        """

        if 'cpufreq' not in te.target.modules:
            logging.getLogger(cls.__name__).warning(
                'cpufreq module not loaded, skipping setting frequency to max')
            return cls._calibrate(te, res_dir)

        with te.target.cpufreq.use_governor('performance'):
            return cls._calibrate(te, res_dir)

class Phase(Loggable):
    """
    Descriptor for an rt-app load phase

    :param duration_s: the phase duration in [s].
    :type duration_s: int

    :param period_ms: the phase period in [ms].
    :type period_ms: int

    :param duty_cycle_pct: the generated load in [%].
    :type duty_cycle_pct: int

    :param cpus: the CPUs on which task execution is restricted during this phase.
    :type cpus: list(int)

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

    def get_rtapp_repr(self, task_name):
        """
        Get a dictionnary representation of the phase as expected by rt-app

        :param task_name: Name of the phase's task (needed for timers)
        :type task_name: str

        :returns: OrderedDict
        """
        phase = OrderedDict()
        # Convert time parameters to integer [us] units
        duration = int(self.duration_s * 1e6)

        # A duty-cycle of 0[%] translates to a 'sleep' phase
        if self.duty_cycle_pct == 0:
            self.logger.info(' | sleep %.6f [s]', duration/1e6)

            phase['loop'] = 1
            phase['sleep'] = duration

        # A duty-cycle of 100[%] translates to a 'run-only' phase
        elif self.duty_cycle_pct == 100:
            self.logger.info(' | batch %.6f [s]', duration/1e6)

            phase['loop'] = 1
            phase['run'] = duration
            if self.barrier_after:
                phase['barrier'] = self.barrier_after

        # A certain number of loops is requires to generate the
        # proper load
        else:
            period = int(self.period_ms * 1e3)

            cloops = -1
            if duration >= 0:
                cloops = int(duration / period)

            sleep_time = period * (100 - self.duty_cycle_pct) / 100
            running_time = period - sleep_time

            self.logger.info(' | duration %.6f [s] (%d loops)',
                             duration/1e6, cloops)
            self.logger.info(' |  period   %6d [us], duty_cycle %3d %%',
                             period, self.duty_cycle_pct)
            self.logger.info(' |  run_time %6d [us], sleep_time %6d [us]',
                             running_time, sleep_time)

            phase['loop'] = cloops
            phase['run'] = running_time
            phase['timer'] = {'ref': task_name, 'period': period}

        if self.cpus is not None:
            phase['cpus'] = self.cpus

        return phase

class RTATask(object):
    """
    Base class for conveniently constructing params to :meth:`RTA.conf`

    This class represents an rt-app task which may contain multiple :class:`Phase`.
    It implements ``__add__`` so that using ``+`` on two tasks concatenates their
    phases. For example ``Ramp() + Periodic()`` would yield an ``RTATask`` that
    executes the default phases for :class:`Ramp` followed by the default phases for
    :class:`Periodic`.
    """

    def __init__(self, delay_s=0, loops=1, sched_policy=None, priority=None):
        self.delay_s = delay_s
        self.loops = loops

        if isinstance(sched_policy, basestring):
            sched_policy = sched_policy.upper()

            if sched_policy not in RTA.sched_policies:
                raise ValueError('scheduling class {} not supported'.format(sched_policy))

        self.sched_policy = sched_policy
        self.priority = priority
        self.phases = []

    def __add__(self, next_phases):
        if next_phases.delay_s:
            # This won't work, because rt-app's "delay" field is per-task and
            # not per-phase. We might be able to implement it by adding a
            # "sleep" event here, but let's not bother unless such a need
            # arises.
            raise ValueError("Can't compose rt-app tasks "
                             "when the second has nonzero 'delay_s'")

        self.phases.extend(next_phases.phases)
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

    :param sched_policy: the scheduler configuration for this task.
    :type sched_policy: dict

    :param cpus: the list of CPUs on which task can run.
                .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=0, end_pct=100, delta_pct=10, time_s=1,
                 period_ms=100, delay_s=0, loops=1, sched_policy=None,
                 priority=None, cpus=None):
        super(Ramp, self).__init__(delay_s, loops, sched_policy, priority)

        if start_pct not in range(0, 101) or end_pct not in range(0, 101):
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

        self.phases = phases

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
                 delay_s=0, loops=1, sched_policy=None, priority=None, cpus=None):
        delta_pct = abs(end_pct - start_pct)
        super(Step, self).__init__(start_pct, end_pct, delta_pct, time_s,
                                   period_ms, delay_s, loops, sched_policy,
                                   priority, cpus)

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
                 delay_s=0, loops=1, sched_policy=None, priority=None, cpus=None):
        super(Pulse, self).__init__(delay_s, loops, sched_policy, priority)

        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        if end_pct not in range(0, 101) or start_pct not in range(0, 101):
            raise ValueError('end_pct and start_pct must be in [0..100] range')
        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        phases = []
        for load in [start_pct, end_pct]:
            if load == 0:
                continue
            phase = Phase(time_s, period_ms, load, cpus)
            phases.append(phase)

        self.phases = phases

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
                 delay_s=0, sched_policy=None, priority=None, cpus=None):
        super(Periodic, self).__init__(duty_cycle_pct, 0, duration_s,
                                       period_ms, delay_s, 1, sched_policy,
                                       priority, cpus)

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
    def __init__(self, barrier, time_s=1, delay_s=0, loops=1, sched_policy=None,
                 priority=None, cpus=None):
        super(RunAndSync, self).__init__(delay_s, loops, sched_policy, priority)

        # This should translate into a phase containing a 'run' event and a
        # 'barrier' event
        self.phases = [Phase(time_s, None, 100, cpus, barrier_after=barrier)]
