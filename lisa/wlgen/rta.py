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
from shlex import quote
import copy
import itertools

from lisa.wlgen.workload import Workload
from lisa.utils import Loggable, ArtifactPath, TASK_COMM_MAX_LEN, groupby, nullcontext
from lisa.pelt import PELT_SCALE

class CalibrationError(RuntimeError):
    """
    Exception raised when the ``rt-app`` calibration is not consistent with the
    CPU capacities in a way or another.
    """
    pass

class RTA(Workload):
    """
    An rt-app workload

    :param json_file: Path to the rt-app json description
    :type json_file: str

    The class constructor only deals with pre-constructed json files.
    For creating rt-app workloads through other means, see :meth:`by_profile`
    and :meth:`by_str`.

    For more information about rt-app itself, see
    https://github.com/scheduler-tools/rt-app
    """

    required_tools = Workload.required_tools + ['rt-app']

    sched_policies = ['OTHER', 'FIFO', 'RR', 'DEADLINE']

    ALLOWED_TASK_NAME_REGEX = r'^[a-zA-Z0-9_]+$'

    def __init__(self, target, name, res_dir=None, json_file=None):
        # Don't add code here, use the early/late init methods instead.
        # This lets us factorize some code for the class methods that serve as
        # alternate constructors.

        self._early_init(target, name, res_dir, json_file)
        self._late_init()

    def _early_init(self, target, name, res_dir, json_file, log_stats=False, trace_events=None):
        """
        Initialize everything that is not related to the contents of the json file
        """
        super().__init__(target, name, res_dir)
        self.log_stats = log_stats
        self.trace_events = trace_events or []

        if not json_file:
            json_file = '{}.json'.format(self.name)

        self.local_json = ArtifactPath.join(self.res_dir, json_file)
        self.remote_json = self.target.path.join(self.run_dir, json_file)

        rta_cmd = self.target.which('rt-app')
        if not rta_cmd:
            raise RuntimeError("No rt-app executable found on the target")

        self.command = '{0:s} {1:s} 2>&1'.format(quote(rta_cmd), quote(self.remote_json))

    def _late_init(self, calibration=None, tasks_names=None):
        """
        Complete initialization with a ready json file

        :parameters: Attributes that have been pre-computed and ended up
          in the json file. Passing them can prevent a needless file read.
        """
        if calibration or not tasks_names:
            with open(self.local_json, "r") as fh:
                desc = json.load(fh)

                if calibration is None:
                    calibration = desc["global"]["calibration"]
                if not tasks_names:
                    tasks_names = list(desc["tasks"].keys())

        self.calibration = calibration
        self.tasks = sorted(tasks_names)

        # Move configuration file to target
        self.target.push(self.local_json, self.remote_json)

    def run(self, cpus=None, cgroup=None, background=False, as_root=False):
        super().run(cpus, cgroup, background, as_root)
        logger = self.get_logger()

        if background:
            # TODO: handle background case
            return

        if not self.log_stats:
            return
        logger.debug('Pulling logfiles to [%s]...', self.res_dir)
        for task in self.tasks:
            # RT-app appends some number to the logs, so we can't predict the
            # exact filename
            logfile = self.target.path.join(self.run_dir, '*{}*.log'.format(task))
            self.target.pull(logfile, self.res_dir)

    def _process_calibration(self, calibration):
        """
        Select CPU or pload value for task calibration
        """
        # This is done at init time rather than at run time, because the
        # calibration value lives in the file
        if isinstance(calibration, int):
            pass
        elif isinstance(calibration, str):
            calibration = calibration.upper()
        elif calibration is None:
            calib_map = self.target.plat_info['rtapp']['calib']
            calibration = min(calib_map.values())
        else:
            raise ValueError('Calibration value "{x}" is cannot be handled'.format(x=calibration))

        return calibration

    @classmethod
    def by_profile(cls, target, name, profile, res_dir=None, default_policy=None,
                   max_duration_s=None, calibration=None,
                   log_stats=False, trace_events=None):
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

        :param calibration: The calibration value to be used by rt-app. This can
          be an integer value or a CPU string (e.g. "CPU0").
        :type calibration: int or str

        :param log_stats: Generate a log file with stats for each task
        :type log_stats: bool

        :param trace_events: A list of trace events to generate.
            For a full list of trace events which can be generated by rt-app,
            refer to the tool documentation:
            https://github.com/scheduler-tools/rt-app/blob/master/doc/tutorial.txt
            By default, no events are generated.
        :type trace_events: list(str)

        A simple profile workload would be::

            task = Periodic(duty_cycle_pct=5)
            rta = RTA.by_profile(target, "test",  {"foo" : task})
            rta.run()
        """
        logger = cls.get_logger()
        self = cls.__new__(cls)
        self._early_init(target, name, res_dir, None, log_stats=log_stats,
                        trace_events=trace_events)

        # Sanity check for task names rt-app uses pthread_setname_np(3) which
        # limits the task name to 16 characters including the terminal '\0' and
        # the rt-app suffix.
        max_size = TASK_COMM_MAX_LEN - len('-XX-XXXX')
        too_long_tids = sorted(
            tid for tid in profile.keys()
            if len(tid) > max_size
        )
        if too_long_tids:
            raise ValueError(
                'Task names too long, please configure your tasks with names shorter than {} characters: {}'.format(
                max_size, too_long_tids
            ))

        invalid_tids = sorted((
            tid for tid in profile.keys()
            if not re.match(cls.ALLOWED_TASK_NAME_REGEX, tid)
        ))
        if invalid_tids:
            raise ValueError(
                'Task names not matching "{}": {}'.format(
                cls.ALLOWED_TASK_NAME_REGEX, invalid_tids,
            ))

        rta_profile = {
            'tasks': {},
            'global': {}
        }

        calibration = self._process_calibration(calibration)

        global_conf = {
            'default_policy': 'SCHED_OTHER',
            'duration': -1 if not max_duration_s else max_duration_s,
            'calibration': calibration,
            # TODO: this can only be enabled when rt-app is running as root.
            # unfortunately, that's currently decided when calling
            # run(as_root=True), at which point we already generated and pushed
            # the JSON
            'lock_pages': False,
            'log_size' : 'file' if log_stats else 'disable',
            'ftrace': ','.join(self.trace_events),
        }

        if max_duration_s:
            logger.warning('Limiting workload duration to %d [s]', max_duration_s)

        if default_policy:
            if default_policy in self.sched_policies:
                global_conf['default_policy'] = 'SCHED_{}'.format(default_policy)
            else:
                raise ValueError('scheduling class {} not supported'.format(default_policy))

        logger.info('Calibration value: %s', global_conf['calibration'])
        logger.info('Default policy: %s', global_conf['default_policy'])

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


            logger.info('------------------------')
            logger.info('task [%s], %s', tid, sched_descr)

            task_conf['delay'] = int(task.delay_s * 1e6)
            logger.info(' | start delay: %.6f [s]', task.delay_s)

            task_conf['loop'] = task.loops
            logger.info(' | loops count: %d', task.loops)

            task_conf['phases'] = OrderedDict()
            rta_profile['tasks'][tid] = task_conf

            pid = 1
            for phase in task.phases:
                phase_name = 'p{}'.format(str(pid).zfill(6))

                logger.info(' + phase_%06d', pid)
                rta_profile['tasks'][tid]['phases'][phase_name] = phase.get_rtapp_repr(tid)

                pid += 1

        # Generate JSON configuration on local file
        with open(self.local_json, 'w') as outfile:
            json.dump(rta_profile, outfile, indent=4, separators=(',', ': '))

        self._late_init(calibration=calibration,
                        tasks_names=list(profile.keys()))
        return self

    @classmethod
    def process_template(cls, template, duration=None, pload=None, log_dir=None,
                         work_dir=None):
        """
        Turn a raw string rt-app description into a JSON dict.
        Also, process some tokens and replace them.

        :param template: The raw string to process
        :type template: str

        :param duration: The value to replace ``__DURATION__`` with
        :type duration: int

        :param pload: The value to replace ``__PVALUE__`` with
        :type pload: int or str

        :param log_dir: The value to replace ``__LOGDIR__`` with
        :type log_dir: str

        :param work_dir: The value to replace ``__WORKDIR__`` with
        :type work_dir: str

        :returns: a JSON dict
        """

        replacements = {
            '__DURATION__' : duration,
            '__PVALUE__'   : pload,
            '__LOGDIR__'   : log_dir,
            '__WORKDIR__'  : work_dir,
        }

        json_str = template
        for placeholder, value in replacements.items():
            if placeholder in template and placeholder is None:
                raise ValueError('Missing value for {} placeholder'.format(placeholder))
            else:
                json_str = json_str.replace(placeholder, json.dumps(value))

        return json.loads(json_str)

    @classmethod
    def by_str(cls, target, name, str_conf, res_dir=None, max_duration_s=None,
               calibration=None):
        """
        Create an rt-app workload using a pure string description

        :param str_conf: The raw string description. This must be a valid json
          description, with the exception of some tokens (see
          :meth:`process_template`) that will be replaced automagically.
        :type str_conf: str

        :param max_duration_s: Maximum duration of the workload.
        :type max_duration_s: int

        :param calibration: The calibration value to be used by rt-app. This can
          be an integer value or a CPU string (e.g. "CPU0").
        :type calibration: int or str
        """

        self = cls.__new__(cls)
        self._early_init(target, name, res_dir, None)

        calibration = self._process_calibration(calibration)

        json_conf = self.process_template(
            str_conf, max_duration_s, calibration, self.run_dir, self.run_dir)

        with open(self.local_json, 'w') as fh:
            json.dump(json_conf, fh)

        tasks_names = [tid for tid in json_conf['tasks']]
        self._late_init(calibration=calibration, tasks_names=tasks_names)

        return self

    @classmethod
    def _calibrate(cls, target, res_dir):
        res_dir = res_dir if res_dir else target .get_res_dir(
            "rta_calib", symlink=False
        )

        pload_regexp = re.compile(r'pLoad = ([0-9]+)ns')
        pload = {}

        logger = cls.get_logger()

        # Create calibration task
        if target.is_rooted:
            max_rtprio = int(target.execute('ulimit -Hr').split('\r')[0])
            logger.debug('Max RT prio: %d', max_rtprio)

            priority = max_rtprio if max_rtprio <= 10 else 10
            sched_policy = 'FIFO'
        else:
            logger.warning('Will use default scheduler class instead of RT since the target is not rooted')
            priority = None
            sched_policy = None

        for cpu in target.list_online_cpus():
            logger.info('CPU%d calibration...', cpu)

            # RT-app will run a calibration for us, so we just need to
            # run a dummy task and read the output
            calib_task = Periodic(
                duty_cycle_pct=100,
                duration_s=0.001,
                period_ms=1,
                priority=priority,
                sched_policy=sched_policy,
            )
            rta = cls.by_profile(target, name="rta_calib_cpu{}".format(cpu),
                                 profile={'task1': calib_task},
                                 calibration="CPU{}".format(cpu),
                                 res_dir=res_dir)

            with rta, target.freeze_userspace():
                rta.run(as_root=target.is_rooted)

            for line in rta.output.split('\n'):
                pload_match = re.search(pload_regexp, line)
                if pload_match is None:
                    continue
                pload[cpu] = int(pload_match.group(1))
                logger.debug('>>> cpu%d: %d', cpu, pload[cpu])

        logger.info('Target RT-App calibration: %s', pload)

        plat_info = target.plat_info

        # Sanity check calibration values for asymmetric systems if we have
        # access to capacities
        try:
            cpu_capacities = plat_info['cpu-capacities']
        except KeyError:
            return pload

        capa_ploads = {
            capacity: {cpu: pload[cpu] for cpu, capa in cpu_caps}
            for capacity, cpu_caps in groupby(cpu_capacities.items(), lambda k_v: k_v[1])
        }

        # Find the min pload per capacity level, i.e. the fastest detected CPU.
        # It is more likely to represent the right pload, as it has suffered
        # from less IRQ slowdown or similar disturbances that might be random.
        capa_pload = {
            capacity: min(ploads.values())
            for capacity, ploads in capa_ploads.items()
        }

        # Sort by capacity
        capa_pload_list = sorted(capa_pload.items())
        # unzip the list of tuples
        _, pload_list = zip(*capa_pload_list)

        # If sorting according to capa was not equivalent to reverse sorting
        # according to pload (small pload=fast cpu)
        if list(pload_list) != sorted(pload_list, reverse=True):
            raise CalibrationError('Calibration values reports big cores less capable than LITTLE cores')

        # Check that the CPU capacities seen by rt-app are similar to the one
        # the kernel uses
        true_capacities = cls.get_cpu_capacities_from_calibrations(pload)
        capa_factors_pct = {
            cpu: true_capacities[cpu] / cpu_capacities[cpu] * 100
            for cpu in cpu_capacities.keys()
        }
        dispersion_pct = max(abs(100 - factor) for factor in capa_factors_pct.values())

        logger.info('CPU capacities according to rt-app workload: {}'.format(true_capacities))

        if dispersion_pct > 2:
            logger.warning('The calibration values are not inversely proportional to the CPU capacities, the duty cycles will be up to {:.2f}% off on some CPUs: {}'.format(dispersion_pct, capa_factors_pct))

        if dispersion_pct > 20:
            raise CalibrationError('The calibration values are not inversely proportional to the CPU capacities. Either rt-app calibration failed, or the rt-app busy loops has a very different instruction mix compared to the workload used to establish the CPU capacities: {}'.format(capa_factors_pct))

        # Map of CPUs X to list of CPUs Ys that are faster than it although CPUs
        # of Ys have a smaller capacity than X
        faster_than_map = {
            cpu1: sorted(
                cpu2
                for cpu2, pload2 in ploads2.items()
                # CPU2 faster than CPU1
                if pload2 < pload1
            )
            for (capa1, ploads1), (capa2, ploads2) in itertools.permutations(capa_ploads.items())
            for cpu1, pload1 in ploads1.items()
            # Only look at permutations in which CPUs of ploads1 are supposed
            # to be faster than the one in ploads2
            if capa1 > capa2
        }

        # Remove empty lists
        faster_than_map = {
            cpu: faster_cpus
            for cpu, faster_cpus in faster_than_map.items()
            if faster_cpus
        }

        if faster_than_map:
            raise CalibrationError('Some CPUs of higher capacities are slower than other CPUs of smaller capacities: {}'.format(faster_than_map))

        return pload

    @classmethod
    def get_cpu_capacities_from_calibrations(cls, calibrations):
        """
        Compute the CPU capacities out of the rt-app calibration values.

        :returns: A mapping of CPU to capacity.

        :param calibrations: Mapping of CPU to pload value.
        :type calibrations: dict
        """

        # calibration values are inversely proportional to the CPU capacities
        inverse_calib = {cpu: 1/calib for cpu, calib in calibrations.items()}

        def compute_capa(cpu):
            # True CPU capacity for the rt-app workload, rather than for the
            # whatever workload was used to compute the CPU capacities exposed by
            # the kernel
            return inverse_calib[cpu] / max(inverse_calib.values()) * PELT_SCALE

        return {cpu: compute_capa(cpu) for cpu in calibrations.keys()}

    @classmethod
    def get_cpu_calibrations(cls, target, res_dir=None):
        """
        Get the rt-ap calibration value for all CPUs.

        :param target: Devlib target to run calibration on.
        :returns: Dict mapping CPU numbers to rt-app calibration values.
        """

        if 'cpufreq' not in target.modules:
            cls.get_logger().warning(
                'cpufreq module not loaded, skipping setting frequency to max')
            cm = nullcontext()
        else:
            cm = target.cpufreq.use_governor('performance')


        with cm, target.disable_idle_states():
            return cls._calibrate(target, res_dir)

class Phase(Loggable):
    """
    Descriptor for an rt-app load phase

    :param duration_s: the phase duration in [s].
    :type duration_s: int

    :param period_ms: the phase period in [ms].
    :type period_ms: int

    :param duty_cycle_pct: the generated load in [%].
    :type duty_cycle_pct: numbers.Number

    :param cpus: the CPUs on which task execution is restricted during this phase.
    :type cpus: list(int)

    :param barrier_after: if provided, the name of the barrier to sync against
                          when reaching the end of this phase. Currently only
                          supported when duty_cycle_pct=100
    :type barrier_after: str
    """

    def __init__(self, duration_s, period_ms, duty_cycle_pct, cpus=None, barrier_after=None,
                 uclamp_min=None, uclamp_max=None):
        if barrier_after and duty_cycle_pct != 100:
            # This could be implemented but currently don't foresee any use.
            raise ValueError('Barriers only supported when duty_cycle_pct=100')

        self.duration_s = duration_s
        self.period_ms = period_ms
        self.duty_cycle_pct = duty_cycle_pct
        self.cpus = cpus
        self.barrier_after = barrier_after
        self.uclamp_min = uclamp_min
        self.uclamp_max = uclamp_max

    def get_rtapp_repr(self, task_name):
        """
        Get a dictionnary representation of the phase as expected by rt-app

        :param task_name: Name of the phase's task (needed for timers)
        :type task_name: str

        :returns: OrderedDict
        """
        logger = self.get_logger()
        phase = OrderedDict()
        # Convert time parameters to integer [us] units
        duration = int(self.duration_s * 1e6)

        # A duty-cycle of 0[%] translates to a 'sleep' phase
        if self.duty_cycle_pct == 0:
            logger.info(' | sleep %.6f [s]', duration/1e6)

            phase['loop'] = 1
            phase['sleep'] = duration

        # A duty-cycle of 100[%] translates to a 'run-only' phase
        elif self.duty_cycle_pct == 100:
            logger.info(' | batch %.6f [s]', duration/1e6)

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
                cloops = duration // period

            sleep_time = period * (100 - self.duty_cycle_pct) // 100
            # rtapp fails to handle floating values correctly
            # https://github.com/scheduler-tools/rt-app/issues/82
            running_time = int(period - sleep_time)

            logger.info(' | duration %.6f [s] (%d loops)',
                             duration/1e6, cloops)
            logger.info(' |  period   %6d [us], duty_cycle %3d %%',
                             period, self.duty_cycle_pct)
            logger.info(' |  run_time %6d [us], sleep_time %6d [us]',
                             running_time, sleep_time)

            phase['loop'] = cloops
            phase['run'] = running_time
            phase['timer'] = {'ref': task_name, 'period': period}

        if self.cpus is not None:
            phase['cpus'] = self.cpus

        if self.uclamp_min is not None:
            phase['util_min'] = self.uclamp_min
            logger.info(' | util_min %7d', self.uclamp_min)

        if self.uclamp_max is not None:
            phase['util_max'] = self.uclamp_max
            logger.info(' | util_max %7d', self.uclamp_max)

        return phase

class RTATask(object):
    """
    Base class for conveniently constructing params to :meth:`RTA.by_profile`

    :param sched_policy: the scheduler policy for this task.
    :type sched_policy: str or None

    This class represents an rt-app task which may contain multiple :class:`Phase`.
    It implements ``__add__`` so that using ``+`` on two tasks concatenates their
    phases. For example ``Ramp() + Periodic()`` would yield an ``RTATask`` that
    executes the default phases for :class:`Ramp` followed by the default phases for
    :class:`Periodic`.
    """

    def __init__(self, delay_s=0, loops=1, sched_policy=None, priority=None):
        self.delay_s = delay_s
        self.loops = loops

        if isinstance(sched_policy, str):
            sched_policy = sched_policy.upper()

            if sched_policy not in RTA.sched_policies:
                raise ValueError('scheduling class {} not supported'.format(sched_policy))

        self.sched_policy = sched_policy
        self.priority = priority
        self.phases = []

    def __add__(self, task):
        # Do not modify the original object which might still be used for other
        # purposes
        new = copy.deepcopy(self)
        # Piggy back on the __iadd__ implementation
        new += task
        return new

    def __iadd__(self, task):
        if task.delay_s:
            # This won't work, because rt-app's "delay" field is per-task and
            # not per-phase. We might be able to implement it by adding a
            # "sleep" event here, but let's not bother unless such a need
            # arises.
            raise ValueError("Can't compose rt-app tasks "
                             "when the second has nonzero 'delay_s'")
        self.phases.extend(task.phases)
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

    :param sched_policy: the scheduler policy for this task.
    :type sched_policy: str or None

    :param cpus: the list of CPUs on which task can run.

        .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=0, end_pct=100, delta_pct=10, time_s=1,
                 period_ms=100, delay_s=0, loops=1, sched_policy=None,
                 priority=None, cpus=None, uclamp_min=None, uclamp_max=None):
        super(Ramp, self).__init__(delay_s, loops, sched_policy, priority)

        if not (0 <= start_pct <= 100 and 0 <= end_pct <= 100):
            raise ValueError('start_pct and end_pct must be in [0..100] range')

        # Make sure the delta goes in the right direction
        sign = +1 if start_pct <= end_pct else -1
        delta_pct = sign * abs(delta_pct)

        steps = list(range(start_pct, end_pct+delta_pct, delta_pct))

        # clip the last step
        steps[-1] = end_pct

        phases = []
        for load in steps:
            if load == 0:
                phase = Phase(time_s, 0, 0, cpus, uclamp_min=uclamp_min,
                              uclamp_max=uclamp_max)
            else:
                phase = Phase(time_s, period_ms, load, cpus,
                              uclamp_min=uclamp_min, uclamp_max=uclamp_max)
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

    :param sched_policy: the scheduler policy for this task.
    :type sched_policy: str or None

    :param cpus: the list of CPUs on which task can run.

        .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=0, end_pct=100, time_s=1, period_ms=100,
                 delay_s=0, loops=1, sched_policy=None, priority=None, cpus=None,
                 uclamp_min=None, uclamp_max=None):
        delta_pct = abs(end_pct - start_pct)
        super().__init__(start_pct, end_pct, delta_pct, time_s,
                                   period_ms, delay_s, loops, sched_policy,
                                   priority, cpus, uclamp_min, uclamp_max)

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

    :param sched_policy: the scheduler policy for this task.
    :type sched_policy: str or None

    :param cpus: the list of CPUs on which task can run

        .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, start_pct=100, end_pct=0, time_s=1, period_ms=100,
                 delay_s=0, loops=1, sched_policy=None, priority=None, cpus=None,
                 uclamp_min=None, uclamp_max=None):
        super(Pulse, self).__init__(delay_s, loops, sched_policy, priority)

        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        if not (0 <= start_pct <= 100 and 0 <= end_pct <= 100):
            raise ValueError('end_pct and start_pct must be in [0..100] range')

        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        phases = []
        for load in [start_pct, end_pct]:
            if load == 0:
                continue
            phase = Phase(time_s, period_ms, load, cpus, uclamp_min=uclamp_min,
                          uclamp_max=uclamp_max)
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

    :param priority: the priority for this task.
    :type priority: int or None

    :param sched_policy: the scheduler policy for this task.
    :type sched_policy: str or None

    :param cpus: the list of CPUs on which task can run.

        .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)
    """

    def __init__(self, duty_cycle_pct=50, duration_s=1, period_ms=100,
                 delay_s=0, sched_policy=None, priority=None, cpus=None,
                 uclamp_min=None, uclamp_max=None):
        super(Periodic, self).__init__(duty_cycle_pct, 0, duration_s,
                                       period_ms, delay_s, 1, sched_policy,
                                       priority, cpus,
                                       uclamp_min=uclamp_min,
                                       uclamp_max=uclamp_max)

class RunAndSync(RTATask):
    """
    Configure a task that runs 100% then waits on a barrier

    :param barrier: name of barrier to wait for. Sleeps until any other tasks
                    that refer to this barrier have reached the barrier too.
    :type barrier: str

    :param time_s: time to run for

    :param delay_s: the delay in seconds before starting.

    :param sched_policy: the scheduler policy for this task.
    :type sched_policy: str or None

    :param cpus: the list of CPUs on which task can run.

        .. note:: if not specified, it can run on all CPUs
    :type cpus: list(int)

    """
    def __init__(self, barrier, time_s=1, delay_s=0, loops=1, sched_policy=None,
                 priority=None, cpus=None, uclamp_min=None, uclamp_max=None):
        super(RunAndSync, self).__init__(delay_s, loops, sched_policy, priority)

        # This should translate into a phase containing a 'run' event and a
        # 'barrier' event
        self.phases = [Phase(time_s, None, 100, cpus, barrier_after=barrier,
                             uclamp_min=uclamp_min, uclamp_max=uclamp_max)]

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
