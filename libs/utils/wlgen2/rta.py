import json
import logging
import os
import re
from collections import OrderedDict

from devlib.utils.misc import ranges_to_list

from workload import Workload

class RTA(Workload):
    """
    :param tasks: Description of the workload using :class:`RTATask`, described
      as {task_name : :class:`RTATask`}
    :type tasks: dict(RTATask)
    """

    def __init__(self, te, res_dir, name, tasks=None, conf=None, default_policy=None,
                 max_duration_s=None, calibration=None):
        super(RTA, self).__init__(te, res_dir)

        self.name = name

        self.tasks = tasks
        self.conf = conf
        self.default_policy = default_policy
        self.max_duration_s = max_duration_s
        self.calibration = calibration

        self.pload = None

        json_name = '{}.json'.format(self.name)
        self.local_json = os.path.join(self.res_dir, json_name)
        self.remote_json = self.te.target.path.join(self.run_dir, json_name)

        if tasks:
            self._init_tasks()
        else:
            self._init_conf()

        # Move configuration file to target
        self.te.target.push(self.local_json, self.remote_json)

        rta_cmd = self.te.target.which('rt-app')
        self.command = '{0:s} {1:s} 2>&1'.format(rta_cmd, self.remote_json)

    def _select_calibration(self):
        # Select CPU or pload value for task calibration
        if self.calibration is not None:
            return min(self.calibration.values())
        else:
            cpus = range(self.te.target.number_of_cpus)
            target_cpu = cpus[-1]
            if 'bl'in self.te.target.modules:
                candidates = sorted(set(self.te.target.bl.bigs).intersection(cpus))
                if candidates:
                    target_cpu = candidates[0]

            return 'CPU{0:d}'.format(target_cpu)

    def _init_tasks(self):
        # Sanity check for task names
        for task in self.tasks.keys():
            if len(task) > 15:
                # rt-app uses pthread_setname_np(3) which limits the task name
                # to 16 characters including the terminal '\0'.
                msg = ('Task name "{}" too long, please configure your tasks '
                       'with names shorter than 16 characters').format(task)
                raise ValueError(msg)

        # Task configuration
        rta_profile = {
            'tasks': {},
            'global': {}
        }

        # Initialize global configuration
        global_conf = {
                'default_policy': 'SCHED_OTHER',
                'duration': -1 if not self.max_duration_s else self.max_duration_s,
                'calibration': self._select_calibration(),
                'logdir': self.run_dir,
            }

        # self._log.warn('Limiting workload duration to %d [s]',
        #                global_conf['duration'])

        # Setup default scheduling class
        if self.default_policy:
            policy = self.default_policy
            if policy not in ['OTHER', 'FIFO', 'RR', 'DEADLINE']:
                raise ValueError('scheduling class {} not supported'\
                        .format(policy))
            global_conf['default_policy'] = 'SCHED_{}'.format(policy)

        #self._log.info('Default policy: %s', global_conf['default_policy'])

        # Setup global configuration
        rta_profile['global'] = global_conf

        # Setup tasks parameters
        for tid in sorted(self.tasks.keys()):
            task = self.tasks[tid]

            # Initialize task configuration
            task_conf = {}

            if not task.sched_policy:
                policy = 'DEFAULT'
            else:
                policy = task.sched_policy.upper()

            if policy == 'DEFAULT':
                task_conf['policy'] = global_conf['default_policy']
                sched_descr = 'sched: using default policy'
            elif policy not in ['OTHER', 'FIFO', 'RR', 'DEADLINE']:
                raise ValueError('scheduling class {} not supported'\
                        .format(task['sclass']))
            else:
                task_conf.update(task.sched_policy)
                task_conf['policy'] = 'SCHED_' + policy
                sched_descr = 'sched: {0:s}'.format(task['sched'])

            # Initialize task phases
            task_conf['phases'] = OrderedDict()

            # self._log.info('------------------------')
            # self._log.info('task [%s], %s', tid, sched_descr)

            if task.delay_s:
                task_conf['delay'] = int(task['delay'] * 1e6)
                # self._log.info(' | start delay: %.6f [s]',
                #                task['delay'])

            task_conf['loop'] = task.loops
            self._log.info(' | loops count: %d', task.loops)

            # Setup task configuration
            rta_profile['tasks'][tid] = task_conf

            # Getting task phase descriptor
            pid=1
            for phase in task.phases:

                # Convert time parameters to integer [us] units
                duration = int(phase.duration_s * 1e6)

                task_phase = OrderedDict()

                # A duty-cycle of 0[%] translates to a 'sleep' phase
                if phase.duty_cycle_pct == 0:
                    # self._log.info(' + phase_%06d: sleep %.6f [s]',
                    #                pid, duration/1e6)

                    task_phase['loop'] = 1
                    task_phase['sleep'] = duration

                # A duty-cycle of 100[%] translates on a 'run-only' phase
                elif phase.duty_cycle_pct == 100:
                    # self._log.info(' + phase_%06d: batch %.6f [s]',
                    #                pid, duration/1e6)
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

                    # self._log.info('+ phase_%06d: duration %.6f [s] (%d loops)',
                    #                pid, duration/1e6, cloops)
                    # self._log.info('|  period   %6d [us], duty_cycle %3d %%',
                    #                period, phase.duty_cycle_pct)
                    # self._log.info('|  run_time %6d [us], sleep_time %6d [us]',
                    #                running_time, sleep_time)

                    task_phase['loop'] = cloops
                    task_phase['run'] = running_time
                    task_phase['timer'] = {'ref': tid, 'period': period}

                rta_profile['tasks'][tid]['phases']['p'+str(pid).zfill(6)] = task_phase

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
                    # self._log.info('|  CPUs affinity: {}'.format(phase.cpus))
                pid += 1

        # Generate JSON configuration on local file
        with open(self.local_json, 'w') as outfile:
            json.dump(rta_profile, outfile, indent=4, separators=(',', ': '))

    def _init_conf(self):
        rtapp_conf = self.conf

        ofile = open(self.local_json, 'w')

        calibration = self.getCalibrationConf()
        # Calibration can either be a string like "CPU1" or an integer, if the
        # former we need to quote it.
        if type(calibration) != int:
            calibration = '"{}"'.format(calibration)

        replacements = {
            '__DURATION__' : str(self.duration),
            '__PVALUE__'   : str(calibration),
            '__LOGDIR__'   : str(self.run_dir),
            '__WORKDIR__'  : '"'+self.te.target.working_directory+'"',
        }

        # Check for inline config
        if not isinstance(self.params['custom'], basestring):
            if isinstance(self.params['custom'], dict):
                # Inline config present, turn it into a file repr
                tmp_json = json.dumps(rtapp_conf,
                    indent=4, separators=(',', ': '), sort_keys=True)
                ifile = tmp_json.splitlines(True)
            else:
                raise ValueError("Value specified for 'params'  can only be "
                                 "a filename or an embedded dictionary")
        else:
            # We assume we are being passed a filename instead of a dict
            self._log.info('Loading custom configuration:')
            self._log.info('   %s', rtapp_conf)
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

        with open(self.local_json) as f:
            conf = json.load(f)
        for tid in conf['tasks']:
            self.tasks[tid] = {'pid': -1}

    @classmethod
    def _calibrate(target):
        pload_regexp = re.compile(r'pLoad = ([0-9]+)ns')
        pload = {}

        # Setup logging
        # log = logging.getLogger('RTApp')

        # Create calibration task
        max_rtprio = int(target.execute('ulimit -Hr').split('\r')[0])
        # log.debug('Max RT prio: %d', max_rtprio)
        if max_rtprio > 10:
            max_rtprio = 10

        calib_task = Periodic(period_ms=100,
                              duty_cycle_pct=50,
                              duration_s=1,
                              sched_policy={
                                  'policy': 'FIFO',
                                  'prio': max_rtprio
                              })
        #rta = RTA(target, 'rta_calib')

        for cpu in target.list_online_cpus():
            # log.info('CPU%d calibration...', cpu)

            #rta.conf(kind='profile', params={'task1': calib_task}, cpus=[cpu])
            rta = RTA(target, tasks={'task1': calib_task})
            rta.run(as_root=True)

            for line in rta.getOutput().split('\n'):
                pload_match = re.search(pload_regexp, line)
                if pload_match is None:
                    continue
                pload[cpu] = int(pload_match.group(1))
                # log.debug('>>> cpu%d: %d', cpu, pload[cpu])

        # log.info('Target RT-App calibration:')
        # log.info("{" + ", ".join('"%r": %r' % (key, pload[key])
        #                          for key in pload) + "}")

        # Sanity check calibration values for big.LITTLE systems
        # if 'bl' in target.modules:
        #     bcpu = target.bl.bigs_online[0]
        #     lcpu = target.bl.littles_online[0]
        #     if pload[bcpu] > pload[lcpu]:
        #         log.warning('Calibration values reports big cores less '
        #                     'capable than LITTLE cores')
        #         raise RuntimeError('Calibration failed: try again or file a bug')
        #     bigs_speedup = ((float(pload[lcpu]) / pload[bcpu]) - 1) * 100
        #     log.info('big cores are ~%.0f%% more capable than LITTLE cores',
        #              bigs_speedup)

        return pload

    @classmethod
    def get_calibration(target):
        """
        Calibrate RT-App on each CPU in the system

        :param target: Devlib target to run calibration on.
        :returns: Dict mapping CPU numbers to RT-App calibration values.
        """

        if 'cpufreq' not in target.modules:
            logging.getLogger(cls.__name__).warning(
                'cpufreq module not loaded, skipping setting frequency to max')
            return cls._calibrate(target)

        with target.cpufreq.use_governor('performance'):
            return cls._calibrate(target)

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

class RTATask(object):
    """
    Base class for conveniently constructing params to :meth:`RTA.conf`

    This class represents an RT-App task which may contain multiple phases. It
    implements ``__add__`` so that using ``+`` on two tasks concatenates their
    phases. For example ``Ramp() + Periodic()`` would yield an ``RTATask`` that
    executes the default phases for ``Ramp`` followed by the default phases for
    ``Periodic``.
    """

    def __init__(self, delay_s=0, loops=1, sched_policy=None):
        self.sched_policy = sched_policy
        self.delay_s = delay_s
        self.loops = loops
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
                 period_ms=100, delay_s=0, loops=1, sched_policy=None, cpus=None):
        super(Ramp, self).__init__(delay_s, loops, sched_policy)

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
                 delay_s=0, loops=1, sched_policy=None, cpus=None):
        delta_pct = abs(end_pct - start_pct)
        super(Step, self).__init__(start_pct, end_pct, delta_pct, time_s,
                                   period_ms, delay_s, loops, sched_policy, cpus)

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
                 delay_s=0, loops=1, sched_policy=None, cpus=None):
        super(Pulse, self).__init__(delay_s, loops, sched_policy)

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
                 delay_s=0, sched_policy=None, cpus=None):
        super(Periodic, self).__init__(duty_cycle_pct, 0, duration_s,
                                       period_ms, delay_s, 1, sched_policy, cpus)

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
                 delay_s=0, loops=1, sched_policy=None, cpus=None):
        super(RunAndSync, self).__init__(delay_s, loops, sched_policy)

        # This should translate into a phase containing a 'run' event and a
        # 'barrier' event
        self.phases = [Phase(time_s, None, 100, cpus,
                                      barrier_after=barrier)]
