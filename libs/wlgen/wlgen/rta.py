
import fileinput
import json
import logging
import os
import re

from wlgen import Workload

class RTA(Workload):

    def __init__(self,
                 target,
                 name,
                 calibration=None):

        # rt-app calibration
        self.pload = calibration

        # TODO: Assume rt-app is pre-installed on target
        # self.target.setup('rt-app')

        super(RTA, self).__init__(target, name, calibration)

        # rt-app executor
        self.executor = 'rt-app'

        # Setup RTA callbacks
        self.setCallback('postrun', self.__postrun)

    @staticmethod
    def calibrate(target):
        pload_regexp = re.compile(r'pLoad = ([0-9]+)ns')
        pload = {}

        # target.cpufreq.save_governors()
        target.cpufreq.set_all_governors('performance')

        for cpu in target.list_online_cpus():

            logging.info('CPU{0:d} calibration...'.format(cpu))

            rta = RTA(target, 'rta_calib')
            rta.conf(kind='periodic', params=[(100000, 50)], cpus=[cpu], duration=1)
            rta.run()

            for line in rta.getOutput().split('\n'):
                pload_match = re.search(pload_regexp, line)
                if (pload_match is None):
                    continue
                pload[cpu] = int(pload_match.group(1))
                logging.debug('>>> cpu{0:d}: {1:d}'.format(cpu, pload[cpu]))

        # target.cpufreq.load_governors()

        logging.info('Target RT-App calibration:')
        logging.info('{0:s}'.format(str(pload)))

        return pload

    def __postrun(self, params):
        destdir = params['destdir']
        if destdir is None:
            return
        logging.debug('Pulling logfiles to [{0:s}]...'.format(destdir))
        for task in self.tasks.keys():
            logfile = "'{0:s}/*{1:s}*.log'"\
                    .format(self.run_dir, task)
            self.target.pull(logfile, destdir)
        logging.debug('Pulling JSON to [%s]...', destdir)
        self.target.pull('{}/{}'.format(self.run_dir, self.json), destdir)
        logfile = '{}/output.log'.format(destdir)
        logging.debug('Saving output on [%s]...', logfile)
        with open(logfile, 'w') as ofile:
            for line in self.output['executor'].split('\n'):
                ofile.write(line+'\n')

    def _getFirstBig(self, cpus=None):
        if cpus:
            for c in cpus:
                if c not in self.target.bl.bigs:
                    continue
                return c
        # Only LITTLE CPUs, thus:
        #  return the first big core of the system
        if self.target.big_core:
            # Big.LITTLE system
            return self.target.bl.bigs[0]
        return 0

    def _getFirstLittle(self, cpus=None):
        # Try to return one LITTLE CPUs among the specified ones
        if cpus:
            for c in cpus:
                if c not in self.target.bl.littles:
                    continue
                return c
        # Only big CPUs, thus:
        #  return the first LITTLE core of the system
        if self.target.little_core:
            # Big.LITTLE system
            return self.target.bl.littles[0]
        return 0

    def getTargetCpu(self, loadref):
        # Select CPU for task calibration, which is the first little
        # of big depending on the loadref tag
        if self.pload:
            if loadref.upper() == 'LITTLE':
                target_cpu = self._getFirstLittle()
                logging.debug('ref on LITTLE cpu: {0:d}'.format(target_cpu))
            else:
                target_cpu = self._getFirstBig()
                logging.debug('ref on big cpu: {0:d}'.format(target_cpu))
        elif self.cpus is None:
            target_cpu = self._getFirstBig()
            logging.debug('ref on cpu: {0:d}'.format(target_cpu))
        else:
            target_cpu = self._getFirstBig(self.cpus)
            logging.debug('ref on cpu: {0:d}'.format(target_cpu))
        return target_cpu

    def getCalibrationConf(self, target_cpu=0):
        if (self.pload is None):
            return 'CPU{0:d}'.format(target_cpu)
        return self.pload[target_cpu]

    def _confCustom(self):

        if self.duration is None:
            raise ValueError('Workload duration not specified')

        target_cpu = self.getTargetCpu(self.loadref)
        calibration = self.getCalibrationConf(target_cpu)

        self.json = '{0:s}_{1:02d}.json'.format(self.name, self.exc_id)
        ofile = open(self.json, 'w')
        ifile = open(self.params['custom'], 'r')
        replacements = {
            '__DURATION__' : str(self.duration),
            '__PVALUE__'   : str(calibration),
            '__LOGDIR__'   : str(self.run_dir),
            '__WORKDIR__'  : '"'+self.target.working_directory+'"',
        }

        for line in ifile:
            for src, target in replacements.iteritems():
                line = line.replace(src, target)
            ofile.write(line)
        ifile.close()
        ofile.close()

        return self.json

    def _confPeriodic(self):

        # periodic task configuration
        target_cpu = self.getTargetCpu(self.loadref)
        self.rta_periodic = {
            'tasks': {},
            'global': {
                'default_policy': 'SCHED_OTHER',
                'duration': 5,
                'calibration': 'CPU'+str(target_cpu),
                'logdir': self.run_dir,
            }
        }

        # Setup calibration data
        calibration = self.getCalibrationConf(target_cpu)
        self.rta_periodic['global']['calibration'] = calibration
        if self.duration is not None:
            self.rta_periodic['global']['duration'] = self.duration
            logging.info('Configure workload duration to {0:d}[s]'\
                        .format(self.rta_periodic['global']['duration']))

        # Setup tasks parameters
        for tid in self.params:
            task = self.params[tid]
            self.rta_periodic['global']['calibration'] = calibration
            self.rta_periodic['tasks'][task['name']] = {
                'loop': -1,
                'run': task['running_time'],
                'timer': {'ref': task['name'], 'period': task['period']}
            }
            # Append task name to the list of this workload tasks
            self.tasks[task['name']] = {'pid': -1}

        # Generate JSON configuraiton on local file
        self.json = '{0:s}_{1:02d}.json'.format(self.name, self.exc_id)
        with open(self.json, 'w') as outfile:
            json.dump(self.rta_periodic, outfile,
                    sort_keys=True, indent=4, separators=(',', ': '))

        return self.json

    def _confProfile(self):

        # periodic task configuration
        target_cpu = self.getTargetCpu(self.loadref)
        self.rta_profile = {
            'tasks': {},
            'global': {
                'default_policy': 'SCHED_OTHER',
                'duration': -1,
                'calibration': 'CPU'+str(target_cpu),
                'logdir': self.run_dir,
            }
        }

        # Setup calibration data
        calibration = self.getCalibrationConf(target_cpu)
        self.rta_profile['global']['calibration'] = calibration
        if self.duration is not None:
            self.rta_profile['global']['duration'] = self.duration
            logging.warn('Limiting workload duration to {0:d}[s]'\
                .format(self.rta_profile['global']['duration']))
        else:
            logging.info('Workload duration defined by longest task')

        # Setup tasks parameters
        for tid in sorted(self.params['profile'].keys()):
            task = self.params['profile'][tid]

            self.rta_profile['tasks'][tid] = {}

            task['sclass'] = task['sclass'].upper()
            if task['sclass'] not in ['OTHER', 'FIFO', 'RR', 'DEADLINE']:
                raise ValueError('scheduling class {} not supported'.format(task['sclass']))
            self.rta_profile['tasks'][tid]['policy'] = 'SCHED_' + task['sclass']
            self.rta_profile['tasks'][tid]['priority'] = task['prio']
            self.rta_profile['tasks'][tid]['phases'] = {}

            logging.info('------------------------')
            logging.info('task [{0:s}], SCHED_{1:s}:'.format(tid, task['sclass']))

            if 'delay' in task.keys():
                if task['delay'] > 0:
                    task['delay'] = int(task['delay'] * 1e6)
                    self.rta_profile['tasks'][tid]['phases']['p000000'] = {}
                    self.rta_profile['tasks'][tid]['phases']['p000000']['sleep'] = task['delay']
                    logging.info(' | start delay: {0:.6f} [s]'\
                            .format(task['delay'] / 1e6))

            if 'loops' not in task.keys():
                task['loops'] = 1
            self.rta_profile['tasks'][tid]['loop'] = task['loops']
            logging.info(' | loops count: {0:d}'.format(task['loops']))

            # Getting task phase descriptor
            pid=1
            for phase in task['phases']:
                (duration, period, duty_cycle) = phase

                # Convert time parameters to integer [us] units
                duration = int(duration * 1e6)
                period = int(period * 1e3)

                # A duty-cycle of 0[%] translates on a 'sleep' phase
                if duty_cycle == 0:

                    logging.info(' + phase_{0:06d}: sleep {1:.6f} [s]'\
                        .format(pid, duration/1e6))

                    task_phase = {
                        'loop': 1,
                        'sleep': duration,
                    }

                # A duty-cycle of 100[%] translates on a 'run-only' phase
                elif duty_cycle == 100:

                    logging.info(' + phase_{0:06d}: batch {1:.6f} [s]'\
                        .format(pid, duration/1e6))

                    task_phase = {
                        'loop': 1,
                        'run': duration,
                    }

                # A certain number of loops is requires to generate the
                # proper load
                else:

                    cloops = -1
                    if (duration >= 0):
                        cloops = int(duration / period)

                    sleep_time = period * (100 - duty_cycle) / 100
                    running_time = period - sleep_time

                    logging.info(' + phase_{0:06d}: duration {1:.6f} [s] ({2:d} loops)'\
                        .format(pid, duration/1e6, cloops))
                    logging.info(' |  period   {0:6d} [us], duty_cycle {1:3d} %'\
                        .format(period, duty_cycle))
                    logging.info(' |  run_time {0:6d} [us], sleep_time {1:6d} [us]'\
                        .format(running_time, sleep_time))

                    task_phase = {
                        'loop': cloops,
                        'run': running_time,
                        'timer': {'ref': tid, 'period': period},
                    }

                self.rta_profile['tasks'][tid]['phases']['p'+str(pid).zfill(6)] = task_phase

                pid+=1

            # Append task name to the list of this workload tasks
            self.tasks[tid] = {'pid': -1}

        # Generate JSON configuration on local file
        self.json = '{0:s}_{1:02d}.json'.format(self.name, self.exc_id)
        with open(self.json, 'w') as outfile:
            json.dump(self.rta_profile, outfile,
                    sort_keys=True, indent=4, separators=(',', ': '))

        return self.json

    @staticmethod
    def ramp(start_pct=0, end_pct=100, delta_pct=10, time_s=1, period_ms=100,
            delay_s=0, loops=1, prio=0, sclass='OTHER'):
        """
        Configure a ramp load.

        This class defines a task which load is a ramp with a configured number
        of steps according to the input paramters.

        Args:
            start_pct (int, [0-100]): the initial load [%], (default 0[%])
            end_pct   (int, [0-100]): the final load [%], (default 100[%])
            delta_pct (int, [0-100]): the load increase/decrease [%], (default 10[%])
                                      increaase if start_prc < end_prc
                                      decrease  if start_prc > end_prc
            time_s    (float): the duration in [s] of each load step, (default 1.0[s])
            period_ms (float): the pediod used to define the load in [ms], (default 100.0[ms])
            delay_s   (float): the delay in [s] before ramp start, (deafault 0[s])
            loops     (int): number of time to repeat the ramp, with the specified delay in between (deafault 0)
            prio      (int): the task priority, (default 0)
            sclass    (string): the class (OTHER|FIFO) of the workload
        """
        task = {}

        task['prio'] = prio
        task['sclass'] = sclass
        task['delay'] = delay_s
        task['loops'] = loops
        task['phases'] = {}

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
                phase = (time_s, 0, 0)
            else:
                phase = (time_s, period_ms, load)
            phases.append(phase)

        task['phases'] = phases

        return task;

    @staticmethod
    def step(start_pct=0, end_pct=100, time_s=1, period_ms=100,
            delay_s=0, loops=1, prio=0, sclass='OTHER'):
        """
        Configure a step load.

        This class defines a task which load is a step with a configured
        initial and final load.

        Args:
            start_pct (int, [0-100]): the initial load [%], (default 0[%])
            end_pct   (int, [0-100]): the final load [%], (default 100[%])
            time_s    (float): the duration in [s] of the start and end load, (default 1.0[s])
            period_ms (float): the pediod used to define the load in [ms], (default 100.0[ms])
            delay_s   (float): the delay in [s] before ramp start, (deafault 0[s])
            loops     (int): number of time to repeat the ramp, with the specified delay in between (deafault 0)
            prio      (int): the task priority, (default 0)
            sclass    (string): the class (OTHER|FIFO) of the workload

        """
        delta_pct = abs(end_pct - start_pct)
        return RTA.ramp(start_pct, end_pct, delta_pct, time_s, period_ms, delay_s, loops, prio, sclass)

    @staticmethod
    def pulse(start_pct=100, end_pct=0, time_s=1, period_ms=100,
            delay_s=0, loops=1, prio=0, sclass='OTHER'):
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

        Args:
            start_pct (int, [0-100]): the initial load [%], (default 0[%])
            end_pct   (int, [0-100]): the final load [%], (default 100[%])
                      NOTE: must be lower than start_pct value
            time_s    (float): the duration in [s] of the start and end load, (default 1.0[s])
                      NOTE: if end_pct is 0, the task end after the start_pct period completed
            period_ms (float): the pediod used to define the load in [ms], (default 100.0[ms])
            delay_s   (float): the delay in [s] before ramp start, (deafault 0[s])
            loops     (int): number of time to repeat the ramp, with the specified delay in between (deafault 0)
            prio      (int): the task priority, (default 0)
            sclass    (string): the class (OTHER|FIFO) of the workload

        """

        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        task = {}

        task['prio'] = prio
        task['sclass'] = sclass
        task['delay'] = delay_s
        task['loops'] = loops
        task['phases'] = {}

        if end_pct not in range(0,101) or start_pct not in range(0,101):
            raise ValueError('end_pct and start_pct must be in [0..100] range')
        if end_pct >= start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        step_pct = start_pct - end_pct

        phases = []
        for load in [start_pct, end_pct]:
            if load == 0:
                continue
            phase = (time_s, period_ms, load)
            phases.append(phase)

        task['phases'] = phases

        return task;

    @staticmethod
    def periodic(duty_cycle_pct=50, duration_s=1, period_ms=100,
            delay_s=0, prio=0, sclass='OTHER'):
        """
        Configure a periodic load.

        This class defines a task which load is periodic with a configured
        period and duty-cycle.

        This class is a specialization of the 'pulse' class since a periodic
        load is generated as a sequence of pulse loads.

        Args:
            cuty_cycle_pct  (int, [0-100]): the pulses load [%], (default 50[%])
            duration_s      (float): the duration in [s] of the entire workload, (default 1.0[s])
            period_ms       (float): the pediod used to define the load in [ms], (default 100.0[ms])
            delay_s         (float): the delay in [s] before ramp start, (deafault 0[s])
            prio            (int): the task priority, (default 0)
            sclass          (string): the class (OTHER|FIFO) of the workload

        """

        return RTA.pulse(duty_cycle_pct, 0, duration_s, period_ms, delay_s, 1, prio, sclass)

    def conf(self,
             kind,
             params,
             duration=None,
             cpus=None,
             cgroup=None,
             run_dir='./',
             loadref='big',
             exc_id=0):
        """
        Configure a workload of a specified kind.

        The rt-app based workload allows to define different classes of
        workloads. The classes supported so far are detailed hereafter.

        Periodic workloads
        ------------------


        Custom workloads
        ----------------


        Profile based workloads
        -----------------------
        When 'kind' is 'profile' the tasks generated by this workload have a
        profile which is defined by a sequence of phases and they are defined
        according to the following grammar:

          params := {task, ...}
          task   := NAME : {SCLASS, PRIO, [phase, ...]}
          phase  := (PTIME, PRIOD, DCYCLE)

        where the terminals are:

          NAME   : string, the task name (max 16 chars)
          SCLASS : string, the scheduling class (OTHER, FIFO, RR)
          PRIO   : int, the pririty of the task
          PTIME  : float, length of the current phase in [s]
          PERIOD : float, task activation interval in [ms]
          DCYCLE : int, task running interval in [0..100]% within each period

        """


        super(RTA, self).conf(kind, params, duration,
                cpus, cgroup, run_dir, exc_id)

        self.loadref = loadref

        # Setup class-specific configuration
        if (kind == 'custom'):
            self._confCustom()
        elif (kind == 'periodic'):
            self._confPeriodic()
        elif (kind == 'profile'):
            self._confProfile()

        # Move configuration file to target
        self.target.push(self.json, self.run_dir)

        self.rta_cmd  = self.target.executables_directory + '/rt-app'
        self.rta_conf = self.run_dir + '/' + self.json
        self.command = '{0:s} {1:s}'.format(self.rta_cmd, self.rta_conf)

        # Set and return the test label
        self.test_label = '{0:s}_{1:02d}'.format(self.name, self.exc_id)
        return self.test_label

