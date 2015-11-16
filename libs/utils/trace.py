
import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pylab as pl
import re
import sys
import trappy

# Configure logging
import logging

class Trace(object):

    def __init__(self, platform, datadir, tasks=None):

        # The platform used to run the experiments
        self.platform = None

        # Dataframe of all events data
        self.trace_data = {}

        # Folder containing all perf data
        self.datadir = None

        # TRAPpy run object
        self.run = None

        # Dynamically registered TRAPpy events
        self.trappy_cls = {}

        # Maximum timespan for all collected events
        self.time_range = 0

        # The dictionary of tasks descriptors available in the dataset
        self.tasks = {}

        # List of events available in the parsed trace
        self.available_events = []

        # Folder containing trace
        self.datadir = datadir

        # Platform descriptor
        self.platform = platform

        self.__registerTraceEvents()
        self.__parseTrace(datadir, tasks)
        self.__computeTimeSpan()

    def __registerTraceEvents(self):

        # Additional standard events
        self.trappy_cls['sched_migrate_task'] = trappy.register_dynamic(
                'SchedMigrateTask', 'sched_migrate_task:', scope='sched');

        self.trappy_cls['sched_wakeup'] = trappy.register_dynamic(
                'SchedWakeup', 'sched_wakeup:', scope='sched', parse_raw=True);
        self.trappy_cls['sched_wakeup_new'] = trappy.register_dynamic(
                'SchedWakeupNew', 'sched_wakeup_new:', scope='sched', parse_raw=True);

        self.trappy_cls['sched_boost_cpu'] = trappy.register_dynamic(
                'SchedBoostCpu', 'sched_boost_cpu:', scope='sched')
        self.trappy_cls['sched_boost_task'] = trappy.register_dynamic(
                'SchedBoostTask', 'sched_boost_task:', scope='sched')

        self.trappy_cls['sched_energy_diff'] = trappy.register_dynamic(
                'SchedEnergyDiff', 'sched_energy_diff:', scope='sched')
        self.trappy_cls['sched_tune_config'] = trappy.register_dynamic(
                'SchedTuneConfig', 'sched_tune_config:', scope='sched')
        self.trappy_cls['sched_tune_tasks_update'] = trappy.register_dynamic(
                'SchedTuneTasksUpdate', 'sched_tune_tasks_update:',
                scope='sched')

    def __parseTrace(self, datadir, tasks):
        logging.debug('Loading [sched] events from trace in [%s]...', datadir)
        self.run = trappy.Run(datadir, scope="sched")

        # Check for events available on the parsed trace
        self.__checkAvailableEvents()
        if len(self.available_events) == 0:
            raise ValueError('The trace does not contain useful events')

        # Setup internal data reference to interesting events/dataframes

        if self.hasEvents('sched_switch'):
            self.trace_data['sswitch'] = \
                self.run.sched_switch.data_frame

        if self.hasEvents('sched_wakeup'):
            self.trace_data['swkp'] = \
                self.run.sched_wakeup.data_frame

        if self.hasEvents('sched_wakeup_new'):
            self.trace_data['swkpn'] = \
                self.run.sched_wakeup_new.data_frame

        if self.hasEvents('sched_cpu_frequency'):
            self.trace_data['pfreq'] = \
                self.run.sched_cpu_frequency.data_frame

        if self.hasEvents('sched_load_avg_cpu'):
            self.trace_data['cload'] = \
                self.run.sched_load_avg_cpu.data_frame

        if self.hasEvents('sched_load_avg_task'):
            self.trace_data['tload'] = \
                self.run.sched_load_avg_task.data_frame
            self.__addClusterColum()

        if self.hasEvents('cpu_capacity'):
            self.trace_data['ccap'] = \
                self.run.cpu_capacity.data_frame
            self.__addCapacityColum()

        if self.hasEvents('sched_boost_cpu'):
            self.trace_data['cboost'] = \
                self.run.sched_boost_cpu.data_frame
            self.__addCpuBoostColums()

        if self.hasEvents('sched_boost_task'):
            self.trace_data['tboost'] = \
                self.run.sched_boost_task.data_frame
            self.__addBoostedColum()

        if self.hasEvents('sched_contrib_scale_f'):
            self.trace_data['scalef'] = \
                self.run.sched_contrib_scale_f.data_frame

        if self.hasEvents('sched_energy_diff'):
            self.trace_data['ediff'] = \
                    self.run.sched_energy_diff.data_frame
            self.__addNormalizedEnergy()

        if self.hasEvents('sched_tune_config'):
            self.trace_data['stune'] = \
                    self.run.sched_tune_config.data_frame

        if self.hasEvents('sched_tune_tasks_update'):
            self.trace_data['utask'] = \
                    self.run.sched_tune_tasks_update.data_frame

        self.__loadTasksNames(tasks)


    def __checkAvailableEvents(self):
        for val in trappy.Run.get_filters(self.run):
            obj = getattr(self.run, val)
            if len(obj.data_frame):
                self.available_events.append(val)
        logging.debug('Events found on trace:')
        for evt in self.available_events:
            logging.debug(' - %s', evt)


    def __loadTasksNames(self, tasks):
        # Try to load tasks names using one of the supported events
        if 'sched_switch' in self.available_events:
            self.getTasks(self.trace_data['sswitch'], tasks,
                name_key='next_comm', pid_key='next_pid')
            return
        if 'sched_load_avg_task' in self.available_events:
            self.getTasks(self.trace_data['tload'], tasks)
            return
        logging.warning('Failed to load tasks names from trace events')

    def hasEvents(self, dataset):
        if dataset in self.available_events:
            return True
        return False

    def __computeTimeSpan(self):
        # Compute time axis range, considering all the parsed events
        ts = sys.maxint
        te = 0

        for key in self.trace_data.keys():
            df = self.trace_data[key]
            if len(df) == 0:
                continue
            if (df.index[0]) < ts:
                ts = df.index[0]
            if (df.index[-1]) > te:
                te = df.index[-1]
            self.time_range = te - ts

        logging.info('Collected events spans a %.3f [s] time interval',
                self.time_range)

    def getTasks(self, dataframe=None,
            task_names=None, name_key='comm', pid_key='pid'):
        # """ Helper function to get PIDs of specified tasks
        #
        #     This method requires a Pandas dataset in input to be used to
        #     fiter out the PIDs of all the specified tasks.
        #     In a dataset is not provided, previouslt filtered PIDs are
        #     returned.
        #     If a list of task names is not provided, the workload defined
        #     task names is used instead.
        #     The specified dataframe must provide at least two columns
        #     reporting the task name and the task PID. The default values of
        #     this colums could be specified using the provided parameters.
        #
        #     :param task_names: The list of tasks to get the PID of (by default
        #                        the workload defined tasks)
        #     :param dataframe: A Pandas datafram containing at least 'pid' and
        #                       'task name' columns. If None, the previously
        #                       filtered PIDs are returned
        #     :param name_key: The name of the dataframe columns containing
        #                      task names
        #     :param pid_key:  The name of the dataframe columns containing
        #                      task PIDs
        # """
        if dataframe is None:
            return self.tasks
        df = dataframe
        if task_names is None:
            task_names = self.tasks.keys()
        logging.debug("Lookup dataset for tasks...")
        for tname in task_names:
            logging.debug("Lookup for task [%s]...", tname)
            results = df[df[name_key] == tname][[name_key,pid_key]]
            if len(results)==0:
                logging.error('  task %16s NOT found', tname)
                continue
            (name, pid) = results.head(1).values[0]
            if name!=tname:
                logging.error('  task %16s NOT found', tname)
                continue
            if (tname not in self.tasks):
                self.tasks[tname] = {}
            pids = list(results[pid_key].unique())
            self.tasks[tname]['pid'] = pids
            logging.info('  task %16s found, pid: %s',
                    tname, self.tasks[tname]['pid'])
        return self.tasks

    def df(self, event):
        """
        Return the PANDAS dataframe with the performance data for the specified
        event
        """
        if self.datadir is None:
            raise ValueError("trace data not (yet) loaded")
        if event not in self.trace_data:
            raise ValueError('Event [{}] not supported. '\
                    'Supported events are: {}'\
                    .format(event, self.trace_data.keys()))
        return self.trace_data[event]

    def __addCapacityColum(self):
        df = self.df('ccap')

        # Add more columns if the energy model is available
        if 'nrg_model' not in self.platform:
            return

        # Add column with LITTLE and big CPUs max capacities
        nrg_model = self.platform['nrg_model']
        max_lcap = nrg_model['little']['cpu']['cap_max']
        max_bcap = nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [tip_lcap], tip_bcap)

    def __addClusterColum(self):
        df = self.df('tload')
        if 'utilization' in df:
            # Convert signals name from v5.0 to v5.1 format
            df.rename(columns={'utilization':'util_avg'}, inplace=True)
            df.rename(columns={'load':'load_avg'}, inplace=True)
            df.rename(columns={'avg_period':'period_contrib'}, inplace=True)
            df.rename(columns={'runnable_avg_sum':'load_sum'}, inplace=True)
            df.rename(columns={'running_avg_sum':'util_sum'}, inplace=True)
        df['cluster'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                ['LITTLE'], 'big')

    def __addCpuBoostColums(self):
        df = self.df('cboost')
        if 'usage' in df:
            # Convert signals name from to v5.1 format
            df.rename(columns={'usage':'util'}, inplace=True)
        df['boosted_util'] = df['util'] + df['margin']


    def __addBoostedColum(self):
        df = self.df('tboost')
        df['boosted_utilization'] = df['utilization'] + df['margin']

    def __addNormalizedEnergy(self):
        if 'nrg_model' not in self.platform:
            return
        nrg_model = self.platform['nrg_model']
        em_lcluster = nrg_model['little']['cluster']
        em_bcluster = nrg_model['big']['cluster']
        em_lcpu = nrg_model['little']['cpu']
        em_bcpu = nrg_model['big']['cpu']
        lcpus = len(self.platform['clusters']['little'])
        bcpus = len(self.platform['clusters']['big'])
        SCHED_LOAD_SCALE = 1024

        power_max = em_lcpu['nrg_max'] * lcpus + em_bcpu['nrg_max'] * bcpus + \
                    em_lcluster['nrg_max'] + em_bcluster['nrg_max']
        print "Maximum estimated system energy: {0:d}".format(power_max)

        df = self.df('ediff')
        df['nrg_diff_pct'] = SCHED_LOAD_SCALE * df.nrg_diff / power_max

        # Tag columns by usage_delta
        ccol = df.usage_delta
        df['usage_delta_group'] = np.select(
            [ccol < 150, ccol < 400, ccol < 600],
            ['< 150', '< 400', '< 600'], '>= 600')

        # Tag columns by nrg_payoff
        ccol = df.nrg_payoff
        df['nrg_payoff_group'] = np.select(
            [ccol > 2e9, ccol > 0, ccol > -2e9],
            ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject'], 'Suboptimal Reject')

