
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

class TraceAnalysis(object):

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

        # Minimum and Maximum x_time to use for all plots
        self.x_min = 0
        self.x_max = self.time_range

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

        # Standard Sched events
        # self.trappy_cls['cpu_frequency'] = trappy.register_dynamic(
        #         'CpuFrequency', 'cpu_frequency')

        # Additional (out-of-tree) custom events
        # self.trappy_cls['sched_load_avg_cpu'] = trappy.register_dynamic(
        #         'SchedLoadAvgCpu', 'sched_load_avg_cpu')
        # self.trappy_cls['sched_load_avg_task'] = trappy.register_dynamic(
        #         'SchedLoadAvgTask', 'sched_load_avg_task')
        self.trappy_cls['sched_contrib_scale_f'] = trappy.register_dynamic(
                'SchedContribScaleF', 'sched_contrib_scale_f')
        self.trappy_cls['cpu_capacity'] = trappy.register_dynamic(
                'CpuCapacity', 'cpu_capacity:', scope='sched')

        self.trappy_cls['sched_boost_cpu'] = trappy.register_dynamic(
                'SchedBoostCpu', 'sched_boost_cpu:', scope='sched')
        self.trappy_cls['sched_boost_task'] = trappy.register_dynamic(
                'SchedBoostTask', 'sched_boost_task:', scope='sched')
        self.trappy_cls['sched_energy_diff'] = trappy.register_dynamic(
                'SchedEnergyDiff', 'sched_energy_diff:', scope='sched')
        self.trappy_cls['sched_tune_config'] = trappy.register_dynamic(
                'SchedTuneConfig', 'sched_tune_config:', scope='sched')
        self.trappy_cls['sched_tune_tasks_update'] = trappy.register_dynamic(
                'SchedTuneTasksUpdate', 'sched_tune_tasks_update:', scope='sched')
        # self.trappy_cls['sched_task_model_update'] = trappy.register_dynamic(
        #         'sched_task_model_update', 'sched_task_model_update')

    def __parseTrace(self, datadir, tasks):
        logging.debug('Loading [sched] events from trace in [%s]...', datadir)
        self.run = trappy.Run(datadir, scope="sched")

        # Check for events available on the parsed trace
        self.__checkAvailableEvents()
        if len(self.available_events) == 0:
            raise ValueError('The trace does not contain useful events for analysis')

        # Setup internal data reference to interesting events/dataframes

        if self.__isAvailable('sched_switch'):
            self.trace_data['sswitch'] = \
                self.run.sched_switch.data_frame

        if self.__isAvailable('sched_cpu_frequency'):
            self.trace_data['pfreq'] = \
                self.run.sched_cpu_frequency.data_frame

        if self.__isAvailable('sched_load_avg_cpu'):
            self.trace_data['cload'] = \
                self.run.sched_load_avg_cpu.data_frame

        if self.__isAvailable('sched_load_avg_task'):
            self.trace_data['tload'] = \
                self.run.sched_load_avg_task.data_frame
            self.__addClusterColum()

        if self.__isAvailable('cpu_capacity'):
            self.trace_data['ccap'] = \
                self.run.cpu_capacity.data_frame
            self.__addCapacityColum()

        if self.__isAvailable('sched_boost_cpu'):
            self.trace_data['cboost'] = \
                self.run.sched_boost_cpu.data_frame
            self.__addCpuBoostColums()

        if self.__isAvailable('sched_boost_task'):
            self.trace_data['tboost'] = \
                self.run.sched_boost_task.data_frame
            self.__addBoostedColum()

        if self.__isAvailable('sched_contrib_scale_f'):
            self.trace_data['scalef'] = \
                self.run.sched_contrib_scale_f.data_frame

        if self.__isAvailable('sched_energy_diff'):
            self.trace_data['ediff'] = \
                    self.run.sched_energy_diff.data_frame
            self.__addNormalizedEnergy()

        if self.__isAvailable('sched_tune_config'):
            self.trace_data['stune'] = \
                    self.run.sched_tune_config.data_frame

        if self.__isAvailable('sched_tune_tasks_update'):
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

    def __isAvailable(self, dataset):
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
        # Reset x axis time range to full scale
        self.setXTimeRange()

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
        #                       'task name' columns
        #                      If None, the previously filtered PIDs are returned
        #     :param name_key: The name of the dataframe columns containing task names
        #     :param pid_key:  The name of the dataframe columns containing task PIDs
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
            (name,pid) = results.head(1).values[0]
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

    def setXTimeRange(self, t_min=None, t_max=None):
        if t_min is None:
            self.x_min = 0
        else:
            self.x_min = t_min
        if t_max is None:
            self.x_max = self.time_range
        else:
            self.x_max = t_max
        logging.info('Set plots time range to (%.6f, %.6f)[s]',
                self.x_min, self.x_max)

    def df(self, event):
        """
        Return the PANDAS dataframe with the performance data for the specified
        event
        """
        if self.datadir is None:
            raise ValueError("trace data not (yet) loaded")
        if event not in self.trace_data:
            raise ValueError('Event [%s] not supported. '\
                    'Supported events are: %s',
                    event, self.trace_data.keys())
        return self.trace_data[event]

    def plotClusterFrequencies(self):
        if not self.__isAvailable('sched_cpu_frequency'):
            logging.warn('Events [sched_cpu_frequency] not found, '\
                    'plot DISABLED!')
            return
        df = self.trace_data

        pd.options.mode.chained_assignment = None

        # Extract LITTLE and big clusters frequencies
        # and scale them to [MHz]
        if len(self.platform['clusters']['little']):
            lfreq = df['pfreq'][df['pfreq'].cpu == self.platform['clusters']['little'][-1]]
            lfreq['state'] = lfreq['state']/1e3
        else:
            lfreq = []
        if len(self.platform['clusters']['big']):
            bfreq = df['pfreq'][df['pfreq'].cpu == self.platform['clusters']['big'][-1]]
            bfreq['state'] = bfreq['state']/1e3
        else:
            bfreq = []

        # Compute AVG frequency for LITTLE cluster
        avg_lfreq = 0
        if len(lfreq) > 0:
            lfreq['timestamp'] = lfreq.index;
            lfreq['delta'] = (lfreq['timestamp'] - lfreq['timestamp'].shift()).fillna(0).shift(-1);
            lfreq['cfreq'] = (lfreq['state'] * lfreq['delta']).fillna(0);
            timespan = lfreq.iloc[-1].timestamp - lfreq.iloc[0].timestamp;
            avg_lfreq = lfreq['cfreq'].sum()/timespan;

        # Compute AVG frequency for big cluster
        avg_bfreq = 0
        if len(bfreq) > 0:
            bfreq['timestamp'] = bfreq.index;
            bfreq['delta'] = (bfreq['timestamp'] - bfreq['timestamp'].shift()).fillna(0).shift(-1);
            bfreq['cfreq'] = (bfreq['state'] * bfreq['delta']).fillna(0);
            timespan = bfreq.iloc[-1].timestamp - bfreq.iloc[0].timestamp;
            avg_bfreq = bfreq['cfreq'].sum()/timespan;

        pd.options.mode.chained_assignment = 'warn'

        # Setup a dual cluster plot
        fig, pltaxes = plt.subplots(2, 1, figsize=(16, 8));
        plt.suptitle("Clusters Frequencies",
                     y=.97, fontsize=16, horizontalalignment='center');

        # Plot Cluster frequencies
        axes = pltaxes[0]
        axes.set_title('big Cluster');
        if avg_bfreq > 0:
            axes.axhline(avg_bfreq, color='r', linestyle='--', linewidth=2);
        axes.set_ylim(
                (self.platform['freqs']['big'][0] - 100000)/1e3,
                (self.platform['freqs']['big'][-1] + 100000)/1e3
        );
        if len(bfreq) > 0:
            bfreq.state.plot(style=['r-'], ax=axes,
                    drawstyle='steps-post', alpha=0.4);
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_ylabel('MHz')
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')

        axes = pltaxes[1]
        axes.set_title('LITTLE Cluster');
        if avg_lfreq > 0:
            axes.axhline(avg_lfreq, color='b', linestyle='--', linewidth=2);
        axes.set_ylim(
                (self.platform['freqs']['little'][0] - 100000)/1e3,
                (self.platform['freqs']['little'][-1] + 100000)/1e3
        );
        if len(lfreq) > 0:
            lfreq.state.plot(style=['b-'], ax=axes,
                    drawstyle='steps-post', alpha=0.4);
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_ylabel('MHz')
        axes.grid(True);

        # Save generated plots into datadir
        figname = '{}/cluster_freqs.png'.format(self.datadir)
        pl.savefig(figname, bbox_inches='tight')

        logging.info('LITTLE cluster average frequency: %.3f GHz',
                avg_lfreq/1e3)
        logging.info('big    cluster average frequency: %.3f GHz',
                avg_bfreq/1e3)

    def __addCapacityColum(self):
        df = self.df('ccap')
        # Rename CPU and Capacity columns
        df.rename(columns={'cpu_id':'cpu'}, inplace=True)
        df.rename(columns={'state':'cur_capacity'}, inplace=True)
        # Add column with LITTLE and big CPUs max capacities
        max_lcap = self.platform.nrg_model['little']['cpu']['cap_max']
        max_bcap = self.platform.nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [tip_lcap], tip_bcap)

    def __plotCPU(self, cpus=None, label=''):
        if cpus is None or len(cpus) == 0:
            return
        if label != '':
            label1 = '{} '.format(label)
            label2 = '_{}s'.format(label.lower())
        df = self.trace_data

        # Plot required CPUs
        fig, pltaxes = plt.subplots(len(cpus), 1, figsize=(16, 3*(len(cpus))));
        plt.suptitle("{}CPUs Signals".format(label1),
                     y=.99, fontsize=16, horizontalalignment='center');

        idx = 0
        for cpu in cpus:

            # Reference axes to be used
            axes = pltaxes
            if (len(cpus) > 1):
                axes = pltaxes[idx]

            # Add CPU utilization
            axes.set_title('{0:s}CPU [{1:d}]'.format(label1, cpu));
            df1 = df['cload'][df['cload'].cpu == cpu]
            if (len(df1)):
                df1[['utilization']].plot(ax=axes, drawstyle='steps-post', alpha=0.4);

            if self.__isAvailable('sched_boost_cpu'):
                df2 = df['cboost'][df['cboost'].cpu == cpu]
                if (len(df2)):
                    df2[['util', 'boosted_util']].plot(
                            ax=axes,
                            style=['m-', 'r-'],
                            drawstyle='steps-post');

            # Add Capacities data if avilable
            if self.__isAvailable('cpu_capacity'):
                df2 = df['ccap'][df['ccap'].cpu == cpu]
                if (len(df2)):
                    data = df2[['cur_capacity', 'tip_capacity', 'max_capacity']]
                    data.plot(ax=axes, style=['m', 'y', 'r'],
                        drawstyle='steps-post')

            axes.set_ylim(0, 1100);
            axes.set_xlim(self.x_min, self.x_max);

            # Disable x-axis timestamp for top-most cpus
            if (len(cpus) > 1 and idx < len(cpus)-1):
                axes.set_xticklabels([])
                axes.set_xlabel('')
            axes.grid(True);

            idx+=1

        # Save generated plots into datadir
        figname = '{}/cpus{}.png'.format(self.datadir, label2)
        pl.savefig(figname, bbox_inches='tight')

    def plotCPU(self, cpus=None):
        if not self.__isAvailable('sched_load_avg_cpu'):
            logging.warn('Events [sched_load_avg_cpu] not found, '\
                    'plot DISABLED!')
            return

        # Filter on specified cpus
        if cpus is None:
            cpus = sorted(self.platform['clusters']['little'] + self.platform['clusters']['big'])

        # Plot: big CPUs
        bcpus = set(cpus) & set(self.platform['clusters']['big'])
        self.__plotCPU(bcpus, "big")

        # Plot: LITTLE CPUs
        lcpus = set(cpus) & set(self.platform['clusters']['little'])
        self.__plotCPU(lcpus, "LITTLE")

    def __addClusterColum(self):
        df = self.df('tload')
        df['cluster'] = np.select([df.cpu.isin(self.platform['clusters']['little'])], ['LITTLE'], 'big')

    def __addCpuBoostColums(self):
        df = self.df('cboost')
        df['boosted_util'] = df['util'] + df['margin']

    def __addBoostedColum(self):
        df = self.df('tboost')
        df['boosted_utilization'] = df['utilization'] + df['margin']

    def plotTasks(self, tasks=None):
        if not self.__isAvailable('sched_load_avg_task'):
            logging.warn('Events [sched_load_avg_task] not found, '\
                    'plot DISABLED!')
            return
        df = self.trace_data['tload']
        self.getTasks(df, tasks)
        tasks_to_plot = sorted(self.tasks)
        if tasks:
            tasks_to_plot = tasks

        # Grid
        gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1]);
        gs.update(wspace=0.1, hspace=0.1);

        for task_name in tasks_to_plot:
            logging.debug('Plotting [%s]', task_name)
            tid = self.tasks[task_name]['pid'][0]

            # Figure
            plt.figure(figsize=(16, 2*6+3));
            plt.suptitle("Task Signals",
                         y=.94, fontsize=16, horizontalalignment='center');

            # Plot load and utilization
            axes = plt.subplot(gs[0,0]);
            axes.set_title('Task [{0:d}:{1:s}] Signals'.format(tid, task_name));
            data = df[df.comm == task_name][['load', 'utilization']]
            data.plot(ax=axes, drawstyle='steps-post');
            # Plot boost utilization if available
            if self.__isAvailable('sched_boost_task'):
                df2 = self.trace_data['tboost']
                data = df2[df2.comm == task_name][['boosted_utilization']]
                if len(data):
                    data.plot(ax=axes, style=['y-'], drawstyle='steps-post');
            axes.set_ylim(0, 1100);
            axes.set_xlim(self.x_min, self.x_max);
            axes.grid(True);
            axes.set_xticklabels([])
            axes.set_xlabel('')

            # Plot CPUs residency
            axes = plt.subplot(gs[1,0]);
            axes.set_title('CPUs residency (green: LITTLE, red: big)');
            axes.set_xlim(0, self.time_range);
            data = df[df.comm == task_name][['cluster', 'cpu']]
            for ccolor, clabel in zip('gr', ['LITTLE', 'big']):
                cdata = data[data.cluster == clabel]
                if (len(cdata) > 0):
                    cdata.plot(ax=axes, style=[ccolor+'+'], legend=False);
            axes.set_ylim(-1, self.platform['cpus_count']+1)
            axes.set_xlim(self.x_min, self.x_max);
            axes.set_ylabel('CPUs')
            axes.grid(True);
            axes.set_xticklabels([])
            axes.set_xlabel('')

            # Plot PELT signals
            axes = plt.subplot(gs[2,0]);
            axes.set_title('PELT Signals');
            data = df[df.comm == task_name][['avg_period', 'runnable_avg_sum', 'running_avg_sum']]
            data.plot(ax=axes, drawstyle='steps-post');
            axes.set_xlim(self.x_min, self.x_max);
            axes.grid(True);

            # Save generated plots into datadir
            figname = '{}/task_util_{}.png'.format(self.datadir, task_name)
            pl.savefig(figname, bbox_inches='tight')

    def __addNormalizedEnergy(self):
        em_lcluster = self.platform.nrg_model['little']['cluster']
        em_bcluster = self.platform.nrg_model['big']['cluster']
        em_lcpu = self.platform.nrg_model['little']['cpu']
        em_bcpu = self.platform.nrg_model['big']['cpu']
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


    def plotEDiffTime(self, tasks=None,
            min_usage_delta=None, max_usage_delta=None,
            min_cap_delta=None, max_cap_delta=None,
            min_nrg_delta=None, max_nrg_delta=None,
            min_nrg_diff=None, max_nrg_diff=None):
        if not self.__isAvailable('sched_energy_diff'):
            logging.warn('Events [sched_energy_diff] not found, plot DISABLED!')
            return
        df = self.df('ediff')

        # Filter on 'tasks'
        if tasks is not None:
            logging.info('Plotting EDiff data just for task(s) [%s]', tasks)
            df = df[df['comm'].isin(tasks)]

        # Filter on 'usage_delta'
        if min_usage_delta is not None:
            logging.info('Plotting EDiff data just with minimum usage_delta of [%d]', min_usage_delta)
            df = df[abs(df['usage_delta']) >= min_usage_delta]
        if max_usage_delta is not None:
            logging.info('Plotting EDiff data just with maximum usage_delta of [%d]', max_usage_delta)
            df = df[abs(df['usage_delta']) <= max_usage_delta]

        # Filter on 'cap_delta'
        if min_cap_delta is not None:
            logging.info('Plotting EDiff data just with minimum cap_delta of [%d]', min_cap_delta)
            df = df[abs(df['cap_delta']) >= min_cap_delta]
        if max_cap_delta is not None:
            logging.info('Plotting EDiff data just with maximum cap_delta of [%d]', max_cap_delta)
            df = df[abs(df['cap_delta']) <= max_cap_delta]

        # Filter on 'nrg_delta'
        if min_nrg_delta is not None:
            logging.info('Plotting EDiff data just with minimum nrg_delta of [%d]', min_nrg_delta)
            df = df[abs(df['nrg_delta']) >= min_nrg_delta]
        if max_nrg_delta is not None:
            logging.info('Plotting EDiff data just with maximum nrg_delta of [%d]', max_nrg_delta)
            df = df[abs(df['nrg_delta']) <= max_nrg_delta]

        # Filter on 'nrg_diff'
        if min_nrg_diff is not None:
            logging.info('Plotting EDiff data just with minimum nrg_diff of [%d]', min_nrg_diff)
            df = df[abs(df['nrg_diff']) >= min_nrg_diff]
        if max_nrg_diff is not None:
            logging.info('Plotting EDiff data just with maximum nrg_diff of [%d]', max_nrg_diff)
            df = df[abs(df['nrg_diff']) <= max_nrg_diff]

        # Grid: setup stats for gris
        gs = gridspec.GridSpec(4, 3, height_ratios=[2,4,2,4]);
        gs.update(wspace=0.1, hspace=0.1);

        # Configure plot
        fig = plt.figure(figsize=(16, 8*2+4*2+2));
        plt.suptitle("EnergyDiff Data",
                     y=.92, fontsize=16, horizontalalignment='center');

        # Plot1: src and dst CPUs
        axes = plt.subplot(gs[0,:]);
        axes.set_title('Source and Destination CPUs');
        df[['src_cpu', 'dst_cpu']].plot(ax=axes, style=['bo', 'r+']);
        axes.set_ylim(-1, self.platform['cpus_count']+1)
        axes.set_xlim(self.x_min, self.x_max);
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')

        # Plot2: energy and capacity variations
        axes = plt.subplot(gs[1,:]);
        axes.set_title('Energy vs Capacity Variations');

        for color, label in zip('gbyr', ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject', 'Suboptimal Reject']):
            subset = df[df.nrg_payoff_group == label]
            if (len(subset) == 0):
                continue
            subset[['nrg_diff_pct']].plot(ax=axes, style=[color+'o']);
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_yscale('symlog')
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')

        # Plot3: energy payoff
        axes = plt.subplot(gs[2,:]);
        axes.set_title('Energy Payoff Values');
        for color, label in zip('gbyr', ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject', 'Suboptimal Reject']):
            subset = df[df.nrg_payoff_group == label]
            if (len(subset) == 0):
                continue
            subset[['nrg_payoff']].plot(ax=axes, style=[color+'o']);
        axes.set_xlim(self.x_min, self.x_max);
        axes.set_yscale('symlog')
        axes.grid(True);
        axes.set_xticklabels([])
        axes.set_xlabel('')

        # Plot4: energy deltas (kernel and host computed values)
        axes = plt.subplot(gs[3,:]);
        axes.set_title('Energy Deltas Values');
        df[['nrg_delta', 'nrg_diff_pct']].plot(ax=axes, style=['ro', 'b+']);
        axes.set_xlim(self.x_min, self.x_max);

        # Save generated plots into datadir
        figname = '{}/ediff_time.png'.format(self.datadir)
        pl.savefig(figname, bbox_inches='tight')


        # Grid: setup stats for gris
        gs = gridspec.GridSpec(1, 3, height_ratios=[2]);
        gs.update(wspace=0.1, hspace=0.1);

        fig = plt.figure(figsize=(16, 4));

        # Plot: usage, capacity and energy distributuions
        axes = plt.subplot(gs[0,0]);
        df[['usage_delta']].hist(ax=axes, bins=60)
        axes = plt.subplot(gs[0,1]);
        df[['cap_delta']].hist(ax=axes, bins=60)
        axes = plt.subplot(gs[0,2]);
        df[['nrg_delta']].hist(ax=axes, bins=60)

        # Save generated plots into datadir
        figname = '{}/ediff_stats.png'.format(self.datadir)
        pl.savefig(figname, bbox_inches='tight')


    def plotEDiffSpace(self, tasks=None,
            min_usage_delta=None, max_usage_delta=None,
            min_cap_delta=None, max_cap_delta=None,
            min_nrg_delta=None, max_nrg_delta=None,
            min_nrg_diff=None, max_nrg_diff=None):
        if not self.__isAvailable('sched_energy_diff'):
            logging.warn('Events [sched_energy_diff] not found, plot DISABLED!')
            return
        df = self.df('ediff')

        # Filter on 'tasks'
        if tasks is not None:
            logging.info('Plotting EDiff data just for task(s) [%s]', tasks)
            df = df[df['comm'].isin(tasks)]

        # Filter on 'usage_delta'
        if min_usage_delta is not None:
            logging.info('Plotting EDiff data just with minimum usage_delta of [%d]', min_usage_delta)
            df = df[abs(df['usage_delta']) >= min_usage_delta]
        if max_usage_delta is not None:
            logging.info('Plotting EDiff data just with maximum usage_delta of [%d]', max_usage_delta)
            df = df[abs(df['usage_delta']) <= max_usage_delta]

        # Filter on 'cap_delta'
        if min_cap_delta is not None:
            logging.info('Plotting EDiff data just with minimum cap_delta of [%d]', min_cap_delta)
            df = df[abs(df['cap_delta']) >= min_cap_delta]
        if max_cap_delta is not None:
            logging.info('Plotting EDiff data just with maximum cap_delta of [%d]', max_cap_delta)
            df = df[abs(df['cap_delta']) <= max_cap_delta]

        # Filter on 'nrg_delta'
        if min_nrg_delta is not None:
            logging.info('Plotting EDiff data just with minimum nrg_delta of [%d]', min_nrg_delta)
            df = df[abs(df['nrg_delta']) >= min_nrg_delta]
        if max_nrg_delta is not None:
            logging.info('Plotting EDiff data just with maximum nrg_delta of [%d]', max_nrg_delta)
            df = df[abs(df['nrg_delta']) <= max_nrg_delta]

        # Filter on 'nrg_diff'
        if min_nrg_diff is not None:
            logging.info('Plotting EDiff data just with minimum nrg_diff of [%d]', min_nrg_diff)
            df = df[abs(df['nrg_diff']) >= min_nrg_diff]
        if max_nrg_diff is not None:
            logging.info('Plotting EDiff data just with maximum nrg_diff of [%d]', max_nrg_diff)
            df = df[abs(df['nrg_diff']) <= max_nrg_diff]

        # Grid: setup grid for P-E space
        gs = gridspec.GridSpec(1, 2, height_ratios=[2]);
        gs.update(wspace=0.1, hspace=0.1);

        fig = plt.figure(figsize=(16, 8));

        # Get min-max of each axes
        x_min = df.nrg_diff_pct.min()
        x_max = df.nrg_diff_pct.max()
        y_min = df.cap_delta.min()
        y_max = df.cap_delta.max()
        axes_min = min(x_min, y_min)
        axes_max = max(x_max, y_max)


        # # Tag columns by usage_delta
        # ccol = df.usage_delta
        # df['usage_delta_group'] = np.select(
        #     [ccol < 150, ccol < 400, ccol < 600],
        #     ['< 150', '< 400', '< 600'], '>= 600')
        #
        # # Tag columns by nrg_payoff
        # ccol = df.nrg_payoff
        # df['nrg_payoff_group'] = np.select(
        #     [ccol > 2e9, ccol > 0, ccol > -2e9],
        #     ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject'], 'Suboptimal Reject')

        # Plot: per usage_delta values
        axes = plt.subplot(gs[0,0]);

        for color, label in zip('bgyr', ['< 150', '< 400', '< 600', '>= 600']):
            subset = df[df.usage_delta_group == label]
            if (len(subset) == 0):
                continue
            plt.scatter(subset.nrg_diff_pct, subset.cap_delta,
                        s=subset.usage_delta,
                        c=color, label='task_usage ' + str(label),
                        axes=axes)

        # Plot space axes
        plt.plot((0, 0), (-1025, 1025), 'y--', axes=axes)
        plt.plot((-1025, 1025), (0,0), 'y--', axes=axes)

        # # Perf cuts
        # plt.plot((0, 100), (0,100*delta_pb), 'b--', label='PB (Perf Boost)')
        # plt.plot((0, -100), (0,-100*delta_pc), 'r--', label='PC (Perf Constraint)')
        #
        # # Perf boost setups
        # for y in range(0,6):
        #     plt.plot((0, 500), (0,y*100), 'g:')
        # for x in range(0,5):
        #     plt.plot((0, x*100), (0,500), 'g:')

        axes.legend(loc=4, borderpad=1);

        plt.xlim(1.1*axes_min, 1.1*axes_max);
        plt.ylim(1.1*axes_min, 1.1*axes_max);

        # axes.title('Performance-Energy Space')
        axes.set_xlabel('Energy diff [%]');
        axes.set_ylabel('Capacity diff [%]');


        # Plot: per usage_delta values
        axes = plt.subplot(gs[0,1]);

        for color, label in zip('gbyr', ['Optimal Accept', 'SchedTune Accept', 'SchedTune Reject', 'Suboptimal Reject']):
            subset = df[df.nrg_payoff_group == label]
            if (len(subset) == 0):
                continue
            plt.scatter(subset.nrg_diff_pct, subset.cap_delta,
                        s=60,
                        c=color,
                        marker='+',
                        label='{} Region'.format(label),
                        axes=axes)
                        # s=subset.usage_delta,

        # Plot space axes
        plt.plot((0, 0), (-1025, 1025), 'y--', axes=axes)
        plt.plot((-1025, 1025), (0,0), 'y--', axes=axes)

        # # Perf cuts
        # plt.plot((0, 100), (0,100*delta_pb), 'b--', label='PB (Perf Boost)')
        # plt.plot((0, -100), (0,-100*delta_pc), 'r--', label='PC (Perf Constraint)')
        #
        # # Perf boost setups
        # for y in range(0,6):
        #     plt.plot((0, 500), (0,y*100), 'g:')
        # for x in range(0,5):
        #     plt.plot((0, x*100), (0,500), 'g:')

        axes.legend(loc=4, borderpad=1);

        plt.xlim(1.1*axes_min, 1.1*axes_max);
        plt.ylim(1.1*axes_min, 1.1*axes_max);

        # axes.title('Performance-Energy Space')
        axes.set_xlabel('Energy diff [%]');
        axes.set_ylabel('Capacity diff [%]');

        plt.title('Performance-Energy Space')

        # Save generated plots into datadir
        figname = '{}/ediff_space.png'.format(self.datadir)
        pl.savefig(figname, bbox_inches='tight')


    def plotSchedTuneConf(self):
        """
        Plot the configuration of the SchedTune
        """
        if not self.__isAvailable('sched_tune_config'):
            logging.warn('Events [sched_tune_config] not found, plot DISABLED!')
            return
        # Grid
        gs = gridspec.GridSpec(2, 1, height_ratios=[4,1]);
        gs.update(wspace=0.1, hspace=0.1);

        # Figure
        plt.figure(figsize=(16, 2*6));
        plt.suptitle("SchedTune Configuration",
                     y=.97, fontsize=16, horizontalalignment='center');

        # Plot: Margin
        axes = plt.subplot(gs[0,0]);
        axes.set_title('Margin');
        data = self.df('stune')[['margin']]
        data.plot(ax=axes, drawstyle='steps-post', style=['b']);
        axes.set_ylim(0, 110);
        axes.set_xlim(self.x_min, self.x_max);
        axes.xaxis.set_visible(False);

        # Plot: Boost mode
        axes = plt.subplot(gs[1,0]);
        axes.set_title('Boost mode');
        data = self.df('stune')[['boostmode']]
        data.plot(ax=axes, drawstyle='steps-post');
        axes.set_ylim(0, 4);
        axes.set_xlim(self.x_min, self.x_max);
        axes.xaxis.set_visible(True);

        # Save generated plots into datadir
        figname = '{}/schedtune_conf.png'.format(self.datadir)
        pl.savefig(figname, bbox_inches='tight')
