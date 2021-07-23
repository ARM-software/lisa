import simpy
import logging
import pandas as pd
import numpy as np
from lisa.utils import Loggable
from lisa._generic import TypedDict, TypedList, SortedTypedList
from lisa.conf import SimpleMultiSrcConf, KeyDesc, LevelKeyDesc, TopLevelKeyDesc, Configurable

class Task(Loggable):
    
    def __init__(self, env, name, sched=None, act_time=None, act_period=None, act_offset=None):
        self.env = env
        self.ev_preempt = env.event()
        self.ev_run = env.event()
        self.pe = None
        self.state = 'blocked'
        self.sched_state = TaskState.TASK_INTERRUPTIBLE
        self.sched = None
        self.activation_time = act_time
        self.activation_period = act_period
        self.activation_offset = act_offset
        self.name = name
        self.pid = None
        self.workload = None
        self.affinity = None
        
        self.logger = self.get_logger()

        if sched:
            self.set_scheduler(sched)
            
    def run(self):
        """
        Task model main loop
        """
        env = self.env
        sched = self.sched
        logger = self.logger

        # Wait until first activation, Task doesn't exist yet from an OS/scheduling point of view.
        if self.activation_offset:
            yield env.timeout(self.activation_offset)
                    
        while True:
            last_activation_time = env.now
            self.state = 'waiting'
            self.workload = Workload_Unit(self.activation_time, self, env.now)
            self.pe = sched.enqueue_task(self)
            
            while 'blocked' not in self.state:
                if 'running' not in self.state:
                    yield self.ev_run # Wait for task to be scheduled
                    self.ev_run = env.event()
                # Task running
                self.state = 'running'

                # Task runs until either completion of the activation or preemption
                completion = env.process(self.pe.execute_workload(self.workload))
                preempted = self.ev_preempt
                ret = yield completion | preempted
                if completion in ret:
                    self.state = 'blocked'
                else:
                    self.state = 'waiting'
                    # Stop task execution on PE
                    completion.interrupt()

            # Activation complete 
            sched.dequeue_task(self)

            # Sleep until next activation (wake-up)
            yield env.timeout(self.activation_period - (env.now - last_activation_time))
            
    def schedule_task(self, pe):
        """
        Start executing the task. Must be called from PE when task starts executing. Task state changes to running.
        """
        self.ev_run.succeed()
        
    def set_scheduler(self, Scheduler):
        """
        Assign scheduler class instance to the task.
        """
        if not self.sched:
            self.sched = Scheduler
            self.pid = Scheduler.add_task(self)
            self.affinity = Scheduler.default_affinity
            self.run_proc = self.env.process(self.run())
        else:
            self.sched = Scheduler

    def set_affinity(self, affinity):
        """
        Set task PE affinity.
        """
        self.affinity = affinity
            
    def preempt(self):
        """
        Preempt task. Must be called from scheduler when task is preempted. Causes task state to transition from running to waiting.
        """
        self.ev_preempt.succeed()
        self.ev_preempt = self.env.event()
        
class Scheduler(Loggable):
    
    def __init__(self, env, pes, trace):
        self.env = env
        self.tick_rate = 1e-3 # 1 ms scheduler tick
        self.sched_period = 4e-3
        self.logger = self.get_logger()
        self.trace = trace
        self.next_pid = 1
        
        self.rq = {}
        self.exec_start = []
        self.tasks = []
        self.pes = []
        idle_task = Task(env, 'swapper')
        idle_task.pid = 0
        idle_task.state = 'blocked'
        self.idle_task = idle_task
        self.default_affinity = {}
        
        for pe in pes:
            self.pes.append(pe)
            pe.init_sched_timer(self)
            pe.run_task(self.idle_task)
            pe.idle()
            self.exec_start.append(0)
            self.default_affinity[pe] = True
            self.rq[pe] = []
    
    def add_task(self, task):
        """
        Register task with scheduler.
        """
        pid = self.next_pid
        self.tasks.append(task)
        self.next_pid += 1
        self.trace.trace_sched_wakeup_new(self.pes[0], task, pid, self.pes[0])
        return pid
        
    def wake_balance(self, task):
        """
        Choose PE a task wake-up. Called internally in scheduler only.
        """
        for pe in self.pes:
            if pe.task == self.idle_task and task.affinity[pe]:
                return pe
        return self.pes[0]
        
    def enqueue_task(self, task):
        """
        Enqueue task on runqueue. Called by a task on activation.
        """
        waking_pe = self.pes[0]

        if task not in self.tasks:
            self.logger.error('Task not registered!')
            return
        
        task.sched_state = TaskState.TASK_RUNNING
        task.pe = self.wake_balance(task)
        self.rq[task.pe].append(task)
        
        self.trace.trace_sched_wakeup(waking_pe, task, task.pe)
        
        self.schedule(task.pe)
        return task.pe
        
    def dequeue_task(self, task):
        """
        Dequeue task from runqueue. Called by task when activation is complete.
        """
        task.sched_state = TaskState.TASK_INTERRUPTIBLE
        if task not in self.tasks:
            self.logger.warning(f'Task {task.name} not registered witch scheduler!')
            return
       
        if task in self.rq[task.pe]:
            self.rq[task.pe].remove(task)
        self.schedule(task.pe)
        return

    def newly_idle_balance(self, dst_pe):
        """
        Balance runqueues when dst_pe runqueue is empty. Called from schedule()
        """
        dst_rq = self.rq[dst_pe]

        if dst_rq:
            return

        for pe in self.pes:
            if pe == dst_pe:
                continue

            src_rq = self.rq[pe]

            if src_rq:
                for task in src_rq:
                    if not task.affinity[pe]:
                        continue
                    src_rq.remove(task)
                    dst_rq.append(task)
                    task.pe = pe
                    self.trace.trace_sched_migrate_task(pe, dst_pe, task)
                    return
    
    def schedule(self, pe):
        """
        Schedule PE. Decides if task switch is necessary.
        """
        rq = self.rq[pe]
        env = self.env
        
        prev_task = pe.task
        next_task = prev_task
            
        # Make sure all runnable tasks, including current, are on the rq.
        if not prev_task == self.idle_task and 'blocked' not in prev_task.state:
            rq.append(prev_task)

        # Nothing runnable, try pulling a task from another rq
        if not rq:
            self.newly_idle_balance(pe)

        # Nothing runnable, run idle-task
        if not rq:
            if prev_task == self.idle_task:
                return prev_task
            next_task = self.idle_task
            self.trace.trace_sched_switch(pe, pe.task, pe.task.sched_state, next_task)
            pe.run_task(next_task)
            self.trace.trace_cpu_idle(pe, 0)
            pe.idle()
            
            return next_task

        sched_slice = self.sched_period/len(rq)

        # We have at least one runnable task which could be the running task 
        if not prev_task.state == 'blocked' and env.now-self.exec_start[pe.cpuid] < sched_slice:
            rq.remove(prev_task)
            return prev_task

        next_task = rq.pop(0)

        if prev_task == next_task:
            return next_task

        # Found a new task to run
        if not prev_task == self.idle_task:
            prev_task.preempt()
        else:
            # Exiting idle-state
            self.trace.trace_cpu_idle(pe, 4294967295)

        self.exec_start[pe.cpuid] = env.now
        self.trace.trace_sched_switch(pe, pe.task, pe.task.sched_state, next_task)
        pe.run_task(next_task)
        
        return next_task
    
    def tick(self, pe):
        """
        Scheduler tick. Called periodically by PE timer.
        """
        self.schedule(pe)

    def pe_next_tick(self, PE):
        """
        Returns time until next desired scheduler tick.
        """
        return self.tick_rate

class PE(Loggable):
    
    def __init__(self, env, cpuid, trace, capacity=1024, dvfs_capacities=None):
        self.env = env
        self.task = None
        self.cpuid = cpuid
        self.logger = self.get_logger()
        self.work_start = None
        self.trace = trace

        self.capacity_current = capacity
        self.capacities = dvfs_capacities
        self.ev_capacity_change = env.event()
    
    def run_task(self, task):
        """
        Execute new task.

        :param task: New task execute.
        """

        self.task = task

        if not task.pid == 0:
            task.schedule_task(self)
        
    def idle(self):
        """
        PE is idle. Called from :class:`Scheduler`.
        """
        return

    def execute_workload(self, workload_unit):
        """
        Set workload unit to execute. Called from :class:`Task` when the task
        starts running.
        """
        try:
            while True:
                # Execute workload unit either completion or PE compute capacity changes.
                completion_ev = self.env.timeout(self._predict_completion(workload_unit))
                workload_unit.work_start = self.env.now
                capacity_ev = self.ev_capacity_change
                capacity_start = self.capacity_current
                ret = yield completion_ev | capacity_ev
                self._workload_progress(workload_unit, workload_unit.work_start, self.env.now, capacity_start)
                if completion_ev in ret:
                    # Activation (workload_unit) has been completed.
                    self.trace.trace_sim_activation_complete(self, self.task, workload_unit.work_created, self.env.now)
                    break

        except simpy.Interrupt as i:
            # Task/workload unit has been preempted by the scheduler.
            self._workload_progress(workload_unit, workload_unit.work_start, self.env.now, capacity_start)

    def set_capacity(self, capacity):
        """
        Set compute capacity of PE.
        """
        self.capacity_current = capacity
        self.ev_capacity_change.succeed()
        self.ev_capacity_change = self.env.event()
        self.trace.trace_cpu_frequency(self, capacity)

    def _predict_completion(self, workload_unit):
        """
        Calculate when workload_unit will complete at current capacity assuming
        no changes in capacity.
        """
        work_left = workload_unit.work
        return work_left * 1024 /  self.capacity_current

    def _workload_progress(self, workload_unit, start, end, capacity):
        """
        Account for workload unit progress.

        :param start: Last update time-stamp.
        :param end: Update until this time-stamp.
        """
        delta = end - start
        delta = delta * capacity / 1024

        if delta >= workload_unit.work:
            workload_unit.work = 0
        else:
            workload_unit.work = workload_unit.work - delta
    
    def _timer_tick(self):
        """
        Simpy process for generating timer ticks.
        """
        while True:
            yield self.env.timeout(self.timer_tick_next)
            self.timer_tick_next = self.sched.pe_next_tick(self)
            self.sched.tick(self)
            
    def init_sched_timer(self, scheduler):
        """
        Initialize PE scheduler timer tick.
        """
        self.sched = scheduler
        
        next_tick = scheduler.pe_next_tick(self)
        if not next_tick:
            return
        
        self.timer_tick_next = next_tick
        self.proc_sched_tick = self.env.process(self._timer_tick())

class Workload_Unit():
    """
    Work associated with a single task activation. Every time a Task activates
    the work described in the Workload_Unit must be executed to complete the activation.
    Workload_Unit describes the amount of work to be completed, the PE model determines 
    the rate of progress and tracks forward progress by reducing the amount of work 
    left in the Workload_Unit.
    """
    def __init__(self, work, task, now):
        # Work is currently just "busy time" @ max capacity.
        # Ideally, it should be number of instuctions split into relevant
        # instruction types.
        self.work = work # Busy time
        self.work_start = None
        self.work_created = now
        self.task = task

class Capacity_Scaling_gov(Loggable):
    """
    Frequency scaling governor. Proof of concept simply cycling through available OPPs.
    """
    def __init__(self, env, pes):
        self.env = env
        self.pes = pes # List of PEs in frequency domain
        self.logger = self.get_logger()
        self.policy = self.env.process(self.policy())

    def policy(self):
        cur_cap = {}
        for pe in self.pes.values():
            cur_cap[pe] = len(pe.capacities)-1
        while True:
            yield self.env.timeout(0.001)
            for pe in self.pes.values():
                cur_cap[pe] = (cur_cap[pe]+1) % len(pe.capacities)
                pe.set_capacity(pe.capacities[cur_cap[pe]])
        
class Trace():
    """
    Linux kernel ftrace compatible tracing.
    """
    def __init__(self, env):
        self.env = env
        self.last_event_time = -1
        self.buffers = {
            'sched_wakeup' : [],
            'sched_wakeup_new' : [],
            'sched_switch' : [],
            'sched_migrate_task' : [],
            'cpu_frequency' : [],
            'cpu_idle' : [],
            'sim_activation_complete' : [],
        }
        self.columns = {
            'sched_wakeup' : ('Time', '__cpu', '__pid', '__comm', 'comm', 'pid', 'prio', 'target_cpu'),
            'sched_wakeup_new' : ('Time', '__cpu', '__pid', '__comm', 'comm', 'pid', 'prio', 'success', 'target_cpu'),
            'sched_switch' : ('Time', '__cpu', '__pid', '__comm', 'prev_comm', 'prev_pid', 'prev_prio', 'prev_state', 'next_comm', 'next_pid', 'next_prio'),
            'sched_migrate_task' : ('Time', '__cpu', '__pid', '__comm', 'comm', 'dest_cpu', 'orig_cpu', 'pid', 'prio'),
            'cpu_frequency' : ('Time', '__cpu', '__pid', '__comm', 'cpu_id', 'state'),
            'cpu_idle' : ('Time', '__cpu', '__pid', '__comm', 'cpu_id', 'state'),
            'sim_activation_complete' : ('Time', '__cpu', '__pid', '__comm', 'cpu_id', 'pid', 'start', 'end'),
        }
        
    def get_dfs(self):
        """
        Return LISA compatible Pandas dataframes with recorded trace events.
        """
        buffers = self.buffers
        columns = self.columns
        
        dfs = {}
        for event in buffers.keys():
            dfs[event] = pd.DataFrame.from_records(buffers[event], columns=columns[event], index='Time')
        
        return dfs
    
    def unique_time_stamp(self):
        """
        Return unique time stamp for trace events.
        LISA requires all time-stamps to be unique.
        """
        env = self.env
        if env.now > self.last_event_time:
            self.last_event_time = env.now
        else:
            self.last_event_time = np.nextafter(self.last_event_time, np.inf)
            
        return self.last_event_time
    
        
    def trace_sched_wakeup(self, waker_pe, task, target_pe):
        if not waker_pe.task:
            waker_pid = 0
            waker_comm = 'swapper'
        else:
            waker_pid = waker_pe.task.pid
            waker_comm = waker_pe.task.name
        task_prio = 100
        self.buffers['sched_wakeup'].append((self.unique_time_stamp(), waker_pe.cpuid, waker_pid, waker_comm, task.name, task.pid, task_prio, target_pe.cpuid))

    def trace_sched_wakeup_new(self, waker_pe, task, task_pid, target_pe):
        if not waker_pe.task:
            waker_pid = 0
            waker_comm = 'swapper'
        else:
            waker_pid = waker_pe.task.pid
            waker_comm = waker_pe.task.name
        task_prio = 100
        success = 1
        self.buffers['sched_wakeup_new'].append((self.unique_time_stamp(), waker_pe.cpuid, waker_pid, waker_comm, task.name, task_pid, task_prio, success, target_pe.cpuid))
        
    def trace_sched_switch(self, cur_pe, prev_task, prev_state, next_task):
        if not cur_pe.task:
            cur_pid = 0
            cur_comm = 'swapper'
        else:
            cur_pid = cur_pe.task.pid
            cur_comm = cur_pe.task.name
        if not prev_task:
            prev_pid = 0
            prev_comm = 'swapper'
        else:
            prev_pid = prev_task.pid
            prev_comm = prev_task.name
        if not next_task:
            next_pid = 0
            next_comm = 'swapper'
        else:
            next_pid = next_task.pid
            next_comm = next_task.name
        prev_prio = 100
        next_task_prio = 100
        
        self.buffers['sched_switch'].append((self.unique_time_stamp(), cur_pe.cpuid, cur_pid, cur_comm, prev_comm, prev_pid, prev_prio, prev_state, next_comm, next_pid, next_task_prio))

    def trace_sched_migrate_task(self, prev_pe, next_pe, task):
        if not prev_pe.task:
            cur_pid = 0
            cur_comm = 'swapper'
        else:
            cur_pid = prev_pe.task.pid
            cur_comm = prev_pe.task.name
        task_prio = 100
        self.buffers['sched_migrate_task'].append((self.unique_time_stamp(), prev_pe.cpuid, cur_pid, cur_comm, task.name, next_pe.cpuid, prev_pe.cpuid, task.pid, task_prio))

    def trace_cpu_frequency(self, pe, capacity):
        cur_pid = pe.task.pid
        cur_comm = pe.task.name
        freq = int(capacity * 1e6)
        self.buffers['cpu_frequency'].append((self.unique_time_stamp(), pe.cpuid, cur_pid, cur_comm, pe.cpuid, freq))

    def trace_cpu_idle(self, pe, state):
        cur_pid = pe.task.pid
        cur_comm = pe.task.name
        self.buffers['cpu_idle'].append((self.unique_time_stamp(), pe.cpuid, cur_pid, cur_comm, pe.cpuid, state))

    def trace_sim_activation_complete(self, pe, task, start, end):
        self.buffers['sim_activation_complete'].append((self.unique_time_stamp(), pe.cpuid, pe.task.pid, pe.task.name, pe.cpuid, task.pid, start, end))


class TaskState():
    TASK_RUNNING = 0x0000
    TASK_INTERRUPTIBLE = 0x0001
    TASK_UNINTERRUPTIBLE = 0x002

class SimulationConf(SimpleMultiSrcConf):
    """
    YAML configuration helper.
    """
    CPUCapacities = TypedDict[int,int]
    FreqList = SortedTypedList[int]
    CPUIdList = SortedTypedList[int]

    STRUCTURE = TopLevelKeyDesc('tpp-conf', 'Simulation configuration', (
        KeyDesc('name', 'Simulation instance name', [str]),
        LevelKeyDesc('cpu', 'PE properties', (
            KeyDesc('capacities', 'Compute capacities of CPUs', [CPUCapacities]),
            KeyDesc('freqs', 'CPU performance levels', [TypedDict[int,FreqList]]),
            KeyDesc('perf-domains', 'CPU performance domain', [TypedList[CPUIdList]]),
        )),
        LevelKeyDesc('workload', 'workload properties', (
            KeyDesc('tasks', 'tasks (busy_time [s], period [s], offset [s])', [TypedDict[int,TypedList[float]]]),
        ))
    ))

class Simulation(Configurable, Loggable):
    """
    Sets up and configures Simulation.
    """
    CONF_CLASS = SimulationConf
    INIT_KWARGS_KEY_MAP = {
        'cpu_capacities': ['cpu','capacities'],
        'cpu_freqs': ['cpu','freqs'],
        'cpu_perf_domains': ['cpu','perf-domains'],
        'task_defs': ['workload','tasks'],
    }

    def __init__(self, name='<noname>', cpu_capacities=None, cpu_freqs=None, cpu_perf_domains=None, task_defs=None):
        super().__init__()
        self.name = name
        self.cpu_capacities = cpu_capacities
        self.cpu_freqs = cpu_freqs
        self.cpu_perf_domains_conf = cpu_perf_domains

        self.pes = {}
        self.env = simpy.Environment()
        self.trace = Trace(self.env)
        self.tasks = []

        self.scaling_govs = []
        self.cpu_perf_doms = []

        self._create_system()
        self._setup_workload(task_defs)

    @classmethod
    def from_conf(cls, conf, **kwargs):
        kwargs = cls.conf_to_init_kwargs(conf)
        cls.check_init_param(**kwargs)
        print(conf.pretty_format())

        return cls(**kwargs)

    def _create_system(self):
        for pe in self.cpu_capacities.items():
            cpuid, capacity = pe
            self.pes[cpuid] = PE(self.env, cpuid, self.trace, capacity=capacity, dvfs_capacities=self.cpu_freqs[cpuid])

        self.scheduler = Scheduler(self.env, self.pes.values(), self.trace)

        for perf_dom in self.cpu_perf_domains_conf:
            pes = {}
            for cpuid in perf_dom:
                pes[cpuid] = self.pes[cpuid]

            self.scaling_govs.append(Capacity_Scaling_gov(self.env, pes))

    def _setup_workload(self, task_defs):
        for t in task_defs.items():
            task_name, prop = t
            act_time, act_period, act_offset = prop
            task = Task(self.env, str(task_name), self.scheduler, act_time=act_time, act_period=act_period, act_offset=act_offset)
            self.tasks.append(task)

    def run(self, until):
        self.get_logger().warning(f'Running simulation...')
        self.env.run(until=until)
        self.get_logger().warning(f'Simulated until {self.env.now}s')
