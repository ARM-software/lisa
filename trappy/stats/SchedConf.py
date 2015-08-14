#    Copyright 2015-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Scheduler specific Functionality for the
stats framework
"""

import numpy as np
from trappy.stats.Trigger import Trigger

WINDOW_SIZE = 0.0001

# Trigger Values
SCHED_SWITCH_IN = 1
SCHED_SWITCH_OUT = -1
NO_EVENT = 0

# Field Names
CPU_FIELD = "__cpu"
NEXT_PID_FIELD = "next_pid"
PREV_PID_FIELD = "prev_pid"
TASK_RUNNING = 1
TASK_NOT_RUNNING = 0
TIME_INVAL = -1
SERIES_SANTIZED = "_sched_sanitized"

def sanitize_asymmetry(series, window=None):
    """Sanitize the cases when a SWITCH_OUT
    happens before a SWITCH in. (The case when
    a process is already running before the started
    """

    if not hasattr(series, SERIES_SANTIZED):

        events = series[series != 0]
        if len(series) >= 2 and len(events):
            if series.values[0] == SCHED_SWITCH_OUT:
                series.values[0] = TASK_NOT_RUNNING

            elif events.values[0] == SCHED_SWITCH_OUT:
                series.values[0] = SCHED_SWITCH_IN
                if window:
                    series.index.values[0] = window[0]

            if series.values[-1] == SCHED_SWITCH_IN:
                series.values[-1] = TASK_NOT_RUNNING

            elif events.values[-1] == SCHED_SWITCH_IN:
                series.values[-1] = SCHED_SWITCH_OUT
                if window:
                    series.index.values[-1] = window[1]

        # No point if the series just has one value and
        # one event. We do not have sufficient data points
        # for any calculation. We should Ideally never reach
        # here.
        elif len(series) == 1:
            series.values[0] = 0

        setattr(series, SERIES_SANTIZED, True)

    return series


def csum(series, window=None, filter_gaps=False):
    """The following actions are done on the
    input series if filter_gaps is set as True

        * A sched_out of duration < WindowSize
          is filtered out
        * The resultant series is summed cumulatively
          and returned

        Args:
            series (pandas.Series)
            window (tuple)
            filter_gaps (boolean)
        Returns:
            pandas.Series
    """

    if filter_gaps:
        series = filter_small_gaps(series)

    series = series.cumsum()
    return select_window(series, window)

def filter_small_gaps(series):
    start = None
    for index, value in series.iteritems():

        if value == SCHED_SWITCH_IN:
            if start == None:
                continue

            if index - start < WINDOW_SIZE:
                series[start] = NO_EVENT
                series[index] = NO_EVENT
            start = None

        if value == SCHED_SWITCH_OUT:
            start = index

    return series

def first_cpu(series, window=None):
    """This aggreator returns the time of
    the first switch in event in the series
    This is returned as a vector of unit length
    so that it can be aggregated and reduced across
    nodes to find the first cpu of a task
    """
    series = select_window(series, window)
    series = series[series == SCHED_SWITCH_IN]
    if len(series):
        return [series.index.values[0]]
    else:
        return [float("inf")]

def select_window(series, window):
    """Library Function to select a portion of
       pandas time series
    """

    if not window:
        return series

    start, stop = window
    ix = series.index
    selector = ((ix >= start) & (ix <= stop))
    window_series = series[selector]
    return window_series

def residency_sum(series, window=None):
    """The input series is processed for
    intervals between a 1 and -1 in order
    to track additive residency of a task


        Args:
            series (pandas.Series)
            window (start, stop): A start stop
                tuple to process only a section of the
                series
        Returns:
            float (scalar)
    """

    if not len(series):
        return 0.0

    org_series = series
    series = select_window(series, window)
    series = sanitize_asymmetry(series, window)

    s_in = series[series == SCHED_SWITCH_IN]
    s_out = series[series == SCHED_SWITCH_OUT]

    if not (len(s_in) and len(s_out)):
        try:
            org_series = sanitize_asymmetry(org_series)
            running = select_window(org_series.cumsum(), window)
            if running.values[0] == TASK_RUNNING and running.values[-1] == TASK_RUNNING:
                return window[1] - window[0]
        except Exception,e:
            pass

    if len(s_in) != len(s_out):
        raise RuntimeError(
            "Unexpected Lengths: s_in={}, s_out={}".format(
                len(s_in),
                len(s_out)))
    else:
        return np.sum(s_out.index.values - s_in.index.values)


def first_time(series, value, window=None):
    """Return the first index where the
       series == value

       if no such index is found
       +inf is returned
    """

    series = select_window(series, window)
    series = series[series == value]

    if not len(series):
        return [float("inf")]

    return [series.index.values[0]]


def period(series, align="start", window=None):
    """Return a tuple of the average
       duration between SWITCH_IN (align=start)
       and SWITCH_OUT (align=end) and the number
       of events
    """

    series = select_window(series, window)
    series = sanitize_asymmetry(series, window)

    if align == "start":
        series = series[series == SCHED_SWITCH_IN]
    elif align == "end":
        series = series[series == SCHED_SWITCH_OUT]

    if len(series) % 2 == 0:
        series = series[:1]

    if not len(series):
        return [(0,0)]

    deltas = np.diff(series.index.values)
    return [(np.sum(deltas), len(deltas))]

def last_time(series, value, window=None):
    """Return the first index where the
       series == value

       if no such index is found
       TIME_INVAL is returned
    """

    series = select_window(series, window)
    series = series[series == value]
    if not len(series):
        return [TIME_INVAL]

    return [series.index.values[-1]]


def binary_correlate(series_x, series_y):
    """Function to Correlate binary Data"""

    if len(series_x) != len(series_y):
        raise ValueError("Cannot compute binary correlation for \
                          unequal vectors")

    agree = len(series_x[series_x == series_y])
    disagree = len(series_x[series_x != series_y])

    return (agree - disagree) / float(len(series_x))

def get_pids_for_process(run, execname, cls=None):
    """Returns the pids for a given process

    Args:
        run (trappy.Run): A trappy.Run object with a sched_switch
            event
        execname (str): The name of the process
        cls (trappy.Base): The SchedSwitch event class

    Returns:
        The list of pids (unique) for the execname

    """

    if not cls:
        try:
            df = run.sched_switch.data_frame
        except AttributeError:
            raise ValueError("SchedSwitch event not found in run")
    else:
        event = getattr(run, cls.name)
        df = event.data_frame

    mask = df["next_comm"].apply(lambda x : True if x.startswith(execname) else False)
    return list(np.unique(df[mask]["next_pid"].values))

def get_task_name(run, pid, cls=None):
    """Returns the execname for pid

    Args:
        run (trappy.Run): A trappy.Run object with a sched_switch
            event
        pid (str): The name of the process
        cls (trappy.Base): The SchedSwitch event class

    Returns:
        The execname for the PID

    """

    if not cls:
        try:
            df = run.sched_switch.data_frame
        except AttributeError:
           raise ValueError("SchedSwitch event not found in run")
    else:
        event = getattr(run, cls.name)
        df = event.data_frame

    df = df[df["__pid"] == pid]
    if not len(df):
        return ""
    else:
        return df["__comm"].values[0]

def sched_triggers(run, pid, sched_switch_class):
    """Returns the list of sched_switch triggers


    Args:
        run (trappy.Run): A run object with SchedSwitch event
        pid (int): pid of the associated task
        sched_switch_class (trappy.Base): The SchedSwitch class

    Returns:
        Lits of triggers
        [0] = switch_in_trigger
        [1] = switch_out_trigger
    """

    if not hasattr(run, "sched_switch"):
        raise ValueError("SchedSwitch event not found in run")

    triggers = []
    triggers.append(sched_switch_in_trigger(run, pid, sched_switch_class))
    triggers.append(sched_switch_out_trigger(run, pid, sched_switch_class))
    return triggers

def sched_switch_in_trigger(run, pid, sched_switch_class):
    """
    Args:
        run (trappy.Run): A run object with SchedSwitch event
        pid (int): pid of the associated task
        sched_switch_class (trappy.Base): The SchedSwitch class


    Returns:
        Trigger on the SchedSwitch: IN
    """

    task_in = {}
    task_in[NEXT_PID_FIELD] = pid

    return Trigger(run,
                   sched_switch_class,              # trappy Event Class
                   task_in,                         # Filter Dictionary
                   SCHED_SWITCH_IN,                 # Trigger Value
                   CPU_FIELD)                       # Primary Pivot

def sched_switch_out_trigger(run, pid, sched_switch_class):
    """
    Args:
        run (trappy.Run): A run object with SchedSwitch event
        pid (int): pid of the associated task
        sched_switch_class (trappy.Base): The SchedSwitch class


    Returns:
        Trigger on the SchedSwitch: OUT
    """

    task_out = {}
    task_out[PREV_PID_FIELD] = pid

    return Trigger(run,
                   sched_switch_class,              # trappy Event Class
                   task_out,                        # Filter Dictionary
                   SCHED_SWITCH_OUT,                # Trigger Value
                   CPU_FIELD)                       # Primary Pivot


def trace_event(series, window=None):
    """This aggregator returns a list of events
       of the type:

        {
            "start" : <start_time>,
             "end"   " <end_time>
        }

    """
    rects = []
    series = select_window(series, window)
    series = sanitize_asymmetry(series, window)

    s_in = series[series == SCHED_SWITCH_IN]
    s_out = series[series == SCHED_SWITCH_OUT]

    if not len(s_in):
        return rects

    if len(s_in) != len(s_out):
        raise RuntimeError(
            "Unexpected Lengths: s_in={}, s_out={}".format(
                len(s_in),
                len(s_out)))

    return np.column_stack((s_in.index.values, s_out.index.values))
