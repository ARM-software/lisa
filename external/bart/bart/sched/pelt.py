#    Copyright 2015-2016 ARM Limited
#    Copyright 2016 Google Inc. All Rights Reserved
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

"""Simple PELT Simulator

This module provides a simple yet useful simulator of the Per-Entity Load
Tracking (PELT) machinery used by the Linux kernel to track the CPU bandwidth
demand of scheduling entities.

This model is based on some simplification assumptions:

* It supports only periodic tasks, which have a constant period and duty-cycle.
* All the time metrics are defined using "PELT samples" (1024us by default) as
  fundamental unit of measure

The simulator is composed of two main components: a PeriodicTask model and a signal
Simulator. Both classes have configuration parameters to defined their specific
behavior. Specifically, the Simulator requires also the PeriodicTask which signal
has to be computed.
"""
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from builtins import range
from builtins import object
import math

from collections import namedtuple as namedtuple
from pandas import DataFrame as DataFrame


##########################################################################
# PeriodicTask Model
##########################################################################

class PeriodicTask(object):

    def __init__(self, period_samples, start_sample=0,
                 run_samples=None, duty_cycle_pct=None,
                 pelt_sample_us=1024, pelt_max=1024):
        """A simulated periodic task

        Task activations are specified in terms of PELT samples.
        A [PELT sample] is by default 1024 us long, this is the time interval
        used by the Linux scheduler to update a PELT signal.

        :param start_sample: task start time [pelt_sample_us]
        :type start_sample: int

        :param period_samples: the period of each activation [pelt_sample_us]
        :type period_samples: int

        :param run_samples: the duration of each activation [pelt_sample_us]
                            This parameter is mutually exclusive with
                            duty_cycle_pct
        :type run_samples: int

        :param duty_cycle_pct: the duty-cycle of each activation
                               This parameter is mutually exclusive with the
                               run_samples
        :type duty_cycle_pct: int

        :param pelt_sample_us: the duration of a PELT sample, 1024us by default
        :type pelt_sample_us: int

        :param pelt_max: the maximum value of the PELT signal, 1024 by default
        :type pelt_max: int
        """

        args = [period_samples, start_sample, pelt_sample_us, pelt_max]
        invalid = any(not isinstance(param, int) or param < 0
                      for param in args)
        if invalid:
            raise ValueError(
                'one of more parameters are not positive integers')
        if run_samples is None and duty_cycle_pct is None:
            raise ValueError('undefined run_samples or duty_cycle_pct')
        if period_samples == 0:
            raise ValueError('period_samples cannot be 0')
        if run_samples and run_samples > period_samples:
            raise ValueError('run_samples bigger than period_samples')

        self.pelt_sample_us = pelt_sample_us
        self.pelt_max = pelt_max

        self.start_sample = start_sample
        self.start_us = self.pelt_sample_us * start_sample
        self.period_samples = period_samples
        self.period_us = self.pelt_sample_us * period_samples

        if duty_cycle_pct is not None:
            self.run_samples = period_samples * float(duty_cycle_pct) / 100
            self.duty_cycle_pct = duty_cycle_pct
        else:
            self.run_samples = run_samples
            self.duty_cycle_pct = int(100. * run_samples / period_samples)
        self.run_us = self.pelt_sample_us * self.run_samples

        self.idle_us = self.period_us - self.run_us
        self.pelt_avg = int(float(self.pelt_max) * self.duty_cycle_pct / 100)

    def isRunning(self, time_ms):
        """Return whether the task is running in the specified time instant.

        The current implementation provide support only for periodic tasks
        which have the following (abstract) execution model:
        ::

            +----+    +----+      Task is RUNNING
            |    |    |    |
                 +----+    +----  Task is SLEEPING

           -+----+----+----+-------------------------- Time --->
            1    2    3    4      PELT Samples
           ^    ^    ^    ^       Task running sampling time
           R    S    R    S       Task Status reported by this function


        Since we sample exactly at PELT samples intervals, the exact time when
        a task starts to run we consider it as if if was running for the
        entire previous PELT interval. To fix this, by exploting the knowledge
        that a task is always configured using an integer number of run/speep
        PELT samples, we can look at the 1us before time wrt the specified
        sample time.
        """
        # The 1us "before time" is always represeting the exact status of the
        # task in that period. This is why we remove 1 in the following
        # computation:
        period_delta_us = (float(time_ms) * 1e3 - 1) % self.period_us
        if period_delta_us <= self.run_us:
            return True
        return False

    def __str__(self):
        return "PeriodicTask(start: {:.3f} [ms], period: {:.3f} [ms], run: {:.3f} [ms], "\
               "duty_cycle: {:.3f} [%], pelt_avg: {:d})"\
            .format(self.start_us/1e3, self.period_us/1e3,
                    self.run_us/1e3, self.duty_cycle_pct,
                    self.pelt_avg)


##########################################################################
# PELT Model
##########################################################################

_PELTStats = namedtuple('PELTStats', [
    'start_s', 'end_s',
    'pelt_avg', 'pelt_init', 'half_life',
    'tmin', 'tmax', 'min', 'max', 'avg', 'std',
    'err', 'err_pct'])


class PELTStats(_PELTStats):
    """Statistics of a computed PELT signal

    This is a collection of metrics computed on a simulated PELT signal:



    =============   ================================================================
     Metric         Description
    =============   ================================================================
    ``start_s``     Start time [s] used for PELT signal simulation
    ``end_s``       End time [s] used for PELT signal simulation
    ``pelt_avg``    Expected average value of the PELT signal
    ``pelt_init``   Initial PELT value
    ``half_life``   End time [ms] of the PELT's half-life parameter
    ``tmin``        Start time [s] used for statistics computation
    ``tmax``        End time [s] used for statistics computation
    ``min``         Minimum PELT value, within [tmin, tmax]
    ``max``         Maximum PELT value, within [tmin, tmax]
    ``avg``         Average of the PELT values, within [tmin, tmax]
    ``std``         Standard deviation of the PELT value, within [tmin, tmax]
    ``err``         Difference between pelt_avg and avg
    ``err_pct``     The avg error percentage, compared to the expected average (pelt_avg)
    =============   ================================================================

    :note: the [start_s, end_s] interval used for statistical computations can be
           smaller than the timespan of the simulated PELT signal. For example, we
           can simulate a PELT signal for 3 [s] but compute statistics only for the
           last 1 [s]. This is allows for example to ignore the initial execution
           of a task while focusing for stats only on a timeframe in which the
           signal can be considered stable.
    """
    pass

_PELTRange = namedtuple('PELTRange', [
    'min_value', 'max_value'])


class PELTRange(_PELTRange):
    """Stability range for the PELT signal of a given PeriodicTask

    Given a PeriodicTask and a specific PELT configuration the simulated PELT
    signal is expected to stabilize in the range [min_value, max_value].

    Also, store the information about when the signal starts becoming stable.
    """
    pass


class Simulator(object):
    """A simple PELT simulator

    This PELT simulator allows to compute and plot the PELT signal for a
    specifed (periodic) task. The simulator can be configured using the main
    settings exposed by PELT: the half-life of a task and its intial value.

    The model is pretty simple and works by generating a new value at each
    PELT sample interval (_sample_us) specified in [us] (1024 by default).

    The model allows to compute statistics on the generated signal, over a
    specified time frame. It provides also an embedded support to plot the
    generated signal along with a set of expected thresholds and values
    (e.g. half-life, min/max stability values)

    The capping of the decay is an experimental features which is also
    supported by the current model.
    """

    _sample_us = 1024
    _signal_max = 1024

    def __init__(self, init_value=0, half_life_ms=32, decay_cap_ms=None):
        """Initialize the PELT simulator for a specified task.

        :param init_value: the initial PELT value for the task
        :type  init_value: int

        :param half_life_ms: the [ms] interval required by a signal to
                             increased/decreased by half of its range
        :type  half_life_ms: int

        :param decay_cap_ms: the [ms] interval after which we do not decay
                             further the PELT signal
        :type  decay_cap_ms: int

        """
        self.init_value = init_value
        self.half_life_ms = half_life_ms
        self.decay_cap_ms = decay_cap_ms

        self._geom_y = pow(0.5, 1 / half_life_ms)
        self._geom_u = float(self._signal_max) * (1. - self._geom_y)

        self.task = None
        self._df = None

    def __str__(self):
        desc = "PELT Simulator configured with\n"
        desc += "  initial value        : {}\n".format(self.init_value)
        desc += "  half life (HL)  [ms] : {}\n".format(self.half_life_ms)
        desc += "  decay capping @ [ms] : {}\n".format(self.decay_cap_ms)
        desc += "  y =    0.5^(1/HL)    : {:6.3f}\n".format(self._geom_y)
        desc += "  u = {:5d}*(1 - y)    : {:6.3f}\n".format(self._signal_max,
                                                            self._geom_u)
        return desc

    def _geomSum(self, u_1, active_us=0):
        """Geometric add the specified value to the u_1 series.

        NOTE: the current implementation assume that the task was active
              (active_us != 0) or sleeping (active_us == 0) a full PELT
              sampling interval (by default 1024[us])
        """
        # Decay previous samples
        u_1 *= self._geom_y
        # Add current sample (if task was active)
        if active_us:
            u_1 += self._geom_u
        return u_1

    def stableRange(self, task):
        """Compute the PELT's signal stability ranges for the specified task

        Here we use:

        =====================  ======================================
         Value                  Definition
        =====================  ======================================
        Half Like              :math:`\lambda`
        Decay factor           :math:`y = 0.5^{(1/\lambda)}`
        Max stable value       :math:`(1-y^r) / (1-y^p)`
        Min stable value       :math:`y^i \\times (1-y^r) / (1-y^p)`
        =====================  ======================================

        Where:

            * :math:`r` is the run time of the task in number of PELT samples
            * :math:`p` is the period of the task in number of PELT samples.
            * :math:`i` is the idle time of the task in number of PELT samples i.e. :math:`i = p - r`

        :param task: the task we want to simulate the PELT signal for, by
                     default is the task used to initialize the Simulator
                     (when specified)
        :type  task: PeriodicTask

        :return: :mod:`PELTRange` instance representing the minimum and maximum
                 value of the PELT signal once stable.
        """

        # Validate input parameters
        if not isinstance(task, PeriodicTask):
            raise ValueError("Wrong time for task parameter")

        def _to_pelt_samples(time_us):
            return time_us / self._sample_us

        # Compute max value
        max_pelt = (1. - pow(self._geom_y, _to_pelt_samples(task.run_us)))
        max_pelt /= (1. - pow(self._geom_y, _to_pelt_samples(task.period_us)))

        # Compute min value, by decaying the maximum value
        min_pelt = max_pelt
        min_pelt *= pow(self._geom_y, _to_pelt_samples(task.idle_us))

        min_pelt *= self._signal_max
        max_pelt *= self._signal_max

        return PELTRange(min_pelt, max_pelt)

    def stableTime(self, task):
        """
        Compute instant of time after which the signal can be considered to be
        stable.

        :returns: float - time after which the signal is stable
        """
        stable_range = self.stableRange(task)
        min_pelt = stable_range.min_value
        max_pelt = stable_range.max_value

        # Validate input parameters
        if self._df is None:
            raise ValueError("no signal computed, run getSignal before")

        if self.init_value < max_pelt:
            # Look for the max (with 0.1% tolerance)
            cond = (self._df.pelt_value > max_pelt * 0.999) & \
                   (self._df.pelt_value <= max_pelt)
        else:
            # Look for the min (with 0.1% tolerance)
            cond = (self._df.pelt_value < min_pelt * 1.001) & \
                   (self._df.pelt_value >= min_pelt)
        return self._df[cond].index[0]

    def getSignal(self, task, start_s=0, end_s=10):
        """Compute the PELT signal for the specified task and interval.

        :param task: the task we want to simulate the PELT signal for, by
                     default is the task used to intialize the Simulator
                     (when specified)
        :type  task: PeriodicTask

        :param start_s: the start time in [s]
        :type  start_s: float

        :param end_s: the end time in [s]
        :type  end_s: float

        :return: :mod:`pandas.DataFrame` instance which reports the computed
                 PELT values at each PELT sample interval. The returned columns are:

                 ================= ==============================================
                 Column             Description
                 ================= ==============================================
                 Time              The PELT sample time
                 PELT_Interval     The PELT sample number
                 Running           A boolean reporting if the task was RUNNING in that
                                   PELT interval
                 pelt_value        The computed PELT signal value at the end of that PELT
                                   interval
                 ================= ==============================================
        """

        # Computed PELT samples
        samples = []
        # Decay capping support
        running_prev = True
        capping = None

        # Validate input parameters
        if not isinstance(task, PeriodicTask):
            raise ValueError("Wrong type for task parameter")

        # Intervals (in PELT samples) for the signal to compute
        self.start_us = _s_to_us(start_s, self._sample_us, nearest_up=False)
        self.end_us = _s_to_us(end_s, self._sample_us)

        # Add initial value to the generated signal
        pelt_value = self.init_value
        t_us = self.start_us
        sample = (_us_to_s(t_us), 0, False, pelt_value)
        samples.append(sample)

        # Compute following PELT samples
        t_us = self.start_us + self._sample_us
        while t_us <= self.end_us:

            # Check if the task was running in the current PELT sample
            running = task.isRunning(_us_to_ms(t_us))

            # Keep track of sleep start and decay capping time
            if self.decay_cap_ms and running_prev and not running:
                capping = t_us + _ms_to_us(self.decay_cap_ms,
                                           self._sample_us)

            # Assume the task was running for all the current PELT sample
            active_us = self._sample_us if running else 0

            # Always update utilization:
            # - when the task is running
            # - when is not running and we are not capping the decay
            if running or not self.decay_cap_ms:
                pelt_value = self._geomSum(pelt_value, active_us)

            # Update decay only up to the capping point
            elif capping and t_us <= capping:
                pelt_value = self._geomSum(pelt_value, active_us)

            # Append PELT sample
            sample = (_us_to_s(t_us), t_us/self._sample_us, running, pelt_value)
            samples.append(sample)

            # Prepare for next sample computation
            running_prev = running
            t_us += self._sample_us

        # Create DataFrame from computed samples
        self._df = DataFrame(
            samples, columns=['Time', 'PELT_Interval', 'Running', 'pelt_value'])
        self._df.set_index('Time', inplace=True)

        # Keep track of the last task we computed the signal for
        self.task = task

        return self._df

    def getStats(self, stats_start_s=0, stats_end_s=None):
        """Compute the stats over a pre-computed PELT signal, considering only
        the specified portion of the signal.

        :param stats_start_s: the start of signal portion to build the stats for
        :type  stats_start_s: float

        :param stats_end_s: the end of signal portion to build the stats for
        :type  stats_end_s: float

        :return: :mod:`PELTStats` instance reporting the metrics of the PELT
                 signal computed for the task specified in the last execution
                 of `getSignal`
        """

        # Validate input parameters
        if self._df is None:
            raise ValueError("no signal computed, run getSignal before")

        # Intervals (in PELT samples) for the statistics to compute
        stats_start_s = max(stats_start_s, self._df.index.min())
        if stats_end_s:
            stats_end_s = min(stats_end_s, self._df.index.max())
        else:
            stats_end_s = self._df.index.max()

        df = self._df[stats_start_s:stats_end_s]

        # Compute stats
        _pelt_avg = self.task.pelt_avg
        _pelt_init = self.init_value
        _half_life = self.half_life_ms
        _tmin = stats_start_s
        _tmax = stats_end_s
        _min = df.pelt_value.min()
        _max = df.pelt_value.max()
        _avg = df.pelt_value.mean()
        _std = df.pelt_value.std()
        _err = _avg - self.task.pelt_avg
        _err_pct = None
        if self.task.pelt_avg:
            _err_pct = 100 * _err / self.task.pelt_avg

        return PELTStats(
            _us_to_s(self.start_us), _us_to_s(self.end_us),
            _pelt_avg, _pelt_init, _half_life,
            _tmin, _tmax, _min, _max,
            _avg, _std, _err, _err_pct)

    @classmethod
    def estimateInitialPeltValue(cls, first_val, first_event_time_s,
                                 start_time_s, half_life_ms):
        """There will typically be a delay between the time when a task starts,
        and the first trace event that logs PELT signals for that task. During
        this time the signal will change from its initial value. This method
        takes the time at which a task starts running and the timestamp and
        value of the first PELT trace event related to that task. It returns an
        estimate for the value the PELT signal had the task started.

        Example:
        ::

            +---------------------+      Task is RUNNING
            |              E      |

          --+--------------+------+--------- Time --->
            ^              ^
          task_start     first PELT event logged for the task

        This method estimates the value of the PELT signal at task_start,
        assuming the task was running in the interval of time between
        task_start and the first PELT event logged of the task.

        :param first_val: value of the very first event realted to the
            task that shows PELT info
        :type first_val: int

        :param first_event_time_s: instant of time in seconds when the first
            PELT event related to the task was generated
        :type first_event_time_s: float

        :param start_time_s: instant of time in seconds when the task starts
            running for the first time
        :type start_time_s: float

        :param half_life_ms: the [ms] interval required by a signal to
                             increased/decreased by half of its range
        :type  half_life_ms: int

        :returns: int - Estimated value of PELT signal when the task starts
        """
        geom_y = pow(0.5, 1/half_life_ms)
        geom_u = float(cls._signal_max) * (1. - geom_y)

        # Compute period of time between when the task started and when the
        # first sched_load_avg_task event was generated
        time_since_start = first_event_time_s - start_time_s

        if time_since_start < 0:
            raise ValueError('First sched_load_avg_* event '
                             'happens before the task starts')
        # Compute number of times the simulated PELT would be updated in this
        # period of time
        updates_since_start = int(time_since_start/(cls._sample_us/1e6))
        pelt_val = first_val
        for i in range(updates_since_start):
            pelt_val = (pelt_val - geom_u) / geom_y

        return pelt_val


##########################################################################
# Utility Functions
##########################################################################

def _s_to_us(time_s, interval_us=1e3, nearest_up=True):
    """Convert [s] into the (not smaller/greater) nearest [us]

    Translate a time expressed in [s] to the nearest time in [us]
    which is an integer multiple of the specified interval_us and it is not
    smaller than the original time_s.

    Example:
    ::
        _s_to_us(1.0)       => 1000000 [us]
        _s_to_us(1.0, 1024) => 1000448 [us]

        _s_to_us(1.1)       => 1100000 [us]
        _s_to_us(1.1, 1024) => 1100800 [us]

    :param time_s: time in seconds
    :type time_s: float

    :param interval_us: the result will be an integer multiple o this value
        (default = 1e3)
    :type time_ms: int

    :param nearest_up: convert to not smaller nearest if True, to not greater
        otherwise (default = True)
    :type nearest_up: bool
    """
    if nearest_up:
        return interval_us * int(math.ceil((1e6 * time_s)/interval_us))
    return interval_us * int(math.floor((1e6 * time_s)/interval_us))


def _ms_to_us(time_ms, interval_us=1e3, nearest_up=True):
    """Convert [ms] into the (not smaller/greater) nearest [us]

    Translate a time expressed in [ms] to the nearest time in [us]
    which is a integer multiple of the specified interval_us and it is not
    smaller than the original time_s.

    Example:
    ::
        _ms_to_us(1.0)       => 1000 [us]
        _ms_to_us(1.0, 1024) => 1024 [us]

        _ms_to_us(1.1)       => 2000 [us]
        _ms_to_us(1.1, 1024) => 2048 [us]

    :param time_ms: time in milliseconds
    :type time_ms: float

    :param interval_us: the result will be an integer multiple o this value
        (default = 1e3)
    :type time_ms: int

    :param nearest_up: convert to not smaller nearest if True, to not greater
        otherwise (default = True)
    :type nearest_up: bool
    """
    if nearest_up:
        return interval_us * int(math.ceil((1e3 * time_ms)/interval_us))
    return interval_us * int(math.floor((1e3 * time_ms)/interval_us))


def _us_to_s(time_us):
    """Convert [us] into (float) [s]
    """
    return time_us/1e6


def _us_to_ms(time_us):
    """Convert [us] into (float) [ms]
    """
    return time_us/1e3


def _ms_to_s(time_ms):
    """Convert [ms] into (float) [s]
    """
    return time_ms/1e3
