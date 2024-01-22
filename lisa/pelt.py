# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

import math
import functools

import pandas as pd

from lisa.datautils import series_envelope_mean

PELT_WINDOW = 1024 * 1024 * 1e-9
"""
PELT window in seconds.
"""

PELT_HALF_LIFE = 32
"""
PELT half-life in number of windows.
"""

PELT_SCALE = 1024


def simulate_pelt(activations, init=0, index=None, clock=None, capacity=None, windowless=False, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE):
    """
    Simulate a PELT signal out of a series of activations.

    :param activations: Series of a task's activations:
        ``1 == running`` and ``0 == sleeping``.
    :type activations: pandas.Series

    :param init: Initial value of the signal
    :type init: float

    :param index: Optional index at which the PELT values should be computed.
        If ``None``, a value will be computed when the task starts sleeping and
        when it wakes up. Note that there is no emulation of scheduler tick
        updating the signal while it's running.
    :type index: pandas.Index

    :param clock: Series of clock values to be used instead of the timestamp index.
    :type clock: pandas.Series

    :param capacity: Capacity of the CPU at all points. This is used to fixup
        the clock on enqueue and dequeue, since the clock is typically provided
        by a PELT event and not the enqueue or dequeue events. If no clock at
        all is passed, the CPU capacity will be used to create one from scratch
        based on the ``activations`` index values.
    :type capacity: pandas.Series or None

    :param window: PELT window in seconds.
    :type window: float

    :param windowless: If ``True``, a windowless simulator is used. This avoids
        the artifacts of the windowing in PELT.
    :type windowless: bool

    :param half_life: PELT half-life in number of windows.
    :type half_life: int

    :param scale: Scale of the signal, i.e. maximum value it can take.
    :type scale: float

    .. note:: PELT windowing is not time-invariant, i.e. it depends on the
        absolute value of the timestamp. This means that the timestamp of the
        activations matters, and it is recommended to use the ``clock``
        parameter to provide the actual clock used by PELT.

        Also note that the kernel uses integer arithmetic with a different way
        of computing the signal. This means that the simulation cannot
        perfectly match the kernel's signal.
    """
    if index is None:
        index = activations.index
    else:
        index = index.union(activations.index)
        activations = activations.reindex(index, method='ffill')

    df = pd.DataFrame({'activations': activations})
    if capacity is not None:
        df['rel_capacity'] = capacity.reindex(df.index, method='ffill') / scale
        df['rel_capacity'] = df['rel_capacity'].bfill()

    if clock is None:
        # If we have the CPU capacity at hand, we can make a fake PELT clock
        # based on it by just scaling the time deltas
        if capacity is not None:
            delta_ = df.index.to_series().diff().shift(-1)
            # Scale each delta with the relative capacity of the CPU
            clock = (delta_ * df['rel_capacity']).cumsum()

            # On enqueue, reset the PELT clock to the ftrace clock to catch up
            # the "lost idle time"
            clock_offset = df.index - clock
            # Select the points at which the task is enqueued
            clock_offset = clock_offset.where(
                cond=((df['activations'] == 0) & (df['activations'].shift(-1) == 1)),
            )
            # Find the clock offset for each enqueue, by removing the offset of
            # any previous enqueue
            clock_offset = clock_offset.dropna().diff()
            #TODO: understand why we have to inflate the offset by this factor
            clock_offset *= 1 + df['rel_capacity']
            # Reindex the offset on the clock and accumulate the offset
            clock_offset = clock_offset.reindex(clock.index, fill_value=0).cumsum()
            clock = clock + clock_offset
        else:
            clock = df.index

    df['clock'] = clock

    # Fix the PELT clock at enqueue and dequeue points, since these indices are
    # typically coming from sched_switch or sched_wakeup, and therefore do not
    # log the PELT clock

    # PELT time scaling over the past 2 samples at all points
    index_diff = df.index.to_series().diff()
    fix_clock_at = df['clock'].isna()

    # If we have access to CPU capacity directly, no need to extrapolate the
    # scale by looking at previous/next pairs of PELT events, we can get it
    # directly
    if capacity is not None:
        time_scale = df['rel_capacity']
        dequeue_timescale = time_scale
        enqueue_timescale = time_scale
    else:
        time_scale = df['clock'].diff() / index_diff
        dequeue_timescale = time_scale.shift()
        enqueue_timescale = time_scale.shift(-2)

    dequeue = (df['activations'] == 0) & (df['activations'].shift() == 1)
    enqueue = (df['activations'] == 1) & (df['activations'].shift() == 0)

    # Fix dequeue:
    # Previous clock value plus the elapsed ftrace time, rescaled using the
    # previous scale
    df.loc[(fix_clock_at & dequeue), 'clock'] = df['clock'].shift() + index_diff * dequeue_timescale
    # Fix enqueue:
    # Next clock value, minus the ftrace time from now to the next clock value,
    # rescaled using the scale of the next PELT event
    df.loc[(fix_clock_at & enqueue), 'clock'] = df['clock'].shift(-1) - index_diff.shift(-1) * enqueue_timescale

    # Ensure the clock is monotonic.
    # The enqueue/dequeue fixup might sometimes generate inconsistent values.
    prev_clock = df['clock'].shift()
    df['clock'] = df['clock'].where(
        cond=(df['clock'] >= prev_clock.fillna(df['clock'])),
        other=prev_clock,
    )

    # If there are still some missing clock values, it means that some
    # activations were too short to have 2 PELT samples, which is required to
    # estimate the time scaling, so we cannot do anything for them
    df.dropna(subset=['clock'], inplace=True)

    df['delta'] = df['clock'].diff()

    # Compute the number of crossed PELT windows between each sample Since PELT
    # windowing is not time invariant (windows are at "millisecond"
    # boundaries), we need non-normalized timestamps
    window_series = df['clock'] // window
    df['crossed_windows'] = window_series.diff()

    # We want to have the entity state in the window between the previous
    # sample and now.
    df['activations'] = df['activations'].shift()

    # First row of "delta" is NaN, and activations reindex may have produced
    # some NaN at the beginning of the dataframe as well
    df.dropna(inplace=True)

    def make_windowed_pelt_sim(init, scale, window, half_life):
        decay = (1 / 2)**(1 / half_life)
        # Alpha as defined in https://en.wikipedia.org/wiki/Moving_average
        alpha = 1 - decay

        # Accumulator of running time within a PELT window
        acc = 0
        # Output signal
        signal = init / scale
        output = signal

        def pelt(row):
            nonlocal acc, signal, output

            # 1=running 0=sleeping
            running = row['activations']
            clock = row['clock']
            delta = row['delta']
            windows = row['crossed_windows'].astype('int')

            # We crossed one or more windows boundaries
            if windows:
                # Handle last piece of the window in which this activation started
                first_window_fraction = window - ((clock - delta) % window)
                first_window_fraction /= window

                acc += running * first_window_fraction
                signal = alpha * acc + (1 - alpha) * signal

                # Handle the windows we fully crossed
                for _ in range(windows - 1):
                    signal = alpha * running + (1 - alpha) * signal

                # Handle the current incomplete window
                last_window_fraction = (clock % window) / window

                # Extrapolate the signal as it would look with the same
                # `running` state at the end of the current window
                extrapolated = running * alpha + (1 - alpha) * signal
                # Take an value between signal and extrapolated based on the
                # current completion of the window. This implements the same
                # idea as introduced by kernel commit:
                #  sched/cfs: Make util/load_avg more stable 625ed2bf049d5a352c1bcca962d6e133454eaaff
                output = signal + last_window_fraction * (extrapolated - signal)

                signal += alpha * running * last_window_fraction
                acc = 0
            # If we are still in the same window, just accumulate the running
            # time
            else:
                acc += running * delta / window

            return output * scale

        return pelt

    def make_windowless_pelt_sim(init, scale, window, half_life):
        tau = _pelt_tau(half_life, window)
        signal = init

        def pelt_after(init, t, running):
            # Compute the the response of the 1st order filter at time "t",
            # with the given initial condition
            # http://fourier.eng.hmc.edu/e59/lectures/e59/node33.html
            exp_ = math.exp(-t / tau)
            non_zero = running * scale * (1 - exp_)
            zero = init * exp_
            return non_zero + zero

        def pelt(row):
            nonlocal signal
            # 1=running 0=sleeping
            running = row['activations']
            delta = row['delta']

            signal = pelt_after(signal, delta, running)
            return signal

        return pelt

    if windowless:
        make_sim = make_windowless_pelt_sim
    else:
        make_sim = make_windowed_pelt_sim

    sim = make_sim(
        init=init,
        window=window,
        half_life=half_life,
        scale=scale,
    )
    df['pelt'] = df.apply(sim, axis=1)
    pelt = df['pelt']
    if pelt.index is not index:
        pelt = pelt.reindex(index, method='ffill')
    return pelt


def _pelt_tau(half_life, window):
    """
    Compute the time constant of an equivalent continuous-time system as
    defined by: ``tau = period * (alpha / (1-alpha))``
    https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
    """

    # Alpha as defined in https://en.wikipedia.org/wiki/Moving_average
    decay = (1 / 2)**(1 / half_life)
    alpha = 1 - decay
    tau = window * ((1 - alpha) / alpha)
    return tau


# Use LRU cache as computing the swing is quite costly
@functools.lru_cache(maxsize=256, typed=True)
def pelt_swing(period, duty_cycle, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE, kind='peak2peak'):
    """
    Compute an approximation of the PELT signal swing for a given periodic task.

    :param period: Period of the task in seconds.
    :type period: float

    :param duty_cycle: Duty cycle of the task.
    :type duty_cycle: float

    :param window: PELT window in seconds.
    :type window: float

    :param half_life: PELT half life, in number of windows.
    :type half_life: int

    :param scale: PELT scale.
    :type scale: float

    :param kind: One of:

        * ``peak2peak``: the peak-to-peak swing of PELT.
        * ``above``: the amplitude of the swing above the average value.
        * ``below``: the amplitude of the swing below the average value.

    :type kind: str

    .. note:: The PELT signal is approximated as a first order filter. This
        does not take into account the averaging inside a window, but the
        window is small enough in practice for that effect to be negligible.
    """
    if duty_cycle in (0, 1):
        return 0

    final = duty_cycle * scale
    stable_t = pelt_settling_time(
        margin=1,
        init=0,
        final=final,
        window=window,
        half_life=half_life,
        scale=scale
    )
    # Ensure we have at least a full period
    stable_t = max(stable_t, period)

    # Align to have an integral number of periods
    stable_t += stable_t % period
    end = stable_t + period
    nr_period = int(end / period)

    run = duty_cycle * period
    sleep =  period - run

    # We only compute one sample per period, so it's as efficient as it can get
    activations = pd.Series([0, 1] * nr_period)
    activations.index = pd.Series([run, sleep] * nr_period).cumsum()

    simulated = simulate_pelt(
        activations,
        # Setting the initial value close to the average of the signal improves
        # convergence time a great deal
        init=duty_cycle * scale,
        window=window,
        half_life=half_life,
        scale=scale,
        # We don't want windowing artifacts to pollute the result.
        windowless=True,
    )
    min_ = simulated.iloc[-1]
    max_ = simulated.iloc[-2]

    assert min_ <= duty_cycle * scale <= max_

    if kind == 'peak2peak':
        return abs(max_ - min_)
    elif kind == 'above':
        return abs(max_ / scale - duty_cycle) * scale
    elif kind == 'below':
        return abs(min_ / scale - duty_cycle) * scale
    else:
        raise ValueError(f'Unknown kind "{kind}"')


def pelt_step_response(t, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE):
    """
    Compute an approximation of the PELT value at time ``t`` when subject to a
    step input (i.e running tasks, PELT starting at 0).

    :param t: Evaluate PELT signal at time = ``t``.
    :type t: float

    :param window: PELT window in seconds.
    :type window: float

    :param half_life: PELT half life, in number of windows.
    :type half_life: int

    :param scale: PELT scale.
    :type scale: float

    """
    tau = _pelt_tau(half_life, window)
    return scale * (1 - math.exp(-t / tau))


def pelt_settling_time(margin=1, init=0, final=PELT_SCALE, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE):
# pylint: disable=unused-argument
    """
    Compute an approximation of the PELT settling time.

    :param margin: How close to the final value we want to get, in PELT units.
    :type margin_pct: float

    :param init: Initial PELT value.
    :type init: float

    :param final: Final PELT value.
    :type final: float

    :param window: PELT window in seconds.
    :type window: float

    :param half_life: PELT half life, in number of windows.
    :type half_life: int

    :param scale: PELT scale.
    :type scale: float

    .. note:: The PELT signal is approximated as a first order filter. This
        does not take into account the averaging inside a window, but the
        window is small enough in practice for that effect to be negligible.
    """
    tau = _pelt_tau(half_life, window)

    # Response of a first order low pass filter:
    # y(t) = u(t) * (1 - exp(-t/tau))
    # We want to find `t` such as the output y(t) is as close as we want from
    # the input u(t):
    # A * u(t) = u(t) * (1 - exp(-t/tau))
    #
    # From which follows:
    # A = (1 - exp(-t/tau))
    # t = -tau * log(1-A)

    if final > init:
        final = final - abs(margin)
        def compute(final):
            A = final / scale
            return - tau * math.log(1 - A)
    elif final < init:
        # y(t) = u(t) * exp(-t/tau)
        # A * u(t) = u(t) * exp(-t/tau)
        # A = exp(-t/tau)
        # t = -tau * log(A)
        def compute(final):
            A = final / scale
            return - tau * math.log(A)
        final = final + abs(margin)
    else:
        return 0

    # Time it takes for 0=>final minus 0=>init
    return abs(compute(final) - compute(init))


def kernel_util_mean(util, plat_info):
# pylint: disable=unused-argument
    """
    Compute the mean of a utilization signal as output by the kernel.

    :param util: Series of utilization over time.
    :type util: pandas.Series

    :param plat_info: Platform info of the kernel used to generate the
        utilization signal.
    :type plat_info: lisa.platforms.platinfo.PlatformInfo

    .. warning:: It is currently only fully accurate for a task with a 512
        utilisation mean.
    """
    return series_envelope_mean(util)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
