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

import pandas as pd
import numpy as np

from lisa.datautils import df_combine_duplicates, df_add_delta, series_mean

PELT_WINDOW = 1024 * 1024 * 1e-9
"""
PELT window in seconds.
"""

PELT_HALF_LIFE = 32
"""
PELT half-life in number of windows.
"""

PELT_SCALE = 1024


def simulate_pelt(activations, init=0, index=None, clock=None, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE):
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

    :param window: PELT window in seconds.
    :type window: float

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
    if index is not None:
        activations = activations.reindex(index, method='ffill')

    df = pd.DataFrame({'activations': activations})
    df['clock'] = clock if clock is not None else df.index
    df['delta'] = df['clock'].diff()

    # Compute the number of crossed PELT windows between each sample Since PELT
    # windowing is not time invariant (windows are at "millisecond"
    # boundaries), we need non-normalized timestamps
    window_series = df['clock'] // window
    df['crossed_windows'] = window_series.diff()

    # First row of "delta" is NaN, and activations reindex may have produced
    # some NaN at the beginning of the dataframe as well
    df.dropna(inplace=True)

    def make_pelt_sim(init, scale, window, half_life):
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

    sim = make_pelt_sim(
        init=init,
        window=window,
        half_life=half_life,
        scale=scale,
    )
    df['pelt'] = df.apply(sim, axis=1)
    return df['pelt']


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


def pelt_swing(period, duty_cycle, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE):
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

    .. note:: The PELT signal is approximated as a first order filter. This
        does not take into account the averaging inside a window, but the
        window is small enough in practice for that effect to be negligible.
    """
    tau = _pelt_tau(half_life, window)
    runtime = duty_cycle * period
    # Compute the step response of a first order after time t=runtime
    swing = scale * (1 - math.exp(-runtime / tau))
    return swing


def pelt_settling_time(margin=1, init=0, final=PELT_SCALE, window=PELT_WINDOW, half_life=PELT_HALF_LIFE, scale=PELT_SCALE):
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
    # A is how close from u(t) we want the output to get after a time `t`
    # From which follows:
    # A = (1 - exp(-t/tau))
    # t = -tau * log(1-A)

    # Since the equation we have is for a step response, i.e. from 0 to a final
    # value
    delta = abs(final - init)
    # Since margin and delta are in the same unit, we don't have to normalize
    # them to `scale` first.
    relative_margin = (margin / delta)
    A = 1 - relative_margin

    settling_time = - tau * math.log(1 - A)
    return settling_time


def kernel_util_mean(util, plat_info):
    """
    Compute the mean of a utilization signal as output by the kernel.

    :param util: Series of utilization over time.
    :type util: pandas.Series

    :param plat_info: Platform info of the kernel used to generate the
        utilization signal.
    :type plat_info: lisa.platforms.platinfo.PlatformInfo

    It handles correctly the missing decay of the util when the task sleeps.
    """
    df = pd.DataFrame(dict(util=util))

    # Find the times where the util is strictly increasing
    df['up'] = (df['util'].diff() > 0)

    # Merge together rows where the util is monotonically increasing/decreasing
    # and compute the time spent in each phase
    df = df_add_delta(df, col='duration', inplace=True)
    func = lambda df: df['duration'].sum()
    df_combine_duplicates(df, func=func, cols=['up'], output_col='duration', inplace=True)

    # If it starts with an deactivation, shift back so that we start on an
    # activation
    if not df['up'].iloc[0]:
        df = df.iloc[1:]

    # Pair activations with the following deactivation
    up_duration = df['duration']
    down_duration = df['duration'].shift(-1)

    duty_cycle = up_duration / (up_duration + down_duration)
    df.loc[df['up'], 'duty_cycle'] = duty_cycle
    df['duty_cycle'].fillna(method='ffill', inplace=True)

    return series_mean(duty_cycle) * PELT_SCALE

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
