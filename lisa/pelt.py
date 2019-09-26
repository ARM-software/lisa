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

import pandas as pd
import numpy as np

def simulate_pelt(activations, init=0, index=None, window=1024*1024*1e-9, half_life=32, scale=1024):
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

    :param window: PELT window in seconds.
    :type window: float

    :param half_life: PELT half-life in number of windows.
    :type half_life: int

    :param scale: Scale of the signal, i.e. maximum value it can take.
    :type scale: float

    .. note:: PELT windowing is not time-invariant, i.e. it depends on the
        absolute value of the timestamp. This means that the timestamp of the
        activations matters, and it is recommended to load the trace without
        time normalization in order to achieve best results. In any case, the
        timestamp of the kernel event may not match exactly what timestamp was
        used to compute the PELT sample, which can lead to a small amount of
        errors.

        Also note that the kernel uses integer arithmetic with a different way
        of computing the signal. This means that the simulation cannot
        perfectly match the kernel's signal.
    """
    if index is not None:
        index = np.concatenate((activations.index, index))
        index.sort()
        activations = activations.reindex(index, method='ffill')
        activations.dropna(inplace=True)

    df = pd.DataFrame({'activations': activations})
    df['time'] = df.index
    df['delta'] = df['time'].diff()
    # First row of "delta" is NaN
    df['delta'].iloc[0] = 0

    # Compute the number of crossed PELT windows between each sample Since PELT
    # windowing is not time invariant (windows are at "millisecond"
    # boundaries), we need non-normalized timestamps
    window_series = df['time'] // window
    df['crossed_windows'] = window_series.diff()
    df['crossed_windows'].iloc[0] = 0

    def make_pelt_sim(init, scale, window, half_life):
        decay = (1/2)**(1/half_life)
        # Alpha as defined in https://en.wikipedia.org/wiki/Moving_average
        alpha = 1 - decay

        # Accumulator of running time within a PELT window
        acc = 0
        # Output signal
        signal = init / scale

        def pelt(row):
            nonlocal acc, signal

            # 1=running 0=sleeping
            running = row['activations']
            time = row['time']
            delta = row['delta']
            windows = row['crossed_windows'].astype('int')

            # We crossed one or more windows boundaries
            if windows:
                # Handle last piece of the window in which this activation started
                first_window_fraction = window - ((time - delta) % window)
                first_window_fraction /= window

                acc += running * first_window_fraction
                signal = alpha * acc + (1-alpha) * signal

                # Handle the windows we fully crossed
                for _ in range(windows - 1):
                    signal = alpha * running + (1-alpha) * signal

                # Handle the current incomplete window
                last_window_fraction = (time % window) / window
                signal += alpha * running * last_window_fraction
                acc = 0
            # If we are still in the same window, just accumulate the running
            # time
            else:
                acc += running * delta / window

            return signal * scale

        return pelt

    sim = make_pelt_sim(
        init=init,
        window=window,
        half_life=half_life,
        scale=scale,
    )
    df['pelt'] = df.apply(sim, axis=1)
    return df['pelt']

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
