#    Copyright 2015-2016 ARM Limited
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

"""
**Signals**

    - Definition

        A signal is a string representation of a TRAPpy event and the
        column in the same event. The signal can be of two types:

            - *Pivoted Signal*

                A pivoted signal has a pivot specified in its event class.
                This means that the signal in the event is a concatenation of different
                signals which belong to different **pivot** nodes. The analysis for pivoted
                signals must be done by decomposing them into pivoted signals for each node.

                For example, an even that represents the load of the CPU can be pivoted on
                :code:`"cpu"` which should be a column in the event's `DataFrame`

            - *Non-Pivoted Signal*

                A non pivoted signal has an event that has no pivot value associated with it.
                This probably means that signal has one component and can be analysed without
                decomposing it into smaller signals.

    - Representation

        The following are valid representations of a signal

        - :code:`"event_name:event_column"`
        - :code:`"trappy.event.class:event_column"`

"""

from trappy.stats.grammar import Parser
from trappy.stats import StatConf
from bart.common.Utils import area_under_curve, interval_sum

# pylint: disable=invalid-name
# pylint: disable=anomalous-backslash-in-string

class SignalCompare(object):

    """
    :param data: TRAPpy FTrace Object
    :type data: :mod:`trappy.ftrace.FTrace`

    :param sig_a: The first signal
    :type sig_a: str

    :param sig_b: The first signal
    :type sig_b: str

    :param config: A dictionary of variables, classes
        and functions that can be used in the statements
    :type config: dict

    :param method: The method to be used for reindexing data
        This can be one of the standard :mod:`pandas.DataFrame`
        methods (eg. pad, bfill, nearest). The default is pad
        or use the last valid observation.
    :type method: str

    :param limit: The number of indices a value will be propagated
        when reindexing. The default is None
    :type limit: int

    :param fill: Whether to fill the NaNs in the data.
        The default value is True.
    :type fill: bool

    .. note::

        Both the signals must have the same pivots. For example:

            - Signal A has a pivot as :code:`"cpu"` which means that
              the trappy event (:mod:`trappy.base.Base`) has a pivot
              parameter which is equal to :code:`"cpu"`. Then the signal B
              should also have :code:`"cpu"` as it's pivot.

            - Signal A and B can both have undefined or None
              as their pivots
    """

    def __init__(self, data, sig_a, sig_b, **kwargs):

        self._parser = Parser(
            data,
            config=kwargs.pop(
                "config",
                None),
            **kwargs)
        self._a = sig_a
        self._b = sig_b
        self._pivot_vals, self._pivot = self._get_signal_pivots()

        # Concatenate the indices by doing any operation (say add)
        self._a_data = self._parser.solve(sig_a)
        self._b_data = self._parser.solve(sig_b)

    def _get_signal_pivots(self):
        """Internal function to check pivot conditions and
        return an intersection of pivot on the signals"""

        sig_a_info = self._parser.inspect(self._a)
        sig_b_info = self._parser.inspect(self._b)

        if sig_a_info["pivot"] != sig_b_info["pivot"]:
            raise RuntimeError("The pivot column for both signals" +
                               "should be same (%s,%s)"
                               % (sig_a_info["pivot"], sig_b_info["pivot"]))

        if sig_a_info["pivot"]:
            pivot_vals = set(
                sig_a_info["pivot_values"]).intersection(sig_b_info["pivot_values"])
            pivoted = sig_a_info["pivot"]
        else:
            pivot_vals = [StatConf.GRAMMAR_DEFAULT_PIVOT]
            pivoted = False

        return pivot_vals, pivoted

    def conditional_compare(self, condition, **kwargs):
        """Conditionally compare two signals

        The conditional comparison of signals has two components:

        - **Value Coefficient** :math:`\\alpha_{v}` which measures the difference in values of
          of the two signals when the condition is true:

          .. math::

                \\alpha_{v} = \\frac{area\_under\_curve(S_A\ |\ C(t)\ is\ true)}
                {area\_under\_curve(S_B\ |\ C(t)\ is\ true)} \\\\

                \\alpha_{v} = \\frac{\int S_A(\{t\ |\ C(t)\})dt}{\int S_B(\{t\ |\ C(t)\})dt}

        - **Time Coefficient** :math:`\\alpha_{t}` which measures the time during which the
          condition holds true.

          .. math::

                \\alpha_{t} = \\frac{T_{valid}}{T_{total}}

        :param condition: A condition that returns a truth value and obeys the grammar syntax
            ::

                "event_x:sig_a > event_x:sig_b"

        :type condition: str

        :param method: The method for area calculation. This can
            be any of the integration methods supported in `numpy`
            or `rect`
        :type param: str

        :param step: The step behaviour for area and time
            summation calculation
        :type step: str

        Consider the two signals A and B as follows:

            .. code::

                A = [0, 0, 0, 3, 3, 0, 0, 0]
                B = [0, 0, 2, 2, 2, 2, 1, 1]


            .. code::


                                                     A = xxxx
                3                 *xxxx*xxxx+        B = ----
                                  |         |
                2            *----*----*----+
                             |    |         |
                1            |    |         *----*----+
                             |    |         |
                0  *x-x-*x-x-+xxxx+         +xxxx*xxxx+
                   0    1    2    3    4    5    6    7

        The condition:

        .. math::

            A > B

        is valid between T=3 and T=5. Therefore,

        .. math::

            \\alpha_v=1.5 \\\\
            \\alpha_t=\\frac{2}{7}

        :returns: There are two cases:

            - **Pivoted Signals**
              ::

                    {
                        "pivot_name" : {
                                "pval_1" : (v1,t1),
                                "pval_2" : (v2, t2)
                        }
                    }
            - **Non Pivoted Signals**

              The tuple of :math:`(\\alpha_v, \\alpha_t)`
        """

        if self._pivot:
            result = {self._pivot: {}}

        mask = self._parser.solve(condition)
        step = kwargs.get("step", "post")

        for pivot_val in self._pivot_vals:

            a_piv = self._a_data[pivot_val]
            b_piv = self._b_data[pivot_val]

            area = area_under_curve(a_piv[mask[pivot_val]], **kwargs)
            try:
                area /= area_under_curve(b_piv[mask[pivot_val]], **kwargs)
            except ZeroDivisionError:
                area = float("nan")

            duration = min(a_piv.last_valid_index(), b_piv.last_valid_index())
            duration -= max(a_piv.first_valid_index(),
                            b_piv.first_valid_index())
            duration = interval_sum(mask[pivot_val], step=step) / duration

            if self._pivot:
                result[self._pivot][pivot_val] = area, duration
            else:
                result = area, duration

        return result

    def get_overshoot(self, **kwargs):
        """Special case for :func:`conditional_compare`
        where the condition is:
        ::

            "sig_a > sig_b"

        :param method: The method for area calculation. This can
            be any of the integration methods supported in `numpy`
            or `rect`
        :type param: str

        :param step: The step behaviour for calculation of area
            and time summation
        :type step: str

        .. seealso::

            :func:`conditional_compare`
        """

        condition = " ".join([self._a, ">", self._b])
        return self.conditional_compare(condition, **kwargs)

    def get_undershoot(self, **kwargs):
        """Special case for :func:`conditional_compare`
        where the condition is:
        ::

            "sig_a < sig_b"

        :param method: The method for area calculation. This can
            be any of the integration methods supported in `numpy`
            or `rect`
        :type param: str

        :param step: The step behaviour for calculation of area
            and time summation
        :type step: str

        .. seealso::

            :func:`conditional_compare`
        """

        condition = " ".join([self._a, "<", self._b])
        return self.conditional_compare(condition, **kwargs)
