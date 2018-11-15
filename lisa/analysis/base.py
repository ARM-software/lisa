# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

import functools
import os
import inspect

import matplotlib.pyplot as plt
from cycler import cycler

from trappy.utils import listify

from lisa.utils import Loggable

# Colorblind-friendly cycle, see https://gist.github.com/thriveth/8560036
COLOR_CYCLES = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = cycler(color=COLOR_CYCLES)

def requires_events(events):
    """
    Decorator for methods that require some given trace events

    :param events: The list of required events
    :type events: list(str)

    The decorate method must inherit from :class:`AnalysisBase`
    """
    def decorator(f):
        @functools.wraps(f)

        def wrapper(self, *args, **kwargs):
            self.check_events(events)
            return f(self, *args, **kwargs)

        # Set an attribute on the wrapper itself, so it can be e.g. added
        # to the method documentation
        wrapper.required_events = events
        return wrapper

    return decorator

class AnalysisBase(Loggable):
    """
    Base class for Analysis modules.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`

    :Design notes:

    Method depending on certain trace events *must* be decorated with
    :meth:`lisa.analysis.base.requires_events`

    Plotting methods *must* return the :class:`matplotlib.axes.Axes` instance
    used by the plotting method. This lets users embed plots into subplots.
    """

    def __init__(self, trace):
        self._trace = trace

    @classmethod
    def setup_plot(cls, width=16, height=4, ncols=1, nrows=1, **kwargs):
        """
        Common helper for setting up a matplotlib plot

        :param width: Width of the plot (inches)
        :type width: int or float

        :param height: Height of each subplot (inches)
        :type height: int or float

        :param ncols: Number of plots on a single row
        :type ncols: int

        :param nrows: Number of plots in a single column
        :type nrows: int

        :Keywords arguments: Extra arguments to pass to :meth:`matplotlib.subplots`

        :returns: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes (or an
          (array of, if ``nrows`` > 1))
        """
        figure, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(width, height * nrows), **kwargs
        )
        # Needed for multirow plots to not overlap with each other
        plt.tight_layout(h_pad=3.5)
        return figure, axes

    @classmethod
    def cycle_colors(cls, axis, nr_cycles):
        """
        Cycle the axis color cycle ``nr_cycles`` forward

        :param axis: The axis to manipulate
        :type axis: matplotlib.axes.Axes

        :param nr_cycles: The number of colors to cycle through.
        :type nr_cycles: int

        .. note::

          This is an absolute cycle, as in, it will always start from the first
          color defined in the color cycle.

        """
        if nr_cycles < 1:
            return

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if nr_cycles > len(colors):
            nr_cycles -= len(colors)

        axis.set_prop_cycle(cycler(color=colors[nr_cycles:] + colors[:nr_cycles]))

    @classmethod
    def get_next_color(cls, axis):
        """
        Get the next color that will be used to draw lines on the axis

        :param axis: The axis
        :type axis: matplotlib.axes.Axes

        .. warning::

          This will consume the color from the cycler, which means it will
          change which color is to be used next.

        """
        # XXX: We're accessing some private data here, so that could break eventually
        # Need to find another way to get the current color from the cycler, or to
        # plot all data from a dataframe in the same color.
        return next(axis._get_lines.prop_cycler)['color']

    def save_plot(self, figure, filepath=None, img_format="png"):
        """
        Save the plot stored in the ``figure``

        :param figure: The plot figure
        :type figure: matplotlib.figure.figure

        :param filepath: The path of the file into which the plot will be saved.
          If ``None``, a path based on the trace directory and the calling method
          will be used.
        :type filepath: str

        :param img_format: The image format to generate
        :type img_format: str
        """
        if filepath is None:
            module = self.__module__
            caller = inspect.stack()[1][3]
            filepath = os.path.join(
                self._trace.plots_dir,
                "{}.{}.{}".format(module, caller, img_format))

        figure.savefig(filepath, format=img_format)

    def check_events(self, required_events):
        """
        Check that certain trace events are available in the trace

        :raises: RuntimeError if some events are not available
        """
        available_events = sorted(set(self._trace.events))
        missing_events = sorted(set(required_events).difference(available_events))

        if missing_events:
            raise RuntimeError(
                "Trace is missing the following required events: {}".format(missing_events))

    def _plot_generic(self, dfr, pivot, filters=None, columns=None,
                     prettify_name=None, width=16, height=4,
                     drawstyle="default", ax=None, title=""):
        """
        Generic trace plotting method

        The values in the column 'pivot' will be used as discriminant

        Let's consider a df with these columns:

        | time | cpu | load_avg | util_avg |
        ====================================
        |  42  |  2  |   1812   |   400    |
        ------------------------------------
        |  43  |  0  |   1337   |   290    |
        ------------------------------------
        |  ..  | ... |    ..    |    ..    |

        To plot the 'util_avg' value of CPU2, the function would be used like so:
        ::
        plot_generic(df, pivot='cpu', filters={'cpu' : [2]}, columns='util_avg')

        CPUs could be compared by using:
        ::
        plot_generic(df, pivot='cpu', filters={'cpu' : [2, 3]}, columns='util_avg')

        :param dfr: Trace dataframe
        :type dfr: `pandas.DataFrame`

        :param pivot: Name of column that will serve as a pivot
        :type pivot: str

        :param filters: Dataframe column filters
        :type filters: dict

        :param columns: Name of columns whose data will be plotted
        :type columns: str or list(str)

        :param prettify_name: user-friendly stringify function for pivot values
        :type prettify_name: callable[str]

        :param width: The width of the plot
        :type width: int

        :param height: The height of the plot
        :type height: int

        :param drawstyle: The drawstyle setting of the plot
        :type drawstyle: str
        """

        if prettify_name is None:
            def prettify_name(name): return '{}={}'.format(pivot, name)

        if pivot not in dfr.columns:
            raise ValueError('Invalid "pivot" parameter value: no {} column'
                             .format(pivot)
            )

        if columns is None:
            # Find available columns
            columns = dfr.columns.tolist()
            columns.remove(pivot)
        else:
            # Filter out unwanted columns
            columns = listify(columns)
            try:
                dfr = dfr[columns + [pivot]]
            except KeyError as err:
                raise ValueError('Invalid "columns" parameter value: {}'
                                 .format(err.message)
                )

        # Apply filters
        if filters is None:
            filters = {}

        for col, vals in filters.items():
            dfr = dfr[dfr[col].isin(vals)]

        setup_plot = False
        if ax is None:
            _, ax = self.setup_plot(width, height)
            setup_plot = True

        matches = dfr[pivot].unique().tolist()

        for match in matches:
            renamed_cols = []
            for col in columns:
                renamed_cols.append('{} {}'.format(prettify_name(match), col))

            plot_dfr = dfr[dfr[pivot] == match][columns]
            plot_dfr.columns = renamed_cols
            plot_dfr.plot(ax=ax, drawstyle=drawstyle)

        if setup_plot:
            ax.set_title(title)

        ax.set_xlim(self._trace.x_min, self._trace.x_max)

        # Extend ylim for better visibility
        cur_lim = ax.get_ylim()
        lim = (cur_lim[0] - 0.1 * (cur_lim[1] - cur_lim[0]),
               cur_lim[1] + 0.1 * (cur_lim[1] - cur_lim[0]))
        ax.set_ylim(lim)

        plt.legend()

        return ax

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
