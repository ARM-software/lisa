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

import os
import inspect
import mimetypes

import matplotlib.pyplot as plt
from cycler import cycler

from lisa.utils import Loggable, get_subclasses

# Colorblind-friendly cycle, see https://gist.github.com/thriveth/8560036
COLOR_CYCLES = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = cycler(color=COLOR_CYCLES)

class AnalysisHelpers(Loggable):
    """
    Helper methods class for Analysis modules.

    :Design notes:

    Plotting methods *must* return the :class:`matplotlib.axes.Axes` instance
    used by the plotting method. This lets users further modify them.
    """

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

        :Keywords arguments: Extra arguments to pass to
          :obj:`matplotlib.pyplot.subplots`

        :returns: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes (or an
          array of, if ``nrows`` > 1))
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

    def _save_plot(self, figure, default_dir, filepath=None, img_format=None, wrapper_level=2):
        if filepath is None:
            img_format = img_format or 'png'
            module = self.__module__
            caller = inspect.stack()[1 + wrapper_level][3]
            filepath = os.path.join(
                default_dir,
                "{}.{}.{}".format(module, caller, img_format))
        else:
            mime_type = mimetypes.guess_type(filepath, strict=False)[0]
            guessed_format = mime_type.split('/')[1].split('.', 1)[-1].split('+')[0]
            img_format = img_format or guessed_format

        figure.savefig(filepath, format=img_format, bbox_inches='tight')

    def save_plot(self, figure, filepath=None, img_format=None):
        """
        Save the plot stored in the ``figure``

        :param figure: The plot figure
        :type figure: matplotlib.figure.Figure

        :param filepath: The path of the file into which the plot will be saved.
          If ``None``, a path based on the trace directory and the calling method
          will be used. The filepath is also used to deduct the image format.
        :type filepath: str

        :param img_format: The image format to generate. Defaults to using
            filepath to guess the type, or "png" if no filepath is given.
        :type img_format: str
        """
        default_dir = '.'
        return self._save_plot(figure, default_dir, filepath, img_format)

    def do_plot(self, plotter, filepath=None, axis=None, **kwargs):
        """
        Simple helper for consistent behavior across methods.

        :returns: An :class:`matplotlib.axes.Axes` containing the plot.

        :param filepath: Path of the file to save the figure in. If `None`, no
            file is saved.
        :type filepath: str or None

        :param axis: instance of :class:`matplotlib.axes.Axes` to plot into.
            If `None`, a new figure and axis are created and returned.
        :type axis: matplotlib.axes.Axes
            or numpy.ndarray(matplotlib.axes.Axes)
            or None

        :param kwargs: keyword arguments forwarded to :meth:`setup_plot`
        :type kwargs: dict
        """

        local_fig = axis is None
        if local_fig:
            fig, axis = self.setup_plot(**kwargs)

        plotter(axis, local_fig)

        if local_fig:
            self.save_plot(fig, filepath)
        return axis

class TraceAnalysisBase(AnalysisHelpers):
    """
    Base class for Analysis modules.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`

    :Design notes:

    Method depending on certain trace events *must* be decorated with
    :meth:`lisa.trace.requires_events`
    """

    def __init__(self, trace):
        self.trace = trace

    def save_plot(self, figure, filepath=None, img_format=None):
        """
        See :meth:`AnalysisHelpers.save_plot`
        """
        default_dir = self.trace.plots_dir
        return self._save_plot(figure, default_dir, filepath, img_format)

    @classmethod
    def get_analysis_classes(cls):
        return {
            subcls.name: subcls
            for subcls in get_subclasses(cls)
            # Classes without a "name" attribute directly defined in their
            # scope will not get registered. That allows having unnamed
            # intermediate base classes that are not meant to be exposed.
            if 'name' in subcls.__dict__
        }

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
