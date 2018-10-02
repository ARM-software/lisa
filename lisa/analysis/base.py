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

import logging
from collections import namedtuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl

from trappy.utils import listify

""" Helper module for Analysis classes """

ResidencyTime = namedtuple('ResidencyTime', ['total', 'active'])
ResidencyData = namedtuple('ResidencyData', ['label', 'residency'])

class AnalysisBase(object):
    """
    Base class for Analysis modules.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    def __init__(self, trace):
        self._log = logging.getLogger('Analysis')
        self._trace = trace

        plat_info = self._trace.plat_info

        # By default assume SMP system
        self._big_cap = 1024
        self._little_cap = 1024
        self._big_cpus = list(range(trace.cpus_count))
        self._little_cpus = []

        if self._trace.has_big_little:
            self._little_cap = plat_info['nrg-model']['little']['cpu']['cap_max']

        if ('clusters' in plat_info and
            'big' in plat_info['clusters'] and
            'little' in plat_info['clusters']):
            self._big_cpus = plat_info['clusters']['big']
            self._little_cpus = plat_info['clusters']['little']

    @classmethod
    def get_subclasses(cls, cls_set=None):
        """Get all indirect sublcasses of AnalysisBase."""
        if cls_set is None:
            cls_set = set()

        for subcls in cls.__subclasses__():
            if subcls not in cls_set:
                cls_set.add(subcls)
                cls_set.update(subcls.get_subclasses(cls_set))

        return cls_set

    def _plot_setup(self, width=16, height=4, ncols=1, nrows=1):
        figure, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(width, height * nrows)
        )
        # Needed for multirow plots to not overlap with each other
        plt.tight_layout(h_pad=3.5)
        return figure, axes

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
            _, ax = self._plot_setup(width, height)
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
