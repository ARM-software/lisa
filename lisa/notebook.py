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
"""
Various utilities for interactive notebooks.
"""

import functools
import collections

import mplcursors
import matplotlib
from matplotlib.backend_bases import MouseButton
from ipywidgets import widgets, Output, HBox, Layout, interact
from IPython.display import display

class WrappingHBox(widgets.HBox):
    """
    HBox that will overflow on multiple lines if the content is too large to
    fit on one line.
    """
    def __init__(self, *args, **kwargs):
        layout = Layout(
            # Overflow items to the next line rather than hiding them
            flex_flow='row wrap',
            # Evenly spread on one line of items
            justify_content='space-around',
        )
        super().__init__(*args, layout=layout, **kwargs)


# Make a subclass so we can integrate better with mplcursors
class _DataframeLinkMarker(matplotlib.lines.Line2D):
    pass


# Tell mplcursors that we are never selecting the marker line, so that it
# will still show the coordinates of the data that were plotted, rather
# than useless coordinates of the marker
@mplcursors.compute_pick.register(_DataframeLinkMarker)
def _(artist, event):
    return None


def axis_link_dataframes(axis, df_list, before=1, after=5, cursor_color='red', follow_cursor=False):
    """
    Link some dataframes to an axis displayed in the interactive matplotlib widget.


    :param axis: Axis to link to.
    :type axis: matplotlib.axes.Axes

    :param df_list: List of pandas dataframe to link.
    :type df_list: list(pandas.DataFrame)

    :param before: Number of dataframe rows to display before the selected
        location.
    :type before: int

    :param after: Number of dataframe rows to display after the selected
        location.
    :type after: int

    :param cursor_color: Color of the vertical line added at the clicked
        location.
    :type cursor_color: str

    :param follow_cursor: If ``True``, the cursor will be followed without the
        need to click.
    :type follow_cursor: bool

    When the user clicks on the graph, a vertical marker will appear and the
    dataframe slice will update to show the relevant row.

    .. note:: This requires the matplotlib widget enabled using ``%matplotlib
        widget`` magic.
    """
    df_list = [df for df in df_list if not df.empty]
    output_list = [widgets.Output() for df in df_list]
    layout = Layout(
        # Overflow items to the next line rather than hiding them
        flex_flow='row wrap',
        # Evenly spread on one line of item when there is more than one item,
        # align left otherwise
        justify_content='space-around' if len(df_list) > 1 else 'flex-start',
    )
    hbox = widgets.HBox(output_list, layout=layout)

    cursor_vline = axis.axvline(color=cursor_color)
    assert type(cursor_vline) is matplotlib.lines.Line2D
    cursor_vline.__class__ = _DataframeLinkMarker

    def show_loc(loc):
        cursor_vline.set_xdata(loc)

        for df, output in zip(df_list, output_list):
            if loc < df.index[0]:
                iloc = 0
            elif loc > df.index[-1]:
                iloc = -1
            else:
                iloc = df.index.get_loc(loc, method='ffill')
            index_loc = df.index[iloc]

            begin = max(iloc - before, 0)
            end = min(iloc + after, len(df))
            sliced_df = df.iloc[begin:end]

            def highlight_row(row):
                if row.name == index_loc:
                    return ['background: lightblue'] * len(row)
                else:
                    return [''] * len(row)

            styler = sliced_df.style.apply(highlight_row, axis=1)
            styler = styler.set_properties(**{
                'text-align': 'left',
                # perserve multiple consecutive spaces
                'white-space': 'pre',
                # Make sure all chars have the same width to preserve column
                # alignments in preformatted strings
                'font-family': 'monospace',
            })

            # wait=True avoids flicker by waiting for new content to be ready
            # to display before clearing the previous one
            output.clear_output(wait=True)
            with output:
                display(styler)

    init_loc = min(df.index[0] for df in df_list)
    show_loc(init_loc)

    def handler(event):
        loc = event.xdata
        return show_loc(loc)

    event = 'motion_notify_event' if follow_cursor else 'button_press_event'
    axis.get_figure().canvas.mpl_connect(event, handler)
    display(hbox)


def axis_cursor_delta(axis, colors=['blue', 'green'], buttons=[MouseButton.LEFT, MouseButton.RIGHT]):
    """
    Display the time delta between two vertical lines drawn on clicks.

    :param axis: Axis to link to.
    :type axis: matplotlib.axes.Axes

    :param colors: List of colors to use for vertical lines.
    :type colors: list(str)

    :param buttons: Mouse buttons to use for each vertical line.
    :type buttons: list(matplotlib.backend_bases.MouseButton)

    .. note:: This requires the matplotlib widget enabled using
        ``%matplotlib widget`` magic.
    """
    delta_widget = widgets.Text(
        value='0',
        placeholder='0',
        description='Cursors delta',
        disabled=False,
    )

    def make_vline(axis, color):
        vline = axis.axvline(color=color)
        assert type(vline) is matplotlib.lines.Line2D
        vline.__class__ = _DataframeLinkMarker
        return vline

    vlines = [
        make_vline(axis, color)
        for color in colors
    ]

    assert len(vlines) == 2
    vlines_map = dict(zip(buttons, vlines))
    vlines_loc = collections.defaultdict(int)

    def handler(event):
        loc = event.xdata
        button = event.button

        vline = vlines_map[button]
        vlines_loc[button] = loc
        vline.set_xdata(loc)
        locs = [
            vlines_loc[button]
            for button in buttons
        ]
        delta = locs[1] - locs[0]
        delta_widget.value = str(delta)

    axis.get_figure().canvas.mpl_connect('button_press_event', handler)
    display(delta_widget)


def interact_tasks(trace, tasks=None, kind=None):
    """
    Decorator to make a block of code parametrized on a task that can be
    selected from a dropdown.

    :param trace: Trace object in use
    :type trace: lisa.trace.Trace

    :param tasks: List of tasks that are available. See ``kind`` for
        alternative way of specifying tasks.
    :type tasks: list(int or str or lisa.trace.TaskID) or None

    :param kind: Alternatively to ``tasks``, a kind can be provided and the
        tasks will be selected from the trace for you. It can be:

            * ``rtapp`` to select all rt-app tasks
            * ``all`` to select all tasks.

    :type kind: str or None

    **Example**::

        trace = Trace('trace.dat')

        # Allow selecting any rtapp task
        @interact_tasks(trace, kind='rtapp')
        def do_plot(task):
            trace.analysis.load_tracking.plot_task_signals(task)
    """
    if tasks is not None:
        tasks = [
            trace.get_task_id(task, update=False)
            for task in tasks
        ]
    else:
        kind = kind or 'all'
        if kind == 'all':
            tasks = trace.task_ids
        elif kind == 'rtapp':
            tasks = trace.analysis.rta.rtapp_tasks
        else:
            raise ValueError('Unknown task kind: {}'.format(kind))

    # Map of friendly names to actual objects
    task_map = {
        str(task): task
        for task in tasks
    }

    def decorator(f):
        @functools.wraps(f)
        @interact
        def wrapper(task=sorted(task_map.keys())):
            task = task_map[task]
            return f(task)
        return wrapper

    return decorator

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
