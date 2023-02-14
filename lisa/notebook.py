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
Various utilities for interactive notebooks, plus some generic plot-related
functions.
"""

import functools
import collections
import warnings
import importlib
import inspect
from uuid import uuid4
from itertools import starmap

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
import holoviews as hv
import bokeh.models
import panel as pn

from cycler import cycler as make_cycler

from ipywidgets import widgets, Layout, interact
from IPython.display import display

from lisa.utils import is_running_ipython, order_as, destroyablecontextmanager, ContextManagerExit

pn.extension('tabulator')

# Enable all backends, finishing with matplotlib so it becomes the default (for
# backward compat). Since this also corresponds to holoviews default, we are
# not loosing anything, as the user will need hv.extension('bokeh') anyway.
_backends = ['plotly', 'bokeh', 'matplotlib']

# If the user selected a backend already, use it as defaults by activating it
# last
_curr_backend = hv.Store.current_backend
if _curr_backend:
    _backends.remove(_curr_backend)
    _backends.append(_curr_backend)

for backend in _backends:
    try:
        importlib.import_module(backend)
    except Exception:
        pass
    else:
        hv.extension(backend)

COLOR_CYCLE = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]
"""
Colorblind-friendly cycle, see https://gist.github.com/thriveth/8560036
"""

plt.rcParams['axes.prop_cycle'] = make_cycler(color=COLOR_CYCLE)


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
class _DataframeLinkMarker(mpl.lines.Line2D):
    pass


# mplcursors is not a dependency anymore as interactive plots are now done with
# bokeh, but keep this around for compatibility in case someone needs
# matplotlib to get a better fixed output and wants a bit of interactivity for
# development as well.
try:
    import mplcursors
except ImportError:
    pass
else:
    # Tell mplcursors that we are never selecting the marker line, so that it
    # will still show the coordinates of the data that were plotted, rather
    # than useless coordinates of the marker
    @mplcursors.compute_pick.register(_DataframeLinkMarker)
    def _(artist, event):
        return None


def _make_vline(axis, *args, **kwargs):
    vline = axis.axvline(*args, **kwargs)
    assert type(vline) is mpl.lines.Line2D # pylint: disable=unidiomatic-typecheck
    vline.__class__ = _DataframeLinkMarker
    vline.set_visible(False)
    return vline


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

    cursor_vline = _make_vline(axis, color=cursor_color)

    def show_loc(loc):
        cursor_vline.set_xdata(loc)
        cursor_vline.set_visible(True)

        for df, output in zip(df_list, output_list):
            if loc < df.index[0]:
                iloc = 0
            elif loc > df.index[-1]:
                iloc = -1
            else:
                iloc = df.index.get_indexer([loc], method='ffill')[0]
            index_loc = df.index[iloc]

            begin = max(iloc - before, 0)
            end = min(iloc + after, len(df))
            sliced_df = df.iloc[begin:end]

            def highlight_row(row):
                if row.name == index_loc: # pylint: disable=cell-var-from-loop
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

    init_loc = min((df.index[0] for df in df_list), default=0)
    show_loc(init_loc)

    def handler(event):
        loc = event.xdata
        return show_loc(loc)

    event = 'motion_notify_event' if follow_cursor else 'button_press_event'
    axis.get_figure().canvas.mpl_connect(event, handler)
    display(hbox)


def axis_cursor_delta(axis, colors=('blue', 'green'), buttons=(MouseButton.LEFT, MouseButton.RIGHT)):
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

    vlines = [
        _make_vline(axis, color=color)
        for color in colors
    ]

    assert len(vlines) == 2
    vlines_map = dict(zip(buttons, vlines))
    vlines_loc = collections.defaultdict(
        lambda: min(axis.get_xbound())
    )

    def handler(event):
        loc = event.xdata
        button = event.button

        vline = vlines_map[button]
        vlines_loc[button] = loc
        vline.set_xdata(loc)
        vline.set_visible(True)
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
            trace.ana.load_tracking.plot_task_signals(task)
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
            tasks = trace.ana.rta.rtapp_tasks
        else:
            raise ValueError(f'Unknown task kind: {kind}')

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


def make_figure(width, height, nrows, ncols, interactive=None, **kwargs):
    """
    Make a :class:`matplotlib.figure.Figure` and its axes.

    :param width: Width of the figure.
    :type width: int

    :param height: Height of the figure.
    :type height: int

    :param interactive: If ``True``, create an interactive figure. Defaults to
        ``True`` when running under IPython, ``False`` otherwise.
    :type interactive: bool or None

    :Variable keyword arguments: Forwarded to :class:`matplotlib.figure.Figure`

    :returns: A tuple of:
        * :class:`matplotlib.figure.Figure`
        * :class:`matplotlib.axes.Axes` as a scalar, an iterable (1D) or iterable of iterable matrix (2D)
    """
    if interactive is None:
        interactive = is_running_ipython()

    if not interactive and tuple(map(int, mpl.__version__.split('.'))) <= (3, 0, 3):
        warnings.warn('This version of matplotlib does not allow saving figures from axis created using Figure(), forcing interactive=True')
        interactive = True

    width *= ncols
    height *= nrows

    if interactive:
        figure, axes = plt.subplots(
            figsize=(width, height),
            nrows=nrows,
            ncols=ncols,
            **kwargs,
        )
    else:
        figure = Figure(figsize=(width, height))
        axes = figure.subplots(ncols=ncols, nrows=nrows, **kwargs)

    return (figure, axes)


def plot_signal(series, name=None, interpolation=None, add_markers=True, vdim=None):
    """
    Plot a signal using ``holoviews`` library.

    :param series: Series of values to plot.
    :type series: pandas.Series

    :param name: Name of the signal. Defaults to the series name.
    :type name: str or None

    :param interpolation: Interpolate type for the signal. Defaults to
        ``steps-post`` which is the correct value for signals encoded as a
        series of updates.
    :type interpolation: str or None

    :param add_markers: Add markers to the plot.
    :type add_markers: bool

    :param vdim: Value axis dimension.
    :type vdim: holoviews.Dimension
    """
    if isinstance(series, pd.DataFrame):
        try:
            col, = series.columns
        except ValueError:
            raise ValueError('Can only pass Series or DataFrame with one column')
        else:
            series = series[col]

    label = name or series.name
    interpolation = interpolation or 'steps-post'
    kdims = [
        # Ensure shared_axes works well across plots.
        # We don't set the unit as this will prevent shared_axes to work if
        # the other plots do not set the unit, which is what usually
        # happens, since only name/label is taken from pandas index names.
        hv.Dimension('Time', unit='s'),
    ]
    vdims = [
        vdim
    ] if vdim is not None else None
    fig = hv.Curve(
        series,
        label=label,
        kdims=kdims,
        vdims=vdims,
    ).opts(
        interpolation=interpolation,
        title=label,
    )
    if add_markers:
        # The "marker" group for Scatter is used to provide marker-specific
        # styling in generic code..
        # TODO: use mute_legend=True once this bug is fixed:
        # https://github.com/holoviz/holoviews/issues/3936
        fig *= hv.Scatter(
            series,
            label=label,
            group='marker',
            kdims=kdims,
            vdims=vdims,
        )
    return fig


# TODO: revisit when this discussion is solved:
# https://github.com/holoviz/holoviews/issues/4988
def _hv_neutral():
    """
    Neutral element of holoviews operations such that
    ``x <op> holoviews_neutral() == x``.

    .. note:: Holoviews currently does not have a perfectly neutral element.
    """
    return hv.Curve([])


def _hv_backend_twinx(backend, display, y_range):
    def hook(plot, element):
        p = plot.state

        if backend == 'bokeh':
            glyph = p.renderers[-1]
            vals = glyph.data_source.data['y']

            if y_range is None:
                _y_range = (vals.min(), vals.max())
            else:
                _y_range = y_range

            name = uuid4().hex
            p.extra_y_ranges.update({
                name: bokeh.models.Range1d(start=_y_range[0], end=_y_range[1])
            })
            glyph.y_range_name = name

            if display:
                p.add_layout(
                    bokeh.models.LinearAxis(y_range_name=name),
                    'right'
                )
        elif backend == 'matplotlib':
            ax = plot.handles['axis']
            twin = ax.twinx()
            plot.handles['axis'] = twin
            if not display:
                twin.get_yaxis().set_ticks([])
            if y_range is not None:
                twin.set_ylim(y_range)
        else:
            raise ValueError(f'Unsupported backend={backend}')

    return hook

def _hv_twinx(fig, display=True, y_range=None):
    """
    Similar to matplotlib's twinx feature where the element's Y axis is
    separated from the default one and drawn on the right of the plot.

    :param display: If ``True``, the ticks will be displayed on the right of
        the plot. Otherwise, it will be hidden.
    :type display: bool

    .. note:: This uses a custom hook for each backend, so it will be disabled
        if the user also set their own hook.
    """
    kwargs = dict(
        display=display,
        y_range=y_range,
    )
    return fig.options(
        backend='bokeh',
        hooks=[_hv_backend_twinx('bokeh', **kwargs)],
    ).options(
        backend='matplotlib',
        hooks=[_hv_backend_twinx('matplotlib', **kwargs)],
    )

def _hv_multi_line_title_hook(plot, element):
    p = plot.state
    # Add in reverse since titles will pile upwards
    lines = list(reversed(plot.title.splitlines()))
    if len(lines) > 1:
        for line in lines:
            title = bokeh.models.Title(
                text=line,
                standoff=1,
            )
            p.add_layout(title, 'above')

        # Add an empty line at the top to provide visual separation
        # with other plots
        p.add_layout(bokeh.models.Title(text=' '), 'above')
        del p.title

    # Adjust the width of the plot so that the title is not truncated
    max_len = max(map(len, lines))
    # Empirical, should probably inspect the title font size instead
    px_per_char = 12
    p.width = max(p.width, max_len * px_per_char)


def _hv_multi_line_title(fig):
    """
    Holoviews hook to allow multiline titles.

    Also enlarges the plot if its too small for its title.
    """
    return fig.options(hooks=[_hv_multi_line_title_hook])


@destroyablecontextmanager
def _hv_set_backend(backend):
    """
    Context manager to work around this issue:
    https://github.com/holoviz/holoviews/issues/4962
    """
    old_backend = hv.Store.current_backend
    try:
        # This is safe to do as long as the backend has been
        # loaded with hv.extension() beforehand, which happens
        # at import time
        hv.Store.set_current_backend(backend)
        yield
    except ContextManagerExit:
        if old_backend:
            hv.Store.set_current_backend(old_backend)


def _hv_link_dataframes(fig, dfs):
    """
    Link the provided dataframes to the holoviews figure.

    :returns: A panel displaying the dataframes and the figure.
    """
    def make_table(i, df):
        event_header = [
            col for col in df.columns
            if (
                col.startswith('__') or
                col == 'event'
            )
        ]
        df = df[order_as(df.columns, event_header)]

        if df.index.name in df.columns:
            df.index = df.index.copy(deep=False)
            df.index.name = ''

        df_widget = pn.widgets.Tabulator(
            df,
            name=df.attrs.get('name', f'dataframe #{i}'),
            formatters={
                'bool': {'type': 'tickCross'}
            },
            # Disable edition of the dataframe
            disabled=True,
            # sortable=False,
            # Ensure some columns are always displayed
            frozen_columns=event_header,
            # For pn.widgets.DataFrame.
            # frozen_columns=len(event_header) + 1,
            height=400,
            # autosize_mode='fit_viewport',
            row_height=25,
            pagination=None,

            # Only relevant for pn.widgets.Tabulator
            theme='simple',
            selectable='toggle',
        )
        return df_widget

    def mark_table_selection(tables):
        def plot(*args):
            xs = [
                table.value.index[x]
                for xs, table in zip(args, tables)
                for x in xs
            ]
            return hv.Overlay(
                [
                    hv.VLine(x).opts(
                        backend='bokeh',
                        line_dash='dashed',
                    )
                    for x in xs
                ]
            )

        tables = list(tables)
        streams = [
            table.param.selection
            for table in tables
        ]
        bound = pn.bind(plot, *streams)
        dmap = hv.DynamicMap(bound).opts(framewise=True)

        return dmap

    def scroll_table(tables):
        def record_taps(x, y):
            try:
                for table in tables:
                    if x is not None:
                        df = table.value
                        i = df.index.get_indexer([x], method='ffill')[0]
                        # This will automatically scroll in the table.
                        # It requires a Python int, a numpy object is not good
                        # enough.
                        table.selection = [int(i)]
            # It is vital to return something, otherwise the plot will
            # disappear, which is much worse than the selection not working
            finally:
                return hv.Points([])

        tap = hv.streams.SingleTap(transient=True)
        dmap = hv.DynamicMap(record_taps, streams=[tap])
        return dmap

    tables = list(starmap(make_table, enumerate(dfs)))
    markers = mark_table_selection(tables)
    scroll = scroll_table(tables)

    # Workaround issue:
    # https://github.com/holoviz/holoviews/issues/5003
    if isinstance(fig, hv.Layout):
        ncols = fig._max_cols
    else:
        ncols = None

    fig = fig * (scroll * markers)

    if ncols is not None:
        fig = fig.cols(ncols)

    if len(tables) > 1:
        tables_widget = pn.Tabs(
            *(
                (table.name, table)
                for table in tables
            ),
            align='start',
        )
    else:
        tables_widget = tables[0]
        tables_widget.align = 'start'

    return pn.Column(
        fig,
        tables_widget,
        sizing_mode='stretch_both',
        align='center',
    )


class _HoloviewsPanelWrapper:
    """
    Dummy base class used to identify classes created by
    :func:`_hv_wrap_fig_cls`.
    """

@functools.lru_cache(maxsize=None)
def _hv_wrap_fig_cls(cls):
    """
    Wrap a holoviews element class so that it is displayed inside a panel but
    still exhibits the holoviews API.

    .. note:: Due to https://github.com/holoviz/holoviews/issues/3577, ``x <op>
        y`` will not work if ``x`` is a holoviews object, but the opposit will
        work.
    """

    def wrap_fig(self, x):
        if x.__class__.__module__.startswith('holoviews'):
            return _hv_fig_to_pane(
                fig=x,
                make_pane=self._make_pane,
            )
        else:
            return x

    def make_wrapper(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            x = f(self._fig, *args, **kwargs)
            return wrap_fig(self, x)

        return wrapper

    def make_op(name):
        def op(self, other):
            f = getattr(self._fig, name)

            # Unwrap the holoviews figure to avoid exceptions
            if isinstance(other, _HoloviewsPanelWrapper):
                other = other._fig

            x = f(other)
            return wrap_fig(self, x)
        return op

    class NewCls(_HoloviewsPanelWrapper):
        def __init__(self, fig, make_pane):
            self._fig = fig
            self._make_pane = make_pane

        def _repr_mimebundle_(self, *args, **kwargs):
            pane = self._make_pane(self._fig)
            return pane._repr_mimebundle_(*args, **kwargs)

        def opts(self, *args, **kwargs):
            return wrap_fig(
                self,
                self._fig.opts(*args, **kwargs),
            )

    for attr, x in inspect.getmembers(cls):
        if (not attr.startswith('_')) and inspect.isfunction(x):
            setattr(NewCls, attr, make_wrapper(x))

    for name in (
        '__add__',
        '__radd__',

        '__sub__',
        '__rsub__',

        '__mul__',
        '__rmul__',

        '__matmul__',
        '__rmatmul__',

        '__truediv__',
        '__rtruediv__',

        '__floordiv__',
        '__rfloordiv__',

        '__mod__',
        '__rmod__',

        '__divmod__',
        '__rdivmod__',

        '__pow__',
        '__rpow__',

        '__and__',
        '__rand__',

        '__xor__',
        '__rxor__',

        '__or__',
        '__ror__',

        '__rshift__',
        '__rrshift__',

        '__lshift__',
        '__rlshift__',
    ):
        if hasattr(cls, name):
            setattr(NewCls, name, make_op(name))

    return NewCls


def _hv_fig_to_pane(fig, make_pane):
    """
    Stop-gap measure until there is a proper solution for:
    https://discourse.holoviz.org/t/replace-holoviews-notebook-rendering-with-a-panel/2519/12
    """
    cls = _hv_wrap_fig_cls(fig.__class__)
    return cls(fig=fig, make_pane=make_pane)



# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
