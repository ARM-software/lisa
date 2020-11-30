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

import io
import os
import inspect
import abc
import textwrap
import base64
import functools
import docutils.core
import contextlib
import warnings
import itertools
import weakref
import copy
from operator import itemgetter
from collections.abc import Iterable

import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# Avoid ambiguity between function name and usual variable name
from cycler import cycler as make_cycler

import mplcursors

from ipywidgets import widgets
from IPython.display import display

from lisa.utils import Loggable, get_subclasses, get_doc_url, get_short_doc, split_paragraphs, update_wrapper_doc, guess_format, is_running_ipython, nullcontext, measure_time
from lisa.trace import MissingTraceEventError, PandasDataDesc
from lisa.notebook import axis_link_dataframes, axis_cursor_delta, WrappingHBox, make_figure
from lisa.generic import TypedList


class AnalysisHelpers(Loggable, abc.ABC):
    """
    Helper methods class for Analysis modules.

    :Design notes:

    Plotting methods *must* return the :class:`matplotlib.axes.Axes` instance
    used by the plotting method. This lets users further modify them.
    """

    @abc.abstractmethod
    def name():
        """
        Name of the analysis class.
        """
        pass

    @classmethod
    def setup_plot(cls, width=16, height=4, ncols=1, nrows=1, interactive=None, link_dataframes=None, cursor_delta=None, **kwargs):
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

        :param link_dataframes: Link the provided dataframes to the axes using
            :func:`lisa.notebook.axis_link_dataframes`
        :type link_dataframes: list(pandas.DataFrame) or None

        :param cursor_delta: Add two vertical lines set with left and right
            clicks, and show the time delta between them in a widget.
        :type cursor_delta: bool or None

        :param interactive: If ``True``, use the pyplot API of matplotlib,
            which integrates well with notebooks. However, it can lead to
            memory leaks in scripts generating lots of plots, in which case it
            is better to use the non-interactive API. Defaults to ``True`` when
            running under IPython or Jupyter notebook, `False`` otherwise.
        :type interactive: bool

        :Keywords arguments: Extra arguments to pass to
          :obj:`matplotlib.figure.Figure.subplots`

        :returns: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes (or an
          array of, if ``nrows`` > 1))
        """

        figure, axes = make_figure(
            interactive=interactive,
            width=width,
            height=height,
            ncols=ncols,
            nrows=nrows,
        )
        if interactive is None:
            interactive = is_running_ipython()

        use_widgets = interactive

        if link_dataframes:
            if not use_widgets:
                cls.get_logger().error('Dataframes can only be linked to axes in interactive widget plots')
            else:
                for axis in figure.axes:
                    axis_link_dataframes(axis, link_dataframes)

        if cursor_delta or cursor_delta is None and use_widgets:
            if not use_widgets and cursor_delta is not None:
                cls.get_logger().error('Cursor delta can only be used in interactive widget plots')
            else:
                for axis in figure.axes:
                    axis_cursor_delta(axis)

        for axis in figure.axes:
            axis.relim(visible_only=True)
            axis.autoscale_view(True)

        # Needed for multirow plots to not overlap with each other
        figure.set_tight_layout(dict(h_pad=3.5))
        return figure, axes

    @classmethod
    @contextlib.contextmanager
    def set_axis_cycler(cls, axis, *cyclers):
        """
        Context manager to set cyclers on an axis (and the default cycler as
        well), and then restore the default cycler.

        .. note:: The given cyclers are merged with the original cycler. The
            given cyclers will override any key of the original cycler, and the
            number of values will be adjusted to the maximum size between all
            of them. This way of merging allows decoupling the length of all
            keys.
        """
        orig_cycler = plt.rcParams['axes.prop_cycle']

        # Get the maximum value length among all cyclers involved
        values_len = max(
            len(values)
            for values in itertools.chain(
                orig_cycler.by_key().values(),
                itertools.chain.from_iterable(
                    cycler.by_key().values()
                    for cycler in cyclers
                ),
            )
        )

        # We can only add together cyclers with the same number of values for
        # each key, so cycle through the provided values, up to the right
        # length
        def pad_values(values):
            values = itertools.cycle(values)
            values = itertools.islice(values, 0, values_len)
            return list(values)

        def pad_cycler(cycler):
            keys = cycler.by_key()
            return {
                key: pad_values(values)
                for key, values in keys.items()
            }

        cycler = {}
        for user_cycler in cyclers:
            cycler.update(pad_cycler(user_cycler))

        # Merge the cyclers and original cycler together, so we still get the
        # original values of the keys not overridden by the given cycler
        parameters = {
            **pad_cycler(orig_cycler),
            **cycler,
        }
        cycler = make_cycler(**parameters)

        def set_cycler(cycler):
            plt.rcParams['axes.prop_cycle'] = cycler
            axis.set_prop_cycle(cycler)

        set_cycler(cycler)
        try:
            yield
        finally:
            # Since there is no way to get the cycler from an Axis,
            # we cannot restore the original one, so use the
            # default one instead
            set_cycler(orig_cycler)

    @classmethod
    @contextlib.contextmanager
    def set_axis_rc_params(cls, axis, rc_params):
        """
        Context manager to set ``matplotlib.rcParams`` while plotting, and then
        restore the default parameters.
        """
        orig = matplotlib.rcParams.copy()
        matplotlib.rcParams.update(rc_params)

        try:
            yield
        finally:
            # matplotlib complains about some deprecated settings being set, so
            # silence it since we are just restoring the original state
            with warnings.catch_warnings():
                matplotlib.rcParams.update(orig)

    @classmethod
    def cycle_colors(cls, axis, nr_cycles=1):
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

        axis.set_prop_cycle(make_cycler(color=colors[nr_cycles:] + colors[:nr_cycles]))

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

    def get_default_plot_path(self, img_format, plot_name, default_dir='.'):
        """
        Return the default path to use to save plots for the analysis.

        :param img_format: Format of the image to save.
        :type img_format: str

        :param plot_name: Middle-name of the plot
        :type plot_name: str

        :param default_dir: Default folder to store plots into.
        :type default_dir: str
        """
        analysis = self.name
        filepath = os.path.join(
            default_dir,
            "{}.{}.{}".format(analysis, plot_name, img_format))

        return filepath

    def save_plot(self, figure, filepath=None, img_format=None):
        """
        Save a :class:`matplotlib.figure.Figure` as an image file.
        """

        img_format = img_format or guess_format(filepath) or 'png'
        caller = inspect.stack()[1][3]

        filepath = filepath or self.get_default_plot_path(
            img_format=img_format,
            plot_name=caller,
        )

        # The suptitle is not taken into account by tight layout by default:
        # https://stackoverflow.com/questions/48917631/matplotlib-how-to-return-figure-suptitle
        suptitle = figure._suptitle
        figure.savefig(
            filepath,
            bbox_extra_artists=[suptitle] if suptitle else None,
            format=img_format,
            bbox_inches='tight'
        )

    def do_plot(self, plotter, axis=None, **kwargs):
        """
        Simple helper for consistent behavior across methods.
        """

        local_fig = axis is None
        if local_fig:
            fig, axis = self.setup_plot(**kwargs)

        plotter(axis, local_fig)
        return axis

    @staticmethod
    def _get_base64_image(axis, fmt='png'):
        if isinstance(axis, numpy.ndarray):
            axis = axis[0]
        figure = axis.get_figure()
        buff = io.BytesIO()
        figure.savefig(buff, format=fmt, bbox_inches='tight')
        buff.seek(0)
        b64_image = base64.b64encode(buff.read())
        return b64_image.decode('utf-8')

    @classmethod
    def get_plot_methods(cls, instance=None):
        obj = instance if instance is not None else cls

        def predicate(f):
            if not callable(f):
                return False

            # "unwrap" bound methods and other similar things
            with contextlib.suppress(AttributeError):
                f = f.__func__

            return (
                f.__name__.startswith('plot_')
                and f is not cls.plot_method.__func__
            )

        return [
            f
            for name, f in inspect.getmembers(obj, predicate=predicate)
        ]

    _FIG_DATA = weakref.WeakKeyDictionary()
    """
    Data that are related to a matplotlib figure and that must not be duplicated by each call.gqq
    """

    @classmethod
    def _get_fig_data(cls, fig, key):
        return cls._FIG_DATA.setdefault(fig, {})[key]

    def _set_fig_data(cls, fig, key, val):
        cls._FIG_DATA.setdefault(fig, {})[key] = val

    def _make_fig_toolbar(self, fig):
        toolbar = WrappingHBox()
        widget_list = []

        label = 'Open in trace viewer'
        open_button = widgets.Button(
            description=label,
            tooltip=label,
            disabled=False,
        )
        open_button.on_click(lambda event: self.trace.show())
        widget_list.append(open_button)

        toolbar.children += tuple(widget_list)
        return toolbar

    @classmethod
    def plot_method(cls, return_axis=False):
        """
        Plot function decorator.

        :param return_axis: If ``True``, the decorated method is expected to
            return a :class:`matplotlib.axes.Axes` instance, by using
            :meth:`do_plot` for example. Otherwise, it is expected to
            take an ``axis`` and ``local_fig`` parameters like the ``plotter``
            given to :meth:`do_plot` and just update the ``axis``.
        :type return_axis: bool

        It allows for automatic plot setup and HTML and reStructuredText output.
        """

        def decorator(f):
            @update_wrapper_doc(
                f,
                added_by=':meth:`{}.{}.plot_method`'.format(
                    AnalysisHelpers.__module__,
                    AnalysisHelpers.__qualname__,
                ),
                description=textwrap.dedent("""
                :returns: An :class:`matplotlib.axes.Axes` containing the plot,
                    or rich formats depending on ``output`` value.

                :param axis: instance of :class:`matplotlib.axes.Axes` to plot into.
                    If `None`, a new figure and axis are created and returned.
                :type axis: matplotlib.axes.Axes
                    or numpy.ndarray(matplotlib.axes.Axes)
                    or None

                :param colors: List of color names to use for the plots.
                :type colors: list(str) or None

                :param linestyles: List of linestyle to use for the plots.
                :type linestyles: list(str) or None

                :param markers: List of marker to use for the plots.
                :type markers: list(str) or None

                :param rc_params: Matplotlib rc params dictionary overlaid on
                    existing settings.
                :type rc_params: dict(str, object) or None

                :param filepath: Path of the file to save the figure in. If
                    `None`, no file is saved.
                :type filepath: str or None

                :param always_save: When ``True``, the plot is always saved
                    even if no ``filepath`` has explicitly been set. In that
                    case, a default path will be used.
                :type always_save: bool

                :param img_format: The image format to generate. Defaults to
                    using filepath to guess the type, or "png" if no filepath is
                    given. `html` and `rst` are supported in addition to
                    matplotlib image formats.
                :type img_format: str

                :param output: Can be ``None`` to return a
                    :class:`matplotlib.axes.Axes`, ``html`` to return an HTML
                    document, or ``rst`` for a reStructuredText output.
                :type output: str or None

                :Variable keyword arguments: Forwarded to
                    :meth:`~lisa.analysis.base.AnalysisHelpers.setup_plot`
                """),
                remove_params=['local_fig'],
                include_kwargs=True,
            )
            def wrapper(self, *args, filepath=None, axis=None, output=None, img_format=None, always_save=False, colors: TypedList[str]=None, linestyles: TypedList[str]=None, markers: TypedList[str]=None, rc_params=None, **kwargs):

                def is_f_param(param):
                    """
                    Return True if the parameter is for `f`, False if it is
                    for setup_plot()
                    """
                    try:
                        desc = inspect.signature(f).parameters[param]
                    except KeyError:
                        return False
                    else:
                        # Passing kwargs=42 to a function taking **kwargs
                        # should not return True here, as we only consider
                        # explicitly listed arguments
                        return desc.kind not in (
                            inspect.Parameter.VAR_KEYWORD,
                            inspect.Parameter.VAR_POSITIONAL,
                        )

                # Factor the *args inside the **kwargs by binding them to the
                # user-facing signature, which is the one of the wrapper.
                kwargs.update(
                    inspect.signature(wrapper).bind_partial(self, *args).arguments
                )

                f_kwargs = {
                    param: val
                    for param, val in kwargs.items()
                    if is_f_param(param)
                }

                img_format = img_format or guess_format(filepath) or 'png'
                local_fig = axis is None

                # When we create the figure ourselves, always save the plot to
                # the default location
                if local_fig and filepath is None and always_save:
                    filepath = self.get_default_plot_path(
                        img_format=img_format,
                        plot_name=f.__name__,
                    )

                cyclers = dict(
                    color=colors,
                    linestyle=linestyles,
                    marker=markers,
                )
                cyclers = {
                    name: value
                    for name, value in cyclers.items()
                    if value
                }
                if cyclers:
                    cyclers = [
                        make_cycler(**{name: value})
                        for name, value in cyclers.items()
                    ]
                    set_cycler = lambda axis: cls.set_axis_cycler(axis, *cyclers)
                else:
                    set_cycler = lambda axis: nullcontext()

                if rc_params:
                    set_rc_params = lambda axis: cls.set_axis_rc_params(axis, rc_params)
                else:
                    set_rc_params = lambda axis: nullcontext()

                # Allow returning an axis directly, or just update a given axis
                if return_axis:
                    # In that case, the function takes all the kwargs
                    with set_cycler(axis), set_rc_params(axis):
                        axis = f(**kwargs, axis=axis)
                else:
                    if local_fig:
                        setup_plot_kwargs = {
                            param: val
                            for param, val in kwargs.items()
                            if param not in f_kwargs
                        }
                        fig, axis = self.setup_plot(**setup_plot_kwargs)

                    f_kwargs.update(
                        axis=axis,
                        local_fig=local_fig,
                    )
                    with set_cycler(axis), set_rc_params(axis):
                        f(**f_kwargs)

                if isinstance(axis, numpy.ndarray):
                    fig = axis[0].get_figure()
                else:
                    fig = axis.get_figure()

                def resolve_formatter(fmt):
                    format_map = {
                        'rst': cls._get_rst_content,
                        'html': cls._get_html,
                    }
                    try:
                        return format_map[fmt]
                    except KeyError:
                        raise ValueError('Unsupported format: {}'.format(fmt))

                if output is None:
                    out = axis

                    # Show the LISA figure toolbar
                    if is_running_ipython():
                        # Make sure we only add one button per figure
                        try:
                            toolbar = self._get_fig_data(fig, 'toolbar')
                        except KeyError:
                            toolbar = self._make_fig_toolbar(fig)
                            self._set_fig_data(fig, 'toolbar', toolbar)
                            display(toolbar)

                        mplcursors.cursor(fig)
                else:
                    out = resolve_formatter(output)(f, [], f_kwargs, axis)

                if filepath:
                    if img_format in ('html', 'rst'):
                        content = resolve_formatter(img_format)(f, [], f_kwargs, axis)

                        with open(filepath, 'wt', encoding='utf-8') as fd:
                            fd.write(content)
                    else:
                        fig.savefig(filepath, format=img_format, bbox_inches='tight')

                return out
            return wrapper
        return decorator

    @staticmethod
    def _get_rst_header(f):
        name = f.__name__
        prefix = 'plot_'
        if name.startswith(prefix):
            name = name[len(prefix):]
        name = name.replace('_', ' ').capitalize()

        try:
            url = get_doc_url(f)
            doc_link = '`[doc] <{url}>`_'.format(url=url)
        except Exception:
            doc_link = ''

        return textwrap.dedent("""
                {name}
                {name_underline}

                {docstring} {link}
            """).format(
            name=name,
            link=doc_link,
            name_underline='=' * len(name),
            docstring=get_short_doc(f),
        )

    @classmethod
    def _get_rst_content(cls, f, args, kwargs, axis):
        kwargs = inspect.signature(f).bind_partial(*args, **kwargs)
        kwargs.apply_defaults()
        kwargs = kwargs.arguments

        fmt = 'png'
        b64_image = cls._get_base64_image(axis, fmt=fmt)

        hidden_params = {
            'self',
            'axis',
            'local_fig',
            'filepath',
            'axis',
            'output',
            'img_format',
            'always_save',
            'kwargs',
            'colors',
            'linestyles',
            'markers',
            'rc_params',
        }
        args_list = ', '.join(
            '{}={}'.format(k, v)
            for k, v in sorted(kwargs.items(), key=itemgetter(0))
            if v is not None and k not in hidden_params
        )

        return textwrap.dedent("""
            .. figure:: data:image/{fmt};base64,{data}
                :alt: {name}
                :align: center
                :width: 100%

                {arguments}
        """).format(
            fmt=fmt,
            data=b64_image,
            arguments=args_list,
            name=f.__qualname__,
        )

    @classmethod
    def _get_rst(cls, f, args, kwargs, axis):
        return cls._get_rst_header(f) + '\n' + cls._get_rst_content(f, args, kwargs, axis)

    @staticmethod
    def _docutils_render(writer, rst, doctitle_xform=False):
        overrides = {
            'input_encoding': 'utf-8',
            # enable/disable promotion of lone top-level section title
            # to document title
            'doctitle_xform': doctitle_xform,
            'initial_header_level': 1,
            # This level will silent unknown roles and directives
            # error.  It is necessary since we are rendering docstring
            # written for Sphinx using docutils, which only understands
            # plain reStructuredText
            'report_level': 4,
        }
        parts = docutils.core.publish_parts(
            source=rst, source_path=None,
            destination_path=None,
            writer_name=writer,
            settings_overrides=overrides,
        )

        return parts

    @classmethod
    def _get_html(cls, *args, **kwargs):
        rst = cls._get_rst(*args, **kwargs)
        parts = cls._docutils_render(writer='html', rst=rst, doctitle_xform=True)
        return parts['whole']


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

    @classmethod
    def cache(cls, f):
        """
        Decorator to enable caching of the output of dataframe getter function
        in the trace cache.

        This will write the dataframe to the swap as well, so processing can be
        skipped completely when possible.
        """
        sig = inspect.signature(f)
        ignored_kwargs = {
            # self
            list(sig.parameters.keys())[0],
        }

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            # Express the arguments as kwargs-only
            params = sig.bind(self, *args, **kwargs)
            params.apply_defaults()
            kwargs = dict(params.arguments)

            trace = self.trace
            spec = dict(
                module=f.__module__,
                func=f.__qualname__,
                # Include the trace window in the spec since that influences
                # what the analysis was seeing
                trace_state=trace.trace_state,
                # Make a deepcopy as it is critical that the PandasDataDesc is
                # not modified under the hood once inserted in the cache
                kwargs=copy.deepcopy({
                    k: v
                    for k, v in kwargs.items()
                    if k not in ignored_kwargs
                }),
            )
            pd_desc = PandasDataDesc(spec=spec)

            cache = trace._cache
            write_swap = trace._write_swap
            try:
                df = cache.fetch(pd_desc)
            except KeyError:
                with measure_time() as measure:
                    df = f(**kwargs)
                compute_cost = measure.exclusive_delta
                cache.insert(pd_desc, df, compute_cost=compute_cost, write_swap=write_swap)

            return df

        return wrapper

    @classmethod
    def get_all_events(cls):
        """
        Returns the set of all events used by any of the methods.
        """

        def predicate(f):
            return callable(f) and hasattr(f, 'used_events')

        return set(itertools.chain.from_iterable(
            attr.used_events.get_all_events()
            for name, attr in inspect.getmembers(cls, predicate=predicate)
        ))

    def get_default_plot_path(self, **kwargs):
        return super().get_default_plot_path(
            default_dir=self.trace.plots_dir,
            **kwargs,
        )

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

    @classmethod
    def call_on_trace(cls, meth, trace, meth_kwargs):
        """
        Call a method of a subclass on a given trace.

        :param meth: Function (method) defined on a subclass.
        :type meth: collections.abc.Callable

        :param trace: Trace object to use
        :type trace: lisa.trace.Trace

        :param meth_kwargs: Dictionary of keyword arguments to pass to ``meth``
        :type meth_kwargs: dict

        It will create an instance of the right analysis, bind the function to
        it and call the resulting bound method with ``meth_kwargs`` extra
        keyword arguments.
        """
        for subcls in cls.get_analysis_classes().values():
            for name, f in inspect.getmembers(subcls):
                if f is meth:
                    break
            else:
                continue
            break
        else:
            raise ValueError('{} is not a method of any subclasses of {}'.format(
                meth.__qualname__,
                cls.__qualname__,
            ))

        # Create an analysis instance and bind the method to it
        analysis = subcls(trace=trace)
        meth = meth.__get__(analysis, type(analysis))

        return meth(**meth_kwargs)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
