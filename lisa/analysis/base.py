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

"""
Base classes to define a new trace analysis.
"""

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
import copy
from operator import itemgetter, attrgetter
import typing

import numpy
# Avoid ambiguity between function name and usual variable name
import holoviews as hv
import bokeh
import bokeh.layouts
import bokeh.models.widgets
import panel as pn
import panel.widgets
import polars as pl
import pandas as pd


from lisa.utils import Loggable, deprecate, get_doc_url, get_short_doc, get_subclasses, guess_format, is_running_ipython, measure_time, memoized, update_wrapper_doc, _import_all_submodules, optional_kwargs, get_parent_namespace
from lisa.trace import _CacheDataDesc
from lisa.notebook import _hv_fig_to_pane, _hv_link_dataframes, _hv_has_options, axis_cursor_delta, axis_link_dataframes, make_figure
from lisa.datautils import _df_to, _pandas_cleanup_df

# Ensure hv.extension() is called
import lisa.notebook

# Make sure we associate each plot method with a single wrapped object, so that
# the resulting wrapper can be used as a key in dictionaries.
@functools.lru_cache(maxsize=None, typed=True)
def _wrap_plot_method(cls, f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    # Wrap the method so that we record the actual class they were
    # looked up on, rather than the base class they happen to be
    # defined in.
    wrapper.__qualname__ = f'{cls.__qualname__}.{f.__name__}'
    wrapper.__module__ = cls.__module__
    return wrapper



class AnalysisHelpers(Loggable, abc.ABC):
    """
    Helper methods class for Analysis modules.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """
        Name of the analysis class.
        """

    @classmethod
    @deprecate('Made irrelevant by the use of holoviews', deprecated_in='2.0', removed_in='4.0')
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
            **kwargs,
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
    @deprecate('Made irrelevant by the use of holoviews', deprecated_in='2.0', removed_in='4.0')
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
        import matplotlib.pyplot as plt
        from cycler import cycler as make_cycler

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
            if axis is not None:
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
    @deprecate('Made irrelevant by the use of holoviews', deprecated_in='2.0', removed_in='4.0')
    def set_axis_rc_params(cls, axis, rc_params):
        """
        Context manager to set ``matplotlib.rcParams`` while plotting, and then
        restore the default parameters.
        """
        import matplotlib
        orig = matplotlib.rcParams.copy()
        matplotlib.rcParams.update(rc_params)

        try:
            yield
        finally:
            # matplotlib complains about some deprecated settings being set, so
            # silence it since we are just restoring the original state
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                matplotlib.rcParams.update(orig)

    @classmethod
    @deprecate('Made irrelevant by the use of holoviews', deprecated_in='2.0', removed_in='4.0')
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
        import matplotlib.pyplot as plt
        from cycler import cycler as make_cycler

        if nr_cycles < 1:
            return

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if nr_cycles > len(colors):
            nr_cycles -= len(colors)

        axis.set_prop_cycle(make_cycler(color=colors[nr_cycles:] + colors[:nr_cycles]))

    @classmethod
    @deprecate('Made irrelevant by the use of holoviews', deprecated_in='2.0', removed_in='4.0')
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
            f"{analysis}.{plot_name}.{img_format}")

        return filepath

    def _fig_as_plot_method(self, fig, **kwargs):
        # Create a throw-away plot method so we don't duplicate the logic
        # in plot_method
        def f(self):
            return fig
        f.__name__ = ''
        f.__qualname__ = ''
        # Decorate after changing the name, otherwise the name of the
        # wrapper will be changed but not the one used for titles
        return AnalysisHelpers.plot_method(f)(self, **kwargs)

    def save_plot(self, figure, filepath=None, img_format=None, backend=None):
        """
        Save a holoviews element or :class:`matplotlib.figure.Figure` as an image file.

        :param figure: Figure to save to a file.
        :type figure: matplotlib.figure.Figure or holoviews.core.Element

        :param filepath: Path to the file to save the plot. If ``None``, a
            default path will be used.
        :type filepath: str or None

        :param img_format: Format of the image. If ``None``, it is guessed from
            the ``filepath``.
        :type img_format: str or None

        :param backend: Holoviews backend to use. If left to ``None``, the
            current backend enabled with ``hv.extension()`` will be used.
        :type backend: str or None
        """
        try:
            from matplotlib.figure import Figure
        except ImportError:
            class Figure:
                pass

        img_format = img_format or guess_format(filepath) or 'png'

        filepath = filepath or self.get_default_plot_path(
            img_format=img_format,
            # Use the caller's name as plot name
            plot_name=inspect.stack()[1].function,
        )

        if isinstance(figure, Figure):
            # The suptitle is not taken into account by tight layout by default:
            # https://stackoverflow.com/questions/48917631/matplotlib-how-to-return-figure-suptitle
            suptitle = figure._suptitle
            figure.savefig(
                filepath,
                bbox_extra_artists=[suptitle] if suptitle else None,
                format=img_format,
                bbox_inches='tight'
            )
        else:
            self._fig_as_plot_method(
                figure,
                filepath=filepath,
                backend=backend,
            )

    @deprecate('Made irrelevant by the use of holoviews', deprecated_in='2.0', removed_in='4.0')
    def do_plot(self, plotter, axis=None, **kwargs):
        """
        Simple helper for consistent behavior across methods.
        """

        local_fig = False
        if local_fig:
            fig, axis = self.setup_plot(**kwargs)

        plotter(axis, local_fig)
        return axis

    @staticmethod
    def _get_base64_image(axis, fmt='png'):
        if isinstance(axis, (numpy.ndarray, list)):
            axis = axis[0]
        figure = axis.get_figure()
        buff = io.BytesIO()
        figure.savefig(buff, format=fmt, bbox_inches='tight')
        buff.seek(0)
        b64_image = base64.b64encode(buff.read())
        return b64_image.decode('utf-8')

    @classmethod
    def _get_doc_methods(cls, prefix, instance=None, ignored=None):
        ignored = set(ignored or [])
        obj = instance if instance is not None else cls

        def predicate(f):
            if not callable(f):
                return False

            # "unwrap" bound methods and other similar things
            with contextlib.suppress(AttributeError):
                f = f.__func__

            return (
                f.__name__.startswith(prefix)
                and f not in ignored
            )

        return [
            _wrap_plot_method(cls, f)
            for name, f in inspect.getmembers(obj, predicate=predicate)
            if f not in ignored
        ]

    @classmethod
    def get_plot_methods(cls, *args, **kwargs):
        return cls._get_doc_methods(
            *args,
            prefix='plot_',
            **kwargs,
            ignored={
                cls.plot_method.__func__,
            }
        )

    def _make_fig_ui(self, fig, *, link_dataframes):
        open_button = pn.widgets.Button(
            name='Open in trace viewer',
            align='center',
        )
        open_button.on_click(lambda event: self.trace.show())

        toolbar = pn.Row(open_button, align='center')

        time_indexed = any(
            'time' in kdim.name.lower()
            for kdims in fig.traverse(attrgetter('kdims'))
            for kdim in kdims
        )

        # Do not automatically link events when the time is not in a key
        # dimension, such as residency bar graphs
        if not link_dataframes and time_indexed:
            link_dataframes = [
                self.ana.notebook.df_all_events(error='log')
            ]

        fig = _hv_link_dataframes(fig, dfs=link_dataframes)
        return pn.Column(
            toolbar,
            fig,
            sizing_mode='stretch_width',
        )

    @classmethod
    def plot_method(cls, f):
        """
        Plot function decorator.

        It provides among other things:

        * automatic plot setup
        * HTML and reStructuredText output.
        * workarounds some holoviews issues
        * integration in other tools
        """

        _decorator = cls.plot_method.__func__
        @update_wrapper_doc(
            f,
            added_by=f':meth:`{_decorator.__module__}.{_decorator.__qualname__}`',
            description=textwrap.dedent("""
            :returns: The return type is determined by the ``output`` parameter.

            :param backend: Holoviews plot library backend to use:

                * ``bokeh``: good support for interactive plots
                * ``matplotlib``: sometimes better static image output, but
                  unpredictable results that more often than not require
                  a fair amount of hacks to get something good.
                * ``plotly``: not supported by LISA but technically
                  available. Since it's very similar to bokeh
                  feature-wise, bokeh should be preferred.

                .. note:: In a notebook, the way to choose which backend should
                    be used to display plots is typically selected with e.g.
                    ``holoviews.extension('bokeh')`` at the beginning of the
                    notebook. The ``backend`` parameter is more intended for
                    expert use where an object of the given library is
                    required, without depending on the environment.

            :type backend: str or None

            :param link_dataframes: Gated by ``output="ui"``. List of
                dataframes to display under the figure, which is dynamically
                linked with it: clicking on the plot will scroll in the
                dataframes and vice versa.
            :type link_dataframes: list(pandas.DataFrame) or None

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

            :param output: Change the return value of the method:

                * ``None``: Equivalent to ``holoviews`` for now. In the future,
                  this will be either ``holoviews`` or ``ui`` if used in an
                  interactive jupyter notebook.
                * ``holoviews``: a bare holoviews element.
                * ``render``: a backend-specific object, such as
                  :class:`matplotlib.figure.Figure` if ``backend='matplotlib'``
                * ``html``: HTML document
                * ``rst``: a snippet of reStructuredText
                * ``ui``: Pseudo holoviews figure, enriched with extra controls.

                  .. note:: No assumption must be made on the return type other
                      than that it can be displayed in a notebook cell output
                      (and with :func:`IPython.display.display`). The public API
                      holoviews is implemented in a best-effort approach, so that
                      ``.options()`` and ``.opts()`` will work, but compositions
                      using e.g. ``x * y`` will not work if ``x`` is a holoviews
                      element.

                      In the midterm, the output type will be changed so that it
                      is a real holoviews object, rather than some sort of proxy.

            :type output: str or None

            :param colors: List of color names to use for the plots.

                .. deprecated:: 2.0
                    This parameter is deprecated, use holoviews APIs to set
                    matplotlib options.

            :type colors: list(str) or None

            :param linestyles: List of linestyle to use for the plots.

                .. deprecated:: 2.0
                    This parameter is deprecated, use holoviews APIs to set
                    matplotlib options.

            :type linestyles: list(str) or None

            :param markers: List of marker to use for the plots.

                .. deprecated:: 2.0
                    This parameter is deprecated, use holoviews APIs to set
                    matplotlib options.

            :type markers: list(str) or None

            :param axis: instance of :class:`matplotlib.axes.Axes` to plot into.
                If `None`, a new figure and axis are created and returned.

                .. deprecated:: 2.0
                    This parameter is deprecated, use holoviews APIs to compose
                    plot elements:
                    http://holoviews.org/user_guide/Composing_Elements.html

            :type axis: matplotlib.axes.Axes
                or numpy.ndarray(matplotlib.axes.Axes)
                or None

            :param rc_params: Matplotlib rc params dictionary overlaid on
                existing settings.

                .. deprecated:: 2.0
                    This parameter is deprecated, use holoviews APIs to set
                    matplotlib options.

            :type rc_params: dict(str, object) or None

            :param _compat_render: Internal parameter not to be used. This
                enables the compatibility mode where ``render=True`` by default
                when matplotlib is the current holoviews backend.
            :type _compat_render: bool
            """),
            include_kwargs=True,
        )
        # Note about default values: the defaults must be chosen so that plot
        # methods can directly call other plot methods internally without
        # unexpected behaviors. Things like _compar_render must therefore
        # default to False here.
        #
        # If for some reason the "user visible" default must be different, it
        # can be changed using the AnalysisProxy(params=dict(...)) when the
        # AnalysisProxy is instanciated in lisa.trace
        def wrapper(self, *args,
            filepath=None,
            output='holoviews',
            img_format=None,
            always_save=False,

            backend=None,
            _compat_render=False,
            link_dataframes=None,
            cursor_delta=None,

            width=None,
            height=None,

            # Deprecated parameters
            rc_params=None,
            axis=None,
            interactive=None,
            colors: typing.Sequence[str]=None,
            linestyles: typing.Sequence[str]=None,
            markers: typing.Sequence[str]=None,

            **kwargs
        ):

            def deprecation_warning(msg):
                warnings.warn(
                    msg,
                    DeprecationWarning,
                    stacklevel=2,
                )

            if interactive is not None:
                deprecation_warning(
                    '"interactive" parameter is deprecated and ignored',
                )
            interactive = is_running_ipython()

            # If the user did not specify a backend, we will return a
            # holoviews object, but we need to know what is the current
            # backend so we can apply the relevant options.
            if backend is None:
                backend = hv.Store.current_backend

            # For backward compat, return a matplotlib Figure when this
            # backend is selected
            if output is None and _compat_render and backend == 'matplotlib':
                output = 'render'

            # Before this point "None" indicates the default.
            if output is None:
                # TODO: Switch the default to be "ui" when interactive once a
                # solution is found for that issue:
                # https://discourse.holoviz.org/t/replace-holoviews-notebook-rendering-with-a-panel/2519/12
                # output = 'ui' if interactive else 'holoviews'
                output = 'holoviews'

            # Deprecated, but allows easy backward compat
            if axis is not None:
                output = 'render'
                deprecation_warning(
                    'axis parameter is deprecated, use holoviews APIs to combine plots (see overloading of ``*`` operator for holoviews elements)'
                )

            if link_dataframes and output != 'ui':
                warnings.warn(f'"link_dataframes" parameter ignored since output != "ui"', stacklevel=2)

            img_format = img_format or guess_format(filepath) or 'png'

            # When we create the figure ourselves, always save the plot to
            # the default location
            if filepath is None and always_save:
                filepath = self.get_default_plot_path(
                    img_format=img_format,
                    plot_name=f.__name__,
                )

            # Factor the *args inside the **kwargs by binding them to the
            # user-facing signature, which is the one of the wrapper.
            kwargs.update(
                inspect.signature(wrapper).bind_partial(self, *args).arguments
            )

            with lisa.notebook._hv_set_backend(backend):
                hv_fig = f(**kwargs)

                # For each element type, only set the option if it has not
                # been set already. This allows the plot method to give
                # customized options that will not be overridden here.
                set_by_method = {}
                for category in ('plot', 'style'):
                    for name, _opts in hv_fig.traverse(
                        lambda element: (
                            element.__class__.name,
                            hv.Store.lookup_options(
                                backend, element, category
                            ).kwargs.keys()
                        )
                    ):
                        set_by_method.setdefault(name, set()).update(_opts)

                def set_options(fig, opts, typs=None, not_typs=None):
                    if typs is None:
                        typs = _hv_has_options(frozenset(opts.keys()), backend=backend)

                    not_typs = set(not_typs or [])
                    typs = {
                        typ
                        for typ in typs
                        if typ not in not_typs
                    }
                    return fig.options(
                        {
                            typ: {
                                k: v
                                for k, v in opts.items()
                                if k not in set_by_method.get(typ, tuple())
                            }
                            for typ in typs
                        },
                        # Specify the backend explicitly, in case the user
                        # asked for a specific backend
                        backend=backend,
                    )

                # Deprecated options
                if colors:
                    deprecation_warning(
                        '"colors" is deprecated and has no effect anymore, use .options() on the resulting holoviews object'
                    )

                if markers:
                    deprecation_warning(
                        '"markers" is deprecated and has no effect anymore, use .options() on the resulting holoviews object'
                    )

                if linestyles:
                    deprecation_warning(
                        '"linestyles" is deprecated and has no effect anymore, use .options() on the resulting holoviews object'
                    )

                if rc_params:
                    deprecation_warning(
                        'rc_params deprecated, use holoviews APIs to set matplotlib parameters'
                    )
                    if backend == 'matplotlib':
                        hv_fig = hv_fig.opts(fig_rcparams=rc_params)
                    else:
                        self.logger.warning('rc_params is only used with matplotlib backend')

                # Markers added by lisa.notebook.plot_signal
                if backend == 'bokeh':
                    marker_opts = dict(
                        # Disable muted legend for now, as they will mute
                        # everything:
                        # https://github.com/holoviz/holoviews/issues/3936
                        # legend_muted=True,
                        muted_alpha=0,
                        tools=[],
                    )
                elif backend == 'matplotlib':
                    # Hide the markers since it clutters static plots, making
                    # them hard to read.
                    marker_opts = dict(
                        visible=False,
                    )
                else:
                    marker_opts = {}

                hv_fig = set_options(
                    hv_fig,
                    opts=marker_opts,
                    typs=('Scatter.marker',),
                )

                # Tools
                if backend == 'bokeh':
                    hv_fig = set_options(
                        hv_fig,
                        opts=dict(tools=[
                            # TODO: revisit:
                            # undo/redo tools are currently broken for some plots:
                            # https://github.com/holoviz/holoviews/issues/5928
                            #
                            # 'undo',
                            # 'redo',
                            'crosshair',
                            'hover',
                        ]),
                        # Setting hover tool for HSpan and VSpan is broken, so
                        # we don't:
                        # https://github.com/holoviz/holoviews/issues/6321
                        not_typs=['HSpan', 'VSpan', 'VLine', 'HLine']
                    ).options(
                        backend=backend,
                        # Sometimes holoviews (or bokeh) decides to put it on
                        # the side, which crops it
                        toolbar='above',
                    )

                # Workaround:
                # https://github.com/holoviz/holoviews/issues/4981
                hv_fig = set_options(
                    hv_fig,
                    opts=dict(color=hv.Cycle()),
                    typs=('Rectangles',),
                )

                # Figure size
                if backend in ('bokeh', 'plotly'):
                    aspect = 4

                    if (width, height) == (None, None):
                        height = 400
                        size = dict(
                            responsive=True,
                        )
                    elif height is None:
                        # There is usually illimited height available. This
                        # will make auto-sizing the height break, so we have to
                        # set it ourselves
                        height = int(width / aspect)
                        size = dict(
                            width=width,
                        )
                    elif width is None:
                        size = dict(
                            # Width is usually limited by the window width, so
                            # responsive mode will work correctly.
                            responsive=True,
                        )
                    else:
                        size = dict(
                            width=width,
                        )

                    # We need to set height in any case, otherwise the
                    # output="ui" will break since plots have illimited
                    # vertical space, and holoviews will then not be able to
                    # provide a height for the object since there is no
                    # external constraint. This will make the layout to squish
                    # it vertically.
                    size['height'] = height

                    hv_fig = set_options(
                        hv_fig,
                        opts=size,
                    )
                elif backend == 'matplotlib':
                    width = 16 if width is None else width
                    height = 4 if height is None else height
                    fig_inches = max(width, height)

                    # Set the 2 options separately so they apply to the maximum
                    # number of types.
                    hv_fig = set_options(
                        hv_fig,
                        opts=dict(aspect=width / height),
                    )
                    # Not doing this on the Layout will prevent getting big
                    # figures, but the "aspect" cannot be set on a Layout
                    hv_fig = set_options(
                        hv_fig,
                        opts=dict(fig_inches=fig_inches),
                    )

                # Use a memoized function to make sure we only do the rendering once
                @memoized
                def rendered_fig():
                    if backend == 'matplotlib':
                        # Make sure to use an interactive renderer for notebooks,
                        # otherwise the plot will not be displayed
                        import holoviews.plotting.mpl
                        renderer = hv.plotting.mpl.MPLRenderer.instance(
                            interactive=interactive
                        )
                        return renderer.get_plot(
                            hv_fig,
                            interactive=interactive,
                            axis=axis,
                            fig=axis.figure if axis else None,
                        ).state
                    else:
                        return hv.renderer(backend).get_plot(hv_fig).state

                def resolve_formatter(fmt):
                    format_map = {
                        'rst': cls._get_rst_content,
                        'sphinx-rst': cls._get_rst_content,
                        'html': cls._get_html,
                        'sphinx-html': cls._get_html,
                    }
                    try:
                        return format_map[fmt]
                    except KeyError:
                        raise ValueError(f'Unsupported format: {fmt}')

                if filepath:
                    if backend in ('bokeh', 'matplotlib') and img_format in ('html', 'sphinx-html', 'rst', 'sphinx-rst'):
                        content = resolve_formatter(img_format)(
                            fmt=img_format,
                            f=f,
                            args=[],
                            kwargs=kwargs,
                            fig=rendered_fig(),
                            backend=backend
                        )

                        with open(filepath, 'wt', encoding='utf-8') as fd:
                            fd.write(content)
                    else:
                        # Avoid cropping the legend on some backends
                        static_fig = set_options(
                            hv_fig,
                            opts=dict(responsive=False),
                        )
                        hv.save(static_fig, filepath, fmt=img_format, backend=backend)

                if output == 'holoviews':
                    out = hv_fig
                # Show the LISA figure toolbar
                elif output == 'ui':
                    # TODO: improve holoviews so we can return holoviews
                    # objects that are displayed with extra widgets around
                    # https://discourse.holoviz.org/t/replace-holoviews-notebook-rendering-with-a-panel/2519/12
                    make_pane = functools.partial(
                        self._make_fig_ui,
                        link_dataframes=link_dataframes,
                    )
                    out = _hv_fig_to_pane(hv_fig, make_pane)
                elif output == 'render':
                    if _compat_render and backend == 'matplotlib':
                        axes = rendered_fig().axes
                        if len(axes) == 1:
                            out = axes[0]
                        else:
                            out = axes
                    else:
                        out = rendered_fig()
                else:
                    out = resolve_formatter(output)(
                        fmt=output,
                        f=f,
                        args=[],
                        kwargs=kwargs,
                        fig=rendered_fig(),
                        backend=backend
                    )

                return out
        return wrapper

    @staticmethod
    def _get_title(f):
        name = f.__name__
        prefix = 'plot_'
        if name.startswith(prefix):
            name = name[len(prefix):]
        name = name.replace('_', ' ').capitalize()
        return name

    @classmethod
    def _get_rst_header(cls, f):
        name = cls._get_title(f)

        try:
            url = get_doc_url(f)
            doc_link = f'`[doc] <{url}>`_'
        except Exception:
            doc_link = ''

        return textwrap.dedent(f"""
                {name}
                {'=' * len(name)}

                {get_short_doc(f, strip_rst=True)} {doc_link}
            """
        )

    @classmethod
    def _get_rst_content(cls, fmt, f, args, kwargs, fig, backend):
        kwargs = inspect.signature(f).bind_partial(*args, **kwargs)
        kwargs.apply_defaults()
        kwargs = kwargs.arguments

        hidden_params = {
            'self',
            'filepath',
            'output',
            'img_format',
            'always_save',

            'backend',
            '_compat_render',
            'link_dataframes',
            'cursor_delta',

            'width',
            'height',

            'colors',
            'linestyles',
            'markers',
            'rc_params',
            'axis',
        }
        args_list = ', '.join(
            f'{k}={v}'
            for k, v in sorted(kwargs.items(), key=itemgetter(0))
            if v is not None and k not in hidden_params
        )

        if backend == 'matplotlib':
            axis = fig.axes
            if len(axis) == 1:
                axis = axis[0]

            fmt = 'png'
            b64_image = cls._get_base64_image(axis, fmt=fmt)

            return textwrap.dedent(f"""
                .. figure:: data:image/{fmt};base64,{b64_image}
                    :alt: {f.__qualname__}
                    :align: center
                    :width: 100%

                    {args_list}
            """)
        elif backend == 'bokeh':
            idt = ' ' * 4
            indent = lambda x: idt + x.replace('\n', '\n' + idt)

            title = args_list
            # Use Sphinx classes to integrate with the theme
            title = f'<p class="caption"><span class="caption-text">{title}</span>'

            js = '\n'.join(bokeh.embed.components(fig))
            # Fixes the exception when using bokeh.io.show() on the same plot.
            # Suggested at:
            # https://stackoverflow.com/questions/39735710/bokeh-models-must-be-owned-by-only-a-single-document
            pn.io.model.remove_root(fig)

            # For a standalone HTML snippet, we need the script tags to import
            # the libraries, but duplicating it in the same page will lead to
            # catastrophic load time, and memory exhaustion so we do it once
            # per page using Sphinx's html_js_files conf to include them.
            if fmt == 'sphinx-rst':
                libs = ''
            else:
                libs = bokeh.resources.CDN.render()

            content = f'<div class="figure align-center">{libs}\n{js}\n{title}</div>'
            return f'.. raw:: html\n\n{indent(content)}'
        else:
            raise ValueError(f'unsupported backend {backend}')

    @classmethod
    def _get_rst(cls, fmt, f, args, kwargs, fig, backend):
        return cls._get_rst_header(f) + '\n' + cls._get_rst_content(
            fmt=fmt,
            f=f,
            args=args,
            kwargs=kwargs,
            fig=fig,
            backend=backend
        )

    @staticmethod
    def _docutils_render(writer, rst, title, doctitle_xform=True):
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
            # Set the line length to always accept our document, since it has a
            # large base64-encoded image in it and docutils will otherwise just
            # replace the document body with an error
            'line_length_limit': len(rst) + 1,
            'title': title,
        }
        parts = docutils.core.publish_parts(
            source=rst, source_path=None,
            destination_path=None,
            writer_name=writer,
            settings_overrides=overrides,
        )

        return parts

    @classmethod
    def _get_html(cls, *, fmt, f, **kwargs):
        fmt_map = {
            'sphinx-html': 'sphinx-rst',
            'html': 'rst',
        }
        rst = cls._get_rst(
            fmt=fmt_map[fmt],
            f=f,
            **kwargs
        )
        parts = cls._docutils_render(
            writer='html',
            rst=rst,
            title=cls._get_title(f)
        )
        return parts['whole']


class TraceAnalysisBase(AnalysisHelpers):
    """
    Base class for Analysis modules.

    :param trace: input Trace object
    :type trace: lisa.trace.Trace

    :Design notes:

    Method depending on certain trace events *must* be decorated with
    :meth:`lisa.trace.requires_events`
    """

    def __init__(self, trace, proxy=None):
        self.trace = trace
        self.ana = proxy or trace.ana

    @classmethod
    def get_df_methods(cls, *args, **kwargs):
        return cls._get_doc_methods(
            *args,
            prefix='df_',
            **kwargs,
            ignored={
                cls.df_method.__func__,
            }
        )

    @optional_kwargs
    @classmethod
    def df_method(cls, f, index=None):
        """
        Dataframe function decorator.

        It provides among other things:

        * Dataframe format conversion
        """

        # Apply caching to all df-returning functions. This way we also
        # guarantee that the df_fmt is properly applied even when data are
        # coming from the cache.
        cached_f = cls.cache(fmt='parquet')(f)

        _decorator = cls.df_method.__func__
        @update_wrapper_doc(
            f,
            added_by=f':meth:`{_decorator.__module__}.{_decorator.__qualname__}`',
            description=textwrap.dedent("""
            :param df_fmt: Format of dataframe to return. One of:

                * ``"pandas"``: :class:`pandas.DataFrame`
                * ``"polars-lazyframe"``: :class:`polars.LazyFrame`

            :type df_fmt: str or None

            :returns: The return type is determined by the dataframe format
                chosen for the trace object.
            """),
            include_kwargs=False,
        )
        # Note about default values: the defaults must be chosen so that df
        # methods can directly call other plot methods internally without
        # unexpected behaviors.
        #
        # If for some reason the "user visible" default must be different, it
        # can be changed using the AnalysisProxy(params=dict(...)) when the
        # AnalysisProxy is instanciated in lisa.trace
        def wrapper(self, *args, df_fmt=None, **kwargs):
            # Ease working with LazyFrames coming from various sources. When
            # they are collect()'ed in f(), they will be created using a common
            # StringCache so Categorical columns can be concatenated and such.
            with pl.StringCache():

                # We might get different types based on whether the content
                # comes from the function directly (could be a pandas object)
                # or from the cache (polars LazyFrame).
                df = cached_f(self, *args, **kwargs)
                assert isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame))

                df_fmt = df_fmt or 'pandas'
                df = _df_to(
                    df,
                    fmt=df_fmt,
                    index=(
                        ('Time' if 'Time' in df.collect_schema().names() else None)
                        if index is None and isinstance(df, (pl.LazyFrame, pl.DataFrame)) else
                        index
                    ),
                )
                return df

        return wrapper

    @optional_kwargs
    @classmethod
    def cache(cls, f, fmt='parquet', ignored_params=None):
        """
        Decorator to enable caching of the output of dataframe getter function
        in the trace cache.

        This will write the return data to the swap as well, so processing can be
        skipped completely when possible.

        :param fmt: Format of the data to write to the cache. This will
            influence the extension of the cache file created. If ``disk-only``
            format is chosen, the data is not retained in memory and the path
            to the allocated cache file is passed as first parameter to the
            wrapped function. This allows manual management of the file's
            content, as well having a path to a file to pass to external tools
            if they can consume the data directly.
        :type fmt: str

        :param ignored_params: Parameters to ignore when trying to hit the
            cache.
        :type ignored_params: list(str)
        """
        ignored_kwargs = set(ignored_params or [])

        sig = inspect.signature(f)
        parameter_names = list(sig.parameters.keys())

        # Ignore "self"
        ignored_kwargs.add(parameter_names[0])

        memory_cache = fmt != 'disk-only'

        if not memory_cache:
            path_param = parameter_names[1]
            ignored_kwargs.add(path_param)

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            # Make some room for the argument we will fill later
            if not memory_cache:
                args = (None,) + args

            # Express the arguments as kwargs-only
            params = sig.bind(self, *args, **kwargs)
            params.apply_defaults()

            trace = self.trace
            spec = dict(
                bound_class=(
                    self.__class__.__module__,
                    self.__class__.__qualname__,
                ),
                module=f.__module__,
                func=f.__qualname__,
                # Include the trace window in the spec since that influences
                # what the analysis was seeing
                trace_state=trace.trace_state,
                # Make a deepcopy as it is critical that the _CacheDataDesc is
                # not modified under the hood once inserted in the cache
                kwargs=copy.deepcopy({
                    k: v
                    for k, v in params.arguments.items()
                    if k not in ignored_kwargs
                }),
            )
            cache_desc = _CacheDataDesc(spec=spec, fmt=fmt)
            cache = trace._cache

            def call_f():
                if not memory_cache:
                    try:
                        swap_path = cache._cache_desc_swap_path(cache_desc, create=True)
                    except Exception as e:
                        swap_path = None
                    params.arguments[path_param] = swap_path

                with measure_time() as measure:
                    data = f(*params.args, **params.kwargs)

                if isinstance(data, pd.DataFrame):
                    data = _pandas_cleanup_df(data)

                if memory_cache:
                    # Do not use measure.exclusive_delta, otherwise a simple
                    # function making thousands of quick calls to a child
                    # function may appear with a low cost, even though it
                    # actually has a high total cost.
                    compute_cost = measure.delta
                else:
                    compute_cost = None

                cache.insert(cache_desc, data, compute_cost=compute_cost, write_swap='best-effort')
                return data

            if memory_cache:
                try:
                    # Be warned that the type of the data returned by the cache
                    # may not match what was inserted. This can happen notably
                    # when a dataframe (from either pandas or polars) is
                    # cached, as it will be stored in a parquet file and
                    # reloaded most likely as a polars LazyFrame.
                    data = cache.fetch(cache_desc)
                except KeyError:
                    data = call_f()
            else:
                data = call_f()

            return data

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
        # Import all the submodules so that we have full visibility over the
        # subclasses.
        import lisa.analysis as ana
        _import_all_submodules(ana.__name__, ana.__path__)

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
        classes = cls.get_analysis_classes().values()
        subcls = get_parent_namespace(meth)
        if subcls in classes:
            # Create an analysis instance and bind the method to it
            analysis = subcls(trace=trace)
            meth = meth.__get__(analysis, type(analysis))

            return meth(**meth_kwargs)
        else:
            raise ValueError(f'Parent class of {meth} is not a registered analysis')

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
