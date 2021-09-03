#! /usr/bin/env python3
#
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

import os
import sys
import argparse
import inspect
from collections import OrderedDict
import contextlib
from tempfile import NamedTemporaryFile
import webbrowser
import time
import types

from lisa.utils import get_short_doc, nullcontext
from lisa.trace import Trace
from lisa.conf import ConfigKeyError
from lisa.analysis.base import TraceAnalysisBase
from lisa.platforms.platinfo import PlatformInfo
from lisa._typeclass import FromString


def error(msg, ret=1):
    print(msg, file=sys.stderr)
    if ret is None:
        return
    else:
        sys.exit(ret)

def make_meth_name(analysis, f):
    name = f.__name__
    analysis = get_analysis_nice_name(analysis)

    def remove_prefix(prefix, name):
        if name.startswith(prefix):
            return name[len(prefix):]
        else:
            return name

    name = remove_prefix('plot_', name)
    # Remove the analysis name from the method name, which is not common but
    # happens for some of the methods. This avoids verbose redundancy when
    # sticking the analysis name in front of it.
    name = remove_prefix(analysis, name)
    name = name.replace('_', '-').strip('-')

    return '{}:{}'.format(analysis, name)

def get_analysis_nice_name(name):
    return name.replace('_', '-')

def get_plots_map():
    plots_map = {}
    for name, cls in TraceAnalysisBase.get_analysis_classes().items():

        methods = [
            meth
            for meth in cls.get_plot_methods()
            # Method that need extra arguments are not usable by this
            # script
            if meth_usable_args(meth)
        ]

        if methods:
            plots_map[name] = {
                make_meth_name(name, meth): meth
                for meth in methods
            }
    return plots_map

def meth_usable_args(f):
    """
    Returns True when the arguments of the ``f`` can be handled.
    """
    sig = inspect.signature(f)
    parameters = OrderedDict(sig.parameters)
    # ignore first param ("self") since these are methods
    parameters.popitem(last=False)
    return not any(
        # Parameters without default value
        param.default == inspect.Parameter.empty
        # and that are not *args or **kwargs
        and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD
        )
        # Parameter without type annotation
        and param.annotation == inspect.Parameter.empty
        for param in parameters.values()
    )

def make_plot_kwargs(meth, file_path, extra_options, backend='bokeh'):
    """
    Make a keyword arguments dict for the given plot method.

    :param meth: Method that will consume the kwargs.
    :type meth: collections.abc.Callable

    :type file_path: Filepath of the generated plot.
    :type file_path: str

    :param extra_options: Extra options passed on the command line, as strings.
    :type param: dict(str, str)
    """
    sig = inspect.signature(meth)

    options = dict(extra_options)
    kwargs = {
        name: FromString(param.annotation).from_str(options[name])
        for name, param in sig.parameters.items()
        if (
            name in options
            and param.annotation is not inspect.Parameter.empty
        )
    }
    kwargs.update(
        filepath=file_path,
        backend=backend,
    )
    return kwargs

@contextlib.contextmanager
def handle_plot_excep(exit_on_error=True):
    try:
        yield
    except ConfigKeyError as e:
        try:
            key = e.args[1]
        except IndexError:
            excep_msg = str(e)
        else:
            excep_msg = 'Please specify --plat-info with the "{}" filled in'.format(e.args[1])
    except Exception as e:
        excep_msg = str(e)
    else:
        excep_msg = None

    if excep_msg:
        error(excep_msg, -1 if exit_on_error else None)

def get_meth_options_help(meth):
    sig = inspect.signature(meth)

    def get_help(cls):
        return FromString(cls).get_format_description(short=True)

    parameters = ', '.join(
        '{} ({})'.format(name, get_help(param.annotation))
        for name, param in sig.parameters.items()
        if param.annotation != inspect.Parameter.empty
    )
    if parameters:
        return '\n    Options: ' + parameters
    else:
        return ''

def get_analysis_listing(plots_map):
    return '\n'.join(
        '* {} analysis:\n  {}\n'.format(
            get_analysis_nice_name(analysis_name),
            '\n  '.join(
                '{name}: {doc}{params}'.format(
                    name=name,
                    doc=get_short_doc(meth),
                    params=get_meth_options_help(meth)
                )
                for name, meth in methods.items()

            ),
        )
        for analysis_name, methods in plots_map.items()
    )

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    plots_map = get_plots_map()
    analysis_nice_name_map = {
        get_analysis_nice_name(name): name
        for name in plots_map.keys()
    }

    parser = argparse.ArgumentParser(description="""
CLI for LISA analysis plots and reports from traces.

Available plots:

{}

""".format(get_analysis_listing(plots_map)),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('trace',
        help='trace-cmd trace.dat, or systrace file',
    )

    parser.add_argument('--normalize-time', action='store_true',
        help='Normalize the time in the plot, i.e. start at 0 instead of uptime timestamp',
    )

    parser.add_argument('--plot', nargs=2, action='append',
        default=[],
        metavar=('PLOT', 'OUTPUT_PATH'),
        help='Create the given plot. If OUTPUT_PATH is "interactive", the plot will be opened in a window',
    )

    parser.add_argument('--plot-analysis', nargs=3, action='append',
        default=[],
        metavar=('ANALYSIS', 'OUTPUT_FOLDER_PATH', 'FORMAT'),
        help='Create all the plots of the given analysis',
    )

    parser.add_argument('--plot-all', nargs=2,
        metavar=('OUTPUT_FOLDER_PATH', 'FORMAT'),
        help='Create all the plots in the given folder',
    )

    parser.add_argument('--best-effort', action='store_true',
        help='Try to generate as many of the requested plots as possible without early termination.',
    )

    parser.add_argument('--window', nargs=2, type=float,
        metavar=('BEGIN', 'END'),
        help='Only plot data between BEGIN and END times',
    )

    parser.add_argument('-X', '--option', nargs=2, action='append',
        default=[],
        metavar=('OPTION_NAME', 'VALUE'),
        help='Pass extra parameters to plot methods, e.g. "-X cpu 1". Mismatching names are ignored.',
    )

    parser.add_argument('--plat-info',
        help='Platform information, necessary for some plots',
    )

    args = parser.parse_args(argv)

    flat_plot_map = {
        plot_name: meth
        for analysis_name, plot_list in plots_map.items()
        for plot_name, meth in plot_list.items()
    }

    if args.plat_info:
        plat_info = PlatformInfo.from_yaml_map(args.plat_info)
    else:
        plat_info = None

    if args.plot_all:
        folder, fmt = args.plot_all
        plot_analysis_spec_list = [
            (get_analysis_nice_name(analysis_name), folder, fmt)
            for analysis_name in plots_map.keys()
        ]
    else:
        plot_analysis_spec_list = []

    plot_analysis_spec_list.extend(args.plot_analysis)

    plot_spec_list = [
        (plot_name, os.path.join(folder, '{}.{}'.format(plot_name, fmt)))
        for analysis_name, folder, fmt in plot_analysis_spec_list
        for plot_name, meth in plots_map[analysis_nice_name_map[analysis_name]].items()
    ]

    plot_spec_list.extend(args.plot)

    # Build minimal event list to speed up trace loading time
    plot_methods = set()
    for plot_name, file_path in plot_spec_list:
        try:
            f = flat_plot_map[plot_name]
        except KeyError:
            error('Unknown plot "{}", see --help'.format(plot_name))

        plot_methods.add(f)

    # If best effort is used, we don't want to trigger exceptions ahead of
    # time. Let it fail for individual plot methods instead, so the trace can
    # be used for the other events
    if args.best_effort:
        events = None
    else:
        events = set()
        for f in plot_methods:
            with contextlib.suppress(AttributeError):
                events.update(f.used_events.get_all_events())

        events = sorted(events)
        print('Parsing trace events: {}'.format(', '.join(events)))

    trace = Trace(args.trace, plat_info=plat_info, events=events, normalize_time=args.normalize_time, write_swap=True)
    if args.window:
        window = args.window
        def clip(l, x, r):
            if x < l:
                return l
            elif x > r:
                return r
            else:
                return x

        window = (
            clip(trace.window[0], window[0], trace.window[1]),
            clip(trace.window[0], window[1], trace.window[1]),
        )
        # There is no overlap between trace and user window, reset to trace
        # window
        if window[0] == window[1]:
            print('Window {} does not overlap with trace time range, maybe you forgot --normalize-time ?'.format(tuple(args.window)))
            window = trace.window

        trace = trace.get_view(window)

    for plot_name, file_path in sorted(plot_spec_list):
        interactive = file_path == 'interactive'
        if interactive:
            outfile_cm = NamedTemporaryFile(suffix='.html')
        else:
            outfile_cm = nullcontext(
                types.SimpleNamespace(name=file_path)
            )
            dirname = os.path.dirname(file_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

        with outfile_cm as outfile:
            _file_path = outfile.name

            f = flat_plot_map[plot_name]
            kwargs = make_plot_kwargs(
                f,
                file_path=_file_path,
                extra_options=args.option
            )

            with handle_plot_excep(exit_on_error=not args.best_effort):
                fig = TraceAnalysisBase.call_on_trace(f, trace, kwargs)

            if interactive:
                webbrowser.open_new(_file_path)
                # Wait for the page to load before the file is removed
                time.sleep(1)


if __name__ == '__main__':
    sys.exit(main())

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
