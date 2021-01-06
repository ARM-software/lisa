# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
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
import contextlib
import subprocess
import inspect
import itertools
import logging
import functools
import re
import types
from collections.abc import Mapping
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from operator import itemgetter

from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives import flag
from docutils import nodes
from docutils.statemachine import ViewList

from sphinx.util.nodes import nested_parse_with_titles

import lisa
import lisa.analysis
from lisa.analysis.base import AnalysisHelpers, TraceAnalysisBase
from lisa.utils import get_subclasses, import_all_submodules, DEPRECATED_MAP, get_sphinx_name, groupby
from lisa.trace import MissingTraceEventError, TraceEventCheckerBase
from lisa.conf import SimpleMultiSrcConf, TopLevelKeyDesc, KeyDesc, LevelKeyDesc
from lisa.version import format_version


class RecursiveDirective(Directive):
    """
    Base class helping nested parsing.

    Options:

    * ``literal``: If set, a literal block will be used, otherwise the text
      will be interpreted as reStructuredText.
    """
    option_spec = {
        'literal': flag,
    }

    def parse_nested(self, txt, source=None):
        """
        Parse text as reStructuredText if the ``literal`` option is not set.

        Otherwise, interpret the text as a line block.
        """
        if 'literal' in self.options:
            node = nodes.literal_block(txt, txt, classes=[])
            # Avoid syntax highlight
            node['language'] = 'text'
            return [node]
        else:
            txt = ViewList(txt.splitlines(), source)
            node = nodes.Element()
            nested_parse_with_titles(self.state, txt, node)
            return node.children


class ExecDirective(RecursiveDirective):
    """
    reStructuredText directive to execute the specified python code and insert
    the output into the document::

        .. exec::

            import sys
            print(sys.version)

    Options:

    * ``literal``: If set, a literal block will be used, otherwise the text
      will be interpreted as reStructuredText.
    """
    has_content = True

    def run(self):
        stdout = io.StringIO()
        code = '\n'.join(self.content)
        with contextlib.redirect_stdout(stdout):
            exec(code)
        out = stdout.getvalue()
        return self.parse_nested(out)


directives.register_directive('exec', ExecDirective)


class RunCommandDirective(RecursiveDirective):
    """
    reStructuredText directive to execute the specified command and insert
    the output into the document::

        .. run-command::
           :capture-stderr:
           :ignore-error:
           :literal:

           exekall --help

    Options:

    * ``literal``: If set, a literal block will be used, otherwise the text
      will be interpreted as reStructuredText.
    * ``capture-stderr``: If set, stderr will be captured in addition to stdout.
    * ``ignore-error``: The return status of the command will be ignored.
      Otherwise, it will raise an exception and building the
      documentation will fail.
    """
    has_content = True
    option_spec = {
        'ignore-error': flag,
        'capture-stderr': flag,
        'literal': flag,
    }

    def run(self):
        options = self.options
        if 'capture-stderr' in options:
            stderr = subprocess.STDOUT
        else:
            stderr = None

        check = False if 'ignore-error' in options else True

        cmd = '\n'.join(self.content)

        out = subprocess.run(
            cmd, shell=True, check=check,
            stdout=subprocess.PIPE, stderr=stderr,
        ).stdout.decode('utf-8')

        return self.parse_nested(out, cmd)


directives.register_directive('run-command', RunCommandDirective)


def is_test(method):
    """
    Check if a method is a test method.
    """
    if not callable(method):
        return False

    with contextlib.suppress(AttributeError):
        if method.__name__.startswith('test_'):
            return True

    # Tests are methods with an annotated return type, with at least
    # one base class with a name containing 'result'
    try:
        ret_type = inspect.signature(method).return_annotation
        base_cls_list = inspect.getmro(ret_type)
    except (ValueError, AttributeError, KeyError):
        return False
    else:
        return any(
            'result' in cls.__qualname__.casefold()
            for cls in base_cls_list
        )


def autodoc_process_test_method(app, what, name, obj, options, lines):
    # Append the list of available test methods for all classes that appear to
    # have some.
    if what == 'class':
        test_list = [
            member
            for member_name, member in inspect.getmembers(obj, is_test)
        ]
        if test_list:
            test_list_doc = '\n:Test methods:\n\n{}\n\n'.format('\n'.join(
                '    * :meth:`~{}`'.format(
                    method.__module__ + '.' + method.__qualname__
                )
                for method in test_list
            ))

            lines.extend(test_list_doc.splitlines())


def autodoc_process_analysis_events(app, what, name, obj, options, lines):
    """
    Append the list of required trace events
    """

    # We look for events in the getter method of properties
    if what == 'property':
        obj = obj.fget

    try:
        used_events = obj.used_events
    except AttributeError:
        return
    else:
        if not isinstance(used_events, TraceEventCheckerBase):
            return

        events_doc = f"\n:Required trace events:\n\n{used_events.doc_str()}\n\n"
        lines.extend(events_doc.splitlines())


class DocPlotConf(SimpleMultiSrcConf):
    """
    Analysis plot method arguments configuration for the documentation.

    {generated_help}
    {yaml_example}
    """
    STRUCTURE = TopLevelKeyDesc('doc-plot-conf', 'Plot methods configuration', (
        # Avoid deepcopy of the value, since it contains a Trace object that we
        # don't want to duplicate for speed reasons
        KeyDesc('plots', 'Mapping of function qualnames to their settings', [Mapping], deepcopy_val=False),
    ))


def autodoc_process_analysis_plots(app, what, name, obj, options, lines, plot_conf):
    if what != 'method':
        return

    plot_methods = set(itertools.chain.from_iterable(
        subclass.get_plot_methods()
        for subclass in get_subclasses(TraceAnalysisBase)
    ))

    if obj not in plot_methods:
        return

    plot_conf = plot_conf['plots']

    default_spec = plot_conf.get('default', {})
    spec = plot_conf.get(obj.__qualname__, {})
    spec = {**default_spec, **spec}
    kwargs = spec.get('kwargs', {})
    trace = spec['trace']

    if spec.get('hide'):
        return

    print(f'Generating plot for {obj.__qualname__}')
    rst_figure = TraceAnalysisBase.call_on_trace(obj, trace, {
        'output': 'rst',
        # avoid memory leaks
        'interactive': False,
        **kwargs
    })
    rst_figure = f'\n:Example plot:\n\n{rst_figure}'
    lines.extend(rst_figure.splitlines())


def autodoc_process_analysis_methods(app, what, name, obj, options, lines):
    """
    Append the list of required trace events
    """
    methods = {
        func: subclass
        for subclass in get_subclasses(TraceAnalysisBase)
        for name, func in inspect.getmembers(subclass, callable)
    }

    try:
        cls = methods[obj]
    except (KeyError, TypeError):
        return
    else:
        on_trace_name = 'trace.analysis.{}.{}'.format(
            cls.name,
            obj.__name__
        )
        extra_doc = f"\n*Called on* :class:`~lisa.trace.Trace` *instances as* ``{on_trace_name}()``\n\n"
        # prepend
        lines[:0] = extra_doc.splitlines()


def get_analysis_list(meth_type):
    rst_list = []

    deprecated = {
        entry['obj']
        for entry in get_deprecated_map().values()
    }

    for subclass in get_subclasses(AnalysisHelpers):
        class_path = f"{subclass.__module__}.{subclass.__qualname__}"
        if meth_type == 'plot':
            meth_list = subclass.get_plot_methods()
        elif meth_type == 'df':
            meth_list = [
                member
                for name, member in inspect.getmembers(subclass, callable)
                if name.startswith('df_')
            ]
        else:
            raise ValueError()

        meth_list = [
            f.__name__
            for f in meth_list
            if f not in deprecated
        ]

        rst_list += [
            ":class:`{analysis_name}<{cls}>`::meth:`~{cls}.{meth}`".format(
                analysis_name=subclass.name,
                cls=class_path,
                meth=meth,
            )
            for meth in meth_list
        ]

    joiner = '\n* '
    return joiner + joiner.join(sorted(rst_list))


def find_dead_links(content):
    """
    Look for HTTP URLs in ``content`` and return a dict of URL to errors when
    trying to open them.
    """
    regex = r"https?://[^\s]+"
    links = re.findall(regex, content)

    @functools.lru_cache(maxsize=None)
    def check(url):
        # Some HTTP servers (including ReadTheDocs) will return 403 Forbidden
        # if no User-Agent is given
        headers={
            'User-Agent': 'Wget/1.13.4 (linux-gnu)',
        }
        request = Request(url, headers=headers)
        try:
            urlopen(request)
        except (HTTPError, URLError) as e:
            return e.reason
        else:
            return None

    errors = {
        link: check(link)
        for link in links
        if check(link) is not None
    }
    return errors


def check_dead_links(filename):
    """
    Check ``filename`` for broken links, and raise an exception if there is any.
    """
    with open(filename) as f:
        dead_links = find_dead_links(f.read())

    if dead_links:
        raise RuntimeError('Found dead links in {}:\n  {}'.format(
            filename,
            '\n  '.join(
                f'{url}: {error}'
                for url, error in dead_links.items()
            )))


def get_deprecated_map():
    """
    Get the mapping of deprecated names with some metadata.
    """

    # Import everything there is to import, so the map is fully populated
    import_all_submodules(lisa)
    return DEPRECATED_MAP

def get_deprecated_table():
    """
    Get a reStructuredText tables with titles for all the deprecated names in
    :mod:`lisa`.
    """

    def indent(string, level=1):
        idt = ' ' * 4
        return string.replace('\n', '\n' + idt * level)

    def make_entry(entry):
        msg = entry.get('msg') or ''
        removed_in = entry.get('removed_in')
        if removed_in is None:
            removed_in = ''
        else:
            removed_in = '*Removed in: {}*\n\n'.format(format_version(removed_in))

        name = get_sphinx_name(entry['obj'], style='rst')
        replaced_by = entry.get('replaced_by')
        if replaced_by is None:
            replaced_by = ''
        else:
            replaced_by = '*Replaced by:* {}\n\n'.format(get_sphinx_name(replaced_by, style='rst'))

        return "* - {name}{msg}{replaced_by}{removed_in}".format(
            name=indent(name + '\n\n'),
            msg=indent(msg + '\n\n' if msg else ''),
            replaced_by=indent(replaced_by),
            removed_in=indent(removed_in),
        )

    def make_table(entries, removed_in):
        entries = '\n'.join(
            make_entry(entry)
            for entry in sorted(entries, key=itemgetter('name'))
        )
        if removed_in:
            if removed_in > lisa.version.version_tuple:
                remove = 'to be removed'
            else:
                remove = 'removed'
            removed_in = ' {} in {}'.format(remove, format_version(removed_in))
        else:
            removed_in = ''

        table = ".. list-table:: Deprecated names{removed_in}\n    :align: left{entries}".format(
            entries=indent('\n\n' + entries),
            removed_in=removed_in,
        )
        header = f'Deprecated names{removed_in}'
        header += '\n' + '+' * len(header)

        return header + '\n\n' + table

    entries = [
        {'name': name, **info}
        for name, info in get_deprecated_map().items()
    ]

    unspecified_removal = [
        entry
        for entry in entries
        if not entry['removed_in']
    ]

    other_entries = [
        entry
        for entry in entries
        if entry not in unspecified_removal
    ]

    tables = []
    tables.append(make_table(unspecified_removal, removed_in=None))
    tables.extend(
        make_table(entries, removed_in=removed_in)
        for removed_in, entries in groupby(other_entries, itemgetter('removed_in'), reverse=True)
    )

    return '\n\n'.join(tables)


def get_xref_type(obj):
    """
    Infer the Sphinx type a cross reference to ``obj`` should have.

    For example, ``:py:class`FooBar`` has the type ``py:class``.
    """
    if isinstance(obj, type):
        if issubclass(obj, BaseException):
            t = 'exc'
        else:
            t = 'class'
    elif isinstance(obj, types.ModuleType):
        t = 'mod'
    elif callable(obj):
        try:
            qualname = obj.__qualname__
        except AttributeError:
            t = 'func'
        else:
            if len(qualname.split('.')) > 1:
                t = 'meth'
            else:
                t = 'func'
    else:
        raise ValueError(f'Cannot infer the xref type of {obj}')

    return f'py:{t}'

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
