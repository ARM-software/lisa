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
from collections.abc import Mapping

from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives import flag
from docutils import nodes
from docutils.statemachine import ViewList

import lisa.analysis
from lisa.analysis.base import AnalysisHelpers, TraceAnalysisBase
from lisa.utils import get_subclasses
from lisa.trace import MissingTraceEventError
from lisa.conf import SimpleMultiSrcConf, TopLevelKeyDesc, KeyDesc, LevelKeyDesc

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
            self.state.nested_parse(txt, self.content_offset, node)
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

    if method.__name__.startswith('test_'):
        return True

    # Tests are methods with an annotated return type, with at least
    # one base class with a name containing 'result'
    try:
        ret_type = inspect.signature(method).return_annotation
        base_cls_list = inspect.getmro(ret_type)
    except (AttributeError, KeyError):
        return False
    else:
        return any(
            'result' in cls.__qualname__.lower()
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
    # Append the list of required trace events
    if what != 'method' or not hasattr(obj, "used_events"):
        return

    events_doc = "\n:Required trace events:\n\n{}\n\n".format(obj.used_events.doc_str())
    lines.extend(events_doc.splitlines())


class DocPlotConf(SimpleMultiSrcConf):
    """
    Analysis plot method arguments configuration for the documentation.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('doc-plot-conf', 'Plot methods configuration', (
        KeyDesc('plots', 'Mapping of function qualnames to their settings', [Mapping]),
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

    print('Generating plot for {}'.format(obj.__qualname__))
    rst_figure = TraceAnalysisBase.call_on_trace(obj, trace, {
        'output': 'rst',
        'always_save': False,
        # avoid memory leaks
        'interactive': False,
        **kwargs
    })
    rst_figure = '\n:Example plot:\n\n{}'.format(rst_figure)
    lines.extend(rst_figure.splitlines())


def get_analysis_list(meth_type):
    rst_list = []

    for subclass in get_subclasses(AnalysisHelpers):
        class_path = "{}.{}".format(subclass.__module__, subclass.__qualname__)
        analysis = subclass.__module__.split(".")[-1]
        if meth_type == 'plot':
            meth_list = [
                f.__name__
                for f in subclass.get_plot_methods()
            ]
        elif meth_type == 'df':
            meth_list = [
                name
                for name, member in inspect.getmembers(subclass, callable)
                if name.startswith('df_')
            ]
        else:
            raise ValueError()

        rst_list += [
            ":class:`{analysis_name}<{cls}>`::meth:`~{cls}.{meth}`".format(
                analysis_name=analysis,
                cls=class_path,
                meth=meth,
            )
            for meth in meth_list
        ]

    joiner = '\n* '
    return joiner + joiner.join(rst_list)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
