#    Copyright 2014-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import sys
import subprocess
from cStringIO import StringIO

from wlauto import Command
from wlauto.core.config.core import settings
from wlauto.core import pluginloader
from wlauto.utils.doc import (get_summary, get_description, get_type_name, format_column, format_body,
                              format_paragraph, indent, strip_inlined_text)
from wlauto.utils.misc import get_pager
from wlauto.utils.terminalsize import get_terminal_size


class ShowCommand(Command):

    name = 'show'

    description = """
    Display documentation for the specified plugin (workload, instrument, etc.).
    """

    def initialize(self, context):
        self.parser.add_argument('name', metavar='EXTENSION',
                                 help='''The name of the plugin for which information will
                                         be shown.''')

    def execute(self, args):
        # pylint: disable=unpacking-non-sequence
        plugin = pluginloader.get_plugin_class(args.name)
        out = StringIO()
        term_width, term_height = get_terminal_size()
        format_plugin(plugin, out, term_width)
        text = out.getvalue()
        pager = get_pager()
        if len(text.split('\n')) > term_height and pager:
            try:
                sp = subprocess.Popen(pager, stdin=subprocess.PIPE)
                sp.communicate(text)
            except OSError:
                self.logger.warning('Could not use PAGER "{}"'.format(pager))
                sys.stdout.write(text)
        else:
            sys.stdout.write(text)


def format_plugin(plugin, out, width):
    format_plugin_name(plugin, out)
    out.write('\n')
    format_plugin_summary(plugin, out, width)
    out.write('\n')
    if hasattr(plugin, 'supported_platforms'):
        format_supported_platforms(plugin, out, width)
        out.write('\n')
    if plugin.parameters:
        format_plugin_parameters(plugin, out, width)
        out.write('\n')
    format_plugin_description(plugin, out, width)


def format_plugin_name(plugin, out):
    out.write('\n{}\n'.format(plugin.name))


def format_plugin_summary(plugin, out, width):
    out.write('{}\n'.format(format_body(strip_inlined_text(get_summary(plugin)), width)))


def format_supported_platforms(plugin, out, width):
    text = 'supported on: {}'.format(', '.join(plugin.supported_platforms))
    out.write('{}\n'.format(format_body(text, width)))


def format_plugin_description(plugin, out, width):
    # skip the initial paragraph of multi-paragraph description, as already
    # listed above.
    description = get_description(plugin).split('\n\n', 1)[-1]
    out.write('{}\n'.format(format_body(strip_inlined_text(description), width)))


def format_plugin_parameters(plugin, out, width, shift=4):
    out.write('parameters:\n\n')
    param_texts = []
    for param in plugin.parameters:
        description = format_paragraph(strip_inlined_text(param.description or ''), width - shift)
        param_text = '{}'.format(param.name)
        if param.mandatory:
            param_text += " (MANDATORY)"
        param_text += '\n{}\n'.format(description)
        param_text += indent('type: {}\n'.format(get_type_name(param.kind)))
        if param.allowed_values:
            param_text += indent('allowed values: {}\n'.format(', '.join(map(str, param.allowed_values))))
        elif param.constraint:
            param_text += indent('constraint: {}\n'.format(get_type_name(param.constraint)))
        if param.default is not None:
            param_text += indent('default: {}\n'.format(param.default))
        param_texts.append(indent(param_text, shift))

    out.write(format_column('\n'.join(param_texts), width))
