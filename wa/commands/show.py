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
from subprocess import call, Popen, PIPE

from wa import Command
from wa.framework import pluginloader
from wa.framework.exception import NotFoundError
from wa.framework.target.descriptor import get_target_descriptions
from wa.utils.doc import (strip_inlined_text, get_rst_from_plugin,
                          get_params_rst, underline)
from wa.utils.misc import which

from devlib.utils.misc import escape_double_quotes

class ShowCommand(Command):

    name = 'show'
    description = 'Display documentation for the specified plugin (workload, instrument, etc.).'

    def initialize(self, context):
        self.parser.add_argument('plugin', metavar='PLUGIN',
                                 help='The name of the plugin to display documentation for.')

    def execute(self, state, args):
        name = args.plugin
        rst_output = None

        plugin = get_plugin(name)
        if plugin:
            rst_output = get_rst_from_plugin(plugin)
        else:
            target = get_target_description(name)
            if target:
                rst_output = get_rst_from_target(target)

        if not rst_output:
            raise NotFoundError('Could not find plugin or alias "{}"'.format(name))

        if which('pandoc'):
            p = Popen(['pandoc', '-f', 'rst', '-t', 'man'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, _ = p.communicate(rst_output)
            # Make sure to double escape back slashes
            output = output.replace('\\', '\\\\\\')
            # Correctly format the tile of the man page
            output = output.replace('.SH', '.TH', 1)
            call('echo "{}" | man -l -'.format(escape_double_quotes(output)), shell=True)
        else:
            print rst_output


def get_plugin(name):
    for plugin in pluginloader.list_plugins():
        if name == plugin.name:
            return plugin
        if hasattr(plugin, 'alias'):
            for alias in plugin.alias:
                if name == alias:
                    return plugin


def get_target_description(name):
    targets = get_target_descriptions()
    for target in targets:
        if name == target.name:
            return target


def get_rst_from_target(target):
    text = underline(target.name, '-')
    if hasattr(target, 'description'):
        desc = strip_inlined_text(target.description or '')
        text += desc
    conn_params_rst = get_params_rst(target.conn_params)
    if conn_params_rst:
        text += underline('\nconnection parameters', '~') + conn_params_rst
    platform_params_rst = get_params_rst(target.platform_params)
    if platform_params_rst:
        text += underline('\nplatform parameters', '~') + platform_params_rst
    target_params_rst = get_params_rst(target.target_params)
    if target_params_rst:
        text += underline('\nconnection parameters', '~') + target_params_rst
    return text + '\n'
