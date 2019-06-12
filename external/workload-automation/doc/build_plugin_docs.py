#!/usr/bin/env python
#    Copyright 2014-2019 ARM Limited
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


import os
import sys

from wa import pluginloader
from wa.framework.configuration.core import RunConfiguration, MetaConfiguration
from wa.framework.target.descriptor import list_target_descriptions
from wa.utils.doc import (strip_inlined_text, get_rst_from_plugin,
                          get_params_rst, underline, line_break)
from wa.utils.misc import capitalize

GENERATE_FOR_PACKAGES = [
    'wa.workloads',
    'wa.instruments',
    'wa.output_processors',
]


def insert_contents_table(title='', depth=1):
    """
    Insert a sphinx directive to insert a contents page with
    a configurable title and depth.
    """
    text = '''\n
.. contents:: {}
   :depth: {}
   :local:\n
'''.format(title, depth)
    return text


def generate_plugin_documentation(source_dir, outdir, ignore_paths):
    # pylint: disable=unused-argument
    pluginloader.clear()
    pluginloader.update(packages=GENERATE_FOR_PACKAGES)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for ext_type in pluginloader.kinds:
        outfile = os.path.join(outdir, '{}s.rst'.format(ext_type))
        with open(outfile, 'w') as wfh:
            wfh.write('.. _{}s:\n\n'.format(ext_type.replace('_', '-')))
            title = ' '.join([capitalize(w) for w in ext_type.split('_')])
            wfh.write(underline('{}s'.format(title)))
            wfh.write(insert_contents_table())
            wfh.write(line_break())
            exts = pluginloader.list_plugins(ext_type)
            sorted_exts = iter(sorted(exts, key=lambda x: x.name))
            try:
                wfh.write(get_rst_from_plugin(next(sorted_exts)))
            except StopIteration:
                return
            for ext in sorted_exts:
                wfh.write(line_break())
                wfh.write(get_rst_from_plugin(ext))


def generate_target_documentation(outdir):
    targets_to_generate = ['generic_android',
                           'generic_linux',
                           'generic_chromeos',
                           'generic_local',
                           'juno_linux',
                           'juno_android']

    intro = (
        '\nThis is a list of commonly used targets and their device '
        'parameters, to see a complete for a complete reference please use the'
        ' WA :ref:`list command <list-command>`.\n\n\n'
    )

    pluginloader.clear()
    pluginloader.update(packages=['wa.framework.target.descriptor'])

    target_descriptors = list_target_descriptions(pluginloader)
    outfile = os.path.join(outdir, 'targets.rst')
    with open(outfile, 'w') as wfh:
        wfh.write(underline('Common Targets'))
        wfh.write(intro)
        for td in sorted(target_descriptors, key=lambda t: t.name):
            if td.name not in targets_to_generate:
                continue
            text = underline(td.name, '~')
            if hasattr(td, 'description'):
                desc = strip_inlined_text(td.description or '')
                text += desc
            text += underline('Device Parameters:', '-')
            text += get_params_rst(td.conn_params)
            text += get_params_rst(td.platform_params)
            text += get_params_rst(td.target_params)
            text += get_params_rst(td.assistant_params)
            wfh.write(text)


def generate_run_config_documentation(outdir):
    generate_config_documentation(RunConfiguration, outdir)


def generate_meta_config_documentation(outdir):
    generate_config_documentation(MetaConfiguration, outdir)


def generate_config_documentation(config, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    config_name = '_'.join(config.name.split())
    outfile = os.path.join(outdir, '{}.rst'.format(config_name))
    with open(outfile, 'w') as wfh:
        wfh.write(get_params_rst(config.config_points))


if __name__ == '__main__':
    generate_plugin_documentation(sys.argv[2], sys.argv[1], sys.argv[3:])
