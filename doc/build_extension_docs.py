#!/usr/bin/env python
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


import os
import sys

from wa import pluginloader
from wa.framework.configuration.core import RunConfiguration, MetaConfiguration
from wa.utils.doc import get_rst_from_plugin, underline, get_params_rst
from wa.utils.misc import capitalize


GENERATE_FOR_PLUGIN = ['workload', 'instrument', 'output_processor', 'target']

def generate_plugin_documentation(source_dir, outdir, ignore_paths):
    pluginloader.clear()
    pluginloader.update(paths=[source_dir], ignore_paths=ignore_paths)
    for ext_type in pluginloader.kinds:
        if not ext_type in GENERATE_FOR_PLUGIN:
            continue
        outfile = os.path.join(outdir, '{}s.rst'.format(ext_type))
        with open(outfile, 'w') as wfh:
            wfh.write('.. _{}s:\n\n'.format(ext_type))
            title = ' '.join([capitalize(w) for w in ext_type.split('_')])
            wfh.write(underline('{}s'.format(title)))
            exts = pluginloader.list_plugins(ext_type)
            for ext in sorted(exts, key=lambda x: x.name):
                wfh.write(get_rst_from_plugin(ext))


def generate_run_config_documentation(outdir):
    generate_config_documentation(RunConfiguration, outdir)


def generate_meta_config_documentation(outdir):
    generate_config_documentation(MetaConfiguration, outdir)


def generate_config_documentation(config, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outfile = os.path.join(outdir, '{}.rst'.format('_'.join(config.name.split())))
    with open(outfile, 'w') as wfh:
        wfh.write(get_params_rst(config.config_points))



if __name__ == '__main__':
    generate_plugin_documentation(sys.argv[2], sys.argv[1], sys.argv[3:])
