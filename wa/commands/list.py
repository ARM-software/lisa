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

from wa import Command
from wa.framework import pluginloader
from wa.framework.target.descriptor import list_target_descriptions
from wa.utils.doc import get_summary
from wa.utils.formatter import DescriptionListFormatter


class ListCommand(Command):

    name = 'list'
    description = 'List available WA plugins with a short description of each.'

    def initialize(self, context):
        kinds = get_kinds()
        self.parser.add_argument('kind', metavar='KIND',
                                 help=('Specify the kind of plugin to list. Must be '
                                       'one of: {}'.format(', '.join(kinds))),
                                 choices=kinds)
        self.parser.add_argument('-n', '--name', 
                                 help='Filter results by the name specified')
        self.parser.add_argument('-o', '--packaged-only', action='store_true',
                                 help='''
                                 Only list plugins packaged with WA itself. Do
                                 not list plugins installed locally or from
                                 other packages.
                                 ''')
        self.parser.add_argument('-p', '--platform', 
                                 help='''
                                 Only list results that are supported by the
                                 specified platform.
                                 ''')

    def execute(self, state, args):
        filters = {}
        if args.name:
            filters['name'] = args.name

        if args.kind == 'targets':
            list_targets()
        else:
            list_plugins(args, filters)


def get_kinds():
    kinds = pluginloader.kinds
    if 'target_descriptor' in kinds:
        kinds.remove('target_descriptor')
        kinds.append('target')
    return ['{}s'.format(name) for name in kinds]


def list_targets():
    targets = list_target_descriptions()
    targets = sorted(targets, key=lambda x: x.name)

    output = DescriptionListFormatter()
    for target in targets:
        output.add_item(target.description or '', target.name)
    print output.format_data()


def list_plugins(args, filters):
    results = pluginloader.list_plugins(args.kind[:-1])
    if filters or args.platform:
        filtered_results = []
        for result in results:
            passed = True
            for k, v in filters.iteritems():
                if getattr(result, k) != v:
                    passed = False
                    break
            if passed and args.platform:
                passed = check_platform(result, args.platform)
            if passed:
                filtered_results.append(result)
    else:  # no filters specified
        filtered_results = results

    if filtered_results:
        output = DescriptionListFormatter()
        for result in sorted(filtered_results, key=lambda x: x.name):
            output.add_item(get_summary(result), result.name)
        print output.format_data()


def check_platform(plugin, platform):
    supported_platforms = getattr(plugin, 'supported_platforms', [])
    if supported_platforms:
        return platform in supported_platforms
    return True
