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


from wlauto import ExtensionLoader, Command, settings
from wlauto.utils.formatter import DescriptionListFormatter
from wlauto.utils.doc import get_summary


class ListCommand(Command):

    name = 'list'
    description = 'List available WA extensions with a short description of each.'

    def initialize(self):
        extension_types = ['{}s'.format(ext.name) for ext in settings.extensions]
        self.parser.add_argument('kind', metavar='KIND',
                                 help=('Specify the kind of extension to list. Must be '
                                       'one of: {}'.format(', '.join(extension_types))),
                                 choices=extension_types)
        self.parser.add_argument('-n', '--name', help='Filter results by the name specified')

    def execute(self, args):
        filters = {}
        if args.name:
            filters['name'] = args.name

        ext_loader = ExtensionLoader(packages=settings.extension_packages, paths=settings.extension_paths)
        results = ext_loader.list_extensions(args.kind[:-1])
        if filters:
            filtered_results = []
            for result in results:
                passed = True
                for k, v in filters.iteritems():
                    if getattr(result, k) != v:
                        passed = False
                        break
                if passed:
                    filtered_results.append(result)
        else:  # no filters specified
            filtered_results = results

        if filtered_results:
            output = DescriptionListFormatter()
            for result in sorted(filtered_results, key=lambda x: x.name):
                output.add_item(get_summary(result), result.name)
            print output.format_data()
