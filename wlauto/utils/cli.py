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

from wlauto.core.version import get_wa_version


def init_argument_parser(parser):
    parser.add_argument('-c', '--config', help='specify an additional config.py')
    parser.add_argument('-v', '--verbose', action='count',
                        help='The scripts will produce verbose output.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode. Note: this implies --verbose.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(get_wa_version()))
    return parser

