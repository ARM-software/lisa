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


from wlauto import LinuxDevice, Parameter


class Xe503c12Chormebook(LinuxDevice):

    name = "XE503C12"
    description = 'A developer-unlocked Samsung XE503C12 running sshd.'
    platform = 'chromeos'

    parameters = [
        Parameter('core_names', default=['a15', 'a15', 'a15', 'a15'], override=True),
        Parameter('core_clusters', default=[0, 0, 0, 0], override=True),
        Parameter('username', default='chronos', override=True),
        Parameter('password', default='', override=True),
        Parameter('password_prompt', default='Password:', override=True),
        Parameter('binaries_directory', default='/home/chronos/bin', override=True),
    ]

    abi = 'armeabi'

