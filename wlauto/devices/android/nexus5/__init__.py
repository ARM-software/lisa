#    Copyright 2013-2015 ARM Limited
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


from wlauto import AndroidDevice, Parameter


class Nexus5Device(AndroidDevice):

    name = 'Nexus5'
    description = """
    Adapter for Nexus 5.

    To be able to use Nexus5 in WA, the following must be true:

        - USB Debugging Mode is enabled.
        - Generate USB debugging authorisation for the host machine

    """

    default_working_directory = '/storage/sdcard0/working'
    has_gpu = True
    max_cores = 4

    parameters = [
        Parameter('core_names', default=['krait400', 'krait400', 'krait400', 'krait400'], override=True),
        Parameter('core_clusters', default=[0, 0, 0, 0], override=True),
    ]
