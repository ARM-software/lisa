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


class GenericDevice(AndroidDevice):
    name = 'generic_android'
    description = """
    Generic Android device. Use this if you do not have a device file for
    your device.

    This implements the minimum functionality that should be supported by
    all android devices.

    """

    default_working_directory = '/storage/sdcard0/working'
    has_gpu = True

    parameters = [
        Parameter('core_names', default=[], override=True),
        Parameter('core_clusters', default=[], override=True),
    ]
