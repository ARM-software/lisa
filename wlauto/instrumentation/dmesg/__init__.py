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

from wlauto import Instrument, Parameter
from wlauto.utils.misc import ensure_file_directory_exists as _f


class DmesgInstrument(Instrument):
    # pylint: disable=no-member,attribute-defined-outside-init
    """
    Collected dmesg output before and during the run.

    """

    name = 'dmesg'

    parameters = [
        Parameter('loglevel', kind=int, allowed_values=range(8),
                  description='Set loglevel for console output.')
    ]

    loglevel_file = '/proc/sys/kernel/printk'

    def setup(self, context):
        if self.loglevel:
            self.old_loglevel = self.device.get_sysfile_value(self.loglevel_file)
            self.device.set_sysfile_value(self.loglevel_file, self.loglevel, verify=False)
        self.before_file = _f(os.path.join(context.output_directory, 'dmesg', 'before'))
        self.after_file = _f(os.path.join(context.output_directory, 'dmesg', 'after'))

    def slow_start(self, context):
        with open(self.before_file, 'w') as wfh:
            wfh.write(self.device.execute('dmesg'))
        context.add_artifact('dmesg_before', self.before_file, kind='data')
        if self.device.is_rooted:
            self.device.execute('dmesg -c', as_root=True)

    def slow_stop(self, context):
        with open(self.after_file, 'w') as wfh:
            wfh.write(self.device.execute('dmesg'))
        context.add_artifact('dmesg_after', self.after_file, kind='data')

    def teardown(self, context):  # pylint: disable=unused-argument
        if self.loglevel:
            self.device.set_sysfile_value(self.loglevel_file, self.old_loglevel, verify=False)


