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


from wlauto import Device


class TestDevice(Device):

    name = 'test-device'

    def __init__(self, *args, **kwargs):
        self.modules = []
        self.boot_called = 0
        self.push_file_called = 0
        self.pull_file_called = 0
        self.execute_called = 0
        self.set_sysfile_int_called = 0
        self.close_called = 0

    def boot(self):
        self.boot_called += 1

    def push_file(self, source, dest):
        self.push_file_called += 1

    def pull_file(self, source, dest):
        self.pull_file_called += 1

    def execute(self, command):
        self.execute_called += 1

    def set_sysfile_int(self, file, value):
        self.set_sysfile_int_called += 1

    def close(self, command):
        self.close_called += 1
