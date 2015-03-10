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


class Cpuinfo(object):

    @property
    def architecture(self):
        for section in self.sections:
            if 'CPU architecture' in section:
                return section['CPU architecture']
            if 'architecture' in section:
                return section['architecture']

    def __init__(self, text):
        self.sections = None
        self.text = None
        self.parse(text)

    def parse(self, text):
        self.sections = []
        current_section = {}
        self.text = text.strip()
        for line in self.text.split('\n'):
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                current_section[key.strip()] = value.strip()
            else:  # not line
                self.sections.append(current_section)
                current_section = {}
        self.sections.append(current_section)
