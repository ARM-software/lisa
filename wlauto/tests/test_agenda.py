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


# pylint: disable=E0611
# pylint: disable=R0201
import os
from StringIO import StringIO
from unittest import TestCase

from nose.tools import assert_equal, assert_in, raises

from wlauto.core.agenda import Agenda
from wlauto.exceptions import ConfigError
from wlauto.utils.serializer import SerializerSyntaxError


YAML_TEST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'test-agenda.yaml')

invalid_agenda_text = """
workloads:
    - id: 1
      workload_parameters:
          test: 1
"""
invalid_agenda = StringIO(invalid_agenda_text)
invalid_agenda.name = 'invalid1.yaml'

duplicate_agenda_text = """
global:
    iterations: 1
workloads:
    - id: 1
      workload_name: antutu
      workload_parameters:
          test: 1
    - id: 1
      workload_name: andebench
"""
duplicate_agenda = StringIO(duplicate_agenda_text)
duplicate_agenda.name = 'invalid2.yaml'

short_agenda_text = """
workloads: [antutu, linpack, andebench]
"""
short_agenda = StringIO(short_agenda_text)
short_agenda.name = 'short.yaml'

default_ids_agenda_text = """
workloads:
    - antutu
    - id: 1
      name: linpack
    - id: test
      name: andebench
      params:
          number_of_threads: 1
    - vellamo
"""
default_ids_agenda = StringIO(default_ids_agenda_text)
default_ids_agenda.name = 'default_ids.yaml'

sectioned_agenda_text = """
sections:
    - id: sec1
      runtime_params:
        dp: one
      workloads:
        - antutu
        - andebench
        - name: linpack
          runtime_params:
            dp: two
    - id: sec2
      runtime_params:
        dp: three
      workloads:
        - antutu
workloads:
    - nenamark
"""
sectioned_agenda = StringIO(sectioned_agenda_text)
sectioned_agenda.name = 'sectioned.yaml'

dup_sectioned_agenda_text = """
sections:
    - id: sec1
      workloads:
        - antutu
    - id: sec1
      workloads:
        - andebench
workloads:
    - nenamark
"""
dup_sectioned_agenda = StringIO(dup_sectioned_agenda_text)
dup_sectioned_agenda.name = 'dup-sectioned.yaml'

caps_agenda_text = """
config:
    device: TC2
global:
    runtime_parameters:
        sysfile_values:
            /sys/test/MyFile: 1
            /sys/test/other file: 2
workloads:
    - id: 1
      name: linpack
"""
caps_agenda = StringIO(caps_agenda_text)
caps_agenda.name = 'caps.yaml'

bad_syntax_agenda_text = """
config:
    # tab on the following line
    reboot_policy: never
workloads:
    - antutu
"""
bad_syntax_agenda = StringIO(bad_syntax_agenda_text)
bad_syntax_agenda.name = 'bad_syntax.yaml'

section_ids_test_text = """
config:
    device: TC2
    reboot_policy: never
workloads:
    - name: bbench
      id: bbench
    - name: audio
sections:
    - id: foo
    - id: bar
"""
section_ids_agenda = StringIO(section_ids_test_text)
section_ids_agenda.name = 'section_ids.yaml'


class AgendaTest(TestCase):

    def test_yaml_load(self):
        agenda = Agenda(YAML_TEST_FILE)
        assert_equal(len(agenda.workloads), 4)

    def test_yaml_missing_field(self):
        try:
            Agenda(invalid_agenda)
        except ConfigError, e:
            assert_in('workload name', e.message)
        else:
            raise Exception('ConfigError was not raised for an invalid agenda.')

    @raises(ConfigError)
    def test_dup_sections(self):
        Agenda(dup_sectioned_agenda)

    @raises(SerializerSyntaxError)
    def test_bad_syntax(self):
        Agenda(bad_syntax_agenda)
