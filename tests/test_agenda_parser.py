#    Copyright 2013-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.
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
import yaml
from collections import defaultdict
from unittest import TestCase

from nose.tools import assert_equal, assert_in, raises, assert_true

from wa.framework.configuration.execution import ConfigManager
from wa.framework.configuration.parsers import AgendaParser
from wa.framework.exception import ConfigError
from wa.utils.types import reset_all_counters

YAML_TEST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'test-agenda.yaml')
YAML_BAD_SYNTAX_FILE = os.path.join(os.path.dirname(__file__), 'data', 'bad-syntax-agenda.yaml')

invalid_agenda_text = """
workloads:
    - id: 1
      workload_parameters:
          test: 1
"""
invalid_agenda = yaml.load(invalid_agenda_text)
invalid_agenda.name = 'invalid1'

duplicate_agenda_text = """
global:
    iterations: 1
workloads:
    - id: 1
      workload_name: antutu
      workload_parameters:
          test: 1
    - id: "1"
      workload_name: benchmarkpi
"""
duplicate_agenda = yaml.load(duplicate_agenda_text)
duplicate_agenda.name = 'invalid2'

short_agenda_text = """
workloads: [antutu, dhrystone, benchmarkpi]
"""
short_agenda = yaml.load(short_agenda_text)
short_agenda.name = 'short'

default_ids_agenda_text = """
workloads:
    - antutu
    - id: wk1
      name: benchmarkpi
    - id: test
      name: dhrystone
      params:
          cpus: 1
    - vellamo
"""
default_ids_agenda = yaml.load(default_ids_agenda_text)
default_ids_agenda.name = 'default_ids'

sectioned_agenda_text = """
sections:
    - id: sec1
      runtime_params:
        dp: one
      workloads:
        - name: antutu
          workload_parameters:
            markers_enabled: True
        - benchmarkpi
        - name: dhrystone
          runtime_params:
            dp: two
    - id: sec2
      runtime_params:
        dp: three
      workloads:
        - antutu
workloads:
    - memcpy
"""
sectioned_agenda = yaml.load(sectioned_agenda_text)
sectioned_agenda.name = 'sectioned'

dup_sectioned_agenda_text = """
sections:
    - id: sec1
      workloads:
        - antutu
    - id: sec1
      workloads:
        - benchmarkpi
workloads:
    - memcpy
"""
dup_sectioned_agenda = yaml.load(dup_sectioned_agenda_text)
dup_sectioned_agenda.name = 'dup-sectioned'


class AgendaTest(TestCase):

    def setUp(self):
        reset_all_counters()
        self.config = ConfigManager()
        self.parser = AgendaParser()

    def test_yaml_load(self):
        self.parser.load_from_path(self.config, YAML_TEST_FILE)
        assert_equal(len(self.config.jobs_config.root_node.workload_entries), 4)

    def test_duplicate_id(self):
        try:
            self.parser.load(self.config, duplicate_agenda, 'test')
        except ConfigError as e:
            assert_in('duplicate', e.message.lower())  # pylint: disable=E1101
        else:
            raise Exception('ConfigError was not raised for an agenda with duplicate ids.')

    def test_yaml_missing_field(self):
        try:
            self.parser.load(self.config, invalid_agenda, 'test')
        except ConfigError as e:
            assert_in('workload name', e.message)
        else:
            raise Exception('ConfigError was not raised for an invalid agenda.')

    def test_defaults(self):
        self.parser.load(self.config, short_agenda, 'test')
        workload_entries = self.config.jobs_config.root_node.workload_entries
        assert_equal(len(workload_entries), 3)
        assert_equal(workload_entries[0].config['workload_name'], 'antutu')
        assert_equal(workload_entries[0].id, 'wk1')

    def test_default_id_assignment(self):
        self.parser.load(self.config, default_ids_agenda, 'test2')
        workload_entries = self.config.jobs_config.root_node.workload_entries
        assert_equal(workload_entries[0].id, 'wk2')
        assert_equal(workload_entries[3].id, 'wk3')

    def test_sections(self):
        self.parser.load(self.config, sectioned_agenda, 'test')
        root_node_workload_entries = self.config.jobs_config.root_node.workload_entries
        leaves = list(self.config.jobs_config.root_node.leaves())
        section1_workload_entries = leaves[0].workload_entries
        section2_workload_entries = leaves[0].workload_entries

        assert_equal(root_node_workload_entries[0].config['workload_name'], 'memcpy')
        assert_true(section1_workload_entries[0].config['workload_parameters']['markers_enabled'])
        assert_equal(section2_workload_entries[0].config['workload_name'], 'antutu')

    @raises(ConfigError)
    def test_dup_sections(self):
        self.parser.load(self.config, dup_sectioned_agenda, 'test')

    @raises(ConfigError)
    def test_bad_syntax(self):
        self.parser.load_from_path(self.config, YAML_BAD_SYNTAX_FILE)
