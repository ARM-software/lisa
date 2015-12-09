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


# pylint: disable=E0611,R0201
import os
import tempfile
from unittest import TestCase

from nose.tools import assert_equal, assert_in, raises

from wlauto.core.bootstrap import ConfigLoader
from wlauto.core.agenda import AgendaWorkloadEntry, AgendaGlobalEntry, Agenda
from wlauto.core.configuration import RunConfiguration
from wlauto.exceptions import ConfigError


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

BAD_CONFIG_TEXT = """device = 'TEST
device_config = 'TEST-CONFIG'"""


LIST_PARAMS_AGENDA_TEXT = """
config:
    instrumentation: [list_params]
    list_params:
        param: [0.1, 0.1, 0.1]
workloads:
    - dhrystone
"""

INSTRUMENTATION_AGENDA_TEXT = """
config:
    instrumentation: [execution_time]
workloads:
    - dhrystone
    - name: angrybirds
      instrumentation: [fsp]
"""


class MockExtensionLoader(object):

    def __init__(self):
        self.aliases = {}
        self.global_param_aliases = {}
        self.extensions = {
            'defaults_workload': DefaultsWorkload(),
            'list_params': ListParamstrument(),
        }

    def get_extension_class(self, name, kind=None):  # pylint: disable=unused-argument
        return self.extensions.get(name, NamedMock(name))

    def resolve_alias(self, name):
        return name, {}

    def get_default_config(self, name):  # pylint: disable=unused-argument
        ec = self.get_extension_class(name)
        return {p.name: p.default for p in ec.parameters}

    def has_extension(self, name):
        return name in self.aliases or name in self.extensions


class MockAgenda(object):

    def __init__(self, *args):
        self.config = {}
        self.global_ = AgendaGlobalEntry()
        self.sections = []
        self.workloads = args


class NamedMock(object):

    def __init__(self, name):
        self.__attrs = {
            'global_alias': None
        }
        self.name = name
        self.parameters = []

    def __getattr__(self, name):
        if name not in self.__attrs:
            self.__attrs[name] = NamedMock(name)
        return self.__attrs[name]


class DefaultsWorkload(object):

    def __init__(self):
        self.name = 'defaults_workload'
        self.parameters = [NamedMock('param')]
        self.parameters[0].default = [1, 2]


class ListParamstrument(object):

    def __init__(self):
        self.name = 'list_params'
        self.parameters = [NamedMock('param')]
        self.parameters[0].default = []


class ConfigLoaderTest(TestCase):

    def setUp(self):
        self.filepath = tempfile.mktemp()
        with open(self.filepath, 'w') as wfh:
            wfh.write(BAD_CONFIG_TEXT)

    def test_load(self):
        test_cfg_file = os.path.join(DATA_DIR, 'test-config.py')
        config = ConfigLoader()
        config.update(test_cfg_file)
        assert_equal(config.device, 'TEST')

    @raises(ConfigError)
    def test_load_bad(self):
        config_loader = ConfigLoader()
        config_loader.update(self.filepath)

    def test_load_duplicate(self):
        config_loader = ConfigLoader()
        config_loader.update(dict(instrumentation=['test']))
        config_loader.update(dict(instrumentation=['test']))
        assert_equal(config_loader.instrumentation, ['test'])

    def tearDown(self):
        os.unlink(self.filepath)


class ConfigTest(TestCase):

    def setUp(self):
        self.config = RunConfiguration(MockExtensionLoader())
        self.config.load_config({'device': 'MockDevice'})

    def test_case(self):
        devparams = {
            'sysfile_values': {
                '/sys/test/MyFile': 1,
                '/sys/test/other file': 2,
            }
        }
        ws = AgendaWorkloadEntry(id='a', iterations=1, name='linpack', runtime_parameters=devparams)
        self.config.set_agenda(MockAgenda(ws))
        spec = self.config.workload_specs[0]
        assert_in('/sys/test/MyFile', spec.runtime_parameters['sysfile_values'])
        assert_in('/sys/test/other file', spec.runtime_parameters['sysfile_values'])

    def test_list_defaults_params(self):
        ws = AgendaWorkloadEntry(id='a', iterations=1,
                                 name='defaults_workload', workload_parameters={'param': [3]})
        self.config.set_agenda(MockAgenda(ws))
        spec = self.config.workload_specs[0]
        assert_equal(spec.workload_parameters, {'param': [3]})

    def test_exetension_params_lists(self):
        a = Agenda(LIST_PARAMS_AGENDA_TEXT)
        self.config.set_agenda(a)
        self.config.finalize()
        assert_equal(self.config.instrumentation['list_params']['param'], [0.1, 0.1, 0.1])

    def test_instrumentation_specification(self):
        a = Agenda(INSTRUMENTATION_AGENDA_TEXT)
        self.config.set_agenda(a)
        self.config.finalize()
        assert_equal(self.config.workload_specs[0].instrumentation, ['execution_time'])
        assert_equal(self.config.workload_specs[1].instrumentation, ['fsp', 'execution_time'])

    def test_remove_instrument(self):
        self.config.load_config({'instrumentation': ['list_params']})
        a = Agenda('{config: {instrumentation: [~list_params] }}')
        self.config.set_agenda(a)
        self.config.finalize()
        assert_equal(self.config.instrumentation, {})

    def test_global_instrumentation(self):
        self.config.load_config({'instrumentation': ['global_instrument']})
        ws = AgendaWorkloadEntry(id='a', iterations=1, name='linpack', instrumentation=['local_instrument'])
        self.config.set_agenda(MockAgenda(ws))
        self.config.finalize()
        assert_equal(self.config.workload_specs[0].instrumentation,
                     ['local_instrument', 'global_instrument'])
