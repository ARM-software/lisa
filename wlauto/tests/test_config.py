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
from wlauto.core.agenda import AgendaWorkloadEntry, AgendaGlobalEntry
from wlauto.core.configuration import RunConfiguration
from wlauto.exceptions import ConfigError


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

BAD_CONFIG_TEXT = """device = 'TEST
device_config = 'TEST-CONFIG'"""


class MockExtensionLoader(object):

    def __init__(self):
        self.aliases = {}
        self.global_param_aliases = {}
        self.extensions = {}

    def get_extension_class(self, name, kind=None):  # pylint: disable=unused-argument
        if name == 'defaults_workload':
            return DefaultsWorkload()
        else:
            return NamedMock(name)

    def resolve_alias(self, name):
        return name, {}

    def get_default_config(self, name):  # pylint: disable=unused-argument
        return {}

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
        if not name in self.__attrs:
            self.__attrs[name] = NamedMock(name)
        return self.__attrs[name]


class DefaultsWorkload(object):

    def __init__(self):
        self.name = 'defaults_workload'
        self.parameters = [NamedMock('param')]
        self.parameters[0].default = [1, 2]


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
                                 name='defaults_workload', workload_parameters={'param':[3]})
        self.config.set_agenda(MockAgenda(ws))
        spec = self.config.workload_specs[0]
        assert_equal(spec.workload_parameters, {'param': [3]})

    def test_global_instrumentation(self):
        self.config.load_config({'instrumentation': ['global_instrument']})
        ws = AgendaWorkloadEntry(id='a', iterations=1, name='linpack', instrumentation=['local_instrument'])
        self.config.set_agenda(MockAgenda(ws))
        self.config.finalize()
        assert_equal(self.config.workload_specs[0].instrumentation,
                     ['local_instrument', 'global_instrument'])
