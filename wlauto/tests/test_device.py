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


# pylint: disable=abstract-method,no-self-use,no-name-in-module
from collections import defaultdict, OrderedDict
from unittest import TestCase

from nose.tools import raises, assert_equal

from wlauto import Device, Parameter, RuntimeParameter, CoreParameter
from wlauto.exceptions import ConfigError


class TestDevice(Device):

    name = 'test-device'
    path_module = 'posixpath'

    parameters = [
        Parameter('core_names', default=['a7', 'a7', 'a15'], override=True),
        Parameter('core_clusters', default=[0, 0, 1], override=True),
    ]

    runtime_parameters = [
        RuntimeParameter('test_param', 'getter', 'setter'),
        RuntimeParameter('test_param2', 'getter', 'setter'),
        CoreParameter('${core}_param', 'core_getter', 'core_setter'),
    ]

    def __init__(self, *args, **kwargs):
        super(TestDevice, self).__init__(*args, **kwargs)
        self.value = None
        self.core_values = defaultdict()

    def getter(self):
        return self.value

    def setter(self, value):
        if self.value is None:
            self.value = value

    def core_getter(self, core):
        return self.core_values.get(core)

    def core_setter(self, core, value):
        self.core_values[core] = value


class RuntimeParametersTest(TestCase):

    def test_runtime_param(self):
        device = _instantiate(TestDevice)
        device.set_runtime_parameters(dict(test_param=5))
        assert_equal(device.value, 5)
        assert_equal(device.get_runtime_parameters().get('test_param'), 5)

    def test_core_param(self):
        device = _instantiate(TestDevice)
        device.set_runtime_parameters(dict(a15_param=1, a7_param=2))
        assert_equal(device.core_values, {'a15': 1, 'a7': 2})
        assert_equal(device.get_runtime_parameters().get('a15_param'), 1)
        assert_equal(device.get_runtime_parameters().get('a7_param'), 2)

    @raises(ConfigError)
    def test_bad_runtime_param(self):
        device = _instantiate(TestDevice)
        device.set_runtime_parameters(dict(bad_param=1))

    def test_get_unset_runtime_params(self):
        device = _instantiate(TestDevice)
        expected = {'test_param': None, 'test_param2': None, 'a15_param': None, 'a7_param': None}
        assert_equal(device.get_runtime_parameters(), expected)

    def test_param_set_order(self):
        device = _instantiate(TestDevice)
        device.set_runtime_parameters(OrderedDict([('test_param2', 1), ('test_param', 5)]))
        assert_equal(device.value, 1)
        device.value = None
        device.set_runtime_parameters(OrderedDict([('test_param', 5), ('test_param2', 1)]))
        assert_equal(device.value, 5)


def _instantiate(cls, *args, **kwargs):
    # Needed to get around Plugin's __init__ checks
    return cls(*args, **kwargs)

