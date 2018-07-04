#    Copyright 2014-2017 ARM Limited
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


# pylint: disable=E0611,R0201,E1101
import os
from unittest import TestCase

from nose.tools import assert_equal, raises, assert_true

from wa.framework.plugin import Plugin, PluginMeta, PluginLoader, Parameter
from wa.utils.types import list_of_ints
from wa import ConfigError


EXTDIR = os.path.join(os.path.dirname(__file__), 'data', 'extensions')


class PluginLoaderTest(TestCase):

    def setUp(self):
        self.loader = PluginLoader(paths=[EXTDIR, ])

    def test_load_device(self):
        device = self.loader.get_device('test-device')
        assert_equal(device.name, 'test-device')

    def test_list_by_kind(self):
        exts = self.loader.list_devices()
        assert_equal(len(exts), 1)
        assert_equal(exts[0].name, 'test-device')



class MyBasePlugin(Plugin):

    name = 'base'
    kind = 'test'

    parameters = [
        Parameter('base'),
    ]

    def __init__(self, **kwargs):
        super(MyBasePlugin, self).__init__(**kwargs)
        self.v1 = 0
        self.v2 = 0
        self.v3 = ''

    def virtual1(self):
        self.v1 += 1
        self.v3 = 'base'

    def virtual2(self):
        self.v2 += 1


class MyAcidPlugin(MyBasePlugin):

    name = 'acid'

    parameters = [
        Parameter('hydrochloric', kind=list_of_ints, default=[1, 2]),
        Parameter('citric'),
        Parameter('carbonic', kind=int),
    ]

    def __init__(self, **kwargs):
        super(MyAcidPlugin, self).__init__(**kwargs)
        self.vv1 = 0
        self.vv2 = 0

    def virtual1(self):
        self.vv1 += 1
        self.v3 = 'acid'

    def virtual2(self):
        self.vv2 += 1


class MyOtherPlugin(MyBasePlugin):

    name = 'other'

    parameters = [
        Parameter('mandatory', mandatory=True),
        Parameter('optional', allowed_values=['test', 'check']),
    ]

class MyOtherOtherPlugin(MyOtherPlugin):

    name = 'otherother'

    parameters = [
        Parameter('mandatory', override=True),
    ]


class MyOverridingPlugin(MyAcidPlugin):

    name = 'overriding'

    parameters = [
        Parameter('hydrochloric', override=True, default=[3, 4]),
    ]


class MyThirdTeerPlugin(MyOverridingPlugin):

    name = 'thirdteer'


class MultiValueParamExt(Plugin):

    name = 'multivalue'
    kind = 'test'

    parameters = [
        Parameter('test', kind=list_of_ints, allowed_values=[42, 7, 73]),
    ]


class PluginMetaTest(TestCase):

    def test_propagation(self):
        acid_params = [p.name for p in MyAcidPlugin.parameters]
        assert_equal(acid_params, ['base', 'hydrochloric', 'citric', 'carbonic'])

    @raises(ValueError)
    def test_duplicate_param_spec(self):
        class BadPlugin(MyBasePlugin):  # pylint: disable=W0612
            parameters = [
                Parameter('base'),
            ]

    def test_param_override(self):
        class OverridingPlugin(MyBasePlugin):  # pylint: disable=W0612
            parameters = [
                Parameter('base', override=True, default='cheese'),
            ]
        assert_equal(OverridingPlugin.parameters['base'].default, 'cheese')

    @raises(ValueError)
    def test_invalid_param_spec(self):
        class BadPlugin(MyBasePlugin):  # pylint: disable=W0612
            parameters = [
                7,
            ]


class ParametersTest(TestCase):

    def test_setting(self):
        myext = MyAcidPlugin(hydrochloric=[5, 6], citric=5, carbonic=42)
        assert_equal(myext.hydrochloric, [5, 6])
        assert_equal(myext.citric, '5')
        assert_equal(myext.carbonic, 42)

    def test_validation_ok(self):
        myext = MyOtherPlugin(mandatory='check', optional='check')
        myext.validate()

    def test_default_override(self):
        myext = MyOverridingPlugin()
        assert_equal(myext.hydrochloric, [3, 4])
        myotherext = MyThirdTeerPlugin()
        assert_equal(myotherext.hydrochloric, [3, 4])

    def test_multivalue_param(self):
        myext = MultiValueParamExt(test=[7, 42])
        myext.validate()
        assert_equal(myext.test, [7, 42])

    @raises(ConfigError)
    def test_bad_multivalue_param(self):
        myext = MultiValueParamExt(test=[5])
        myext.validate()

    @raises(ConfigError)
    def test_validation_no_mandatory(self):
        myext = MyOtherPlugin(optional='check')
        myext.validate()

    @raises(ConfigError)
    def test_validation_no_mandatory_in_derived(self):
        MyOtherOtherPlugin()

    @raises(ConfigError)
    def test_validation_bad_value(self):
        myext = MyOtherPlugin(mandatory=1, optional='invalid')
        myext.validate()

