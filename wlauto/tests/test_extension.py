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


# pylint: disable=E0611,R0201,E1101
from unittest import TestCase

from nose.tools import assert_equal, raises, assert_true

from wlauto.core.extension import Extension, Parameter, Param, ExtensionMeta, Module
from wlauto.utils.types import list_of_ints
from wlauto.exceptions import ConfigError


class MyMeta(ExtensionMeta):

    virtual_methods = ['validate', 'virtual1', 'virtual2']


class MyBaseExtension(Extension):

    __metaclass__ = MyMeta

    name = 'base'

    parameters = [
        Parameter('base'),
    ]

    def __init__(self, **kwargs):
        super(MyBaseExtension, self).__init__(**kwargs)
        self.v1 = 0
        self.v2 = 0
        self.v3 = ''

    def virtual1(self):
        self.v1 += 1
        self.v3 = 'base'

    def virtual2(self):
        self.v2 += 1


class MyAcidExtension(MyBaseExtension):

    name = 'acid'

    parameters = [
        Parameter('hydrochloric', kind=list_of_ints, default=[1, 2]),
        'citric',
        ('carbonic', int),
    ]

    def __init__(self, **kwargs):
        super(MyAcidExtension, self).__init__(**kwargs)
        self.vv1 = 0
        self.vv2 = 0

    def virtual1(self):
        self.vv1 += 1
        self.v3 = 'acid'

    def virtual2(self):
        self.vv2 += 1


class MyOtherExtension(MyBaseExtension):

    name = 'other'

    parameters = [
        Param('mandatory', mandatory=True),
        Param('optional', allowed_values=['test', 'check']),
    ]

class MyOtherOtherExtension(MyOtherExtension):

    name = 'otherother'

    parameters = [
        Param('mandatory', override=True),
    ]


class MyOverridingExtension(MyAcidExtension):

    name = 'overriding'

    parameters = [
        Parameter('hydrochloric', override=True, default=[3, 4]),
    ]


class MyThirdTeerExtension(MyOverridingExtension):

    name = 'thirdteer'


class MultiValueParamExt(Extension):

    name = 'multivalue'

    parameters = [
        Parameter('test', kind=list_of_ints, allowed_values=[42, 7, 73]),
    ]


class MyCoolModule(Module):

    name = 'cool_module'

    capabilities = ['fizzle']

    def initialize(self, context):
        self.fizzle_factor = 0  # pylint: disable=attribute-defined-outside-init

    def fizzle(self):
        self.fizzle_factor += 1


class MyEvenCoolerModule(Module):

    name = 'even_cooler_module'

    capabilities = ['fizzle']

    def fizzle(self):
        self.owner.self_fizzle_factor += 2


class MyModularExtension(Extension):

    name = 'modular'

    parameters = [
        Parameter('modules', override=True, default=['cool_module']),
    ]


class MyOtherModularExtension(Extension):

    name = 'other_modular'

    parameters = [
        Parameter('modules', override=True, default=[
            'cool_module',
            'even_cooler_module',
        ]),
    ]

    def __init__(self, **kwargs):
        super(MyOtherModularExtension, self).__init__(**kwargs)
        self.self_fizzle_factor = 0


class FakeLoader(object):

    modules = [
        MyCoolModule,
        MyEvenCoolerModule,
    ]

    def get_module(self, name, owner, **kwargs):  # pylint: disable=unused-argument
        for module in self.modules:
            if module.name == name:
                return _instantiate(module, owner)


class ExtensionMetaTest(TestCase):

    def test_propagation(self):
        acid_params = [p.name for p in MyAcidExtension.parameters]
        assert_equal(acid_params, ['modules', 'base', 'hydrochloric', 'citric', 'carbonic'])

    @raises(ValueError)
    def test_duplicate_param_spec(self):
        class BadExtension(MyBaseExtension):  # pylint: disable=W0612
            parameters = [
                Parameter('base'),
            ]

    def test_param_override(self):
        class OverridingExtension(MyBaseExtension):  # pylint: disable=W0612
            parameters = [
                Parameter('base', override=True, default='cheese'),
            ]
        assert_equal(OverridingExtension.parameters['base'].default, 'cheese')

    @raises(ValueError)
    def test_invalid_param_spec(self):
        class BadExtension(MyBaseExtension):  # pylint: disable=W0612
            parameters = [
                7,
            ]

    def test_virtual_methods(self):
        acid = _instantiate(MyAcidExtension)
        acid.virtual1()
        assert_equal(acid.v1, 1)
        assert_equal(acid.vv1, 1)
        assert_equal(acid.v2, 0)
        assert_equal(acid.vv2, 0)
        assert_equal(acid.v3, 'acid')
        acid.virtual2()
        acid.virtual2()
        assert_equal(acid.v1, 1)
        assert_equal(acid.vv1, 1)
        assert_equal(acid.v2, 2)
        assert_equal(acid.vv2, 2)

    def test_initialization(self):
        class MyExt(Extension):
            name = 'myext'
            values = {'a': 0}
            def __init__(self, *args, **kwargs):
                super(MyExt, self).__init__(*args, **kwargs)
                self.instance_init = 0
            def initialize(self, context):
                self.values['a'] += 1

        class MyChildExt(MyExt):
            name = 'mychildext'
            def initialize(self, context):
                self.instance_init += 1

        ext = _instantiate(MyChildExt)
        ext.initialize(None)

        assert_equal(MyExt.values['a'], 1)
        assert_equal(ext.instance_init, 1)

    def test_initialization_happens_once(self):
        class MyExt(Extension):
            name = 'myext'
            values = {'a': 0}
            def __init__(self, *args, **kwargs):
                super(MyExt, self).__init__(*args, **kwargs)
                self.instance_init = 0
                self.instance_validate = 0
            def initialize(self, context):
                self.values['a'] += 1
            def validate(self):
                self.instance_validate += 1

        class MyChildExt(MyExt):
            name = 'mychildext'
            def initialize(self, context):
                self.instance_init += 1
            def validate(self):
                self.instance_validate += 1

        ext1 = _instantiate(MyExt)
        ext2 = _instantiate(MyExt)
        ext3 = _instantiate(MyChildExt)
        ext1.initialize(None)
        ext2.initialize(None)
        ext3.initialize(None)
        ext1.validate()
        ext2.validate()
        ext3.validate()

        assert_equal(MyExt.values['a'], 1)
        assert_equal(ext1.instance_init, 0)
        assert_equal(ext3.instance_init, 1)
        assert_equal(ext1.instance_validate, 1)
        assert_equal(ext3.instance_validate, 2)


class ParametersTest(TestCase):

    def test_setting(self):
        myext = _instantiate(MyAcidExtension, hydrochloric=[5, 6], citric=5, carbonic=42)
        assert_equal(myext.hydrochloric, [5, 6])
        assert_equal(myext.citric, '5')
        assert_equal(myext.carbonic, 42)

    def test_validation_ok(self):
        myext = _instantiate(MyOtherExtension, mandatory='check', optional='check')
        myext.validate()

    def test_default_override(self):
        myext = _instantiate(MyOverridingExtension)
        assert_equal(myext.hydrochloric, [3, 4])
        myotherext = _instantiate(MyThirdTeerExtension)
        assert_equal(myotherext.hydrochloric, [3, 4])

    def test_multivalue_param(self):
        myext = _instantiate(MultiValueParamExt, test=[7, 42])
        myext.validate()
        assert_equal(myext.test, [7, 42])

    @raises(ConfigError)
    def test_bad_multivalue_param(self):
        myext = _instantiate(MultiValueParamExt, test=[5])
        myext.validate()

    @raises(ConfigError)
    def test_validation_no_mandatory(self):
        myext = _instantiate(MyOtherExtension, optional='check')
        myext.validate()

    @raises(ConfigError)
    def test_validation_no_mandatory_in_derived(self):
        _instantiate(MyOtherOtherExtension)

    @raises(ConfigError)
    def test_validation_bad_value(self):
        myext = _instantiate(MyOtherExtension, mandatory=1, optional='invalid')
        myext.validate()

    @raises(ValueError)
    def test_duplicate_param_override(self):
        class DuplicateParamExtension(MyBaseExtension):  # pylint: disable=W0612
            parameters = [
                Parameter('base', override=True, default='buttery'),
                Parameter('base', override=True, default='biscuit'),
            ]

    @raises(ValueError)
    def test_overriding_new_param(self):
        class DuplicateParamExtension(MyBaseExtension):  # pylint: disable=W0612
            parameters = [
                Parameter('food', override=True, default='cheese'),
            ]

class ModuleTest(TestCase):

    def test_fizzle(self):
        myext = _instantiate(MyModularExtension)
        myext.load_modules(FakeLoader())
        assert_true(myext.can('fizzle'))
        myext.fizzle()
        assert_equal(myext.fizzle_factor, 1)

    def test_self_fizzle(self):
        myext = _instantiate(MyOtherModularExtension)
        myext.load_modules(FakeLoader())
        myext.fizzle()
        assert_equal(myext.self_fizzle_factor, 2)


def _instantiate(cls, *args, **kwargs):
    # Needed to get around Extension's __init__ checks
    return cls(*args, **kwargs)
