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


# pylint: disable=W0231,W0613,E0611,W0603,R0201
from unittest import TestCase

from nose.tools import assert_equal, assert_raises

from wa.utils.exec_control import (init_environment, reset_environment,
                                   activate_environment, once,
                                   once_per_class, once_per_instance)

class TestClass(object):

    called = 0

    def __init__(self):
        self.count = 0

    @once
    def called_once(self):
        TestClass.called += 1

    @once
    def initilize_once(self):
        self.count += 1

    @once_per_class
    def initilize_once_per_class(self):
        self.count += 1

    @once_per_instance
    def initilize_once_per_instance(self):
        self.count += 1


class SubClass(TestClass):

    def __init__(self):
        super(SubClass, self).__init__()

    @once
    def initilize_once(self):
        super(SubClass, self).initilize_once()
        self.count += 1

    @once_per_class
    def initilize_once_per_class(self):
        super(SubClass, self).initilize_once_per_class()
        self.count += 1

    @once_per_instance
    def initilize_once_per_instance(self):
        super(SubClass, self).initilize_once_per_instance()
        self.count += 1


class SubSubClass(SubClass):

    def __init__(self):
        super(SubSubClass, self).__init__()

    @once
    def initilize_once(self):
        super(SubSubClass, self).initilize_once()
        self.count += 1

    @once_per_class
    def initilize_once_per_class(self):
        super(SubSubClass, self).initilize_once_per_class()
        self.count += 1

    @once_per_instance
    def initilize_once_per_instance(self):
        super(SubSubClass, self).initilize_once_per_instance()
        self.count += 1


class AnotherClass(object):

    def __init__(self):
        self.count = 0

    @once
    def initilize_once(self):
        self.count += 1

    @once_per_class
    def initilize_once_per_class(self):
        self.count += 1

    @once_per_instance
    def initilize_once_per_instance(self):
        self.count += 1


class AnotherSubClass(TestClass):

    def __init__(self):
        super(AnotherSubClass, self).__init__()

    @once
    def initilize_once(self):
        super(AnotherSubClass, self).initilize_once()
        self.count += 1

    @once_per_class
    def initilize_once_per_class(self):
        super(AnotherSubClass, self).initilize_once_per_class()
        self.count += 1

    @once_per_instance
    def initilize_once_per_instance(self):
        super(AnotherSubClass, self).initilize_once_per_instance()
        self.count += 1


class EnvironmentManagementTest(TestCase):

    def test_duplicate_environment(self):
        init_environment('ENVIRONMENT')
        assert_raises(ValueError, init_environment, 'ENVIRONMENT')

    def test_reset_missing_environment(self):
        assert_raises(ValueError, reset_environment, 'MISSING')

    def test_reset_current_environment(self):
        activate_environment('CURRENT_ENVIRONMENT')
        t1 = TestClass()
        t1.initilize_once()
        assert_equal(t1.count, 1)

        reset_environment()
        t1.initilize_once()
        assert_equal(t1.count, 2)

    def test_switch_environment(self):
        activate_environment('ENVIRONMENT1')
        t1 = TestClass()
        t1.initilize_once()
        assert_equal(t1.count, 1)

        activate_environment('ENVIRONMENT2')
        t1.initilize_once()
        assert_equal(t1.count, 2)

        activate_environment('ENVIRONMENT1')
        t1.initilize_once()
        assert_equal(t1.count, 2)

    def test_reset_environment_name(self):
        activate_environment('ENVIRONMENT')
        t1 = TestClass()
        t1.initilize_once()
        assert_equal(t1.count, 1)

        reset_environment('ENVIRONMENT')
        t1.initilize_once()
        assert_equal(t1.count, 2)


class ParentOnlyOnceEvironmentTest(TestCase):
    def test_sub_classes(self):
        sc = SubClass()
        asc = AnotherSubClass()

        sc.called_once()
        assert_equal(sc.called, 1)
        asc.called_once()
        assert_equal(asc.called, 1)


class OnlyOnceEnvironmentTest(TestCase):

    def setUp(self):
        activate_environment('TEST_ENVIRONMENT')

    def tearDown(self):
        reset_environment('TEST_ENVIRONMENT')

    def test_single_instance(self):
        t1 = TestClass()
        ac = AnotherClass()

        t1.initilize_once()
        assert_equal(t1.count, 1)

        t1.initilize_once()
        assert_equal(t1.count, 1)

        ac.initilize_once()
        assert_equal(ac.count, 1)


    def test_mulitple_instances(self):
        t1 = TestClass()
        t2 = TestClass()

        t1.initilize_once()
        assert_equal(t1.count, 1)

        t2.initilize_once()
        assert_equal(t2.count, 0)


    def test_sub_classes(self):
        t1 = TestClass()
        sc = SubClass()
        ss = SubSubClass()
        asc = AnotherSubClass()

        t1.initilize_once()
        assert_equal(t1.count, 1)

        sc.initilize_once()
        sc.initilize_once()
        assert_equal(sc.count, 1)

        ss.initilize_once()
        ss.initilize_once()
        assert_equal(ss.count, 1)

        asc.initilize_once()
        asc.initilize_once()
        assert_equal(asc.count, 1)


class OncePerClassEnvironmentTest(TestCase):

    def setUp(self):
        activate_environment('TEST_ENVIRONMENT')

    def tearDown(self):
        reset_environment('TEST_ENVIRONMENT')

    def test_single_instance(self):
        t1 = TestClass()
        ac = AnotherClass()

        t1.initilize_once_per_class()
        assert_equal(t1.count, 1)

        t1.initilize_once_per_class()
        assert_equal(t1.count, 1)

        ac.initilize_once_per_class()
        assert_equal(ac.count, 1)


    def test_mulitple_instances(self):
        t1 = TestClass()
        t2 = TestClass()

        t1.initilize_once_per_class()
        assert_equal(t1.count, 1)

        t2.initilize_once_per_class()
        assert_equal(t2.count, 0)


    def test_sub_classes(self):
        t1 = TestClass()
        sc1 = SubClass()
        sc2 = SubClass()
        ss1 = SubSubClass()
        ss2 = SubSubClass()
        asc = AnotherSubClass()

        t1.initilize_once_per_class()
        assert_equal(t1.count, 1)

        sc1.initilize_once_per_class()
        sc2.initilize_once_per_class()
        assert_equal(sc1.count, 1)
        assert_equal(sc2.count, 0)

        ss1.initilize_once_per_class()
        ss2.initilize_once_per_class()
        assert_equal(ss1.count, 1)
        assert_equal(ss2.count, 0)

        asc.initilize_once_per_class()
        assert_equal(asc.count, 1)


class OncePerInstanceEnvironmentTest(TestCase):

    def setUp(self):
        activate_environment('TEST_ENVIRONMENT')

    def tearDown(self):
        reset_environment('TEST_ENVIRONMENT')

    def test_single_instance(self):
        t1 = TestClass()
        ac = AnotherClass()

        t1.initilize_once_per_instance()
        assert_equal(t1.count, 1)

        t1.initilize_once_per_instance()
        assert_equal(t1.count, 1)

        ac.initilize_once_per_instance()
        assert_equal(ac.count, 1)


    def test_mulitple_instances(self):
        t1 = TestClass()
        t2 = TestClass()

        t1.initilize_once_per_instance()
        assert_equal(t1.count, 1)

        t2.initilize_once_per_instance()
        assert_equal(t2.count, 1)


    def test_sub_classes(self):
        t1 = TestClass()
        sc = SubClass()
        ss = SubSubClass()
        asc = AnotherSubClass()

        t1.initilize_once_per_instance()
        assert_equal(t1.count, 1)

        sc.initilize_once_per_instance()
        sc.initilize_once_per_instance()
        assert_equal(sc.count, 2)

        ss.initilize_once_per_instance()
        ss.initilize_once_per_instance()
        assert_equal(ss.count, 3)

        asc.initilize_once_per_instance()
        asc.initilize_once_per_instance()
        assert_equal(asc.count, 2)
