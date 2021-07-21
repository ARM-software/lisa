# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os
import copy
from unittest import TestCase

import pytest

from lisa.conf import MultiSrcConf, KeyDesc, LevelKeyDesc, TopLevelKeyDesc, DerivedKeyDesc, DeferredValue
from lisa._generic import TypedList
from .utils import StorageTestCase, HOST_PLAT_INFO, HOST_TARGET_CONF

""" A test suite for the MultiSrcConf subclasses."""


class TestMultiSrcConfBase(StorageTestCase):
    """
    A test class that exercise various APIs of MultiSrcConf
    """
    __test__ = False

    def test_serialization(self):
        path = os.path.join(self.res_dir, "conf.serialized.yml")
        self.conf.to_path(path)
        self.conf.from_path(path)

    def test_conf_file(self):
        path = os.path.join(self.res_dir, "conf.yml")
        self.conf.to_yaml_map(path)
        self.conf.from_yaml_map(path)

    def test_add_src(self):
        updated_conf = copy.deepcopy(self.conf)
        # Add the same values in a new source. This is guaranteed to be valid
        updated_conf.add_src('foo', self.conf)
        assert dict(updated_conf) == dict(self.conf)

    def test_disallowed_key(self):
        with pytest.raises(KeyError):
            self.conf['this-key-does-not-exists-and-is-not-allowed']

    def test_copy(self):
        assert dict(self.conf) == dict(copy.copy(self.conf))

    def test_deepcopy(self):
        assert dict(self.conf) == dict(copy.deepcopy(self.conf))


class TestPlatformInfo(TestMultiSrcConfBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make copies to avoid mutating the original one
        self.conf = copy.copy(HOST_PLAT_INFO)


class TestTargetConf(TestMultiSrcConfBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make copies to avoid mutating the original one
        self.conf = copy.copy(HOST_TARGET_CONF)


def compute_derived(base_conf):
    return base_conf['foo'] + sum(base_conf['bar']) + base_conf['sublevel']['subkey']


INTERNAL_STRUCTURE = (
    KeyDesc('foo', 'foo help', [int]),
    KeyDesc('bar', 'bar help', [TypedList[int]]),
    KeyDesc('multitypes', 'multitypes help', [TypedList[int], str, None]),
    LevelKeyDesc('sublevel', 'sublevel help', (
        KeyDesc('subkey', 'subkey help', [int]),
    )),
    DerivedKeyDesc('derived', 'derived help', [int],
        [['foo'], ['bar'], ['sublevel', 'subkey']], compute_derived),
)


class TestConf(MultiSrcConf):
    __test__ = False

    STRUCTURE = TopLevelKeyDesc('lisa-self-test-test-conf', 'lisa self test',
        INTERNAL_STRUCTURE
    )


class TestConfWithDefault(MultiSrcConf):
    __test__ = False

    STRUCTURE = TopLevelKeyDesc('lisa-self-test-test-conf-with-default', 'lisa self test',
        INTERNAL_STRUCTURE
    )

    DEFAULT_SRC = {
        'bar': [0, 1, 2],
    }


class TestMultiSrcConf(TestMultiSrcConfBase):
    def test_add_src_one_key(self):
        conf = copy.deepcopy(self.conf)
        conf_src = {'foo': 22}

        conf.add_src('mysrc', conf_src)

        goal = dict(self.conf)
        goal.update(conf_src)
        assert dict(conf) == goal

        assert conf.resolve_src('foo') == 'mysrc'

    def test_disallowed_val(self):
        with pytest.raises(TypeError):
            self.conf.add_src('bar', {'foo': ['a', 'b']})

    def test_multitypes(self):
        conf = copy.deepcopy(self.conf)
        conf.add_src('mysrc', {'multitypes': 'a'})
        conf.add_src('mysrc', {'multitypes': [1, 2]})
        conf.add_src('mysrc', {'multitypes': None})


class TestTestConf(TestMultiSrcConf):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = TestConf()

    def test_unset_key(self):
        with pytest.raises(KeyError):
            self.conf['foo']

    def test_derived(self):
        conf = copy.deepcopy(self.conf)
        conf.add_src('mysrc', {'foo': 1})
        # Two missing base keys
        with pytest.raises(KeyError):
            conf['derived']
        conf.add_src('mysrc2', {
            'bar': [1, 2],
            'sublevel': {
                'subkey': 42
            }
        })
        assert conf['derived'] == 46

    def test_derived_with_deferred_base(self):
        conf = copy.deepcopy(self.conf)
        conf.add_src('mysrc', {
            'foo': DeferredValue(lambda: 1),
            'bar': [1, 2],
            'sublevel': {
                'subkey': 42
            }
        })
        # Check that the value we get is transitively a DeferredValue
        val = conf.get_key('derived', eval_deferred=False)
        assert isinstance(val, DeferredValue)

        # Regular access will evaluate all the bases
        assert conf['derived'] == 46

    def test_force_src_nested(self):
        conf = copy.deepcopy(self.conf)
        conf.add_src('mysrc', {'bar': [6, 7]})

        # Check without any actual source
        conf.force_src_nested({
            'bar': ['src-that-does-not-exist', 'another-one-that-does-not-exists'],
        })
        with pytest.raises(KeyError):
            conf['bar']

        # Check the first existing source is taken
        conf.force_src_nested({
            'bar': ['src-that-does-not-exist', 'mysrc2', 'mysrc', 'this-src-does-not-exist', 'mysrc'],
        })
        assert conf['bar'] == [6, 7]

        # Add one source that was specified earlier, and that has priority
        conf.add_src('mysrc2', {'bar': [99, 100]})
        assert conf['bar'] == [99, 100]

        # Reset the source priority, so the last source added will win
        conf.force_src('bar', None)
        assert conf['bar'] == [99, 100]

        src_map = conf.get_src_map('bar')
        assert list(src_map.keys()) == ['mysrc2', 'mysrc']


class TestTestConfWithDefault(TestMultiSrcConf):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = TestConfWithDefault()

    def test_default_src(self):
        ref = dict(TestConfWithDefault.DEFAULT_SRC)
        # A freshly built object still has all the level keys, even if it has
        # no leaves
        ref['sublevel'] = {}
        assert dict(self.conf) == ref

    def test_add_src_one_key_fallback(self):
        conf = copy.deepcopy(self.conf)
        # Since there is a default value for this key, it should not impact the
        # result
        conf_src = {'bar': [1, 2]}

        conf.add_src('bar', conf_src, fallback=True)

        assert dict(conf) == dict(self.conf)
        assert conf.resolve_src('bar') == 'default'

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
