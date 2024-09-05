# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, Arm Limited and contributors.
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

from pathlib import Path
from tempfile import NamedTemporaryFile
from io import StringIO
import json

from .utils import StorageTestCase

from lisa.utils import Serializable, UnknownTagPlaceholder


class EqDict:
    def __eq__(self, other):
        assert self.__class__ == other.__class__
        return self.__dict__ == other.__dict__


class MySerializable(EqDict, Serializable):
    def __init__(self, x):
        self.x = x


class MyGetState(MySerializable):
    def __getstate__(self):
        # We have to wrap in a dict, otherwise we get issues with self.x ==
        # None (pickle fails and ruamel.yaml incorrectly reloads self.x as an
        # empty dict instead).
        return {'val': self.x}

    def __setstate__(self, x):
        self.x = x['val']


class LoadUser(Serializable):
    @classmethod
    def from_yaml(cls, path):
        return cls._from_path(path, fmt='yaml')


class MyClass(EqDict):
    pass


class MyExcep(Exception):
    def __eq__(self, other):
        assert self.__class__ is other.__class__
        return self.args == other.args


class MyList(list):
    pass


class MyDict(dict):
    pass


class TestSerializable(StorageTestCase):

    def _test(self, obj, avoid_fmt=None):
        """
        Test that serialization works correctly
        """
        def test(obj, fmt):
            with NamedTemporaryFile(dir=self.res_dir) as f:
                path = Path(f.name)
                obj.to_path(path, fmt=fmt)
                obj2 = obj.__class__.from_path(path, fmt=fmt)

            assert obj == obj2


        fmts = [
            fmt
            for fmt in ('yaml', 'pickle')
            if fmt not in (avoid_fmt or [])
        ]
        for wrapper in (MySerializable, MyGetState):
            for fmt in fmts:
                test(wrapper(obj), fmt)

    def test_int(self):
        self._test(42)

    def test_float(self):
        self._test(42.42)

    def test_none(self):
        self._test(None)

    def test_list(self):
        self._test([42, 43])

    def test_custom_list(self):
        # list subclasses are important as they get serialized with
        # !!python/object/new and some list items.
        self._test(MyList([42, 43]))

    def test_dict(self):
        self._test({42: 43})

    def test_custom_dict(self):
        # list subclasses are important as they get serialized with
        # !!python/object/new and some dict items.
        self._test(MyDict({42: 43}))

    def test_tuple(self):
        self._test((42, 43))

    def test_complex(self):
        self._test(complex(42, 43))

    def test_class(self):
        # Test serializing the class, which should serialize by name.
        self._test(MyClass)

    def test_exception(self):
        # exceptions are important as they get serialized using
        # !!python/object/apply
        self._test(MyExcep('hello'))

    def test_module(self):
        def test(mod):
            # pickle does not support modules
            return self._test(mod, avoid_fmt={'pickle'})

        import lisa.utils
        test(lisa.utils)

        import lisa
        test(lisa)

    def test_include(self):
        def load(data, trust):
            with NamedTemporaryFile(dir=self.res_dir) as f_trusted:
                path_trusted = Path(f_trusted.name)
                with NamedTemporaryFile(dir=self.res_dir) as f_untrusted:
                    path_untrusted = Path(f_untrusted.name)

                    path_untrusted.write_text(data)

                    include_tag = 'include' if trust else 'include-untrusted'
                    data_trusted = f'!{include_tag} {json.dumps(str(path_untrusted))}'
                    path_trusted.write_text(data_trusted)

                    return LoadUser.from_yaml(path_trusted)

        # Create an unsafe parser before anything else. This way, if
        # ruamel.yaml leaks some state from one parser to the next, we will
        # attempt to decode untrusted values using some unsafe parser feature
        # and we will get test failures.
        assert load('', trust=True) == None

        # smoke test on a builtin value
        assert load('42', trust=True) == 42
        assert load('42', trust=False) == 42

        # Normal include allows untrusted tags
        assert load('!!python/object:tests.test_serialize.MyClass {}', trust=True) == MyClass()

        # Check this will not reload as a MyClass() instance, since we are not
        # trusting the content of the data.
        x = load('!!python/object:tests.test_serialize.MyClass {}', trust=False)
        assert isinstance(x, UnknownTagPlaceholder)
        assert x.tag == 'tag:yaml.org,2002:python/object:tests.test_serialize.MyClass'


