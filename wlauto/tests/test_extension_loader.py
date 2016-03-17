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
from unittest import TestCase

from nose.tools import assert_equal, assert_greater

from wlauto.core.pluginloader import PluginLoader


EXTDIR = os.path.join(os.path.dirname(__file__), 'data', 'plugins')


class PluginLoaderTest(TestCase):

    def test_load_device(self):
        loader = PluginLoader(paths=[EXTDIR, ], load_defaults=False)
        device = loader.get_device('test-device')
        assert_equal(device.name, 'test-device')

    def test_list_by_kind(self):
        loader = PluginLoader(paths=[EXTDIR, ], load_defaults=False)
        exts = loader.list_devices()
        assert_equal(len(exts), 1)
        assert_equal(exts[0].name, 'test-device')

    def test_clear_and_reload(self):
        loader = PluginLoader()
        assert_greater(len(loader.list_devices()), 1)
        loader.clear()
        loader.update(paths=[EXTDIR, ])
        devices = loader.list_devices()
        assert_equal(len(devices), 1)
        assert_equal(devices[0].name, 'test-device')
        assert_equal(len(loader.list_plugins()), 1)

