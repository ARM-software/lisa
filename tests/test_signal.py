#    Copyright 2018 ARM Limited
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

import logging
import unittest

from nose.tools import assert_equal, assert_true, assert_false

import wa.framework.signal as signal


class Callable(object):

    def __init__(self, val):
        self.val = val

    def __call__(self):
        return self.val


class TestSignalDisconnect(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_ctr = 0

    def setUp(self):
        signal.connect(self._call_me_once, 'first')
        signal.connect(self._call_me_once, 'second')

    def test_handler_disconnected(self):
        signal.send('first')
        signal.send('second')

    def _call_me_once(self):
        assert_equal(self.callback_ctr, 0)
        self.callback_ctr += 1
        signal.disconnect(self._call_me_once, 'first')
        signal.disconnect(self._call_me_once, 'second')


class TestPriorityDispatcher(unittest.TestCase):

    def setUp(self):
        # Stop logger output interfering with nose output in the console.
        logger = logging.getLogger('signal')
        logger.setLevel(logging.CRITICAL)

    def test_ConnectNotify(self):
        one = Callable(1)
        two = Callable(2)
        three = Callable(3)
        signal.connect(
            two,
            'test',
            priority=200
        )
        signal.connect(
            one,
            'test',
            priority=100
        )
        signal.connect(
            three,
            'test',
            priority=300
        )
        result = [i[1] for i in signal.send('test')]
        assert_equal(result, [3, 2, 1])

    def test_wrap_propagate(self):
        d = {'before': False, 'after': False, 'success': False}

        def before():
            d['before'] = True

        def after():
            d['after'] = True

        def success():
            d['success'] = True

        signal.connect(before, signal.BEFORE_WORKLOAD_SETUP)
        signal.connect(after, signal.AFTER_WORKLOAD_SETUP)
        signal.connect(success, signal.SUCCESSFUL_WORKLOAD_SETUP)

        caught = False
        try:
            with signal.wrap('WORKLOAD_SETUP'):
                raise RuntimeError()
        except RuntimeError:
            caught = True

        assert_true(d['before'])
        assert_true(d['after'])
        assert_true(caught)
        assert_false(d['success'])
