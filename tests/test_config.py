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

import unittest
from nose.tools import assert_equal

from wa.framework.configuration.execution import ConfigManager
from wa.utils.misc import merge_config_values


class TestConfigUtils(unittest.TestCase):

    def test_merge_values(self):
        test_cases = [
            # base, other, expected_result
            ('a', 3, 3),
            ('a', [1, 2], ['a', 1, 2]),
            ({1: 2}, [3, 4], [{1: 2}, 3, 4]),
            (set([2]), [1, 2, 3], [2, 1, 3]),
            ([1, 2, 3], set([2]), set([1, 2, 3])),
            ([1, 2], None, [1, 2]),
            (None, 'a', 'a'),
        ]
        for v1, v2, expected in test_cases:
            result = merge_config_values(v1, v2)
            assert_equal(result, expected)
            if v2 is not None:
                assert_equal(type(result), type(v2))



class TestConfigParser(unittest.TestCase):

    def test_param_merge(self):
        config = ConfigManager()

        config.load_config({'workload_params': {'one': 1, 'three': {'ex': 'x'}}, 'runtime_params': {'aye': 'a'}}, 'file_one')
        config.load_config({'workload_params': {'two': 2, 'three': {'why': 'y'}}, 'runtime_params': {'bee': 'b'}}, 'file_two')

        assert_equal(
            config.jobs_config.job_spec_template['workload_parameters'],
            {'one': 1, 'two': 2, 'three': {'why': 'y'}},
        )
        assert_equal(
            config.jobs_config.job_spec_template['runtime_parameters'],
            {'aye': 'a', 'bee': 'b'},
        )
