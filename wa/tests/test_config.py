import unittest
from nose.tools import assert_equal

from wa.framework.configuration import merge_config_values


class TestConfigUtils(unittest.TestCase):

    def test_merge_values(self):
        test_cases = [
            ('a', 3, 3),
            ('a', [1, 2], ['a', 1, 2]),
            ({1: 2}, [3, 4], [{1: 2}, 3, 4]),
            (set([2]), [1, 2, 3], [2, 1, 2, 3]),
            ([1, 2, 3], set([2]), set([1, 2, 3])),
            ([1, 2], None, [1, 2]),
            (None, 'a', 'a'),
        ]
        for v1, v2, expected in test_cases:
            assert_equal(merge_config_values(v1, v2), expected)

