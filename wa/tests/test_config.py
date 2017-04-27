import unittest
from nose.tools import assert_equal

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

