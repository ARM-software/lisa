import unittest
from nose.tools import assert_equal
from mock.mock import Mock

from wa.utils.misc import resolve_cpus, resolve_unique_domain_cpus

class TestRuntimeParameterUtils(unittest.TestCase):

    def test_resolve_cpu(self):
        # Set up a mock target
        mock = Mock()
        mock.big_core = "A72"
        mock.little_core = "A53"
        mock.core_names = ['A72', 'A72', 'A53', 'A53']
        mock.number_of_cpus = 4
        def mock_core_cpus(core):
            return [i for i, c in enumerate(mock.core_names) if c == core]
        def mock_online_cpus():
            return [0, 1, 2]
        def mock_offline_cpus():
            return [3]
        def mock_related_cpus(core):
            if core in [0, 1]:
                return [0, 1]
            elif core in [2, 3]:
                return [2, 3]

        mock.list_online_cpus = mock_online_cpus
        mock.list_offline_cpus = mock_offline_cpus
        mock.core_cpus = mock_core_cpus
        mock.core_cpus = mock_core_cpus
        mock.cpufreq.get_related_cpus = mock_related_cpus

        # Check retrieving cpus from a given prefix
        assert_equal(resolve_cpus('A72', mock), [0, 1])
        assert_equal(resolve_cpus('A53', mock), [2, 3])
        assert_equal(resolve_cpus('big', mock), [0, 1])
        assert_equal(resolve_cpus('little', mock), [2, 3])
        assert_equal(resolve_cpus('', mock), [0, 1, 2, 3])
        assert_equal(resolve_cpus('cpu0', mock), [0])
        assert_equal(resolve_cpus('cpu3', mock), [3])

        # Check get unique domain cpus
        assert_equal(resolve_unique_domain_cpus('A72', mock), [0])
        assert_equal(resolve_unique_domain_cpus('A53', mock), [2])
        assert_equal(resolve_unique_domain_cpus('big', mock), [0])
        assert_equal(resolve_unique_domain_cpus('little', mock), [2])
        assert_equal(resolve_unique_domain_cpus('', mock), [0, 2])
        assert_equal(resolve_unique_domain_cpus('cpu0', mock), [0])
        assert_equal(resolve_unique_domain_cpus('cpu3', mock), [2])
