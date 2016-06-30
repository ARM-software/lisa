# pylint: disable=R0201

from unittest import TestCase

from nose.tools import assert_equal, assert_is
from mock.mock import MagicMock, Mock

from wlauto.exceptions import ConfigError
from wlauto.core.configuration.tree import Node
from wlauto.core.configuration.configuration import (ConfigurationPoint, Configuration,
                                                     JobsConfiguration)

#       A1
#     /    \
#   B1      B2
#  /  \    /  \
# C1  C2  C3  C4
#      \
#      D1
a1 = Node("A1")
b1 = a1.add_section("B1")
b2 = a1.add_section("B2")
c1 = b1.add_section("C1")
c2 = b1.add_section("C2")
c3 = b2.add_section("C3")
c4 = b2.add_section("C4")
d1 = c2.add_section("D1")


class NodeTest(TestCase):

    def test_node(self):
        node = Node(1)
        assert_equal(node.config, 1)
        assert_is(node.parent, None)
        assert_equal(node.workloads, [])
        assert_equal(node.children, [])

    def test_add_workload(self):
        node = Node(1)
        node.add_workload(2)
        assert_equal(node.workloads, [2])

    def test_add_section(self):
        node = Node(1)
        new_node = node.add_section(2)
        assert_equal(len(node.children), 1)
        assert_is(node.children[0], new_node)
        assert_is(new_node.parent, node)
        assert_equal(node.is_leaf, False)
        assert_equal(new_node.is_leaf, True)

    def test_descendants(self):
        for got, expected in zip(b1.descendants(), [c1, d1, c2]):
            print "GOT:{} EXPECTED:{}".format(got.config, expected.config)
            assert_is(got, expected)
        print "----"
        for got, expected in zip(a1.descendants(), [c1, d1, c2, b1, c3, c4, b2]):
            print "GOT:{} EXPECTED:{}".format(got.config, expected.config)
            assert_is(got, expected)

    def test_ancestors(self):
        for got, expected in zip(d1.ancestors(), [c2, b1, a1]):
            print "GOT:{} EXPECTED:{}".format(got.config, expected.config)
            assert_is(got, expected)
        for _ in a1.ancestors():
            raise Exception("A1 is the root, it shouldn't have ancestors")

    def test_leaves(self):
        for got, expected in zip(a1.leaves(), [c1, d1, c3, c4]):
            print "GOT:{} EXPECTED:{}".format(got.config, expected.config)
            assert_is(got, expected)
        print "----"
        for got, expected in zip(d1.leaves(), [d1]):
            print "GOT:{} EXPECTED:{}".format(got.config, expected.config)
            assert_is(got, expected)


class ConfigurationPointTest(TestCase):

    def test_match(self):
        cp1 = ConfigurationPoint("test1", aliases=["foo", "bar"])
        cp2 = ConfigurationPoint("test2", aliases=["fizz", "buzz"])

        assert_equal(cp1.match("test1"), True)
        assert_equal(cp1.match("foo"), True)
        assert_equal(cp1.match("bar"), True)
        assert_equal(cp1.match("fizz"), False)
        assert_equal(cp1.match("NOT VALID"), False)

        assert_equal(cp2.match("test2"), True)
        assert_equal(cp2.match("fizz"), True)
        assert_equal(cp2.match("buzz"), True)
        assert_equal(cp2.match("foo"), False)
        assert_equal(cp2.match("NOT VALID"), False)

    def test_set_value(self):
        cp1 = ConfigurationPoint("test", default="hello")
        cp2 = ConfigurationPoint("test", mandatory=True)
        cp3 = ConfigurationPoint("test", mandatory=True, default="Hello")
        cp4 = ConfigurationPoint("test", default=["hello"], merge=True, kind=list)
        cp5 = ConfigurationPoint("test", kind=int)
        cp6 = ConfigurationPoint("test5", kind=list, allowed_values=[1, 2, 3, 4, 5])

        mock = Mock()
        mock.name = "ConfigurationPoint Unit Test"

        # Testing defaults and basic functionality
        cp1.set_value(mock)
        assert_equal(mock.test, "hello")
        cp1.set_value(mock, value="there")
        assert_equal(mock.test, "there")

        # Testing mandatory flag
        err_msg = 'No values specified for mandatory parameter "test" in ' \
                  'ConfigurationPoint Unit Test'
        with self.assertRaisesRegexp(ConfigError, err_msg):
            cp2.set_value(mock)
        cp3.set_value(mock)  # Should ignore mandatory
        assert_equal(mock.test, "Hello")

        # Testing Merging - not in depth that is done in the unit test for merge_config
        cp4.set_value(mock, value=["there"])
        assert_equal(mock.test, ["Hello", "there"])

        # Testing type conversion
        cp5.set_value(mock, value="100")
        assert_equal(isinstance(mock.test, int), True)
        msg = 'Bad value "abc" for test; must be an integer'
        with self.assertRaisesRegexp(ConfigError, msg):
            cp5.set_value(mock, value="abc")

        # Testing that validation is not called when no value is set
        # if it is it will error because it cannot iterate over None
        cp6.set_value(mock)

    def test_validation(self):
        def is_even(value):
            if value % 2:
                return False
            return True

        cp1 = ConfigurationPoint("test", kind=int, allowed_values=[1, 2, 3, 4, 5])
        cp2 = ConfigurationPoint("test", kind=list, allowed_values=[1, 2, 3, 4, 5])
        cp3 = ConfigurationPoint("test", kind=int, constraint=is_even)
        cp4 = ConfigurationPoint("test", kind=list, mandatory=True, allowed_values=[1, 99])
        mock = MagicMock()
        mock.name = "ConfigurationPoint Validation Unit Test"

        # Test allowed values
        cp1.validate_value(mock.name, 1)
        with self.assertRaises(ConfigError):
            cp1.validate_value(mock.name, 100)
        with self.assertRaises(ConfigError):
            cp1.validate_value(mock.name, [1, 2, 3])

        # Test allowed values for lists
        cp2.validate_value(mock.name, [1, 2, 3])
        with self.assertRaises(ConfigError):
            cp2.validate_value(mock.name, [1, 2, 100])

        # Test constraints
        cp3.validate_value(mock.name, 2)
        cp3.validate_value(mock.name, 4)
        cp3.validate_value(mock.name, 6)
        msg = '"3" failed constraint validation for "test" in "ConfigurationPoint' \
              ' Validation Unit Test".'
        with self.assertRaisesRegexp(ConfigError, msg):
            cp3.validate_value(mock.name, 3)

        with self.assertRaises(ValueError):
            ConfigurationPoint("test", constraint=100)

        # Test "validate" methods
        mock.test = None
        # Mandatory config point not set
        with self.assertRaises(ConfigError):
            cp4.validate(mock)
        cp1.validate(mock)  # cp1 doesnt have mandatory set
        cp4.set_value(mock, value=[99])
        cp4.validate(mock)

    def test_get_type_name(self):
        def dummy():
            pass
        types = [str, list, int, dummy]
        names = ["str", "list", "integer", "dummy"]
        for kind, name in zip(types, names):
            cp = ConfigurationPoint("test", kind=kind)
            assert_equal(cp.get_type_name(), name)


# Subclass just to add some config points to use in testing
class TestConfiguration(Configuration):
    name = "Test Config"
    __configuration = [
        ConfigurationPoint("test1", default="hello"),
        ConfigurationPoint("test2", mandatory=True),
        ConfigurationPoint("test3", default=["hello"], merge=True, kind=list),
        ConfigurationPoint("test4", kind=int, default=123),
        ConfigurationPoint("test5", kind=list, allowed_values=[1, 2, 3, 4, 5]),
    ]
    configuration = {cp.name: cp for cp in __configuration}


class ConfigurationTest(TestCase):

    def test(self):
        # Test loading defaults
        cfg = TestConfiguration()
        expected = {
            "test1": "hello",
            "test2": None,
            "test3": ["hello"],
            "test4": 123,
            "test5": None,
        }
        # If a cfg point is not set an attribute with value None should still be created
        for name, value in expected.iteritems():
            assert_equal(getattr(cfg, name), value)

        # Testing pre finalization "set"
        cfg.set("test1", "there")
        assert_equal(cfg.test1, "there")  # pylint: disable=E1101
        with self.assertRaisesRegexp(ConfigError, 'Unknown Test Config configuration "nope"'):
            cfg.set("nope", 123)

        # Testing setting values from a dict
        new_values = {
            "test1": "This",
            "test2": "is",
            "test3": ["a"],
            "test4": 7357,
            "test5": [5],
        }
        cfg.update_config(new_values)
        new_values["test3"] = ["hello", "a"]  # This cfg point has merge == True
        for k, v in new_values.iteritems():
            assert_equal(getattr(cfg, k), v)

        # Test finalization

        # This is a madatory cfg point so finalization should fail
        cfg.configuration["test2"].set_value(cfg, value=None, check_mandatory=False)
        msg = 'No value specified for mandatory parameter "test2" in Test Config'
        with self.assertRaisesRegexp(ConfigError, msg):
            cfg.finalize()
        assert_equal(cfg._finalized, False)  # pylint: disable=W0212

        # Valid finalization
        cfg.set("test2", "is")
        cfg.finalize()
        assert_equal(cfg._finalized, True)  # pylint: disable=W0212

        # post finalization set should failed
        with self.assertRaises(RuntimeError):
            cfg.set("test2", "is")


class JobsConfigurationTest(TestCase):

    def test_set_global_config(self):
        jc = JobsConfiguration()

        jc.set_global_config("workload_name", "test")
        assert_equal(jc.root_node.config.workload_name, "test")
        # Aliased names (e.g. "name") should be resolved by the parser
        # before being passed here.

        with self.assertRaises(ConfigError):
            jc.set_global_config("unknown", "test")

        jc.finalise_global_config()
        with self.assertRaises(RuntimeError):
            jc.set_global_config("workload_name", "test")

    def test_tree_manipulation(self):
        jc = JobsConfiguration()

        workloads = [123, "hello", True]
        for w in workloads:
            jc.add_workload(w)
        assert_equal(jc.root_node.workloads, workloads)

        jc.add_section("section", workloads)
        assert_equal(jc.root_node.children[0].config, "section")
        assert_equal(jc.root_node.workloads, workloads)

    def test_generate_job_specs(self):

    # disable_instruments
    # only_run_ids
