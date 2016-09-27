# pylint: disable=R0201
from copy import deepcopy, copy

from unittest import TestCase

from nose.tools import assert_equal, assert_is
from mock.mock import Mock

from wlauto.exceptions import ConfigError
from wlauto.core.configuration.tree import SectionNode
from wlauto.core.configuration.configuration import (ConfigurationPoint,
                                                     Configuration,
                                                     RunConfiguration,
                                                     merge_using_priority_specificity,
                                                     get_type_name)
from wlauto.core.configuration.plugin_cache import PluginCache, GENERIC_CONFIGS
from wlauto.utils.types import obj_dict
#       A1
#     /    \
#   B1      B2
#  /  \    /  \
# C1  C2  C3  C4
#      \
#      D1
a1 = SectionNode({"id": "A1"})
b1 = a1.add_section({"id": "B1"})
b2 = a1.add_section({"id": "B2"})
c1 = b1.add_section({"id": "C1"})
c2 = b1.add_section({"id": "C2"})
c3 = b2.add_section({"id": "C3"})
c4 = b2.add_section({"id": "C4"})
d1 = c2.add_section({"id": "D1"})

DEFAULT_PLUGIN_CONFIG = {
    "device_config": {
        "a": {
            "test3": ["there"],
            "test5": [5, 4, 3],
        },
        "b": {
            "test4": 1234,
        },
    },
    "some_device": {
        "a": {
            "test3": ["how are"],
            "test2": "MANDATORY",
        },
        "b": {
            "test3": ["you?"],
            "test5": [1, 2, 3],
        }
    }
}


def _construct_mock_plugin_cache(values=None):
    if values is None:
        values = deepcopy(DEFAULT_PLUGIN_CONFIG)

    plugin_cache = Mock(spec=PluginCache)
    plugin_cache.sources = ["a", "b", "c", "d", "e"]

    def get_plugin_config(plugin_name):
        return values[plugin_name]
    plugin_cache.get_plugin_config.side_effect = get_plugin_config

    def get_plugin_parameters(_):
        return TestConfiguration.configuration
    plugin_cache.get_plugin_parameters.side_effect = get_plugin_parameters

    return plugin_cache


class TreeTest(TestCase):

    def test_node(self):
        node = SectionNode(1)
        assert_equal(node.config, 1)
        assert_is(node.parent, None)
        assert_equal(node.workload_entries, [])
        assert_equal(node.children, [])

    def test_add_workload(self):
        node = SectionNode(1)
        node.add_workload(2)
        assert_equal(len(node.workload_entries), 1)
        wk = node.workload_entries[0]
        assert_equal(wk.config, 2)
        assert_is(wk.parent, node)

    def test_add_section(self):
        node = SectionNode(1)
        new_node = node.add_section(2)
        assert_equal(len(node.children), 1)
        assert_is(node.children[0], new_node)
        assert_is(new_node.parent, node)
        assert_equal(node.is_leaf, False)
        assert_equal(new_node.is_leaf, True)

    def test_descendants(self):
        for got, expected in zip(b1.descendants(), [c1, d1, c2]):
            assert_equal(got.config, expected.config)
        for got, expected in zip(a1.descendants(), [c1, d1, c2, b1, c3, c4, b2]):
            assert_equal(got.config, expected.config)

    def test_ancestors(self):
        for got, expected in zip(d1.ancestors(), [c2, b1, a1]):
            assert_equal(got.config, expected.config)
        for _ in a1.ancestors():
            raise Exception("A1 is the root, it shouldn't have ancestors")

    def test_leaves(self):
        for got, expected in zip(a1.leaves(), [c1, d1, c3, c4]):
            assert_equal(got.config, expected.config)
        for got, expected in zip(d1.leaves(), [d1]):
            assert_equal(got.config, expected.config)

    def test_source_name(self):
        assert_equal(a1.name, 'section "A1"')
        global_section = SectionNode({"id": "global"})
        assert_equal(global_section.name, "globally specified configuration")

        a1.add_workload({'id': 'wk1'})
        assert_equal(a1.workload_entries[0].name, 'workload "wk1" from section "A1"')
        global_section.add_workload({'id': 'wk2'})
        assert_equal(global_section.workload_entries[0].name, 'workload "wk2"')


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
        #Test invalid default
        with self.assertRaises(ValueError):
            # pylint: disable=W0612
            bad_cp = ConfigurationPoint("test", allowed_values=[1], default=100)

        def is_even(value):
            if value % 2:
                return False
            return True

        cp1 = ConfigurationPoint("test", kind=int, allowed_values=[1, 2, 3, 4, 5])
        cp2 = ConfigurationPoint("test", kind=list, allowed_values=[1, 2, 3, 4, 5])
        cp3 = ConfigurationPoint("test", kind=int, constraint=is_even)
        cp4 = ConfigurationPoint("test", kind=list, mandatory=True, allowed_values=[1, 99])
        mock = obj_dict()
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
            assert_equal(get_type_name(cp.kind), name)


# Subclass to add some config points for use in testing
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

    def test_merge_using_priority_specificity(self):
        # Test good configs
        plugin_cache = _construct_mock_plugin_cache()
        expected_result = {
            "test1": "hello",
            "test2": "MANDATORY",
            "test3": ["hello", "there", "how are", "you?"],
            "test4": 1234,
            "test5": [1, 2, 3],
        }
        result = merge_using_priority_specificity("device_config", "some_device", plugin_cache)
        assert_equal(result, expected_result)

        # Test missing mandatory parameter
        plugin_cache = _construct_mock_plugin_cache(values={
            "device_config": {
                "a": {
                    "test1": "abc",
                },
            },
            "some_device": {
                "b": {
                    "test5": [1, 2, 3],
                }
            }
        })
        msg = 'No value specified for mandatory parameter "test2" in some_device.'
        with self.assertRaisesRegexp(ConfigError, msg):
            merge_using_priority_specificity("device_config", "some_device", plugin_cache)

        # Test conflict
        plugin_cache = _construct_mock_plugin_cache(values={
            "device_config": {
                "e": {
                    'test2': "NOT_CONFLICTING"
                }
            },
            "some_device": {
                'a': {
                    'test2': "CONFLICT1"
                },
                'b': {
                    'test2': "CONFLICT2"
                },
                'c': {
                    'test2': "CONFLICT3"
                },
            },
        })
        msg = ('Error in "e":\n'
               '\t"device_config" configuration "test2" has already been specified more specifically for some_device in:\n'
               '\t\ta\n'
               '\t\tb\n'
               '\t\tc')
        with self.assertRaisesRegexp(ConfigError, msg):
            merge_using_priority_specificity("device_config", "some_device", plugin_cache)

        # Test invalid entries
        plugin_cache = _construct_mock_plugin_cache(values={
            "device_config": {
                "a": {
                    "NOT_A_CFG_POINT": "nope"
                }
            },
            "some_device": {}
        })
        msg = ('Error in "a":\n\t'
               'Invalid entry\(ies\) for "some_device" in "device_config": "NOT_A_CFG_POINT"')
        with self.assertRaisesRegexp(ConfigError, msg):
            merge_using_priority_specificity("device_config", "some_device", plugin_cache)

        plugin_cache = _construct_mock_plugin_cache(values={
            "some_device": {
                "a": {
                    "NOT_A_CFG_POINT": "nope"
                }
            },
            "device_config": {}
        })
        msg = ('Error in "a":\n\t'
               'Invalid entry\(ies\) for "some_device": "NOT_A_CFG_POINT"')
        with self.assertRaisesRegexp(ConfigError, msg):
            merge_using_priority_specificity("device_config", "some_device", plugin_cache)

    # pylint: disable=no-member
    def test_configuration(self):
        # Test loading defaults
        cfg = TestConfiguration()
        expected = {
            "test1": "hello",
            "test3": ["hello"],
            "test4": 123,
        }
        assert_equal(cfg.to_pod(), expected)
        # If a cfg point is not set an attribute with value None should still be created
        assert_is(cfg.test2, None)
        assert_is(cfg.test5, None)

        # Testing set
        # Good value
        cfg.set("test1", "there")
        assert_equal(cfg.test1, "there")  # pylint: disable=E1101
        # Unknown value
        with self.assertRaisesRegexp(ConfigError, 'Unknown Test Config configuration "nope"'):
            cfg.set("nope", 123)
        # check_mandatory
        with self.assertRaises(ConfigError):
            cfg.set("test2", value=None)
        cfg.set("test2", value=None, check_mandatory=False)
        # parameter constraints are tested in the ConfigurationPoint unit test
        # since this just calls through to `ConfigurationPoint.set_value`

        # Test validation
        msg = 'No value specified for mandatory parameter "test2" in Test Config'
        with self.assertRaisesRegexp(ConfigError, msg):
            cfg.validate()
        cfg.set("test2", 1)
        cfg.validate()

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

        #Testing podding
        pod = cfg.to_pod()
        new_pod = TestConfiguration.from_pod(copy(pod), None).to_pod()
        assert_equal(pod, new_pod)

        #invalid pod entry
        pod = {'invalid_entry': "nope"}
        msg = 'Invalid entry\(ies\) for "Test Config": "invalid_entry"'
        with self.assertRaisesRegexp(ConfigError, msg):
            TestConfiguration.from_pod(pod, None)

        #failed pod validation
        pod = {"test1": "testing"}
        msg = 'No value specified for mandatory parameter "test2" in Test Config.'
        with self.assertRaisesRegexp(ConfigError, msg):
            TestConfiguration.from_pod(pod, None)

    def test_run_configuration(self):
        plugin_cache = _construct_mock_plugin_cache()

        # Test `merge_device_config``
        run_config = RunConfiguration()
        run_config.set("device", "some_device")
        run_config.merge_device_config(plugin_cache)

        # Test `to_pod`
        expected_pod = {
            "device": "some_device",
            "device_config": {
                "test1": "hello",
                "test2": "MANDATORY",
                "test3": ["hello", "there", "how are", "you?"],
                "test4": 1234,
                "test5": [1, 2, 3],
            },
            "execution_order": "by_iteration",
            "reboot_policy": "as_needed",
            "retry_on_status": ['FAILED', 'PARTIAL'],
            "max_retries": 3,
        }
        pod = run_config.to_pod()
        assert_equal(pod, expected_pod)

        # Test to_pod > from_pod
        new_pod = RunConfiguration.from_pod(copy(pod), plugin_cache).to_pod()
        assert_equal(pod, new_pod)

        # from_pod with invalid device_config
        pod['device_config']['invalid_entry'] = "nope"
        msg = 'Invalid entry "invalid_entry" for device "some_device".'
        with self.assertRaisesRegexp(ConfigError, msg):
            RunConfiguration.from_pod(copy(pod), plugin_cache)

        # from_pod with no device_config
        pod.pop("device_config")
        msg = 'No value specified for mandatory parameter "device_config".'
        with self.assertRaisesRegexp(ConfigError, msg):
            RunConfiguration.from_pod(copy(pod), plugin_cache)

    def test_generate_job_spec(self):
        pass


class PluginCacheTest(TestCase):

    param1 = ConfigurationPoint("param1", aliases="test_global_alias")
    param2 = ConfigurationPoint("param2", aliases="some_other_alias")
    param3 = ConfigurationPoint("param3")

    plugin1 = obj_dict(values={
        "name": "plugin 1",
        "parameters": [
            param1,
            param2,
        ]
    })
    plugin2 = obj_dict(values={
        "name": "plugin 2",
        "parameters": [
            param1,
            param3,
        ]
    })

    def get_plugin(self, name):
        if name == "plugin 1":
            return self.plugin1
        if name == "plugin 2":
            return self.plugin2

    def has_plugin(self, name):
        return name in ["plugin 1", "plugin 2"]

    def make_mock_cache(self):
        mock_loader = Mock()
        mock_loader.get_plugin_class.side_effect = self.get_plugin
        mock_loader.list_plugins = Mock(return_value=[self.plugin1, self.plugin2])
        mock_loader.has_plugin.side_effect = self.has_plugin
        return PluginCache(loader=mock_loader)

    def test_get_params(self):
        plugin_cache = self.make_mock_cache()

        expected_params = {
            self.param1.name: self.param1,
            self.param2.name: self.param2,
        }

        assert_equal(expected_params, plugin_cache.get_plugin_parameters("plugin 1"))

    def test_global_aliases(self):
        plugin_cache = self.make_mock_cache()

        # Check the alias map
        expected_map = {
            "plugin 1": {
                self.param1.aliases: self.param1,
                self.param2.aliases: self.param2,
            },
            "plugin 2": {
                self.param1.aliases: self.param1,
            }
        }
        expected_set = set(["test_global_alias", "some_other_alias"])

        assert_equal(expected_map, plugin_cache._global_alias_map)
        assert_equal(expected_set, plugin_cache._list_of_global_aliases)
        assert_equal(True, plugin_cache.is_global_alias("test_global_alias"))
        assert_equal(False, plugin_cache.is_global_alias("not_a_global_alias"))

        # Error when adding to unknown source
        with self.assertRaises(RuntimeError):
            plugin_cache.add_global_alias("adding", "too", "early")

        # Test adding sources
        for x in xrange(5):
            plugin_cache.add_source(x)
        assert_equal([0, 1, 2, 3, 4], plugin_cache.sources)

        # Error when adding non plugin/global alias/generic
        with self.assertRaises(RuntimeError):
            plugin_cache.add_global_alias("unknow_alias", "some_value", 0)

        # Test adding global alias values
        plugin_cache.add_global_alias("test_global_alias", "some_value", 0)
        expected_aliases = {"test_global_alias": {0: "some_value"}}
        assert_equal(expected_aliases, plugin_cache.global_alias_values)

    def test_add_config(self):
        plugin_cache = self.make_mock_cache()

        # Test adding sources
        for x in xrange(5):
            plugin_cache.add_source(x)
        assert_equal([0, 1, 2, 3, 4], plugin_cache.sources)

        # Test adding plugin config
        plugin_cache.add_config("plugin 1", "param1", "some_other_value", 0)
        expected_plugin_config = {"plugin 1": {0: {"param1": "some_other_value"}}}
        assert_equal(expected_plugin_config, plugin_cache.plugin_configs)

        # Test adding generic config
        for name in GENERIC_CONFIGS:
            plugin_cache.add_config(name, "param1", "some_value", 0)
            expected_plugin_config[name] = {}
            expected_plugin_config[name][0] = {"param1": "some_value"}
        assert_equal(expected_plugin_config, plugin_cache.plugin_configs)

    def test_get_plugin_config(self):
        plugin_cache = self.make_mock_cache()
        for x in xrange(5):
            plugin_cache.add_source(x)

        # Add some global aliases
        plugin_cache.add_global_alias("test_global_alias", "1", 0)
        plugin_cache.add_global_alias("test_global_alias", "2", 4)
        plugin_cache.add_global_alias("test_global_alias", "3", 3)

        # Test if they are being merged in source order
        expected_config = {
            "param1": "2",
            "param2": None,
        }
        assert_equal(expected_config, plugin_cache.get_plugin_config("plugin 1"))

        # Add some plugin specific config
        plugin_cache.add_config("plugin 1", "param1", "3", 0)
        plugin_cache.add_config("plugin 1", "param1", "4", 2)
        plugin_cache.add_config("plugin 1", "param1", "5", 1)

        # Test if they are being merged in source order on top of the global aliases
        expected_config = {
            "param1": "4",
            "param2": None,
        }
        assert_equal(expected_config, plugin_cache.get_plugin_config("plugin 1"))

    def test_merge_using_priority_specificity(self):
        plugin_cache = self.make_mock_cache()
        for x in xrange(5):
            plugin_cache.add_source(x)

        # Add generic configs
        plugin_cache.add_config("device_config", "param1", '1', 1)
        plugin_cache.add_config("device_config", "param1", '2', 2)
        assert_equal(plugin_cache.get_plugin_config("plugin 1", generic_name="device_config"),
                     {"param1": '2', "param2": None})

        # Add specific configs at same level as generic config
        plugin_cache.add_config("plugin 1", "param1", '3', 2)
        assert_equal(plugin_cache.get_plugin_config("plugin 1", generic_name="device_config"),
                     {"param1": '3', "param2": None})

        # Add specific config at higher level
        plugin_cache.add_config("plugin 1", "param1", '4', 3)
        assert_equal(plugin_cache.get_plugin_config("plugin 1", generic_name="device_config"),
                     {"param1": '4', "param2": None})

        # Add generic config at higher level - should be an error
        plugin_cache.add_config("device_config", "param1", '5', 4)
        msg = 'Error in "4":\n' \
              '\t"device_config" configuration "param1" has already been specified' \
              ' more specifically for plugin 1 in:\n' \
              '\t\t2, 3'
        with self.assertRaisesRegexp(ConfigError, msg):
            plugin_cache.get_plugin_config("plugin 1", generic_name="device_config")
