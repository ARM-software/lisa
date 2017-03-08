import os
from unittest import TestCase
from copy import copy

from nose.tools import assert_equal  # pylint: disable=E0611
from mock.mock import Mock, MagicMock, call

from wlauto.exceptions import ConfigError
from wlauto.core.configuration.parsers import *  # pylint: disable=wildcard-import
from wlauto.core.configuration.parsers import _load_file, _collect_valid_id, _resolve_params_alias
from wlauto.core.configuration import RunConfiguration, JobGenerator, PluginCache, ConfigurationPoint
from wlauto.core.configuration.configuration import MetaConfiguration
from wlauto.utils.types import toggle_set, reset_counter


class TestFunctions(TestCase):

    def test_load_file(self):
        # This does not test read_pod

        # Non-existant file
        with self.assertRaises(ValueError):
            _load_file("THIS-IS-NOT-A-FILE", "test file")
        base_path = os.path.dirname(os.path.realpath(__file__))

        # Top level entry not a dict
        with self.assertRaisesRegexp(ConfigError, r".+ does not contain a valid test file structure; top level must be a dict\."):
            _load_file(os.path.join(base_path, "data", "test-agenda-not-dict.yaml"), "test file")

        # Yaml syntax error
        with self.assertRaisesRegexp(ConfigError, r"Error parsing test file .+: Syntax Error on line 1"):
            _load_file(os.path.join(base_path, "data", "test-agenda-bad-syntax.yaml"), "test file")

        # Ideal case
        _load_file(os.path.join(base_path, "data", "test-agenda.yaml"), "test file")

    def test_get_aliased_param(self):
        # Ideal case
        cp1 = ConfigurationPoint("test", aliases=[
            'workload_parameters',
            'workload_params',
            'params'
        ])

        d_correct = {"workload_parameters": [1, 2, 3],
                     "instruments": [2, 3, 4],
                     "some_other_param": 1234}
        assert_equal(get_aliased_param(cp1, d_correct, default=[], pop=False), [1, 2, 3])

        # Two aliases for the same parameter given
        d_duplicate = {"workload_parameters": [1, 2, 3],
                       "workload_params": [2, 3, 4]}
        with self.assertRaises(ConfigError):
            get_aliased_param(cp1, d_duplicate, default=[])

        # Empty dict
        d_none = {}
        assert_equal(get_aliased_param(cp1, d_none, default=[]), [])

        # Aliased parameter not present in dict
        d_not_present = {"instruments": [2, 3, 4],
                         "some_other_param": 1234}
        assert_equal(get_aliased_param(cp1, d_not_present, default=1), 1)

        # Testing pop functionality
        assert_equal("workload_parameters" in d_correct, True)
        get_aliased_param(cp1, d_correct, default=[])
        assert_equal("workload_parameters" in d_correct, False)

    def test_merge_result_processor_instruments(self):
        non_merge = {
            "instrumentation": toggle_set(["one", "two"]),
        }
        expected_non_merge = copy(non_merge)
        merge_result_processors_instruments(non_merge)
        assert_equal(non_merge, expected_non_merge)

        no_overlap = {
            "instrumentation": ["one", "two"],
            "result_processors": ["three", "~four"],
        }
        expected_no_overlap = {"instrumentation": toggle_set(["one", "two", "three", "~four"])}
        merge_result_processors_instruments(no_overlap)
        assert_equal(no_overlap, expected_no_overlap)

        non_conflicting = {
            "instrumentation": ["one", "two"],
            "result_processors": ["two", "three"],
        }
        expected_non_conflicting = {"instrumentation": toggle_set(["one", "two", "three"])}
        merge_result_processors_instruments(non_conflicting)
        assert_equal(non_conflicting, expected_non_conflicting)

        conflict = {
            "instrumentation": ["one", "two"],
            "result_processors": ["~two", "three"],
        }
        with self.assertRaises(ConfigError):
            merge_result_processors_instruments(conflict)

    def test_collect_valid_id(self):

        msg = 'Invalid unit_test ID "uses-a-dash"; IDs cannot contain a "-"'
        with self.assertRaisesRegexp(ConfigError, msg):
            _collect_valid_id("uses-a-dash", set(), "unit_test")

        msg = 'Invalid unit_test ID "global"; is a reserved ID'
        with self.assertRaisesRegexp(ConfigError, msg):
            _collect_valid_id("global", set(), "unit_test")

        msg = 'Duplicate unit_test ID "duplicate"'
        with self.assertRaisesRegexp(ConfigError, msg):
            _collect_valid_id("duplicate", set(["duplicate"]), "unit_test")

    def test_resolve_params_alias(self):
        test = {"params": "some_value"}
        _resolve_params_alias(test, "new_name")
        assert_equal(test, {"new_name_parameters": "some_value"})

        # Test it only affects "params"
        _resolve_params_alias(test, "new_name")
        assert_equal(test, {"new_name_parameters": "some_value"})

        test["params"] = "some_other_value"
        with self.assertRaises(ConfigError):
            _resolve_params_alias(test, "new_name")


class TestConfigParser(TestCase):

    def test_error_cases(self):
        wa_config = Mock(spec=MetaConfiguration)
        wa_config.configuration = MetaConfiguration.configuration
        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration
        config_parser = ConfigParser(wa_config,
                                     run_config,
                                     Mock(spec=JobGenerator),
                                     Mock(spec=PluginCache))

        # "run_name" can only be in agenda config sections
        #' and is handled by AgendaParser
        err = 'Error in "Unit test":\n' \
              '"run_name" can only be specified in the config section of an agenda'
        with self.assertRaisesRegexp(ConfigError, err):
            config_parser.load({"run_name": "test"}, "Unit test")

        # Instrument and result_processor lists in the same config cannot
        # have conflicting entries.
        err = 'Error in "Unit test":\n' \
              '"instrumentation" and "result_processors" have conflicting entries:'
        with self.assertRaisesRegexp(ConfigError, err):
            config_parser.load({"instruments": ["one", "two", "three"],
                                "result_processors": ["~one", "~two", "~three"]},
                               "Unit test")

    def test_config_points(self):
        wa_config = Mock(spec=MetaConfiguration)
        wa_config.configuration = MetaConfiguration.configuration

        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration

        jobs_config = Mock(spec=JobGenerator)
        plugin_cache = Mock(spec=PluginCache)
        config_parser = ConfigParser(wa_config, run_config, jobs_config, plugin_cache)

        cfg = {
            "assets_repository": "/somewhere/",
            "logging": "verbose",
            "project": "some project",
            "project_stage": "stage 1",
            "iterations": 9001,
            "workload_name": "name"
        }
        config_parser.load(cfg, "Unit test")
        wa_config.set.assert_has_calls([
            call("assets_repository", "/somewhere/"),
            call("logging", "verbose")
        ], any_order=True)
        run_config.set.assert_has_calls([
            call("project", "some project"),
            call("project_stage", "stage 1")
        ], any_order=True)
        jobs_config.set_global_value.assert_has_calls([
            call("iterations", 9001),
            call("workload_name", "name"),
            call("instrumentation", toggle_set())
        ], any_order=True)

        # Test setting global instruments including a non-conflicting duplicate ("two")
        jobs_config.reset_mock()
        instruments_and_result_processors = {
            "instruments": ["one", "two"],
            "result_processors": ["two", "three"]
        }
        config_parser.load(instruments_and_result_processors, "Unit test")
        jobs_config.set_global_value.assert_has_calls([
            call("instrumentation", toggle_set(["one", "two", "three"]))
        ], any_order=True)

        # Testing a empty config
        jobs_config.reset_mock()
        config_parser.load({}, "Unit test")
        jobs_config.set_global_value.assert_has_calls([], any_order=True)
        wa_config.set.assert_has_calls([], any_order=True)
        run_config.set.assert_has_calls([], any_order=True)


class TestAgendaParser(TestCase):

    # Tests Phase 1 & 2
    def test_valid_structures(self):
        wa_config = Mock(spec=MetaConfiguration)
        wa_config.configuration = MetaConfiguration.configuration
        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration
        jobs_config = Mock(spec=JobGenerator)
        plugin_cache = Mock(spec=PluginCache)
        agenda_parser = AgendaParser(wa_config, run_config, jobs_config, plugin_cache)

        msg = 'Error in "Unit Test":\n\tInvalid agenda, top level entry must be a dict'
        with self.assertRaisesRegexp(ConfigError, msg):
            agenda_parser.load(123, "Unit Test")

        def _test_bad_type(name, source, msg):
            error_msg = msg.format(source=source, name=name)
            with self.assertRaisesRegexp(ConfigError, error_msg):
                agenda_parser.load({name: 123}, source)

        msg = 'Error in "{source}":\n\tInvalid entry "{name}" - must be a dict'
        _test_bad_type("config", "Unit Test", msg)
        _test_bad_type("global", "Unit Test", msg)

        msg = 'Error in "Unit Test":\n\tInvalid entry "{name}" - must be a list'
        _test_bad_type("sections", "Unit Test", msg)
        _test_bad_type("workloads", "Unit Test", msg)

        msg = 'Error in "Unit Test":\n\tInvalid top level agenda entry\(ies\): "{name}"'
        _test_bad_type("not_a_valid_entry", "Unit Test", msg)

    # Test Phase 3
    def test_id_collection(self):
        wa_config = Mock(spec=MetaConfiguration)
        wa_config.configuration = MetaConfiguration.configuration
        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration
        jobs_config = Mock(spec=JobGenerator)
        plugin_cache = Mock(spec=PluginCache)
        agenda_parser = AgendaParser(wa_config, run_config, jobs_config, plugin_cache)

        agenda = {
            "workloads": [
                {"id": "test1"},
                {"id": "test2"},
            ],
            "sections": [
                {"id": "section1",
                 "workloads": [
                     {"id": "section1_workload"}
                 ]}
            ]
        }
        workloads, sections = agenda_parser.load(agenda, "Unit Test")
        assert_equal(sections, set(["section1"]))
        assert_equal(workloads, set(["test1", "test2", "section1_workload"]))

    # Test Phase 4
    def test_id_assignment(self):
        wa_config = Mock(spec=MetaConfiguration)
        wa_config.configuration = MetaConfiguration.configuration
        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration
        jobs_config = Mock(spec=JobGenerator)
        plugin_cache = Mock(spec=PluginCache)
        agenda_parser = AgendaParser(wa_config, run_config, jobs_config, plugin_cache)

        # Helper function
        def _assert_ids(ids, expected):
            ids_set = set(ids)
            assert_equal(len(ids), len(ids_set))
            assert_equal(ids_set, set(expected))

        def _assert_workloads_sections(jobs_config, expected_sect, expected_wk):
            wk_ids = [wk[0][0]['id'] for wk in jobs_config.add_workload.call_args_list]
            # section workloads
            for s in jobs_config.add_section.call_args_list:
                wk_ids += [wk['id'] for wk in s[0][1]]
            #sections
            sec_ids = set([s[0][0]['id'] for s in jobs_config.add_section.call_args_list])
            _assert_ids(wk_ids, set(expected_wk))
            _assert_ids(sec_ids, set(expected_sect))
            _reset_jobs_config(jobs_config)

        def _reset_jobs_config(jobs_config):
            jobs_config.reset_mock()
            reset_counter("wk")
            reset_counter("s")

        # Test auto id assignment
        auto_id = {
            "workloads": [
                {"name": 1},
                {"name": 2},
                {"name": 3},
            ],
            "sections": [
                {"name": 4,
                 "workloads": [
                     {"name": 7},
                     {"name": 8},
                     {"name": 9},
                 ]},
                {"name": 5},
                {"name": 6},
            ]
        }
        agenda_parser.load(auto_id, "Unit Test")
        _assert_workloads_sections(jobs_config, ["s1", "s2", "s3"],
                                   ["wk1", "wk2", "wk3", "wk4", "wk5", "wk6"])

        # Test user defined IDs
        user_ids = {
            "workloads": [
                {"id": "user1"},
                {"name": "autoid1"},
            ],
            "sections": [
                {"id": "user_section1",
                 "workloads": [
                     {"name": "autoid2"}
                 ]}
            ]
        }
        agenda_parser.load(user_ids, "Unit Test")
        _assert_workloads_sections(jobs_config, ["user_section1"],
                                   ["user1", "wk1", "wk2"])

        # Test auto asigned ID already present
        used_auto_id = {
            "workloads": [
                {"id": "wk2"},
                {"name": 2},
                {"name": 3},
            ],
        }
        agenda_parser.load(used_auto_id, "Unit Test")
        _assert_workloads_sections(jobs_config, [], ["wk1", "wk2", "wk3"])

        # Test string workload
        string = {
            "workloads": [
                "test"
            ]
        }
        agenda_parser.load(string, "Unit Test")
        workload = jobs_config.add_workload.call_args_list[0][0][0]
        assert_equal(isinstance(workload, dict), True)
        assert_equal(workload['workload_name'], "test")




class TestCommandLineArgsParser(TestCase):
    wa_config = Mock(spec=MetaConfiguration)
    run_config = Mock(spec=RunConfiguration)
    jobs_config = Mock(spec=JobGenerator)

    cmd_args = MagicMock(
        verbosity=1,
        output_directory="my_results",
        instruments_to_disable=["abc", "def", "ghi"],
        only_run_ids=["wk1", "s1_wk4"],
        some_other_setting="value123"
    )
    CommandLineArgsParser(cmd_args, wa_config, jobs_config)
    wa_config.set.assert_has_calls([call("verbosity", 1)], any_order=True)
    jobs_config.disable_instruments.assert_has_calls([
        call(toggle_set(["~abc", "~def", "~ghi"]))
    ], any_order=True)
    jobs_config.only_run_ids.assert_has_calls([call(["wk1", "s1_wk4"])], any_order=True)
