import os
from unittest import TestCase

from nose.tools import assert_equal  # pylint: disable=E0611
from mock.mock import Mock, MagicMock, call

from wlauto.exceptions import ConfigError
from wlauto.core.configuration.parsers import (get_aliased_param,
                                               _load_file, ConfigParser, EnvironmentVarsParser,
                                               CommandLineArgsParser)
from wlauto.core.configuration import (WAConfiguration, RunConfiguration, JobsConfiguration,
                                       PluginCache)
from wlauto.utils.types import toggle_set


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
        d_correct = {"workload_parameters": [1, 2, 3],
                     "instruments": [2, 3, 4],
                     "some_other_param": 1234}
        assert_equal(get_aliased_param(d_correct, [
            'workload_parameters',
            'workload_params',
            'params'
        ], default=[], pop=False), [1, 2, 3])

        # Two aliases for the same parameter given
        d_duplicate = {"workload_parameters": [1, 2, 3],
                       "workload_params": [2, 3, 4]}
        with self.assertRaises(ConfigError):
            get_aliased_param(d_duplicate, [
                'workload_parameters',
                'workload_params',
                'params'
            ], default=[])

        # Empty dict
        d_none = {}
        assert_equal(get_aliased_param(d_none, [
            'workload_parameters',
            'workload_params',
            'params'
        ], default=[]), [])

        # Aliased parameter not present in dict
        d_not_present = {"instruments": [2, 3, 4],
                         "some_other_param": 1234}
        assert_equal(get_aliased_param(d_not_present, [
            'workload_parameters',
            'workload_params',
            'params'
        ], default=1), 1)

        # Testing pop functionality
        assert_equal("workload_parameters" in d_correct, True)
        get_aliased_param(d_correct, [
            'workload_parameters',
            'workload_params',
            'params'
        ], default=[])
        assert_equal("workload_parameters" in d_correct, False)


class TestConfigParser(TestCase):

    def test_error_cases(self):
        wa_config = Mock(spec=WAConfiguration)
        wa_config.configuration = WAConfiguration.configuration
        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration
        config_parser = ConfigParser(wa_config,
                                     run_config,
                                     Mock(spec=JobsConfiguration),
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
        wa_config = Mock(spec=WAConfiguration)
        wa_config.configuration = WAConfiguration.configuration

        run_config = Mock(spec=RunConfiguration)
        run_config.configuration = RunConfiguration.configuration

        jobs_config = Mock(spec=JobsConfiguration)
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
        print jobs_config.set_global_config.call_args_list
        jobs_config.set_global_config.assert_has_calls([
            call("iterations", 9001),
            call("workload_name", "name"),
            call("instrumentation", toggle_set()),
            call("instrumentation", toggle_set())
        ], any_order=True)

        # Test setting global instruments including a non-conflicting duplicate ("two")
        jobs_config.reset_mock()
        instruments_and_result_processors = {
            "instruments": ["one", "two"],
            "result_processors": ["two", "three"]
        }
        config_parser.load(instruments_and_result_processors, "Unit test")
        jobs_config.set_global_config.assert_has_calls([
            call("instrumentation", toggle_set(["one", "two"])),
            call("instrumentation", toggle_set(["two", "three"]))
        ], any_order=True)

        # Testing a empty config
        jobs_config.reset_mock()
        config_parser.load({}, "Unit test")
        jobs_config.set_global_config.assert_has_calls([], any_order=True)
        wa_config.set.assert_has_calls([], any_order=True)
        run_config.set.assert_has_calls([], any_order=True)


class TestEnvironmentVarsParser(TestCase):

    def test_environmentvarsparser(self):
        wa_config = Mock(spec=WAConfiguration)
        calls = [call('user_directory', '/testdir'),
                 call('plugin_paths', ['/test', '/some/other/path', '/testy/mc/test/face'])]

        # Valid env vars
        valid_environ = {"WA_USER_DIRECTORY": "/testdir",
                         "WA_PLUGIN_PATHS": "/test:/some/other/path:/testy/mc/test/face"}
        EnvironmentVarsParser(wa_config, valid_environ)
        wa_config.set.assert_has_calls(calls)

        # Alternative env var name
        wa_config.reset_mock()
        alt_valid_environ = {"WA_USER_DIRECTORY": "/testdir",
                             "WA_EXTENSION_PATHS": "/test:/some/other/path:/testy/mc/test/face"}
        EnvironmentVarsParser(wa_config, alt_valid_environ)
        wa_config.set.assert_has_calls(calls)

        # Test that WA_EXTENSION_PATHS gets merged with WA_PLUGIN_PATHS.
        # Also checks that other enviroment variables don't cause errors
        wa_config.reset_mock()
        calls = [call('user_directory', '/testdir'),
                 call('plugin_paths', ['/test', '/some/other/path']),
                 call('plugin_paths', ['/testy/mc/test/face'])]
        ext_and_plgin = {"WA_USER_DIRECTORY": "/testdir",
                         "WA_PLUGIN_PATHS": "/test:/some/other/path",
                         "WA_EXTENSION_PATHS": "/testy/mc/test/face",
                         "RANDOM_VAR": "random_value"}
        EnvironmentVarsParser(wa_config, ext_and_plgin)
        # If any_order=True then the calls can be in any order, but they must all appear
        wa_config.set.assert_has_calls(calls, any_order=True)

        # No WA enviroment variables present
        wa_config.reset_mock()
        EnvironmentVarsParser(wa_config, {"RANDOM_VAR": "random_value"})
        wa_config.set.assert_not_called()


class TestCommandLineArgsParser(TestCase):
    wa_config = Mock(spec=WAConfiguration)
    run_config = Mock(spec=RunConfiguration)
    jobs_config = Mock(spec=JobsConfiguration)

    cmd_args = MagicMock(
        verbosity=1,
        output_directory="my_results",
        instruments_to_disable=["abc", "def", "ghi"],
        only_run_ids=["wk1", "s1_wk4"],
        some_other_setting="value123"
    )
    CommandLineArgsParser(cmd_args, wa_config, run_config, jobs_config)
    wa_config.set.assert_has_calls([call("verbosity", 1)], any_order=True)
    run_config.set.assert_has_calls([call("output_directory", "my_results")], any_order=True)
    jobs_config.disable_instruments.assert_has_calls([
        call(toggle_set(["~abc", "~def", "~ghi"]))
    ], any_order=True)
    jobs_config.only_run_ids.assert_has_calls([call(["wk1", "s1_wk4"])], any_order=True)


class TestAgendaParser(TestCase):
    pass
