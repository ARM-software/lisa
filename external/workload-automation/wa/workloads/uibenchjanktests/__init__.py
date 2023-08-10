#    Copyright 2019 ARM Limited
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
import re

from wa import Parameter, ApkWorkload, PackageHandler, TestPackageHandler, ConfigError
from wa.utils.types import list_or_string
from wa.framework.exception import WorkloadError


class Uibenchjanktests(ApkWorkload):

    name = 'uibenchjanktests'
    description = """
        Runs a particular test (or list of tests) of the UIBench JankTests_
        test suite. The suite is provided by Google as an automated version
        of the UIBench testbench for the Android UI.
        The workload supports running the default set of tests without
        restarting the app or running an arbitrary set of tests with
        restarting the app in between each test.

        .. _JankTests: https://android.googlesource.com/platform/platform_testing/+/master/tests/jank/uibench/src/com/android/uibench/janktests
    """
    package_names = ['com.android.uibench.janktests']
    _DUT_PACKAGE = 'com.android.test.uibench'
    _DEFAULT_CLASS = 'UiBenchJankTests'
    _OUTPUT_SECTION_REGEX = re.compile(
        r'(\s*INSTRUMENTATION_STATUS: gfx-[\w-]+=[-+\d.]+\n)+'
        r'\s*INSTRUMENTATION_STATUS_CODE: (?P<code>[-+\d]+)\n?', re.M)
    _OUTPUT_GFXINFO_REGEX = re.compile(
        r'INSTRUMENTATION_STATUS: (?P<name>[\w-]+)=(?P<value>[-+\d.]+)')

    parameters = [
        Parameter('tests', kind=list_or_string,
                  description="""
                  Tests to be run. Defaults to running every available
                  subtest in alphabetical order. The app will be restarted
                  for each subtest, unlike when using full=True.
                  """, default=None, aliases=['test']),
        Parameter('full', kind=bool, default=False,
                  description="""
                  Runs the full suite of tests that the app defaults to
                  when no subtests are specified. The actual tests and their
                  order might depend on the version of the app. The subtests
                  will be run back to back without restarting the app in between.
                  """),
        Parameter('wait', kind=bool, default=True,
                  description='Forces am instrument to wait until the '
                  'instrumentation terminates before terminating itself. The '
                  'net effect is to keep the shell open until the tests have '
                  'finished. This flag is not required, but if you do not use '
                  'it, you will not see the results of your tests.'),
        Parameter('raw', kind=bool, default=True,
                  description='Outputs results in raw format. Use this flag '
                  'when you want to collect performance measurements, so that '
                  'they are not formatted as test results. This flag is '
                  'designed for use with the flag -e perf true.'),
        Parameter('instrument_args', kind=dict, default={},
                  description='Extra arguments for am instrument.'),
        Parameter('no_hidden_api_checks', kind=bool, default=False,
                  description='Disables restrictions on the use of hidden '
                  'APIs.'),
    ]

    def __init__(self, target, **kwargs):
        super(Uibenchjanktests, self).__init__(target, **kwargs)

        if 'iterations' not in self.instrument_args:
            self.instrument_args['iterations'] = 1

        self.dut_apk = PackageHandler(
            self,
            package_name=self._DUT_PACKAGE,
            variant=self.variant,
            strict=self.strict,
            version=self.version,
            force_install=self.force_install,
            install_timeout=self.install_timeout,
            uninstall=self.uninstall,
            exact_abi=self.exact_abi,
            prefer_host_package=self.prefer_host_package,
            clear_data_on_reset=self.clear_data_on_reset)
        self.apk = TestPackageHandler(
            self,
            package_name=self.package_name,
            variant=self.variant,
            strict=self.strict,
            version=self.version,
            force_install=self.force_install,
            install_timeout=self.install_timeout,
            uninstall=self.uninstall,
            exact_abi=self.exact_abi,
            prefer_host_package=self.prefer_host_package,
            clear_data_on_reset=self.clear_data_on_reset,
            instrument_args=self.instrument_args,
            raw_output=self.raw,
            instrument_wait=self.wait,
            no_hidden_api_checks=self.no_hidden_api_checks)

    def validate(self):
        if self.full and self.tests is not None:
            raise ConfigError("Can't select subtests while 'full' is True")

    def initialize(self, context):
        super(Uibenchjanktests, self).initialize(context)
        self.dut_apk.initialize(context)
        self.dut_apk.initialize_package(context)

        self.output = {}

        # Full run specified, don't select subtests
        if self.full:
            self.apk.args['class'] = '{}.{}'.format(
                self.package_names[0], self._DEFAULT_CLASS
            )
            return

        self.available_tests = {
            test: cl for test, cl in self.apk.apk_info.methods
            if test.startswith('test')
        }

        # default to running all tests in alphabetical order
        # pylint: disable=access-member-before-definition
        if not self.tests:
            self.tests = sorted(self.available_tests.keys())
        # raise error if any of the tests are not available
        elif any([t not in self.available_tests for t in self.tests]):
            msg = 'Unknown test(s) specified. Known tests: {}'
            known_tests = '\n'.join(self.available_tests.keys())
            raise ValueError(msg.format(known_tests))

    def run(self, context):
        # Full run, just run the activity directly
        if self.full:
            self.apk.start_activity()
            self.apk.wait_instrument_over()
            self.output['full'] = self.apk.instrument_output
            return

        for test in self.tests:
            self.apk.args['class'] = '{}.{}#{}'.format(
                self.package_names[0],
                self.available_tests[test], test
            )
            self.apk.setup(context)
            self.apk.start_activity()
            try:
                self.apk.wait_instrument_over()
            except WorkloadError as e:
                self.logger.warning(str(e))
            self.output[test] = self.apk.instrument_output

    def update_output(self, context):
        super(Uibenchjanktests, self).update_output(context)
        for test, test_output in self.output.items():
            for section in self._OUTPUT_SECTION_REGEX.finditer(test_output):
                if int(section.group('code')) != -1:
                    msg = 'Run failed (INSTRUMENTATION_STATUS_CODE: {}). See log.'
                    raise RuntimeError(msg.format(section.group('code')))
                for metric in self._OUTPUT_GFXINFO_REGEX.finditer(section.group()):
                    context.add_metric(metric.group('name'), metric.group('value'),
                                       classifiers={'test_name': test})

    def teardown(self, context):
        super(Uibenchjanktests, self).teardown(context)
        self.dut_apk.teardown()
