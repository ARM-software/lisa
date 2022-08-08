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

from wa import Parameter, ApkWorkload, PackageHandler, TestPackageHandler


class Uibenchjanktests(ApkWorkload):

    name = 'uibenchjanktests'
    description = """
        Runs a particular test of the UIBench JankTests_ test suite. The suite
        is provided by Google as an automated version of the UIBench testbench
        for the Android UI.

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
        Parameter('test', kind=str,
                  description='Test to be run. Defaults to full run.'),
        Parameter('wait', kind=bool, default=True,
                  description='Forces am instrument to wait until the '
                  'instrumentation terminates before terminating itself. The '
                  'net effect is to keep the shell open until the tests have '
                  'finished. This flag is not required, but if you do not use '
                  'it, you will not see the results of your tests.'),
        Parameter('raw', kind=bool, default=False,
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

    def initialize(self, context):
        super(Uibenchjanktests, self).initialize(context)
        self.dut_apk.initialize(context)
        self.dut_apk.initialize_package(context)
        if 'class' not in self.apk.args:
            class_for_method = dict(self.apk.apk_info.methods)
            class_for_method[None] = self._DEFAULT_CLASS
            try:
                method = class_for_method[self.test]
            except KeyError as e:
                msg = 'Unknown test "{}". Known tests:\n\t{}'
                known_tests = '\n\t'.join(
                    m for m in class_for_method.keys()
                    if m is not None and m.startswith('test'))
                raise ValueError(msg.format(e, known_tests))
            klass = '{}.{}'.format(self.package_names[0], method)

            if self.test:
                klass += '#{}'.format(self.test)
            self.apk.args['class'] = klass

    def run(self, context):
        self.apk.start_activity()
        self.apk.wait_instrument_over()

    def update_output(self, context):
        super(Uibenchjanktests, self).update_output(context)
        output = self.apk.instrument_output
        for section in self._OUTPUT_SECTION_REGEX.finditer(output):
            if int(section.group('code')) != -1:
                msg = 'Run failed (INSTRUMENTATION_STATUS_CODE: {}). See log.'
                raise RuntimeError(msg.format(section.group('code')))
            for metric in self._OUTPUT_GFXINFO_REGEX.finditer(section.group()):
                context.add_metric(metric.group('name'), metric.group('value'))

    def teardown(self, context):
        super(Uibenchjanktests, self).teardown(context)
        self.dut_apk.teardown()
