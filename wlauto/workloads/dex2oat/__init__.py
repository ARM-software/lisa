#    Copyright 2014-2015 ARM Limited
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

# pylint: disable=no-member,attribute-defined-outside-init
import re
import os
import sys

from wlauto import Workload, Parameter, ExtensionLoader
from wlauto.exceptions import WorkloadError
from wlauto.utils.android import ApkInfo
import wlauto.common.android.resources


class Dex2oatBenchmark(Workload):

    name = 'dex2oat'
    description = """
        Benchmarks the execution time of dex2oat (a key part of APK installation process).

        ART is a new Android runtime in KitKat, which replaces Dalvik VM. ART uses Ahead-Of-Time
        compilation. It pre-compiles ODEX files used by Dalvik using dex2oat tool as part of APK
        installation process.

        This workload benchmarks the time it take to compile an APK using dex2oat, which has a
        significant impact on the total APK installation time, and therefore  user experience.

        """

    supported_platforms = ['android']
    command_template = 'dex2oat --dex-file={} --oat-file={} --instruction-set={} --dump-timing'
    run_timeout = 5 * 60

    parameters = [
        Parameter('instruction_set', default='arm64',
                  allowed_values=['arm', 'arm64', 'x86', 'x86_64', 'mips'],
                  description="""Specifies the instruction set to compile for.  Only options supported by
                                 the target device can be used."""),
    ]

    def init_resources(self, context):
        # TODO: find a better APK to use for this.
        peacekeeper = ExtensionLoader().get_workload('peacekeeper', self.device)
        self.apk_file = context.resolver.get(wlauto.common.android.resources.ApkFile(peacekeeper), version='chrome')
        self.package = ApkInfo(self.apk_file).package

    def setup(self, context):
        if self.device.getprop('persist.sys.dalvik.vm.lib.2') != 'libart.so':
            raise WorkloadError('Android system must be using ART (rather than Dalvik) in order for dex2oat to work.')
        supported = [eabi == 'armeabi' and 'arm' or eabi.split('-')[0]
                     for eabi in self.device.supported_eabi]
        if self.instruction_set not in supported:
            message = 'Instruction set "{}" is not supported by the device; (supported: {})'
            raise WorkloadError(message.format(self.instruction_set, supported))

        on_device_apk = self.device.path.join(self.device.working_directory,
                                              os.path.basename(self.apk_file))
        self.on_device_oat = on_device_apk.replace('.apk', '-{}.oat'.format(self.instruction_set))
        self.command = self.command_template.format(on_device_apk, self.on_device_oat, self.instruction_set)

        if not self.device.file_exists(on_device_apk):
            self.device.push_file(self.apk_file, on_device_apk)

    def run(self, context):
        self.device.execute(self.command, self.run_timeout)

    def update_result(self, context):
        """
        Retrieve the last dex2oat time from the logs. That will correspond with the run() method.
        The compilation time does not.

        Pulls out the compilation time and dex2oat execution time:
            I/dex2oat ( 2522):     1.8s Compile Dex File
            I/dex2oat ( 2522): dex2oat took 2.366s (threads: 6)


        """
        logcat_log = os.path.join(context.output_directory, 'logcat.log')
        self.device.dump_logcat(logcat_log)

        regex_time = re.compile("^I\/dex2oat \( *[0-9]+\): dex2oat took (?P<time>[0-9]+\.?[0-9]*)(?P<unit>m?s)")
        regex_comp_time = re.compile("^I\/dex2oat \( *[0-9]+\): +(?P<time>[0-9]*\.?[0-9]*)(?P<unit>m?s) Compile Dex File")
        time_data, comp_time_data = None, None
        with open(logcat_log) as fh:
            for line in fh:
                match = regex_time.search(line)

                if match:
                    time_data = match.groupdict()

                match = regex_comp_time.search(line)

                if match:
                    comp_time_data = match.groupdict()
        # Last dex2oat time wins.
        if time_data is not None:
            time = time_data['time']
            if time_data['unit'] == "s":
                time = float(time) * 1000.0
            context.result.add_metric('dex2oat_time', time, "ms", lower_is_better=True)

        if comp_time_data is not None:
            time = comp_time_data['time']
            if comp_time_data['unit'] == "s":
                time = float(time) * 1000.0
            context.result.add_metric('dex2oat_comp_time', time, "ms", lower_is_better=True)

    def teardown(self, context):
        self.device.delete_file(self.on_device_oat)

