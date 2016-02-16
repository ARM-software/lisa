#    Copyright 2013-2015 ARM Limited
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

import os
import xml.etree.ElementTree as ET

from wlauto import AndroidUiAutoBenchmark


class Cfbench(AndroidUiAutoBenchmark):

    name = 'cfbench'
    description = """
    CF-Bench is (mainly) CPU and memory benchmark tool specifically designed to
    be able to handle multi-core devices, produce a fairly stable score, and
    test both native as well managed code performance.

    https://play.google.com/store/apps/details?id=eu.chainfire.cfbench&hl=en

    From the website:

    It tests specific device properties you do not regularly see tested by other
    benchmarks, and runs in a set timeframe.

    It does produce some "final" scores, but as with every benchmark, you should
    take those with a grain of salt. It is simply not theoretically possible to
    produce a single number that accurately describes a device's performance.

    .. note:: This workload relies on the device being rooted

    """
    package = 'eu.chainfire.cfbench'
    activity = '.MainActivity'
    run_timeout = 5 * 60  # seconds
    summary_metrics = ['overall_score']

    cfbench_params = ['java_mdflops', 'native_memory_read', 'java_msflops', 'native_disk_read', 'native_score', 'java_efficiency_memory_read',
                      'native_mips', 'native_mdflops', 'java_score', 'native_memory_write', 'java_memory_write', 'native_mallocs', 'native_msflops',
                      'java_mips', 'java_efficiency_mdflops', 'overall_score', 'java_memory_read', 'java_efficiency_memory_write', 'java_efficiency_mips',
                      'java_efficiency_msflops', 'native_disk_write']

    def update_result(self, context):
        super(Cfbench, self).update_result(context)
        device_results_file = os.path.join(self.device.package_data_directory,
                                           self.package,
                                           'shared_prefs', 'eu.chainfire.cfbench_preferences.xml ')
        self.device.execute('cp {} {}'.format(device_results_file, self.device.working_directory), as_root=True)
        self.device.pull(os.path.join(self.device.working_directory, 'eu.chainfire.cfbench_preferences.xml'), context.output_directory)
        result_file = os.path.join(context.output_directory, 'eu.chainfire.cfbench_preferences.xml')
        tree = ET.parse(result_file)
        root = tree.getroot()
        for child in root:
            if child.attrib['name'] in self.cfbench_params:
                if '%' in child.text:
                    value = float(child.text.split('%')[0]) / 100
                else:
                    value = int(child.text)
                context.result.add_metric(child.attrib['name'], value)


