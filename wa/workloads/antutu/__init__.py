#    Copyright 2014-2018 ARM Limited
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
import os
import time

from wa import ApkUiautoWorkload, ApkWorkload, WorkloadError, Parameter, ApkFile, File


class Antutu(ApkUiautoWorkload):

    name = 'antutu'
    package_names = ['com.antutu.ABenchMark']
    regex_matches_v7 = [re.compile(r'CPU Maths Score (.+)'),
                        re.compile(r'CPU Common Score (.+)'),
                        re.compile(r'CPU Multi Score (.+)'),
                        re.compile(r'GPU Marooned Score (.+)'),
                        re.compile(r'GPU Coastline Score (.+)'),
                        re.compile(r'GPU Refinery Score (.+)'),
                        re.compile(r'Data Security Score (.+)'),
                        re.compile(r'Data Processing Score (.+)'),
                        re.compile(r'Image Processing Score (.+)'),
                        re.compile(r'User Experience Score (.+)'),
                        re.compile(r'RAM Score (.+)'),
                        re.compile(r'ROM Score (.+)')]
    regex_matches_v8 = [re.compile(r'CPU Mathematical Operations Score (.+)'),
                        re.compile(r'CPU Common Algorithms Score (.+)'),
                        re.compile(r'CPU Multi-Core Score (.+)'),
                        re.compile(r'GPU Terracotta Score (.+)'),
                        re.compile(r'GPU Coastline Score (.+)'),
                        re.compile(r'GPU Refinery Score (.+)'),
                        re.compile(r'Data Security Score (.+)'),
                        re.compile(r'Data Processing Score (.+)'),
                        re.compile(r'Image Processing Score (.+)'),
                        re.compile(r'User Experience Score (.+)'),
                        re.compile(r'RAM Access Score (.+)'),
                        re.compile(r'ROM APP IO Score (.+)'),
                        re.compile(r'ROM Sequential Read Score (.+)'),
                        re.compile(r'ROM Sequential Write Score (.+)'),
                        re.compile(r'ROM Random Access Score (.+)')]
    regex_matches_v9 = [re.compile(r'CPU Mathematical Operations Score (.+)'),
                        re.compile(r'CPU Common Algorithms Score (.+)'),
                        re.compile(r'CPU Multi-Core Score (.+)'),
                        re.compile(r'GPU Terracotta Score (.+)'),
                        re.compile(r'GPU Swordsman Score (.+)'),
                        re.compile(r'GPU Refinery Score (.+)'),
                        re.compile(r'Data Security Score (.+)'),
                        re.compile(r'Data Processing Score (.+)'),
                        re.compile(r'Image Processing Score (.+)'),
                        re.compile(r'User Experience Score (.+)'),
                        re.compile(r'Video CTS Score (.+)'),
                        re.compile(r'Video Decode Score (.+)'),
                        re.compile(r'RAM Access Score (.+)'),
                        re.compile(r'ROM APP IO Score (.+)'),
                        re.compile(r'ROM Sequential Read Score (.+)'),
                        re.compile(r'ROM Sequential Write Score (.+)'),
                        re.compile(r'ROM Random Access Score (.+)')]
    regex_matches_v10 = [re.compile(r'CPU Mathematical Operations Score (.+)'),
                         re.compile(r'CPU Common Algorithms Score (.+)'),
                         re.compile(r'CPU Multi-Core Score (.+)'),
                         re.compile(r'GPU Seasons Score (.+)'),
                         re.compile(r'GPU Coastline2 Score (.+)'),
                         re.compile(r'RAM Bandwidth Score (.+)'),
                         re.compile(r'RAM Latency Score (.+)'),
                         re.compile(r'ROM APP IO Score (.+)'),
                         re.compile(r'ROM Sequential Read Score (.+)'),
                         re.compile(r'ROM Sequential Write Score (.+)'),
                         re.compile(r'ROM Random Access Score (.+)'),
                         re.compile(r'Data Security Score (.+)'),
                         re.compile(r'Data Processing Score (.+)'),
                         re.compile(r'Document Processing Score (.+)'),
                         re.compile(r'Image Decoding Score (.+)'),
                         re.compile(r'Image Processing Score (.+)'),
                         re.compile(r'User Experience Score (.+)'),
                         re.compile(r'Video CTS Score (.+)'),
                         re.compile(r'Video Decoding Score (.+)'),
                         re.compile(r'Video Editing Score (.+)')]
    description = '''
    Executes Antutu 3D, UX, CPU and Memory tests

    Test description:
    1. Open Antutu application
    2. Execute Antutu benchmark

    Known working APK version: 8.0.4
    '''

    supported_versions = ['7.0.4', '7.2.0',
                          '8.0.4', '8.1.9', '8.4.5',
                          '9.1.6', '9.2.9',
                          '10.0.1-OB1', '10.0.6-OB6', '10.1.9', '10.2.1', '10.4.3']

    parameters = [
        Parameter('version', kind=str, allowed_values=supported_versions, override=True,
                  description=(
                      '''Specify the version of Antutu to be run.
                      If not specified, the latest available version will be used.
                      ''')
                  )
    ]

    def __init__(self, device, **kwargs):
        super(Antutu, self).__init__(device, **kwargs)
        self.gui.timeout = 1200

    def initialize(self, context):
        super(Antutu, self).initialize(context)
        #Install the supporting benchmark
        supporting_apk = context.get_resource(ApkFile(self, package='com.antutu.benchmark.full'))
        self.target.install(supporting_apk)
        #Ensure the orientation is set to portrait
        self.target.set_rotation(0)

    def setup(self, context):
        self.gui.uiauto_params['version'] = self.version
        super(Antutu, self).setup(context)

    def extract_scores(self, context, regex_version):
        #pylint: disable=no-self-use, too-many-locals
        cpu = []
        gpu = []
        ux = []
        mem = []
        expected_results = len(regex_version)
        logcat_file = context.get_artifact_path('logcat')
        with open(logcat_file, errors='replace') as fh:
            for line in fh:
                for regex in regex_version:
                    match = regex.search(line)
                    if match:
                        try:
                            result = float(match.group(1))
                        except ValueError:
                            result = float('NaN')
                        entry = regex.pattern.rsplit(None, 1)[0]
                        context.add_metric(entry, result, lower_is_better=False)
                        #Calculate group scores if 'CPU' in entry:
                        if 'CPU' in entry:
                            cpu.append(result)
                            cpu_result = sum(cpu)
                        if 'GPU' in entry:
                            gpu.append(result)
                            gpu_result = sum(gpu)
                        if any([i in entry for i in ['Data', 'Document', 'Image', 'User', 'Video']]):
                            ux.append(result)
                            ux_result = sum(ux)
                        if any([i in entry for i in ['RAM', 'ROM']]):
                            mem.append(result)
                            mem_result = sum(mem)
                        expected_results -= 1
        if expected_results > 0:
            msg = "The Antutu workload has failed. Expected {} scores, Detected {} scores."
            raise WorkloadError(msg.format(len(regex_version), expected_results))

        context.add_metric('CPU Total Score', cpu_result, lower_is_better=False)
        context.add_metric('GPU Total Score', gpu_result, lower_is_better=False)
        context.add_metric('UX Total Score', ux_result, lower_is_better=False)
        context.add_metric('MEM Total Score', mem_result, lower_is_better=False)

        #Calculate overall scores
        overall_result = float(cpu_result + gpu_result + ux_result + mem_result)
        context.add_metric('Overall Score', overall_result, lower_is_better=False)

    def update_output(self, context):
        super(Antutu, self).update_output(context)
        if self.version.startswith('10'):
            self.extract_scores(context, self.regex_matches_v10)
        if self.version.startswith('9'):
            self.extract_scores(context, self.regex_matches_v9)
        if self.version.startswith('8'):
            self.extract_scores(context, self.regex_matches_v8)
        if self.version.startswith('7'):
            self.extract_scores(context, self.regex_matches_v7)


class AntutuBDP(ApkWorkload):

    name = "antutu_bdp"
    description = '''
    Workload for executing the BDP versions of the Antutu APK.

    This will only work with specific APKS provided by Antutu but does
    unlock command line automation and the capturing of a result file
    as opposed to using UiAuto and Regex.

    Known working version: 10.4.3-domesticAndroidFullBdp
    '''
    activity = 'com.android.module.app.ui.start.ABenchMarkStart --ez isExternal true --es whereTo "test"'
    package_names = ['com.antutu.ABenchMark']

    def initialize(self, context):
        super(AntutuBDP, self).initialize(context)
        #Set the files and directories we need
        self.test_dir = os.path.join(self.target.external_storage_app_dir, 'com.antutu.ABenchMark', 'files', '.antutu')
        self.settings_xml = context.get_resource(File(self, 'settings.xml'))
        self.result_file = os.path.join(self.target.external_storage, 'Documents', 'antutu', 'last_result.json')
        self.output_file = os.path.join(context.output_directory, 'antutu_results.json')
        self.supporting_apk = context.get_resource(ApkFile(self, package='com.antutu.benchmark.full'))

    def setup(self, context):
        super(AntutuBDP, self).setup(context)
        #Install the supporting benchmark
        self.logger.info("Installing the supporting APK")
        self.target.install(self.supporting_apk)
        #Launch the apk to initialize the test dir, then kill it
        self.target.execute('am start {}/com.android.module.app.ui.test.activity.ActivityScoreBench'.format(self.apk.package))
        self.target.execute('am force-stop {}'.format(self.apk.package))
        #Copy the settings.xml to the test dir
        self.target.push(self.settings_xml, self.test_dir)
        #Ensure the orientation is set to portrait
        self.target.set_rotation(0)
        #Remove any pre-existing test results
        if self.target.file_exists(self.result_file):
            self.target.execute('rm {}'.format(self.result_file))

    def run(self, context):
        super(AntutuBDP, self).run(context)
        #Launch the tests
        self.target.execute('am start -n {}/{}'.format(self.apk.package, self.activity))
        #Wait 10 minutes, then begin polling every 30s for the test result to appear
        self.logger.debug("Waiting 10 minutes before starting to poll for the results file.")
        time.sleep(600)
        #Poll for another 15 minutes, 20 minutes total before timing out
        end_time = time.time() + 900
        while time.time() < end_time:
            if self.target.file_exists(self.result_file):
                self.logger.debug("Result file found.")
                return True
            time.sleep(30)
            self.logger.debug("File not found yet. Continuing polling.")
        self.logger.warning("File not found within the configured timeout period. Exiting test.")
        return False

    def update_output(self, context):
        super(AntutuBDP, self).update_output(context)
        self.target.pull(self.result_file, self.output_file)
        context.add_artifact('antutu_result', self.output_file, kind='data', description='Antutu output from target')

    def teardown(self, context):
        super(AntutuBDP, self).teardown(context)
        #Remove the test results file
        self.target.execute('rm {}'.format(self.result_file))
        #Remove the supporting APK
        if self.target.is_installed(self.supporting_apk):
            self.target.uninstall(self.supporting_apk)
