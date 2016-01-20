#    Copyright 2012-2015 ARM Limited
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
# pylint: disable=E1101,W0201
import os
import time
import urllib
import tarfile
import shutil
import json
import re

from collections import defaultdict

from wlauto import settings, Workload, Parameter, Alias, Executable
from wlauto.exceptions import ConfigError
from wlauto.utils.types import boolean

DEFAULT_BBENCH_FILE = "http://bbench.eecs.umich.edu/bbench/bbench_2.0.tgz"
DOWNLOADED_FILE_NAME = "bbench_2.0.tgz"
BBENCH_SERVER_NAME = 'bbench_server'
PATCH_FILES = os.path.join(os.path.dirname(__file__), "patches")
DEFAULT_AUDIO_FILE = "http://archive.org/download/PachelbelsCanoninD/Canon_in_D_Piano.mp3"
DEFAULT_AUDIO_FILE_NAME = 'Canon_in_D_Piano.mp3'


class BBench(Workload):

    name = 'bbench'
    description = """
    BBench workload opens the built-in browser and navigates to, and
    scrolls through, some preloaded web pages and ends the workload by trying to
    connect to a local server it runs after it starts. It can also play the
    workload while it plays an audio file in the background.

    """

    summary_metrics = ['Mean Latency']

    parameters = [
        Parameter('with_audio', kind=boolean, default=False,
                  description=('Specifies whether an MP3 should be played in the background during '
                               'workload execution.')),
        Parameter('server_timeout', kind=int, default=300,
                  description='Specifies the timeout (in seconds) before the server is stopped.'),
        Parameter('force_dependency_push', kind=boolean, default=False,
                  description=('Specifies whether to push dependency files to the device to the device '
                               'if they are already on it.')),
        Parameter('audio_file', default=os.path.join(settings.dependencies_directory, 'Canon_in_D_Piano.mp3'),
                  description=('The (on-host) path to the audio file to be played. This is only used if '
                               '``with_audio`` is ``True``.')),
        Parameter('perform_cleanup', kind=boolean, default=False,
                  description='If ``True``, workload files on the device will be deleted after execution.'),
        Parameter('clear_file_cache', kind=boolean, default=True,
                  description='Clear the the file cache on the target device prior to running the workload.'),
        Parameter('browser_package', default='com.android.browser',
                  description='Specifies the package name of the device\'s browser app.'),
        Parameter('browser_activity', default='.BrowserActivity',
                  description='Specifies the startup activity  name of the device\'s browser app.'),
    ]

    aliases = [
        Alias('bbench_with_audio', with_audio=True),
    ]

    supported_platforms = ['android']

    def setup(self, context):  # NOQA
        self.bbench_on_device = '/'.join([self.device.working_directory, 'bbench'])
        self.bbench_server_on_device = os.path.join(self.device.working_directory, BBENCH_SERVER_NAME)
        self.audio_on_device = os.path.join(self.device.working_directory, DEFAULT_AUDIO_FILE_NAME)
        self.index_noinput = 'file:///{}'.format(self.bbench_on_device) + '/index_noinput.html'

        if not os.path.isdir(os.path.join(self.dependencies_directory, "sites")):
            self._download_bbench_file()
        if self.with_audio and not os.path.isfile(self.audio_file):
            self._download_audio_file()

        if not os.path.isdir(self.dependencies_directory):
            raise ConfigError('Bbench directory does not exist: {}'.format(self.dependencies_directory))
        self._apply_patches()

        if self.with_audio:
            if self.force_dependency_push or not self.device.file_exists(self.audio_on_device):
                self.device.push_file(self.audio_file, self.audio_on_device, timeout=120)

        # Push the bbench site pages and http server to target device
        if self.force_dependency_push or not self.device.file_exists(self.bbench_on_device):
            self.logger.debug('Copying bbench sites to device.')
            self.device.push_file(self.dependencies_directory, self.bbench_on_device, timeout=300)

        # Push the bbench server
        host_binary = context.resolver.get(Executable(self, self.device.abi, 'bbench_server'))
        device_binary = self.device.install(host_binary)
        self.luanch_server_command = '{} {}'.format(device_binary, self.server_timeout)

        # Open the browser with default page
        self.device.execute('am start -n  {}/{} about:blank'.format(self.browser_package, self.browser_activity))
        time.sleep(5)

        # Stop the browser if already running and wait for it to stop
        self.device.execute('am force-stop {}'.format(self.browser_package))
        time.sleep(5)

        # Clear the logs
        self.device.clear_logcat()

        # clear browser cache
        self.device.execute('pm clear {}'.format(self.browser_package))
        if self.clear_file_cache:
            self.device.execute('sync')
            self.device.set_sysfile_value('/proc/sys/vm/drop_caches', 3)

        #On android 6+ the web browser requires permissions to access the sd card
        if self.device.get_sdk_version() >= 23:
            self.device.execute("pm grant com.android.browser android.permission.READ_EXTERNAL_STORAGE")
            self.device.execute("pm grant com.android.browser android.permission.WRITE_EXTERNAL_STORAGE")

        # Launch the background music
        if self.with_audio:
            self.device.execute('am start -W -S -n com.android.music/.MediaPlaybackActivity -d {}'.format(self.audio_on_device))

    def run(self, context):
        # Launch the bbench
        self.device.execute('am start -n  {}/{} {}'.format(self.browser_package, self.browser_activity, self.index_noinput))
        time.sleep(5)  # WA1 parity
        # Launch the server waiting for Bbench to complete
        self.device.execute(self.luanch_server_command, self.server_timeout)

    def update_result(self, context):
        # Stop the browser
        self.device.execute('am force-stop {}'.format(self.browser_package))

        # Stop the music
        if self.with_audio:
            self.device.execute('am force-stop com.android.music')

        # Get index_no_input.html
        indexfile = os.path.join(self.device.working_directory, 'bbench/index_noinput.html')
        self.device.pull_file(indexfile, context.output_directory)

        # Get the logs
        output_file = os.path.join(self.device.working_directory, 'browser_bbench_logcat.txt')
        self.device.execute('logcat -v time -d > {}'.format(output_file))
        self.device.pull_file(output_file, context.output_directory)

        metrics = _parse_metrics(os.path.join(context.output_directory, 'browser_bbench_logcat.txt'),
                                 os.path.join(context.output_directory, 'index_noinput.html'),
                                 context.output_directory)
        for key, values in metrics:
            for i, value in enumerate(values):
                metric = '{}_{}'.format(key, i) if i else key
                context.result.add_metric(metric, value, units='ms', lower_is_better=True)

    def teardown(self, context):
        if self.perform_cleanup:
            self.device.execute('rm -r {}'.format(self.bbench_on_device))
            self.device.execute('rm {}'.format(self.audio_on_device))

    def _download_audio_file(self):
        self.logger.debug('Downloadling audio file.')
        urllib.urlretrieve(DEFAULT_AUDIO_FILE, self.audio_file)

    def _download_bbench_file(self):
        # downloading the file to bbench_dir
        self.logger.debug('Downloading bbench dependencies.')
        full_file_path = os.path.join(self.dependencies_directory, DOWNLOADED_FILE_NAME)
        urllib.urlretrieve(DEFAULT_BBENCH_FILE, full_file_path)

        # Extracting Bbench to bbench_dir/
        self.logger.debug('Extracting bbench dependencies.')
        tar = tarfile.open(full_file_path)
        tar.extractall(os.path.dirname(self.dependencies_directory))

        # Removing not needed files and the compressed file
        os.remove(full_file_path)
        youtube_dir = os.path.join(self.dependencies_directory, 'sites', 'youtube')
        os.remove(os.path.join(youtube_dir, 'www.youtube.com', 'kp.flv'))
        os.remove(os.path.join(youtube_dir, 'kp.flv'))

    def _apply_patches(self):
        self.logger.debug('Applying patches.')
        shutil.copy(os.path.join(PATCH_FILES, "bbench.js"), self.dependencies_directory)
        shutil.copy(os.path.join(PATCH_FILES, "results.html"), self.dependencies_directory)
        shutil.copy(os.path.join(PATCH_FILES, "index_noinput.html"), self.dependencies_directory)
        shutil.copy(os.path.join(PATCH_FILES, "bbc.html"),
                    os.path.join(self.dependencies_directory, "sites", "bbc", "www.bbc.co.uk", "index.html"))
        shutil.copy(os.path.join(PATCH_FILES, "cnn.html"),
                    os.path.join(self.dependencies_directory, "sites", "cnn", "www.cnn.com", "index.html"))
        shutil.copy(os.path.join(PATCH_FILES, "twitter.html"),
                    os.path.join(self.dependencies_directory, "sites", "twitter", "twitter.com", "index.html"))


def _parse_metrics(logfile, indexfile, output_directory):  # pylint: disable=R0914
    regex_bbscore = re.compile(r'(?P<head>\w+)=(?P<val>\w+)')
    regex_bbmean = re.compile(r'Mean = (?P<mean>[0-9\.]+)')
    regex_pagescore_head = re.compile(r'metrics:(\w+),(\d+)')
    regex_pagescore_tail = re.compile(r',(\d+.\d+)')
    regex_indexfile = re.compile(r'<body onload="startTest\((.*)\)">')
    settings_dict = defaultdict()

    with open(indexfile) as fh:
        for line in fh:
            match = regex_indexfile.search(line)
            if match:
                settings_dict['iterations'], settings_dict['scrollDelay'], settings_dict['scrollSize'] = match.group(1).split(',')
    with open(logfile) as fh:
        results_dict = defaultdict(list)
        for line in fh:
            if 'metrics:Mean' in line:
                results_list = regex_bbscore.findall(line)
                results_dict['Mean Latency'].append(regex_bbmean.search(line).group('mean'))
                if results_list:
                    break
            elif 'metrics:' in line:
                page_results = [0]
                match = regex_pagescore_head.search(line)
                name, page_results[0] = match.groups()
                page_results.extend(regex_pagescore_tail.findall(line[match.end():]))
                for val in page_results[:-2]:
                    results_list.append((name, int(float(val))))

        setting_names = ['siteIndex', 'CGTPreviousTime', 'scrollDelay', 'scrollSize', 'iterations']
        for k, v in results_list:
            if k not in setting_names:
                results_dict[k].append(v)

        sorted_results = sorted(results_dict.items())

        with open(os.path.join(output_directory, 'settings.json'), 'w') as wfh:
            json.dump(settings_dict, wfh)

    return sorted_results
