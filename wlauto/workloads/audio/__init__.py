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

from wlauto import settings, Workload, Parameter
from wlauto.exceptions import ConfigError
from wlauto.utils.types import boolean


DEFAULT_AUDIO_FILE_URL = "http://archive.org/download/PachelbelsCanoninD/Canon_in_D_Piano.mp3"


class Audio(Workload):

    name = 'audio'
    description = """
    Audio workload plays an MP3 file using the built-in music player. By default,
    it plays Canon_in_D_Pieano.mp3 for 30 seconds.

    """
    supported_platforms = ['android']

    parameters = [
        Parameter('duration', kind=int, default=30,
                  description='The duration the music will play for in seconds.'),
        Parameter('audio_file', default=os.path.join(settings.dependencies_directory, 'Canon_in_D_Piano.mp3'),
                  description='''The (on-host) path to the audio file to be played.

                                 .. note:: If the default file is not present locally, it will be downloaded.
                  '''),
        Parameter('perform_cleanup', kind=boolean, default=False,
                  description='If ``True``, workload files on the device will be deleted after execution.'),
        Parameter('clear_file_cache', kind=boolean, default=True,
                  description='Clear the the file cache on the target device prior to running the workload.')
    ]

    def init_resources(self, context):
        if not os.path.isfile(self.audio_file):
            self._download_audio_file()

    def setup(self, context):
        self.on_device_file = os.path.join(self.device.working_directory,
                                           os.path.basename(self.audio_file))

        self.device.push_file(self.audio_file, self.on_device_file, timeout=120)

        # Open the browser with default page
        self.device.execute('am start -n  com.android.browser/.BrowserActivity about:blank')
        time.sleep(5)

        # Stop the browser if already running and wait for it to stop
        self.device.execute('am force-stop com.android.browser')
        time.sleep(5)

        # Clear the logs
        self.device.clear_logcat()

        # Clear browser cache
        self.device.execute('pm clear com.android.browser')

        if self.clear_file_cache:
            self.device.execute('sync')
            self.device.set_sysfile_value('/proc/sys/vm/drop_caches', 3)

        # Start the background music
        self.device.execute('am start -W -S -n com.android.music/.MediaPlaybackActivity -d {}'.format(self.on_device_file))

        # Launch the browser to blank the screen
        self.device.execute('am start -W -n  com.android.browser/.BrowserActivity about:blank')
        time.sleep(5)  # Wait for browser to be properly launched

    def run(self, context):
        time.sleep(self.duration)

    def update_result(self, context):
        # Stop the browser
        self.device.execute('am force-stop com.android.browser')
        # Stop the audio
        self.device.execute('am force-stop com.android.music')

    def teardown(self, context):
        if self.perform_cleanup:
            self.device.delete_file(self.on_device_file)

    def _download_audio_file(self):
        self.logger.debug('Downloading audio file from {}'.format(DEFAULT_AUDIO_FILE_URL))
        urllib.urlretrieve(DEFAULT_AUDIO_FILE_URL, self.audio_file)

