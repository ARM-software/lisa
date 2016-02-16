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

# pylint: disable=E1101,E0203,W0201

import os
import time
import urllib
from collections import defaultdict

from wlauto import Workload, settings, Parameter, Alias
from wlauto.exceptions import ConfigError, WorkloadError
from wlauto.utils.misc import ensure_directory_exists as _d
from wlauto.utils.types import boolean

DOWNLOAD_URLS = {
    '1080p': 'http://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_surround.avi',
    '720p': 'http://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_surround.avi',
    '480p': 'http://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_surround-fix.avi'
}


class VideoWorkload(Workload):
    name = 'video'
    description = """
    Plays a video file using the standard android video player for a predetermined duration.

    The video can be specified either using ``resolution`` workload parameter, in which case
    `Big Buck Bunny`_ MP4 video of that resolution will be downloaded and used, or using
    ``filename`` parameter, in which case the video file specified will be used.


    .. _Big Buck Bunny: http://www.bigbuckbunny.org/

    """
    supported_platforms = ['android']

    parameters = [
        Parameter('play_duration', kind=int, default=20,
                  description='Playback duration of the video file. This become the duration of the workload.'),
        Parameter('resolution', default='720p', allowed_values=['480p', '720p', '1080p'],
                  description='Specifies which resolution video file to play.'),
        Parameter('filename',
                  description="""
                   The name of the video file to play. This can be either a path
                   to the file anywhere on your file system, or it could be just a
                   name, in which case, the workload will look for it in
                   ``~/.workloads_automation/dependency/video``
                   *Note*: either resolution or filename should be specified, but not both!
                  """),
        Parameter('force_dependency_push', kind=boolean, default=False,
                  description="""
                  If true, video will always be pushed to device, regardless
                  of whether the file is already on the device.  Default is ``False``.
                  """),
    ]

    aliases = [
        Alias('video_720p', resolution='720p'),
        Alias('video_1080p', resolution='1080p'),
    ]

    @property
    def host_video_file(self):
        if not self._selected_file:
            if self.filename:
                if self.filename[0] in './' or len(self.filename) > 1 and self.filename[1] == ':':
                    filepath = os.path.abspath(self.filename)
                else:
                    filepath = os.path.join(self.video_directory, self.filename)
                if not os.path.isfile(filepath):
                    raise WorkloadError('{} does not exist.'.format(filepath))
                self._selected_file = filepath
            else:
                files = self.video_files[self.resolution]
                if not files:
                    url = DOWNLOAD_URLS[self.resolution]
                    filepath = os.path.join(self.video_directory, os.path.basename(url))
                    self.logger.debug('Downloading {}...'.format(filepath))
                    urllib.urlretrieve(url, filepath)
                    self._selected_file = filepath
                else:
                    self._selected_file = files[0]
                    if len(files) > 1:
                        self.logger.warn('Multiple files for 720p found. Using {}.'.format(self._selected_file))
                        self.logger.warn('Use \'filename\'parameter instead of \'resolution\' to specify a different file.')
        return self._selected_file

    def init_resources(self, context):
        self.video_directory = _d(os.path.join(settings.dependencies_directory, 'video'))
        self.video_files = defaultdict(list)
        self.enum_video_files()
        self._selected_file = None

    def setup(self, context):
        on_device_video_file = os.path.join(self.device.working_directory, os.path.basename(self.host_video_file))
        if self.force_dependency_push or not self.device.file_exists(on_device_video_file):
            self.logger.debug('Copying {} to device.'.format(self.host_video_file))
            self.device.push(self.host_video_file, on_device_video_file, timeout=120)
        self.device.execute('am start -n  com.android.browser/.BrowserActivity about:blank')
        time.sleep(5)
        self.device.execute('am force-stop com.android.browser')
        time.sleep(5)
        self.device.clear_logcat()
        command = 'am start -W -S -n com.android.gallery3d/.app.MovieActivity -d {}'.format(on_device_video_file)
        self.device.execute(command)

    def run(self, context):
        time.sleep(self.play_duration)

    def update_result(self, context):
        self.device.execute('am force-stop com.android.gallery3d')

    def teardown(self, context):
        pass

    def validate(self):
        if (self.resolution and self.filename) and (self.resolution != self.parameters['resolution'].default):
            raise ConfigError('Ether resolution *or* filename must be specified; but not both.')

    def enum_video_files(self):
        for filename in os.listdir(self.video_directory):
            for resolution in self.parameters['resolution'].allowed_values:
                if resolution in filename:
                    self.video_files[resolution].append(os.path.join(self.video_directory, filename))

