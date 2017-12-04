# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import os
import time
import urllib

from wa import ApkWorkload, Parameter, ConfigError, WorkloadError
from wa.framework.configuration.core import settings
from wa.utils.types import boolean
from wa.utils.misc import ensure_directory_exists
from devlib.utils.android import grant_app_permissions

# Regexps for benchmark synchronization
REGEXPS = {
    'start'         : '.*Displayed com.google.android.exoplayer2.demo/.PlayerActivity',
    'duration'      : '.*period \[(?P<duration>[0-9]+.*)\]',
    'end'           : '.*state \[.+, .+, E\]',
    'dropped_frames': '.*droppedFrames \[(?P<session_time>[0-9]+\.[0-9]+), (?P<count>[0-9]+)\]'
}


DOWNLOAD_URLS = {
    'mp4_1080p': 'http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4',
    'mov_720p':  'http://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov',
    'mov_480p':  'http://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov',
    'ogg_128kbps': 'http://upload.wikimedia.org/wikipedia/commons/c/ca/Tchaikovsky_-_Romeo_and_Juliet_Ouverture_-_Antal_Dorati_(1959).ogg',
}


class ExoPlayer(ApkWorkload):
    """
    Android ExoPlayer

    ExoPlayer is the basic video player library that is used by the YouTube
    android app. The aim of this workload is to test a proxy for YouTube
    performance on targets where running the real YouTube app is not possible
    due its dependencies.

    ExoPlayer sources: https://github.com/google/ExoPlayer

    The 'demo' application is used by this workload.  It can easily be built by
    loading the ExoPlayer sources into Android Studio.

    Version r2.4.0 built from commit d979469 is known to work

    Produces a metric 'exoplayer_dropped_frames' - this is the count of frames
    that Exoplayer itself reports as dropped. This is not the same thing as the
    dropped frames reported by gfxinfo.
    """

    name = 'exoplayer'

    video_directory = os.path.join(settings.dependencies_directory, name)

    package_names = ['com.google.android.exoplayer2.demo']
    versions = ['2.4.0']
    action = 'com.google.android.exoplayer.demo.action.VIEW'
    default_format = 'mov_720p'

    parameters = [
        Parameter('version', allowed_values=versions, default=versions[-1], override=True),
        Parameter('duration', kind=int, default=20,
                  description="""
                  Playback duration of the video file. This becomes the duration of the workload.
                  If provided must be shorter than the length of the media.
                  """),
        Parameter('format', allowed_values=DOWNLOAD_URLS.keys(),
                  description="""
                  Specifies which format video file to play. Default is {}
                  """.format(default_format)),
        Parameter('filename',
                  description="""
                   The name of the video file to play. This can be either a path
                   to the file anywhere on your file system, or it could be just a
                   name, in which case, the workload will look for it in
                   ``{}``
                   *Note*: either format or filename should be specified, but not both!
                  """.format(video_directory)),
        Parameter('force_dependency_push', kind=boolean, default=False,
                  description="""
                  If true, video will always be pushed to device, regardless
                  of whether the file is already on the device.  Default is ``False``.
                  """),
    ]

    def validate(self):
        if self.format and self.filename:
            raise ConfigError('Either format *or* filename must be specified; but not both.')

        if not self.format and not self.filename:
            self.format = self.default_format

    def _find_host_video_file(self):
        """Pick the video file we're going to use, download it if necessary"""
        if self.filename:
            if self.filename[0] in './' or len(self.filename) > 1 and self.filename[1] == ':':
                filepath = os.path.abspath(self.filename)
            else:
                filepath = os.path.join(self.video_directory, self.filename)
            if not os.path.isfile(filepath):
                raise WorkloadError('{} does not exist.'.format(filepath))
            return filepath
        else:
            # Search for files we've already downloaded
            files = []
            for filename in os.listdir(self.video_directory):
                format_ext, format_resolution = self.format.split('_')
                _, file_ext = os.path.splitext(filename)
                if file_ext == '.' + format_ext and format_resolution in filename:
                    files.append(os.path.join(self.video_directory, filename))

            if not files:
                # Download a file with the requested format
                url = DOWNLOAD_URLS[self.format]
                filepath = os.path.join(self.video_directory, os.path.basename(url))
                self.logger.info('Downloading {} to {}...'.format(url, filepath))
                urllib.urlretrieve(url, filepath)
                return filepath
            else:
                if len(files) > 1:
                    self.logger.warn('Multiple files found for {} format. Using {}.'
                                     .format(self.format, files[0]))
                    self.logger.warn('Use "filename"parameter instead of '
                                     '"format" to specify a different file.')
                return files[0]

    def init_resources(self, context):
        # Needs to happen first, as it sets self.format, which is required by
        # _find_host_video_file
        self.validate()

        ensure_directory_exists(self.video_directory)
        self.host_video_file = self._find_host_video_file()

    def setup(self, context):
        super(ExoPlayer, self).setup(context)

        grant_app_permissions(self.target, self.package)

        self.device_video_file = self.target.path.join(self.target.working_directory,
                                                       os.path.basename(self.host_video_file))
        if self.force_dependency_push or not self.target.file_exists(self.device_video_file):
            self.logger.info('Copying {} to device.'.format(self.host_video_file))
            self.target.push(self.host_video_file, self.device_video_file, timeout=120)

        self.play_cmd = 'am start -a {} -d "file://{}"'.format(self.action,
                                                               self.device_video_file)

        self.monitor = self.target.get_logcat_monitor(REGEXPS.values())
        self.monitor.start()

    def run(self, context):
        self.target.execute(self.play_cmd)

        self.monitor.wait_for(REGEXPS['start'])
        self.logger.info('Playing media file')

        line = self.monitor.wait_for(REGEXPS['duration'])[0]
        media_duration_s = int(round(float(re.search(REGEXPS['duration'], line)
                                           .group('duration'))))

        self.logger.info('Media duration is {} seconds'.format(media_duration_s))

        if self.duration > media_duration_s:
            raise ConfigError(
                "'duration' param ({}) longer than media duration ({})".format(
                    self.duration, media_duration_s))

        if self.duration:
            self.logger.info('Waiting {} seconds before ending playback'
                           .format(self.duration))
            time.sleep(self.duration)
        else:
            self.logger.info('Waiting for playback completion ({} seconds)'
                           .format(media_duration_s))
            self.monitor.wait_for(REGEXPS['end'], timeout = media_duration_s + 30)

    def update_output(self, context):
        regex = re.compile(REGEXPS['dropped_frames'])

        dropped_frames = 0
        for line in self.monitor.get_log():
            match = regex.match(line)
            if match:
                dropped_frames += int(match.group('count'))

        context.add_metric('exoplayer_dropped_frames', dropped_frames,
                           lower_is_better=True)

    def teardown(self, context):
        super(ExoPlayer, self).teardown(context)
        self.monitor.stop()
