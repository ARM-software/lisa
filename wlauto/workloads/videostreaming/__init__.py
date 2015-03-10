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

from wlauto import AndroidUiAutoBenchmark, Parameter
import wlauto.common.android.resources


class Videostreaming(AndroidUiAutoBenchmark):
    name = 'videostreaming'
    description = """
    Uses the FREEdi video player to search, stream and play the specified
    video content from YouTube.

    """
    name = 'videostreaming'
    package = 'tw.com.freedi.youtube.player'
    activity = '.MainActivity'

    parameters = [
        Parameter('video_name', kind=str,
                  description='Name of the video to be played.'),
        Parameter('resolution', kind=str, default='320p', allowed_values=['320p', '720p', '1080p'],
                  description='Resolution of the video to be played. If video_name is set'
                  'this setting will be ignored'),
        Parameter('sampling_interval', kind=int, default=20,
                  description="""
                  Time interval, in seconds, after which the status of the video playback to
                  be monitoreThe elapsed time of the video playback is
                  monitored after after every ``sampling_interval`` seconds and
                  compared against the actual time elapsed and the previous
                  sampling point. If the video elapsed time is less that
                  (sampling time - ``tolerance``) , then the playback is aborted as
                  the video has not been playing continuously.
                  """),
        Parameter('tolerance', kind=int, default=3,
                  description="""
                  Specifies the amount, in seconds, by which sampling time is
                  allowed to deviate from elapsed video playback time. If the delta
                  is greater than this value (which could happen due to poor network
                  connection), workload result will be invalidated.
                  """),
        Parameter('run_timeout', kind=int, default=200,
                  description='The duration in second for which to play the video'),
    ]

    def init_resources(self, context):
        self.uiauto_params['tolerance'] = self.tolerance
        self.uiauto_params['sampling_interval'] = self.sampling_interval
        if self.video_name and self.video_name != "":
            self.uiauto_params['video_name'] = self.video_name.replace(" ", "0space0")  # hack to get around uiautomator limitation
        else:
            self.uiauto_params['video_name'] = "abkk sathe {}".format(self.resolution).replace(" ", "0space0")
        self.apk_file = context.resolver.get(wlauto.common.android.resources.ApkFile(self))
        self.uiauto_file = context.resolver.get(wlauto.common.android.resources.JarFile(self))
        self.device_uiauto_file = self.device.path.join(self.device.working_directory,
                                                        os.path.basename(self.uiauto_file))
        if not self.uiauto_package:
            self.uiauto_package = os.path.splitext(os.path.basename(self.uiauto_file))[0]
