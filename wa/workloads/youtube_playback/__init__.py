#    Copyright 2017 ARM Limited
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

import time

from wa import Parameter, ApkWorkload


class YoutubePlayback(ApkWorkload):
    """
    Simple Youtube video playback

    This triggers a video streaming playback on Youtube. Unlike the more
    featureful "youtube" workload, this performs no other action that starting
    the video via an intent and then waiting for a certain amount of playback
    time. This is therefore only useful when you are confident that the content
    on the end of the provided URL is stable - that means the video should have
    no advertisements attached.
    """
    name = 'youtube_playback'

    package_names = ['com.google.android.youtube']
    action = 'android.intent.action.VIEW'

    parameters = [
        Parameter('video_url', default='https://www.youtube.com/watch?v=YE7VzlLtp-4',
                  description='URL of video to play'),
        Parameter('duration', kind=int, default=20,
                  description='Number of seconds of video to play'),
    ]

    def setup(self, context):
        super(YoutubePlayback, self).setup(context)

        self.command = 'am start -a {} {}'.format(self.action, self.video_url)

    def run(self, context):
        self.target.execute(self.command)

        time.sleep(self.duration)
