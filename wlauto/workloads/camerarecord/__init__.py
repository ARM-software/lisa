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

from wlauto import UiAutomatorWorkload, Parameter


class Camerarecord(UiAutomatorWorkload):

    name = 'camerarecord'
    description = """
    Uses in-built Android camera app to record the video for given interval
    of time.

    """
    package = 'com.google.android.gallery3d'
    activity = 'com.android.camera.CameraActivity'
    run_timeout = 0

    parameters = [
        Parameter('recording_time', kind=int, default=60,
                  description='The video recording time in seconds.'),
    ]

    def __init__(self, device, **kwargs):
        super(Camerarecord, self).__init__(device)
        self.uiauto_params['recording_time'] = self.recording_time  # pylint: disable=E1101
        self.run_timeout = 3 * self.uiauto_params['recording_time']

    def setup(self, context):
        super(Camerarecord, self).setup(context)
        self.device.execute('am start -n {}/{}'.format(self.package, self.activity))

    def teardown(self, context):
        self.device.execute('am force-stop {}'.format(self.package))
        super(Camerarecord, self).teardown(context)
