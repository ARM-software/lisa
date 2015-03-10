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

# pylint: disable=E1101

from wlauto import UiAutomatorWorkload, Parameter


class Cameracapture(UiAutomatorWorkload):

    name = 'cameracapture'
    description = """
    Uses in-built Android camera app to take photos.

    """
    package = 'com.google.android.gallery3d'
    activity = 'com.android.camera.CameraActivity'

    parameters = [
        Parameter('no_of_captures', kind=int, default=5,
                  description='Number of photos to be taken.'),
        Parameter('time_between_captures', kind=int, default=5,
                  description='Time, in seconds, between two consecutive camera clicks.'),
    ]

    def __init__(self, device, **kwargs):
        super(Cameracapture, self).__init__(device, **kwargs)
        self.uiauto_params['no_of_captures'] = self.no_of_captures
        self.uiauto_params['time_between_captures'] = self.time_between_captures

    def setup(self, context):
        super(Cameracapture, self).setup(context)
        self.device.execute('am start -n {}/{}'.format(self.package, self.activity))

    def update_result(self, context):
        pass

    def teardown(self, context):
        super(Cameracapture, self).teardown(context)
