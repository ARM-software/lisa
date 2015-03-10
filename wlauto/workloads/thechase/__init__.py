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
import time

from wlauto import ApkWorkload, Parameter


class TheChase(ApkWorkload):

    name = 'thechase'
    description = """
    The Chase demo showcasing the capabilities of Unity game engine.

    This demo, is a static video-like game demo, that demonstrates advanced features
    of the unity game engine. It loops continuously until terminated.

    """

    package = 'com.unity3d.TheChase'
    activity = 'com.unity3d.player.UnityPlayerNativeActivity'
    install_timeout = 200
    view = 'SurfaceView'

    parameters = [
        Parameter('duration', kind=int, default=70,
                  description=('Duration, in seconds, note that the demo loops the same (roughly) 60 '
                               'second sceene until stopped.')),
    ]

    def run(self, context):
        time.sleep(self.duration)

