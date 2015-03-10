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

from wlauto import GameWorkload, Parameter


class EpicCitadel(GameWorkload):

    name = 'citadel'
    description = """
    Epic Citadel demo showcasing Unreal Engine 3.

    The game has very rich graphics details. The workload only moves around its
    environment for the specified time.

    """
    package = 'com.epicgames.EpicCitadel'
    activity = '.UE3JavaApp'
    install_timeout = 120

    parameters = [
        Parameter('duration', kind=int, default=60,
                  description=('Duration, in seconds, of the run (may need to be adjusted for '
                               'different devices.')),
    ]

    def run(self, context):
        super(EpicCitadel, self).run(context)
        time.sleep(self.duration)
