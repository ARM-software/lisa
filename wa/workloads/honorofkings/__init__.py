#    Copyright 2025 ARM Limited
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

import os

from wa import ApkReventWorkload, Parameter


class HoK(ApkReventWorkload):
    name = 'honorofkings'
    uninstall = False
    clear_data_on_reset = False  # Don't clear assets on exit
    requires_network = True  # The game requires network connection
    description = (
        'Launch a match replay in Honor of Kings.\n\n'
        'The game must already have a user logged in and the plugins downloaded.'
    )
    package_names = [
        'com.levelinfinite.sgameGlobal',
        'com.tencent.tmgp.sgame',
    ]

    parameters = [
        Parameter(
            'activity',
            kind=str,
            default='.SGameGlobalActivity',
            description='Activity name of Honor of Kings game.',
        ),
        Parameter(
            'replay_file',
            kind=str,
            default='replay.abc',
            description='Honor of Kings Replay file name.',
        ),
    ]

    def setup(self, context):
        upload_dir = self.target.path.join(
            self.target.external_storage_app_dir,
            self.apk.apk_info.package,
            'files',
            'Replay'
        )
        replay_file = os.path.join(self.dependencies_directory, self.replay_file)
        self.logger.debug('Uploading "%s" to "%s"...', replay_file, upload_dir)
        self.target.push(replay_file, upload_dir)

        super().setup(context)
