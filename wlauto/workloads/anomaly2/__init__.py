#    Copyright 2014-2015 ARM Limited
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
import json

from wlauto.common.android.workload import GameWorkload
from wlauto.exceptions import WorkloadError, DeviceError


class Anomaly2(GameWorkload):

    name = 'anomaly2'
    description = """
    Anomaly 2 game demo and benchmark.

    Plays three scenes from the game, benchmarking each one. Scores reported are intended to
    represent overall perceived quality of the game, based not only on raw FPS but also factors
    like smoothness.

    """
    package = 'com.elevenbitstudios.anomaly2Benchmark'
    activity = 'com.android.Game11Bits.MainActivity'
    loading_time = 30
    asset_file = 'obb:com.elevenbitstudios.anomaly2Benchmark.tar.gz'

    def reset(self, context):
        pass

    def update_result(self, context):
        super(Anomaly2, self).update_result(context)
        sent_blobs = {'data': []}
        with open(self.logcat_log) as fh:
            for line in fh:
                if 'sendHttpRequest: json = ' in line:
                    data = json.loads(line.split('json = ')[1])
                    sent_blobs['data'].append(data)
                    if 'scene' not in data['intValues']:
                        continue
                    scene = data['intValues']['scene']
                    score = data['intValues']['score']
                    fps = data['floatValues']['fps']
                    context.result.add_metric('scene_{}_score'.format(scene), score)
                    context.result.add_metric('scene_{}_fps'.format(scene), fps)
        outfile = os.path.join(context.output_directory, 'anomaly2.json')
        with open(outfile, 'wb') as wfh:
            json.dump(sent_blobs, wfh, indent=4)

    def teardown(self, context):
        self.device.execute('am force-stop {}'.format(self.package))

