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


import os
import re
import time

from wlauto import AndroidUiAutoBenchmark


class Smartbench(AndroidUiAutoBenchmark):

    name = 'smartbench'
    description = """
    Smartbench is a multi-core friendly benchmark application that measures the
    overall performance of an android device. It reports both Productivity and
    Gaming Index.

    https://play.google.com/store/apps/details?id=com.smartbench.twelve&hl=en

    From the website:

    It will be better prepared for the quad-core world. Unfortunately this also
    means it will run slower on older devices. It will also run slower on
    high-resolution tablet devices. All 3D tests are now rendered in full native
    resolutions so naturally it will stress hardware harder on these devices.
    This also applies to higher resolution hand-held devices.
    """
    package = 'com.smartbench.twelve'
    activity = '.Smartbench2012'
    summary_metrics = ['Smartbench: valueGame', 'Smartbench: valueProd']
    run_timeout = 10 * 60

    prod_regex = re.compile('valueProd=(\d+)')
    game_regex = re.compile('valueGame=(\d+)')

    def update_result(self, context):
        super(Smartbench, self).update_result(context)
        with open(self.logcat_log) as fh:
            text = fh.read()
            match = self.prod_regex.search(text)
            prod = int(match.group(1))
            match = self.game_regex.search(text)
            game = int(match.group(1))
            context.result.add_metric('Smartbench: valueProd', prod)
            context.result.add_metric('Smartbench: valueGame', game)
