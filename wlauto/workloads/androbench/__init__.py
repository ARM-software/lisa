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

from wlauto import AndroidUiAutoBenchmark


class Androbench(AndroidUiAutoBenchmark):
    name = 'androbench'
    description = """Androbench measures the storage performance of device"""
    package = 'com.andromeda.androbench2'
    activity = '.main'
    run_timeout = 10 * 60

    def update_result(self, context):
        super(Androbench, self).update_result(context)
        db = '/data/data/com.andromeda.androbench2/databases/history.db'
        qs = 'select * from history'
        res = 'results.raw'
        os.system('adb shell sqlite3 %s "%s" > %s' % (db, qs, res))
        fhresults = open("results.raw", "rb")
        results = fhresults.readlines()[0].split('|')
        context.result.add_metric('Sequential Read ', results[8], 'MB/s')
        context.result.add_metric('Sequential Write ', results[9], 'MB/s')
        context.result.add_metric('Random Read ', results[10], 'MB/s')
        context.result.add_metric('Random Write ', results[12], 'MB/s')
        os.system('rm results.raw')
