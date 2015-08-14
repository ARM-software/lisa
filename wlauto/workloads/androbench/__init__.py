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

from wlauto import AndroidBenchmark
from uiautomator import Device


class Androbench(AndroidBenchmark):

    name = 'androbench'
    description = """Androbench measures the storage performance of device"""
    package = 'com.andromeda.androbench2'
    activity = '.main'
    device=''

    def setup(self, context):
		global device
		os.system('adb devices > deviceinfo')
		devinf = open('deviceinfo','rb')
		dev = devinf.readlines()[1].split('\t')[0]
		devinf.close()
		device=Device(dev)
		os.system('rm deviceinfo')


    def run(self, context):
		global device,package,activity
		os.system('adb shell pm clear com.andromeda.androbench2')		
		os.system('adb shell am start -n com.andromeda.androbench2/.main')
		while True :
			if device(text="Measure your storage performance").exists :
				time.sleep(1)
				break
				
		if device(textStartsWith="Micro").exists :
			device(textStartsWith="Micro").click()
		if device(text="Yes").exists :
			device(textStartsWith="Yes").click()
		
		
		while True :
			if device(text="Cancel").exists :
				device(text="Cancel").click()
				time.sleep(1)
				break
			

    def update_result(self, context):
        super(Androbench, self).update_result(context)
        os.system('adb shell cp /data/data/com.andromeda.androbench2/databases/history.db /sdcard/results.db')
        os.system('adb pull /sdcard/results.db .')
        os.system('sqlite3 results.db "select * from history" > results.raw')
        fhresults=open("results.raw","rb")
        results=fhresults.readlines()[0].split('|')
        context.result.add_metric('Sequential Read MB/s', results[8])
        context.result.add_metric('Sequential Write MB/s', results[9])
        context.result.add_metric('Random Read MB/s', results[10])
        context.result.add_metric('Random Write MB/s', results[12])
        os.system('rm results.raw')
        
    

    def teardown(self, context):
        pass

    def validate(self):
        pass
