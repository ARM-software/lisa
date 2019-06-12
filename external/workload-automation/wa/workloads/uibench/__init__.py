#    Copyright 2013-2019 ARM Limited
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

from wa import Parameter, ApkWorkload


class Uibench(ApkWorkload):

    name = 'uibench'
    description = """
        Runs a particular activity of the UIBench_ workload suite. The suite
        is provided by Google as a testbench for the Android UI.

        .. _UIBench: https://android.googlesource.com/platform/frameworks/base/+/refs/heads/master/tests/UiBench/
    """
    package_names = ['com.android.test.uibench']
    loading_time = 1

    parameters = [
        Parameter('activity', kind=str,
                  description="""
                  The UIBench activity to be run. Each activity corresponds to
                  a test. If this parameter is ignored, the application is
                  launched in its main menu. Please note that the available
                  activities vary between versions of UIBench (which follow
                  AOSP versioning) and the availability of the services under
                  test may depend on the version of the target Android. We
                  recommend using the APK of UIBench corresponding to the
                  Android version, enforced through the ``version`` parameter to
                  this workload.
                  """),
        Parameter('duration', kind=int, default=10,
                  description="""
                  As activities do not finish, this workload will terminate
                  UIBench after the given duration.
                  """),
    ]

    def run(self, context):
        super(Uibench, self).run(context)
        self.target.sleep(self.duration)
