#    Copyright 2013-2018 ARM Limited
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

# pylint: disable=E1101,W0201

import os
import re

from wa import Workload, Parameter, File
from wa.framework.exception import WorkloadError, TargetError
from wa.utils.exec_control import once


class Recentfling(Workload):

    name = 'recentfling'
    description = """
    Tests UI jank on android devices.

    For this workload to work, ``recentfling.sh`` and ``defs.sh`` must be placed
    in ``~/.workload_automation/dependencies/recentfling/``. These can be found
    in the `AOSP Git repository <https://android.googlesource.com/platform/system/extras/+/master/tests/workloads>`_.

    To change the apps that are opened at the start of the workload you will need
    to modify the ``defs.sh`` file. You will need to add your app to ``dfltAppList``
    and then add a variable called ``{app_name}Activity`` with the name of the
    activity to launch (where ``{add_name}`` is the name you put into ``dfltAppList``).

    You can get a list of activities available on your device by running
    ``adb shell pm list packages -f``
    """
    supported_platforms = ['android']

    parameters = [
        Parameter('loops', kind=int, default=3,
                  description="The number of test iterations."),
        Parameter('start_apps', kind=bool, default=True,
                  description="""
                  If set to ``False``,no apps will be started before flinging
                  through the recent apps list (in which the assumption is
                  there are already recently started apps in the list.
                  """),
        Parameter('device_name', kind=str, default=None,
                  description="""
                  If set, recentfling will use the fling parameters for this
                  device instead of automatically guessing the device.  This can
                  also be used if the device is not supported by recentfling,
                  but its screensize is similar to that of one that is supported.

                  For possible values, check your recentfling.sh.  At the time
                  of writing, valid values are: 'shamu', 'hammerhead', 'angler',
                  'ariel', 'mtp8996', 'bullhead' or 'volantis'.
                  """),
    ]

    @once
    def initialize(self, context):  # pylint: disable=no-self-use
        if self.target.get_sdk_version() < 23:
            raise WorkloadError("This workload relies on ``dumpsys gfxinfo`` \
                                 only present in Android M and onwards")

        defs_host = context.get_resource(File(self, "defs.sh"))
        Recentfling.defs_target = self.target.install(defs_host)
        recentfling_host = context.get_resource(File(self, "recentfling.sh"))
        Recentfling.recentfling_target = self.target.install(recentfling_host)

    def setup(self, context):
        args = '-i {} '.format(self.loops)
        if not self.start_apps:
            args += '-N '
        if self.device_name is not None:
            args += '-d {}'.format(self.device_name)

        self.cmd = "echo $$>{workdir}/pidfile; cd {bindir}; exec ./recentfling.sh {args}; rm {workdir}/pidfile".format(
            workdir=self.target.working_directory,
            bindir=self.target.executables_directory, args=args)

        self._kill_recentfling()
        self.target.ensure_screen_is_on()

    def run(self, context):
        self.output = ""
        try:
            self.output = self.target.execute(self.cmd, timeout=120)
        except KeyboardInterrupt:
            self._kill_recentfling()
            raise

    def update_output(self, context):
        loop_group_names = ["90th Percentile", "95th Percentile", "99th Percentile", "Jank", "Jank%"]
        count = 0
        p = re.compile("Frames: \d+ latency: (?P<pct90>\d+)/(?P<pct95>\d+)/(?P<pct99>\d+) Janks: (?P<jank>\d+)\((?P<jank_pct>\d+)%\)")
        for line in self.output.strip().splitlines():
            match = p.search(line)
            if match:
                if line.startswith("AVE: "):
                    group_names = ["Average " + g for g in loop_group_names]
                    classifiers = {"loop": "Average"}
                else:
                    count += 1
                    group_names = loop_group_names
                    classifiers = {"loop": count}

                for (name, metric) in zip(group_names, match.groups()):
                    context.add_metric(name, metric,
                                       classifiers=classifiers)

    @once
    def finalize(self, context):
        self.target.uninstall_executable(self.recentfling_target)
        self.target.uninstall_executable(self.defs_target)

    def _kill_recentfling(self):
        command = 'cat {}/pidfile'.format(self.target.working_directory)
        try:
            pid = self.target.execute(command)
            if pid.strip():
                self.target.kill(pid.strip(), signal='SIGKILL')
        except TargetError:
            # recentfling is not running
            pass
