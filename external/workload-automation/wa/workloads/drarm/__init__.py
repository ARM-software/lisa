#    Copyright 2023 ARM Limited
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
import pandas as pd
from wa import ApkWorkload, Parameter
from devlib.exception import TargetStableCalledProcessError


class DrArm(ApkWorkload):

    name = 'drarm'
    package_names = ['com.Arm.DrArm']
    activity = 'com.unity3d.player.UnityPlayerActivity'
    view = "SurfaceView[com.Arm.DrArm/com.unity3d.player.UnityPlayerActivity](BLAST)"
    install_timeout = 200
    description = """
    Dr. Arm’s Amazing Adventures is a “Souls-Like” Mobile Action Role Playing Game developed at Arm.
    """

    parameters = [
        Parameter('timeout', kind=int, default=126,
                  description='The amount of time the game should run for'),
        Parameter('auto_demo', kind=bool, default=False,
                  description='Start the demo automatically'),
        Parameter('show_fps', kind=bool, default=False,
                  description='Show the FPS count window in-game'),
        Parameter('adpf', kind=bool, default=True,
                  description='Enable ADPF'),
        Parameter('adpf_auto', kind=bool, default=True,
                  description='Enable automatic ADPF mode'),
        Parameter('adpf_logging', kind=bool, default=False,
                  description='Enable ADPF logging'),
        Parameter('verbose_log', kind=bool, default=False,
                  description='Emit reported stats as debug logs'),
        Parameter('adpf_interventions', kind=bool, default=True,
                  description='Enable ADPF interventions'),
        Parameter('target_vsyncs', kind=int, default=1,
                  description='the number of vsyncs to target to a frame (1 = current display rate)'),
        Parameter('target_framerate', kind=int, default=None,
                  description='Target framerate for the application'),
        Parameter('fps_report_file', kind=str, default=None,
                  description='File name that the ADPF FPS report should use.'),
        Parameter('fixed_time_step', kind=float, default=None,
                  description='Time, in seconds, that should be used in advancing the simulation'),
    ]

    @property
    def apk_arguments(self):
        args = {
            'showFPS': int(self.show_fps),
            'doAdpf': int(self.adpf),
            'adpfMode': int(self.adpf_auto),
            'adpfLogging': int(self.adpf_logging),
            'verboseLog': int(self.verbose_log),
            'adpfInterventions': int(self.adpf_interventions),
            'targetVsyncs': self.target_vsyncs,
            'autoDemo': int(self.auto_demo),
        }

        if self.target_framerate is not None:
            args['targetFramerate'] = self.target_framerate

        if self.fixed_time_step is not None:
            args['fixedTimeStep'] = self.fixed_time_step

        if self.fps_report_file is not None:
            args['fpsReportFileName'] = self.fps_report_file

        return args

    def run(self, context):
        self.target.sleep(self.timeout)

    def update_output(self, context):
        super(DrArm, self).update_output(context)
        outfile_glob = self.target.path.join(
            self.target.external_storage_app_dir, self.apk.package, 'files', '*.csv'
        )

        try:
            ls_output = self.target.execute('ls {}'.format(outfile_glob))
        except TargetStableCalledProcessError:
            self.logger.warning('Failed to find the ADPF report file.')
            return

        on_target_output_files = [f.strip() for f in ls_output.split('\n') if f]

        self.logger.info('Extracting the ADPF FPS report from target...')
        for file in on_target_output_files:
            host_output_file = os.path.join(context.output_directory, os.path.basename(file))
            self.target.pull(file, host_output_file)
            context.add_artifact('adpf', host_output_file, kind='data',
                                 description='ADPF report log in CSV format.')

            adpf_df = pd.read_csv(host_output_file)
            if not adpf_df.empty:
                context.add_metric('Average FPS', round(adpf_df['average fps'].mean(), 2))
                context.add_metric('Frame count', int(adpf_df['# frame count'].iloc[-1]))
