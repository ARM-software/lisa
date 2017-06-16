#!/usr/bin/env python
#    Copyright 2017 ARM Limited, Google and contributors
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
from time import sleep
from devlib.exception import TargetError
from analysis_tool import AnalysisTool

class AndroidGfx(AnalysisTool):

    def get_frame_drops(self, df):
        if not 'func' in df.columns:
            return 0
        return len(df[(df.func == 'FrameMissed') & (df.data == '1')])

    def run_analysis(self):
        df = self.trace.data_frame.trace_event('tracing_mark_write')
        print {'frame_drops': self.get_frame_drops(df)}

# Run the analysis and provide results
if __name__ == "__main__":
    AndroidGfx().run_main()
