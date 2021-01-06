# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
import numpy as np

from lisa.analysis.base import TraceAnalysisBase
from lisa.trace import requires_events
from lisa.datautils import df_delta

class EASAnalysis(TraceAnalysisBase):
    """
	Energy Aware Scheduler (EAS) specific analysis.

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """
    name = 'eas'

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @TraceAnalysisBase.cache
    @requires_events('sched_pre_feec', 'sched_post_feec')
    def df_feec_delta(self, tasks=None):
        """
        DataFrame containing ``find_energy_efficient_cpu`` (feec) related
        information.

        :param tasks: Task names or PIDs or ``(pid, comm)`` to look for.
        :type tasks: list(int or str or tuple(int, str))

        :returns: a :class:`pandas.DataFrame` indexed by ``Time`` with:

          - A ``pid`` column.
          - A ``comm`` column.
          - A ``__cpu`` column (the cpu feec was executed on).
          - A ``prev_cpu`` column (the cpu the task was running on).
          - A ``dst_cpu`` column (the cpu selected by feec).
          - A ``delta`` column (duration of feec function call).
        """
        pre_df = self.trace.df_event('sched_pre_feec')
        post_df = self.trace.df_event('sched_post_feec')

        # Filter the tasks.
        if tasks:
            pre_df = df_filter_task_ids(pre_df, tasks)
            post_df = df_filter_task_ids(post_df, tasks)

        pre_df = pre_df.drop(columns=['__comm', '__pid'])
        post_df = post_df.drop(columns=['__comm', '__pid'])

        # Also group by '__cpu': the events must be emitted from the same cpu.
        return df_delta(pre_df, post_df, ['pid', 'comm', '__cpu'])


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
