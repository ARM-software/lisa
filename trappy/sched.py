#    Copyright 2015-2015 ARM Limited
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


"""Definitions of scheduler events registered by the Run class"""

from trappy.base import Base
from trappy.dynamic import register_dynamic
from trappy.run import Run

class SchedLoadAvgSchedGroup(Base):
    """Corresponds to Linux kernel trace event sched_load_avg_sched_group"""

    unique_word = "sched_load_avg_sg:"
    """The unique word that will be matched in a trace line"""

    name = "sched_load_avg_sched_group"
    """The name of the :mod:`pandas.DataFrame` member that will be created in a
    :mod:`trappy.run.Run` object"""

    _cpu_mask_column = "cpus"

    def __init__(self):
        super(SchedLoadAvgSchedGroup, self).__init__(
            unique_word=self.unique_word,
        )

    def finalize_object(self):
        """This condition is necessary to force column 'cpus' to be printed
        as 8 digits w/ leading 0
        """
        if self._cpu_mask_column in self.data_frame.columns:
            dfr = self.data_frame[self._cpu_mask_column].apply('{:0>8}'.format)
            self.data_frame[self._cpu_mask_column] = dfr

Run.register_class(SchedLoadAvgSchedGroup, "sched")

class SchedLoadAvgTask(Base):
    """Corresponds to Linux kernel trace event sched_load_avg_task"""

    unique_word = "sched_load_avg_task:"
    """The unique word that will be matched in a trace line"""

    name = "sched_load_avg_task"
    """The name of the :mod:`pandas.DataFrame` member that will be created in a
    :mod:`trappy.run.Run` object"""

    def __init__(self):
        super(SchedLoadAvgTask, self).__init__(
            unique_word=self.unique_word,
        )

    def get_pids(self, key=""):
        """Returns a list of (comm, pid) that contain
        'key' in their 'comm'."""
        dfr = self.data_frame.drop_duplicates(subset=['comm', 'pid'])
        dfr = dfr.ix[:, ['comm', 'pid']]

        return dfr[dfr['comm'].str.contains(key)].values.tolist()

Run.register_class(SchedLoadAvgTask, "sched")

# pylint doesn't like globals that are not ALL_CAPS
# pylint: disable=invalid-name
SchedLoadAvgCpu = register_dynamic("SchedLoadAvgCpu",
                                   "sched_load_avg_cpu:",
                                   "sched")
"""Load and Utilization Signals for CPUs"""

SchedContribScaleFactor = register_dynamic("SchedContribScaleFactor",
                                           "sched_contrib_scale_f:",
                                           "sched")
"""Event to register tracing of contrib factor"""

SchedCpuCapacity = register_dynamic("SchedCpuCapacity",
                                    "sched_cpu_capacity:",
                                    "sched")
"""Event to register tracing of CPU capacities"""

SchedSwitch = register_dynamic("SchedSwitch",
                               "sched_switch",
                               "sched",
                               parse_raw=True)
"""Register SchedSwitch Event"""
# pylint: enable=invalid-name

class SchedCpuFrequency(Base):
    """Corresponds to Linux kernel trace event power/cpu_frequency"""

    unique_word = "cpu_frequency:"
    """The unique word that will be matched in a trace line"""

    name = "sched_cpu_frequency"
    """The name of the :mod:`pandas.DataFrame` member that will be created in a
    :mod:`trappy.run.Run` object"""

    def __init__(self):
        super(SchedCpuFrequency, self).__init__(
            unique_word=self.unique_word,
        )

    def finalize_object(self):
        """This renaming is necessary because our cpu related pivot is 'cpu'
        and not 'cpu_id'. Otherwise you cannot 'mix and match' with other
        classes
        """
        self.data_frame.rename(columns={'cpu_id':'cpu'}, inplace=True)

Run.register_class(SchedCpuFrequency, "sched")
