# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        sched.py
# ----------------------------------------------------------------
# $
#

"""Definitions of scheduler events registered by the Run class"""

from cr2.base import Base
from cr2.dynamic import register_dynamic
from cr2.run import Run

class SchedLoadAvgSchedGroup(Base):
    """Corresponds to Linux kernel trace event sched_load_avg_sched_group"""
    unique_word = "sched_load_avg_sg:"
    name = "sched_load_avg_sched_group"
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
    name = "sched_load_avg_task"

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

SchedContribScaleFactor = register_dynamic("SchedContribScaleFactor",
                                           "sched_contrib_scale_f:",
                                           "sched")

SchedCpuCapacity = register_dynamic("SchedCpuCapacity",
                                    "sched_cpu_capacity:",
                                    "sched")

SchedSwitch = register_dynamic("SchedSwitch",
                               "sched_switch",
                               "sched",
                               parse_raw=True)
# pylint: enable=invalid-name

class SchedCpuFrequency(Base):
    """Corresponds to Linux kernel trace event power/cpu_frequency"""
    unique_word = "cpu_frequency:"
    name = "sched_cpu_frequency"

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
