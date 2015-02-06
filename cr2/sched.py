#!/usr/bin/python
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

from base import Base

class SchedLoadAvgSchedGroup(Base):
    """Corresponds to Linux kernel trace event sched_load_avg_sched_group"""
    unique_word="sched_load_avg_sg:"
    name="sched_load_avg_sched_group"
    _cpu_mask_column = "cpus"

    def __init__(self, path=None):
        super(SchedLoadAvgSchedGroup, self).__init__(
            basepath=path,
            unique_word=self.unique_word,
        )

    def finalize_object(self):
        """This condition is necessary to force column 'cpus' to be printed
        as 8 digits w/ leading 0
        """
        if self._cpu_mask_column in self.data_frame.columns:
            self.data_frame[self._cpu_mask_column] = self.data_frame[self._cpu_mask_column].apply('{:0>8}'.format)

class SchedLoadAvgTask(Base):
    """Corresponds to Linux kernel trace event sched_load_avg_task"""
    unique_word="sched_load_avg_task:"
    name="sched_load_avg_task"

    def __init__(self, path=None):
        super(SchedLoadAvgTask, self).__init__(
            basepath=path,
            unique_word=self.unique_word,
        )

    def get_pids(self, key=""):
        """Returns a list of (comm, pid) that contain
        'key' in their 'comm'."""
        df = self.data_frame.drop_duplicates(subset=['comm','pid']).ix[:,['comm','pid']]

        return df[df['comm'].str.contains(key)].values.tolist()

class SchedLoadAvgCpu(Base):
    """Corresponds to Linux kernel trace event sched_load_avg_cpu"""
    unique_word="sched_load_avg_cpu:"
    name="sched_load_avg_cpu"

    def __init__(self, path=None):
        super(SchedLoadAvgCpu, self).__init__(
            basepath=path,
            unique_word=self.unique_word,
        )

class SchedContribScaleFactor(Base):
    """Corresponds to Linux kernel trace event sched_contrib_scale_factor"""
    unique_word="sched_contrib_scale_f:"
    name="sched_contrib_scale_factor"

    def __init__(self, path=None):
        super(SchedContribScaleFactor, self).__init__(
            basepath=path,
            unique_word=self.unique_word,
        )

class SchedCpuCapacity(Base):
    """Corresponds to Linux kernel trace event sched_cpu_capacity"""
    unique_word="sched_cpu_capacity:"
    name="sched_cpu_capacity"

    def __init__(self, path=None):
        super(SchedCpuCapacity, self).__init__(
            basepath=path,
            unique_word=self.unique_word,
        )

class SchedCpuFrequency(Base):
    """Corresponds to Linux kernel trace event power/cpu_frequency"""
    unique_word="cpu_frequency:"
    name="sched_cpu_frequency"

    def __init__(self, path=None):
        super(SchedCpuFrequency, self).__init__(
            basepath=path,
            unique_word=self.unique_word,
        )

    def finalize_object(self):
        """This renaming is necessary because our cpu related pivot is 'cpu'
        and not 'cpu_id'. Otherwise you cannot 'mix and match' with other
        classes
        """
        self.data_frame.rename(columns={'cpu_id':'cpu'}, inplace=True)
