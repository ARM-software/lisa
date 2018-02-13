# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

""" SchedTune Analysis Module """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from analysis_module import AnalysisModule
from devlib.utils.misc import memoized
from trappy.utils import listify

# Configure logging
import logging


class STuneAnalysis(AnalysisModule):
    """
    Support for SchedTune signals anaysis

    :param trace: input Trace object
    :type trace: :mod:`libs.utils.Trace`
    """

    def __init__(self, trace):
        super(STuneAnalysis, self).__init__(trace)

###############################################################################
# DataFrame Getter Methods
###############################################################################

    @memoized
    def _dfg_tasks_updates(self, cpus=None, cgroups=None):
        if not self._trace.hasEvents('sched_tune_tasks_update'):
            logging.warn('Events [sched_tune_tasks_updates] not found, '
                         'cannot get schedtune tasks updates!')
            return None

        # Get SchedTune Accouting data
        df = self._dfg_trace_event('sched_tune_tasks_update')

        # Get number of CPUs to plot
        if cpus:
            # Filter on requested CPUs
            cpus = listify(cpus)
            df = df[df.cpu.isin(cpus)]

        # Get number of CGROUPs to plot
        if cgroups:
            # Filter on requested CGROUPs
            cgroups = listify(cgroups)
            df = df[df.idx.isin(cgroups)]

        return df


###############################################################################
# Plotting Methods
###############################################################################

    def plotTasksUpdate(self, cpus=None, cgroups=None, xlim=(0,None)):
        """ Plot SchedTune accouting on specified cpus/cgroups
        """

        if not self._trace.hasEvents('sched_tune_tasks_update'):
            logging.warn('Event [sched_tune_tasks_update] not found, plot DISABLED!')
            return
        df = self._dfg_tasks_updates(cpus, cgroups)
        if df is None:
            return

        # Get number of CPUs to plot
        cpus = df.cpu.unique()
        cpus_cnt = cpus.size
        cpus = sorted(cpus.tolist())

        # Ger number of CGROUPS to plot
        cgroups = df.idx.unique()
        cgroups_cnt = cgroups.size
        cgroups = sorted(cgroups.tolist())

        # Get maximum number of RUNNABLE tasks in a single RQ
        ymax = df.tasks.max() + 1

        logging.info("Plotting SchedTune Accounting [%d CGroups, %d CPUs]",
                     cgroups_cnt, cpus_cnt)
        gs = gridspec.GridSpec(cpus_cnt, cgroups_cnt);
        fig = plt.figure(figsize=(6*cgroups_cnt, 3*cpus_cnt));

        for cpu_idx, cpu in enumerate(cpus):
            for cg_idx, cg in enumerate(cgroups):
                axes = plt.subplot(gs[cpu_idx, cg_idx])

                # Get accouting data for current CPU and CGroup
                data = df[(df.cpu == cpu) & (df.idx == cg)][['tasks']]
                if not data.empty:
                    data.plot(ax=axes, drawstyle='steps-post', xlim=xlim, ylim=(0, ymax))

                axes.set_title("CPU_{:02d}, CGroup_{:02d}".format(cpu, cg))
                axes.grid(True)
                if cpu_idx < cpus_cnt-1:
                    axes.set_xlabel('')
                    axes.set_xticklabels([])

                # Set Y ticks to be integer
                axes.yaxis.set_ticks(np.arange(0, ymax, 1))

        logging.info("Detected maximum number of RUNNABLE tasks: %d", ymax-1)
        if ymax > 100:
            logging.warn("Suspecious number of RUNNABLE tasks, ")
            logging.warn("Check for possible core accouting issue!")

# vim :set tabstop=4 shiftwidth=4 expandtab
