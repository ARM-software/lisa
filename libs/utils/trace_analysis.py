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

import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pylab as pl
import re
import sys
import trappy
import operator
from trappy.utils import listify
from devlib.utils.misc import memoized
from collections import namedtuple

# Configure logging
import logging

NON_IDLE_STATE = 4294967295

ResidencyTime = namedtuple('ResidencyTime', ['total', 'active'])
ResidencyData = namedtuple('ResidencyData', ['label', 'residency'])

class TraceAnalysis(object):

    def __init__(self, trace, tasks=None, plotsdir=None, prefix=''):
        """
        Support for plotting a standard set of trace singals and events
        """

        self.trace = trace
        self.tasks = tasks
        self.plotsdir = plotsdir
        self.prefix = prefix

        # Keep track of the Trace::platform
        self.platform = trace.platform

        # Plotsdir is byb default the trace dir
        if self.plotsdir is None:
            self.plotsdir = self.trace.data_dir

        # Minimum and Maximum x_time to use for all plots
        self.x_min = 0
        self.x_max = self.trace.time_range

        # Reset x axis time range to full scale
        t_min = self.trace.window[0]
        t_max = self.trace.window[1]
        self.setXTimeRange(t_min, t_max)

    def setXTimeRange(self, t_min=None, t_max=None):
        if t_min is None:
            self.x_min = 0
        else:
            self.x_min = t_min
        if t_max is None:
            self.x_max = self.trace.time_range
        else:
            self.x_max = t_max
        logging.info('Set plots time range to (%.6f, %.6f)[s]',
                self.x_min, self.x_max)

    def __addCapacityColum(self):
        df = self.trace.df('cpu_capacity')
        # Rename CPU and Capacity columns
        df.rename(columns={'cpu_id':'cpu'}, inplace=True)
        # Add column with LITTLE and big CPUs max capacities
        nrg_model = self.platform['nrg_model']
        max_lcap = nrg_model['little']['cpu']['cap_max']
        max_bcap = nrg_model['big']['cpu']['cap_max']
        df['max_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [max_lcap], max_bcap)
        # Add LITTLE and big CPUs "tipping point" threshold
        tip_lcap = 0.8 * max_lcap
        tip_bcap = 0.8 * max_bcap
        df['tip_capacity'] = np.select(
                [df.cpu.isin(self.platform['clusters']['little'])],
                [tip_lcap], tip_bcap)

