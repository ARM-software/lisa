
# import glob
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# import pylab as pl
# import re
# import sys
# import trappy

# Configure logging
import logging

class Filters(object):

    def __init__(self, trace, tasks=None):
        self.trace = trace
        self.tasks = tasks

    def topBigTasks(self, max_tasks=10, min_samples=100, min_utilization=None):
        """
        Tasks which had a 'utilization' bigger than the specified threshold
        """

        if min_utilization is None:
            min_utilization = self.trace.platform['nrg_model']['little']['cpu']['cap_max']

        df = self.trace.df('tload')
        big_tasks_events = df[df.utilization > min_utilization]
        big_tasks = big_tasks_events.pid.unique()

        big_tasks_count = big_tasks.size
        print 'Total {} tasks with at least {} "utilization" samples > {}'\
                .format(big_tasks_count, min_samples, min_utilization)

        big_tasks_stats = big_tasks_events.groupby('pid')\
                            .describe(include=['object']);
        big_tasks_pids = big_tasks_stats.unstack()['comm']\
                            .sort(columns=['count'], ascending=False)
        big_tasks_pids = big_tasks_pids[big_tasks_pids['count'] > min_samples]

        big_topmost = big_tasks_pids.head(max_tasks)
        print 'Top {} "big" tasks:'.format(max_tasks)
        print big_topmost

        return list(big_topmost.index)

