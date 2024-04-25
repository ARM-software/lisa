#    Copyright 2017-2018 ARM Limited
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

import re
import sys
import os.path
from collections import defaultdict

from devlib.exception import TargetStableError, HostError
from devlib.module import Module
from devlib.platform.gem5 import Gem5SimulationPlatform
from devlib.utils.gem5 import iter_statistics_dump, GEM5STATS_ROI_NUMBER


class Gem5ROI:
    def __init__(self, number, target):
        self.target = target
        self.number = number
        self.running = False
        self.field = 'ROI::{}'.format(number)

    def start(self):
        if self.running:
            return False
        self.target.execute('m5 roistart {}'.format(self.number))
        self.running = True
        return True

    def stop(self):
        if not self.running:
            return False
        self.target.execute('m5 roiend {}'.format(self.number))
        self.running = False
        return True

class Gem5StatsModule(Module):
    '''
    Module controlling Region of Interest (ROIs) markers, satistics dump
    frequency and parsing statistics log file when using gem5 platforms.

    ROIs are identified by user-defined labels and need to be booked prior to
    use. The translation of labels into gem5 ROI numbers will be performed
    internally in order to avoid conflicts between multiple clients.
    '''
    name = 'gem5stats'

    @staticmethod
    def probe(target):
        return isinstance(target.platform, Gem5SimulationPlatform)

    def __init__(self, target):
        super(Gem5StatsModule, self).__init__(target)
        self._current_origin = 0
        self._stats_file_path = os.path.join(target.platform.gem5_out_dir,
                                            'stats.txt')
        self.rois = {}
        self._dump_pos_cache = {0: 0}

    def book_roi(self, label):
        if label in self.rois:
            raise KeyError('ROI label {} already used'.format(label))
        if len(self.rois) >= GEM5STATS_ROI_NUMBER:
            raise RuntimeError('Too many ROIs reserved')
        all_rois = set(range(GEM5STATS_ROI_NUMBER))
        used_rois = set([roi.number for roi in self.rois.values()])
        avail_rois = all_rois - used_rois
        self.rois[label] = Gem5ROI(list(avail_rois)[0], self.target)

    def free_roi(self, label):
        if label not in self.rois:
            raise KeyError('ROI label {} not reserved yet'.format(label))
        self.rois[label].stop()
        del self.rois[label]

    def roi_start(self, label):
        if label not in self.rois:
            raise KeyError('Incorrect ROI label: {}'.format(label))
        if not self.rois[label].start():
            raise TargetStableError('ROI {} was already running'.format(label))

    def roi_end(self, label):
        if label not in self.rois:
            raise KeyError('Incorrect ROI label: {}'.format(label))
        if not self.rois[label].stop():
            raise TargetStableError('ROI {} was not running'.format(label))

    def start_periodic_dump(self, delay_ns=0, period_ns=10000000):
        # Default period is 10ms because it's roughly what's needed to have
        # accurate power estimations
        if delay_ns < 0 or period_ns < 0:
            msg = 'Delay ({}) and period ({}) for periodic dumps must be positive'
            raise ValueError(msg.format(delay_ns, period_ns))
        self.target.execute('m5 dumpresetstats {} {}'.format(delay_ns, period_ns))

    def match(self, keys, rois_labels, base_dump=0):
        '''
        Extract specific values from the statistics log file of gem5

        :param keys: a list of key name or regular expression patterns that
            will be matched in the fields of the statistics file. ``match()``
            returns only the values of fields matching at least one these
            keys.
        :type keys: list

        :param rois_labels: list of ROIs labels. ``match()`` returns the
            values of the specified fields only during dumps spanned by at
            least one of these ROIs.
        :type rois_label: list

        :param base_dump: dump number from which ``match()`` should operate. By
            specifying a non-zero dump number, one can virtually truncate
            the head of the stats file and ignore all dumps before a specific
            instant. The value of ``base_dump`` will typically (but not
            necessarily) be the result of a previous call to ``next_dump_no``.
            Default value is 0.
        :type base_dump: int

        :returns: a dict indexed by key parameters containing a dict indexed by
        ROI labels containing an in-order list of records for the key under
        consideration during the active intervals of the ROI.

        Example of return value:
         * Result of match(['sim_'],['roi_1']):
            {
                'sim_inst':
                {
                    'roi_1': [265300176, 267975881]
                }
                'sim_ops':
                {
                    'roi_1': [324395787, 327699419]
                }
                'sim_seconds':
                {
                    'roi_1': [0.199960, 0.199897]
                }
                'sim_freq':
                {
                    'roi_1': [1000000000000, 1000000000000]
                }
                'sim_ticks':
                {
                    'roi_1': [199960234227, 199896897330]
                }
            }
        '''
        records = defaultdict(lambda: defaultdict(list))
        for record, active_rois in self.match_iter(keys, rois_labels, base_dump):
            for key in record:
                for roi_label in active_rois:
                    records[key][roi_label].append(record[key])
        return records

    def match_iter(self, keys, rois_labels, base_dump=0):
        '''
        Yield specific values dump-by-dump from the statistics log file of gem5

        :param keys: same as ``match()``
        :param rois_labels: same as ``match()``
        :param base_dump: same as ``match()``
        :returns: a pair containing:
            1. a dict storing the values corresponding to each of the found keys
            2. the list of currently active ROIs among those passed as parameters

        Example of return value:
         * Result of match_iter(['sim_'],['roi_1', 'roi_2']).next()
            (
                {
                    'sim_inst': 265300176,
                    'sim_ops': 324395787,
                    'sim_seconds': 0.199960,
                    'sim_freq': 1000000000000,
                    'sim_ticks': 199960234227,
                },
                [ 'roi_1 ' ]
            )
        '''
        for label in rois_labels:
            if label not in self.rois:
                raise KeyError('Impossible to match ROI label {}'.format(label))
            if self.rois[label].running:
                self.logger.warning('Trying to match records in statistics file'
                        ' while ROI {} is running'.format(label))

        # Construct one large regex that concatenates all keys because
        # matching one large expression is more efficient than several smaller
        all_keys_re = re.compile('|'.join(keys))

        def roi_active(roi_label, dump):
            roi = self.rois[roi_label]
            return (roi.field in dump) and (int(dump[roi.field]) == 1)

        with open(self._stats_file_path, 'r') as stats_file:
            self._goto_dump(stats_file, base_dump)
            for dump in iter_statistics_dump(stats_file):
                active_rois = [l for l in rois_labels if roi_active(l, dump)]
                if active_rois:
                    rec = {k: dump[k] for k in dump if all_keys_re.search(k)}
                    yield (rec, active_rois)

    def next_dump_no(self):
        '''
        Returns the number of the next dump to be written to the stats file.

        For example, if next_dump_no is called while there are 5 (0 to 4) full
        dumps in the stats file, it will return 5. This will be usefull to know
        from which dump one should match() in the future to get only data from
        now on.
        '''
        with open(self._stats_file_path, 'r') as stats_file:
            # _goto_dump reach EOF and returns the total number of dumps + 1
            return self._goto_dump(stats_file, sys.maxsize)

    def _goto_dump(self, stats_file, target_dump):
        if target_dump < 0:
            raise HostError('Cannot go to dump {}'.format(target_dump))

        # Go to required dump quickly if it was visited before
        if target_dump in self._dump_pos_cache:
            stats_file.seek(self._dump_pos_cache[target_dump])
            return target_dump
        # Or start from the closest dump already visited before the required one
        prev_dumps = filter(lambda x: x < target_dump, self._dump_pos_cache.keys())
        curr_dump = max(prev_dumps)
        curr_pos = self._dump_pos_cache[curr_dump]
        stats_file.seek(curr_pos)

        # And iterate until target_dump
        dump_iterator = iter_statistics_dump(stats_file)
        while curr_dump < target_dump:
            try:
                next(dump_iterator)
            except StopIteration:
                break
            # End of passed dump is beginning og next one
            curr_pos = stats_file.tell()
            curr_dump += 1
        self._dump_pos_cache[curr_dump] = curr_pos
        return curr_dump
