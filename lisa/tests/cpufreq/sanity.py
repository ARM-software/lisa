# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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
import os

from lisa.tests.base import Result, ResultBundle, TestBundle
from lisa.wlgen.sysbench import Sysbench
from lisa.target import Target
from lisa.utils import ArtifactPath, groupby, nullcontext
from lisa.analysis.tasks import TasksAnalysis

class UserspaceSanityItem(TestBundle):
    """
    Record the number of sysbench events on a given CPU at a given frequency.
    """

    def __init__(self, res_dir, plat_info, cpu, freq, work):
        super().__init__(res_dir, plat_info)

        self.cpu = cpu
        self.freq = freq
        self.work = work

    @classmethod
    def _from_target(cls, target:Target, *, res_dir:ArtifactPath, cpu, freq, switch_governor=True,) -> 'UserspaceSanityItem':
        """
        Create a :class:`UserspaceSanityItem` from a live :class:`lisa.target.Target`.

        :param cpu: CPU to run on.
        :type cpu: int

        :param freq: Frequency to run at.
        :type freq: int

        :param switch_governor: Switch the governor to userspace, and undo it at the end.
            If that has been done in advance, not doing it for every item saves substantial time.
        :type switch_governor: bool
        """


        sysbench = Sysbench(target, "sysbench", res_dir)

        cm = target.cpufreq.use_governor('userspace') if switch_governor else nullcontext()
        with cm:
            target.cpufreq.set_frequency(cpu, freq)
            sysbench.run(cpus=[cpu], max_duration_s=1)

        work = sysbench.output.nr_events
        return cls(res_dir, target.plat_info, cpu, freq, work)


class UserspaceSanity(TestBundle):
    """
    A class for making sure the userspace governor behaves sanely

    :param sanity_items: A list of :class:`UserspaceSanityItem`.
    :type sanity_items: list(UserspaceSanityItem)
    """

    def __init__(self, res_dir, plat_info, sanity_items):
        super().__init__(res_dir, plat_info)

        self.sanity_items = sanity_items

    @classmethod
    def _from_target(cls, target:Target, *, res_dir:ArtifactPath=None,
                     freq_count_limit=5) -> 'UserspaceSanity':
        """
        Factory method to create a bundle using a live target

        :param freq_count_limit: The maximum amount of frequencies to test
        :type freq_count_limit: int

        This will run Sysbench at different frequencies using the userspace
        governor
        """
        sanity_items = []

        plat_info = target.plat_info
        with target.cpufreq.use_governor("userspace"):
            for domain in plat_info['freq-domains']:
                cpu = domain[0]
                freqs = plat_info['freqs'][cpu]

                if len(freqs) > freq_count_limit:
                    freqs = freqs[::len(freqs) // freq_count_limit +
                                    (1 if len(freqs) % 2 else 0)]

                for freq in freqs:
                    item_res_dir = ArtifactPath.join(res_dir, 'CPU{}@{}'.format(cpu, freq))
                    os.makedirs(item_res_dir)
                    item = UserspaceSanityItem.from_target(
                        target=target,
                        cpu=cpu,
                        freq=freq,
                        res_dir=item_res_dir,
                        # We already did that once and for all, so that we
                        # don't spend too much time endlessly switching back
                        # and forth between governors
                        switch_governor=False,
                    )
                    sanity_items.append(item)

        return cls(res_dir, plat_info, sanity_items)

    def test_performance_sanity(self) -> ResultBundle:
        """
        Assert that higher CPU frequency leads to more work done
        """
        res = ResultBundle.from_bool(True)

        cpu_items = {
            cpu: {
                # We expect only one item per frequency
                item.freq: item
                for item in freq_items
            }
            for cpu, freq_items in groupby(self.sanity_items, key=lambda item: item.cpu)
        }

        failed = []
        passed = True
        for cpu, freq_items in cpu_items.items():
            sorted_items = sorted(freq_items.values(), key=lambda item: item.freq)
            work = [item.work for item in sorted_items]
            if work != sorted(work):
                passed = False
                failed.append(cpu)

        res = ResultBundle.from_bool(passed)
        work_metric = {
            cpu: {freq: item.work for freq, item in freq_items.items()}
            for cpu, freq_items in cpu_items.items()
        }
        res.add_metric('CPUs work', work_metric)
        res.add_metric('Failed CPUs', failed)

        return res

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
