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

from lisa.tests.kernel.test_bundle import Result, ResultBundle, TestBundle
from lisa.wlgen.sysbench import Sysbench
from lisa.env import TestEnv, ArtifactPath

class UserspaceSanity(TestBundle):
    """
    A class for making sure the userspace governor behaves sanely

    :param cpu_work: A description of the amount of work done on one CPU of
      each frequency domain ({cpu : {freq : work}}
    :type cpu_work: dict
    """

    def __init__(self, res_dir, cpu_work):
        super().__init__(res_dir)

        self.cpu_work = cpu_work

    @classmethod
    def _from_testenv(cls, te, res_dir, freq_count_limit):
        cpu_work = {}
        sysbench = Sysbench(te, "sysbench", res_dir)

        with te.target.cpufreq.use_governor("userspace"):
            for domain in te.target.cpufreq.iter_domains():
                cpu = domain[0]
                cpu_work[cpu] = {}
                freqs = te.target.cpufreq.list_frequencies(cpu)

                if len(freqs) > freq_count_limit:
                    freqs = freqs[::len(freqs) // freq_count_limit +
                                  (1 if len(freqs) % 2 else 0)]

                for freq in freqs:
                    te.target.cpufreq.set_frequency(cpu, freq)
                    sysbench.run(cpus=[cpu], max_duration_s=1)
                    cpu_work[cpu][freq] = sysbench.output.nr_events

        return cls(res_dir, cpu_work)

    @classmethod
    def from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None,
                     freq_count_limit=5) -> 'UserspaceSanity':
        """
        Factory method to create a bundle using a live target

        :param freq_count_limit: The maximum amount of frequencies to test
        :type freq_count_limit: int

        This will run Sysbench at different frequencies using the userspace
        governor
        """
        return super().from_testenv(te, res_dir, freq_count_limit=freq_count_limit)

    def test_performance_sanity(self) -> ResultBundle:
        """
        Assert that higher CPU frequency leads to more work done
        """
        res = ResultBundle.from_bool(True)

        for cpu, freq_work in self.cpu_work.items():
            sorted_freqs = sorted(freq_work.keys())
            work = [freq_work[freq] for freq in sorted_freqs]

            if not work == sorted(work):
                res.result = Result.FAILED

            res.add_metric("CPU{} work".format(cpu), freq_work)

        return res

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
