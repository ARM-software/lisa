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

import sys

from lisa.kernel_tests.test_bundle import TestMetric, ResultBundle, TestBundle
from lisa.wlgen.sysbench import Sysbench

class CapacitySanityCheck(TestBundle):
    """
    A class for making sure capacity values make sense on a given target

    :param capacity_work: A description of the amount of work done on the
      target, per capacity value ({capacity : work})
    :type capacity_work: dict
    """

    def __init__(self, res_dir, capacity_work):
        super(CapacitySanityCheck, self).__init__(res_dir)

        self.capacity_work = capacity_work

    @classmethod
    def _from_target(cls, te, res_dir):
        with te.target.cpufreq.use_governor("performance"):
            sysbench = Sysbench(te, "sysbench", res_dir)

            cpu_capacities = te.target.sched.get_capacities()
            capa_work = {capa : sys.maxint for capa in cpu_capacities.values()}
            for cpu in cpu_capacities.keys():
                sysbench.run(cpus=[cpu], max_duration_s=1)
                # We could save the work done on each CPU, but we can make
                # things simpler and just store the smallest amount of work done
                # per capacity value.
                capa = cpu_capacities[cpu]
                capa_work[capa] = min(capa_work[capa], sysbench.output.nr_events)

        return cls(res_dir, capa_work)

    def test_capacity_sanity(self):
        """
        Assert that CPU capacity increase leads to more work done
        """
        sorted_capacities = sorted(self.capacity_work.keys())
        res = ResultBundle(True)

        for capacity, work in self.capacity_work.items():
            res.add_metric("Performance @{} capacity".format(capacity), work)

        for idx, capacity in enumerate(sorted_capacities[1:]):
            if self.capacity_work[capacity] <= self.capacity_work[sorted_capacities[idx]]:
                res.passed = False

        return res
