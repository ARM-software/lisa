#    Copyright 2017-2017 ARM Limited
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

from bart.common.Utils import select_window

from test import LisaTest, experiment_test

WORKLOAD_DURATION_S = 5

REQUIRED_CPU_ACTIVE_TIME_PCT = 95

class HeavyLoadTest(LisaTest):
    """
    Test an EAS system under heavy load

    Runs N 100% RT-App threads where N is the number of CPUs, and checks that
    those tasks were spread across all CPUs in the system by asserting that all
    CPUs were fully utilized up until the first task completed.
    """

    test_conf = {
        'ftrace' : {
            'events' : ['cpu_idle', 'sched_switch'],
        },
        'modules' : ['cgroups'], # Required by freeze_userspace flag
    }

    experiments_conf = {
        "wloads" : {
            "n_heavy_tasks" : {
                "type" : "rt-app",
                "conf" : {
                    "class" : "profile",
                    "params" : {
                        "wmig" : {
                            "kind" : "Periodic",
                            "params" : {
                                "duty_cycle_pct": 100,
                                "duration_s": WORKLOAD_DURATION_S,
                            },
                            # Create one task for each cpu
                            "tasks" : "cpus",
                        },
                    },
                },
            },
        },
        "confs" : [{
            'tag' : 'energy_aware',
            'flags' : ['ftrace', 'freeze_userspace'],
            'sched_features' : 'ENERGY_AWARE',
        }]
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(HeavyLoadTest, cls).runExperiments(*args, **kwargs)

    @experiment_test
    def test_tasks_spread(self, experiment, tasks):
        trace = self.get_trace(experiment)
        start, _ = self.get_window(experiment)
        end = min(self.get_end_times(experiment).values())
        duration = end - start

        total_cpu_time = 0
        active_proportions = []
        for cpu, _ in enumerate(self.target.core_names):
            cpu_active = trace.getCPUActiveSignal(cpu)
            if cpu_active is None:
                raise RuntimeError(
                    "Couldn't get CPU-active signal. "
                    "Is the 'cpu_idle' ftrace event enabled in the kernel?")

            # Add extra events to cpu_active signal so that it matches the
            # window exactly
            new_index = sorted(cpu_active.index.tolist() + [start, end])
            cpu_active = cpu_active.reindex(new_index, method='ffill')

            active_time = trace.integrate_square_wave(cpu_active[start:end])
            active_proportions.append(active_time / duration)

        if any(a < (REQUIRED_CPU_ACTIVE_TIME_PCT / 100.)
               for a in active_proportions):

            proportions_str = ""
            for cpu, _ in enumerate(self.target.core_names):
                proportions_str += " {:3d} {:5.1f}%\n".format(
                    cpu, active_proportions[cpu]*100)

            raise AssertionError(
                "Some CPUs were less than {}% utilized\n"
                " CPU active proportions:\n{}".format(
                    REQUIRED_CPU_ACTIVE_TIME_PCT, proportions_str))

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
