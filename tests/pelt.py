from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from hypothesis import given
from hypothesis.strategies import integers, tuples, none, one_of
import unittest
from sys import maxsize

from bart.sched.pelt import *

# Required to use `int` not `long` henx ma=maxint
nonneg_ints = lambda mi=0, ma=maxint: integers(min_value=mi, max_value=ma)

# Generate a Simulator
simulator_args = lambda: tuples(
    nonneg_ints(0, 1024),         # init_value
    nonneg_ints(1, 256),          # half_life_ms
    one_of(nonneg_ints(), none()) # decay_cap_ms
)


# Generate args for PeriodicTask::__init__ args using run_samples
periodic_task_args_samples = lambda: tuples(
    nonneg_ints(1), # period_samples
    nonneg_ints(),  # start_sample
    nonneg_ints(),  # run_samples
    none(),         # duty_cycle_pct
).filter(lambda period___run___: period___run___[2] <= period___run___[0])

# Generate args for PeriodicTask::__init__ args using duty_cycle_pct
periodic_task_args_pct = lambda: tuples(
    nonneg_ints(1),   # period_samples
    nonneg_ints(),    # start_sample
    none(),           # run_samples
    integers(0, 100), # duty_cycle_pct
)

# Generate a PeriodicTask using args from one of the above two strategies
periodic_task_args = lambda: one_of(periodic_task_args_samples(),
                                    periodic_task_args_pct())

# Generate a tuple of ordered integers less than 20
signal_range = lambda: tuples(nonneg_ints(0, 200),
                              nonneg_ints(0, 200)).map(sorted)

class TestSimulator(unittest.TestCase):
    @given(periodic_task_args(), simulator_args())
    def test_stable_range_range(self, task_args, sim_args):
        """Test that the stable range's max_value is within expected bounds"""
        task = PeriodicTask(*task_args)
        sim = Simulator(*sim_args)

        stable_range = sim.stableRange(task)
        self.assertLessEqual(stable_range.max_value, sim._signal_max)
        self.assertGreaterEqual(stable_range.min_value, 0)

    @given(periodic_task_args(), simulator_args())
    def test_signal_within_range(self, task_args, sim_args):
        """Test that the simulated signal falls within the expected bounds"""
        task = PeriodicTask(*task_args)
        sim = Simulator(*sim_args)

        signal = sim.getSignal(task)
        signal_max = signal.max()['pelt_value']
        signal_min = signal.min()['pelt_value']
        self.assertLessEqual(signal_max, sim._signal_max)
        self.assertGreaterEqual(signal_min, 0)

    @given(periodic_task_args(), simulator_args(), signal_range())
    def test_signal_time_range(self, task_args, sim_args, signal_range):
        """Test that the result of getSignal covers the requested range"""
        task = PeriodicTask(*task_args)
        sim = Simulator(*sim_args)
        start_s, end_s = signal_range

        signal = sim.getSignal(task, start_s, end_s)

        # Should start no earlier than 1 sample before start_s
        earliest_start = min(0, start_s - (im._sample_us/1.e6))
        self.assertGreaterEqual(signal.index[0], earliest_start)
        # Should start no later than start_s
        self.assertLessEqual(signal.index[0], start_s)

        # Should start no earlier than end_s
        self.assertGreaterEqual(signal.index[-1], end_s)
        # Should end no later than 1 sample after end_s
        latest_start = end_s + (sim._sample_us/1.e6)
        self.assertLessEqual(signal.index[-1], latest_start)

if __name__ == "__main__":
    unittest.main()
