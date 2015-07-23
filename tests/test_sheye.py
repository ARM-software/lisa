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
# File:        test_sheye.py
# ----------------------------------------------------------------
# $
#

import utils_tests
import unittest
import cr2
from sheye.SchedAssert import SchedAssert
from cr2.stats.Topology import Topology

@unittest.skipUnless(utils_tests.trace_cmd_installed(),
                     "trace-cmd not installed")
class TestSchedAssert(utils_tests.SetupDirectory):

    def __init__(self, *args, **kwargs):

        self.BIG = [1,2]
        self.LITTLE = [0, 3, 4, 5]
        self.clusters = [self.BIG, self.LITTLE]
        self.topology = Topology(clusters=self.clusters)
        super(TestSchedAssert, self).__init__(
             [("raw_trace.dat", "trace.dat")],
             *args,
             **kwargs)

    def test_get_runtime(self):

        r = cr2.Run()
        # The ls process is process we are
        # testing against with pre calculated
        # values
        process = "ls"

        # Complete duration
        expected_time = 0.0034740000264719129
        s = SchedAssert(r, self.topology, execname=process)
        self.assertAlmostEqual(s.getRuntime(), expected_time, places=9)
        self.assertAlmostEqual(s.getRuntime(), expected_time, places=9)

        # Non Interrupted Window
        window=(0.0034, 0.003525)
        expected_time=0.000125
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time, places=9)

        # Interrupted Window
        window=(0.0030, 0.0032)
        expected_time=0.000166
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time, places=9)

        # A window with multiple interruptions
        window = (0.0027, 0.0036)
        expected_time = 0.000817
        self.assertAlmostEqual(s.getRuntime(window=window), expected_time, places=9)
