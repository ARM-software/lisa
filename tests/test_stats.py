#    Copyright 2015-2015 ARM Limited
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

import unittest
from trappy.stats.Topology import Topology
from trappy.stats.Trigger import Trigger

import trappy
from trappy.base import Base
import pandas as pd
from pandas.util.testing import assert_series_equal


class TestTopology(unittest.TestCase):

    def test_add_to_level(self):
        """Test level creation"""

        level_groups = [[1, 2], [0, 3, 4, 5]]
        level = "test_level"
        topology = Topology()
        topology.add_to_level(level, level_groups)
        check_groups = topology.get_level(level)

        self.assertTrue(topology.has_level(level))
        self.assertEqual(level_groups, check_groups)

    def test_flatten(self):
        """Test Topology: flatten"""

        level_groups = [[1, 2], [0, 3, 4, 5]]
        level = "test_level"
        topology = Topology()
        topology.add_to_level(level, level_groups)
        flattened = [0, 1, 2, 3, 4, 5]

        self.assertEqual(flattened, topology.flatten())

    def test_cpu_topology_construction(self):
        """Test CPU Topology Construction"""

        cluster_0 = [0, 3, 4, 5]
        cluster_1 = [1, 2]
        clusters = [cluster_0, cluster_1]
        topology = Topology(clusters=clusters)

        # Check cluster level creation
        cluster_groups = [[0, 3, 4, 5], [1, 2]]
        self.assertTrue(topology.has_level("cluster"))
        self.assertEqual(cluster_groups, topology.get_level("cluster"))

        # Check cpu level creation
        cpu_groups = [[0], [1], [2], [3], [4], [5]]
        self.assertTrue(topology.has_level("cpu"))
        self.assertEqual(cpu_groups, topology.get_level("cpu"))

        # Check "all" level
        all_groups = [[0, 1, 2, 3, 4, 5]]
        self.assertEqual(all_groups, topology.get_level("all"))

    def test_level_span(self):
        """TestTopology: level_span"""

        level_groups = [[1, 2], [0, 3, 4, 5]]
        level = "test_level"
        topology = Topology()
        topology.add_to_level(level, level_groups)

        self.assertEqual(topology.level_span(level), 2)

    def test_group_index(self):
        """TestTopology: get_index"""

        level_groups = [[1, 2], [0, 3, 4, 5]]
        level = "test_level"
        topology = Topology()
        topology.add_to_level(level, level_groups)

        self.assertEqual(topology.get_index(level, [1, 2]), 0)
        self.assertEqual(topology.get_index(level, [0, 3, 4, 5]), 1)

# TODO: Remove inheritance from SetupDirectory when a
# Run object can be created without a trace file
from utils_tests import SetupDirectory


class BaseTestStats(SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(BaseTestStats, self).__init__(
            [("../doc/trace_stats.dat", "trace.dat")],
            *args,
            **kwargs)

    def setUp(self):

        super(BaseTestStats, self).setUp()
        run = trappy.Run()
        data = {

            "identifier": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
            "result": [
                "fire",
                "blank",
                "fire",
                "blank",
                "fire",
                "blank",
            ],
        }

        index = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], name="Time")
        data_frame = pd.DataFrame(data, index=index)
        run.add_parsed_event("aim_and_fire", data_frame)
        self._run = run
        self.topology = Topology(clusters=[[0], [1]])
