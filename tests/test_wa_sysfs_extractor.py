#    Copyright 2015-2017 ARM Limited
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


import os
import subprocess
import unittest

import utils_tests

import trappy.wa

class TestWASysfsExtractor(utils_tests.SetupDirectory):
    """Test the WA specific interface to get parameters from a sysfs extractor"""
    def __init__(self, *args, **kwargs):
        self.wa_sysfs_fname = "WA_sysfs_extract.tar.xz"
        super(TestWASysfsExtractor, self).__init__(
            [(self.wa_sysfs_fname, self.wa_sysfs_fname)],
            *args, **kwargs)

    def setUp(self):
        super(TestWASysfsExtractor, self).setUp()
        subprocess.check_call(["tar", "xf", self.wa_sysfs_fname])

    def test_get_parameters(self):
        """Test that we can get the parameters of a sysfs extractor output"""

        os.chdir("..")
        thermal_params = trappy.wa.SysfsExtractor(self.out_dir).get_parameters()
        self.assertEquals(thermal_params["cdev0_weight"], 1024)
        self.assertEquals(thermal_params["cdev1_weight"], 768)
        self.assertEquals(thermal_params["trip_point_0_temp"], 72000)
        self.assertEquals(thermal_params["policy"], "power_allocator")

    def test_print_thermal_params(self):
        """Test that printing the thermal params doesn't bomb"""

        trappy.wa.SysfsExtractor(".").pretty_print_in_ipython()

class TestWASysfsExtractorFailMode(unittest.TestCase):
    """Test the failure modes of the Workload Automation sysfs extractor"""

    def test_get_params_invalid_directory(self):
        """An invalid directory for trappy.wa.SysfsExtractor doesn't bomb"""

        sysfs_extractor = trappy.wa.SysfsExtractor(".")
        self.assertEquals(sysfs_extractor.get_parameters(), {})

        sysfs_extractor.pretty_print_in_ipython()
