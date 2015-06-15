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
# File:        test_wa_sysfs_extractor.py
# ----------------------------------------------------------------
# $
#

import os
import subprocess

import utils_tests

class TestWASysfsExtractor(utils_tests.SetupDirectory):
    """Test the WA specific interface to get parameters from a sysfs extractor"""
    def __init__(self, *args, **kwargs):
        self.wa_sysfs_fname = "WA_sysfs_extract.tar.xz"
        super(TestWASysfsExtractor, self).__init__(
            [(self.wa_sysfs_fname, self.wa_sysfs_fname)],
            *args, **kwargs)

    def setUp(self):
        super(TestWASysfsExtractor, self).setUp()
        subprocess.call(["tar", "xf", self.wa_sysfs_fname])

    def test_get_parameters(self):
        """Test that we can get the parameters of a sysfs extractor output"""

        from cr2.wa.sysfs_extractor import SysfsExtractor

        os.chdir("..")
        thermal_params = SysfsExtractor(self.out_dir).get_parameters()
        self.assertEquals(thermal_params["cdev0_weight"], 1024)
        self.assertEquals(thermal_params["cdev1_weight"], 768)
        self.assertEquals(thermal_params["trip_point_0_temp"], 72000)
        self.assertEquals(thermal_params["policy"], "power_allocator")
