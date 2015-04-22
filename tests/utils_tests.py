#!/usr/bin/python
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
# File:        utils_tests.py
# ----------------------------------------------------------------
# $
#

import unittest
import os
import shutil
import subprocess
import tempfile

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def trace_cmd_installed():
    """Return true if trace-cmd is installed, false otherwise"""
    with open(os.devnull) as devnull:
        try:
            subprocess.check_call(["trace-cmd", "options"], stdout=devnull)
        except OSError:
            return False

    return True

class SetupDirectory(unittest.TestCase):
    def __init__(self, files_to_copy, *args, **kwargs):
        self.files_to_copy = files_to_copy
        super(SetupDirectory, self).__init__(*args, **kwargs)

    def setUp(self):
        self.previous_dir = os.getcwd()

        self.out_dir = tempfile.mkdtemp()
        os.chdir(self.out_dir)

        for src_fname, dst_fname in self.files_to_copy:
            src_fname = os.path.join(TESTS_DIRECTORY, src_fname)
            shutil.copy(src_fname, os.path.join(self.out_dir, dst_fname))

    def tearDown(self):
        os.chdir(self.previous_dir)
        shutil.rmtree(self.out_dir)
