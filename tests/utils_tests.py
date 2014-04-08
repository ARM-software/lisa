#!/usr/bin/python

import unittest
import os, sys
import shutil, tempfile

TESTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(TESTS_DIRECTORY, "..", "cr2"))

class SetupDirectory(unittest.TestCase):
    def __init__(self, files_to_copy, *args, **kwargs):
        self.files_to_copy = files_to_copy
        super(SetupDirectory, self).__init__(*args, **kwargs)

    def setUp(self):
        self.previous_dir = os.getcwd()

        self.out_dir = tempfile.mkdtemp()
        os.chdir(self.out_dir)

        for fname in self.files_to_copy:
            src_fname = os.path.join(TESTS_DIRECTORY, fname)
            shutil.copy(src_fname, self.out_dir)

    def tearDown(self):
        os.chdir(self.previous_dir)
        shutil.rmtree(self.out_dir)
