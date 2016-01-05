#    Copyright 2015-2016 ARM Limited
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


class TestBART(SetupDirectory):

    def __init__(self, *args, **kwargs):
        super(TestBART, self).__init__(
            [
                ("./trace.txt", "trace.txt"),
                ("./trace.raw.txt", "trace.raw.txt")
            ],
            *args,
            **kwargs)
