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


from datetime import date
import os
import subprocess
import unittest


def copyright_is_valid(fname):
    """Return True if fname has a valid copyright"""
    with open(fname) as fin:
        # Read the first 2K of the file.  If the copyright is not there, you
        # are probably doing something wrong
        lines = fin.readlines(2048)

    # Either the first or the second line must have a "#    Copyright:" line
    if not lines[0].startswith("#    Copyright"):
        if lines[1].startswith("#    Copyright"):
            # Drop the first line to align the copyright to lines[0]
            lines = lines[1:]
        else:
            return False

    # The copyright mentions ARM Limited
    if "ARM Limited" not in lines[0]:
        return False

    # The Copyright includes the current year
    current_year = date.today().year
    if str(current_year) not in lines[0]:
        return False

    # It's the apache license
    if "#     http://www.apache.org/licenses/LICENSE-2.0\n" != lines[6]:
        return False

    return True


class TestCopyRight(unittest.TestCase):
    def test_copyrights(self):
        """Check that all files have valid copyrights"""
        files_in_repo = subprocess.check_output(["git", "ls-files"])

        for fname in files_in_repo.split():
            if os.path.splitext(fname)[1] == ".py":
                if not copyright_is_valid(fname):
                    print("Invalid copyright in {}".format(fname))
                    self.fail()
