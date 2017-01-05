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


from datetime import date
from glob import glob
import os
import re
import unittest


def copyright_is_valid(fname):
    """Return True if fname has a valid copyright"""
    with open(fname) as fin:
        # Read the first 2K of the file.  If the copyright is not there, you
        # are probably doing something wrong
        lines = fin.readlines(2048)

    # Either the first or the second line must have a "Copyright:" line
    first_line = re.compile(r"(#| \*)    Copyright")
    try:
        if not first_line.search(lines[0]):
            if first_line.search(lines[1]):
                # Drop the first line to align the copyright to lines[0]
                lines = lines[1:]
            else:
                return False
    except IndexError:
        return False

    # The copyright mentions ARM Limited
    if "ARM Limited" not in lines[0]:
        return False

    apache_line = 6
    if "Google Inc" in lines[1]:
        apache_line += 1

    # The Copyright includes the current year
    current_year = date.today().year
    if str(current_year) not in lines[0]:
        return False

    # It's the apache license
    if "http://www.apache.org/licenses/LICENSE-2.0" not in lines[apache_line]:
        return False

    return True


class TestCopyRight(unittest.TestCase):
    def test_copyrights(self):
        """Check that all files have valid copyrights"""

        tests_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(tests_dir)
        patterns_to_ignore = {}

        for root, dirs, files in os.walk(base_dir):
            if ".gitignore" in files:
                fname = os.path.join(root, ".gitignore")
                with open(fname) as fin:
                    lines = fin.readlines()

            patterns_to_ignore[root] = [l.strip() for l in lines]

            files_to_ignore = []
            for directory, patterns in patterns_to_ignore.iteritems():
                if root.startswith(directory):
                    for pat in patterns:
                        pat = os.path.join(root, pat)
                        files_to_ignore.extend(glob(pat))

            for dirname in dirs:
                full_dirname = os.path.join(root, dirname)
                if full_dirname in files_to_ignore:
                    dirs.remove(dirname)


            for fname in files:
                fname = os.path.join(root, fname)
                if fname in files_to_ignore:
                    continue

                extension = os.path.splitext(fname)[1]
                if extension in [".py", ".js", ".css"]:
                    if not copyright_is_valid(fname):
                        print("Invalid copyright in {}".format(fname))
                        self.fail()

            if '.git' in dirs:
                dirs.remove('.git')
