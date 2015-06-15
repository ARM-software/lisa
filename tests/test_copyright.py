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
# File:        test_copyright.py
# ----------------------------------------------------------------
# $
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

    # Either the first or the second line must have a "# $Copyright:" line
    if lines[0] != "# $Copyright:\n":
        if lines[1] == "# $Copyright:\n":
            # Drop the first line to align the copyright to lines[0]
            lines = lines[1:]
        else:
            return False

    # There's a (C) COPYRIGHT which includes the current year
    if "(C) COPYRIGHT" not in lines[4]:
        return False

    current_year = date.today().year
    if str(current_year) not in lines[4]:
        return False

    # There's a "File: $fname" line that matches the current file
    if "File: " not in lines[10]:
        return False

    if os.path.basename(fname) not in lines[10]:
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
