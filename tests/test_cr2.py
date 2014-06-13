#!/usr/bin/python

import os
import matplotlib, tempfile

import cr2
from test_thermal import TestThermalBase

class TestCR2(TestThermalBase):
    def test_summary_plots(self):
        """Test summary_plots()

        Can't check that the graphs are ok, so just see that the method doesn't blow up"""

        cr2.summary_plots()
        cr2.summary_plots(width=14, title="Foo")
        matplotlib.pyplot.close('all')

    def test_summary_other_dir(self):
        """Test summary_plots() with another directory"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        cr2.summary_plots(path=self.out_dir)
        matplotlib.pyplot.close('all')

        # Sanity check that the test actually ran from another directory
        self.assertEquals(os.getcwd(), other_random_dir)
