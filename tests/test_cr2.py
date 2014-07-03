#!/usr/bin/python

import os
import matplotlib, tempfile

import cr2
from test_thermal import BaseTestThermal

class TestCR2(BaseTestThermal):
    def __init__(self, *args, **kwargs):
        super(TestCR2, self).__init__(*args, **kwargs)
        self.map_label = {"0000000f": "A7", "000000f0": "A15"}
        self.actor_order = ["GPU", "A15", "A7"]

    def test_summary_plots(self):
        """Test summary_plots()

        Can't check that the graphs are ok, so just see that the method doesn't blow up"""

        cr2.summary_plots(self.actor_order, self.map_label)
        matplotlib.pyplot.close('all')

        cr2.summary_plots(self.actor_order, self.map_label, width=14,
                          title="Foo")
        matplotlib.pyplot.close('all')

    def test_summary_other_dir(self):
        """Test summary_plots() with another directory"""

        other_random_dir = tempfile.mkdtemp()
        os.chdir(other_random_dir)

        cr2.summary_plots(self.actor_order, self.map_label, path=self.out_dir)
        matplotlib.pyplot.close('all')

        # Sanity check that the test actually ran from another directory
        self.assertEquals(os.getcwd(), other_random_dir)
