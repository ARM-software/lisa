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
# File:        sysfs_extractor.py
# ----------------------------------------------------------------
# $
#

import os
import pandas as pd
import re

class SysfsExtractor(object):
    """Operate on the parameters of a dump of Workload Automation's sysfs extractor instrumentation.

    path is the path to the workload in a output directory created by
    WA.

    """

    def __init__(self, path):
        self.thermal_path = os.path.join(path, "after", "sys", "devices",
                                         "virtual", "thermal", "thermal_zone0")
        self.properties = ["integral_cutoff", "k_d", "k_i", "k_po", "k_pu",
                           "policy", "sustainable_power"]

        for fname in os.listdir(self.thermal_path):
            if re.search(r"cdev\d+_weight", fname):
                self.properties.append(fname)
            elif re.search(r"trip_point_\d+_temp", fname):
                self.properties.append(fname)

    def get_parameters(self):
        """Get the parameters from a sysfs extractor dump

        WorkloadAutomation (WA) can dump sysfs values using its
        sysfs_extractor instrumentation.  Parse the tree and return the
        thermal parameters as a dict of key and values where the keys are
        the names of the files and values its corresponding values.

        """

        ret = {}

        for property_name in self.properties:
            property_path = os.path.join(self.thermal_path, property_name)

            with open(property_path) as fin:
                contents = fin.read()
                # Trim trailing newline
                contents = contents[:-1]

                try:
                    ret[property_name] = int(contents)
                except ValueError:
                    ret[property_name] = contents

        return ret

    def pretty_print_in_ipython(self):
        """Print parameters extracted from sysfs from a WA run in a pretty HTML table.

        This won't work if the code is not running in an ipython notebook."""

        from IPython.display import display, HTML

        params = self.get_parameters()

        params_items = [(key, [value]) for key, value in sorted(params.items())]
        dfr = pd.DataFrame.from_items(params_items, orient="index",
                                      columns=["Value"])
        display(HTML(dfr.to_html(header=False)))
