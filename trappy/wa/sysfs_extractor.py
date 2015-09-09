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


import os
import pandas as pd
import re

class SysfsExtractor(object):
    """Operate on the parameters of a dump of Workload Automation's
       sysfs extractor instrumentation.

    :param path: The path to the workload in a output directory created by
        WA.
    :type path: str
    """

    def __init__(self, path):
        self.thermal_path = os.path.join(path, "after", "sys", "devices",
                                         "virtual", "thermal", "thermal_zone0")
        self.properties = ["integral_cutoff", "k_d", "k_i", "k_po", "k_pu",
                           "policy", "sustainable_power"]

        try:
            sysfs_files = os.listdir(self.thermal_path)
        except OSError:
            sysfs_files = []

        for fname in sysfs_files:
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

            if not os.path.isfile(property_path):
                continue

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

        # Don't print anything if we couldn't find any parameters
        if len(params) == 0:
            return

        params_items = [(key, [value]) for key, value in sorted(params.items())]
        dfr = pd.DataFrame.from_items(params_items, orient="index",
                                      columns=["Value"])
        display(HTML(dfr.to_html(header=False)))
