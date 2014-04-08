#!/usr/bin/python
"""Process the output of the power allocator's PI controller in the current directory's trace.dat"""

import pandas as pd
from thermal import BaseThermal

class PIController(BaseThermal):
    def __init__(self, path=None):
        super(PIController, self).__init__(
            basepath=path,
            unique_word="thermal_power_allocator_pi",
        )
