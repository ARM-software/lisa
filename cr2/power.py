#!/usr/bin/python
"""Process the output of the cpu_cooling devices in the current directory's trace.dat"""

import pandas as pd
from thermal import BaseThermal

class Power(BaseThermal):
    def __init__(self, path=None):
        super(Power, self).__init__(
            basepath=path,
            unique_word="thermal_power_limit",
        )
