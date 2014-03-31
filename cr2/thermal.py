#!/usr/bin/python3
"""Process the output of the power allocator trace in the current directory's trace.dat"""

import os
import subprocess

class Thermal(object):
    def __init__(self):
        if not os.path.isfile("trace.txt"):
            self.__run_trace_cmd_report()


    def __run_trace_cmd_report(self):
        """Run "trace-cmd report > trace.txt".  Overwrites the contents of trace.txt if it exists."""
        with open(os.devnull) as devnull:
            out = subprocess.check_output(["trace-cmd", "report"], stderr=devnull)

        with open("trace.txt", "w") as f:
            f.write(out)
