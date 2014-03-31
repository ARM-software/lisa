#!/usr/bin/python3
"""Process the output of the power allocator trace in the current directory's trace.dat"""

import os
import re

class Thermal(object):
    def __init__(self):
        if not os.path.isfile("trace.txt"):
            self.__run_trace_cmd_report()


    def __run_trace_cmd_report(self):
        """Run "trace-cmd report > trace.txt".  Overwrites the contents of trace.txt if it exists."""
        from subprocess import check_output

        with open(os.devnull) as devnull:
            out = check_output(["trace-cmd", "report"], stderr=devnull)

        with open("trace.txt", "w") as f:
            f.write(out)

    def write_thermal_csv(self):
        pat_timestamp = re.compile(r"([0-9]+\.[0-9]+):")
        pat_data = re.compile(r"[A-Za-z0-9_]+=([0-9]+) ")
        header = ""

        with open("thermal.csv", "w") as fout:
            with open("trace.txt") as fin:
                for line in fin:
                    if not re.search("Ptot_out", line):
                        continue

                    line = line[:-1]

                    m = re.search(pat_timestamp, line)
                    timestamp = m.group(1)

                    semi_idx = line.index(" : ")
                    data_str = line[semi_idx + 3:]

                    if not header:
                        header = re.sub(r"([A-Za-z0-9_]+)=[0-9]+ ", r"\1,", data_str)
                        header = header[:-1]
                        header = "time," + header + "\n"
                        fout.write(header)

                    parsed_data = re.sub(pat_data, r"\1,", data_str)
                    # Drop the last comma
                    parsed_data = parsed_data[:-1]

                    parsed_data = timestamp + "," + parsed_data + "\n"
                    fout.write(parsed_data)
