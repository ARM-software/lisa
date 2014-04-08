#!/usr/bin/python
"""Compare runs 2

Second version of the compare runs script, to compare two traces of the power allocator governor"""

import csv, re
import pandas as pd

class CR2(object):
    """Compare two traces of the power allocator governor"""
    def get_results(self):
        """Return a pd.DataFrame with the results

        The DataFrame has one row: "score" and as many columns as
        benchmarks were found.  For benchmarks that have a score
        result, that's what's used.  For benchmarks with FPS_* result,
        that's the score.  E.g. glbenchmark "score" is it's fps"""

        pat_result = re.compile("score|FPS_")
        res_dict = {}

        with open("results.csv") as fin:
            results = csv.reader(fin)

            for row in results:
                if (re.match(pat_result, row[3])):
                    res_dict[row[0]] = pd.Series([int(row[4])], index=["score"])

        return pd.DataFrame(res_dict)
