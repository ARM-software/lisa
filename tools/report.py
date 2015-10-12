#!/usr/bin/python

import sys
sys.path.insert(1, "./libs")
from utils.report import Results

if __name__ == "__main__":
    results = Results('./results_latest')
    results.report()

