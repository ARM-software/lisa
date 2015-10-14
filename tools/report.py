#!/usr/bin/python

import sys
sys.path.insert(1, "./libs")
from utils.report import Report

if __name__ == "__main__":
    Report('./results_latest')

