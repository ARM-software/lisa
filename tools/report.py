#!/usr/bin/python

import sys
sys.path.insert(1, "./libs")
from utils.report import Report

import argparse

parser = argparse.ArgumentParser(
        description='EAS RFC Configuration Comparator.')
parser.add_argument('--bases', type=str,
        default='bases_',
        help='Regexp of BASE configurations to compare with')
parser.add_argument('--tests',  type=str,
        default='tests_',
        help='Regexp of TEST configurations to compare against')
parser.add_argument('--results', type=str,
        default='./results_latest',
        help='Folder containing experimental results')

if __name__ == "__main__":
    args = parser.parse_args()
    Report(args.results, compare=[(args.bases, args.tests)])

