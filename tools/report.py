#!/usr/bin/python
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

