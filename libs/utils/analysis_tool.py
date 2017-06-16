# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited, Google and contributors.
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

import argparse
import logging
import os
import select
from trace import Trace
from conf import LisaLogging
from devlib.utils.misc import memoized

class AnalysisTool(object):
    """
    A base class for LISA analysis command-line tools.

    This class is intended to be subclassed in order to create a custom
    analysis tool.

    :param options_parser: A parser object passed by the specific tool
                           to parse additional options if any (optional).
    :type options_parser: ArgumentParser
    """

    def __init__(self, options_parser=None):
        self.options_parser = options_parser

    def _setup_logging(self):
        """
        Setup logging for the analysis tool.

        Analysis tools shouldn't output to standard output as their
        output can be used by other tools. Configure the logger to
        output only error messages.
        """
        self._log = logging.getLogger('AnalysisTool')
        level = logging.INFO if self.args.log_stdout else logging.ERROR
        LisaLogging.setup(level=level)

    def _parse_args(self):
        """
        Parse command line arguments given to the tool, also use the parser
        object if one is passed by the more specific class so that the
        tool specific options are also parsed.
        """
        parser = self.options_parser
        if not parser:
            parser = argparse.ArgumentParser(
                    description='LISA Analysis tool configuration')

        # General options that all tools accept
        parser.add_argument('--trace', type=str,
                help='Path of the trace file or directory to be used',
                required=True)

        parser.add_argument('--log-stdout', action='store_true',
                help='Output logging to stdout',
                required=False)

        self.args = parser.parse_args()

    def run_analysis(self):
        """
        Run an analysis and store results

        The specific analysis tool should override this and perform
        analysis and provide results in self.output.
        """
        raise NotImplementedError('Analysis tool needs to define run_analysis')

    def run_main(self):
        """
        The main function that sets up, runs the analysis and provides
        output to the user.

        The specific analysis tool should call this function to start the analysis
        which will perform analysis and report the result. The tool should also
        override run_analysis to provide custom analysis and optionally override
        run_output to provide custom output.
        """

        self._parse_args()
        self._setup_logging()

        # Create trace object
        self._log.info('Creating trace object...')
        self.trace = Trace(None, self.args.trace)

        self.run_analysis()
