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

import json
import os
import re
import logging
import logging.config

class LisaLogging(object):

    @classmethod
    def setup(self, filepath='logging.conf', level=logging.INFO):
        """
        Initialize logging used for all the LISA modules.

        :param filepath: the relative or absolute path of the logging configuration to use.
                         Relative path uses the LISA_HOME environment variable
                         has base folder.
        :type filepath: str

        :param level: the default log level to enable, INFO by default
        :type level: logging.<level> or int in [0..50]
        """

        # Load the specified logfile using an absolute path
        basepath = os.getenv('LISA_HOME')
        filepath = os.path.join(basepath, filepath)
        if not os.path.exists(filepath):
            raise ValueError('Logging configuration file not found in: {}'\
                             .format(filepath))
        logging.config.fileConfig(filepath)
        logging.getLogger().setLevel(level)

        logging.info('Using LISA logging configuration:')
        logging.info('  %s', filepath)

class JsonConf(object):
    """
    Class for parsing a JSON superset with comments.

    Simply strips comments and then uses the standard JSON parser.

    :param filename: Path to file to parse
    :type filename: str
    """

    def __init__(self, filename):
        self.filename = filename
        self.json = None

    def load(self):
        """
        Parse a JSON file

        First remove comments and then use the json module package
        Comments look like :

        ::

            // ...

        or

        ::

            /*
            ...
            */

        """

        # Setup logging
        self._log = logging.getLogger('JsonConf')

        if not os.path.isfile(self.filename):
            raise RuntimeError(
                'Missing configuration file: {}'.format(self.filename)
            )
        self._log.debug('loading JSON...')

        with open(self.filename) as fh:
            content = ''.join(fh.readlines())

            ## Looking for comments
            match = JSON_COMMENTS_RE.search(content)
            while match:
                # single line comment
                content = content[:match.start()] + content[match.end():]
                match = JSON_COMMENTS_RE.search(content)

            # Allow trailing commas in dicts an lists in JSON
            # Note that this simple implementation will mangle things like:
            # {"config": ", }"}
            content = re.sub(r',[ \t\r\n]+}', '}', content)
            content = re.sub(r',[ \t\r\n]+\]', ']', content)

            # Return json file
            self.json = json.loads(content, parse_int=int)
            self._log.debug('Loaded JSON configuration:')
            self._log.debug('   %s', self.json)

        return self.json

    def show(self):
        """
        Pretty-print content of parsed JSON
        """
        print(json.dumps(self.json, indent=4))

# Regular expression for comments
JSON_COMMENTS_RE = re.compile(
    r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)



# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
