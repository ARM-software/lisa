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
import logging
import os
import re

class JsonConf(object):

    def __init__(self, filename):
        self.filename = filename
        self.json = None

    def load(self):
        """ Parse a JSON file
            First remove comments and then use the json module package
            Comments look like :
                // ...
            or
                /*
                ...
                */
        """
        if not os.path.isfile(self.filename):
            raise RuntimeError(
                'Missing configuration file: {}'.format(self.filename)
            )
        logging.debug('loading JSON...')

        with open(self.filename) as fh:
            content = ''.join(fh.readlines())

            ## Looking for comments
            match = JSON_COMMENTS_RE.search(content)
            while match:
                # single line comment
                content = content[:match.start()] + content[match.end():]
                match = JSON_COMMENTS_RE.search(content)

            # Return json file
            self.json = json.loads(content, parse_int=int)
            logging.debug('Loaded JSON configuration:\n%s', self.json)

        return self.json

    def show(self):
        print json.dumps(self.json, indent=4)

# Regular expression for comments
JSON_COMMENTS_RE = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)


