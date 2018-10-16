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
import os.path
import re
import logging
import logging.config

from lisa.utils import Loggable

BASEPATH = os.getenv('LISA_HOME')
# This will catch both unset variable and variable set to an empty string
if not BASEPATH:
    logging.getLogger(__name__).warning('LISA_HOME env var is not set, LISA may misbehave.')


def setup_logging(filepath='logging.conf', level=logging.INFO):
    """
    Initialize logging used for all the LISA modules.

    :param filepath: the relative or absolute path of the logging
                     configuration to use. Relative path uses the
                     :data:`env.BASEPATH` as base folder.
    :type filepath: str

    :param level: the default log level to enable, INFO by default
    :type level: logging.<level> or int in [0..50]
    """

    # Load the specified logfile using an absolute path
    filepath = os.path.join(BASEPATH, filepath)
    if not os.path.exists(filepath):
        raise ValueError('Logging configuration file not found in: {}'\
                         .format(filepath))
    logging.config.fileConfig(filepath)
    logging.getLogger().setLevel(level)

    logging.info('Using LISA logging configuration:')
    logging.info('  %s', filepath)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
