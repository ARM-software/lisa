#    Copyright 2014-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import logging

from twisted.python import log

__all__ = ['debug', 'info', 'warning', 'error', 'critical', 'start_logging']

debug = lambda x: log.msg(x, logLevel=logging.DEBUG)
info = lambda x: log.msg(x, logLevel=logging.INFO)
warning = lambda x: log.msg(x, logLevel=logging.WARNING)
error = lambda x: log.msg(x, logLevel=logging.ERROR)
critical = lambda x: log.msg(x, logLevel=logging.CRITICAL)


class CustomLoggingObserver(log.PythonLoggingObserver):

    def __init__(self, loggerName="twisted"):
        super(CustomLoggingObserver, self).__init__(loggerName)
        if hasattr(self, '_newObserver'):  # new vesions of Twisted
            self.logger = self._newObserver.logger  # pylint: disable=no-member

    def emit(self, eventDict):
        if 'logLevel' in eventDict:
            level = eventDict['logLevel']
        elif eventDict['isError']:
            level = logging.ERROR
        else:
            # All of that just just to override this one line from
            # default INFO level...
            level = logging.DEBUG
        text = log.textFromEventDict(eventDict)
        if text is None:
            return
        self.logger.log(level, text)


logObserver = CustomLoggingObserver()
logObserver.start()


def start_logging(level, fmt='%(asctime)s %(levelname)-8s: %(message)s'):
    logging.basicConfig(level=getattr(logging, level), format=fmt)

