#    Copyright 2013-2015 ARM Limited
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


# pylint: disable=E1101
import logging
import string
import threading

import colorama

from wlauto.core.bootstrap import settings
import wlauto.core.signal as signal


COLOR_MAP = {
    logging.DEBUG: colorama.Fore.BLUE,
    logging.INFO: colorama.Fore.GREEN,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.ERROR: colorama.Fore.RED,
    logging.CRITICAL: colorama.Style.BRIGHT + colorama.Fore.RED,
}

RESET_COLOR = colorama.Style.RESET_ALL


def init_logging(verbosity):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    error_handler = ErrorSignalHandler(logging.DEBUG)
    root_logger.addHandler(error_handler)

    console_handler = logging.StreamHandler()
    if verbosity == 1:
        console_handler.setLevel(logging.DEBUG)
        if 'colour_enabled' in settings.logging and not settings.logging['colour_enabled']:
            console_handler.setFormatter(LineFormatter(settings.logging['verbose_format']))
        else:
            console_handler.setFormatter(ColorFormatter(settings.logging['verbose_format']))
    else:
        console_handler.setLevel(logging.INFO)
        if 'colour_enabled' in settings.logging and not settings.logging['colour_enabled']:
            console_handler.setFormatter(LineFormatter(settings.logging['regular_format']))
        else:
            console_handler.setFormatter(ColorFormatter(settings.logging['regular_format']))
    root_logger.addHandler(console_handler)

    logging.basicConfig(level=logging.DEBUG)


def add_log_file(filepath, level=logging.DEBUG):
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(LineFormatter(settings.logging['file_format']))
    root_logger.addHandler(file_handler)


class ErrorSignalHandler(logging.Handler):
    """
    Emits signals for ERROR and WARNING level traces.

    """

    def emit(self, record):
        if record.levelno == logging.ERROR:
            signal.send(signal.ERROR_LOGGED, self)
        elif record.levelno == logging.WARNING:
            signal.send(signal.WARNING_LOGGED, self)


class ColorFormatter(logging.Formatter):
    """
    Formats logging records with color and prepends record info
    to each line of the message.

        BLUE for DEBUG logging level
        GREEN for INFO logging level
        YELLOW for WARNING logging level
        RED for ERROR logging level
        BOLD RED for CRITICAL logging level

    """

    def __init__(self, fmt=None, datefmt=None):
        super(ColorFormatter, self).__init__(fmt, datefmt)
        template_text = self._fmt.replace('%(message)s', RESET_COLOR + '%(message)s${color}')
        template_text = '${color}' + template_text + RESET_COLOR
        self.fmt_template = string.Template(template_text)

    def format(self, record):
        self._set_color(COLOR_MAP[record.levelno])

        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        d = record.__dict__
        parts = []
        for line in record.message.split('\n'):
            d.update({'message': line.strip('\r')})
            parts.append(self._fmt % d)

        return '\n'.join(parts)

    def _set_color(self, color):
        self._fmt = self.fmt_template.substitute(color=color)


class LineFormatter(logging.Formatter):
    """
    Logs each line of the message separately.

    """

    def __init__(self, fmt=None, datefmt=None):
        super(LineFormatter, self).__init__(fmt, datefmt)

    def format(self, record):
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        d = record.__dict__
        parts = []
        for line in record.message.split('\n'):
            d.update({'message': line.strip('\r')})
            parts.append(self._fmt % d)

        return '\n'.join(parts)


class BaseLogWriter(object):

    def __init__(self, name, level=logging.DEBUG):
        """
        File-like object class designed to be used for logging from streams
        Each complete line (terminated by new line character) gets logged
        at DEBUG level. In complete lines are buffered until the next new line.

        :param name: The name of the logger that will be used.

        """
        self.logger = logging.getLogger(name)
        self.buffer = ''
        if level == logging.DEBUG:
            self.do_write = self.logger.debug
        elif level == logging.INFO:
            self.do_write = self.logger.info
        elif level == logging.WARNING:
            self.do_write = self.logger.warning
        elif level == logging.ERROR:
            self.do_write = self.logger.error
        else:
            raise Exception('Unknown logging level: {}'.format(level))

    def flush(self):
        # Defined to match the interface expected by pexpect.
        return self

    def close(self):
        if self.buffer:
            self.logger.debug(self.buffer)
            self.buffer = ''
        return self

    def __del__(self):
        # Ensure we don't lose bufferd output
        self.close()


class LogWriter(BaseLogWriter):

    def write(self, data):
        data = data.replace('\r\n', '\n').replace('\r', '\n')
        if '\n' in data:
            parts = data.split('\n')
            parts[0] = self.buffer + parts[0]
            for part in parts[:-1]:
                self.do_write(part)
            self.buffer = parts[-1]
        else:
            self.buffer += data
        return self


class LineLogWriter(BaseLogWriter):

    def write(self, data):
        self.do_write(data)


class StreamLogger(threading.Thread):
    """
    Logs output from a stream in a thread.

    """

    def __init__(self, name, stream, level=logging.DEBUG, klass=LogWriter):
        super(StreamLogger, self).__init__()
        self.writer = klass(name, level)
        self.stream = stream
        self.daemon = True

    def run(self):
        line = self.stream.readline()
        while line:
            self.writer.write(line.rstrip('\n'))
            line = self.stream.readline()
        self.writer.close()
