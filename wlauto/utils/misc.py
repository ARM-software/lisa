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


"""
Miscellaneous functions that don't fit anywhere else.

"""
from __future__ import division
import os
import sys
import re
import math
import imp
import string
import threading
import signal
import subprocess
import pkgutil
import traceback
import logging
import random
import hashlib
from datetime import datetime, timedelta
from operator import mul, itemgetter
from StringIO import StringIO
from itertools import cycle, groupby
from functools import partial
from distutils.spawn import find_executable

import yaml
from dateutil import tz

from devlib.utils.misc import ABI_MAP, check_output, walk_modules, \
                              ensure_directory_exists, ensure_file_directory_exists, \
                              merge_dicts, merge_lists, normalize, convert_new_lines, \
                              escape_quotes, escape_single_quotes, escape_double_quotes, \
                              isiterable, getch, as_relative, ranges_to_list, \
                              list_to_ranges, list_to_mask, mask_to_list, which, \
                              get_cpu_mask, unique

check_output_logger = logging.getLogger('check_output')


# Defined here rather than in wlauto.exceptions due to module load dependencies
class TimeoutError(Exception):
    """Raised when a subprocess command times out. This is basically a ``WAError``-derived version{}
    of ``subprocess.CalledProcessError``, the thinking being that while a timeout could be due to
    programming error (e.g. not setting long enough timers), it is often due to some failure in the
    environment, and there fore should be classed as a "user error"."""

    def __init__(self, command, output):
        super(TimeoutError, self).__init__('Timed out: {}'.format(command))
        self.command = command
        self.output = output

    def __str__(self):
        return '\n'.join([self.message, 'OUTPUT:', self.output or ''])


def diff_tokens(before_token, after_token):
    """
    Creates a diff of two tokens.

    If the two tokens are the same it just returns returns the token
    (whitespace tokens are considered the same irrespective of type/number
    of whitespace characters in the token).

    If the tokens are numeric, the difference between the two values
    is returned.

    Otherwise, a string in the form [before -> after] is returned.

    """
    if before_token.isspace() and after_token.isspace():
        return after_token
    elif before_token.isdigit() and after_token.isdigit():
        try:
            diff = int(after_token) - int(before_token)
            return str(diff)
        except ValueError:
            return "[%s -> %s]" % (before_token, after_token)
    elif before_token == after_token:
        return after_token
    else:
        return "[%s -> %s]" % (before_token, after_token)


def prepare_table_rows(rows):
    """Given a list of lists, make sure they are prepared to be formatted into a table
    by making sure each row has the same number of columns and stringifying all values."""
    rows = [map(str, r) for r in rows]
    max_cols = max(map(len, rows))
    for row in rows:
        pad = max_cols - len(row)
        for _ in xrange(pad):
            row.append('')
    return rows


def write_table(rows, wfh, align='>', headers=None):  # pylint: disable=R0914
    """Write a column-aligned table to the specified file object."""
    if not rows:
        return
    rows = prepare_table_rows(rows)
    num_cols = len(rows[0])

    # cycle specified alignments until we have max_cols of them. This is
    # consitent with how such cases are handled in R, pandas, etc.
    it = cycle(align)
    align = [it.next() for _ in xrange(num_cols)]

    cols = zip(*rows)
    col_widths = [max(map(len, c)) for c in cols]
    row_format = ' '.join(['{:%s%s}' % (align[i], w) for i, w in enumerate(col_widths)])
    row_format += '\n'

    if headers:
        wfh.write(row_format.format(*headers))
        underlines = ['-' * len(h) for h in headers]
        wfh.write(row_format.format(*underlines))

    for row in rows:
        wfh.write(row_format.format(*row))


def get_null():
    """Returns the correct null sink based on the OS."""
    return 'NUL' if os.name == 'nt' else '/dev/null'


def get_traceback(exc=None):
    """
    Returns the string with the traceback for the specifiec exc
    object, or for the current exception exc is not specified.

    """
    if exc is None:
        exc = sys.exc_info()
    if not exc:
        return None
    tb = exc[2]
    sio = StringIO()
    traceback.print_tb(tb, file=sio)
    del tb  # needs to be done explicitly see: http://docs.python.org/2/library/sys.html#sys.exc_info
    return sio.getvalue()


def _check_remove_item(the_list, item):
    """Helper function for merge_lists that implements checking wether an items
    should be removed from the list and doing so if needed. Returns ``True`` if
    the item has been removed and ``False`` otherwise."""
    if not isinstance(item, basestring):
        return False
    if not item.startswith('~'):
        return False
    actual_item = item[1:]
    if actual_item in the_list:
        del the_list[the_list.index(actual_item)]
    return True


VALUE_REGEX = re.compile(r'(\d+(?:\.\d+)?)\s*(\w*)')

UNITS_MAP = {
    's': 'seconds',
    'ms': 'milliseconds',
    'us': 'microseconds',
    'ns': 'nanoseconds',
    'V': 'volts',
    'A': 'amps',
    'mA': 'milliamps',
    'J': 'joules',
}


def parse_value(value_string):
    """parses a string representing a numerical value and returns
    a tuple (value, units), where value will be either int or float,
    and units will be a string representing the units or None."""
    match = VALUE_REGEX.search(value_string)
    if match:
        vs = match.group(1)
        value = float(vs) if '.' in vs else int(vs)
        us = match.group(2)
        units = UNITS_MAP.get(us, us)
        return (value, units)
    else:
        return (value_string, None)


def get_meansd(values):
    """Returns mean and standard deviation of the specified values."""
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    sd = math.sqrt(sum([(v - mean) ** 2 for v in values]) / len(values))
    return mean, sd


def geomean(values):
    """Returns the geometric mean of the values."""
    return reduce(mul, values) ** (1.0 / len(values))


def capitalize(text):
    """Capitalises the specified text: first letter upper case,
    all subsequent letters lower case."""
    if not text:
        return ''
    return text[0].upper() + text[1:].lower()


def utc_to_local(dt):
    """Convert naive datetime to local time zone, assuming UTC."""
    return dt.replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())


def local_to_utc(dt):
    """Convert naive datetime to UTC, assuming local time zone."""
    return dt.replace(tzinfo=tz.tzlocal()).astimezone(tz.tzutc())


def load_class(classpath):
    """Loads the specified Python class. ``classpath`` must be a fully-qualified
    class name (i.e. namspaced under module/package)."""
    modname, clsname = classpath.rsplit('.', 1)
    return getattr(__import__(modname), clsname)


def get_pager():
    """Returns the name of the system pager program."""
    pager = os.getenv('PAGER')
    if pager is None:
        pager = find_executable('less')
    if pager is None:
        pager = find_executable('more')
    return pager


def enum_metaclass(enum_param, return_name=False, start=0):
    """
    Returns a ``type`` subclass that may be used as a metaclass for
    an enum.

    Paremeters:

        :enum_param: the name of class attribute that defines enum values.
                     The metaclass will add a class attribute for each value in
                     ``enum_param``. The value of the attribute depends on the type
                     of ``enum_param`` and on the values of ``return_name``. If
                     ``return_name`` is ``True``, then the value of the new attribute is
                     the name of that attribute; otherwise, if ``enum_param`` is a ``list``
                     or a ``tuple``, the value will be the index of that param in
                     ``enum_param``, optionally offset by ``start``, otherwise, it will
                     be assumed that ``enum_param`` implementa a dict-like inteface and
                     the value will be ``enum_param[attr_name]``.
        :return_name: If ``True``, the enum values will the names of enum attributes. If
                      ``False``, the default, the values will depend on the type of
                      ``enum_param`` (see above).
        :start: If ``enum_param`` is a list or a tuple, and ``return_name`` is ``False``,
                this specifies an "offset" that will be added to the index of the attribute
                within ``enum_param`` to form the value.


    """
    class __EnumMeta(type):
        def __new__(mcs, clsname, bases, attrs):
            cls = type.__new__(mcs, clsname, bases, attrs)
            values = getattr(cls, enum_param, [])
            if return_name:
                for name in values:
                    setattr(cls, name, name)
            else:
                if isinstance(values, list) or isinstance(values, tuple):
                    for i, name in enumerate(values):
                        setattr(cls, name, i + start)
                else:  # assume dict-like
                    for name in values:
                        setattr(cls, name, values[name])
            return cls
    return __EnumMeta


_bash_color_regex = re.compile('\x1b\[[0-9;]+m')


def strip_bash_colors(text):
    return _bash_color_regex.sub('', text)


def format_duration(seconds, sep=' ', order=['day', 'hour', 'minute', 'second']):  # pylint: disable=dangerous-default-value
    """
    Formats the specified number of seconds into human-readable duration.

    """
    if isinstance(seconds, timedelta):
        td = seconds
    else:
        td = timedelta(seconds=seconds)
    dt = datetime(1, 1, 1) + td
    result = []
    for item in order:
        value = getattr(dt, item, None)
        if item is 'day':
            value -= 1
        if not value:
            continue
        suffix = '' if value == 1 else 's'
        result.append('{} {}{}'.format(value, item, suffix))
    return sep.join(result)


def get_article(word):
    """
    Returns the appropriate indefinite article for the word (ish).

    .. note:: Indefinite article assignment in English is based on
              sound rather than spelling, so this will not work correctly
              in all case; e.g. this will return ``"a hour"``.

    """
    return'an' if word[0] in 'aoeiu' else 'a'


def get_random_string(length):
    """Returns a random ASCII string of the specified length)."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in xrange(length))


class LoadSyntaxError(Exception):

    def __init__(self, message, filepath, lineno):
        super(LoadSyntaxError, self).__init__(message)
        self.filepath = filepath
        self.lineno = lineno

    def __str__(self):
        message = 'Syntax Error in {}, line {}:\n\t{}'
        return message.format(self.filepath, self.lineno, self.message)


RAND_MOD_NAME_LEN = 30
BAD_CHARS = string.punctuation + string.whitespace
TRANS_TABLE = string.maketrans(BAD_CHARS, '_' * len(BAD_CHARS))


def to_identifier(text):
    """Converts text to a valid Python identifier by replacing all
    whitespace and punctuation."""
    return re.sub('_+', '_', text.translate(TRANS_TABLE))


def load_struct_from_python(filepath=None, text=None):
    """Parses a config structure from a .py file. The structure should be composed
    of basic Python types (strings, ints, lists, dicts, etc.)."""
    if not (filepath or text) or (filepath and text):
        raise ValueError('Exactly one of filepath or text must be specified.')
    try:
        if filepath:
            modname = to_identifier(filepath)
            mod = imp.load_source(modname, filepath)
        else:
            modname = get_random_string(RAND_MOD_NAME_LEN)
            while modname in sys.modules:  # highly unlikely, but...
                modname = get_random_string(RAND_MOD_NAME_LEN)
            mod = imp.new_module(modname)
            exec text in mod.__dict__  # pylint: disable=exec-used
        return dict((k, v)
                    for k, v in mod.__dict__.iteritems()
                    if not k.startswith('_'))
    except SyntaxError as e:
        raise LoadSyntaxError(e.message, filepath, e.lineno)


def load_struct_from_yaml(filepath=None, text=None):
    """Parses a config structure from a .yaml file. The structure should be composed
    of basic Python types (strings, ints, lists, dicts, etc.)."""
    if not (filepath or text) or (filepath and text):
        raise ValueError('Exactly one of filepath or text must be specified.')
    try:
        if filepath:
            with open(filepath) as fh:
                return yaml.load(fh)
        else:
            return yaml.load(text)
    except yaml.YAMLError as e:
        lineno = None
        if hasattr(e, 'problem_mark'):
            lineno = e.problem_mark.line  # pylint: disable=no-member
        raise LoadSyntaxError(e.message, filepath=filepath, lineno=lineno)


def load_struct_from_file(filepath):
    """
    Attempts to parse a Python structure consisting of basic types from the specified file.
    Raises a ``ValueError`` if the specified file is of unkown format; ``LoadSyntaxError`` if
    there is an issue parsing the file.

    """
    extn = os.path.splitext(filepath)[1].lower()
    if (extn == '.py') or (extn == '.pyc') or (extn == '.pyo'):
        return load_struct_from_python(filepath)
    elif extn == '.yaml':
        return load_struct_from_yaml(filepath)
    else:
        raise ValueError('Unknown format "{}": {}'.format(extn, filepath))


def open_file(filepath):
    """
    Open the specified file path with the associated launcher in an OS-agnostic way.

    """
    if os.name == 'nt':  # Windows
        return os.startfile(filepath)  # pylint: disable=no-member
    elif sys.platform == 'darwin':  # Mac OSX
        return subprocess.call(['open', filepath])
    else:  # assume Linux or similar running a freedesktop-compliant GUI
        return subprocess.call(['xdg-open', filepath])


def sha256(path, chunk=2048):
    """Calculates SHA256 hexdigest of the file at the specified path."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        buf = fh.read(chunk)
        while buf:
            h.update(buf)
            buf = fh.read(chunk)
    return h.hexdigest()


def urljoin(*parts):
    return '/'.join(p.rstrip('/') for p in parts)


# From: http://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python/
def istextfile(fileobj, blocksize=512):
    """ Uses heuristics to guess whether the given file is text or binary,
        by reading a single block of bytes from the file.
        If more than 30% of the chars in the block are non-text, or there
        are NUL ('\x00') bytes in the block, assume this is a binary file.
    """
    _text_characters = (b''.join(chr(i) for i in range(32, 127)) +
                        b'\n\r\t\f\b')

    block = fileobj.read(blocksize)
    if b'\x00' in block:
        # Files with null bytes are binary
        return False
    elif not block:
        # An empty file is considered a valid text file
        return True

    # Use translate's 'deletechars' argument to efficiently remove all
    # occurrences of _text_characters from the block
    nontext = block.translate(None, _text_characters)
    return float(len(nontext)) / len(block) <= 0.30
