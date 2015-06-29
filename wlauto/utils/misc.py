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
from datetime import datetime, timedelta
from operator import mul, itemgetter
from StringIO import StringIO
from itertools import cycle, groupby
from functools import partial
from distutils.spawn import find_executable

import yaml
from dateutil import tz


# ABI --> architectures list
ABI_MAP = {
    'armeabi': ['armeabi', 'armv7', 'armv7l', 'armv7el', 'armv7lh'],
    'arm64': ['arm64', 'armv8', 'arm64-v8a'],
}


def preexec_function():
    # Ignore the SIGINT signal by setting the handler to the standard
    # signal handler SIG_IGN.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Change process group in case we have to kill the subprocess and all of
    # its children later.
    # TODO: this is Unix-specific; would be good to find an OS-agnostic way
    #       to do this in case we wanna port WA to Windows.
    os.setpgrp()


check_output_logger = logging.getLogger('check_output')


# Defined here rather than in wlauto.exceptions due to module load dependencies
class TimeoutError(Exception):
    """Raised when a subprocess command times out. This is basically a ``WAError``-derived version
    of ``subprocess.CalledProcessError``, the thinking being that while a timeout could be due to
    programming error (e.g. not setting long enough timers), it is often due to some failure in the
    environment, and there fore should be classed as a "user error"."""

    def __init__(self, command, output):
        super(TimeoutError, self).__init__('Timed out: {}'.format(command))
        self.command = command
        self.output = output

    def __str__(self):
        return '\n'.join([self.message, 'OUTPUT:', self.output or ''])


def check_output(command, timeout=None, ignore=None, **kwargs):
    """This is a version of subprocess.check_output that adds a timeout parameter to kill
    the subprocess if it does not return within the specified time."""
    # pylint: disable=too-many-branches
    if ignore is None:
        ignore = []
    elif isinstance(ignore, int):
        ignore = [ignore]
    elif not isinstance(ignore, list) and ignore != 'all':
        message = 'Invalid value for ignore parameter: "{}"; must be an int or a list'
        raise ValueError(message.format(ignore))
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')

    def callback(pid):
        try:
            check_output_logger.debug('{} timed out; sending SIGKILL'.format(pid))
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass  # process may have already terminated.

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               preexec_fn=preexec_function, **kwargs)

    if timeout:
        timer = threading.Timer(timeout, callback, [process.pid, ])
        timer.start()

    try:
        output, error = process.communicate()
    finally:
        if timeout:
            timer.cancel()

    retcode = process.poll()
    if retcode:
        if retcode == -9:  # killed, assume due to timeout callback
            raise TimeoutError(command, output='\n'.join([output, error]))
        elif ignore != 'all' and retcode not in ignore:
            raise subprocess.CalledProcessError(retcode, command, output='\n'.join([output, error]))
    return output, error


def walk_modules(path):
    """
    Given package name, return a list of all modules (including submodules, etc)
    in that package.

    """
    root_mod = __import__(path, {}, {}, [''])
    mods = [root_mod]
    for _, name, ispkg in pkgutil.iter_modules(root_mod.__path__):
        submod_path = '.'.join([path, name])
        if ispkg:
            mods.extend(walk_modules(submod_path))
        else:
            submod = __import__(submod_path, {}, {}, [''])
            mods.append(submod)
    return mods


def ensure_directory_exists(dirpath):
    """A filter for directory paths to ensure they exist."""
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return dirpath


def ensure_file_directory_exists(filepath):
    """
    A filter for file paths to ensure the directory of the
    file exists and the file can be created there. The file
    itself is *not* going to be created if it doesn't already
    exist.

    """
    ensure_directory_exists(os.path.dirname(filepath))
    return filepath


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


def merge_dicts(*args, **kwargs):
    if not len(args) >= 2:
        raise ValueError('Must specify at least two dicts to merge.')
    func = partial(_merge_two_dicts, **kwargs)
    return reduce(func, args)


def _merge_two_dicts(base, other, list_duplicates='all', match_types=False,  # pylint: disable=R0912,R0914
                     dict_type=dict, should_normalize=True, should_merge_lists=True):
    """Merge dicts normalizing their keys."""
    merged = dict_type()
    base_keys = base.keys()
    other_keys = other.keys()
    norm = normalize if should_normalize else lambda x, y: x

    base_only = []
    other_only = []
    both = []
    union = []
    for k in base_keys:
        if k in other_keys:
            both.append(k)
        else:
            base_only.append(k)
            union.append(k)
    for k in other_keys:
        if k in base_keys:
            union.append(k)
        else:
            union.append(k)
            other_only.append(k)

    for k in union:
        if k in base_only:
            merged[k] = norm(base[k], dict_type)
        elif k in other_only:
            merged[k] = norm(other[k], dict_type)
        elif k in both:
            base_value = base[k]
            other_value = other[k]
            base_type = type(base_value)
            other_type = type(other_value)
            if (match_types and (base_type != other_type) and
                    (base_value is not None) and (other_value is not None)):
                raise ValueError('Type mismatch for {} got {} ({}) and {} ({})'.format(k, base_value, base_type,
                                                                                       other_value, other_type))
            if isinstance(base_value, dict):
                merged[k] = _merge_two_dicts(base_value, other_value, list_duplicates, match_types, dict_type)
            elif isinstance(base_value, list):
                if should_merge_lists:
                    merged[k] = _merge_two_lists(base_value, other_value, list_duplicates, dict_type)
                else:
                    merged[k] = _merge_two_lists([], other_value, list_duplicates, dict_type)

            elif isinstance(base_value, set):
                merged[k] = norm(base_value.union(other_value), dict_type)
            else:
                merged[k] = norm(other_value, dict_type)
        else:  # Should never get here
            raise AssertionError('Unexpected merge key: {}'.format(k))

    return merged


def merge_lists(*args, **kwargs):
    if not len(args) >= 2:
        raise ValueError('Must specify at least two lists to merge.')
    func = partial(_merge_two_lists, **kwargs)
    return reduce(func, args)


def _merge_two_lists(base, other, duplicates='all', dict_type=dict):  # pylint: disable=R0912
    """
    Merge lists, normalizing their entries.

    parameters:

        :base, other: the two lists to be merged. ``other`` will be merged on
                      top of base.
        :duplicates: Indicates the strategy of handling entries that appear
                     in both lists. ``all`` will keep occurrences from both
                     lists; ``first`` will only keep occurrences from
                     ``base``; ``last`` will only keep occurrences from
                     ``other``;

                     .. note:: duplicate entries that appear in the *same* list
                               will never be removed.

    """
    if not isiterable(base):
        base = [base]
    if not isiterable(other):
        other = [other]
    if duplicates == 'all':
        merged_list = []
        for v in normalize(base, dict_type) + normalize(other, dict_type):
            if not _check_remove_item(merged_list, v):
                merged_list.append(v)
        return merged_list
    elif duplicates == 'first':
        base_norm = normalize(base, dict_type)
        merged_list = normalize(base, dict_type)
        for v in base_norm:
            _check_remove_item(merged_list, v)
        for v in normalize(other, dict_type):
            if not _check_remove_item(merged_list, v):
                if v not in base_norm:
                    merged_list.append(v)  # pylint: disable=no-member
        return merged_list
    elif duplicates == 'last':
        other_norm = normalize(other, dict_type)
        merged_list = []
        for v in normalize(base, dict_type):
            if not _check_remove_item(merged_list, v):
                if v not in other_norm:
                    merged_list.append(v)
        for v in other_norm:
            if not _check_remove_item(merged_list, v):
                merged_list.append(v)
        return merged_list
    else:
        raise ValueError('Unexpected value for list duplicates argument: {}. '.format(duplicates) +
                         'Must be in {"all", "first", "last"}.')


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


def normalize(value, dict_type=dict):
    """Normalize values. Recursively normalizes dict keys to be lower case,
    no surrounding whitespace, underscore-delimited strings."""
    if isinstance(value, dict):
        normalized = dict_type()
        for k, v in value.iteritems():
            key = k.strip().lower().replace(' ', '_')
            normalized[key] = normalize(v, dict_type)
        return normalized
    elif isinstance(value, list):
        return [normalize(v, dict_type) for v in value]
    elif isinstance(value, tuple):
        return tuple([normalize(v, dict_type) for v in value])
    else:
        return value


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


def convert_new_lines(text):
    """ Convert new lines to a common format.  """
    return text.replace('\r\n', '\n').replace('\r', '\n')


def escape_quotes(text):
    """Escape quotes, and escaped quotes, in the specified text."""
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\'', '\\\'').replace('\"', '\\\"')


def escape_single_quotes(text):
    """Escape single quotes, and escaped single quotes, in the specified text."""
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\'', '\'\\\'\'')


def escape_double_quotes(text):
    """Escape double quotes, and escaped double quotes, in the specified text."""
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\"', '\\\"')


def getch(count=1):
    """Read ``count`` characters from standard input."""
    if os.name == 'nt':
        import msvcrt  # pylint: disable=F0401
        return ''.join([msvcrt.getch() for _ in xrange(count)])
    else:  # assume Unix
        import tty  # NOQA
        import termios  # NOQA
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(count)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def isiterable(obj):
    """Returns ``True`` if the specified object is iterable and
    *is not a string type*, ``False`` otherwise."""
    return hasattr(obj, '__iter__') and not isinstance(obj, basestring)


def utc_to_local(dt):
    """Convert naive datetime to local time zone, assuming UTC."""
    return dt.replace(tzinfo=tz.tzutc()).astimezone(tz.tzlocal())


def local_to_utc(dt):
    """Convert naive datetime to UTC, assuming local time zone."""
    return dt.replace(tzinfo=tz.tzlocal()).astimezone(tz.tzutc())


def as_relative(path):
    """Convert path to relative by stripping away the leading '/' on UNIX or
    the equivant on other platforms."""
    path = os.path.splitdrive(path)[1]
    return path.lstrip(os.sep)


def get_cpu_mask(cores):
    """Return a string with the hex for the cpu mask for the specified core numbers."""
    mask = 0
    for i in cores:
        mask |= 1 << i
    return '0x{0:x}'.format(mask)


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


def which(name):
    """Platform-independent version of UNIX which utility."""
    if os.name == 'nt':
        paths = os.getenv('PATH').split(os.pathsep)
        exts = os.getenv('PATHEXT').split(os.pathsep)
        for path in paths:
            testpath = os.path.join(path, name)
            if os.path.isfile(testpath):
                return testpath
            for ext in exts:
                testpathext = testpath + ext
                if os.path.isfile(testpathext):
                    return testpathext
        return None
    else:  # assume UNIX-like
        try:
            return check_output(['which', name])[0].strip()
        except subprocess.CalledProcessError:
            return None


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
        raise LoadSyntaxError(e.message, e.filepath, e.lineno)


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
            lineno = e.problem_mark.line
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


def unique(alist):
    """
    Returns a list containing only unique elements from the input list (but preserves
    order, unlike sets).

    """
    result = []
    for item in alist:
        if item not in result:
            result.append(item)
    return result


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


def ranges_to_list(ranges_string):
    """Converts a sysfs-style ranges string, e.g. ``"0,2-4"``, into a list ,e.g ``[0,2,3,4]``"""
    values = []
    for rg in ranges_string.split(','):
        if '-' in rg:
            first, last = map(int, rg.split('-'))
            values.extend(xrange(first, last + 1))
        else:
            values.append(int(rg))
    return values


def list_to_ranges(values):
    """Converts a list, e.g ``[0,2,3,4]``, into a sysfs-style ranges string, e.g. ``"0,2-4"``"""
    range_groups = []
    for _, g in groupby(enumerate(values), lambda (i, x): i - x):
        range_groups.append(map(itemgetter(1), g))
    range_strings = []
    for group in range_groups:
        if len(group) == 1:
            range_strings.append(str(group[0]))
        else:
            range_strings.append('{}-{}'.format(group[0], group[-1]))
    return ','.join(range_strings)


def list_to_mask(values, base=0x0):
    """Converts the specified list of integer values into
    a bit mask for those values. Optinally, the list can be
    applied to an existing mask."""
    for v in values:
        base |= (1 << v)
    return base


def mask_to_list(mask):
    """Converts the specfied integer bitmask into a list of
    indexes of bits that are set in the mask."""
    size = len(bin(mask)) - 2  # because of "0b"
    return [size - i - 1 for i in xrange(size)
            if mask & (1 << size - i - 1)]
