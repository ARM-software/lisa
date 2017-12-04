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
import subprocess
import traceback
import logging
import random
import hashlib
from datetime import datetime, timedelta
from operator import mul
from StringIO import StringIO
from itertools import chain, cycle
from distutils.spawn import find_executable

import yaml
from dateutil import tz

from devlib.exception import TargetError
from devlib.utils.misc import (ABI_MAP, check_output, walk_modules,
                               ensure_directory_exists, ensure_file_directory_exists,
                               normalize, convert_new_lines, get_cpu_mask, unique,
                               escape_quotes, escape_single_quotes, escape_double_quotes,
                               isiterable, getch, as_relative, ranges_to_list, memoized,
                               list_to_ranges, list_to_mask, mask_to_list, which)

check_output_logger = logging.getLogger('check_output')


# Defined here rather than in wa.exceptions due to module load dependencies
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


def categorize(v):
    if hasattr(v, 'merge_with') and hasattr(v, 'merge_into'):
        return 'o'
    elif hasattr(v, 'iteritems'):
        return 'm'
    elif isiterable(v):
        return 's'
    elif v is None:
        return 'n'
    else:
        return 'c'


def merge_config_values(base, other):
    """
    This is used to merge two objects, typically when setting the value of a
    ``ConfigurationPoint``. First, both objects are categorized into

        c: A scalar value. Basically, most objects. These values
           are treated as atomic, and not mergeable.
        s: A sequence. Anything iterable that is not a dict or
           a string (strings are considered scalars).
        m: A key-value mapping. ``dict`` and its derivatives.
        n: ``None``.
        o: A mergeable object; this is an object that implements both
          ``merge_with`` and ``merge_into`` methods.

    The merge rules based on the two categories are then as follows:

        (c1, c2) --> c2
        (s1, s2) --> s1 . s2
        (m1, m2) --> m1 . m2
        (c, s) --> [c] . s
        (s, c) --> s . [c]
        (s, m) --> s . [m]
        (m, s) --> [m] . s
        (m, c) --> ERROR
        (c, m) --> ERROR
        (o, X) --> o.merge_with(X)
        (X, o) --> o.merge_into(X)
        (X, n) --> X
        (n, X) --> X

    where:

        '.'  means concatenation (for maps, contcationation of (k, v) streams
             then converted back into a map). If the types of the two objects
             differ, the type of ``other`` is used for the result.
        'X'  means "any category"
        '[]' used to indicate a literal sequence (not necessarily a ``list``).
             when this is concatenated with an actual sequence, that sequencies
             type is used.

    notes:

        - When a mapping is combined with a sequence, that mapping is
          treated as a scalar value.
        - When combining two mergeable objects, they're combined using
          ``o1.merge_with(o2)`` (_not_ using o2.merge_into(o1)).
        - Combining anything with ``None`` yields that value, irrespective
          of the order. So a ``None`` value is eqivalent to the corresponding
          item being omitted.
        - When both values are scalars, merging is equivalent to overwriting.
        - There is no recursion (e.g. if map values are lists, they will not
          be merged; ``other`` will overwrite ``base`` values). If complicated
          merging semantics (such as recursion) are required, they should be
          implemented within custom mergeable types (i.e. those that implement
          ``merge_with`` and ``merge_into``).

    While this can be used as a generic "combine any two arbitry objects"
    function, the semantics have been selected specifically for merging
    configuration point values.

    """
    cat_base = categorize(base)
    cat_other = categorize(other)

    if cat_base == 'n':
        return other
    elif cat_other == 'n':
        return base

    if cat_base == 'o':
        return base.merge_with(other)
    elif cat_other == 'o':
        return other.merge_into(base)

    if cat_base == 'm':
        if cat_other == 's':
            return merge_sequencies([base], other)
        elif cat_other == 'm':
            return merge_maps(base, other)
        else:
            message = 'merge error ({}, {}): "{}" and "{}"'
            raise ValueError(message.format(cat_base, cat_other, base, other))
    elif cat_base == 's':
        if cat_other == 's':
            return merge_sequencies(base, other)
        else:
            return merge_sequencies(base, [other])
    else:  # cat_base == 'c'
        if cat_other == 's':
            return merge_sequencies([base], other)
        elif cat_other == 'm':
            message = 'merge error ({}, {}): "{}" and "{}"'
            raise ValueError(message.format(cat_base, cat_other, base, other))
        else:
            return other


def merge_sequencies(s1, s2):
    return type(s2)(unique(chain(s1, s2)))


def merge_maps(m1, m2):
    return type(m2)(chain(m1.iteritems(), m2.iteritems()))


def merge_dicts_simple(base, other):
    result = base.copy()
    for key, value in (other or {}).iteritems():
        result[key] = merge_config_values(result.get(key), value)
    return result


def touch(path):
    with open(path, 'w'):
        pass


def get_object_name(obj):
    if hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, 'im_func'):
        return '{}.{}'.format(get_object_name(obj.im_class),
                              obj.im_func.func_name)
    elif hasattr(obj, 'func_name'):
        return obj.func_name
    elif hasattr(obj, '__name__'):
        return obj.__name__
    elif hasattr(obj, '__class__'):
        return obj.__class__.__name__
    return None


def resolve_cpus(name, target):
    """
    Returns a list of cpu numbers that corresponds to a passed name.
    Allowed formats are:
        - 'big'
        - 'little'
        - '<core_name> e.g. 'A15'
        - 'cpuX'
        - 'all' - returns all cpus
        - '' - Empty name will also return all cpus
    """
    cpu_list = range(target.number_of_cpus)

    # Support for passing cpu no directly
    if isinstance(name, int):
        cpu = name
        if cpu not in cpu_list:
            message = 'CPU{} is not available, must be in {}'
            raise ValueError(message.format(cpu, cpu_list))
        return [cpu]

    # Apply to all cpus
    if not name or name.lower() == 'all':
        return cpu_list
    # Deal with big.little substitution
    elif name.lower() == 'big':
        name = target.big_core
        if not name:
            raise ValueError('big core name could not be retrieved')
    elif name.lower() == 'little':
        name = target.little_core
        if not name:
            raise ValueError('little core name could not be retrieved')

    # Return all cores with specified name
    if name in target.core_names:
        return target.core_cpus(name)

    # Check if core number has been supplied.
    else:
        core_no = re.match('cpu([0-9]+)', name, re.IGNORECASE)
        if core_no:
            cpu = int(core_no.group(1))
            if cpu not in cpu_list:
                message = 'CPU{} is not available, must be in {}'
                raise ValueError(message.format(cpu, cpu_list))
            return [cpu]
        else:
            msg = 'Unexpected core name "{}"'
            raise ValueError(msg.format(name))

@memoized
def resolve_unique_domain_cpus(name, target):
    """
    Same as `resolve_cpus` above but only returns only the first cpu
    in each of the different frequency domains. Requires cpufreq.
    """
    cpus = resolve_cpus(name, target)
    if not target.has('cpufreq'):
        msg = 'Device does not appear to support cpufreq; ' \
              'Cannot obtain cpu domain information'
        raise TargetError(msg)

    unique_cpus = []
    domain_cpus = []
    for cpu in cpus:
        if cpu not in domain_cpus:
            domain_cpus = target.cpufreq.get_related_cpus(cpu)
        if domain_cpus[0] not in unique_cpus:
            unique_cpus.append(domain_cpus[0])
    return unique_cpus
