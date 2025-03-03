#    Copyright 2013-2018 ARM Limited
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

import errno
import hashlib
import importlib
import inspect
import logging
import math
import os
import pathlib
import random
import re
import shutil
import string
import subprocess
import sys
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import reduce  # pylint: disable=redefined-builtin
from operator import mul
from tempfile import gettempdir, NamedTemporaryFile
from time import sleep
from io import StringIO
# pylint: disable=wrong-import-position,unused-import
from itertools import chain, cycle

try:
    from shutil import which as find_executable
except ImportError:
    from distutils.spawn import find_executable  # pylint: disable=no-name-in-module, import-error

from dateutil import tz

# pylint: disable=wrong-import-order
from devlib.exception import TargetError
from devlib.utils.misc import (ABI_MAP, check_output, walk_modules,
                               ensure_directory_exists, ensure_file_directory_exists,
                               normalize, convert_new_lines, get_cpu_mask, unique,
                               isiterable, getch, as_relative, ranges_to_list, memoized,
                               list_to_ranges, list_to_mask, mask_to_list, which,
                               to_identifier, safe_extract, LoadSyntaxError)

check_output_logger = logging.getLogger('check_output')

file_lock_logger = logging.getLogger('file_lock')
at_write_logger = logging.getLogger('at_write')


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
    rows = [list(map(str, r)) for r in rows]
    max_cols = max(list(map(len, rows)))
    for row in rows:
        pad = max_cols - len(row)
        for _ in range(pad):
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
    align = [next(it) for _ in range(num_cols)]

    cols = list(zip(*rows))
    col_widths = [max(list(map(len, c))) for c in cols]
    if headers:
        col_widths = [max([c, len(h)]) for c, h in zip(col_widths, headers)]
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
    if not isinstance(item, str):
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
    mod = importlib.import_module(modname)
    cls = getattr(mod, clsname)
    if isinstance(cls, type):
        return cls
    else:
        raise ValueError(f'The classpath "{classpath}" does not point at a class: {cls}')


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
        td = timedelta(seconds=seconds or 0)
    dt = datetime(1, 1, 1) + td
    result = []
    for item in order:
        value = getattr(dt, item, None)
        if item == 'day':
            value -= 1
        if not value:
            continue
        suffix = '' if value == 1 else 's'
        result.append('{} {}{}'.format(value, item, suffix))
    return sep.join(result) if result else 'N/A'


def get_article(word):
    """
    Returns the appropriate indefinite article for the word (ish).

    .. note:: Indefinite article assignment in English is based on
              sound rather than spelling, so this will not work correctly
              in all case; e.g. this will return ``"a hour"``.

    """
    return 'an' if word[0] in 'aoeiu' else 'a'


def get_random_string(length):
    """Returns a random ASCII string of the specified length)."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def import_path(filepath, module_name=None):
    """
    Programmatically import the given Python source file under the name
    ``module_name``. If ``module_name`` is not provided, a stable name based on
    ``filepath`` will be created. Note that this module name cannot be relied
    on, so don't make write import statements assuming this will be stable in
    the future.
    """
    if not module_name:
        path = pathlib.Path(filepath).resolve()
        id_ = to_identifier(str(path))
        module_name = f'wa._user_import.{id_}'

    try:
        return sys.modules[module_name]
    except KeyError:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        else:
            # We could return the "module" object, but that would not take into
            # account any manipulation the module did on sys.modules when
            # executing. To be consistent with the import statement, re-lookup
            # the module name.
            return sys.modules[module_name]


def load_struct_from_python(filepath):
    """Parses a config structure from a .py file. The structure should be composed
    of basic Python types (strings, ints, lists, dicts, etc.)."""

    try:
        mod = import_path(filepath)
    except SyntaxError as e:
        raise LoadSyntaxError(e.message, filepath, e.lineno)
    else:
        return {
            k: v
            for k, v in inspect.getmembers(mod)
            if not k.startswith('_')
        }


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
    _text_characters = (b''.join(chr(i) for i in range(32, 127))
                        + b'\n\r\t\f\b')

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
    elif hasattr(v, 'items'):
        return 'm'
    elif isiterable(v):
        return 's'
    elif v is None:
        return 'n'
    else:
        return 'c'


# pylint: disable=too-many-return-statements,too-many-branches
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
    return type(m2)(chain(iter(m1.items()), iter(m2.items())))


def merge_dicts_simple(base, other):
    result = base.copy()
    for key, value in (other or {}).items():
        result[key] = merge_config_values(result.get(key), value)
    return result


def touch(path):
    with open(path, 'w'):
        pass


def get_object_name(obj):
    if hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, '__func__') and hasattr(obj, '__self__'):
        return '{}.{}'.format(get_object_name(obj.__self__.__class__),
                              obj.__func__.__name__)
    elif hasattr(obj, 'func_name'):
        return obj.__name__
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
    cpu_list = list(range(target.number_of_cpus))

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


def format_ordered_dict(od):
    """
    Provide a string representation of ordered dict that is similar to the
    regular dict representation, as that is more concise and easier to read
    than the default __str__ for OrderedDict.
    """
    return '{{{}}}'.format(', '.join('{}={}'.format(k, v)
                                     for k, v in od.items()))


@contextmanager
def atomic_write_path(path, mode='w'):
    """
    Gets a file path to write to which will be replaced with the original
     file path to simulate an atomic write from the point of view
    of other processes. This is achieved by writing to a tmp file and
    replacing the exiting file to prevent inconsistencies.
    """
    tmp_file = None
    try:
        tmp_file = NamedTemporaryFile(mode=mode, delete=False,
                                      suffix=os.path.basename(path))
        at_write_logger.debug('')
        yield tmp_file.name
        os.fsync(tmp_file.file.fileno())
    finally:
        if tmp_file:
            tmp_file.close()
    at_write_logger.debug('Moving {} to {}'.format(tmp_file.name, path))
    safe_move(tmp_file.name, path)


def safe_move(src, dst):
    """
    Taken from: https://alexwlchan.net/2019/03/atomic-cross-filesystem-moves-in-python/

    Rename a file from ``src`` to ``dst``.

    *   Moves must be atomic.  ``shutil.move()`` is not atomic.
    *   Moves must work across filesystems and ``os.rename()`` can
        throw errors if run across filesystems.

    So we try ``os.rename()``, but if we detect a cross-filesystem copy, we
    switch to ``shutil.move()`` with some wrappers to make it atomic.
    """
    try:
        os.rename(src, dst)
    except OSError as err:

        if err.errno == errno.EXDEV:
            # Generate a unique ID, and copy `<src>` to the target directory
            # with a temporary name `<dst>.<ID>.tmp`.  Because we're copying
            # across a filesystem boundary, this initial copy may not be
            # atomic.  We intersperse a random UUID so if different processes
            # are copying into `<dst>`, they don't overlap in their tmp copies.
            copy_id = uuid.uuid4()
            tmp_dst = "%s.%s.tmp" % (dst, copy_id)
            shutil.copyfile(src, tmp_dst)

            # Then do an atomic rename onto the new name, and clean up the
            # source image.
            os.rename(tmp_dst, dst)
            os.unlink(src)
        else:
            raise


@contextmanager
def lock_file(path, timeout=30):
    """
    Enable automatic locking and unlocking of a file path given. Used to
    prevent synchronisation issues between multiple wa processes.
    Uses a default timeout of 30 seconds which should be overridden for files
    that are expect to be unavailable for longer periods of time.
    """

    # Import here to avoid circular imports
    # pylint: disable=wrong-import-position,cyclic-import, import-outside-toplevel
    from wa.framework.exception import ResourceError

    locked = False
    l_file = 'wa-{}.lock'.format(path)
    l_file = os.path.join(gettempdir(), l_file.replace(os.path.sep, '_'))
    file_lock_logger.debug('Acquiring lock on "{}"'.format(path))
    try:
        while timeout:
            try:
                open(l_file, 'x').close()
                locked = True
                file_lock_logger.debug('Lock acquired on "{}"'.format(path))
                break
            except FileExistsError:
                msg = 'Failed to acquire lock on "{}" Retrying...'
                file_lock_logger.debug(msg.format(l_file))
                sleep(1)
                timeout -= 1
        else:
            msg = 'Failed to acquire lock file "{}" within the timeout. \n' \
                  'If there are no other running WA processes please delete ' \
                  'this file and retry.'
            raise ResourceError(msg.format(os.path.abspath(l_file)))
        yield
    finally:
        if locked and os.path.exists(l_file):
            os.remove(l_file)
            file_lock_logger.debug('Lock released "{}"'.format(path))
