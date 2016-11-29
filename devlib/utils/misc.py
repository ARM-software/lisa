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
import string
import threading
import signal
import subprocess
import pkgutil
import logging
import random
import ctypes
from operator import itemgetter
from itertools import groupby
from functools import partial

import wrapt

# ABI --> architectures list
ABI_MAP = {
    'armeabi': ['armeabi', 'armv7', 'armv7l', 'armv7el', 'armv7lh'],
    'arm64': ['arm64', 'armv8', 'arm64-v8a', 'aarch64'],
}

# Vendor ID --> CPU part ID --> CPU variant ID --> Core Name
# None means variant is not used.
CPU_PART_MAP = {
    0x41: {  # ARM
        0x926: {None: 'ARM926'},
        0x946: {None: 'ARM946'},
        0x966: {None: 'ARM966'},
        0xb02: {None: 'ARM11MPCore'},
        0xb36: {None: 'ARM1136'},
        0xb56: {None: 'ARM1156'},
        0xb76: {None: 'ARM1176'},
        0xc05: {None: 'A5'},
        0xc07: {None: 'A7'},
        0xc08: {None: 'A8'},
        0xc09: {None: 'A9'},
        0xc0e: {None: 'A17'},
        0xc0f: {None: 'A15'},
        0xc14: {None: 'R4'},
        0xc15: {None: 'R5'},
        0xc17: {None: 'R7'},
        0xc18: {None: 'R8'},
        0xc20: {None: 'M0'},
        0xc60: {None: 'M0+'},
        0xc21: {None: 'M1'},
        0xc23: {None: 'M3'},
        0xc24: {None: 'M4'},
        0xc27: {None: 'M7'},
        0xd01: {None: 'A32'},
        0xd03: {None: 'A53'},
        0xd04: {None: 'A35'},
        0xd07: {None: 'A57'},
        0xd08: {None: 'A72'},
        0xd09: {None: 'A73'},
    },
    0x4e: {  # Nvidia
        0x0: {None: 'Denver'},
    },
    0x51: {  # Qualcomm
        0x02d: {None: 'Scorpion'},
        0x04d: {None: 'MSM8960'},
        0x06f: {  # Krait
            0x2: 'Krait400',
            0x3: 'Krait450',
        },
        0x205: {0x1: 'KryoSilver'},
        0x211: {0x1: 'KryoGold'},
    },
    0x56: {  # Marvell
        0x131: {
            0x2: 'Feroceon 88F6281',
        }
    },
}


def get_cpu_name(implementer, part, variant):
    part_data = CPU_PART_MAP.get(implementer, {}).get(part, {})
    if None in part_data:  # variant does not determine core Name for this vendor
        name = part_data[None]
    else:
        name = part_data.get(variant)
    return name


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


# Defined here rather than in devlib.exceptions due to module load dependencies
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


def check_output(command, timeout=None, ignore=None, inputtext=None, **kwargs):
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
                               stdin=subprocess.PIPE,
                               preexec_fn=preexec_function, **kwargs)

    if timeout:
        timer = threading.Timer(timeout, callback, [process.pid, ])
        timer.start()

    try:
        output, error = process.communicate(inputtext)
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
            return check_output(['which', name])[0].strip()  # pylint: disable=E1103
        except subprocess.CalledProcessError:
            return None


_bash_color_regex = re.compile('\x1b\\[[0-9;]+m')


def strip_bash_colors(text):
    return _bash_color_regex.sub('', text)


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


__memo_cache = {}


def reset_memo_cache():
    __memo_cache.clear()


def __get_memo_id(obj):
    """
    An object's id() may be re-used after an object is freed, so it's not
    sufficiently unique to identify params for the memo cache (two different
    params may end up with the same id). this attempts to generate a more unique
    ID string.
    """
    obj_id = id(obj)
    try:
        return '{}/{}'.format(obj_id, hash(obj))
    except TypeError:  # obj is not hashable
        obj_pyobj = ctypes.cast(obj_id, ctypes.py_object)
        # TODO: Note: there is still a possibility of a clash here. If Two
        # different objects get assigned the same ID, an are large and are
        # identical in the first thirty two bytes. This shouldn't be much of an
        # issue in the current application of memoizing Target calls, as it's very
        # unlikely that a target will get passed large params; but may cause
        # problems in other applications, e.g. when memoizing results of operations
        # on large arrays. I can't really think of a good way around that apart
        # form, e.g., md5 hashing the entire raw object, which will have an
        # undesirable impact on performance.
        num_bytes = min(ctypes.sizeof(obj_pyobj), 32)
        obj_bytes = ctypes.string_at(ctypes.addressof(obj_pyobj), num_bytes)
        return '{}/{}'.format(obj_id, obj_bytes)


@wrapt.decorator
def memoized(wrapped, instance, args, kwargs):
    """A decorator for memoizing functions and methods."""
    func_id = repr(wrapped)

    def memoize_wrapper(*args, **kwargs):
        id_string = func_id + ','.join([__get_memo_id(a) for a in  args])
        id_string += ','.join('{}={}'.format(k, v)
                              for k, v in kwargs.iteritems())
        if id_string not in __memo_cache:
            __memo_cache[id_string] = wrapped(*args, **kwargs)
        return __memo_cache[id_string]

    return memoize_wrapper(*args, **kwargs)

