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
from __future__ import division
from contextlib import contextmanager
from functools import partial, reduce, wraps
from itertools import groupby
from operator import itemgetter
from weakref import WeakKeyDictionary, WeakSet

import ctypes
import functools
import logging
import os
import pkgutil
import random
import re
import signal
import string
import subprocess
import sys
import threading
import types
import wrapt
import warnings


try:
    from contextlib import ExitStack
except AttributeError:
    from contextlib2 import ExitStack

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from past.builtins import basestring

# pylint: disable=redefined-builtin
from devlib.exception import HostError, TimeoutError


# ABI --> architectures list
ABI_MAP = {
    'armeabi': ['armeabi', 'armv7', 'armv7l', 'armv7el', 'armv7lh', 'armeabi-v7a'],
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
    0x42: {  # Broadcom
        0x516: {None: 'Vulcan'},
    },
    0x43: {  # Cavium
        0x0a1: {None: 'Thunderx'},
        0x0a2: {None: 'Thunderx81xx'},
    },
    0x4e: {  # Nvidia
        0x0: {None: 'Denver'},
    },
    0x50: {  # AppliedMicro
        0x0: {None: 'xgene'},
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
        0x800: {None: 'Falkor'},
    },
    0x53: {  # Samsung LSI
        0x001: {0x1: 'MongooseM1'},
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
    # Change process group in case we have to kill the subprocess and all of
    # its children later.
    # TODO: this is Unix-specific; would be good to find an OS-agnostic way
    #       to do this in case we wanna port WA to Windows.
    os.setpgrp()


check_output_logger = logging.getLogger('check_output')
# Popen is not thread safe. If two threads attempt to call it at the same time,
# one may lock up. See https://bugs.python.org/issue12739.
check_output_lock = threading.RLock()


def get_subprocess(command, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    with check_output_lock:
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   preexec_fn=preexec_function,
                                   **kwargs)
    return process


def check_subprocess_output(process, timeout=None, ignore=None, inputtext=None):
    output = None
    error = None
    # pylint: disable=too-many-branches
    if ignore is None:
        ignore = []
    elif isinstance(ignore, int):
        ignore = [ignore]
    elif not isinstance(ignore, list) and ignore != 'all':
        message = 'Invalid value for ignore parameter: "{}"; must be an int or a list'
        raise ValueError(message.format(ignore))

    try:
        output, error = process.communicate(inputtext, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        timeout_expired = e
    else:
        timeout_expired = None

    # Currently errors=replace is needed as 0x8c throws an error
    output = output.decode(sys.stdout.encoding or 'utf-8', "replace") if output else ''
    error = error.decode(sys.stderr.encoding or 'utf-8', "replace") if error else ''

    if timeout_expired:
        raise TimeoutError(process.args, output='\n'.join([output, error]))

    retcode = process.poll()
    if retcode and ignore != 'all' and retcode not in ignore:
        raise subprocess.CalledProcessError(retcode, process.args, output, error)

    return output, error


def check_output(command, timeout=None, ignore=None, inputtext=None, **kwargs):
    """This is a version of subprocess.check_output that adds a timeout parameter to kill
    the subprocess if it does not return within the specified time."""
    process = get_subprocess(command, **kwargs)
    return check_subprocess_output(process, timeout=timeout, ignore=ignore, inputtext=inputtext)


def walk_modules(path):
    """
    Given package name, return a list of all modules (including submodules, etc)
    in that package.

    :raises HostError: if an exception is raised while trying to import one of the
                       modules under ``path``. The exception will have addtional
                       attributes set: ``module`` will be set to the qualified name
                       of the originating module, and ``orig_exc`` will contain
                       the original exception.

    """

    def __try_import(path):
        try:
            return __import__(path, {}, {}, [''])
        except Exception as e:
            he = HostError('Could not load {}: {}'.format(path, str(e)))
            he.module = path
            he.exc_info = sys.exc_info()
            he.orig_exc = e
            raise he

    root_mod = __try_import(path)
    mods = [root_mod]
    if not hasattr(root_mod, '__path__'):
        # root is a module not a package -- nothing to walk
        return mods
    for _, name, ispkg in pkgutil.iter_modules(root_mod.__path__):
        submod_path = '.'.join([path, name])
        if ispkg:
            mods.extend(walk_modules(submod_path))
        else:
            submod = __try_import(submod_path)
            mods.append(submod)
    return mods

def redirect_streams(stdout, stderr, command):
    """
    Update a command to redirect a given stream to /dev/null if it's
    ``subprocess.DEVNULL``.

    :return: A tuple (stdout, stderr, command) with stream set to ``subprocess.PIPE``
        if the `stream` parameter was set to ``subprocess.DEVNULL``.
    """
    def redirect(stream, redirection):
        if stream == subprocess.DEVNULL:
            suffix = '{}/dev/null'.format(redirection)
        elif stream == subprocess.STDOUT:
            suffix = '{}&1'.format(redirection)
            # Indicate that there is nothing to monitor for stderr anymore
            # since it's merged into stdout
            stream = subprocess.DEVNULL
        else:
            suffix = ''

        return (stream, suffix)

    stdout, suffix1 = redirect(stdout, '>')
    stderr, suffix2 = redirect(stderr, '2>')

    command = 'sh -c {} {} {}'.format(quote(command), suffix1, suffix2)
    return (stdout, stderr, command)

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
    base_keys = list(base.keys())
    other_keys = list(other.keys())
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
        for k, v in value.items():
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

def sanitize_cmd_template(cmd):
    msg = (
        '''Quoted placeholder should not be used, as it will result in quoting the text twice. {} should be used instead of '{}' or "{}" in the template: '''
    )
    for unwanted in ('"{}"', "'{}'"):
        if unwanted in cmd:
            warnings.warn(msg + cmd, stacklevel=2)
            cmd = cmd.replace(unwanted, '{}')

    return cmd

def escape_quotes(text):
    """
    Escape quotes, and escaped quotes, in the specified text.

    .. note:: :func:`pipes.quote` should be favored where possible.
    """
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\'', '\\\'').replace('\"', '\\\"')


def escape_single_quotes(text):
    """
    Escape single quotes, and escaped single quotes, in the specified text.

    .. note:: :func:`pipes.quote` should be favored where possible.
    """
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\'', '\'\\\'\'')


def escape_double_quotes(text):
    """
    Escape double quotes, and escaped double quotes, in the specified text.

    .. note:: :func:`pipes.quote` should be favored where possible.
    """
    return re.sub(r'\\("|\')', r'\\\\\1', text).replace('\"', '\\\"')


def escape_spaces(text):
    """
    Escape spaces in the specified text

    .. note:: :func:`pipes.quote` should be favored where possible.
    """
    return text.replace(' ', '\\ ')


def getch(count=1):
    """Read ``count`` characters from standard input."""
    if os.name == 'nt':
        import msvcrt  # pylint: disable=F0401
        return ''.join([msvcrt.getch() for _ in range(count)])
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


def commonprefix(file_list, sep=os.sep):
    """
    Find the lowest common base folder of a passed list of files.
    """
    common_path = os.path.commonprefix(file_list)
    cp_split = common_path.split(sep)
    other_split = file_list[0].split(sep)
    last = len(cp_split) - 1
    if cp_split[last] != other_split[last]:
        cp_split = cp_split[:-1]
    return sep.join(cp_split)


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


# This matches most ANSI escape sequences, not just colors
_bash_color_regex = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

def strip_bash_colors(text):
    return _bash_color_regex.sub('', text)


def get_random_string(length):
    """Returns a random ASCII string of the specified length)."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


class LoadSyntaxError(Exception):

    @property
    def message(self):
        if self.args:
            return self.args[0]
        return str(self)

    def __init__(self, message, filepath, lineno):
        super(LoadSyntaxError, self).__init__(message)
        self.filepath = filepath
        self.lineno = lineno

    def __str__(self):
        message = 'Syntax Error in {}, line {}:\n\t{}'
        return message.format(self.filepath, self.lineno, self.message)


RAND_MOD_NAME_LEN = 30
BAD_CHARS = string.punctuation + string.whitespace
# pylint: disable=no-member
if sys.version_info[0] == 3:
    TRANS_TABLE = str.maketrans(BAD_CHARS, '_' * len(BAD_CHARS))
else:
    TRANS_TABLE = string.maketrans(BAD_CHARS, '_' * len(BAD_CHARS))


def to_identifier(text):
    """Converts text to a valid Python identifier by replacing all
    whitespace and punctuation and adding a prefix if starting with a digit"""
    if text[:1].isdigit():
        text = '_' + text
    return re.sub('_+', '_', str(text).translate(TRANS_TABLE))


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
            first, last = list(map(int, rg.split('-')))
            values.extend(range(first, last + 1))
        else:
            values.append(int(rg))
    return values


def list_to_ranges(values):
    """Converts a list, e.g ``[0,2,3,4]``, into a sysfs-style ranges string, e.g. ``"0,2-4"``"""
    range_groups = []
    for _, g in groupby(enumerate(values), lambda i_x: i_x[0] - i_x[1]):
        range_groups.append(list(map(itemgetter(1), g)))
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
    return [size - i - 1 for i in range(size)
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
def memoized(wrapped, instance, args, kwargs):  # pylint: disable=unused-argument
    """
    A decorator for memoizing functions and methods.

    .. warning:: this may not detect changes to mutable types. As long as the
                 memoized function was used with an object as an argument
                 before, the cached result will be returned, even if the
                 structure of the object (e.g. a list) has changed in the mean time.

    """
    func_id = repr(wrapped)

    def memoize_wrapper(*args, **kwargs):
        id_string = func_id + ','.join([__get_memo_id(a) for a in  args])
        id_string += ','.join('{}={}'.format(k, __get_memo_id(v))
                              for k, v in kwargs.items())
        if id_string not in __memo_cache:
            __memo_cache[id_string] = wrapped(*args, **kwargs)
        return __memo_cache[id_string]

    return memoize_wrapper(*args, **kwargs)

@contextmanager
def batch_contextmanager(f, kwargs_list):
    """
    Return a context manager that will call the ``f`` callable with the keyword
    arguments dict in the given list, in one go.

    :param f: Callable expected to return a context manager.

    :param kwargs_list: list of kwargs dictionaries to be used to call ``f``.
    :type kwargs_list: list(dict)
    """
    with ExitStack() as stack:
        for kwargs in kwargs_list:
            stack.enter_context(f(**kwargs))
        yield


@contextmanager
def nullcontext(enter_result=None):
    """
    Backport of Python 3.7 ``contextlib.nullcontext``

    This context manager does nothing, so it can be used as a default
    placeholder for code that needs to select at runtime what context manager
    to use.

    :param enter_result: Object that will be bound to the target of the with
        statement, or `None` if nothing is specified.
    :type enter_result: object
    """
    yield enter_result


class tls_property:
    """
    Use it like `property` decorator, but the result will be memoized per
    thread. When the owning thread dies, the values for that thread will be
    destroyed.

    In order to get the values, it's necessary to call the object
    given by the property. This is necessary in order to be able to add methods
    to that object, like :meth:`_BoundTLSProperty.get_all_values`.

    Values can be set and deleted as well, which will be a thread-local set.
    """

    @property
    def name(self):
        return self.factory.__name__

    def __init__(self, factory):
        self.factory = factory
        # Lock accesses to shared WeakKeyDictionary and WeakSet
        self.lock = threading.RLock()

    def __get__(self, instance, owner=None):
        return _BoundTLSProperty(self, instance, owner)

    def _get_value(self, instance, owner):
        tls, values = self._get_tls(instance)
        try:
            return tls.value
        except AttributeError:
            # Bind the method to `instance`
            f = self.factory.__get__(instance, owner)
            obj = f()
            tls.value = obj
            # Since that's a WeakSet, values will be removed automatically once
            # the threading.local variable that holds them is destroyed
            with self.lock:
                values.add(obj)
            return obj

    def _get_all_values(self, instance, owner):
        with self.lock:
            # Grab a reference to all the objects at the time of the call by
            # using a regular set
            tls, values = self._get_tls(instance=instance)
            return set(values)

    def __set__(self, instance, value):
        tls, values = self._get_tls(instance)
        tls.value = value
        with self.lock:
            values.add(value)

    def __delete__(self, instance):
        tls, values = self._get_tls(instance)
        with self.lock:
            values.discard(tls.value)
        del tls.value

    def _get_tls(self, instance):
        dct = instance.__dict__
        name = self.name
        try:
            # Using instance.__dict__[self.name] is safe as
            # getattr(instance, name) will return the property instead, as
            # the property is a descriptor
            tls = dct[name]
        except KeyError:
            with self.lock:
                # Double check after taking the lock to avoid a race
                if name not in dct:
                    tls = (threading.local(), WeakSet())
                    dct[name] = tls

        return tls

    @property
    def basic_property(self):
        """
        Return a basic property that can be used to access the TLS value
        without having to call it first.

        The drawback is that it's not possible to do anything over than
        getting/setting/deleting.
        """
        def getter(instance, owner=None):
            prop = self.__get__(instance, owner)
            return prop()

        return property(getter, self.__set__, self.__delete__)

class _BoundTLSProperty:
    """
    Simple proxy object to allow either calling it to get the TLS value, or get
    some other informations by calling methods.
    """
    def __init__(self, tls_property, instance, owner):
        self.tls_property = tls_property
        self.instance = instance
        self.owner = owner

    def __call__(self):
        return self.tls_property._get_value(
            instance=self.instance,
            owner=self.owner,
        )

    def get_all_values(self):
        """
        Returns all the thread-local values currently in use in the process for
        that property for that instance.
        """
        return self.tls_property._get_all_values(
            instance=self.instance,
            owner=self.owner,
        )


class InitCheckpointMeta(type):
    """
    Metaclass providing an ``initialized`` and ``is_in_use`` boolean attributes
    on instances.

    ``initialized`` is set to ``True`` once the ``__init__`` constructor has
    returned. It will deal cleanly with nested calls to ``super().__init__``.

    ``is_in_use`` is set to ``True`` when an instance method is being called.
    This allows to detect reentrance.
    """
    def __new__(metacls, name, bases, dct, **kwargs):
        cls = super().__new__(metacls, name, bases, dct, **kwargs)
        init_f = cls.__init__

        @wraps(init_f)
        def init_wrapper(self, *args, **kwargs):
            self.initialized = False
            self.is_in_use = False

            # Track the nesting of super()__init__ to set initialized=True only
            # when the outer level is finished
            try:
                stack = self._init_stack
            except AttributeError:
                stack = []
                self._init_stack = stack

            stack.append(init_f)
            try:
                x = init_f(self, *args, **kwargs)
            finally:
                stack.pop()

            if not stack:
                self.initialized = True
                del self._init_stack

            return x

        cls.__init__ = init_wrapper

        # Set the is_in_use attribute to allow external code to detect if the
        # methods are about to be re-entered.
        def make_wrapper(f):
            if f is None:
                return None

            @wraps(f)
            def wrapper(self, *args, **kwargs):
                f_ = f.__get__(self, self.__class__)
                initial_state = self.is_in_use
                try:
                    self.is_in_use = True
                    return f_(*args, **kwargs)
                finally:
                    self.is_in_use = initial_state

            return wrapper

        # This will not decorate methods defined in base classes, but we cannot
        # use inspect.getmembers() as it uses __get__ to bind the attributes to
        # the class, making staticmethod indistinguishible from instance
        # methods.
        for name, attr in cls.__dict__.items():
            # Only wrap the methods (exposed as functions), not things like
            # classmethod or staticmethod
            if (
                name not in ('__init__', '__new__') and
                isinstance(attr, types.FunctionType)
            ):
                setattr(cls, name, make_wrapper(attr))
            elif isinstance(attr, property):
                prop = property(
                    fget=make_wrapper(attr.fget),
                    fset=make_wrapper(attr.fset),
                    fdel=make_wrapper(attr.fdel),
                    doc=attr.__doc__,
                )
                setattr(cls, name, prop)

        return cls


class InitCheckpoint(metaclass=InitCheckpointMeta):
    """
    Inherit from this class to set the :class:`InitCheckpointMeta` metaclass.
    """
    pass


def groupby_value(dct):
    """
    Process the input dict such that all keys sharing the same values are
    grouped in a tuple, used as key in the returned dict.
    """
    key = itemgetter(1)
    items = sorted(dct.items(), key=key)
    return {
        tuple(map(itemgetter(0), _items)): v
        for v, _items in groupby(items, key=key)
    }
