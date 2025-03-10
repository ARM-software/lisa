#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
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

import abc
import argparse
import collections
import contextlib
import copy
import datetime
import enum
import fcntl
import fnmatch
import functools
import gc
import glob
import gzip
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import lzma
import math
import multiprocessing
import mimetypes
import numbers
import os
import os.path
import pickle
import pathlib
import queue
import re
import select
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import tempfile
import types
import urllib.parse
import uuid

import requests
import ruamel.yaml

# If these modules are not available, DBus features will not be used.
try:
    import pydbus
    import gi.repository
    from gi.repository import GLib
except ImportError:
    DBUS_CAN_BE_ENABLED = False
else:
    DBUS_CAN_BE_ENABLED = True


def mask_signals(unblock=False):
    # Allow the handler to run again, now that we know it is safe to have
    # exceptions propagating.
    if unblock:
        try:
            sig_exception_lock.release()
        # We don't care if the lock was unlocked.
        except RuntimeError:
            pass

    action = signal.SIG_UNBLOCK if unblock else signal.SIG_BLOCK
    signal.pthread_sigmask(action, {
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGHUP,
    })


def filter_keys(mapping, remove=None, keep=None):
    return {
        k: v
        for k, v in mapping.items()
        if (
            (remove is None or k not in remove)
            and (keep is None or k in keep)
        )
    }


def natural_sort_key(s):
    """
    Key suitable for alphanumeric sort, but sorts numbers contained in the
    string as numbers.
    """
    def parse_int(s):
        try:
            return int(s)
        except ValueError:
            return s
    return tuple(parse_int(x) for x in re.split(r'(\d+)', s))


sig_exception_lock = threading.Lock()


def raise_sig_exception(sig, frame):
    """Turn some signals into exceptions that can be caught by user code."""
    # Mask signals here to avoid killing the process when it is handling the
    # exception. We assume that nothing else will happen afterwards, so there
    # should be no need for unmasking these signals. If it is needed, it should
    # be done explicitly using mask_signals(unblock=True).
    # To avoid competing signal handlers, we acquire a lock that will be
    # released only when the signals are unblocked explicitly by the code. In
    # the meantime, this handler will not do anything. This avoids the handler
    # raising an exception while the exception raised from a previous
    # invocation is being handled.
    if sig_exception_lock.acquire(blocking=False):
        mask_signals()

        if sig == signal.SIGTERM or sig == signal.SIGINT or sig == signal.SIGHUP:
            # Raise KeyboardInterrupt just like for SIGINT, so that the logging
            # module does not swallow if it is raised when logging functions
            # are executing.
            raise KeyboardInterrupt()
        else:
            error(f'Unknown signal {sig}')
    else:
        return


signal.signal(signal.SIGTERM, raise_sig_exception)
signal.signal(signal.SIGINT, raise_sig_exception)
signal.signal(signal.SIGHUP, raise_sig_exception)

# Upon catching these interrupts, the script will cleanly exit and will save
# the current state to a report.
SILENT_EXCEPTIONS = (KeyboardInterrupt, BrokenPipeError)


class MLString:
    """
    Efficient multiline string, behaving mostly like an str.

    Such string can be called to append a line to it, and str methods can be
    used. Note that these methods will return a rendered string and will not
    modify the object itself.
    """

    def __init__(self, lines=None, empty=False):
        self._empty = empty
        self._line_list = list(lines) if lines else []
        # Keep a cache of the rendered string. This favors speed over memory.
        self._rendered = None

    def tabulate(self, separator=' ', filler=None):
        """
        Return a new MLString with the lines formatted so that the columns
        are properly aligned.

        separator is a regex so splitting on multiple delimiters at once is
        possible.
        """

        if filler is None:
            filler = separator

        split_regex = re.compile(separator)

        def do_split(line):
            return (cell for cell in split_regex.split(line) if cell)

        # "Transpose" the list of lines into a list of columns
        table_cols = list(itertools.zip_longest(
            *(do_split(line) for line in self._line_list),
            fillvalue=''
        ))

        # Compute the width of each column according to the longest cell in it
        table_cols_width = (
            # +1 to keep at least one filler string between columns
            max(len(cell) for cell in col) + 1
            for col in table_cols
        )

        # Reformat all cells to fit the width of their column
        table_cols = [
            [
                '{cell:{filler}<{w}}'.format(cell=cell, filler=filler, w=width)
                for cell in col
            ]
            for width, col in zip(table_cols_width, table_cols)
        ]
        # Transpose back the columns to lines
        table_lines = (
            ''.join(cells).rstrip(filler)
            for cells in zip(*table_cols)
        )

        return MLString(lines=table_lines)

    def __call__(self, line=''):
        """Append a line by simply calling the object."""
        if not self._empty:
            self._line_list.append(line)
            # Invalidate the previously rendered string
            self._rendered = None

    def __str__(self):
        string = self._rendered
        # Render the string only once if the list has not been modified.
        if string is None:
            string = '\n'.join(str(line) for line in self._line_list)
            self._rendered = string

        return string

    def __iter__(self):
        return iter(self._line_list)

    def __bool__(self):
        return bool(self._line_list)

    def __getattr__(self, attr):
        """Make all strings methods available."""
        return getattr(str(self), attr)


class BisectRet(enum.Enum):
    """
    Git bisect return code as described in git bisect run documentation.
    """

    # Not Applicable. Used when the step does not want to take part in the
    # bisect decision
    NA = -1
    # Commit is non testable and must be skipped by git bisect
    UNTESTABLE = 125
    GOOD = 0
    BAD = 1
    # Bisect must abort due to non-recoverable error (the board can't boot
    # anymore for example)
    ABORT = 253
    # Yield to the caller of bisector, without taking any decision
    YIELD = 252

    @property
    def lower_name(self):
        """
        Make the lowercase name available through a property so that it can be
        used directly from format strings (they cannot call functions).
        """
        return self.name.lower()

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag, node.name)

    @classmethod
    def from_yaml(cls, constructor, node):
        name = constructor.construct_scalar(node)
        return cls.__members__[name]


# monkey-patch the class since class attributes defined in the class's scope
# are post-processed by the enum.Enum metaclass. That means we use
# register_class() instead of ruamel.yaml.yaml_object() decorator as well.
BisectRet.yaml_tag = '!git-bisect'

# This will map to an "abort" meaning for "git bisect run"
GENERIC_ERROR_CODE = 254
assert GENERIC_ERROR_CODE not in (e.value for e in BisectRet)


def parse_step_options(opts_seq):
    """
    Parse the steps options passed with -o command line parameters.

    :param opts_seq: sequence of command line options as returned by argparse.
                     Their format is: <name or category pattern>.<key>=<value>
                     If <name or cat pattern> is missing, '*' is assumed. It
                     will also be passed to all functions. The pattern is
                     matched using fnmatch.  If <value> is missing, True is
                     assumed.

    :returns: An OrderedDict that maps steps name pattern to a dict of key and
              values.
    """

    # Use an OrderedDict so that the command line order is used to know what
    # option overrides which one.
    options_map = collections.OrderedDict()

    for opt in opts_seq:
        step_k_v = opt.split('=', 1)

        step_k = step_k_v[0]
        step_k = step_k.split('.', 1)
        if len(step_k) >= 2:
            step_pattern, k = step_k
        else:
            # If no pattern was specified, assume '*'
            step_pattern = '*'
            k = step_k[0]

        # Replace '-' by '_' in key names.
        k = k.replace('-', '_')
        v = step_k_v[1] if len(step_k_v) >= 2 else ''
        options_map.setdefault(step_pattern, dict())[k] = v

    return options_map


def parse_iterations(opt):
    """Parse a string representing the number of iterations to be run.
    Special value "inf" means infinite number of iterations.
    """
    if opt == 'inf':
        return opt

    try:
        i = int(opt)
    except ValueError:
        raise argparse.ArgumentTypeError('iteration number must be an integer or "inf" for infinity.')
    if i < 0:
        raise argparse.ArgumentTypeError('iteration number must be positive.')

    return i


def parse_timeout(timeout):
    """Parse a string representing a timeout.

    It handles "inf" for infinite timeout (i.e. no timeout), and the s, m, h
    suffixes.  Default unit is seconds.
    """
    if timeout is None or timeout == 'inf':
        return 0

    kwargs_map = {'h': 'hours', 'm': 'minutes', 's': 'seconds'}
    suffix = timeout[-1]
    if suffix.isdigit():
        arg_name = kwargs_map['s']
    else:
        try:
            arg_name = kwargs_map[suffix]
        except KeyError as e:
            raise argparse.ArgumentTypeError('Cannot parse time spec "{timeout}": unrecognized suffix "{suffix}"'.format(**locals())) from e
        timeout = timeout[:-len(suffix)]

    try:
        delta = datetime.timedelta(**{arg_name: int(timeout)})
    except ValueError as e:
        raise argparse.ArgumentTypeError('Cannot parse time spec "{timeout}"'.format(**locals())) from e
    return int(delta.total_seconds())


class Param(abc.ABC):
    """
    Abstract Base Class of steps parameter.

    Such parameters take care of parsing strings specified on the command line
    and convert them to the right type before they are passed to the method.

    When no value is specified for such parameter, True boolean is assumed.
    Subclasses need to handle that with a default value, or raise an
    argparse.ArgumentTypeError exception as appropriate.
    """

    type_desc = None
    """Short description of the type. It will be displayed in the help."""

    def __init__(self, desc=''):
        self.help = desc

    def parse(self, val):
        """
        Parse an arbitrary value.

        Some types will be accepted without modification, str will undergo
        parsing.
        """
        if isinstance(val, str):
            val = self.parse_str(val)

        if val is not Default:
            excep = argparse.ArgumentTypeError('Invalid value format.')
            try:
                res = self.validate_val(val)
            except Exception as e:
                raise excep from e

            if not res:
                raise excep

        return val

    def validate_val(self, val):
        """
        Validate an object to be used as the value of the parameter.

        The validation is done after string parsing. That will catch non-string
        objects passed to the function, or non-string objects coming from the
        YAML file.

        :return: True if the object satisfies the properties, False otherwise
        """
        return True

    def __str__(self):
        return self.__class__.__name__ + '<' + self.type_desc + '>'

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def parse_str(self, val):
        """
        Parse a string argument.

        Strings are passed when the option is specified on the command line so
        that it is converted to the right type before being handled to the
        methods.
        """
        pass


class TimeoutParam(Param):
    """Step parameter holding a timeout value."""
    type_desc = 'int or "inf"'

    def parse_str(self, val):
        return parse_timeout(val)

    def validate_val(self, val):
        return isinstance(val, numbers.Integral)


class IterationParam(Param):
    """Step parameter holding a number of iterations."""
    type_desc = 'int or "inf"'

    def parse_str(self, val):
        return parse_iterations(val)

    def validate_val(self, val):
        return isinstance(val, numbers.Integral) or val == 'inf'


class BoolOrStrParam(Param):
    """Step parameter holding a string."""
    type_desc = 'str'

    @property
    def type_desc(self):
        if self.allow_empty:
            return 'str'
        else:
            return 'non-empty str'

    def __init__(self, *args, allow_empty=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_empty = allow_empty

    def parse_str(self, val):
        if not self.allow_empty and not val:
            raise argparse.ArgumentTypeError('A non-empty string must be specified')
        return val

    def validate_val(self, val):
        return isinstance(val, str) or isinstance(val, bool)


class CommaListParam(Param):
    """Step parameter holding a comma-separated list of strings."""
    type_desc = 'comma-separated list'

    def parse_str(self, val):
        return val.split(',')

    def validate_val(self, val):
        return all(isinstance(v, str) for v in val)


class CommaListRangesParam(CommaListParam):
    type_desc = 'comma-separated list of integer ranges'

    def parse_str(self, val):
        ranges = super().parse_str(val)

        values = list()
        for spec in ranges:
            try:
                first, last = spec.split('-')
            except ValueError:
                first = spec
                last = spec

            values.extend(i for i in range(int(first), int(last) + 1))

        return values

    def validate_val(self, val):
        return all(isinstance(v, numbers.Integral) for v in val)


class EnvListParam(Param):
    """Step parameter holding a list of environment variables values."""
    type_desc = 'env var list'

    def parse(self, val):
        # Convert Mapping[str, str] to Mapping[str, List[str]]
        if isinstance(val, collections.abc.Mapping):
            val = collections.OrderedDict(
                (var, val_list)
                if not isinstance(val_list, str) else
                (var, [val_list])

                for var, val_list in val.items()
            )

        return super().parse(val)

    def parse_str(self, val):
        var_separator = '%%'
        val_separator = '%'

        spec_list = val.split(var_separator)
        var_map = collections.OrderedDict()
        for spec in spec_list:
            var, val_spec = spec.split('=', 1)
            val_list = val_spec.split(val_separator)
            var_map[var] = val_list

        return var_map

    def validate_val(self, val):
        return (
            isinstance(val, collections.abc.Mapping) and
            all(
                isinstance(var, str) and not isinstance(val_list, str)
                for var, val_list in val.items()
            )
        )


class IntParam(Param):
    """Step parameter holding an integer."""
    type_desc = 'int'

    def parse_str(self, val):
        return int(val)

    def validate_val(self, val):
        return isinstance(val, numbers.Integral)


class BoolParam(Param):
    """Step parameter holding a boolean value."""
    type_desc = 'bool'

    def parse_str(self, val):
        val = val.lower()
        return (
            val == 'yes' or val == 'y' or
            val == 'true' or
            val == 'on' or
            val == '1' or
            # Empty string means the user just specified the option
            val == ''
        )

    def validate_val(self, val):
        return isinstance(val, bool)


class ChoiceOrBoolParam(Param):
    """
    Step parameter holding a boolean value or an item chosen among a predefined
    set.
    """

    @property
    def type_desc(self):
        return '{choices} or bool'.format(
            choices=','.join(
                f'"{choice}"'
                for choice in self.choices
            )
        )

    def __init__(self, choices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = choices

    def parse_str(self, val):
        if val in self.choices:
            return val
        else:
            return BoolParam().parse_str(val)

    def validate_val(self, val):
        return isinstance(val, bool) or val in self.choices


def info(msg):
    """Write a log message at the INFO level."""
    BISECTOR_LOGGER.info(msg)


def debug(msg):
    """Write a log message at the DEBUG level."""
    BISECTOR_LOGGER.debug(msg)


def warn(msg):
    """Write a log message at the WARNING level."""
    BISECTOR_LOGGER.warning(msg)


def error(msg):
    """Write a log message at the ERROR level."""
    BISECTOR_LOGGER.error(msg)


class _DefaultType:
    """
    Class of the Default singleton. It is used as a placeholder value for method parameters.
    When assigning the Default singleton to an attribute, it will:

    - Check if the attribute already exists. If yes, its value is retained.
    - Otherwise, lookup the attribute in the ``attr_init`` dict of the class.

    This behaviour allows setting a default value to an attribute, without
    overwriting a pre-existing value. Partial reinitialization of deserialized
    objects becomes easy, provided that arguments are assigned to attibutes
    before being through that attribute.
    """

    def __bool__(self):
        """Evaluate to False to allow this pattern:
            val = Default
            self.XXX = val or another_value
        This will store "another_value" in self.XXX .
        """
        return False

    def __str__(self):
        return 'Default'

    def __repr__(self):
        return str(self)


# Default is a singleton like None
Default = _DefaultType()


class SerializableMeta(type):
    """
    Metaclass for all serializable classes.

    It ensures the class will always have a ``name`` and ``cat``
    attributes, as well as constructing a ChainMap from the ``attr_init``
    dictionaries of its base classes. That means that usual inheritance rules
    will apply to the content of ``attr_init`` .
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dct = self.__dict__

        # Pick up the class attribute, or the attr_init and ultimately, just
        # use the name of the class itself. That is not recommended since it
        # will make the class name directly part of an external interface.
        self.name = (
            dct.get('name') or
            dct.get('attr_init', {}).get('name') or
            self.__name__
        )

        # Also make the category available for better introspection of the class
        self.cat = (
            dct.get('cat') or
            dct.get('attr_init', {}).get('cat')
        )

        # Make the class-specific attr_init available under the name _attr_init
        self._attr_init = dct.get('attr_init', {})

        # Turn attr_init attribute into a ChainMap that will look for the
        # values in all base classes, following the MRO. This could be done
        # as a property but it's more efficient to do it here once and for all
        # at classe creation time.
        self.attr_init = collections.ChainMap(*(
            base._attr_init for base in self.__mro__
            if hasattr(base, '_attr_init')
        ))

    def set_attr_init(cls, attr, val):
        """Set an initial value for instance attributes."""
        # Since attr_init is a ChainMap, the value will be stored in the
        # underlying dictionary for the given class only.
        cls.attr_init[attr] = val


class Serializable(metaclass=SerializableMeta):
    """
    Base class of all serializable classes.

    Subclasses MUST implement a ``yaml_tag`` class attribute, in order to
    have a stable tag associated with them that can survive accross class
    renames and refactoring.
    """

    # Default, should be overriden by subclasses
    attr_init = dict(
        name=None,
    )
    """All keys of this dictionary will be copied into fresh instance attributes
    when the class is instantiated.
    """

    dont_save = []
    "Attributes that will not be saved to the Pickle or YAML file."

    def __new__(cls, *args, **kwargs):
        # Avoid passing *args and **kwargs to object.__new__ since it will
        # complain otherwise.
        obj = super().__new__(cls)

        # Get a fresh independant copy of init values for every new instance.
        # Note that immutable types (like integers and strings) are not
        # actually copied, so the cost is pretty low.
        for attr, val in cls.attr_init.items():
            setattr(obj, attr, val)

        return obj

    def __setattr__(self, attr, val):
        """Use the default value if the attribute does not exist yet."""
        if val is Default:
            if hasattr(self, attr):
                return
            cls = type(self)
            try:
                val = cls.attr_init[attr]
            except KeyError as e:
                raise AttributeError(f'Cannot find attribute "{attr}"') from e

        super().__setattr__(attr, val)

    def __getstate__(self):
        """
        Determine what state must be preserved when the instance is serialized.

        Attributes with a value equal to the corresponding value in
        ``attr_init`` will not be serialized, since they will be recreated
        when the instance is deserialized. That limits the number of attributes
        that are actually stored, thereby limiting the scope of backward
        compatibility issues.  It also means that an update to the code could
        change the default value, so the deserialized instance will not
        necessarily be equal to the serialized one.

        Attributes listed in ``dont_save`` will not be serialized.
        """
        cls = type(self)
        attr_init_map = cls.attr_init
        dont_save_set = set(self.dont_save)
        dont_save_set.add('dont_save')
        return {
            k: v for k, v in self.__dict__.items()
            if not (
                # Don't save the attributes that are explicitly excluded
                (k in dont_save_set) or
                # Don't save the attributes that have the default init value.
                (k in attr_init_map and attr_init_map[k] == v)
            )
        }


class StepResultBase(Serializable):
    """
    Base class of all step results.

    Step results are returned by the :meth:`StepBase.run` method of steps.
    It must implement ``bisect_ret`` containing a value of :class:`BisectRet` .
    """

    def filtered_bisect_ret(self, steps_filter=None):
        """Dummy implementation to simplify bisect status aggreagation code."""
        return self.bisect_ret


class StepResult(StepResultBase):
    """
    Basic step result.

    This step results support storing the return values of multiple trials of a
    step, along its corresponding logs.
    """
    yaml_tag = '!basic-step-result'

    def __init__(self, step, res_list, bisect_ret):
        self.step = step
        self.bisect_ret = bisect_ret

        self.res_list = [
            {
                'ret': int(ret) if ret is not None else None,
                'log': self._format_log(log),
            }
            for ret, log in res_list
        ]

    def __str__(self):
        return 'exit status {self.ret}, bisect code {self.bisect_ret.name}'.format(self=self)

    @property
    def ret(self):
        """The meaningful return value is the one from the last trial."""
        return self.res_list[-1]['ret']

    @property
    def log(self):
        """Get the log that goes with the return code."""
        return self.res_list[-1]['log']

    @staticmethod
    def _format_log(log):
        """
        Format a log so it can be neatly serialized using the block style
        strings in YAML. This makes the YAML serialized object readable and
        greppable.
        """
        # If there are some non-printable characters in the string, or if it
        # encounters a newline preceeded by a blank space, it falls back to
        # double-quoted string so we just remove these white spaces before new
        # lines. This should not be an issue for the log.
        return '\n'.join(
            line.rstrip().replace('\t', ' ' * 4)
            for line in log.decode().splitlines()
        )


def terminate_process(p, kill_timeout):
    """
    Attempt to terminate a process `p`, and kill it after
    `kill_timeout` if it is not dead yet.
    """
    with contextlib.suppress(ProcessLookupError):
        p.terminate()
        time.sleep(0.1)
        if p.poll() is None:
            time.sleep(kill_timeout)
            p.kill()
            time.sleep(0.1)


def read_stdout(p, timeout=None, kill_timeout=3):
    """
    Read the standard output of a given process, and terminates it when the
    timeout expires.

    Note that it will not close stdout file descriptor, so
    the caller needs to take care of that, since it created the process.
    """

    stdout_fd = p.stdout.fileno()
    stdout_list = list()
    timed_out = False

    # fcntl command to get the size of the buffer of the pipe. If we read
    # that much, we are sure to drain the pipe completely.
    F_GETPIPE_SZ = 1032
    pipe_capacity = fcntl.fcntl(stdout_fd, F_GETPIPE_SZ)
    # We need to be able to check regularly if the process is alive, so
    # reading must not be blocking.
    os.set_blocking(stdout_fd, False)

    watch_stdout = True
    begin_ts = time.monotonic()
    while watch_stdout:
        # Use a timeout here so we have a chance checking that the process
        # is alive and that the timeout has not expired at a regular
        # interval.
        select.select([stdout_fd], [], [], 1)

        # Check the elapsed time, so we kill the process if it takes too
        # much time to complete
        if timeout and time.monotonic() - begin_ts >= timeout:
            timed_out = True
            terminate_process(p, kill_timeout)

        # In case the pipe is still opened by some daemonized child when
        # the subprocess we spawned has already died, we don't  want to
        # wait forever, so we do one last round and then close the pipe.
        watch_stdout = p.poll() is None

        # Drain the pipe by reading as much as possible. If that does not
        # drain the pipe for one reason or another, the remaining output
        # will be lost.
        try:
            stdout = os.read(stdout_fd, pipe_capacity)
        # The read would block but the fil descriptor is non-blocking.
        except BlockingIOError:
            stdout = b''
        # Either there is some data available, or all the write ends of the
        # pipe are closed.
        else:
            # os.read returns an empty bytestring when reaching EOF, which
            # means all writing ends of the pipe are closed.
            if not stdout:
                watch_stdout = False

        # Store the output and print it on the output.
        if stdout:
            write_stdout(stdout)
            stdout_list.append(stdout)

    ret = p.wait()
    assert ret is not None
    ret = None if timed_out else ret

    return (ret, b''.join(stdout_list))


def write_stdout(txt):
    sys.stdout.buffer.write(txt)
    sys.stdout.buffer.flush()
    LOG_FILE.buffer.write(txt)
    LOG_FILE.buffer.flush()


def call_process(cmd, *args, merge_stderr=True, **kwargs):
    """
    Call a given command with the given arguments and return its standard
    output.

    :param cmd: name of the command to run
    :param args: command line arguments passed to the command
    :param merge_stderr: merge stderr with stdout.
    :Variable keyword arguments: Forwarded to :func:`subprocess.check_output`
    """
    cmd = tuple(str(arg) for arg in cmd)
    if merge_stderr:
        kwargs['stderr'] = subprocess.STDOUT
    try:
        return subprocess.check_output(cmd, *args, **kwargs).decode()
    except subprocess.CalledProcessError as e:
        e.stdout = e.stdout.decode()
        raise


def git_cleanup(repo='./'):
    """Forcefully clean and reset a git repository."""
    info('Cleaning up git worktree (git reset --hard and git clean -fdx) ...')
    try:
        call_process(['git', '-C', repo, 'reset', '--hard'])
        call_process(['git', '-C', repo, 'clean', '-fdx'])
    except subprocess.CalledProcessError as e:
        warn(f'Failed to clean git worktree: {e}')


@contextlib.contextmanager
def enforce_git_cleanup(do_cleanup=True, repo='./'):
    """Context manager allowing to cleanup a git repository at entry and exit.

    :param do_cleanup: does the cleanup if True, otherwise does nothing.
    """
    if do_cleanup:
        git_cleanup(repo=repo)
    yield
    if do_cleanup:
        git_cleanup(repo=repo)


def get_git_sha1(repo='./', ref='HEAD', abbrev=12):
    """Get the SHA1 of given ref in the given git repository.

    :param abbrev: length of the SHA1 returned.
    """
    try:
        return call_process(
            ('git', '-C', repo, 'rev-parse', str(ref))
        ).strip()[:abbrev]
    except subprocess.CalledProcessError as e:
        err = e.stdout.replace('\n', '. ')
        debug(f'{repo} is not a Git repository: {err}')
        return '<commit sha1 not available>'


def get_steps_kwarg_parsers(cls, method_name):
    """
    Analyze a method to extract the parameters with command line parsers.

    The parser must be a :class:`Param` so it handles parsing of arguments
    coming from the command line. It must be specified in the ``options``
    class attribute.

    :param method_name: name of the method to analyze
    :param cls: class of the method to analyze
    :returns: map of parameter names to their Param object
    """

    method = getattr(cls, method_name)
    meth_options = cls.options.get(method_name, dict())
    return meth_options


def get_step_kwargs(step_cat, step_name, cls, method_name, user_options):
    """
    Compute the keyword arguments map for a given method that belongs to a
    specific step.

    :param step_cat: category of the step the method belongs to
    :param step_name: name of the step the method belongs to
    :param user_options: Map of steps categories or names patterns to options.
    :returns: A map suitable to pass as `**kwargs` to the method. Note that the
              option values are not modified, the method is expected to apply
              its own processing to convert strings to the expected types if
              necessary.
    """

    if not user_options:
        return dict()

    kwarg_name_set = set(get_steps_kwarg_parsers(cls, method_name).keys())

    # The precedence is given by the order on the command line.
    options = dict()
    for key in itertools.chain(
        match_step_name(step_cat, user_options.keys()),
        match_step_name(step_name, user_options.keys())
    ):
        options.update(user_options[key])

    kwargs = {
        k: v
        for k, v in options.items()
        if k in kwarg_name_set
    }

    return kwargs


class StepMeta(abc.ABCMeta, type(Serializable)):
    """
    Metaclass of all steps.

    Wraps ``__init__()`` and ``report()`` to preprocess the values of the
    parameters annotated with a command line option parser.
    """
    def __new__(meta_cls, name, bases, dct):
        # Wrap some methods to preprocess some parameters that have a parser
        # defined using the "options" class attribute.
        for method_name in (
            name for name in ('__init__', 'report')
            if name in dct
        ):
            wrapper = meta_cls.wrap_method(dct, method_name)
            dct[method_name] = wrapper

        return super().__new__(meta_cls, name, bases, dct)

    @staticmethod
    def wrap_method(dct, method_name):
        """Wrap a method to preprocess argument values using the parser given
        in dct['options'].
        """
        method = dct[method_name]
        sig = inspect.signature(method)

        # Stub object used to satisfy the API of get_steps_kwarg_parsers
        cls_stub = types.SimpleNamespace(**dct)
        parser_map = get_steps_kwarg_parsers(cls_stub, method_name)

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            # Map all the arguments to named parameters
            # Some arguments may not be present, but we want the exception to
            # raised when the method is called to make it more obvious.

            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            # Preprocess the values when they are an str
            for param, parser in parser_map.items():
                try:
                    val = bound_args.arguments[param]
                except KeyError:
                    continue

                try:
                    bound_args.arguments[param] = parser.parse(val)
                except Exception as e:
                    raise ValueError('Invalid value format "{val}" for option "{param}": {e}'.format(
                        e=e,
                        param=param,
                        val=val,
                    )) from e

            # Call the wrapped method with the preprocessed values.
            return method(*bound_args.args, **bound_args.kwargs)
        return wrapper


class IterationCounterStack(list):
    """list subclass that can be pretty printed as dot-separated list.
    This is intended to be used for nested iteration counters.
    """

    def __format__(self, fmt):
        return str(self).__format__(fmt)

    def __str__(self):
        return '.'.join(str(i) for i in self)


class StepABC(Serializable, metaclass=StepMeta):
    """
    Abstract Base Class of all steps.

    """

    options = dict(
        __init__={},
        report=dict(
            verbose=BoolParam('Increase verbosity'),
        ),
    )

    @abc.abstractmethod
    def __init__(self):
        """
        Keyword-only parameters can be annotated with an instance of
        :class:`Param` in order to support step-specific cmd-line parameters.
        """
        pass

    @abc.abstractmethod
    def run(self, i_stack, service_hub):
        """
        Run the step.

        The return values of this method will be collected and stored.
        """
        pass

    @abc.abstractmethod
    def report(self, step_res_seq,
        verbose=True
    ):
        """
        Display the list of results gathered when executing :meth:`run`.

        Keyword-only parameters are supported as for ``__init__()``
        """
        pass


class StepBase(StepABC):
    """
    Base class of most steps.

    It supports running shell commands, saving their stdout and stderr as well
    as their exit status.
    """

    # Class attributes will not be serialized, which makes the YAML report less
    # cluttered and limits the damages when removing attributes.
    attr_init = dict(
        cat='<no category>',
        name='<no name>',
        shell='/bin/bash',
        trials=1,
        cmd='',
        timeout=parse_timeout('inf'),
        # After sending SIGTERM, wait for kill_timeout and if the process is not
        # dead yet, send SIGKILL
        kill_timeout=3,
        bail_out=True,
        # Use systemd-run to avoid escaping daemons.
        use_systemd_run=False,
        env=collections.OrderedDict(),
    )
    options = dict(
        __init__=dict(
            cmd=BoolOrStrParam('shell command to be executed'),
            shell=BoolOrStrParam('shell to execute the command in'),
            trials=IntParam('number of times the command will be retried if it does not return 0'),
            timeout=TimeoutParam('timeout in seconds before sending SIGTERM to the command, or "inf" for infinite timeout'),
            kill_timeout=TimeoutParam('time to wait before sending SIGKILL after having sent SIGTERM'),
            bail_out=BoolParam('start a new iteration if the command fails, without executing remaining steps for this iteration'),
            use_systemd_run=BoolParam('use systemd-run to run the command. This allows cleanup of daemons spawned by the command (using cgroups), and using a private /tmp that is also cleaned up automatically'),
            env=EnvListParam('environment variables with a list of values that will be used for each iterations, wrapping around. The string format is: VAR1=val1%val2%...%%VAR2=val1%val2%.... In YAML, it is a map of var names to list of values. A single string can be supplied instead of a list of values.'),
        ),
        report=dict(
            verbose=BoolParam('increase verbosity'),
            show_basic=BoolParam('show command exit status for all iterations'),
            ignore_non_issue=BoolParam('consider only iteration with non-zero command exit status'),
            iterations=CommaListRangesParam('comma-separated list of iterations to consider. Inclusive ranges can be specified with <first>-<last>'),
            export_logs=BoolOrStrParam('export the logs to the given directory'),
        )
    )

    def __init__(self, name=Default, cat=Default,
                cmd=Default,
                shell=Default,
                trials=Default,
                timeout=Default,
                kill_timeout=Default,
                bail_out=Default,
                use_systemd_run=Default,
                env=Default,
            ):
        # Only assign the instance attribute if it is not the default coming
        # from the class attribute.
        self.cat = cat
        self.name = name
        self.shell = shell
        self.trials = trials
        self.timeout = timeout
        self.kill_timeout = kill_timeout
        self.bail_out = bail_out
        self.use_systemd_run = use_systemd_run
        self.env = env

        if self.use_systemd_run and not shutil.which('systemd-run'):
            warn('systemd-run not found in the PATH, not using systemd-run.')
            self.use_systemd_run = False

        # Assign first to consume Default
        self.cmd = cmd
        self.cmd = self.cmd.strip()

    def reinit(self, *args, **kwargs):
        # Calling __init__ again will only override specified parameters, the
        # other ones will keep their current value thanks to Default
        # behavior.
        self.__init__(*args, **kwargs)
        return self

    @classmethod
    def help(cls):
        out = MLString()
        indent = 4 * ' '
        title = '{cls.name} ({cls.attr_init[cat]})'.format(cls=cls)
        out('\n' + title + '\n' + '-' * len(title))

        base_list = [
            base.name for base in inspect.getmro(cls)
            if issubclass(base, StepBase) and not (
                base is StepBase
                or base is cls
            )
        ]
        if base_list:
            out('Inherits from: ' + ', '.join(base_list) + '\n')

        if cls.__doc__:
            out(indent + textwrap.dedent(cls.__doc__).strip().replace('\n', '\n' + indent) + '\n')

        for pretty, method_name in (
                ('run', '__init__'),
                ('report', 'report'),
        ):
            parser_map = get_steps_kwarg_parsers(cls, method_name)
            if not parser_map:
                continue
            out(indent + f'{pretty}:')
            for name, param in sorted(parser_map.items(), key=lambda k_v: k_v[0]):
                name = name.replace('_', '-')
                opt_header = indent + '  -o {name}= ({type_desc}) '.format(
                    name=name,
                    type_desc=param.type_desc,
                )

                content_indent = indent * 3
                opt_content = ('\n' + content_indent).join(
                    textwrap.wrap(param.help, width=80 - len(content_indent))
                )

                out(opt_header + '\n' + content_indent + opt_content + '\n')
            out('')

        return out

    def _run_cmd(self, i_stack, env=None):
        """Run ``cmd`` command and records the result."""

        res_list = list()

        if env:
            info('Setting environment variables {}'.format(', '.join(
                f'{k}={v}' for k, v in env.items()
            )))
        else:
            env = dict()

        # Merge the dictionaries, env takes precedence
        user_env = env
        # A deepcopy is required, otherwise we modify our own environment as
        # well
        env = copy.deepcopy(os.environ)
        env.update(user_env)

        for j in range(1, self.trials + 1):
            # Rebuild the command at every trial to make sure we get a unique
            # scope name every time.
            # Use systemd-run to start the command in a systemd scope. This
            # avoids leftover processes after the main one is killed.
            if self.use_systemd_run:
                if self.shell:
                    subcmd = [self.shell, '-c', self.cmd]
                else:
                    subcmd = [self.cmd]
                script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                scope_name = script_name + '-' + uuid.uuid1().hex[:10]

                # Ideally, we would pass -G but it is not supported before
                # systemd 236 so we call reset-failed manually at the end
                cmd = ['systemd-run', '--user', '-q', '--scope',
                    '-p', f'TimeoutStopSec={self.kill_timeout}',
                    '--unit', scope_name, '--'
                ]
                cmd.extend(subcmd)
                cmd_str = ' '.join(shlex.quote(x) for x in cmd)
                cmd = [item.encode('utf-8') for item in cmd]

                # We already start the shell with systemd, Popen does not need
                # to do it.
                shell_exe = None
            else:
                cmd_str = self.cmd
                cmd = self.cmd.encode('utf-8')
                shell_exe = self.shell

            info('Starting (#{j}) {self.cat} step ({self.name}): {cmd_str} ...'.format(
                self=self,
                cmd_str=cmd_str,
                j=j
            ))

            temp_env = copy.copy(env)
            temp_env.update({
                # most nested loop iteration count
                'BISECT_ITERATION': str(i_stack[-1]),
                # stack of nested loop iteration counts
                'BISECT_ITERATION_STACK': ' '.join(str(i) for i in i_stack),
                'BISECT_TRIAL': str(j),
            })

            # Update with the variables that come from the user
            for var, val_list in self.env.items():
                idx = (i_stack[-1] - 1) % len(val_list)
                temp_env[var] = val_list[idx]

            p = None
            try:
                p = subprocess.Popen(
                    args=cmd,
                    stdout=subprocess.PIPE,
                    # It is necessary to merge stdout and
                    # stderr to avoid garbage output with a
                    # mix of stdout and stderr.
                    stderr=subprocess.STDOUT,
                    shell=shell_exe is not None,
                    executable=shell_exe,
                    # Unbuffered output, necessary to avoid issues in
                    # read_stdout()
                    bufsize=0,
                    close_fds=True,
                    env=temp_env,
                )

                returncode, stdout = read_stdout(p, self.timeout, self.kill_timeout)
                timed_out = returncode is None
            finally:
                if p is not None:
                    # Make sure we don't leak opened file descriptors
                    p.stdout.close()
                    # Always kill the subprocesses, even if an exception was raised.
                    if self.use_systemd_run:
                        subprocess.call(['systemctl', '--user', 'kill',
                            scope_name + '.scope',
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        # Also remove the "failed" status of the scope unit in
                        # case it has it, so it can be garbage collected.
                        # That is equivalent to running "systemd-run -G" but
                        # this option was only added in systemd 236.
                        subprocess.call(['systemctl', 'reset-failed',
                            scope_name + '.scope',
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        terminate_process(p, self.kill_timeout)

            # Record the results
            # None returncode means it timed out
            res_list.append((returncode, stdout))

            # It timed out
            if timed_out:
                info('{self.cat} step ({self.name}) timed out after {timeout}.'.format(
                    self=self,
                    timeout=datetime.timedelta(seconds=self.timeout),
                ))
                if self.trials - j >= 1:
                    info(f'retrying (#{j}) ...')
                continue

            # It exectuded in time
            else:
                # If non-zero return code, try again ...
                if returncode != 0:
                    info('{self.cat} step ({self.name}) failed with exit status {returncode}.'.format(**locals()))
                    if self.trials - j >= 1:
                        info(f'retrying (#{j}) ...')
                    continue
                # Zero return code, we can proceed
                else:
                    break

        trial_status = f', tried {j} times' if j > 1 else ''

        last_ret = res_list[-1][0]
        if last_ret is None:
            exit_reason = 'timed out after {timeout}{trial_status}'.format(
                timeout=datetime.timedelta(seconds=self.timeout),
                trial_status=trial_status,
            )
        else:
            exit_reason = 'exit status {last_ret}{trial_status}'.format(**locals())

        info('Finished {self.cat} step ({self.name}) ({exit_reason}).'.format(**locals()))
        return res_list

    def run(self, i_stack, service_hub):
        return StepResult(
            step=self,
            res_list=self._run_cmd(i_stack),
            bisect_ret=BisectRet.NA,
        )

    def _get_exported_logs_dir(self, base_dir, i_stack):
        """
        Returns the directory where to store the logs of the command for the
        given iteration.
        """
        # If a boolean is given, it means no path is specified. We use a default
        # one.
        if isinstance(base_dir, bool):
            base_dir = 'logs'

        i_stack_str = [str(i) for i in i_stack]
        dirname = os.path.join(base_dir, self.name, *i_stack_str)
        ensure_dir(os.path.join(dirname, 'foo'))
        return dirname

    def report(self, step_res_seq, service_hub,
               verbose=False,
               show_basic=True,
               ignore_non_issue=False,
               iterations=[],
               export_logs=False
            ):
        """Print out a report for a list of executions results created using
        the run() method.
        """

        considered_iteration_set = set(iterations)

        out_list = list()
        # Not showing basic info means not showing anything.
        out = MLString(empty=not show_basic)
        out(f'command: {self.cmd}')

        for i_stack, res in step_res_seq:
            # Ignore the iterations we are not interested in
            if considered_iteration_set and i_stack[0] not in considered_iteration_set:
                continue

            step = res.step

            # Save the logs to the given directory, sorted by step name, then
            # iteration number and all the trials.
            if export_logs:
                for trial, trial_res in enumerate(res.res_list):
                    trial += 1
                    ret = trial_res['ret']
                    log = trial_res['log']
                    if ignore_non_issue and ret == 0:
                        continue

                    log_name = 'log_{trial}_{ret}'.format(
                        ret=ret,
                        trial=trial,
                    )

                    log_path = self._get_exported_logs_dir(export_logs, i_stack)
                    log_path = os.path.join(log_path, log_name)

                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write(log)
                        f.write('-' * 12 + '\nExit status: ' + str(trial_res['ret']) + '\n')

            if (not verbose and res.ret == 0 and
                (res.bisect_ret == BisectRet.GOOD
                 or res.bisect_ret == BisectRet.NA)
                    ):
                if not ignore_non_issue:
                    out(f'#{i_stack: <2}: OK')
            else:
                out('#{i_stack: <2}: returned {ret}, bisect {res.bisect_ret.lower_name}'.format(
                    ret=res.ret if res.ret is not None else '<timeout>',
                    res=res,
                    i_stack=i_stack,
                ))

        return out


class YieldStep(StepBase):
    """
    Abort current iteration with the yield return code.

    If the specified command returns a non-zero return code, bisector will
    abort the current iteration with the yield return code.

    .. note:: This step can only be used under the main macrostep (it cannot be
        used in nested macrosteps).
    """
    yaml_tag = '!yield-step'

    attr_init = dict(
        cat='yield',
        name='yield',
        every_n_iterations=1,
        cmd='exit 1',
    )

    options = dict(
        __init__=dict(
            every_n_iterations=IntParam('The step will be a no-op except if the iteration number can be evenly divided by that value.'),
            **StepBase.options['__init__']
        )
    )

    def __init__(self,
                every_n_iterations=Default,
                **kwargs
            ):
        super().__init__(**kwargs)
        self.every_n_iterations = every_n_iterations

    def run(self, i_stack, service_hub):
        skipped = bool(i_stack[-1] % self.every_n_iterations)
        if not skipped:
            res_list = self._run_cmd(i_stack)

            # Return value is the one from the last trial
            ret = res_list[-1][0]
            bisect_ret = BisectRet.YIELD if ret != 0 else BisectRet.NA
            step_res = StepResult(
                step=self,
                res_list=res_list,
                bisect_ret=bisect_ret,
            )
        else:
            step_res = StepResult(
                step=self,
                res_list=[(0, b'<skipped>')],
                bisect_ret=BisectRet.NA,
            )
        return step_res


class ShellStep(StepBase):
    """
    Execute a command in a shell.

    Stdout and stderr are merged and logged. The exit status of the command
    will have no influence of the bisect status.
    """
    yaml_tag = '!shell-step'
    attr_init = dict(
        cat='shell',
        name='shell',
    )


class TestShellStep(ShellStep):
    """
    Similar to :class:`ShellStep` .

    Non-zero exit status of the command will be interpreted as a bisect bad
    status, and zero exit status as bisect good.
    """
    yaml_tag = '!test-step'
    attr_init = dict(
        cat='test',
        name='test',
    )

    def run(self, i_stack, service_hub):
        res_list = self._run_cmd(i_stack)

        # Return value is the one from the last trial
        ret = res_list[-1][0]

        bisect_ret = BisectRet.BAD if ret != 0 else BisectRet.GOOD

        return StepResult(
            step=self,
            res_list=res_list,
            bisect_ret=bisect_ret,
        )


class BuildStep(ShellStep):
    """
    Similar to :class:`ShellStep` .

    Non-zero exit status of the command will be interpreted as a bisect
    untestable status, and zero exit status as bisect good.
    """
    yaml_tag = '!build-step'
    attr_init = dict(
        cat='build',
        name='build',
    )

    def run(self, i_stack, service_hub):
        res_list = self._run_cmd(i_stack)

        ret = res_list[-1][0]

        if ret != 0:
            bisect_ret = BisectRet.UNTESTABLE
        else:
            bisect_ret = BisectRet.GOOD

        return StepResult(
            step=self,
            res_list=res_list,
            bisect_ret=bisect_ret,
        )


class FlashStep(ShellStep):
    """
    Similar to :class:`ShellStep` .

    Non-zero exit status of the command will be interpreted as a bisect
    abort, and zero exit status ignored.
    """
    yaml_tag = '!flash-step'
    attr_init = dict(
        cat='flash',
        name='flash',
    )

    def run(self, i_stack, service_hub):
        res_list = self._run_cmd(i_stack)

        ret_list = [item[0] for item in res_list]
        all_failed = all(
            ret is None or ret != 0
            for ret in ret_list
        )
        # If flashing works, we don't take it into account, otherwise we abort
        # the bisection since the board can't be flashed anymore.
        bisect_ret = BisectRet.ABORT if all_failed else BisectRet.NA

        return StepResult(
            step=self,
            res_list=res_list,
            bisect_ret=bisect_ret,
        )


class RebootStep(ShellStep):
    """
    Similar to :class:`ShellStep` .

    Non-zero exit status of the command will be interpreted as a bisect
    abort, and zero exit status as bisect good.
    """
    yaml_tag = '!reboot-step'
    attr_init = dict(
        cat='boot',
        name='reboot',
    )

    def run(self, i_stack, service_hub):
        res_list = self._run_cmd(i_stack)

        # Return value is the one from the last trial
        ret = res_list[-1][0]

        last_boot_failed = ret is None or ret != 0
        # The board is unresponsive and cannot boot anymore so we need to
        # abort
        bisect_ret = BisectRet.ABORT if last_boot_failed else BisectRet.GOOD

        res = StepResult(
            step=self,
            res_list=res_list,
            bisect_ret=bisect_ret,
        )

        return res


class ExekallStepResult(StepResult):
    """
    Result of a LISA (exekall) step.

    It collects an :class:`exekall.engine.ValueDB`, as well as the path of the
    LISA result directory in addition than to what is collected by
    :class:`StepResult` .
    """
    yaml_tag = '!exekall-step-result'
    attr_init = dict(
        name='LISA-test',
    )

    def __init__(self, results_path, db, **kwargs):
        super().__init__(**kwargs)
        self.results_path = results_path
        self.db = db


class Deprecated:
    """
    Base class to inherit from to mark a subclass as deprecated.

    This removes it from the documented classes, and will generally mask it.
    """
    pass


class LISATestStepResult(ExekallStepResult, Deprecated):
    """
    .. deprecated:: 1.0
        Deprecated alias for :class:`ExekallStepResult`, it is only kept around
        to be able to reload old YAML and pickle reports.
    """
    yaml_tag = '!LISA-test-step-result'


class LISATestStep(ShellStep):
    """
    Execute an exekall LISA test command and collect
    :class:`exekall.engine.ValueDB`. Also compress the result directory and
    record its path. It will also define some environment variables that are
    expected to be used by the command to be able to locate resources to
    collect.
    """

    # YAML tag mentions exekall, so we can support other engines easily if
    # needed without breaking the tag compat
    yaml_tag = '!exekall-LISA-test-step'
    attr_init = dict(
        cat='test',
        name='LISA-test',
        delete_artifact_hidden=True,
        compress_artifact=True,
        upload_artifact=False,
        delete_artifact=False,
        prune_db=True,
        cmd='lisa-test',
    )

    options = dict(
        __init__=dict(
            delete_artifact_hidden=BoolParam('Remove hidden files and folders inside the artifacts'),
            compress_artifact=BoolParam('compress the exekall artifact directory in an archive'),
            upload_artifact=BoolParam('upload the exekall artifact directory to Artifactorial as the execution goes, and delete the local archive.'),
            delete_artifact=BoolParam('delete the exekall artifact directory to Artifactorial as the execution goes.'),
            prune_db=BoolParam("Prune exekall's ValueDB so that only roots values are preserved. That allows smaller reports that are faster to load"),
            # Some options are not supported
            **filter_keys(StepBase.options['__init__'], remove={'trials'}),
        ),
        report=dict(
            verbose=StepBase.options['report']['verbose'],
            show_basic=StepBase.options['report']['show_basic'],
            iterations=StepBase.options['report']['iterations'],
            show_rates=BoolParam('show percentages of failure, error, skipped, undecided and passed tests'),
            show_dist=BoolParam('show graphical distribution of issues among iterations with a one letter code: passed=".", failed="F", error="#", skipped="s", undecided="u"'),
            show_pass_rate=BoolParam('always show the pass rate of tests, even when there are failures or errors as well'),
            show_details=ChoiceOrBoolParam(['msg'], 'show details of results. Use "msg" for only a brief message'),
            show_artifact_dirs=BoolParam('show exekall artifact directory for all iterations'),
            testcase=CommaListParam('show only the untagged test cases matching one of the patterns in the comma-separated list. * can be used to match any part of the name'),
            result_uuid=CommaListParam('show only the test results with a UUID in the comma-separated list.'),
            ignore_testcase=CommaListParam('completely ignore untagged test cases matching one of the patterns in the comma-separated list. * can be used to match any part of the name.'),
            ignore_non_issue=BoolParam('consider only tests that failed or had an error'),
            ignore_non_error=BoolParam('consider only tests that had an error'),
            ignore_excep=CommaListParam('ignore the given comma-separated list of exceptions class name patterns that caused tests error. This will also match on base classes of the exception.'),
            dump_artifact_dirs=BoolOrStrParam('write the list of exekall artifact directories to a file. Useful to implement garbage collection of unreferenced artifact archives'),
            export_db=BoolOrStrParam('export a merged exekall ValueDB, merging it with existing ValueDB if the file exists', allow_empty=False),
            export_logs=BoolOrStrParam('export the logs and artifact directory symlink to the given directory'),
            download=BoolParam('Download the exekall artifact archives if necessary'),
            upload_artifact=BoolParam('upload the artifact directory to an artifacts service and update the in-memory report. Following env var are needed: ARTIFACTORY_FOLDER or ARTIFACTORIAL_FOLDER set to the folder URL, and ARTIFACTORY_TOKEN or ARTIFACTORIAL_TOKEN. Note: --export should be used to save the report with updated paths'),
        )

    )

    def __init__(self,
                delete_artifact_hidden=Default,
                compress_artifact=Default,
                upload_artifact=Default,
                delete_artifact=Default,
                prune_db=Default,
                **kwargs
            ):
        kwargs['trials'] = 1
        super().__init__(**kwargs)

        self.upload_artifact = upload_artifact
        # upload_artifact implies compress_artifact, in order to have an
        # archive instead of a folder.
        # It also implies deleting the local artifact folder, since it has been
        # uploaded.
        if self.upload_artifact:
            compress_artifact = True
            delete_artifact = True

        self.delete_artifact_hidden = delete_artifact_hidden
        self.compress_artifact = compress_artifact
        self.delete_artifact = delete_artifact
        self.prune_db = prune_db

    def run(self, i_stack, service_hub):
        from exekall.utils import NoValue
        from exekall.engine import ValueDB

        artifact_path = os.getenv(
            'EXEKALL_ARTIFACT_ROOT',
            # default value
            './exekall_artifacts'
        )

        # Create a unique artifact dir
        date = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
        name = f'{date}_{uuid.uuid4().hex}'
        artifact_path = os.path.join(artifact_path, name)

        # This also strips the trailing /, which is needed later on when
        # archiving the artifact.
        artifact_path = os.path.realpath(artifact_path)

        env = {
            # exekall will use that folder directly, so it has to be empty and
            # cannot be reused for another invocation
            'EXEKALL_ARTIFACT_DIR': str(artifact_path),
        }

        res_list = self._run_cmd(i_stack, env=env)
        ret = res_list[-1][0]

        if ret is None or ret != 0:
            bisect_ret = BisectRet.BAD
        else:
            bisect_ret = BisectRet.GOOD

        db_path = os.path.join(artifact_path, 'VALUE_DB.pickle.xz')
        try:
            db = ValueDB.from_path(db_path)
        except Exception as e:
            warn(f'Could not read DB at {db_path}: {e}')
            db = None
        else:
            if self.prune_db:
                # Prune the DB so we only keep the root values in it, or the
                # exceptions
                root_froz_val_uuids = {
                    froz_val.uuid
                    for froz_val in db.get_roots()
                }

                def prune_predicate(froz_val):
                    return not (
                        # Keep the root values, usually ResultBundle's. We
                        # cannot compare the object themselves since they have
                        # been copied to the pruned DB.
                        froz_val.uuid in root_froz_val_uuids
                        or
                        # keep errors and values leading to them
                        froz_val.value is NoValue
                    )
                db = db.prune_by_predicate(prune_predicate)

        # Remove the hidden folders and files in the artifacts
        if self.delete_artifact_hidden:
            def remove_hidden(root, name, rm):
                if name.startswith('.'):
                    rm(os.path.join(root, name))

            for root, dirs, files in os.walk(artifact_path, topdown=False):
                for name in files:
                    remove_hidden(root, name, os.remove)
                for name in dirs:
                    remove_hidden(root, name, shutil.rmtree)

        # Compress artifact directory
        if self.compress_artifact:
            try:
                orig_artifact_path = artifact_path
                # Create a compressed tar archive
                info(f'Compressing exekall artifact directory {artifact_path} ...')
                archive_name = shutil.make_archive(
                    base_name=artifact_path,
                    format='gztar',
                    root_dir=os.path.join(artifact_path, '..'),
                    base_dir=os.path.split(artifact_path)[-1],
                )
                info('exekall artifact directory {artifact_path} compressed as {archive_name}'.format(
                    artifact_path=artifact_path,
                    archive_name=archive_name
                ))

                # From now on, the artifact_path is the path to the archive.
                artifact_path = os.path.abspath(archive_name)

                # Delete the original artifact directory since we archived it
                # successfully.
                info(f'Deleting exekall artifact root directory {orig_artifact_path} ...')
                shutil.rmtree(str(orig_artifact_path))
                info(f'exekall artifact directory {orig_artifact_path} deleted.')

            except Exception as e:
                warn(f'Failed to compress exekall artifact: {e}')

        artifact_local_path = artifact_path
        delete_artifact = self.delete_artifact

        # If an upload service is available, upload the traces as we go
        if self.upload_artifact:
            upload_service = service_hub.upload
            if upload_service:
                try:
                    artifact_path = upload_service.upload(path=artifact_path)
                except Exception as e:
                    error('Could not upload exekall artifact, will not delete the folder: ' + str(e))
                    # Avoid deleting the artifact if something went wrong, so
                    # we still have the local copy to salvage using
                    # bisector report
                    delete_artifact = False
            else:
                error('No upload service available, could not upload exekall artifact. The artifacts will not be deleted.')
                delete_artifact = False

        if delete_artifact:
            info(f'Deleting exekall artifact: {artifact_local_path}')
            try:
                os.remove(artifact_local_path)
            except Exception as e:
                error('Could not delete local artifact {path}: {e}'.format(
                    e=e,
                    path=artifact_local_path,
                ))

        return ExekallStepResult(
            step=self,
            res_list=res_list,
            bisect_ret=bisect_ret,
            results_path=artifact_path,
            db=db,
        )

    def report(self, step_res_seq, service_hub,
               verbose=False,
               show_basic=False,
               show_rates=True,
               show_dist=False,
               show_details=False,
               show_pass_rate=False,
               show_artifact_dirs=False,
               testcase=[],
               result_uuid=[],
               ignore_testcase=[],
               iterations=[],
               ignore_non_issue=False,
               ignore_non_error=False,
               ignore_excep=[],
               dump_artifact_dirs=False,
               export_db=False,
               export_logs=False,
               download=True,
               upload_artifact=False
            ):
        """Print out a report for a list of executions artifact created using
        the run() method.
        """

        from exekall.utils import get_name, NoValue
        from exekall.engine import ValueDB

        from lisa.tests.base import ResultBundleBase, Result

        if verbose:
            show_basic = True
            show_rates = True
            show_dist = True
            show_artifact_dirs = True
            show_details = True
            ignore_non_issue = False

        out = MLString()

        considered_testcase_set = set(testcase)
        considered_uuid_set = set(result_uuid)
        ignored_testcase_set = set(ignore_testcase)
        considered_iteration_set = set(iterations)

        # Read the ValueDB from exekall to know the failing tests
        testcase_map = dict()
        filtered_step_res_seq = list()
        for step_res_item in step_res_seq:
            i_stack, step_res = step_res_item
            bisect_ret = BisectRet.NA
            any_entries = False

            # Ignore the iterations we are not interested in
            if considered_iteration_set and i_stack[0] not in considered_iteration_set:
                continue

            db = step_res.db
            if not db:
                warn("No exekall ValueDB for {step_name} step, iteration {i}".format(
                    step_name=step_res.step.name,
                    i=i_stack
                ))
                continue
            else:

                # Gather all result bundles
                for froz_val in db.get_roots():
                    untagged_testcase_id = froz_val.get_id(qual=False, with_tags=False)

                    # Ignore tests we are not interested in
                    if (
                        (considered_testcase_set and not any(
                            fnmatch.fnmatch(untagged_testcase_id, pattern)
                            for pattern in considered_testcase_set
                        ))
                        or
                        (considered_uuid_set and
                            froz_val.uuid not in considered_uuid_set
                        )
                        or
                        (ignored_testcase_set and any(
                            fnmatch.fnmatch(untagged_testcase_id, pattern)
                            for pattern in ignored_testcase_set
                        ))
                    ):
                        continue

                    testcase_id = froz_val.get_id(qual=False, with_tags=True)
                    entry = {
                        'testcase_id': testcase_id,
                        'i_stack': i_stack,
                        'results_path': step_res.results_path,
                    }

                    is_ignored = False

                    try:
                        # We only show the 1st exception, others are hidden
                        excep_froz_val = list(froz_val.get_excep())[0]
                    except IndexError:
                        excep_froz_val = None

                    if excep_froz_val:
                        excep = excep_froz_val.excep
                        if isinstance(excep, ResultBundleBase):
                            result = 'skipped'
                            try:
                                short_msg = excep.metrics['skipped-reason'].data
                            except KeyError:
                                short_msg = ''
                            short_msg = short_msg or ''
                            msg = ''
                        else:
                            result = 'error'
                            short_msg = str(excep)
                            msg = excep_froz_val.excep_tb

                        entry['result'] = result

                        type_name = get_name(type(excep))
                        is_ignored |= any(
                            any(
                                fnmatch.fnmatch(get_name(cls), pattern)
                                for pattern in ignore_excep
                            )
                            for cls in inspect.getmro(type(excep))
                        )


                        entry['details'] = (type_name, short_msg, msg)
                    else:
                        val = froz_val.value
                        result_map = {
                            Result.PASSED: 'passed',
                            Result.FAILED: 'failure',
                            Result.UNDECIDED: 'undecided',
                        }
                        # If that is a ResultBundle, use its result to get
                        # most accurate info, otherwise just assume it is a
                        # bool-ish value
                        try:
                            result = val.result
                            msg = '\n'.join(
                                f'{metric}: {value}'
                                for metric, value in val.metrics.items()
                            )
                        except AttributeError:
                            result = Result.PASSED if val else Result.FAILED
                            msg = str(val)

                        # If the result is not something known, assume undecided
                        result = result_map.get(result, 'undecided')

                        entry['result'] = result
                        type_name = get_name(type(val))
                        short_msg = result
                        entry['details'] = (type_name, short_msg, msg)

                    if ignore_non_error and entry['result'] != 'error':
                        is_ignored = True

                    entry['froz_val'] = froz_val
                    entry['db'] = db

                    # Ignored testcases will not contribute to the number of
                    # iterations
                    if is_ignored:
                        continue

                    if entry['result'] in ('error', 'failure'):
                        bisect_ret = BisectRet.BAD
                    elif entry['result'] == 'passed':
                        # Only change from None to GOOD but not from BAD to GOOD
                        bisect_ret = BisectRet.GOOD if bisect_ret is BisectRet.NA else bisect_ret

                    any_entries = True
                    testcase_map.setdefault(testcase_id, []).append(entry)

            # Update the ExekallStepResult bisect result based on the DB
            # content.
            step_res.bisect_ret = bisect_ret

            # Filter out non-interesting iterations of that step
            if any_entries and not (ignore_non_issue and bisect_ret != BisectRet.BAD):
                filtered_step_res_seq.append(step_res_item)

        step_res_seq = filtered_step_res_seq

        if show_artifact_dirs:
            out('Results directories:')

        # Apply processing on selected results
        for i_stack, step_res in step_res_seq:

            # Upload the results and update the result path
            if upload_artifact and os.path.exists(step_res.results_path):
                upload_service = service_hub.upload
                if upload_service:
                    try:
                        url = upload_service.upload(path=step_res.results_path)
                        step_res.results_path = url
                    except Exception as e:
                        error('Could not upload LISA results: ' + str(e))
                else:
                    error('No upload service available, could not upload LISA results.')

            if show_artifact_dirs:
                out('    #{i_stack: <2}: {step_res.results_path}'.format(**locals()))

            # Accumulate the results path to a file, that can be used to garbage
            # collect all results path that are not referenced by any report.
            if dump_artifact_dirs:
                with open(dump_results_dir, 'a') as f:
                    f.write(step_res.results_path + '\n')

            # Add extra information to the dumped logs
            if export_logs:
                archive_path = step_res.results_path
                log_dir = self._get_exported_logs_dir(export_logs, i_stack)

                archive_basename = os.path.basename(archive_path)
                archive_dst = 'lisa_artifacts.' + archive_basename.split('.', 1)[1]
                archive_dst = os.path.join(log_dir, archive_dst)

                url = urllib.parse.urlparse(archive_path)
                # If this is a URL, we download it
                if download and url.scheme.startswith('http'):
                    info('Downloading {archive_path} to {archive_dst} ...'.format(
                        archive_path=archive_path,
                        archive_dst=archive_dst,
                    ))
                    download_service = service_hub.download
                    if download_service:
                        try:
                            download_service.download(url=archive_path,
                                                      path=archive_dst)
                        except Exception as e:
                            error('Could not retrieve {archive_path}: {e}'.format(
                                archive_path=archive_path,
                                e=e
                            ))
                    else:
                        error('No download service available.')

                # Otherwise, assume it is a file and symlink it alongside
                # the logs.
                else:
                    # Make sure we overwrite the symlink if it is already there.
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(archive_dst)
                    os.symlink(archive_path, archive_dst)

        out('')

        # Always execute that for the potential side effects like exporting the
        # logs.
        # Do not propagate ignore_non_issue since exekall will always return 0
        # exit code.
        basic_report = super().report(
            step_res_seq, service_hub, export_logs=export_logs,
            show_basic=show_basic, verbose=verbose,
            iterations=iterations,
        )

        if show_basic:
            out(basic_report)

        # Contains the percentage of skipped, failed, error and passed
        # iterations for every testcase.
        testcase_stats = dict()
        table_out = MLString()
        dist_out = MLString()

        def key(k_v):
            id_, entry_list = k_v
            return (natural_sort_key(id_), entry_list)

        for testcase_id, entry_list in sorted(testcase_map.items(), key=key):
            # We only count the iterations where the testcase was run
            iteration_n = len(entry_list)
            stats = dict()
            testcase_stats[testcase_id] = stats

            iterations_summary = [entry['result'] for entry in entry_list]
            stats['iterations_summary'] = iterations_summary
            stats['counters'] = {
                'total': iteration_n
            }
            stats['events'] = dict()
            for issue, pretty_issue in (
                ('passed', 'passed'),
                ('skipped', 'skipped'),
                ('undecided', 'undecided'),
                ('failure', 'FAILED'),
                ('error', 'ERROR'),
            ):
                filtered_entry_list = [
                    entry
                    for entry in entry_list
                    if entry['result'] == issue
                ]

                issue_n = len(filtered_entry_list)
                issue_pc = (100 * issue_n) / iteration_n
                stats['counters'][issue] = issue_n
                # The event value for a given issue at a given iteration will
                # be True if the issue appeared, False otherwise.
                stats['events'][issue] = [
                    summ_issue == issue
                    for summ_issue in iterations_summary
                ]

                if issue == 'passed':
                    show_this_rate = (
                        (
                            # Show pass rate if we were explicitly asked for
                            show_pass_rate
                            or (
                                issue_n > 0
                                and (
                                    # Show if we are going to show the details,
                                    # since we need a "header" line
                                    show_details
                                    # Show passed if we got 100% pass rate
                                    or set(iterations_summary) == {'passed'}
                                )
                            )
                            # in any case, hide it we explicitly asked for
                        ) and not ignore_non_issue
                    )
                elif issue in ('skipped', 'undecided'):
                    show_this_rate = issue_n > 0 and not ignore_non_issue
                else:
                    show_this_rate = issue_n > 0

                show_this_rate &= show_rates

                if show_this_rate:
                    table_out(
                        '{testcase_id}: {pretty_issue} {issue_n}/{iteration_n} ({issue_pc:.1f}%)'.format(
                            testcase_id=testcase_id,
                            pretty_issue=pretty_issue,
                            issue_n=issue_n,
                            iteration_n=iteration_n,
                            issue_pc=issue_pc
                        )
                    )

                if show_details and not (
                    ignore_non_issue
                    and issue in ('passed', 'skipped', 'undecided')
                ):
                    for entry in filtered_entry_list:
                        i_stack = entry['i_stack']
                        results_path = '\n' + entry['results_path'] if show_artifact_dirs else ''
                        exception_name, short_msg, msg = entry['details']
                        uuid_ = entry['froz_val'].uuid

                        if show_details == 'msg':
                            msg = ''
                        else:
                            msg = ':\n' + msg

                        if '\n' in short_msg:
                            short_msg = short_msg.strip() + '\n'
                        else:
                            msg += ' '

                        table_out(
                            '   #{i_stack: <2}) UUID={uuid_} ({exception_name}) {short_msg}{results_path}{msg}'.format(
                                **locals()
                            ).replace('\n', '\n\t')
                        )

            if show_dist and iterations_summary:
                issue_letter = {
                    'passed': '.',
                    'failure': 'F',
                    'skipped': 's',
                    'undecided': 'u',
                    'error': '#',
                }
                dist = ''.join(
                    issue_letter[issue]
                    for issue in iterations_summary
                )
                dist_out('{testcase_id}:\n\t{dist}\n'.format(
                    testcase_id=testcase_id,
                    dist=dist,
                ))

        if show_details:
            out(table_out)
        else:
            out(table_out.tabulate(' ', ' '))

        out()
        out(dist_out)

        counts = {
            issue: sum(
                # If there is a non-zero count for that issue for that test, we
                # account it
                bool(stats['counters'][issue])
                for stats in testcase_stats.values()
            )
            for issue in ('error', 'failure', 'undecided', 'skipped', 'total')
        }
        nr_tests = len(testcase_map)
        assert counts['total'] == nr_tests

        # Only account for tests that only passed and had no other issues
        counts['passed'] = sum(
            set(stats['iterations_summary']) == {'passed'}
            for stats in testcase_stats.values()
        )

        out(
            'Error: {counts[error]}/{total}, '
            'Failed: {counts[failure]}/{total}, '
            'Undecided: {counts[undecided]}/{total}, '
            'Skipped: {counts[skipped]}/{total}, '
            'Passed: {counts[passed]}/{total}'.format(
                counts=counts,
                total=nr_tests,
            )
        )

        # Write-out a merged DB
        if export_db:

            entry_list = [
                entry
                for entry_list in testcase_map.values()
                for entry in entry_list
            ]

            # Prune the DBs so we only keep the root values we selected
            # previously and all its parents.
            def get_parents_uuid(froz_val):
                yield froz_val.uuid
                for parent_froz_val in froz_val.values():
                    yield from get_parents_uuid(parent_froz_val)

            allowed_uuids = set(itertools.chain.from_iterable(
                get_parents_uuid(entry['froz_val'])
                for entry in entry_list
            ))

            def prune_predicate(froz_val):
                return froz_val.uuid not in allowed_uuids

            db_list = [
                db.prune_by_predicate(prune_predicate)
                for db in {entry['db'] for entry in entry_list}
            ]

            if db_list:
                with contextlib.suppress(FileNotFoundError):
                    existing_db = ValueDB.from_path(export_db)
                    db_list.append(existing_db)

                merged_db = ValueDB.merge(db_list)
                merged_db.to_path(export_db, optimize=False)

        return out


class ExekallLISATestStep(LISATestStep, Deprecated):
    """
    .. deprecated:: 1.0
        Deprecated alias for :class:`LISATestStep`, it is only kept around to
        be able to reload old pickle reports.
    """

    # Use the former name
    attr_init = copy.copy(LISATestStep.attr_init)
    attr_init['name'] = 'exekall-LISA-test'


class StepNotifService:
    """Allows steps to send notifications."""

    def __init__(self, slave_manager):
        self.slave_manager = slave_manager

    def notif(self, step, msg, display_time=3):
        self.slave_manager.signal.StepNotif = (step.name, msg, display_time)


def urlretrieve(url, path):
    response = requests.get(url)
    # Raise an exception is the request failed
    response.raise_for_status()

    # If that is a broken symlink, get rid of it
    if not os.path.exists(path) and os.path.islink(path):
        os.unlink(path)

    with open(path, 'wb') as f:
        f.write(response.content)


class ArtifactsService(abc.ABC):
    @abc.abstractmethod
    def upload(self, path, url):
        """Upload a file"""
        pass
    @abc.abstractmethod
    def download(self, url, path):
        """Download a file"""
        pass


class ArtifactorialService(ArtifactsService):
    """Upload/download files to/from Artifactorial."""

    def __init__(self, token=None):
        """
        :param token: Token granting the read & write access to the url
                      It is not mandatory for the download service.
        """

        self.token = token or os.getenv('ARTIFACTORIAL_TOKEN')

    def upload(self, path, url=None):
        """
        Upload a file to Artifactorial.

        :param path: path to the file to upload
        :param url: URL of the Artifactorial folder to upload to,
                    or env var ARTIFACTORIAL_FOLDER
        """

        token = self.token
        if not token:
            raise ValueError("An Artifactorial token must be provided through ARTIFACTORIAL_TOKEN env var")

        dest = url or os.getenv('ARTIFACTORIAL_FOLDER')
        if not dest:
            raise ValueError("An Artifactorial folder URL must be provided through ARTIFACTORIAL_FOLDER env var")

        if not os.path.exists(path):
            warn(f'Cannot upload {path} as it does not exists.')
            return path

        info(f'Uploading {path} to {dest} ...')

        data = dict(token=token)
        with open(path, 'rb') as f:
            content = f.read()
        files = dict(path=(os.path.basename(path), content))
        response = requests.post(dest, data=data, files=files)

        if response.ok:
            uploaded_link = response.text.split('/', 1)[1]
            url = list(urllib.parse.urlparse(dest))
            url[2] += '/' + uploaded_link
            url = urllib.parse.urlunparse(url)
            info('{path} uploaded to {url} ...'.format(
                path=path,
                url=url
            ))
            path = url
        # Return the path unmodified if it failed
        else:
            warn('Failed to upload {path} to {dest} (HTTP {error}/{reason}): {msg}'.format(
                path=path,
                dest=dest,
                error=response.status_code,
                reason=response.reason,
                msg=response.text,
            ))

        return path

    def download(self, url, path):
        """
        Download a file from Artifactorial: anonymous access.

        :param url: URL of the Artifactorial file to download
        :param path: path to the downloaded file
        """
        urlretrieve(url, path)


class ArtifactoryService(ArtifactsService):
    """Upload/download files to/from Artifactory. """

    def __init__(self, token=None):
        """
        :param token: Token granting the read & write access to the url,
                      or env var ARTIFACTORY_TOKEN
        """

        self.token = token or os.getenv('ARTIFACTORY_TOKEN')

        if not self.token:
            raise ValueError("An Artifactory token must be provided through ARTIFACTORY_TOKEN env var")

    def upload(self, path, url=None):
        """
        Upload a file to Artifactory.

        :param path: path to the file to upload
        :param url: URL of the Artifactory folder to upload to,
                    or env var ARTIFACTORY_FOLDER
        """

        token = self.token

        dest = url or os.getenv('ARTIFACTORY_FOLDER')
        if not dest:
            raise ValueError("An Artifactory folder URL must be provided through ARTIFACTORY_FOLDER env var")

        dest = dest + "/" + os.path.basename(path)

        if not os.path.exists(path):
            warn(f'Cannot upload {path} as it does not exists.')
            return path

        info(f'Uploading {path} to {dest} ...')

        with open(path, 'rb') as f:
            content = f.read()
            md5sum = hashlib.md5(content).hexdigest();

        headers = {
            "X-Checksum-Md5": md5sum,
            "X-JFrog-Art-Api": token
        }

        response = requests.put(dest, headers=headers, data=content)

        if response.ok:
            url = response.json().get("downloadUri")
            info('{path} uploaded to {url} ...'.format(
                path=path,
                url=url
            ))
            path = url
        # Return the path unmodified if it failed
        else:
            warn('Failed to upload {path} to {dest} (HTTP {error}/{reason}): {msg}'.format(
                path=path,
                dest=dest,
                error=response.status_code,
                reason=response.reason,
                msg=response.text,
            ))

        return path

    def download(self, url, path):
        """
        Download a file from an Artifacts service.

        :param path: path to the file to download
        """

        headers = {
            "X-JFrog-Art-Api": self.token
        }

        response = requests.get(url, headers=headers)
        # Raise an exception is the request failed
        response.raise_for_status()

        # If that is a broken symlink, get rid of it
        if not os.path.exists(path) and os.path.islink(path):
            os.unlink(path)

        with open(path, 'wb') as f:
            f.write(response.content)


def update_json(path, mapping):
    """
    Update a JSON file with the given mapping.

    That allows accumulation of keys coming from different sources in one file.
    Newly inserted keys will override existing keys with the same name.
    """
    # This is a bit dodgy since we keep accumulating to a file that may
    # already exist before bisector is run. However, we don't really have the
    # choice if we want to accumulate all results in the same file (as opposed
    # to one file per step).
    try:
        with open(path) as f:
            data = json.load(f)
    except BaseException:
        data = dict()
    data.update(mapping)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_steps_list(yaml_path):
    return load_yaml(yaml_path)['steps']


def load_yaml(yaml_path):
    """Load the sequence of steps from a YAML file."""
    try:
        with open(yaml_path, encoding='utf-8') as f:
            mapping = Report._get_yaml().load(f)
    except (ruamel.yaml.parser.ParserError, ruamel.yaml.scanner.ScannerError) as e:
        raise ValueError('Could not parse YAML file {yaml_path}: {e}'.format(**locals())) from e

    except FileNotFoundError as e:
        raise FileNotFoundError('Could not read YAML steps file: {e}'.format(**locals())) from e

    else:
        # If that's not a mapping, we consider the file as empty. This will
        # trigger exceptions in the right place when keys are accessed.
        if isinstance(mapping, collections.abc.Mapping):
            return mapping
        else:
            return dict()


def get_class(full_qual_name):
    """
    Parse the class name of an entry in a YAML steps map file and return the
    module name and the class name. It will also append the necessary
    directories to PYTHONPATH if the class comes from another module.

    :param full_qual_name: The qualified name of the class that is suitable
                           for serialization, e.g.:
                           /path/to/mod.py:mymodule.LISA-step
                           Note: the name is not the Python class name, but the
                           value of the ``name`` attribute of the class.
                           The path to the script can be omitted if the module
                           is directly importable (i.e. in sys.path). The module
                           can be omitted for class names that are already
                           imported in the script namespace.
    """
    tokens = full_qual_name.rsplit(':', 1)

    # If there is no script path specified (e.g. module.Class)
    if len(tokens) <= 1:
        script_name = __file__
        tokens = full_qual_name.rsplit('.', 1)
        if len(tokens) <= 1:
            cls_name = tokens[0]
        else:
            mod_name, cls_name = tokens
            # Import the module to make sure the class is available
            module = importlib.import_module(mod_name)

    # If a script name is specified as well, e.g. :
    # /path/to/script.py:Class
    # script.py:Class
    # In the 2nd example, the path is considered to be relative to the YAML file
    # In both cases, the module name is "script".
    else:
        script_name, cls_name = tokens
        mod_name = os.path.splitext(os.path.basename(script_name))[0]

        # Import the module to make sure the class is available
        import_file(script_name, mod_name)

    cls = get_step_by_name(cls_name)
    cls.src_file = script_name
    return cls


def import_file(script_name, name=None):
    if name is None:
        name = inspect.getmodulename(script_name)

    spec = importlib.util.spec_from_file_location(name, script_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_files(src_files):
    try:
        import_excep = ModuleNotFoundError
    # ModuleNotFoundError does not exists in Python < 3.6
    except NameError:
        import_excep = ImportError

    for src in src_files:
        try:
            import_file(src)
        except (import_excep, FileNotFoundError) as e:
            pass


def get_subclasses(cls):
    """Get the subclasses recursively."""
    return _get_subclasses(cls) - {cls}


def _get_subclasses(cls):
    cls_set = {cls}
    for subcls in cls.__subclasses__():
        cls_set |= _get_subclasses(subcls)
    return cls_set


def get_step_by_name(name):
    for cls in get_subclasses(StepBase):
        if cls.name == name:
            return cls
    raise ValueError(f'Could not find a class matching "{name}".')


def instantiate_step(spec, step_options):
    """
    Find the right Step* class using Step*.name attribute and build an instance
    of it.
    """
    cls_name = spec['class']
    # This creates a new dictionary so we can delete the 'class' key without
    # modifying the original dict.
    # Also, we replace hypens by underscore in key names for better-looking
    # yaml.
    spec = {key.replace('-', '_'): val for key, val in spec.items()}
    del spec['class']

    cls = get_class(cls_name)
    name = spec.get('name', cls.name)
    cat = spec.get('cat', cls.name)

    for identifer in (name, cat):
        if '.' in identifer:
            raise ValueError('Dots are forbidden in steps names or categories.')

    spec.update(get_step_kwargs(cat, name, cls, '__init__', step_options))

    # If this is a MacroStep, we propagate some extra parameters so nested
    # MacroStep behave properly.
    if issubclass(cls, MacroStep):
        spec.update({
            'step_options': step_options,
        })

    return cls(**spec)


class StepSeqResult(StepResultBase):
    """
    Result of a sequence of steps.

    The main purpose is to store the results of one iteration of a
    :class:`MacroStep` .
    """
    yaml_tag = '!step-seq-result'

    attr_init = dict(
        # For backward compatibility
        step_res_run_times={},
    )

    def __init__(self, step, steps_res, run_time=0, step_res_run_times=None):
        # self.step is not directly useful at the moment but may become useful
        # since the data stored in self is directly under control of the MacroStep.
        self.step = step
        self.steps_res = steps_res
        self.run_time = round(run_time, 6)
        self.step_res_run_times = step_res_run_times or {}

    @property
    def bisect_ret(self):
        return self._filtered_bisect_ret()

    def _filtered_bisect_ret(self, steps_set=None, steps_filter=None, ignore_yield=False):
        """
        The set of steps to consider is precomputed by the caller so the
        overall complexity depends on the steps hierarchy structure and not the
        amount of results that is stored.

        :param steps_set: set of steps to consider. All other steps will be
                          ignored.
        :param steps_filter: :class:`StepFilter` instance that will be passed
                             along to allow nested :class:`MacroStep` to filter
                             their own steps.
        :param ignore_yield: ignore :class:`YieldStep` steps.
        """
        if steps_set is not None:
            steps_res = (
                step_res for step_res in self.steps_res
                if step_res.step in steps_set
            )
        else:
            steps_res = self.steps_res

        bisect_ret_stats = collections.Counter(
            res.filtered_bisect_ret(steps_filter) for res in steps_res
        )

        # If there are no results at all, this is untestable
        if not bisect_ret_stats:
            bisect_ret = BisectRet.NA

        # If one step ask for bisection abort, return abort.
        elif bisect_ret_stats[BisectRet.ABORT]:
            bisect_ret = BisectRet.ABORT

        # If one step ask for bisection yield, return yield.
        elif not ignore_yield and bisect_ret_stats[BisectRet.YIELD]:
            bisect_ret = BisectRet.YIELD

        # If one step labels the commit as untestable, return untestable.
        elif bisect_ret_stats[BisectRet.UNTESTABLE]:
            bisect_ret = BisectRet.UNTESTABLE

        # If there is at least one step that labeled the commit as bad,
        # return bad
        elif bisect_ret_stats[BisectRet.BAD]:
            bisect_ret = BisectRet.BAD

        # If everything went fine, the commit is good
        else:
            bisect_ret = BisectRet.GOOD

        return bisect_ret


class ServiceHub:
    def __init__(self, **kwargs):
        for name, service in kwargs.items():
            setattr(self, name, service)

    def __getattr__(self, attr):
        """If a service is missing, None is returned."""
        return getattr(super(), attr, None)

    def register_service(self, name, service):
        setattr(self, name, service)

    def unregister_service(self, name):
        try:
            delattr(self, name)
            return True
        except AttributeError:
            return False


class MacroStepResult(StepResultBase):
    """
    Result of the execution of a :class:`MacroStep` .

    It mainly contains a sequence of :class:`StepSeqResult` instances, each of them
    corresponding to one iteration of the :class:`MacroStep` .
    """

    yaml_tag = '!macro-step-result'
    attr_init = dict(
        name='macro',
    )

    def __init__(self, step, res_list):
        self.step = step
        self.res_list = res_list

    @property
    def avg_run_time(self):
        return statistics.mean(
            res.run_time for res in self.res_list
        )

    @property
    def bisect_ret(self):
        return self.filtered_bisect_ret()

    def filtered_bisect_ret(self, steps_filter=None, ignore_yield=False):
        bisect_ret = BisectRet.UNTESTABLE

        if not self.res_list:
            pass

        elif self.step.stat_test:
            steps_set = self.step.filter_steps(steps_filter)

            # Maps the items in the list to their number of occurences
            bisect_ret_stats = collections.Counter(
                res._filtered_bisect_ret(
                    steps_set, steps_filter,
                    ignore_yield=ignore_yield
                )
                for res in self.res_list
            )

            if bisect_ret_stats[BisectRet.ABORT]:
                bisect_ret = BisectRet.ABORT
            elif bisect_ret_stats[BisectRet.YIELD]:
                bisect_ret = BisectRet.YIELD
            # If one iteration labels the commit as untestable, return untestable.
            elif bisect_ret_stats[BisectRet.UNTESTABLE]:
                bisect_ret = BisectRet.UNTESTABLE
            else:
                bad_cnt = bisect_ret_stats[BisectRet.BAD]
                good_cnt = bisect_ret_stats[BisectRet.GOOD]
                good_bad_cnt_sum = bad_cnt + good_cnt

                # If there was no good or bad result, this is inconclusive
                if good_bad_cnt_sum == 0:
                    bisect_ret = BisectRet.UNTESTABLE
                else:
                    # We only consider the iterations that returned GOOD or BAD, but
                    # we ignore UNTESTABLE and such.
                    bad_percent = (100 * bad_cnt) / good_bad_cnt_sum

                    # Carry out the statistical test to check whether it is likely
                    # that the bad_percent average is ok.
                    bisect_ret = self.step.stat_test.test(bad_percent, good_bad_cnt_sum)
        else:
            warn('No statistical test registered for macro step {self.step.name}'.format(
                self=self,
            ))

        return bisect_ret


def match_step_name(name, pattern_list):
    """Find the patterns matching the step name, and return them in a list."""
    return [
        pattern for pattern in pattern_list
        if fnmatch.fnmatch(name, pattern)
    ]


class StepFilter:
    """
    Filter the steps of a :class:`MacroStep` . This allows restricting the display
    to a subset of steps.
    """

    def __init__(self, skip=set(), only=set()):
        self.skip = set(skip)
        self.only = set(only)

    def filter(self, steps_list):
        to_remove = self.skip
        to_keep = self.only

        if not (to_remove or to_keep):
            steps_list = copy.copy(steps_list)
        else:
            steps_list = [
                step for step in steps_list
                if not (
                    match_step_name(step.cat, to_remove) or
                    match_step_name(step.name, to_remove)
                )
            ]

            if to_keep:
                steps_list = [
                    step for step in steps_list
                    if (
                        match_step_name(step.cat, to_keep) or
                        match_step_name(step.name, to_keep)
                    )
                ]

        return steps_list


class MacroStep(StepBase):
    """
    Provide a loop-like construct to the steps definitions.

    All sub-steps will be executed in order. This sequence will be repeated
    for the given number of iterations or until it times out. The steps
    definition YAML configuration file is interpreted assuming an implicit
    toplevel :class:`MacroStep` that is parameterized using command line options.
    """

    yaml_tag = '!macro-step'

    attr_init = dict(
        name='macro',
        cat='macro',
        timeout=parse_timeout('inf'),
        iterations=parse_iterations('inf'),
        steps_list=[],

        bail_out_early=False,
    )

    options = dict(
        __init__=dict(
            iterations=IterationParam('number of iterations'),
            timeout=TimeoutParam('time after which no new iteration will be started'),
            bail_out_early=BoolParam('start a new iteration when a step returned bisect status bad or untestable and skip all remaining steps'),
        ),
        report=dict()
    )

    """Groups a set of steps in a logical sequence."""

    def __init__(self, *, steps=None, name=Default, cat=Default,
                stat_test=Default, step_options=None,
                iterations=Default,
                timeout=Default,
                bail_out_early=Default
            ):
        if step_options is None:
            step_options = dict()

        self.name = name
        self.cat = cat
        self.bail_out_early = bail_out_early
        self.stat_test = stat_test

        self.timeout = timeout
        # There are no guarantees that <iterations> number of iterations were
        # actually executed. This is only stored to allow resuming nested
        # MacroStep properly.
        self.iterations = iterations

        # We are intializing the steps for the first time
        if steps:
            self.steps_list = [
                instantiate_step(spec, step_options)
                for spec in steps
            ]

    def reinit(self, *args, steps=None, step_options=None, **kwargs):
        # Calling __init__ again will only override specified parameters, the
        # other ones will keep their current value thanks to Default
        # behavior.

        # We do not pass "steps" to avoid rebuilding a new step list
        self.__init__(*args, **kwargs)

        # If we have some step options, we also partially reinitialize the steps
        if step_options:
            for step in self.steps_list:
                step_kwargs = get_step_kwargs(
                    step.cat, step.name, type(step),
                    '__init__', step_options
                )

                # Pass down MacroStep-specific information
                if isinstance(step, MacroStep):
                    step_kwargs['step_options'] = step_options
                step.reinit(**step_kwargs)

        return self

    def iteration_range(self, start=1):
        # Create the generator used for running the iterations
        if self.iterations == 'inf':
            _range = itertools.count(start)
        else:
            _range = range(start, self.iterations + 1)

        begin_ts = time.monotonic()
        for i in _range:
            # If we have consumed all the time available, we break
            elapsed_ts = time.monotonic() - begin_ts

            # Report the elapsed time
            elapsed = datetime.timedelta(seconds=elapsed_ts)
            info(f'Elapsed time: {elapsed}')

            if self.timeout and elapsed_ts > self.timeout:
                info('{self.cat} step ({self.name}) timing out after {timeout}, stopping ...'.format(
                    self=self,
                    timeout=datetime.timedelta(seconds=self.timeout),
                ))
                break
            else:
                yield i

    def filter_steps(self, steps_filter=None):
        if steps_filter is None:
            steps_list = copy.copy(self.steps_list)
        else:
            steps_list = steps_filter.filter(self.steps_list)
        return steps_list

    def _run_steps(self, i_stack, service_hub):
        step_res_list = list()

        info('Starting {self.cat} step ({self.name}) iteration #{i} ...'.format(i=i_stack, self=self))

        # Run the steps
        step_res_run_times = {}
        begin_ts = time.monotonic()
        for step in self.steps_list:
            step_begin_ts = time.monotonic()
            res = step.run(i_stack, service_hub)
            step_end_ts = time.monotonic()
            step_res_list.append(res)
            step_res_run_times[res] = step_end_ts - step_begin_ts

            # If the bisect must be aborted, there is no point in carrying over,
            # even when bail_out_early=False.
            if res.bisect_ret == BisectRet.ABORT:
                info('{step.cat} step ({step.name}) requested bisect abortion, aborting ...'.format(step=step))
                break

            # Yielding to the caller of bisector
            if res.bisect_ret == BisectRet.YIELD:
                info('{step.cat} step ({step.name}) requested yielding ...'.format(step=step))
                break

            # If the commit is not testable or bad, bail out early
            if self.bail_out_early and step.bail_out and (
                (res.bisect_ret == BisectRet.UNTESTABLE) or
                (res.bisect_ret == BisectRet.BAD)
            ):
                break

        # Report the iteration execution time
        end_ts = time.monotonic()
        delta_ts = end_ts - begin_ts
        info('{self.cat} step ({self.name}) iteration #{i} executed in {delta_ts}.'.format(
            i=i_stack,
            delta_ts=datetime.timedelta(seconds=delta_ts),
            self=self,
        ))

        return StepSeqResult(
            step=self,
            steps_res=step_res_list,
            run_time=delta_ts,
            step_res_run_times=step_res_run_times,
        )

    def run(self, i_stack, service_hub):
        res_list = list()

        for i in self.iteration_range():
            i_stack.append(i)

            res = self._run_steps(i_stack, service_hub)
            res_list.append(res)

            # Propagate ABORT and YIELD
            if res.bisect_ret in (BisectRet.ABORT, BisectRet.YIELD):
                break

            i_stack.pop()

        return MacroStepResult(
            step=self,
            res_list=res_list,
        )

    def toplevel_run(self, report_options, slave_manager=None,
            previous_res=None, service_hub=None):
        """
        Equivalent of :meth:`run` method with additional signal handling and
        report generation at every step. Must only be called once for the root
        :class:`MacroStep` .
        """

        service_hub = ServiceHub() if service_hub is None else service_hub

        iteration_start = 1
        if previous_res:
            res_list = previous_res.res_list
            # Account for the iterations already completed.
            iteration_start = len(res_list) + 1
        else:
            res_list = list()

        macrostep_res = MacroStepResult(step=self, res_list=res_list)
        report = Report(
            macrostep_res,
            **report_options
        )

        natural_termination = False
        try:
            # Unmask the signals before starting iterating, in case they were
            # blocked by a previous toplevel_run() call.
            mask_signals(unblock=True)

            if slave_manager:
                slave_manager.signal.State = 'running'

            for i in self.iteration_range(start=iteration_start):
                # Save the report after each iteration, to make early results
                # available. We store BisectRet.UNTESTABLE since we have not
                # completed yet, so it is not significant.
                # Saving the report at the beginning avoids doing it twice at
                # the the last iteration, when it could take some time to save
                # to disk.

                path, url = report.save(upload_service=service_hub.upload)

                if slave_manager:
                    slave_manager.signal.Iteration = i
                    if url is not None:
                        slave_manager.signal.ReportPath = url

                i_stack = IterationCounterStack([i])
                res = self._run_steps(i_stack, service_hub)
                res_list.append(res)

                # Propagate ABORT
                if res.bisect_ret == BisectRet.ABORT:
                    if slave_manager:
                        slave_manager.signal.State = 'aborted'
                    break

                # Propagate YIELD
                if res.bisect_ret == BisectRet.YIELD:
                    if slave_manager:
                        slave_manager.signal.State = 'yielded'
                    break

                if slave_manager:
                    while slave_manager.pause_loop.is_set():
                        info('Iterations paused.')
                        slave_manager.signal.State = 'paused'

                        while not slave_manager.continue_loop.wait():
                            pass
                        slave_manager.signal.State = 'running'
                        info('Iterations resumed ...')

                    if slave_manager.stop_loop.is_set():
                        info('Iterations stopped, exiting ...')
                        break

                # Report Estimated Time of Arrival, given the remaining number
                # of steps.
                if isinstance(self.iterations, numbers.Integral):
                    eta = datetime.timedelta(
                        seconds=res.run_time * (self.iterations + 1 - i)
                    )
                    info(f'ETA: {eta}')

            # If the loop naturally terminates (i.e. not with break)
            else:
                natural_termination = True

        # If the execution is interrupted by an asynchronous signal, we just stop
        # where we are so that we can save the current results and exit cleanly.
        except SILENT_EXCEPTIONS as e:
            info('Interrupting current iteration and exiting ...')

        finally:
            # From this point, we mask the signals to avoid dying stupidly just
            # before being able to save the report. The signal handler will
            # take care of masking signals before raising any exception so we
            # don't need to handle that here.
            mask_signals()

            if slave_manager:
                state = 'completed' if natural_termination else 'stopped'
                slave_manager.signal.State = state

        path, url = report.save(upload_service=service_hub.upload)
        if slave_manager and url is not None:
            slave_manager.signal.ReportPath = url

        return report

    def report(self, macrostep_res_seq, service_hub, steps_filter=None,
            step_options=dict()):
        """Report the results of nested steps."""

        out = MLString()

        # Only show the steps that we care about
        steps_list = self.filter_steps(steps_filter)
        steps_set = set(steps_list)

        step_res_map = collections.defaultdict(list)
        for i_stack, macrostep_res in macrostep_res_seq:
            if not macrostep_res.res_list:
                return 'No iteration information found for {self.cat} step ({self.name}).'.format(self=self)

            out('Average iteration runtime: {}\n'.format(
                datetime.timedelta(seconds=macrostep_res.avg_run_time),
            ))
            step_res_run_times = {}
            for i, macrostep_i_res in enumerate(macrostep_res.res_list):
                i += 1
                i_stack_ = copy.copy(i_stack)
                i_stack_.append(i)
                for step_res in macrostep_i_res.steps_res:
                    step = step_res.step
                    # Ignore steps that are not part of the list
                    if not step in steps_set:
                        continue

                    step_res_run_times.update(macrostep_i_res.step_res_run_times)

                    # Get the step result and the associated iteration number
                    step_res_map[step].append((i_stack_, step_res))

        # Display the steps in the right order
        for step in steps_list:
            step_res_list = step_res_map[step]
            if not step_res_list:
                out('Warning: no result for {step.cat} step ({step.name})'.format(step=step))
                continue

            # Select the kwargs applicable for this step
            kwargs = get_step_kwargs(step.cat, step.name, type(step), 'report', step_options)

            # Pass down some MacroStep-specific parameters
            if isinstance(step, MacroStep):
                kwargs.update({
                    'step_options': step_options,
                    'steps_filter': steps_filter,
                })

            report_str = step.report(
                step_res_list,
                service_hub,
                **kwargs
            )

            # Build a temporary iteration result to aggregate the results of
            # any given step, as if all these results were coming from a list
            # of steps. This allows quickly spotting if any of the iterations
            # went wrong. This must be done after calling step.report,
            # so that it has a chance to modify the results.
            bisect_ret = StepSeqResult(
                step,
                (res[1] for res in step_res_list),
            ).bisect_ret

            run_time_list = [
                step_res_run_times.get(res[1])
                for res in step_res_list
            ]
            run_time_list = [
                runtime for runtime in run_time_list
                if runtime is not None
            ]
            if run_time_list:
                avg_run_time = statistics.mean(run_time_list)
                avg_run_time = datetime.timedelta(seconds=int(avg_run_time))
                avg_run_time = f' in {avg_run_time}'
            else:
                avg_run_time = ''

            out('{step.cat}/{step.name} ({step.__class__.name}){avg_run_time} [{bisect_ret}]'.format(
                avg_run_time=avg_run_time,
                step=step,
                bisect_ret=bisect_ret.name,
            ))
            indent = ' ' * 4
            if report_str:
                out(indent + report_str.strip().replace('\n', '\n' + indent) + '\n')

        return out

    def get_step_src_files(self, src_file_set=None):
        if src_file_set is None:
            src_file_set = set()

        for step in self.steps_list:
            if isinstance(step, MacroStep):
                step.get_step_src_files(src_file_set)

            try:
                src_file = type(step).src_file
            except AttributeError:
                src_file = __file__

            if src_file != __file__:
                src_file_set.add(src_file)

        return src_file_set


class StatTestMeta(abc.ABCMeta, type(Serializable)):
    """Metaclass of all statistical tests."""
    pass


class StatTestABC(Serializable, metaclass=StatTestMeta):
    """Abstract Base Class of all statistical tests."""
    pass
    @abc.abstractmethod
    def test(self, failure_rate, iteration_n):
        """failure_rate ranges in [0;1] and is the number of failed iterations
        divided by the number of properly executed iterations.
        iteration_n is the number of iterations that executed with a clear
        bisect return value (GOOD or BAD). It returns a BisectRet value.
        """
        pass


class BasicStatTest(StatTestABC):
    """Basic statistical test using a threshold for the failure rate."""

    yaml_tag = '!bad-percentage-stat-test'

    def __init__(self, allowed_bad_percent=0):
        self.allowed_bad_percent = allowed_bad_percent

    def test(self, failure_rate, iteration_n):
        return BisectRet.GOOD if failure_rate <= self.allowed_bad_percent else BisectRet.BAD


class BinomStatTest(BasicStatTest):
    """
    Binomial test on the failure rate.

    TODO: replace that with a Fisher exact test
    """

    yaml_tag = '!binomial-stat-test'

    def __init__(self, good_failure, good_sample_size, bad_failure, commit_n, overall_failure):
        assert good_failure < bad_failure

        # We need scipy for statistical tests.
        self.scipy_stats = importlib.import_module('scipy.stats')

        # The probability of the bisect session to fail because one step failed
        # is:
        # bisect_failure = 1 - (1 - step_failure) ** commit_n
        # Therefore, the allowed failure rate for every bisect step is:
        step_failure = 1 - (1 - overall_failure) ** (1 / commit_n)

        if step_failure > 0.5:
            raise ValueError('Risk of bisect step being wrong is higher than 0.5: {step_failure}'.format(**locals()))

        # There are 2 cases:
        # * the commit is good: the only source of risk is alpha
        # * the commit is bad: the only source of is beta
        # That means that the allowed failure rate for a step is assigned to
        # alpha and beta, since only one of them is actually a risk in a given
        # situation.

        # Probability of marking as bad although it is a good commit
        alpha = step_failure
        # Probability of marking as good although it is bad
        beta = step_failure

        # Convert into success rate
        good = 1 - good_failure
        bad = 1 - bad_failure

        # Determine the worst value for the actual mean, given that we only
        # know the mean through the results of a previous experiment. There is
        # a risk step_failure that the mean is actually smaller than that, in
        # which case the test we are going to setup will not give the expected
        # performances. We are a bit conservative and decide to assume this
        # worst case.

        # We increment by small steps so we are not too pessimistic.
        step_size = 0.0001
        # We start by centering the mean on the observed one, and we shift it
        # left until we reach the point where observing n_good would become
        # very unlikely.
        n_good = good * good_sample_size
        # We actually proceed in multiple steps, starting with coarse grained
        # exploration and then refining the step size. Doing such dichotomy is
        # alright since we know that the function is monotonic.  Initialize
        # such that the first step will not bring worst_good below 0.
        step_size = good
        worst_good = good
        direction = 0
        error = 0.01 / 100
        n = math.ceil(math.log(step_size, 2) - math.log(error, 2))
        for i in range(n):
            step_size /= 2
            worst_good += direction * step_size
            p = self.scipy_stats.binom.sf(n_good, good_sample_size, worst_good)
            direction = -1 if (p >= step_failure) else 1

        good = worst_good

        if good <= bad:
            raise ValueError('It is impossible to reliably distinguish between good and bad distributions, since good is not know with enough accuracy. The bad failure rate must be at least {least_bad:.2f}%'.format(
                least_bad=(1 - good) * 100,
            ))

        # We don't try to be smart here and just try every sample size until we
        # hit the target alpha and beta. This is fast enough since we only have
        # a few hundreds or thousands iterations at most.
        for N in itertools.count(1):
            # Substract one since PPF function rounds to the next value
            crit_val = self.scipy_stats.binom.ppf(alpha, N, good) - 1
            actual_beta = self.scipy_stats.binom.sf(crit_val, N, bad)
            if actual_beta <= beta:
                break

        actual_alpha = self.scipy_stats.binom.cdf(crit_val, N, good)

        self.iteration_n = N
        self.alpha = actual_alpha
        self.good = good
        self.overall_failure = overall_failure

        info('Using binomial test: given number of bisect steps={commit_n}, risk of bisect not converging to the right commit={overall_failure:.2f}%, expected failure rate={good_failure:.2f}%, bad failure rate={bad_failure:.2f}%'.format(
            commit_n=commit_n,
            overall_failure=overall_failure * 100,
            good_failure=good_failure * 100,
            bad_failure=bad_failure * 100,
        ))

    def test(self, failure_rate, iteration_n):
        # We convert back percentage to [0;1]
        failure_rate /= 100
        # Number of successful iterations
        success_n = (1 - failure_rate) * iteration_n
        # We don't reuse self.iteration_n, since there is no guarantee that the
        # tests executed that many iterations. They could have been aborted for
        # example, or an iteration could have been skipped for other reasons.
        # If that happens, the power of the test (1-beta) will be reduced and
        # overall_failure cannot be guaranteed anymore, but we can still give
        # an answer.
        # We do a one-sided test to check whether the failure_rate is equal or
        # greater than the bad failure rate. Since we test for success, the
        # alternate hypothesis is "observed success rate lower than expected
        # success rate".
        pval = self.scipy_stats.binom_test(
            success_n, iteration_n, self.good, alternative="less"
        )
        return BisectRet.GOOD if pval > self.alpha else BisectRet.BAD


def do_steps_help(cls_list):
    """Print out the help for the given steps classes."""
    for cls in cls_list:
        print(cls.help())

    return 0


def do_run(slave_manager, iteration_n, stat_test, steps_filter=None,
        bail_out_early=False, inline_step_list=[], steps_path=None,
        report_options=None, overall_timeout=0, step_options=None,
        git_clean=False, resume_path=None,
        service_hub=None):
    """Run the specified list of steps."""

    # Top level MacroStep options coming from the command line
    macro_step_options = dict(
        name='main',
        stat_test=stat_test,
        iterations=iteration_n,
        timeout=overall_timeout,
        bail_out_early=bail_out_early,
        step_options=step_options,
    )
    if resume_path and not os.path.exists(resume_path):
        warn('Report {path} does not exist, starting from scratch ...'.format(
            path=resume_path,
        ))
        resume_path = None

    # Restore the MacroStep from a report. This allows resuming from the report.
    if resume_path:
        info(f'Resuming execution from {resume_path} ...')
        report = Report.load(resume_path, steps_path, use_cache=False)

        # If there no steps path, we need to carry over the src files list so
        # it is not forgotten
        if steps_path is None:
            report_options['src_files'] = report.preamble.src_files
        previous_res = report.result
        # Partially reinitialize the steps, with the updated command line options
        macro_step = report.result.step.reinit(**macro_step_options)

    # Otherwise, create a fresh MacroStep
    else:
        # Steps taken from a YAML description
        if steps_path:
            steps_list = load_steps_list(steps_path)

        # Inline steps defined from the command line
        else:
            steps_list = [
                {
                    'class': spec[0],
                    'name': spec[1],
                }
                for spec in inline_step_list
            ]

        # Create the top level MacroStep that will run the steps in order
        macro_step = MacroStep(
            steps=steps_list,
            **macro_step_options
        )
        previous_res = None

    # Handle --skip
    macro_step.steps_list = macro_step.filter_steps(steps_filter)

    step_list_str = ', '.join(
        f'{step.cat}/{step.name}'
        for step in macro_step.steps_list
    )
    if overall_timeout:
        timeout_str = ' or {tmout}'.format(
            tmout=datetime.timedelta(seconds=overall_timeout),
        )
    else:
        timeout_str = ''
    info('Will run steps {step_list_str} for {iteration_n} iterations{timeout_str}.'.format(
        **locals()
    ))

    # Make sure that we always cleanup when requested.
    with enforce_git_cleanup(git_clean):
        report = macro_step.toplevel_run(
            report_options,
            slave_manager=slave_manager,
            previous_res=previous_res,
            service_hub=service_hub,
        )

    # Only take into account YIELD results if this is not the last iteration
    if (
        macro_step.iterations == 'inf'
        or len(report.result.res_list) < macro_step.iterations
    ):
        bisect_ret = report.bisect_ret
    else:
        bisect_ret = report.filtered_bisect_ret(ignore_yield=True)

    return bisect_ret.value, report


# Compute the SHA1 of the script itself, to identify the version of the tool
# that was used to generate a given report.
with open(__file__, 'rb') as f:
    TOOL_SHA1 = hashlib.sha1(f.read()).hexdigest()


class ReportPreamble(Serializable):
    """
    Informations split out from the :class:`Report` so that they can be used to
    prepare the environment before loading the report body. It is useful to
    make sure all the modules are imported prior to steps results
    restoration.
    """
    yaml_tag = '!report-preamble'

    attr_init = dict(
        src_files=[]
    )

    def __init__(self, report_version, src_files):
        self.src_files = src_files
        # Version of the format that could be used to change behavior if
        # backward compatibility is needed.
        self.report_version = report_version
        self.tool_sha1 = TOOL_SHA1
        self.preamble_version = 0


def check_report_path(path, probe_file):
    def probe_open_f(open_f, path):
        try:
            with open_f(path) as f:
                f.read(1)
        except Exception:
            return False
        else:
            return True

    # Default to uncompressed file
    open_f = open
    mime_map = {
        'gzip': gzip.open,
        'xz': lzma.open,
    }

    if probe_file:
        for f in mime_map.values():
            if probe_open_f(f, path):
                open_f = f
                break
    else:
        guessed_mime = mimetypes.guess_type(path)[1]
        open_f = mime_map.get(guessed_mime, open_f)

    path = pathlib.Path(path)
    compo = path.name.split('.')
    # By default, assume YAML unless pickle is explicitly present
    is_yaml = '.pickle' not in path.name
    return (open_f, is_yaml)


def disable_gc(f):
    """
    Decorator to disable garbage collection during the execution of the
    decorated function.

    This can speed-up code that creates a lot of objects in a short amount of
    time.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        gc.disable()
        try:
            return f(*args, **kwargs)
        # re-enable the garbage collector in all cases
        finally:
            gc.enable()

    return wrapper


class Report(Serializable):
    """
    Report body containg the result of the top level :class:`MacroStep` .
    """

    yaml_tag = '!report'
    # The preamble is saved separately
    dont_save = ['preamble', 'path']

    REPORT_CACHE_TEMPLATE = '{report_filename}.cache.pickle'

    def __init__(self, macrostep_res, description='', path=None, src_files=None):
        self.creation_time = datetime.datetime.now()
        self.result = macrostep_res
        self.description = description
        self.path = path

        src_files = src_files or sorted(macrostep_res.step.get_step_src_files())
        self.preamble = ReportPreamble(
            report_version=0,
            src_files=src_files,
        )

    @classmethod
    def _get_yaml(cls, typ='unsafe'):
        """
        Create and initialize YAML document manager class attribute.

        .. note:: That method should only be called once when the class is
            created.
        """
        yaml = ruamel.yaml.YAML(typ=typ)

        # Make the relevant classes known to YAML
        for cls_to_register in get_subclasses(Serializable):
            yaml.register_class(cls_to_register)

        # Register BisectRet as that cannot be done through a metaclass as
        # usual since it is an enum.Enum
        yaml.register_class(BisectRet)

        # Without allow_unicode, escape sequences are used to represent unicode
        # characters in plain ASCII
        yaml.allow_unicode = True
        yaml.default_flow_style = True

        # ruamel.yaml documentation states that sequence indent less than
        # (offset + 2) might lead to invalid output
        # https://yaml.readthedocs.io/en/latest/detail.html#indentation-of-block-sequences
        offset = 0
        sequence = offset + 2
        yaml.indent(mapping=1, sequence=sequence, offset=offset)

        # Dump OrderedDict as regular dictionaries, since we will reload map as
        # OrderedDict as well
        def map_representer(dumper, data):
            return dumper.represent_dict(data.items())

        def map_constructor(loader, node):
            return collections.OrderedDict(loader.construct_pairs(node))

        yaml.representer.add_representer(collections.OrderedDict, map_representer)
        yaml.constructor.add_constructor(str(yaml.resolver.DEFAULT_MAPPING_TAG), map_constructor)

        # Since strings are immutable, we can memoized the output to deduplicate
        # strings.
        @functools.lru_cache(maxsize=None, typed=True)
        def str_representer(dumper, data):
            # Use block style for multiline strings
            style = '|' if '\n' in data else None
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)

        yaml.representer.add_representer(str, str_representer)
        return yaml

    def save(self, path=None, upload_service=None):
        """Save the report to the specified path.

        :param path: Used to save the report is not None, otherwise use the
             existing ``path`` attribute.

        """
        if path:
            self.path = path

        ensure_dir(self.path)

        open_f, is_yaml = check_report_path(self.path, probe_file=False)
        # File in which the report is written, before being renamed to its
        # final name
        temp_path = os.path.join(
            os.path.dirname(self.path),
            '.{filename}.temp'.format(filename=os.path.basename(self.path))
        )

        # Save to YAML
        if is_yaml:
            # The file needs to be opened as utf-8 since the underlying stream
            # will need to accept utf-8 data in its write() method.
            with open_f(temp_path, 'wt', encoding='utf-8') as yaml_f:
                self._get_yaml().dump_all(
                    (self.preamble, self),
                    yaml_f,
                )

        # Save to Pickle
        else:
            pickle_data = dict(
                preamble=self.preamble,
                report=self
            )
            with open_f(temp_path, 'wb') as f:
                # Temporary workaround this bug:
                # https://bugs.python.org/issue43460
                #
                # Note: this will make deserialization dependent on
                # availibility of the "exekall" package, but this avoids
                # copy-pasting a whole class so it should be ok.
                try:
                    from exekall._utils import ExceptionPickler
                except ImportError:
                    pickle.dump(pickle_data, f, protocol=4)
                else:
                    ExceptionPickler.dump_file(f, pickle_data, protocol=4)

        # Rename the file once we know for sure that writing to the temporary
        # report completed with success
        os.replace(temp_path, self.path)

        # Upload if needed
        url = None
        if upload_service:
            try:
                url = upload_service.upload(path=self.path)
                info('Uploaded report ({path}) to {url}'.format(
                    path=self.path,
                    url=url
                ))
            except Exception as e:
                error('while uploading the report: ' + str(e))

        return (path, url)

    @classmethod
    @disable_gc
    def _do_load(cls, path, steps_path):
        def import_steps_from_yaml(yaml_path):
            """Make sure the steps classes are available before deserializing."""
            if not yaml_path:
                return
            try:
                steps_list = load_steps_list(yaml_path)
            except FileNotFoundError as e:
                return e

            _import_steps_from_yaml(steps_list)

        def _import_steps_from_yaml(steps_list):
            for spec in steps_list:
                cls_name = spec['class']
                # Import the module so that all the classes are created and their
                # yaml_tag registered for deserialization.
                cls = get_class(cls_name)

                # Recursively visit all the steps definitions.
                macrostep_names = {
                    macrostep_cls.name for macrostep_cls in MacroStep.__subclasses__()
                } | {MacroStep.name}
                if cls.name in macrostep_names:
                    _import_steps_from_yaml(spec['steps'])

        open_f, is_yaml = check_report_path(path, probe_file=True)

        def import_modules(steps_path, src_files):
            # Try to import the steps defined in steps_path if specified
            if steps_path:
                excep = import_steps_from_yaml(steps_path)
            # Then try to import the files as recorded in the report
            else:
                excep = import_files(src_files)
            return excep

        # Read as YAML or Pickle depending on the filename.
        if is_yaml:
            # Get the generator that will parse the YAML documents
            with open_f(path, 'rt', encoding='utf-8') as f:
                documents = cls._get_yaml().load_all(f)

                # First document is the preamble
                try:
                    preamble = next(documents)
                except Exception as e:
                    raise ValueError('Could not load the preamble in report: {path}'.format(
                        path=path
                    )) from e

                # Import the modules before deserializing the report document
                # so that all the necessary classes are created. Some exceptions
                # are just saved and only printed if some issues appear when
                # loading the report itself.
                excep = import_modules(steps_path, preamble.src_files)

                # Get the report and link the preamble to it
                try:
                    report = next(documents)
                except Exception as e:
                    if excep is not None:
                        error(excep)
                    raise
        else:
            excep = None
            try:
                # If some modules are missing, this will fail. Avoiding it
                # would require either pickling a dict containing itself
                # pickled objects, or using the shelve module. The latter
                # will use different filenames on different platforms and
                # is not reliable, so we stick with something simple. YAML
                # should be used for more advanced use cases.
                with open_f(path, 'rb') as f:
                    pickle_data = pickle.load(f)

                preamble = pickle_data['preamble']
                excep = import_modules(steps_path, preamble.src_files)
                report = pickle_data['report']
            except Exception as e:
                if excep is not None:
                    error(excep)
                raise

        # Tie the preamble to its report
        report.preamble = preamble

        # Update the path so it can be saved again
        report.path = path

        if report.preamble.tool_sha1 != TOOL_SHA1:
            debug('Loading a report generated with an earlier version of the tool.')
            debug('Version used to generate the report: ' + report.preamble.tool_sha1)
            debug('Current version: ' + TOOL_SHA1)

        return report

    @classmethod
    def load(cls, path, steps_path=None, use_cache=False, service_hub=None):
        """
        Restore a serialized :class:`Report` from the given file.

        :param path: Path to the Report to load. Can be an http URL or a local
                     filename.

        :param steps_path: Path to the steps definition YAML file. It is only
                           needed if the report contains foreign classes that
                           must be imported before restoring the report.
        :param use_cache: If True, will generate a cached version of the report
                          using the fastest on-disk format available. The cache
                          is invalidated if the modification time of the file
                          is newer than that of the cache.

        """

        url = urllib.parse.urlparse(path)
        # If this is a URL, we download it
        if url.scheme.startswith('http'):
            # Make sure the infered file type matches by using the whole
            # original name
            suffix = os.path.basename(url.path)
            with tempfile.NamedTemporaryFile(suffix=suffix) as temp_report:
                path=temp_report.name
                download_service = service_hub.download
                if download_service:
                    try:
                        download_service.download(url=url.geturl(), path=path)
                        return cls._load(path, steps_path, use_cache=False)
                    except Exception as e:
                        raise ValueError('Could not download report: ' + str(e)) from e
                else:
                    raise ValueError('No download service available.')
        else:
            return cls._load(path, steps_path, use_cache)

    @classmethod
    def _load(cls, path, steps_path, use_cache):
        write_cache = False
        if use_cache:
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            cache_filename = os.path.join(
                dirname, cls.REPORT_CACHE_TEMPLATE.format(report_filename=basename))

            write_cache = True
            if os.path.exists(cache_filename):
                cache_mtime = os.path.getmtime(cache_filename)
                report_mtime = os.path.getmtime(path)
                # If the cache is as recent as the report, use it
                if cache_mtime >= report_mtime:
                    path = cache_filename
                    write_cache = False

        try:
            report = cls._do_load(path, steps_path)
        except (ImportError, ruamel.yaml.constructor.ConstructorError) as e:
            raise ValueError('Some steps are relying on modules that cannot be found. Use --steps to point to the steps YAML config: {e}'.format(
                e=e,
            )) from e
        except (FileNotFoundError, IsADirectoryError) as e:
            raise ValueError('Could not open the report file {e.filename}: {e.strerror}'.format(e=e)) from e
        except pickle.UnpicklingError as e:
            raise ValueError('Could not parse the pickle report {path}: {e}'.format(
                e=e,
                path=path,
            )) from e
        except (ruamel.yaml.scanner.ScannerError, ruamel.yaml.composer.ComposerError) as e:
            raise ValueError('Could not parse the YAML report {path}: {e}'.format(
                e=e,
                path=path,
            )) from e

        # Create the cached report if needed
        if write_cache:
            report.save(cache_filename)

        return report

    @property
    def bisect_ret(self):
        return self.filtered_bisect_ret()

    def filtered_bisect_ret(self, *args, **kwargs):
        return self.result.filtered_bisect_ret(*args, **kwargs)

    def show(self, *args, service_hub=None, stat_test=None, steps_filter=None, **kwargs):
        """Show the report results."""
        # Update the stat test used to compute the overall bisect result
        if stat_test is not None:
            self.result.step.stat_test = stat_test

        service_hub = ServiceHub() if service_hub is None else service_hub

        out = MLString()
        out(f'Description: {self.description}')
        out(f'Creation time: {self.creation_time}\n')
        out(self.result.step.report(
            [(IterationCounterStack(), self.result)],
            service_hub=service_hub,
            steps_filter=steps_filter,
            *args, **kwargs
        ))

        # Always ignore YIELD here, since it's only useful for run-time
        bisect_ret = self.filtered_bisect_ret(steps_filter, ignore_yield=True)

        out('Overall bisect result: {} commit'.format(
            bisect_ret.lower_name
        ))

        return out, bisect_ret

    def __str__(self):
        return str(self.show()[0])


def ensure_dir(file_path):
    """
    Ensure that the parent directory of the `file_path` exists and creates if
    necessary.
    """
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def format_placeholders(string, placeholder_map):
    return string.format(**placeholder_map)


DBUS_SERVER_BUS_NAME = 'org.bisector.Server'
"Well-known DBus name used by the monitor-server."
DBUS_SLAVE_BOOK_PATH = '/org/bisector/SlaveBook'
"DBus path under which the :class:`DBusSlaveBook` is published by the monitor-server."

DBUS_SLAVE_BUS_NAME_TEMPLATE = 'org.bisector.Slave-{id}'
"Template of the DBus well-known name used by slaves."
DBUS_SLAVE_MANAGER_PATH = '/org/bisector/SlaveManager'
"DBus path under which the :class:`DBusSlaveManager` is published by the slaves."


def get_dbus_bus():
    return pydbus.SessionBus()


def dbus_variant(v):
    """
    Build a :class:`gi.repository.GLib.Variant` instance out of a Python
    object.
    """
    if isinstance(v, str):
        v = GLib.Variant('s', v)
    # Signed 64 bits value for integers.
    elif isinstance(v, int):
        v = GLib.Variant('x', v)
    elif isinstance(v, bool):
        v = GLib.Variant('b', v)
    elif isinstance(v, float):
        v = GLib.Variant('d', v)
    elif isinstance(v, collections.Iterable):
        v = GLib.Variant('av', [dbus_variant(i) for i in v])
    else:
        raise ValueError('GLib.Variant not known for type {}'.format(type(v)))

    return v


def dbus_variant_dict(dct):
    """
    Build a dictionary with :class:`gi.repository.GLib.Variant` values, so it
    can be used to create DBus types a{sv} objects.
    """
    return {k: dbus_variant(v) for k, v in dct.items()}


class dbus_property(property):
    """
    Like standard property decorator, except that signal_prop_change(<prop
    name>) will be called when the setter function returns, to automatically
    emit the PropertiesChanged signal.
    """

    def setter(self, content=True, iface=None):
        """Unlike property.setter, this is a decorator with parameters."""
        super_setter = super().setter

        def _setter(f):
            @functools.wraps(f)
            def wrapped_f(w_self, *args, **kwargs):
                r = f(w_self, *args, **kwargs)
                name = f.__name__

                nonlocal iface
                if iface is None:
                    iface = w_self.prop_iface

                if content:
                    invalidate = []
                    changed = {name: getattr(w_self, name)}
                else:
                    invalidate = [name]
                    changed = {}

                w_self.PropertiesChanged(iface, changed, invalidate)

                return r
            return super_setter(wrapped_f)
        return _setter


class PipeSetter:
    """Write to a pipe when an attribute is set."""

    def __init__(self, pipe, attrs):
        super().__setattr__('attrs', set(attrs))
        super().__setattr__('pipe', pipe)

        # Make sure writing to the pipe is not blocking. This would hang the
        # main thread which is not acceptable.
        os.set_blocking(self.pipe.fileno(), False)

    def __setattr__(self, attr, val):
        if not attr in self.attrs:
            raise AttributeError('Attribute {attr} not settable through the pipe'.format(
                attr=attr,
            ))
        try:
            self.pipe.send((attr, val))
            return True
        except (BlockingIOError, BrokenPipeError):
            return False


class DBusSlaveManager:
    """
    Object shared between the main process and the DBus slave thread.

    It interacts with the bus to influence the execution of the main thread.
    """

    # Interface on which DBus Properties are exposed by default
    prop_iface = 'org.bisector.SlaveManager'

    dbus = """
    <node>
      <interface name='org.bisector.SlaveManager'>
        <method name='Control'>
          <arg type='s' name='cmd' direction='in' />
          <arg type='b' name='response' direction='out' />
        </method>

        <property name="State" type="(ss)" access="read">
          <annotation name="org.freedesktop.DBus.Property.EmitsChangedSignal" value="true"/>
        </property>

        <property name="Iteration" type="t" access="read">
          <annotation name="org.freedesktop.DBus.Property.EmitsChangedSignal" value="true"/>
        </property>

        <property name="StepNotif" type="(sst)" access="read">
          <annotation name="org.freedesktop.DBus.Property.EmitsChangedSignal" value="true"/>
        </property>

        <property name="Description" type="s" access="read" />
        <property name="PID" type="t" access="read" />
        <property name="RunTime" type="t" access="read" />
        <property name="StartTs" type="t" access="read" />
        <property name="StartMonotonicTs" type="t" access="read" />
        <property name="ReportPath" type="s" access="read" />
        <property name="LogPath" type="s" access="read" />
        <property name="SupportedCommands" type="as" access="read" />
      </interface>
    </node>
    """

    def __init__(self, desc, pid, start_ts, start_monotonic_ts, report_path, log_path,
            path=DBUS_SLAVE_MANAGER_PATH):

        self.pause_loop = threading.Event()
        self.stop_loop = threading.Event()
        self.continue_loop = threading.Event()

        self.setter_pipe = multiprocessing.Pipe(duplex=False)
        # Used by the main thread to set properties
        self.signal = PipeSetter(self.setter_pipe[1], attrs={
            'State', 'Iteration', 'StepNotif', 'ReportPath',
        })

        self.path = path

        # Signaled properties
        self._iteration = 0
        self._state = ('stopped', 'initialized')
        self._step_notif = ('', '', 0)
        self._report_path = report_path

        # Passive properties
        self.Description = desc
        self.PID = pid
        self.StartTs = start_ts
        self.StartMonotonicTs = start_monotonic_ts
        self.LogPath = log_path

    if DBUS_CAN_BE_ENABLED:
        PropertiesChanged = pydbus.generic.signal()

    @dbus_property
    def ReportPath(self):
        return self._report_path

    @ReportPath.setter(content=True)
    def ReportPath(self, i):
        self._report_path = i

    @dbus_property
    def RunTime(self):
        return math.floor(time.monotonic() - self.StartMonotonicTs)

    @dbus_property
    def State(self):
        return self._state

    @State.setter(content=True)
    def State(self, new_state):
        old_state = self._state[1]
        self._state = (old_state, new_state)

    @dbus_property
    def Iteration(self):
        return self._iteration

    @Iteration.setter(content=True)
    def Iteration(self, i):
        self._iteration = i

    @dbus_property
    def StepNotif(self):
        return self._step_notif

    @StepNotif.setter(content=True)
    def StepNotif(self, notif_spec):
        self._step_notif = notif_spec

    def publish(self, bus, bus_name_template=DBUS_SLAVE_BUS_NAME_TEMPLATE):
        bus_name = bus_name_template.format(id=self.PID)
        try:
            # Register the object before taking ownership of the name, so that
            # if the bus name is already taken, we still expose the object using
            # our unique connection name.
            bus.register_object(self.path, self, None)
            bus.request_name(bus_name)
            return True
        except GLib.GError:
            return False

    def consume_prop(self):
        """This must only be called from the DBus thread, since it emits a
        DBus signal.
        """
        if not self.setter_pipe[0].poll(0):
            return False
        else:
            prop, val = self.setter_pipe[0].recv()
            setattr(self, prop, val)

            return True

    SupportedCommands = ('pause', 'stop', 'continue', 'kill')

    def Control(self, cmd):
        if cmd not in self.SupportedCommands:
            return False

        if cmd == 'pause':
            self.continue_loop.clear()
            self.pause_loop.set()
        elif cmd == 'stop':
            self.continue_loop.clear()
            self.stop_loop.set()

        elif cmd == 'continue':
            self.continue_loop.set()
            self.pause_loop.clear()
            self.stop_loop.clear()

        elif cmd == 'kill':
            os.kill(os.getpid(), signal.SIGTERM)

        return True


class DBusSlaveThread:
    def __init__(self, properties):
        self._loop = None
        self.slave_manager = DBusSlaveManager(**properties)

        init_done = threading.Event()

        thread = threading.Thread(
            target=self.thread_main,
            kwargs=dict(
                init_done=init_done,
            ),
            daemon=True,
        )
        thread.start()

        # Wait until the thread is ready
        init_done.wait()

    def thread_main(self, init_done):
        try:
            return self._thread_main(init_done)
        except SILENT_EXCEPTIONS:
            pass
        # Unexpected exceptions will kill the whole process. This makes sure
        # that this dbus slave thread cannot die unexpectedly, leading to some
        # blocking reads on the pipe or similar issues.
        except BaseException:
            # Print the exception before killing the whole process.
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            init_done.set()
            self.quit_thread()

    def quit_thread(self):
        if self._loop:
            ctx = self._loop.get_context()
            # Wait until there is no pending event anymore
            while ctx.pending():
                pass
            self._loop.quit()
            # Give a bit of time for any outstanding DBus transaction to finish
            # before we exit.
            time.sleep(0.1)

    def _thread_main(self, init_done):
        # Needed for old GLib versions.
        if tuple(int(i) for i in gi.__version__.split('.')) <= (3, 11):
            GLib.threads_init()

        # Check for the monitor service registration before creating the
        # GLib.MainLoop, since it installs its SIGINT handler that needs loop.run()
        # to be effective.
        bus = get_dbus_bus()

        # Since GLib installs its own handler for SIGINT that will only quit the
        # loop and raise KeyboardInterrupt in this thread, we also send SIGTERM, so
        # that the main thread will pick it up and terminate. It is okay since we
        # handle both signals in the same way.
        # See https://github.com/beetbox/audioread/issues/63
        def SIGINT_handler():
            os.kill(os.getpid(), signal.SIGTERM)
        GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, SIGINT_handler)

        # After this point, we must call loop.run() to be able to handle SIGINT.
        # If this thread dies, SIGINT will have no effect.
        loop = GLib.MainLoop()
        self._loop = loop

        # Monitor the event pipe.
        def consume_prop(channel, cond):
            self.slave_manager.consume_prop()
            # Return True so that the callback will be called again
            return True
        GLib.io_add_watch(
            self.slave_manager.setter_pipe[0].fileno(),
            GLib.IO_IN, consume_prop
        )

        # Publish the SlaveManager on the bus, so it can be directly accessed
        self.slave_manager.publish(bus)

        # Make ourselves known to the SlaveBook.
        def register_slave(changed_bus_name=DBUS_SERVER_BUS_NAME, old_owner='', new_owner=''):
            # We are only interested in one well known bus name
            if changed_bus_name != DBUS_SERVER_BUS_NAME:
                return

            bus_name = bus.con.get_unique_name()
            location = (bus_name, self.slave_manager.path)
            try:
                slave_book = bus.get(DBUS_SERVER_BUS_NAME, DBUS_SLAVE_BOOK_PATH)
                # Make sure we don't block for too long, in case the server is
                # not responsive.
                slave_book.RegisterSlave(location, timeout=2)
            # Ignore failures, as the server may not be running.
            except BaseException:
                pass

        # Register in the SlaveBook when the bus name shows up
        admin = bus.get('.DBus')
        admin.NameOwnerChanged.connect(register_slave)

        # Does one attempt before letting main thread to carry on, to make sure
        # an existing SlaveBook can connect to the events if needed.
        register_slave()

        # Inform that the init is done once we start executing the main loop.
        # That ensures that we are ready to handle DBus requests when init_done
        # is set.
        GLib.idle_add(lambda: init_done.set())

        # Handle requests, even when KeyboardInterrupt is raised. This allows
        # sending messages even when the main thread is shutting down, so this
        # thread will just die when everything is over.
        while True:
            with contextlib.suppress(SILENT_EXCEPTIONS):
                loop.run()


def parse_slave_props(slave_props):
    props = copy.copy(slave_props)

    props['StartTs'] = datetime.datetime.fromtimestamp(props['StartTs'])
    props['RunTime'] = datetime.timedelta(seconds=props['RunTime'])

    return props


def send_cmd(slave_id, slave_manager, cmd):
    info(f'Sending {cmd} to slave {slave_id} ...')
    try:
        slave_manager.Control(cmd)
    except GLib.GError:
        info('Command {cmd} sent to slave {id} failed.'.format(
            cmd=cmd, id=slave_id))
    else:
        info('Command {cmd} sent successfully to slave {id}.'.format(
            cmd=cmd, id=slave_id))


def do_monitor(slave_id, args):
    bus = get_dbus_bus()

    try:
        slave_book = bus.get(DBUS_SERVER_BUS_NAME, DBUS_SLAVE_BOOK_PATH)
    except GLib.GError:
        slave_book = None

    # If we know which slave ID we are interested in, we directly contact the
    # SlaveManager.
    if slave_id:
        slave_bus_name = DBUS_SLAVE_BUS_NAME_TEMPLATE.format(id=slave_id)
        admin = bus.get('.DBus')
        try:
            # Make sure we use the unique connection name
            slave_bus_name = admin.GetNameOwner(slave_bus_name)
            slave_manager = bus.get(slave_bus_name, DBUS_SLAVE_MANAGER_PATH)
        except GLib.GError:
            error(f"No DBus bus name {slave_bus_name}")
            return GENERIC_ERROR_CODE
        slave_manager_list = [slave_manager]
    else:
        if not slave_book:
            error("No dbus server found")
            return GENERIC_ERROR_CODE

        slave_manager_list = DBusSlaveBook.get_slaves_manager(bus, slave_book)

    # Get the properties for all slaves
    slaves_map = dict()
    for slave_manager in slave_manager_list:
        try:
            props = slave_manager.GetAll('org.bisector.SlaveManager')
        except GLib.GError:
            continue
        props = parse_slave_props(props)
        pid = props['PID']
        slaves_map[pid] = (slave_manager, props)

    if not slaves_map:
        info("No running instance")
        return 0

    # Execute commands
    indent = 4 * ' '
    for slave_id, (slave_manager, props) in sorted(slaves_map.items()):
        if args.list:
            # Adapt the length to number of columns to be sure to print one
            # entry per line.
            cols, lines = shutil.get_terminal_size((9999999, 9999999))
            out = "{props[PID]} [{props[State][1]}] #{props[Iteration]} {props[Description]}".format(
                props=props,
            )
            print(out[:cols])

        if args.prop:
            try:
                print(props[args.prop])
            except BaseException:
                warn('Property {prop} not found on slave {slave_id}'.format(
                    prop=args.prop,
                    slave_id=slave_id
                ))
                info('Available properties on slave {slave_id}: {prop}'.format(
                    slave_id=slave_id,
                    prop=', '.join(props.keys())
                ))

        if args.status:
            print(textwrap.dedent("""\
                Process {props[PID]} :
                    Description: {props[Description]}
                    State: {props[State][1]}
                    Iteration: {props[Iteration]}
                    Started at: {props[StartTs]}
                    Elapsed time: {props[RunTime]}
                    PID: {props[PID]}
                    Report: {props[ReportPath]}
                    Log: {props[LogPath]}
            """.format(props=props)))

        if args.log:
            pager = os.environ.get('PAGER', 'cat')
            subprocess.call([pager, props['LogPath']])

        if args.notif:
            if not slave_book:
                error('Needs to connect to the DBus server.')
            else:
                enable = (args.notif[0] == 'enable')
                prop = args.notif[1]
                location = (slave_manager._bus_name, slave_manager._path)
                try:
                    slave_book.SetDesktopNotif(location, prop, enable)
                except GLib.Error:
                    error('Could not set notifications for "{prop}" property on slave {location}'.format(
                        location=location,
                        prop=prop
                    ))

        if args.report is not None:
            report_path = props['ReportPath']
            argv = ['report', report_path]
            argv.extend(args.report)
            _main(argv)
            print('\n' + '#' * 80)

        if args.pause:
            send_cmd(slave_id, slave_manager, 'pause')
        if args.stop:
            send_cmd(slave_id, slave_manager, 'stop')
        if args.continue_:
            send_cmd(slave_id, slave_manager, 'continue')
        if args.kill:
            send_cmd(slave_id, slave_manager, 'kill')


def do_monitor_server(default_notif):
    bus = get_dbus_bus()
    # Make sure we can interrupt the loop
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    loop = GLib.MainLoop()

    def handle_name_lost(name):
        info(f'Lost ownership of DBus name "{name}", exiting ...')
        loop.quit()

    slave_book = DBusSlaveBook(bus, handle_name_lost, default_notif=default_notif)

    loop.run()


class DBusSlaveBook:
    dbus = """
    <node>
      <interface name='org.bisector.SlaveBook'>
        <method name='RegisterSlave'>
          <arg type='(so)' name='location' direction='in' />
        </method>

        <method name='GetSlavesLocation'>
          <arg type='a(so)' name='locations' direction='out' />
        </method>

        <method name='SetDesktopNotif'>
          <arg type='(so)' name='location' direction='in' />
          <arg type='s' name='prop' direction='in' />
          <arg type='b' name='enable' direction='in' />
        </method>

      </interface>
    </node>
    """

    def __init__(self, bus, name_lost_callback,
            bus_name=DBUS_SERVER_BUS_NAME, path=DBUS_SLAVE_BOOK_PATH,
            default_notif=None):
        self.slaves_map = dict()
        self._desktop_notif = collections.defaultdict(set)

        self.name_lost_callback = name_lost_callback
        self.bus_name = bus_name
        self.path = path
        self.bus = bus
        self.default_notif = default_notif or ('all', True)

        # Will be needed to display desktop notifications
        try:
            self.notif_proxy = bus.get('.Notifications')
        except GLib.Error:
            warn('Unable to connect to desktop notification server.')
            self.notif_proxy = None

        # First, make the ourselves available so we can service requests right
        # after we own the bus name.
        self.bus.register_object(self.path, self, None)

        # Get the ownership of the bus name and publish the object. If it fails,
        # act like we lost the name.
        if not self._request_bus_name():
            self.name_lost_callback(self.bus_name)

    def _request_bus_name(self, replace=True):
        # Make sure we are prepared to give up the bus name we are going to
        # acquire.
        admin = self.bus.get('.DBus')
        admin.NameLost.connect(self.name_lost_callback)

        # Try to own the bus name. This may fail if we don't replace it and if
        # there is a running instance.
        try:
            self.bus.request_name(self.bus_name, allow_replacement=True, replace=replace)
            return True
        except GLib.GError:
            return False

    def get_slave_manager(self, location, timeout=1):
        """Get the SlaveManager instance of a slave."""
        slave_manager = self.slaves_map[location]
        try:
            slave_manager.Ping(timeout=timeout)
        except GLib.GError as e:
            # Timeout, so we don't unregister it yet.
            if e.code == 24:
                pass
            # The slave is dead, unregister it.
            else:
                self.unregister_slave(location)

            raise KeyError(f'Slave {location} is unresponsive.')
        return slave_manager

    def unregister_slave(self, location):
        info(f'Unregistering slave {location}')
        del self.slaves_map[location]

    def prune_slaves(self):
        # Use a list to be able to modify the slaves_map while iterating
        for location in list(self.slaves_map.keys()):
            try:
                self.get_slave_manager(location)
            except KeyError:
                pass

    def GetSlavesLocation(self):
        self.prune_slaves()
        return sorted(self.slaves_map.keys())

    def SetDesktopNotif(self, location, prop, enable):
        verb = 'Enabling' if enable else 'Disabling'
        info('{verb} desktop notifications for {prop} changes for slave {location}'.format(
            verb=verb,
            location=location,
            prop=prop
        ))

        slave_manager = self.get_slave_manager(location)
        slave_props = slave_manager.GetAll('org.bisector.SlaveManager')
        slave_props = [prop.lower() for prop in slave_props.keys()]
        if prop != 'all':
            if prop not in slave_props:
                raise ValueError("Property {prop} not available on slave at location {location}".format(
                    location=location,
                    prop=prop
                ))
            slave_props = [prop]

        try:
            for prop in slave_props:
                if enable:
                    self._desktop_notif[location].add(prop)
                else:
                    self._desktop_notif[location].remove(prop)
        except KeyError:
            pass

    def consume_slave_prop(self, location, slave_manager, changed, invalidated):
        # Send a desktop notification
        if location in self._desktop_notif and self.notif_proxy:
            slave_props = slave_manager.GetAll('org.bisector.SlaveManager')
            props = parse_slave_props(slave_props)

            summary = 'Bisector process {pid}'.format(pid=props['PID'])
            display_time = 5

            for prop, val in changed.items():
                if prop == 'State':
                    old_state, new_state = val
                    # Display longer state change
                    display_time *= 3
                    msg = '{old_state} → {new_state}'.format(
                        old_state=old_state,
                        new_state=new_state,
                    )
                elif prop == 'Iteration':
                    msg = f'Started iteration #{val}'
                elif prop == 'StepNotif':
                    msg = '{name}: {msg}'.format(name=val[0], msg=val[1])
                    display_time = val[2]
                else:
                    debug(f'Unhandled property change {prop}')

                body = '{props[Description]}\nElapsed time: {props[RunTime]}\n\n{msg}'.format(
                    props=props,
                    msg=msg,
                )
                prop_set = self._desktop_notif[location]
                if 'all' in prop_set or prop.lower() in prop_set:
                    self.notif_proxy.Notify('bisector', 0, 'dialog-information',
                        summary, body, [], {}, display_time * 1000)

    def RegisterSlave(self, location):
        bus_name, path = location
        # Skip slaves we already know about
        if location in self.slaves_map:
            return

        try:
            slave_manager = self.bus.get(bus_name, path)
        except GLib.Error:
            return

        self.slaves_map[location] = slave_manager

        # Enable the notifications by default
        self.SetDesktopNotif(location, self.default_notif[0], enable=self.default_notif[1])

        def prop_handler(iface, changed, invalidated):
            return self.consume_slave_prop(location, slave_manager, changed, invalidated)
        slave_manager.PropertiesChanged.connect(prop_handler)

        info(f'Registered slave {location}')

    @staticmethod
    def get_slaves_manager(bus, slave_book):
        """Helper to get the list of SlaveManager out of a list of slaves
        location, requesting it from a SlaveBook (DBus object proxy).
        """
        return [
            bus.get(con_name, path) for con_name, path
            in slave_book.GetSlavesLocation()
        ]


class YAMLCLIOptionsAction(argparse.Action):
    """Custom argparse.Action that extracts command line options from a YAML
    file. Using "required=True" for any argument will break that since the
    parser needs to parse partial command lines.
    """

    def __call__(self, parser, args, value, option_string):
        config_path = value
        info(f'Loading configuration from {config_path} ...')

        yaml_map = load_yaml(config_path)
        config_option_list = shlex.split(yaml_map.get('cmd-line', ''))

        yaml_args = parser.parse_args(config_option_list)

        overriden_names = list()
        for name, val in vars(yaml_args).items():
            default_val = parser.get_default(name)
            existing_val = getattr(args, name)
            arg_present = existing_val != default_val

            # If we find a default value and no value was provided yet, we use
            # that so we can use this options on its own
            if val == parser.get_default(name):
                if not arg_present:
                    setattr(args, name, val)
            # Override if the argument has a non-default value
            else:
                if arg_present:
                    overriden_names.append(name)

                # Extend the list for action="append" arguments
                if isinstance(existing_val, list):
                    existing_val.extend(val)
                # Or just override what was already there
                else:
                    setattr(args, name, val)

        if overriden_names:
            info('Overriding command line options: {names}'.format(
                names=', '.join('"' + name + '"' for name in overriden_names)
            ))


def _main(argv):
    global LOG_FILE
    global SHOW_TRACEBACK

    parser = argparse.ArgumentParser(description="""
    Git-bisect-compatible command sequencer.

    bisector allows to run commands with timeout handling in predefined
    sequences (steps). The output is saved in a report that can be later
    analyzed. The exit status of all steps is merged to obtain a git bisect
    compatible value which is returned. The error code {generic_code} is used
    when bisector itself encounters an error.

    SIGNAL HANDLING

    The script can be stopped with SIGINT (Ctrl-C) or SIGTERM (kill or timeout
    commands) or SIGHUP (when the terminal dies) and will save the completed
    iterations to the report. The execution can later be resumed with --resume.
    """.format(generic_code=GENERIC_ERROR_CODE),
        formatter_class=argparse.RawTextHelpFormatter,
        # Allow passing CLI options through a file
        fromfile_prefix_chars='@',
    )

    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    step_help_parser = subparsers.add_parser('step-help',
    description="""
    Steps options help. The default category of the step is specified in
    parenthesis after the step name.

    OPTION TYPES

    bool: If the option is not specified, default to false. If the option
    is specified with no value, true is assumed. "true", "false", "yes"/"y" and
    "no"/"n" are also supported.

    comma-separated list: simple comma separated list of strings.
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    run_parser = subparsers.add_parser('run',
        description="""
        Run the given steps in a loop and record the result in a report file.
        The report file can be inspected using the "report" subcommand. The exit
        status is suitable for git bisect.
        """
    )
    report_parser = subparsers.add_parser('report',
        description="""Analyze a report generated by run command. The exit status
        is suitable for git bisect.

        In most cases, step options (``-o``) will act as filters to ignore
        parts of the data before computing the overall bisect result.
        """
    )
    edit_parser = subparsers.add_parser('edit',
        description="""Modify the properties of the steps in an existing report."""
    )
    dbus_service_parser = subparsers.add_parser('monitor-server',
        description="""Start the DBus server to allow monitoring all running
        instances. Note that the server is not necessary to monitor a specific
        run instance."""
    )
    monitor_parser = subparsers.add_parser('monitor',
        description="""Monitor and control a running instance."""
    )

    # Options common to all parsers
    for subparser in (parser, run_parser, report_parser, dbus_service_parser, monitor_parser):
        subparser.add_argument('--cli-options', action=YAMLCLIOptionsAction,
            help="""YAML file containing command line option string in a
"cmd-line" toplevel key. Therefore, the same file can be used
for --steps.  The options are inserted at the location of
--cli-options in the command line. That can be used to control what
can be overriden by configuration file and what is forced by the
command line""")

    cmd_metavar = ('TRIALS#', 'TIMEOUT', 'COMMAND')

    # Common options of report-aware commands
    for subparser in (run_parser, report_parser, edit_parser):
        subparser.add_argument('-o', '--option', action='append',
            default=[],
            help="""Step-specific options. Can be repeated. The format is <step
            category or name pattern>.<option>=<value>. If the step name or
            category is omitted, it will be passed to all steps. See
            "step-help" subcommand for the available options. Options specified
            this way override the steps definition file, and can be used to
            modify the report when used with run --resume or edit
            subcommand.""")

        subparser.add_argument('--debug', action='store_true',
            help="""Show Python traceback to help debugging this script.""")

    run_step_group = run_parser.add_mutually_exclusive_group()
    for subparser in (run_step_group, report_parser):
        subparser.add_argument('--steps',
            help="""YAML configuration of steps to run. The steps definitions
            lives under a "step" toplevel key.""")

    for subparser in (run_parser, report_parser):
        stat_test_group = subparser.add_mutually_exclusive_group()
        stat_test_group.add_argument('--allowed-bad', type=int,
            default=0,
            help="""Percentage of bad iterations that still result in a good
            overall result.""")

        # TODO: replace by a Fisher exact test
        #  stat_test_group.add_argument('--stat', nargs=5,
        #  metavar=('REFERENCE_FAILURE%', '#REFERENCE_SAMPLE_SIZE',
        #  'BAD_FAILURE%', 'WRONG_RESULT%', '#TESTED_COMMITS'),
        #  help="""Carry out a binomial test on the iterations results. This
        #  overrides --iterations. The parameters are 1) reference expected
        #  failure rate of the test in percent, 2) the sample size that gave
        #  the reference failure rate, 3) the bad failure rate we are trying
        #  to find the root of, 4) the probability of git bisect converging to
        #  the wrong commit, 5) the number of commits git bisect will need to
        #  analyze before finishing the bisect (roughly log2(number of commits
        #  in the tested series)).""")

        subparser.add_argument('--skip', action='append',
            default=[],
            help="""Step name or category to skip. Can be repeated.""")

        subparser.add_argument('--only', action='append',
            default=[],
            help="""Step name or category to execute. Can be repeated.""")

    # Options for edit subcommand
    edit_parser.add_argument('report',
        help="Report to edit.")

    edit_parser.add_argument('--steps',
        help="""YAML configuration of steps used to find new paths of
        classes if necessary. It is otherwise ignored and --option must be
        used to edit the report.""")

    # Options for run subcommand
    run_parser.add_argument('--git-clean', action='store_true',
        help="""Run git reset --hard and git clean -fdx before and after the
        steps are executed. It is useful for automated bisect to avoid
        checkout failure that leads to bisect abortion. WARNING: this will
        erase all changes and untracked files from the worktree.""")

    run_step_group.add_argument('--inline', '-s', nargs=2, action='append',
        metavar=('CLASS', 'NAME'),
        default=[],
        help="""Class and name of inline step. Can be repeated to build a list
        of steps in the order of appearance.""")

    run_parser.add_argument('-n', '--iterations', type=parse_iterations,
        # Use "inf" as default so that if only --timeout is specified, it will
        # iterate until the timeout expires instead of just 1 iteration.
        default=parse_iterations("inf"),
        help="""Number of iterations. "inf" means an infinite number of
        iterations.""")

    run_parser.add_argument('--timeout', type=parse_timeout,
        default=parse_timeout('inf'),
        help='''Timeout after which no more itertions will be started. "inf"
        means infinite timeout (i.e. no timeout).''')

    run_parser.add_argument('--log',
        help="""Log all events to the log file.
        By default, a log file is created as <report file>.log .""")

    run_parser.add_argument('--report',
        help="""Execution report containing output and return values for all
        steps. Can be displayed using the "report" subcommand. The {commit}
        placeholder is replaced with the truncated sha1 of HEAD in current
        directory, and {date} by a string built from the current date and time.
        If the file name ends with .pickle or .pickle.gz, a Pickle file is
        created, otherwise YAML format is used. YAML is good for archiving but
        slow to generate and load, Pickle format cannot expected to be backward
        compatible with different versions of the tool but can be faster to read
        and write. CAVEAT: Pickle format will not handle references to modules
        that are not in sys.path.""")

    run_parser.add_argument('--overwrite', action='store_true',
        help="""Overwrite existing report files.""")

    run_parser.add_argument('--resume', action='store_true',
        help="""Resume execution from the report specified with --report. The
        steps will be extracted from the report instead of from the command
        line. The number of completed iterations will be deducted from the
        specified number of iterations.""")

    run_parser.add_argument('--desc',
        default='Executed from ' + os.getcwd() + ' ({commit}, {date})',
        help="""Report description. Can use {commit} and {date}
        placeholders.""")

    run_parser.add_argument('--early-bailout', action='store_true',
        help="""Restart a new iteration if a step marked the commit as bad or
        non-testable. Bisect abortion will still take place even without this
        option.""")

    run_parser.add_argument('--upload-report', action='store_true',
        help="""Continuously upload the report to an artifacts service after
        every iteration. This relies on the following environment variables:
        ARTIFACTORY_FOLDER or ARTIFACTORIAL_FOLDER set to the folder's URL
        and ARTIFACTORY_TOKEN or ARTIFACTORIAL_TOKEN set to the token.
        Remember to use the right step option to upload the results as they
        are computed if desired.
        """)

    dbus_group = run_parser.add_mutually_exclusive_group()
    dbus_group.add_argument('--dbus', dest='dbus', action='store_true',
        help="""Try enable DBus API if the necessary dependencies are installed.""")
    dbus_group.add_argument('--no-dbus', dest='dbus', action='store_false',
        help="""Disable DBus even when pydbus module is available.""")

    # Options for report subcommand
    report_parser.add_argument('report',
        help="Read back a previous session saved using --report option of run subcommand.")

    report_parser.add_argument('--export',
        help="""Export the report as a Pickle or YAML file. File format is
        infered from the filename. If it ends with .pickle, a Pickle file is
        created, otherwise YAML format is used.""")

    report_parser.add_argument('--cache', action='store_true',
        help="""When loading a report, create a cache file named "{template}"
        using the fastest format available. It is the reused until the original
        file is modified. This is mostly useful when working with big YAML
        files that are long to load.""".format(
            template=Report.REPORT_CACHE_TEMPLATE
        )
    )

    # Options for step-help subcommand
    step_help_parser.add_argument('steps', nargs='*',
        default=sorted(
            cls.name for cls in get_subclasses(StepBase)
            if not issubclass(cls, Deprecated)
        ),
        help="Steps name to get help of, or nothing for all steps.")

    # Options for monitor subcommand
    monitor_parser.add_argument('slave_id', metavar='PID',
        default='all',
        type=(lambda slave_id: None if slave_id == 'all' else int(slave_id)),
        help='Slave PID to act on or "all". Start a monitor-server before using "all".')

    monitor_parser.add_argument('--status', action='store_true',
        help="""Show status.""")

    monitor_parser.add_argument('--prop',
        help="""Show given property.""")

    monitor_parser.add_argument('--list', action='store_true',
        help="""List all slaves.""")

    cmd_group = monitor_parser.add_mutually_exclusive_group()

    cmd_group.add_argument('--pause', action='store_true',
        help="""Pause at the end of the iteration.""")

    cmd_group.add_argument('--stop', action='store_true',
        help="""Stop at the end of the iteration.""")

    cmd_group.add_argument('--continue', action='store_true', dest='continue_',
        help="""Continue paused or stopped instance.""")

    cmd_group.add_argument('--kill', action='store_true',
        help="""Kill the given instance.""")

    cmd_group.add_argument('--log', action='store_true',
        help="""Show log in $PAGER.""")

    cmd_group.add_argument('--report', nargs=argparse.REMAINDER,
        help="""Equivalent to running bisector report, all remaining options
        being passed to it.
        """)

    for group in (cmd_group, dbus_service_parser):
        group.add_argument('--notif', nargs=2,
            metavar=('enable/disable', 'PROPERTY'),
            help="Enable and disable desktop notifications when the given property changes. 'all' will select all properties.")

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        if e.code != 0:
            e.code = GENERIC_ERROR_CODE
        raise

    if not args.subcommand:
        parser.print_help()
        return GENERIC_ERROR_CODE

    if args.subcommand == 'step-help':
        cls_list = [get_step_by_name(name) for name in args.steps]
        return do_steps_help(cls_list)

    if args.subcommand == 'monitor-server':
        if args.notif:
            notif = (args.notif[1], args.notif[0] == 'enable')
        else:
            notif = None
        return do_monitor_server(notif)

    elif args.subcommand == 'monitor':
        return do_monitor(
            slave_id=args.slave_id,
            args=args,
        )

    SHOW_TRACEBACK = args.debug
    service_hub = ServiceHub()

    artifacts_service = None
    try:
        artifacts_service = ArtifactoryService()
    except Exception:
        try:
            artifacts_service = ArtifactorialService()
        except Exception:
            error('No artifacts service could be initialised.')
        else:
            info('Artifactorial Service was initialized.')
    else:
        info('Artifactory Service was initialized.')

    if artifacts_service:
        service_hub.register_service('upload', artifacts_service)
        service_hub.register_service('download', artifacts_service)

    # required=True cannot be used in add_argument since it breaks
    # YAMLCLIOptionsAction, so we just handle that manually
    if not args.report:
        error('A report must be specified, see --help')
        return GENERIC_ERROR_CODE

    # Placeholders used to format various parameters.
    placeholder_map = {
        'commit': get_git_sha1(ref='HEAD'),
        'date': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    steps_path = args.steps
    report_path = format_placeholders(args.report, placeholder_map)
    step_options = parse_step_options(args.option)

    if args.subcommand == 'edit':
        report = Report.load(report_path, steps_path, service_hub=service_hub)
        # Partially reinitialize the steps, with the updated command line options
        report.result.step.reinit(step_options=step_options)
        report.save()

        return 0

    # Common options processing for "report" and "run" subcommands

    iteration_n = None
    # TODO: replace by Fisher exact test
    #  binom_stat = args.stat
    binom_stat = None

    allowed_bad_percent = args.allowed_bad
    steps_filter = StepFilter(args.skip, args.only)

    # Use a binomial statistical test if we can otherwise just test for an
    # acceptable threshold of bad iterations.
    if binom_stat:
        good_failure = float(binom_stat[0]) / 100
        good_sample_size = int(binom_stat[1])
        bad_failure = float(binom_stat[2]) / 100
        overall_failure = float(binom_stat[3]) / 100
        commit_n = int(binom_stat[4])
        stat_test = BinomStatTest(good_failure, good_sample_size, bad_failure, commit_n, overall_failure)
        # The binomial test overrides the number of iterations to be able to
        # guarantee an overall failure rate (i.e. the probability of the script
        # giving a wrong answer).
        iteration_n = stat_test.iteration_n
    else:
        stat_test = BasicStatTest(allowed_bad_percent)

    if args.subcommand == 'run':
        # We do not enable DBus if the modules are not available
        if DBUS_CAN_BE_ENABLED:
            enable_dbus = args.dbus
        else:
            enable_dbus = False
            # The user wants DBus, but we can't give it since the modules
            # have not been imported properly
            if args.dbus:
                info('DBus monitoring disabled due to missing dependencies')

        overwrite_report = args.overwrite
        inline_step_list = args.inline
        upload_report = args.upload_report
        git_clean = args.git_clean
        overall_timeout = args.timeout
        bail_out_early = args.early_bailout

        # Do not expose an upload service since the steps will use it if
        # available.
        if upload_report:
            if not service_hub.upload:
                raise ValueError('No upload service is available to use with --upload-report.')
        else:
            service_hub.unregister_service('upload')

        iteration_n = iteration_n or args.iterations
        desc = format_placeholders(args.desc, placeholder_map)
        report_options = {
            'description': desc,
            'path': report_path,
        }
        log_path = args.log or report_path + '.log'

        log_file_mode = 'w'
        if args.resume:
            # When resuming, we don't want to override the existing log
            # file but append to it instead
            log_file_mode = 'a'
            resume_path = report_path
        else:
            resume_path = None

        if not resume_path and os.path.exists(report_path) and not overwrite_report:
            error('File already exists, use --overwrite to overwrite: {path}'.format(
                path=report_path,
            ))
            return GENERIC_ERROR_CODE

        if not (inline_step_list or steps_path or resume_path):
            error('Steps must be specified either using --steps or --inline')
            return GENERIC_ERROR_CODE

        # This log file is available for any function in the global scope
        ensure_dir(log_path)
        LOG_FILE = open(log_path, log_file_mode, encoding='utf-8')

        # Log to the main log file as well as on the console.
        file_handler = logging.StreamHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(BISECTOR_FORMATTER)
        BISECTOR_LOGGER.addHandler(file_handler)

        info(f'Description: {desc}')

        pid = os.getpid()
        info(f'PID: {pid}')

        info('Steps definition: {steps}'.format(
            steps=os.path.abspath(steps_path) if steps_path else '<command line>',
        ))

        info(f'Report: {report_path}')
        info(f'Log: {log_path}')

        if enable_dbus:
            debug('Enabling DBus interface ...')
            dbus_slave_thread = DBusSlaveThread(
                properties=dict(
                    desc=desc,
                    pid=pid,
                    start_ts=math.floor(datetime.datetime.now().timestamp()),
                    start_monotonic_ts=math.floor(time.monotonic()),
                    report_path=os.path.abspath(report_path),
                    log_path=os.path.abspath(log_path),
                )
            )
            # TODO: make a service from that
            slave_manager = dbus_slave_thread.slave_manager
            service_hub.register_service('notif', StepNotifService(slave_manager))
        else:
            debug('Not enabling DBus interface.')
            dbus_slave_thread = None
            slave_manager = None

        # Create the report and save it
        ret, report = do_run(
            slave_manager=slave_manager,
            iteration_n=iteration_n,
            stat_test=stat_test,
            steps_filter=steps_filter,
            bail_out_early=bail_out_early,
            inline_step_list=inline_step_list,
            steps_path=steps_path,
            report_options=report_options,
            overall_timeout=overall_timeout,
            step_options=step_options,
            git_clean=git_clean,
            resume_path=resume_path,
            service_hub=service_hub,
        )

        # Display the summary
        print('\n')
        print(report)

        if dbus_slave_thread:
            dbus_slave_thread.quit_thread()

        return ret

    elif args.subcommand == 'report':
        export_path = args.export
        use_cache = args.cache

        report = Report.load(report_path, steps_path, use_cache=use_cache,
                            service_hub=service_hub)

        out, bisect_ret = report.show(
            service_hub=service_hub,
            steps_filter=steps_filter,
            stat_test=stat_test,
            step_options=step_options,
        )
        print(out)

        # Export the report to another format, after potential modifications
        if export_path:
            export_path = format_placeholders(export_path, placeholder_map)
            report.save(export_path)

        return bisect_ret.value


# TODO: avoid the global variable by redirecting stdout to a tee (stdout and
# file) inheriting from io.TextIOWrapper and contextlib.redirect_stdout()

# Log file opened for the duration of the execution. We initialize it as
# the standard error, and it will be overidden by a call to open() when
# we know which file needs to actually be opened (CLI parameter).
LOG_FILE = sys.stderr

# Might be changed by _main()
SHOW_TRACEBACK = True

BISECTOR_LOGGER = logging.getLogger('BISECTOR')
BISECTOR_FORMATTER = logging.Formatter('[%(name)s][%(asctime)s] %(levelname)s  %(message)s')


def main(argv=sys.argv[1:]):

    BISECTOR_LOGGER.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(BISECTOR_FORMATTER)
    BISECTOR_LOGGER.addHandler(console_handler)

    return_code = None
    try:
        return_code = _main(argv=argv)
    # Quietly exit for these exceptions
    except SILENT_EXCEPTIONS:
        pass
    except SystemExit as e:
        return_code = e.code
    # Catch-all
    except Exception as e:
        if SHOW_TRACEBACK:
            error(
                'Exception traceback:\n' +
                ''.join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                ))
        # Always show the concise message
        error(e)

        return_code = GENERIC_ERROR_CODE
    finally:
        # We only flush without closing in case it is stderr, to avoid hidding
        # exception traceback. It will be closed when the process ends in any
        # case.
        LOG_FILE.flush()

    if return_code is None:
        return_code = 0

    sys.exit(return_code)


if __name__ == '__main__':
    print('bisector needs to be installed using pip and called through its shim, not executed directly', file=sys.stderr)
    sys.exit(2)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
