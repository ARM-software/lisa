#    Copyright 2018 ARM Limited
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

import io
import base64
import functools
import gzip
import glob
import os
import re
import time
import logging
import posixpath
import subprocess
import sys
import tarfile
import tempfile
import threading
import uuid
import xml.dom.minidom
import copy
from collections import namedtuple, defaultdict
from contextlib import contextmanager
from pipes import quote
from past.builtins import long
from past.types import basestring
from numbers import Number
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from enum import Enum

from devlib.host import LocalConnection, PACKAGE_BIN_DIRECTORY
from devlib.module import get_module
from devlib.platform import Platform
from devlib.exception import (DevlibTransientError, TargetStableError,
                              TargetNotRespondingError, TimeoutError,
                              TargetTransientError, KernelConfigKeyError,
                              TargetError, HostError, TargetCalledProcessError) # pylint: disable=redefined-builtin
from devlib.utils.ssh import SshConnection
from devlib.utils.android import AdbConnection, AndroidProperties, LogcatMonitor, adb_command, adb_disconnect, INTENT_FLAGS
from devlib.utils.misc import memoized, isiterable, convert_new_lines, groupby_value
from devlib.utils.misc import commonprefix, merge_lists
from devlib.utils.misc import ABI_MAP, get_cpu_name, ranges_to_list
from devlib.utils.misc import batch_contextmanager, tls_property, nullcontext
from devlib.utils.types import integer, boolean, bitmask, identifier, caseless_string, bytes_regex


FSTAB_ENTRY_REGEX = re.compile(r'(\S+) on (.+) type (\S+) \((\S+)\)')
ANDROID_SCREEN_STATE_REGEX = re.compile('(?:mPowerState|mScreenOn|mWakefulness|Display Power: state)=([0-9]+|true|false|ON|OFF|DOZE|Asleep|Awake)',
                                        re.IGNORECASE)
ANDROID_SCREEN_RESOLUTION_REGEX = re.compile(r'cur=(?P<width>\d+)x(?P<height>\d+)')
ANDROID_SCREEN_ROTATION_REGEX = re.compile(r'orientation=(?P<rotation>[0-3])')
DEFAULT_SHELL_PROMPT = re.compile(r'^.*(shell|root|juno)@?.*:[/~]\S* *[#$] ',
                                  re.MULTILINE)
KVERSION_REGEX = re.compile(
    r'(?P<version>\d+)(\.(?P<major>\d+)(\.(?P<minor>\d+)(-rc(?P<rc>\d+))?)?)?(-(?P<commits>\d+)-g(?P<sha1>[0-9a-fA-F]{7,}))?'
)

GOOGLE_DNS_SERVER_ADDRESS = '8.8.8.8'


installed_package_info = namedtuple('installed_package_info', 'apk_path package')


def call_conn(f):
    """
    Decorator to be used on all :class:`devlib.target.Target` methods that
    directly use a method of ``self.conn``.

    This ensures that if a call to any of the decorated method occurs while
    executing, a new connection will be created in order to avoid possible
    deadlocks. This can happen if e.g. a target's method is called from
    ``__del__``, which could be executed by the garbage collector, interrupting
    another call to a method of the connection instance.

    .. note:: This decorator could be applied directly to all methods with a
        metaclass or ``__init_subclass__`` but it could create issues when
        passing target methods as callbacks to connections' methods.
    """

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        reentered = self.conn.is_in_use
        disconnect = False
        try:
            # If the connection was already in use we need to use a different
            # instance to avoid reentrancy deadlocks. This can happen even in
            # single threaded code via __del__ implementations that can be
            # called at any point.
            if reentered:
                # Shallow copy so we can use another connection instance
                _self = copy.copy(self)
                _self.conn = _self.get_connection()
                assert self.conn is not _self.conn
                disconnect = True
            else:
                _self = self
            return f(_self, *args, **kwargs)
        finally:
            if disconnect:
                _self.disconnect()

    return wrapper


class Target(object):

    path = None
    os = None
    system_id = None

    default_modules = [
        'hotplug',
        'cpufreq',
        'cpuidle',
        'cgroups',
        'hwmon',
    ]

    @property
    def core_names(self):
        return self.platform.core_names

    @property
    def core_clusters(self):
        return self.platform.core_clusters

    @property
    def big_core(self):
        return self.platform.big_core

    @property
    def little_core(self):
        return self.platform.little_core

    @property
    def is_connected(self):
        return self.conn is not None

    @property
    def connected_as_root(self):
        return self.conn and self.conn.connected_as_root

    @property
    def is_rooted(self):
        if self._is_rooted is None:
            try:
                self.execute('ls /', timeout=5, as_root=True)
                self._is_rooted = True
            except(TargetError, TimeoutError):
                self._is_rooted = False

        return self._is_rooted or self.connected_as_root

    @property
    @memoized
    def needs_su(self):
        return not self.connected_as_root and self.is_rooted

    @property
    @memoized
    def kernel_version(self):
        return KernelVersion(self.execute('{} uname -r -v'.format(quote(self.busybox))).strip())

    @property
    def hostid(self):
        return int(self.execute('{} hostid'.format(self.busybox)).strip(), 16)

    @property
    def hostname(self):
        return self.execute('{} hostname'.format(self.busybox)).strip()

    @property
    def os_version(self):  # pylint: disable=no-self-use
        return {}

    @property
    def model(self):
        return self.platform.model

    @property
    def abi(self):  # pylint: disable=no-self-use
        return None

    @property
    def supported_abi(self):
        return [self.abi]

    @property
    @memoized
    def cpuinfo(self):
        return Cpuinfo(self.execute('cat /proc/cpuinfo'))

    @property
    @memoized
    def number_of_cpus(self):
        num_cpus = 0
        corere = re.compile(r'^\s*cpu\d+\s*$')
        output = self.execute('ls /sys/devices/system/cpu', as_root=self.is_rooted)
        for entry in output.split():
            if corere.match(entry):
                num_cpus += 1
        return num_cpus

    @property
    @memoized
    def number_of_nodes(self):
        cmd = 'cd /sys/devices/system/node && {busybox} find . -maxdepth 1'.format(busybox=quote(self.busybox))
        try:
            output = self.execute(cmd, as_root=self.is_rooted)
        except TargetStableError:
            return 1
        else:
            nodere = re.compile(r'^\./node\d+\s*$')
            num_nodes = 0
            for entry in output.splitlines():
                if nodere.match(entry):
                    num_nodes += 1
            return num_nodes

    @property
    @memoized
    def list_nodes_cpus(self):
        nodes_cpus = []
        for node in range(self.number_of_nodes):
            path = self.path.join('/sys/devices/system/node/node{}/cpulist'.format(node))
            output = self.read_value(path)
            nodes_cpus.append(ranges_to_list(output))
        return nodes_cpus

    @property
    @memoized
    def config(self):
        try:
            return KernelConfig(self.execute('zcat /proc/config.gz'))
        except TargetStableError:
            for path in ['/boot/config-$({} uname -r)'.format(self.busybox), '/boot/config']:
                try:
                    return KernelConfig(self.execute('cat {}'.format(path)))
                except TargetStableError:
                    pass
        return KernelConfig('')

    @property
    @memoized
    def user(self):
        return self.getenv('USER')

    @property
    @memoized
    def page_size_kb(self):
        cmd = "cat /proc/self/smaps | {0} grep KernelPageSize | {0} head -n 1 | {0} awk '{{ print $2 }}'"
        return int(self.execute(cmd.format(self.busybox)) or 0)

    @property
    def shutils(self):
        if self._shutils is None:
            self._setup_shutils()
        return self._shutils

    @tls_property
    def _conn(self):
        return self.get_connection()

    # Add a basic property that does not require calling to get the value
    conn = _conn.basic_property

    def __init__(self,
                 connection_settings=None,
                 platform=None,
                 working_directory=None,
                 executables_directory=None,
                 connect=True,
                 modules=None,
                 load_default_modules=True,
                 shell_prompt=DEFAULT_SHELL_PROMPT,
                 conn_cls=None,
                 is_container=False
                 ):

        self._is_rooted = None
        self.connection_settings = connection_settings or {}
        # Set self.platform: either it's given directly (by platform argument)
        # or it's given in the connection_settings argument
        # If neither, create default Platform()
        if platform is None:
            self.platform = self.connection_settings.get('platform', Platform())
        else:
            self.platform = platform
        # Check if the user hasn't given two different platforms
        if 'platform' in self.connection_settings:
            if connection_settings['platform'] is not platform:
                raise TargetStableError('Platform specified in connection_settings '
                                   '({}) differs from that directly passed '
                                   '({})!)'
                                   .format(connection_settings['platform'],
                                    self.platform))
        self.connection_settings['platform'] = self.platform
        self.working_directory = working_directory
        self.executables_directory = executables_directory
        self.modules = modules or []
        self.load_default_modules = load_default_modules
        self.shell_prompt = bytes_regex(shell_prompt)
        self.conn_cls = conn_cls
        self.is_container = is_container
        self.logger = logging.getLogger(self.__class__.__name__)
        self._installed_binaries = {}
        self._installed_modules = {}
        self._cache = {}
        self._shutils = None
        self._file_transfer_cache = None
        self.busybox = None

        if load_default_modules:
            module_lists = [self.default_modules]
        else:
            module_lists = []
        module_lists += [self.modules, self.platform.modules]
        self.modules = merge_lists(*module_lists, duplicates='first')
        self._update_modules('early')
        if connect:
            self.connect()

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            # Avoid sharing the connection instance with the original target,
            # so that each target can live its own independent life
            if k != '_conn'
        }

    # connection and initialization

    def connect(self, timeout=None, check_boot_completed=True):
        self.platform.init_target_connection(self)
        # Forcefully set the thread-local value for the connection, with the
        # timeout we want
        self.conn = self.get_connection(timeout=timeout)
        if check_boot_completed:
            self.wait_boot_complete(timeout)
        self.check_connection()
        self._resolve_paths()
        self.execute('mkdir -p {}'.format(quote(self.working_directory)))
        self.execute('mkdir -p {}'.format(quote(self.executables_directory)))
        self.busybox = self.install(os.path.join(PACKAGE_BIN_DIRECTORY, self.abi, 'busybox'), timeout=30)
        self.conn.busybox = self.busybox
        self.platform.update_from_target(self)
        self._update_modules('connected')
        if self.platform.big_core and self.load_default_modules:
            self._install_module(get_module('bl'))

    def check_connection(self):
        """
        Check that the connection works without obvious issues.
        """
        out = self.execute('true', as_root=False)
        if out.strip():
            raise TargetStableError('The shell seems to not be functional and adds content to stderr: {}'.format(out))

    def disconnect(self):
        connections = self._conn.get_all_values()
        for conn in connections:
            conn.close()

    def get_connection(self, timeout=None):
        if self.conn_cls is None:
            raise ValueError('Connection class not specified on Target creation.')
        return self.conn_cls(timeout=timeout, **self.connection_settings)  # pylint: disable=not-callable

    def wait_boot_complete(self, timeout=10):
        raise NotImplementedError()

    def setup(self, executables=None):
        self._setup_shutils()

        for host_exe in (executables or []):  # pylint: disable=superfluous-parens
            self.install(host_exe)

        # Check for platform dependent setup procedures
        self.platform.setup(self)

        # Initialize modules which requires Buxybox (e.g. shutil dependent tasks)
        self._update_modules('setup')

        self.execute('mkdir -p {}'.format(quote(self._file_transfer_cache)))

    def reboot(self, hard=False, connect=True, timeout=180):
        if hard:
            if not self.has('hard_reset'):
                raise TargetStableError('Hard reset not supported for this target.')
            self.hard_reset()  # pylint: disable=no-member
        else:
            if not self.is_connected:
                message = 'Cannot reboot target becuase it is disconnected. ' +\
                          'Either connect() first, or specify hard=True ' +\
                          '(in which case, a hard_reset module must be installed)'
                raise TargetTransientError(message)
            self.reset()
            # Wait a fixed delay before starting polling to give the target time to
            # shut down, otherwise, might create the connection while it's still shutting
            # down resulting in subsequent connection failing.
            self.logger.debug('Waiting for target to power down...')
            reset_delay = 20
            time.sleep(reset_delay)
            timeout = max(timeout - reset_delay, 10)
        if self.has('boot'):
            self.boot()  # pylint: disable=no-member
        self.conn.connected_as_root = None
        if connect:
            self.connect(timeout=timeout)

    # file transfer

    @contextmanager
    def _xfer_cache_path(self, name):
        """
        Context manager to provide a unique path in the transfer cache with the
        basename of the given name.
        """
        # Use a UUID to avoid race conditions on the target side
        xfer_uuid = uuid.uuid4().hex
        folder = self.path.join(self._file_transfer_cache, xfer_uuid)
        # Make sure basename will work on folders too
        name = os.path.normpath(name)
        # Ensure the name is relative so that os.path.join() will actually
        # join the paths rather than ignoring the first one.
        name = './{}'.format(os.path.basename(name))

        check_rm = False
        try:
            self.makedirs(folder)
            # Don't check the exit code as the folder might not even exist
            # before this point, if creating it failed
            check_rm = True
            yield self.path.join(folder, name)
        finally:
            self.execute('rm -rf -- {}'.format(quote(folder)), check_exit_code=check_rm)

    def _prepare_xfer(self, action, sources, dest, pattern=None, as_root=False):
        """
        Check the sanity of sources and destination and prepare the ground for
        transfering multiple sources.
        """

        once = functools.lru_cache(maxsize=None)

        _target_cache = {}
        def target_paths_kind(paths, as_root=False):
            def process(x):
                x = x.strip()
                if x == 'notexist':
                    return None
                else:
                    return x

            _paths = [
                path
                for path in paths
                if path not in _target_cache
            ]
            if _paths:
                cmd = '; '.join(
                    'if [ -d {path} ]; then echo dir; elif [ -e {path} ]; then echo file; else echo notexist; fi'.format(
                        path=quote(path)
                    )
                    for path in _paths
                )
                res = self.execute(cmd, as_root=as_root)
                _target_cache.update(zip(_paths, map(process, res.split())))

            return [
                _target_cache[path]
                for path in paths
            ]

        _host_cache = {}
        def host_paths_kind(paths, as_root=False):
            def path_kind(path):
                if os.path.isdir(path):
                    return 'dir'
                elif os.path.exists(path):
                    return 'file'
                else:
                    return None

            for path in paths:
                if path not in _host_cache:
                    _host_cache[path] = path_kind(path)

            return [
                _host_cache[path]
                for path in paths
            ]

        # TODO: Target.remove() and Target.makedirs() would probably benefit
        # from being implemented by connections, with the current
        # implementation in ConnectionBase. This would allow SshConnection to
        # use SFTP for these operations, which should be cheaper than
        # Target.execute()
        if action == 'push':
            src_excep = HostError
            src_path_kind = host_paths_kind

            _dst_mkdir = once(self.makedirs)
            dst_path_join = self.path.join
            dst_paths_kind = target_paths_kind
            dst_remove_file = once(functools.partial(self.remove, as_root=as_root))
        elif action == 'pull':
            src_excep = TargetStableError
            src_path_kind = target_paths_kind

            _dst_mkdir = once(functools.partial(os.makedirs, exist_ok=True))
            dst_path_join = os.path.join
            dst_paths_kind = host_paths_kind
            dst_remove_file = once(os.remove)
        else:
            raise ValueError('Unknown action "{}"'.format(action))

        # Handle the case where path is None
        def dst_mkdir(path):
            if path:
                _dst_mkdir(path)

        def rewrite_dst(src, dst):
            new_dst = dst_path_join(dst, os.path.basename(src))

            src_kind, = src_path_kind([src], as_root)
            # Batch both checks to avoid a costly extra execute()
            dst_kind, new_dst_kind = dst_paths_kind([dst, new_dst], as_root)

            if src_kind == 'file':
                if dst_kind == 'dir':
                    if new_dst_kind == 'dir':
                        raise IsADirectoryError(new_dst)
                    if new_dst_kind == 'file':
                        dst_remove_file(new_dst)
                        return new_dst
                    else:
                        return new_dst
                elif dst_kind == 'file':
                    dst_remove_file(dst)
                    return dst
                else:
                    dst_mkdir(os.path.dirname(dst))
                    return dst
            elif src_kind == 'dir':
                if dst_kind == 'dir':
                    # Do not allow writing over an existing folder
                    if new_dst_kind == 'dir':
                        raise FileExistsError(new_dst)
                    if new_dst_kind == 'file':
                        raise FileExistsError(new_dst)
                    else:
                        return new_dst
                elif dst_kind == 'file':
                    raise FileExistsError(dst_kind)
                else:
                    dst_mkdir(os.path.dirname(dst))
                    return dst
            else:
                raise FileNotFoundError(src)

        if pattern:
            if not sources:
                raise src_excep('No file matching source pattern: {}'.format(pattern))

            if dst_paths_kind([dest]) != ['dir']:
                raise NotADirectoryError('A folder dest is required for multiple matches but destination is a file: {}'.format(dest))

        # TODO: since rewrite_dst() will currently return a different path for
        # each source, it will not bring anything. In order to be useful,
        # connections need to be able to understand that if the destination is
        # an empty folder, the source is supposed to be transfered into it with
        # the same basename.
        return groupby_value({
            src: rewrite_dst(src, dest)
            for src in sources
        })

    @call_conn
    def push(self, source, dest, as_root=False, timeout=None, globbing=False):  # pylint: disable=arguments-differ
        source = str(source)
        dest = str(dest)

        sources = glob.glob(source) if globbing else [source]
        mapping = self._prepare_xfer('push', sources, dest, pattern=source if globbing else None, as_root=as_root)

        def do_push(sources, dest):
            return self.conn.push(sources, dest, timeout=timeout)

        if as_root:
            for sources, dest in mapping.items():
                for source in sources:
                    with self._xfer_cache_path(source) as device_tempfile:
                        do_push([source], device_tempfile)
                        self.execute("mv -f -- {} {}".format(quote(device_tempfile), quote(dest)), as_root=True)
        else:
            for sources, dest in mapping.items():
                do_push(sources, dest)

    def _expand_glob(self, pattern, **kwargs):
        """
        Expand the given path globbing pattern on the target using the shell
        globbing.
        """
        # Since we split the results based on new lines, forbid them in the
        # pattern
        if '\n' in pattern:
            raise ValueError(r'Newline character \n are not allowed in globbing patterns')

        # If the pattern is in fact a plain filename, skip the expansion on the
        # target to avoid an unncessary command execution.
        #
        # fnmatch char list from: https://docs.python.org/3/library/fnmatch.html
        special_chars = ['*', '?', '[', ']']
        if not any(char in pattern for char in special_chars):
            return [pattern]

        # Characters to escape that are impacting parameter splitting, since we
        # want the pattern to be given in one piece. Unfortunately, there is no
        # fool-proof way of doing that without also escaping globbing special
        # characters such as wildcard which would defeat the entire purpose of
        # that function.
        for c in [' ', "'", '"']:
            pattern = pattern.replace(c, '\\' + c)

        cmd = "exec printf '%s\n' {}".format(pattern)
        # Make sure to use the same shell everywhere for the path globbing,
        # ensuring consistent results no matter what is the default platform
        # shell
        cmd = '{} sh -c {} 2>/dev/null'.format(quote(self.busybox), quote(cmd))
        # On some shells, match failure will make the command "return" a
        # non-zero code, even though the command was not actually called
        result = self.execute(cmd, strip_colors=False, check_exit_code=False, **kwargs)
        paths = result.splitlines()
        if not paths:
            raise TargetStableError('No file matching: {}'.format(pattern))

        return paths

    @call_conn
    def pull(self, source, dest, as_root=False, timeout=None, globbing=False, via_temp=False):  # pylint: disable=arguments-differ
        source = str(source)
        dest = str(dest)

        if globbing:
            sources = self._expand_glob(source, as_root=as_root)
        else:
            sources = [source]

        # The SSH server might not have the right permissions to read the file,
        # so use a temporary copy instead.
        via_temp |= as_root

        mapping = self._prepare_xfer('pull', sources, dest, pattern=source if globbing else None, as_root=as_root)

        def do_pull(sources, dest):
            self.conn.pull(sources, dest, timeout=timeout)

        if via_temp:
            for sources, dest in mapping.items():
                for source in sources:
                    with self._xfer_cache_path(source) as device_tempfile:
                        self.execute("cp -r -- {} {}".format(quote(source), quote(device_tempfile)), as_root=as_root)
                        self.execute("{} chmod 0644 -- {}".format(self.busybox, quote(device_tempfile)), as_root=as_root)
                        do_pull([device_tempfile], dest)
        else:
            for sources, dest in mapping.items():
                do_pull(sources, dest)

    def get_directory(self, source_dir, dest, as_root=False):
        """ Pull a directory from the device, after compressing dir """
        # Create all file names
        tar_file_name = source_dir.lstrip(self.path.sep).replace(self.path.sep, '.')
        # Host location of dir
        outdir = os.path.join(dest, tar_file_name)
        # Host location of archive
        tar_file_name = '{}.tar'.format(tar_file_name)
        tmpfile = os.path.join(dest, tar_file_name)

        # If root is required, use tmp location for tar creation.
        tar_file_cm = self._xfer_cache_path if as_root else nullcontext

        # Does the folder exist?
        self.execute('ls -la {}'.format(quote(source_dir)), as_root=as_root)

        with tar_file_cm(tar_file_name) as tar_file_name:
            # Try compressing the folder
            try:
                self.execute('{} tar -cvf {} {}'.format(
                    quote(self.busybox), quote(tar_file_name), quote(source_dir)
                ), as_root=as_root)
            except TargetStableError:
                self.logger.debug('Failed to run tar command on target! ' \
                                'Not pulling directory {}'.format(source_dir))
            # Pull the file
            if not os.path.exists(dest):
                os.mkdir(dest)
            self.pull(tar_file_name, tmpfile)
            # Decompress
            with tarfile.open(tmpfile, 'r') as f:
                f.extractall(outdir)
            os.remove(tmpfile)

    # execution

    def _prepare_cmd(self, command, force_locale):
        # Force the locale if necessary for more predictable output
        if force_locale:
            # Use an explicit export so that the command is allowed to be any
            # shell statement, rather than just a command invocation
            command = 'export LC_ALL={} && {}'.format(quote(force_locale), command)

        # Ensure to use deployed command when availables
        if self.executables_directory:
            command = "export PATH={}:$PATH && {}".format(quote(self.executables_directory), command)

        return command

    @call_conn
    def execute(self, command, timeout=None, check_exit_code=True,
                as_root=False, strip_colors=True, will_succeed=False,
                force_locale='C'):

        command = self._prepare_cmd(command, force_locale)
        return self.conn.execute(command, timeout=timeout,
                check_exit_code=check_exit_code, as_root=as_root,
                strip_colors=strip_colors, will_succeed=will_succeed)

    @call_conn
    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False,
                   force_locale='C', timeout=None):
        command = self._prepare_cmd(command, force_locale)
        bg_cmd = self.conn.background(command, stdout, stderr, as_root)
        if timeout is not None:
            timer = threading.Timer(timeout, function=bg_cmd.cancel)
            timer.daemon = True
            timer.start()
        return bg_cmd

    def invoke(self, binary, args=None, in_directory=None, on_cpus=None,
               redirect_stderr=False, as_root=False, timeout=30):
        """
        Executes the specified binary under the specified conditions.

        :binary: binary to execute. Must be present and executable on the device.
        :args: arguments to be passed to the binary. The can be either a list or
               a string.
        :in_directory:  execute the binary in the  specified directory. This must
                        be an absolute path.
        :on_cpus:  taskset the binary to these CPUs. This may be a single ``int`` (in which
                   case, it will be interpreted as the mask), a list of ``ints``, in which
                   case this will be interpreted as the list of cpus, or string, which
                   will be interpreted as a comma-separated list of cpu ranges, e.g.
                   ``"0,4-7"``.
        :as_root: Specify whether the command should be run as root
        :timeout: If the invocation does not terminate within this number of seconds,
                  a ``TimeoutError`` exception will be raised. Set to ``None`` if the
                  invocation should not timeout.

        :returns: output of command.
        """
        command = binary
        if args:
            if isiterable(args):
                args = ' '.join(args)
            command = '{} {}'.format(command, args)
        if on_cpus:
            on_cpus = bitmask(on_cpus)
            command = '{} taskset 0x{:x} {}'.format(quote(self.busybox), on_cpus, command)
        if in_directory:
            command = 'cd {} && {}'.format(quote(in_directory), command)
        if redirect_stderr:
            command = '{} 2>&1'.format(command)
        return self.execute(command, as_root=as_root, timeout=timeout)

    def background_invoke(self, binary, args=None, in_directory=None,
                          on_cpus=None, as_root=False):
        """
        Executes the specified binary as a background task under the
        specified conditions.

        :binary: binary to execute. Must be present and executable on the device.
        :args: arguments to be passed to the binary. The can be either a list or
               a string.
        :in_directory:  execute the binary in the  specified directory. This must
                        be an absolute path.
        :on_cpus:  taskset the binary to these CPUs. This may be a single ``int`` (in which
                   case, it will be interpreted as the mask), a list of ``ints``, in which
                   case this will be interpreted as the list of cpus, or string, which
                   will be interpreted as a comma-separated list of cpu ranges, e.g.
                   ``"0,4-7"``.
        :as_root: Specify whether the command should be run as root

        :returns: the subprocess instance handling that command
        """
        command = binary
        if args:
            if isiterable(args):
                args = ' '.join(args)
            command = '{} {}'.format(command, args)
        if on_cpus:
            on_cpus = bitmask(on_cpus)
            command = '{} taskset 0x{:x} {}'.format(quote(self.busybox), on_cpus, command)
        if in_directory:
            command = 'cd {} && {}'.format(quote(in_directory), command)
        return self.background(command, as_root=as_root)

    def kick_off(self, command, as_root=False):
        raise NotImplementedError()

    # sysfs interaction

    def read_value(self, path, kind=None):
        output = self.execute('cat {}'.format(quote(path)), as_root=self.needs_su).strip()  # pylint: disable=E1103
        if kind:
            return kind(output)
        else:
            return output

    def read_int(self, path):
        return self.read_value(path, kind=integer)

    def read_bool(self, path):
        return self.read_value(path, kind=boolean)

    @contextmanager
    def revertable_write_value(self, path, value, verify=True):
        orig_value = self.read_value(path)
        try:
            self.write_value(path, value, verify)
            yield
        finally:
            self.write_value(path, orig_value, verify)

    def batch_revertable_write_value(self, kwargs_list):
        return batch_contextmanager(self.revertable_write_value, kwargs_list)

    def write_value(self, path, value, verify=True):
        value = str(value)

        if verify:
            # Check in a loop for a while since updates to sysfs files can take
            # some time to be observed, typically when a write triggers a
            # lengthy kernel-side request, and the read is based on some piece
            # of state that may take some time to be updated by the write
            # request, such as hotplugging a CPU.
            cmd = '''
orig=$(cat {path} 2>/dev/null || printf "")
printf "%s" {value} > {path} || exit 10
if [ {value} != "$orig" ]; then
   trials=0
   while [ "$(cat {path} 2>/dev/null)" != {value} ]; do
       if [ $trials -ge 10 ]; then
           cat {path}
           exit 11
       fi
       sleep 0.01
       trials=$((trials + 1))
   done
fi
'''
        else:
            cmd = '{busybox} printf "%s" {value} > {path}'
        cmd = cmd.format(busybox=quote(self.busybox), path=quote(path), value=quote(value))

        try:
            self.execute(cmd, check_exit_code=True, as_root=True)
        except TargetCalledProcessError as e:
            if e.returncode == 10:
                raise TargetStableError('Could not write "{value}" to {path}: {e.output}'.format(
                    value=value, path=path, e=e))
            elif verify and e.returncode == 11:
                out = e.output
                message = 'Could not set the value of {} to "{}" (read "{}")'.format(path, value, out)
                raise TargetStableError(message)
            else:
                raise

    def reset(self):
        try:
            self.execute('reboot', as_root=self.needs_su, timeout=2)
        except (TargetError, subprocess.CalledProcessError):
            # on some targets "reboot" doesn't return gracefully
            pass
        self.conn.connected_as_root = None

    @call_conn
    def check_responsive(self, explode=True):
        try:
            self.conn.execute('ls /', timeout=5)
            return True
        except (DevlibTransientError, subprocess.CalledProcessError):
            if explode:
                raise TargetNotRespondingError('Target {} is not responding'.format(self.conn.name))
            return False

    # process management

    def kill(self, pid, signal=None, as_root=False):
        signal_string = '-s {}'.format(signal) if signal else ''
        self.execute('{} kill {} {}'.format(self.busybox, signal_string, pid), as_root=as_root)

    def killall(self, process_name, signal=None, as_root=False):
        for pid in self.get_pids_of(process_name):
            try:
                self.kill(pid, signal=signal, as_root=as_root)
            except TargetStableError:
                pass

    def get_pids_of(self, process_name):
        raise NotImplementedError()

    def ps(self, **kwargs):
        raise NotImplementedError()

    # files

    def makedirs(self, path, as_root=False):
        self.execute('mkdir -p {}'.format(quote(path)), as_root=as_root)

    def file_exists(self, filepath):
        command = 'if [ -e {} ]; then echo 1; else echo 0; fi'
        output = self.execute(command.format(quote(filepath)), as_root=self.is_rooted)
        return boolean(output.strip())

    def directory_exists(self, filepath):
        output = self.execute('if [ -d {} ]; then echo 1; else echo 0; fi'.format(quote(filepath)))
        # output from ssh my contain part of the expression in the buffer,
        # split out everything except the last word.
        return boolean(output.split()[-1])  # pylint: disable=maybe-no-member

    def list_file_systems(self):
        output = self.execute('mount')
        fstab = []
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            match = FSTAB_ENTRY_REGEX.search(line)
            if match:
                fstab.append(FstabEntry(match.group(1), match.group(2),
                                        match.group(3), match.group(4),
                                        None, None))
            else:  # assume pre-M Android
                fstab.append(FstabEntry(*line.split()))
        return fstab

    def list_directory(self, path, as_root=False):
        raise NotImplementedError()

    def get_workpath(self, name):
        return self.path.join(self.working_directory, name)

    def tempfile(self, prefix='', suffix=''):
        names = tempfile._get_candidate_names()  # pylint: disable=W0212
        for _ in range(tempfile.TMP_MAX):
            name = next(names)
            path = self.get_workpath(prefix + name + suffix)
            if not self.file_exists(path):
                return path
        raise IOError('No usable temporary filename found')

    def remove(self, path, as_root=False):
        self.execute('rm -rf -- {}'.format(quote(path)), as_root=as_root)

    # misc
    def core_cpus(self, core):
        return [i for i, c in enumerate(self.core_names) if c == core]

    def list_online_cpus(self, core=None):
        path = self.path.join('/sys/devices/system/cpu/online')
        output = self.read_value(path)
        all_online = ranges_to_list(output)
        if core:
            cpus = self.core_cpus(core)
            if not cpus:
                raise ValueError(core)
            return [o for o in all_online if o in cpus]
        else:
            return all_online

    def list_offline_cpus(self):
        online = self.list_online_cpus()
        return [c for c in range(self.number_of_cpus)
                if c not in online]

    def getenv(self, variable):
        return self.execute('echo ${}'.format(variable)).rstrip('\r\n')

    def capture_screen(self, filepath):
        raise NotImplementedError()

    def install(self, filepath, timeout=None, with_name=None):
        raise NotImplementedError()

    def uninstall(self, name):
        raise NotImplementedError()

    def get_installed(self, name, search_system_binaries=True):
        # Check user installed binaries first
        if self.file_exists(self.executables_directory):
            if name in self.list_directory(self.executables_directory):
                return self.path.join(self.executables_directory, name)
        # Fall back to binaries in PATH
        if search_system_binaries:
            for path in self.getenv('PATH').split(self.path.pathsep):
                try:
                    if name in self.list_directory(path):
                        return self.path.join(path, name)
                except TargetStableError:
                    pass  # directory does not exist or no executable permissions

    which = get_installed

    def install_if_needed(self, host_path, search_system_binaries=True, timeout=None):

        binary_path = self.get_installed(os.path.split(host_path)[1],
                                         search_system_binaries=search_system_binaries)
        if not binary_path:
            binary_path = self.install(host_path, timeout=timeout)
        return binary_path

    def is_installed(self, name):
        return bool(self.get_installed(name))

    def bin(self, name):
        return self._installed_binaries.get(name, name)

    def has(self, modname):
        return hasattr(self, identifier(modname))

    def lsmod(self):
        lines = self.execute('lsmod').splitlines()
        entries = []
        for line in lines[1:]:  # first line is the header
            if not line.strip():
                continue
            parts = line.split()
            name = parts[0]
            size = int(parts[1])
            use_count = int(parts[2])
            if len(parts) > 3:
                used_by = ''.join(parts[3:]).split(',')
            else:
                used_by = []
            entries.append(LsmodEntry(name, size, use_count, used_by))
        return entries

    def insmod(self, path):
        target_path = self.get_workpath(os.path.basename(path))
        self.push(path, target_path)
        self.execute('insmod {}'.format(quote(target_path)), as_root=True)


    def extract(self, path, dest=None):
        """
        Extract the specified on-target file. The extraction method to be used
        (unzip, gunzip, bunzip2, or tar) will be based on the file's extension.
        If ``dest`` is specified, it must be an existing directory on target;
        the extracted contents will be placed there.

        Note that, depending on the archive file format (and therefore the
        extraction method used), the original archive file may or may not exist
        after the extraction.

        The return value is the path to the extracted contents.  In case of
        gunzip and bunzip2, this will be path to the extracted file; for tar
        and uzip, this will be the directory with the extracted file(s)
        (``dest`` if it was specified otherwise, the directory that contained
        the archive).

        """
        for ending in ['.tar.gz', '.tar.bz', '.tar.bz2',
                       '.tgz', '.tbz', '.tbz2']:
            if path.endswith(ending):
                return self._extract_archive(path, 'tar xf {} -C {}', dest)

        ext = self.path.splitext(path)[1]
        if ext in ['.bz', '.bz2']:
            return self._extract_file(path, 'bunzip2 -f {}', dest)
        elif ext == '.gz':
            return self._extract_file(path, 'gunzip -f {}', dest)
        elif ext == '.zip':
            return self._extract_archive(path, 'unzip {} -d {}', dest)
        else:
            raise ValueError('Unknown compression format: {}'.format(ext))

    def sleep(self, duration):
        timeout = duration + 10
        self.execute('sleep {}'.format(duration), timeout=timeout)

    def read_tree_tar_flat(self, path, depth=1, check_exit_code=True,
                              decode_unicode=True, strip_null_chars=True):
        command = 'read_tree_tgz_b64 {} {} {}'.format(quote(path), depth,
                                                  quote(self.working_directory))
        output = self._execute_util(command, as_root=self.is_rooted,
                                    check_exit_code=check_exit_code)

        result = {}

        # Unpack the archive in memory
        tar_gz = base64.b64decode(output)
        tar_gz_bytes = io.BytesIO(tar_gz)
        tar_buf = gzip.GzipFile(fileobj=tar_gz_bytes).read()
        tar_bytes = io.BytesIO(tar_buf)
        with tarfile.open(fileobj=tar_bytes) as tar:
            for member in tar.getmembers():
                try:
                    content_f = tar.extractfile(member)
                # ignore exotic members like sockets
                except Exception:
                    continue
                # if it is a file and not a folder
                if content_f:
                    content = content_f.read()
                    if decode_unicode:
                        try:
                            content = content.decode('utf-8').strip()
                            if strip_null_chars:
                                content = content.replace('\x00', '').strip()
                        except UnicodeDecodeError:
                            content = ''

                    name = self.path.join(path, member.name)
                    result[name] = content

        return result

    def read_tree_values_flat(self, path, depth=1, check_exit_code=True):
        command = 'read_tree_values {} {}'.format(quote(path), depth)
        output = self._execute_util(command, as_root=self.is_rooted,
                                    check_exit_code=check_exit_code)

        accumulator = defaultdict(list)
        for entry in output.strip().split('\n'):
            if ':' not in entry:
                continue
            path, value = entry.strip().split(':', 1)
            accumulator[path].append(value)

        result = {k: '\n'.join(v).strip() for k, v in accumulator.items()}
        return result

    def read_tree_values(self, path, depth=1, dictcls=dict,
                         check_exit_code=True, tar=False, decode_unicode=True,
                         strip_null_chars=True):
        """
        Reads the content of all files under a given tree

        :path: path to the tree
        :depth: maximum tree depth to read
        :dictcls: type of the dict used to store the results
        :check_exit_code: raise an exception if the shutil command fails
        :tar: fetch the entire tree using tar rather than just the value (more
              robust but slower in some use-cases)
        :decode_unicode: decode the content of tar-ed files as utf-8
        :strip_null_chars: remove '\x00' chars from the content of utf-8
                           decoded files

        :returns: a tree-like dict with the content of files as leafs
        """
        if not tar:
            value_map = self.read_tree_values_flat(path, depth, check_exit_code)
        else:
            value_map = self.read_tree_tar_flat(path, depth, check_exit_code,
                                                decode_unicode,
                                                strip_null_chars)
        return _build_path_tree(value_map, path, self.path.sep, dictcls)

    def install_module(self, mod, **params):
        mod = get_module(mod)
        if mod.stage == 'early':
            msg = 'Module {} cannot be installed after device setup has already occoured.'
            raise TargetStableError(msg)

        if mod.probe(self):
            self._install_module(mod, **params)
        else:
            msg = 'Module {} is not supported by the target'.format(mod.name)
            raise TargetStableError(msg)

    # internal methods

    def _setup_shutils(self):
        shutils_ifile = os.path.join(PACKAGE_BIN_DIRECTORY, 'scripts', 'shutils.in')
        tmp_dir = tempfile.mkdtemp()
        shutils_ofile = os.path.join(tmp_dir, 'shutils')
        with open(shutils_ifile) as fh:
            lines = fh.readlines()
        with open(shutils_ofile, 'w') as ofile:
            for line in lines:
                line = line.replace("__DEVLIB_BUSYBOX__", self.busybox)
                ofile.write(line)
        self._shutils = self.install(shutils_ofile)
        os.remove(shutils_ofile)
        os.rmdir(tmp_dir)

    @call_conn
    def _execute_util(self, command, timeout=None, check_exit_code=True, as_root=False):
        command = '{} sh {} {}'.format(quote(self.busybox), quote(self.shutils), command)
        return self.conn.execute(command, timeout, check_exit_code, as_root)

    def _extract_archive(self, path, cmd, dest=None):
        cmd = '{} ' + cmd  # busybox
        if dest:
            extracted = dest
        else:
            extracted = self.path.dirname(path)
        cmdtext = cmd.format(quote(self.busybox), quote(path), quote(extracted))
        self.execute(cmdtext)
        return extracted

    def _extract_file(self, path, cmd, dest=None):
        cmd = '{} ' + cmd  # busybox
        cmdtext = cmd.format(quote(self.busybox), quote(path))
        self.execute(cmdtext)
        extracted = self.path.splitext(path)[0]
        if dest:
            self.execute('mv -f {} {}'.format(quote(extracted), quote(dest)))
            if dest.endswith('/'):
                extracted = self.path.join(dest, self.path.basename(extracted))
            else:
                extracted = dest
        return extracted

    def _update_modules(self, stage):
        for mod_name in copy.copy(self.modules):
            if isinstance(mod_name, dict):
                mod_name, params = list(mod_name.items())[0]
            else:
                params = {}
            mod = get_module(mod_name)
            if not mod.stage == stage:
                continue
            if mod.probe(self):
                self._install_module(mod, **params)
            else:
                msg = 'Module {} is not supported by the target'.format(mod.name)
                self.modules.remove(mod_name)
                if self.load_default_modules:
                    self.logger.debug(msg)
                else:
                    self.logger.warning(msg)

    def _install_module(self, mod, **params):
        name = mod.name
        if name not in self._installed_modules:
            self.logger.debug('Installing module {}'.format(name))
            try:
                mod.install(self, **params)
            except Exception as e:
                self.logger.error('Module "{}" failed to install on target: {}'.format(name, e))
                raise
            self._installed_modules[name] = mod
            if name not in self.modules:
                self.modules.append(name)
        else:
            self.logger.debug('Module {} is already installed.'.format(name))

    def _resolve_paths(self):
        raise NotImplementedError()

    def is_network_connected(self):
        self.logger.debug('Checking for internet connectivity...')

        timeout_s = 5
        # It would be nice to use busybox for this, but that means we'd need
        # root (ping is usually setuid so it can open raw sockets to send ICMP)
        command = 'ping -q -c 1 -w {} {} 2>&1'.format(timeout_s,
                                                      quote(GOOGLE_DNS_SERVER_ADDRESS))

        # We'll use our own retrying mechanism (rather than just using ping's -c
        # to send multiple packets) so that we don't slow things down in the
        # 'good' case where the first packet gets echoed really quickly.
        attempts = 5
        for _ in range(attempts):
            try:
                self.execute(command)
                return True
            except TargetStableError as e:
                err = str(e).lower()
                if '100% packet loss' in err:
                    # We sent a packet but got no response.
                    # Try again - we don't want this to fail just because of a
                    # transient drop in connection quality.
                    self.logger.debug('No ping response from {} after {}s'
                                      .format(GOOGLE_DNS_SERVER_ADDRESS, timeout_s))
                    continue
                elif 'network is unreachable' in err:
                    # No internet connection at all, we can fail straight away
                    self.logger.debug('Network unreachable')
                    return False
                else:
                    # Something else went wrong, we don't know what, raise an
                    # error.
                    raise

        self.logger.debug('Failed to ping {} after {} attempts'.format(
            GOOGLE_DNS_SERVER_ADDRESS, attempts))
        return False


class LinuxTarget(Target):

    path = posixpath
    os = 'linux'

    @property
    @memoized
    def abi(self):
        value = self.execute('uname -m').strip()
        for abi, architectures in ABI_MAP.items():
            if value in architectures:
                result = abi
                break
        else:
            result = value
        return result

    @property
    @memoized
    def os_version(self):
        os_version = {}
        command = 'ls /etc/*-release /etc*-version /etc/*_release /etc/*_version 2>/dev/null'
        version_files = self.execute(command, check_exit_code=False).strip().split()
        for vf in version_files:
            name = self.path.basename(vf)
            output = self.read_value(vf)
            os_version[name] = convert_new_lines(output.strip()).replace('\n', ' ')
        return os_version

    @property
    @memoized
    def system_id(self):
        return self._execute_util('get_linux_system_id').strip()

    def __init__(self,
                 connection_settings=None,
                 platform=None,
                 working_directory=None,
                 executables_directory=None,
                 connect=True,
                 modules=None,
                 load_default_modules=True,
                 shell_prompt=DEFAULT_SHELL_PROMPT,
                 conn_cls=SshConnection,
                 is_container=False,
                 ):
        super(LinuxTarget, self).__init__(connection_settings=connection_settings,
                                          platform=platform,
                                          working_directory=working_directory,
                                          executables_directory=executables_directory,
                                          connect=connect,
                                          modules=modules,
                                          load_default_modules=load_default_modules,
                                          shell_prompt=shell_prompt,
                                          conn_cls=conn_cls,
                                          is_container=is_container)

    def wait_boot_complete(self, timeout=10):
        pass

    @call_conn
    def kick_off(self, command, as_root=False):
        command = 'sh -c {} 1>/dev/null 2>/dev/null &'.format(quote(command))
        return self.conn.execute(command, as_root=as_root)

    def get_pids_of(self, process_name):
        """Returns a list of PIDs of all processes with the specified name."""
        # result should be a column of PIDs with the first row as "PID" header
        result = self.execute('ps -C {} -o pid'.format(quote(process_name)),  # NOQA
                              check_exit_code=False).strip().split()
        if len(result) >= 2:  # at least one row besides the header
            return list(map(int, result[1:]))
        else:
            return []

    def ps(self, threads=False, **kwargs):
        ps_flags = '-eo'
        if threads:
            ps_flags = '-eLo'
        command = 'ps {} user,pid,tid,ppid,vsize,rss,wchan,pcpu,state,fname'.format(ps_flags)

        lines = iter(convert_new_lines(self.execute(command)).split('\n'))
        next(lines)  # header

        result = []
        for line in lines:
            parts = re.split(r'\s+', line, maxsplit=9)
            if parts and parts != ['']:
                result.append(PsEntry(*(parts[0:1] + list(map(int, parts[1:6])) + parts[6:])))

        if not kwargs:
            return result
        else:
            filtered_result = []
            for entry in result:
                if all(getattr(entry, k) == v for k, v in kwargs.items()):
                    filtered_result.append(entry)
            return filtered_result

    def list_directory(self, path, as_root=False):
        contents = self.execute('ls -1 {}'.format(quote(path)), as_root=as_root)
        return [x.strip() for x in contents.split('\n') if x.strip()]

    def install(self, filepath, timeout=None, with_name=None):  # pylint: disable=W0221
        destpath = self.path.join(self.executables_directory,
                                  with_name and with_name or self.path.basename(filepath))
        self.push(filepath, destpath, timeout=timeout)
        self.execute('chmod a+x {}'.format(quote(destpath)), timeout=timeout)
        self._installed_binaries[self.path.basename(destpath)] = destpath
        return destpath

    def uninstall(self, name):
        path = self.path.join(self.executables_directory, name)
        self.remove(path)

    def capture_screen(self, filepath):
        if not self.is_installed('scrot'):
            self.logger.debug('Could not take screenshot as scrot is not installed.')
            return
        try:

            tmpfile = self.tempfile()
            cmd = 'DISPLAY=:0.0 scrot {} && {} date -u -Iseconds'
            ts = self.execute(cmd.format(quote(tmpfile), quote(self.busybox))).strip()
            filepath = filepath.format(ts=ts)
            self.pull(tmpfile, filepath)
            self.remove(tmpfile)
        except TargetStableError as e:
            if "Can't open X dispay." not in e.message:
                raise e
            message = e.message.split('OUTPUT:', 1)[1].strip()  # pylint: disable=no-member
            self.logger.debug('Could not take screenshot: {}'.format(message))

    def _resolve_paths(self):
        if self.working_directory is None:
            self.working_directory = self.path.join(self.execute("pwd").strip(), 'devlib-target')
        self._file_transfer_cache = self.path.join(self.working_directory, '.file-cache')
        if self.executables_directory is None:
            self.executables_directory = self.path.join(self.working_directory, 'bin')


class AndroidTarget(Target):

    path = posixpath
    os = 'android'
    ls_command = ''

    @property
    @memoized
    def abi(self):
        return self.getprop()['ro.product.cpu.abi'].split('-')[0]

    @property
    @memoized
    def supported_abi(self):
        props = self.getprop()
        result = [props['ro.product.cpu.abi']]
        if 'ro.product.cpu.abi2' in props:
            result.append(props['ro.product.cpu.abi2'])
        if 'ro.product.cpu.abilist' in props:
            for abi in props['ro.product.cpu.abilist'].split(','):
                if abi not in result:
                    result.append(abi)

        mapped_result = []
        for supported_abi in result:
            for abi, architectures in ABI_MAP.items():
                found = False
                if supported_abi in architectures and abi not in mapped_result:
                    mapped_result.append(abi)
                    found = True
                    break
            if not found and supported_abi not in mapped_result:
                mapped_result.append(supported_abi)
        return mapped_result

    @property
    @memoized
    def os_version(self):
        os_version = {}
        for k, v in self.getprop().iteritems():
            if k.startswith('ro.build.version'):
                part = k.split('.')[-1]
                os_version[part] = v
        return os_version

    @property
    def adb_name(self):
        return getattr(self.conn, 'device', None)

    @property
    def adb_server(self):
        return getattr(self.conn, 'adb_server', None)

    @property
    @memoized
    def android_id(self):
        """
        Get the device's ANDROID_ID. Which is

            "A 64-bit number (as a hex string) that is randomly generated when the user
            first sets up the device and should remain constant for the lifetime of the
            user's device."

        .. note:: This will get reset on userdata erasure.

        """
        output = self.execute('content query --uri content://settings/secure --projection value --where "name=\'android_id\'"').strip()
        return output.split('value=')[-1]

    @property
    @memoized
    def system_id(self):
        return self._execute_util('get_android_system_id').strip()

    @property
    @memoized
    def external_storage(self):
        return self.execute('echo $EXTERNAL_STORAGE').strip()

    @property
    @memoized
    def screen_resolution(self):
        output = self.execute('dumpsys window displays')
        match = ANDROID_SCREEN_RESOLUTION_REGEX.search(output)
        if match:
            return (int(match.group('width')),
                    int(match.group('height')))
        else:
            return (0, 0)

    def __init__(self,
                 connection_settings=None,
                 platform=None,
                 working_directory=None,
                 executables_directory=None,
                 connect=True,
                 modules=None,
                 load_default_modules=True,
                 shell_prompt=DEFAULT_SHELL_PROMPT,
                 conn_cls=AdbConnection,
                 package_data_directory="/data/data",
                 is_container=False,
                 ):
        super(AndroidTarget, self).__init__(connection_settings=connection_settings,
                                            platform=platform,
                                            working_directory=working_directory,
                                            executables_directory=executables_directory,
                                            connect=connect,
                                            modules=modules,
                                            load_default_modules=load_default_modules,
                                            shell_prompt=shell_prompt,
                                            conn_cls=conn_cls,
                                            is_container=is_container)
        self.package_data_directory = package_data_directory
        self._init_logcat_lock()

    def _init_logcat_lock(self):
        self.clear_logcat_lock = threading.Lock()

    def __getstate__(self):
        dct = super().__getstate__()
        return {
            k: v
            for k, v in dct.items()
            if k not in ('clear_logcat_lock',)
        }

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        self._init_logcat_lock()

    def reset(self, fastboot=False):  # pylint: disable=arguments-differ
        try:
            self.execute('reboot {}'.format(fastboot and 'fastboot' or ''),
                         as_root=self.needs_su, timeout=2)
        except (DevlibTransientError, subprocess.CalledProcessError):
            # on some targets "reboot" doesn't return gracefully
            pass
        self.conn.connected_as_root = None

    def wait_boot_complete(self, timeout=10):
        start = time.time()
        boot_completed = boolean(self.getprop('sys.boot_completed'))
        while not boot_completed and timeout >= time.time() - start:
            time.sleep(5)
            boot_completed = boolean(self.getprop('sys.boot_completed'))
        if not boot_completed:
            # Raise a TargetStableError as this usually happens because of
            # an issue with Android more than a timeout that is too small.
            raise TargetStableError('Connected but Android did not fully boot.')

    def connect(self, timeout=30, check_boot_completed=True):  # pylint: disable=arguments-differ
        device = self.connection_settings.get('device')
        super(AndroidTarget, self).connect(timeout=timeout, check_boot_completed=check_boot_completed)

    def kick_off(self, command, as_root=None):
        """
        Like execute but closes adb session and returns immediately, leaving the command running on the
        device (this is different from execute(background=True) which keeps adb connection open and returns
        a subprocess object).
        """
        if as_root is None:
            as_root = self.needs_su
        try:
            command = 'cd {} && {} nohup {} &'.format(quote(self.working_directory), quote(self.busybox), command)
            self.execute(command, timeout=1, as_root=as_root)
        except TimeoutError:
            pass

    def __setup_list_directory(self):
        # In at least Linaro Android 16.09 (which was their first Android 7 release) and maybe
        # AOSP 7.0 as well, the ls command was changed.
        # Previous versions default to a single column listing, which is nice and easy to parse.
        # Newer versions default to a multi-column listing, which is not, but it does support
        # a '-1' option to get into single column mode. Older versions do not support this option
        # so we try the new version, and if it fails we use the old version.
        self.ls_command = 'ls -1'
        try:
            self.execute('ls -1 {}'.format(quote(self.working_directory)), as_root=False)
        except TargetStableError:
            self.ls_command = 'ls'

    def list_directory(self, path, as_root=False):
        if self.ls_command == '':
            self.__setup_list_directory()
        contents = self.execute('{} {}'.format(self.ls_command, quote(path)), as_root=as_root)
        return [x.strip() for x in contents.split('\n') if x.strip()]

    def install(self, filepath, timeout=None, with_name=None):  # pylint: disable=W0221
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            return self.install_apk(filepath, timeout)
        else:
            return self.install_executable(filepath, with_name, timeout)

    def uninstall(self, name):
        if self.package_is_installed(name):
            self.uninstall_package(name)
        else:
            self.uninstall_executable(name)

    def get_pids_of(self, process_name):
        result = []
        search_term = process_name[-15:]
        for entry in self.ps():
            if search_term in entry.name:
                result.append(entry.pid)
        return result

    def ps(self, threads=False, **kwargs):
        maxsplit = 9 if threads else 8
        command = 'ps'
        if threads:
            command = 'ps -AT'

        lines = iter(convert_new_lines(self.execute(command)).split('\n'))
        next(lines)  # header
        result = []
        for line in lines:
            parts = line.split(None, maxsplit)
            if not parts:
                continue

            wchan_missing = False
            if len(parts) == maxsplit:
                wchan_missing = True

            if not threads:
                # Duplicate PID into TID location.
                parts.insert(2, parts[1])

            if wchan_missing:
                # wchan was blank; insert an empty field where it should be.
                parts.insert(6, '')
            result.append(PsEntry(*(parts[0:1] + list(map(int, parts[1:6])) + parts[6:])))
        if not kwargs:
            return result
        else:
            filtered_result = []
            for entry in result:
                if all(getattr(entry, k) == v for k, v in kwargs.items()):
                    filtered_result.append(entry)
            return filtered_result

    def capture_screen(self, filepath):
        on_device_file = self.path.join(self.working_directory, 'screen_capture.png')
        cmd = 'screencap -p  {} && {} date -u -Iseconds'
        ts = self.execute(cmd.format(quote(on_device_file), quote(self.busybox))).strip()
        filepath = filepath.format(ts=ts)
        self.pull(on_device_file, filepath)
        self.remove(on_device_file)

    # Android-specific

    def input_tap(self, x, y):
        command = 'input tap {} {}'
        self.execute(command.format(x, y))

    def input_tap_pct(self, x, y):
        width, height = self.screen_resolution

        x = (x * width) // 100
        y = (y * height) // 100

        self.input_tap(x, y)

    def input_swipe(self, x1, y1, x2, y2):
        """
        Issue a swipe on the screen from (x1, y1) to (x2, y2)
        Uses absolute screen positions
        """
        command = 'input swipe {} {} {} {}'
        self.execute(command.format(x1, y1, x2, y2))

    def input_swipe_pct(self, x1, y1, x2, y2):
        """
        Issue a swipe on the screen from (x1, y1) to (x2, y2)
        Uses percent-based positions
        """
        width, height = self.screen_resolution

        x1 = (x1 * width) // 100
        y1 = (y1 * height) // 100
        x2 = (x2 * width) // 100
        y2 = (y2 * height) // 100

        self.input_swipe(x1, y1, x2, y2)

    def swipe_to_unlock(self, direction="diagonal"):
        width, height = self.screen_resolution
        if direction == "diagonal":
            start = 100
            stop = width - start
            swipe_height = height * 2 // 3
            self.input_swipe(start, swipe_height, stop, 0)
        elif direction == "horizontal":
            swipe_height = height * 2 // 3
            start = 100
            stop = width - start
            self.input_swipe(start, swipe_height, stop, swipe_height)
        elif direction == "vertical":
            swipe_middle = width / 2
            swipe_height = height * 2 // 3
            self.input_swipe(swipe_middle, swipe_height, swipe_middle, 0)
        else:
            raise TargetStableError("Invalid swipe direction: {}".format(direction))

    def getprop(self, prop=None):
        props = AndroidProperties(self.execute('getprop'))
        if prop:
            return props[prop]
        return props

    def capture_ui_hierarchy(self, filepath):
        on_target_file = self.get_workpath('screen_capture.xml')
        self.execute('uiautomator dump {}'.format(on_target_file))
        self.pull(on_target_file, filepath)
        self.remove(on_target_file)

        parsed_xml = xml.dom.minidom.parse(filepath)
        with open(filepath, 'w') as f:
            if sys.version_info[0] == 3:
                f.write(parsed_xml.toprettyxml())
            else:
                f.write(parsed_xml.toprettyxml().encode('utf-8'))

    def is_installed(self, name):
        return super(AndroidTarget, self).is_installed(name) or self.package_is_installed(name)

    def package_is_installed(self, package_name):
        return package_name in self.list_packages()

    def list_packages(self):
        output = self.execute('pm list packages')
        output = output.replace('package:', '')
        return output.split()

    def get_package_version(self, package):
        output = self.execute('dumpsys package {}'.format(quote(package)))
        for line in convert_new_lines(output).split('\n'):
            if 'versionName' in line:
                return line.split('=', 1)[1]
        return None

    def get_package_info(self, package):
        output = self.execute('pm list packages -f {}'.format(quote(package)))
        for entry in output.strip().split('\n'):
            rest, entry_package = entry.rsplit('=', 1)
            if entry_package != package:
                continue
            _, apk_path = rest.split(':')
            return installed_package_info(apk_path, entry_package)

    def get_sdk_version(self):
        try:
            return int(self.getprop('ro.build.version.sdk'))
        except (ValueError, TypeError):
            return None

    def install_apk(self, filepath, timeout=None, replace=False, allow_downgrade=False):  # pylint: disable=W0221
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            flags = []
            if replace:
                flags.append('-r')  # Replace existing APK
            if allow_downgrade:
                flags.append('-d')  # Install the APK even if a newer version is already installed
            if self.get_sdk_version() >= 23:
                flags.append('-g')  # Grant all runtime permissions
            self.logger.debug("Replace APK = {}, ADB flags = '{}'".format(replace, ' '.join(flags)))
            if isinstance(self.conn, AdbConnection):
                return adb_command(self.adb_name, "install {} {}".format(' '.join(flags), quote(filepath)),
                                   timeout=timeout, adb_server=self.adb_server)
            else:
                dev_path = self.get_workpath(filepath.rsplit(os.path.sep, 1)[-1])
                self.push(quote(filepath), dev_path, timeout=timeout)
                result = self.execute("pm install {} {}".format(' '.join(flags), quote(dev_path)), timeout=timeout)
                self.remove(dev_path)
                return result
        else:
            raise TargetStableError('Can\'t install {}: unsupported format.'.format(filepath))

    def grant_package_permission(self, package, permission):
        try:
            return self.execute('pm grant {} {}'.format(quote(package), quote(permission)))
        except TargetStableError as e:
            if 'is not a changeable permission type' in e.message:
                pass # Ignore if unchangeable
            elif 'Unknown permission' in e.message:
                pass # Ignore if unknown
            elif 'has not requested permission' in e.message:
                pass # Ignore if not requested
            elif 'Operation not allowed' in e.message:
                pass # Ignore if not allowed
            else:
                raise

    def refresh_files(self, file_list):
        """
        Depending on the android version and root status, determine the
        appropriate method of forcing a re-index of the mediaserver cache for a given
        list of files.
        """
        if self.is_rooted or self.get_sdk_version() < 24:  # MM and below
            common_path = commonprefix(file_list, sep=self.path.sep)
            self.broadcast_media_mounted(common_path, self.is_rooted)
        else:
            for f in file_list:
                self.broadcast_media_scan_file(f)

    def broadcast_media_scan_file(self, filepath):
        """
        Force a re-index of the mediaserver cache for the specified file.
        """
        command = 'am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d {}'
        self.execute(command.format(quote('file://' + filepath)))

    def broadcast_media_mounted(self, dirpath, as_root=False):
        """
        Force a re-index of the mediaserver cache for the specified directory.
        """
        command = 'am broadcast -a  android.intent.action.MEDIA_MOUNTED -d {} '\
                  '-n com.android.providers.media/.MediaScannerReceiver'
        self.execute(command.format(quote('file://'+dirpath)), as_root=as_root)

    def install_executable(self, filepath, with_name=None, timeout=None):
        self._ensure_executables_directory_is_writable()
        executable_name = with_name or os.path.basename(filepath)
        on_device_file = self.path.join(self.working_directory, executable_name)
        on_device_executable = self.path.join(self.executables_directory, executable_name)
        self.push(filepath, on_device_file, timeout=timeout)
        if on_device_file != on_device_executable:
            self.execute('cp {} {}'.format(quote(on_device_file), quote(on_device_executable)),
                         as_root=self.needs_su, timeout=timeout)
            self.remove(on_device_file, as_root=self.needs_su)
        self.execute("chmod 0777 {}".format(quote(on_device_executable)), as_root=self.needs_su)
        self._installed_binaries[executable_name] = on_device_executable
        return on_device_executable

    def uninstall_package(self, package):
        if isinstance(self.conn, AdbConnection):
            adb_command(self.adb_name, "uninstall {}".format(quote(package)), timeout=30,
                        adb_server=self.adb_server)
        else:
            self.execute("pm uninstall {}".format(quote(package)), timeout=30)

    def uninstall_executable(self, executable_name):
        on_device_executable = self.path.join(self.executables_directory, executable_name)
        self._ensure_executables_directory_is_writable()
        self.remove(on_device_executable, as_root=self.needs_su)

    def dump_logcat(self, filepath, filter=None, logcat_format=None, append=False,
                    timeout=60):  # pylint: disable=redefined-builtin
        op = '>>' if append else '>'
        filtstr = ' -s {}'.format(quote(filter)) if filter else ''
        formatstr = ' -v {}'.format(quote(logcat_format)) if logcat_format else ''
        logcat_opts = '-d' + formatstr + filtstr
        if isinstance(self.conn, AdbConnection):
            command = 'logcat {} {} {}'.format(logcat_opts, op, quote(filepath))
            adb_command(self.adb_name, command, timeout=timeout, adb_server=self.adb_server)
        else:
            dev_path = self.get_workpath('logcat')
            command = 'logcat {} {} {}'.format(logcat_opts, op, quote(dev_path))
            self.execute(command, timeout=timeout)
            self.pull(dev_path, filepath)
            self.remove(dev_path)

    def clear_logcat(self):
        locked = self.clear_logcat_lock.acquire(blocking=False)
        if locked:
            try:
                if isinstance(self.conn, AdbConnection):
                    adb_command(self.adb_name, 'logcat -c', timeout=30, adb_server=self.adb_server)
                else:
                    self.execute('logcat -c', timeout=30)
            finally:
                self.clear_logcat_lock.release()

    def get_logcat_monitor(self, regexps=None):
        return LogcatMonitor(self, regexps)

    @call_conn
    def wait_for_device(self, timeout=30):
        self.conn.wait_for_device()

    @call_conn
    def reboot_bootloader(self, timeout=30):
        self.conn.reboot_bootloader()

    def is_screen_on(self):
        output = self.execute('dumpsys power')
        match = ANDROID_SCREEN_STATE_REGEX.search(output)
        if match:
            if 'DOZE' in match.group(1).upper():
                return True
            if match.group(1) == 'Asleep':
                return False
            if match.group(1) == 'Awake':
                return True
            return boolean(match.group(1))
        else:
            raise TargetStableError('Could not establish screen state.')

    def ensure_screen_is_on(self, verify=True):
        if not self.is_screen_on():
            self.execute('input keyevent 26')
        if verify and not self.is_screen_on():
             raise TargetStableError('Display cannot be turned on.')

    def ensure_screen_is_on_and_stays(self, verify=True, mode=7):
        self.ensure_screen_is_on(verify=verify)
        self.set_stay_on_mode(mode)

    def ensure_screen_is_off(self, verify=True):
        # Allow 2 attempts to help with cases of ambient display modes
        # where the first attempt will switch the display fully on.
        for _ in range(2):
            if self.is_screen_on():
                self.execute('input keyevent 26')
                time.sleep(0.5)
        if verify and self.is_screen_on():
             msg = 'Display cannot be turned off. Is always on display enabled?'
             raise TargetStableError(msg)

    def set_auto_brightness(self, auto_brightness):
        cmd = 'settings put system screen_brightness_mode {}'
        self.execute(cmd.format(int(boolean(auto_brightness))))

    def get_auto_brightness(self):
        cmd = 'settings get system screen_brightness_mode'
        return boolean(self.execute(cmd).strip())

    def set_brightness(self, value):
        if not 0 <= value <= 255:
            msg = 'Invalid brightness "{}"; Must be between 0 and 255'
            raise ValueError(msg.format(value))
        self.set_auto_brightness(False)
        cmd = 'settings put system screen_brightness {}'
        self.execute(cmd.format(int(value)))

    def get_brightness(self):
        cmd = 'settings get system screen_brightness'
        return integer(self.execute(cmd).strip())

    def set_screen_timeout(self, timeout_ms):
        cmd = 'settings put system screen_off_timeout {}'
        self.execute(cmd.format(int(timeout_ms)))

    def get_screen_timeout(self):
        cmd = 'settings get system screen_off_timeout'
        return int(self.execute(cmd).strip())

    def get_airplane_mode(self):
        cmd = 'settings get global airplane_mode_on'
        return boolean(self.execute(cmd).strip())

    def get_stay_on_mode(self):
        cmd = 'settings get global stay_on_while_plugged_in'
        return int(self.execute(cmd).strip())

    def set_airplane_mode(self, mode):
        root_required = self.get_sdk_version() > 23
        if root_required and not self.is_rooted:
            raise TargetStableError('Root is required to toggle airplane mode on Android 7+')
        mode = int(boolean(mode))
        cmd = 'settings put global airplane_mode_on {}'
        self.execute(cmd.format(mode))
        self.execute('am broadcast -a android.intent.action.AIRPLANE_MODE '
                     '--ez state {}'.format(mode), as_root=root_required)

    def get_auto_rotation(self):
        cmd = 'settings get system accelerometer_rotation'
        return boolean(self.execute(cmd).strip())

    def set_auto_rotation(self, autorotate):
        cmd = 'settings put system accelerometer_rotation {}'
        self.execute(cmd.format(int(boolean(autorotate))))

    def set_natural_rotation(self):
        self.set_rotation(0)

    def set_left_rotation(self):
        self.set_rotation(1)

    def set_inverted_rotation(self):
        self.set_rotation(2)

    def set_right_rotation(self):
        self.set_rotation(3)

    def get_rotation(self):
        output = self.execute('dumpsys input')
        match = ANDROID_SCREEN_ROTATION_REGEX.search(output)
        if match:
            return int(match.group('rotation'))
        else:
            return None

    def set_rotation(self, rotation):
        if not 0 <= rotation <= 3:
            raise ValueError('Rotation value must be between 0 and 3')
        self.set_auto_rotation(False)
        cmd = 'settings put system user_rotation {}'
        self.execute(cmd.format(rotation))

    def set_stay_on_never(self):
        self.set_stay_on_mode(0)

    def set_stay_on_while_powered(self):
        self.set_stay_on_mode(7)

    def set_stay_on_mode(self, mode):
        if not 0 <= mode <= 7:
            raise ValueError('Screen stay on mode must be between 0 and 7')
        cmd = 'settings put global stay_on_while_plugged_in {}'
        self.execute(cmd.format(mode))

    def open_url(self, url, force_new=False):
        """
        Start a view activity by specifying an URL

        :param url: URL of the item to display
        :type url: str

        :param force_new: Force the viewing application to be relaunched
            if it is already running
        :type force_new: bool
        """
        cmd = 'am start -a android.intent.action.VIEW -d {}'

        if force_new:
            cmd = cmd + ' -f {}'.format(INTENT_FLAGS['ACTIVITY_NEW_TASK'] |
                                        INTENT_FLAGS['ACTIVITY_CLEAR_TASK'])

        self.execute(cmd.format(quote(url)))

    def homescreen(self):
        self.execute('am start -a android.intent.action.MAIN -c android.intent.category.HOME')

    def _resolve_paths(self):
        if self.working_directory is None:
            self.working_directory = self.path.join(self.external_storage, 'devlib-target')
        self._file_transfer_cache = self.path.join(self.working_directory, '.file-cache')
        if self.executables_directory is None:
            self.executables_directory = '/data/local/tmp/bin'

    def _ensure_executables_directory_is_writable(self):
        matched = []
        for entry in self.list_file_systems():
            if self.executables_directory.rstrip('/').startswith(entry.mount_point):
                matched.append(entry)
        if matched:
            entry = sorted(matched, key=lambda x: len(x.mount_point))[-1]
            if 'rw' not in entry.options:
                self.execute('mount -o rw,remount {} {}'.format(quote(entry.device),
                                                                quote(entry.mount_point)),
                             as_root=True)
        else:
            message = 'Could not find mount point for executables directory {}'
            raise TargetStableError(message.format(self.executables_directory))

    _charging_enabled_path = '/sys/class/power_supply/battery/charging_enabled'

    @property
    def charging_enabled(self):
        """
        Whether drawing power to charge the battery is enabled

        Not all devices have the ability to enable/disable battery charging
        (e.g. because they don't have a battery). In that case,
        ``charging_enabled`` is None.
        """
        if not self.file_exists(self._charging_enabled_path):
            return None
        return self.read_bool(self._charging_enabled_path)

    @charging_enabled.setter
    def charging_enabled(self, enabled):
        """
        Enable/disable drawing power to charge the battery

        Not all devices have this facility. In that case, do nothing.
        """
        if not self.file_exists(self._charging_enabled_path):
            return
        self.write_value(self._charging_enabled_path, int(bool(enabled)))

FstabEntry = namedtuple('FstabEntry', ['device', 'mount_point', 'fs_type', 'options', 'dump_freq', 'pass_num'])
PsEntry = namedtuple('PsEntry', 'user pid tid ppid vsize rss wchan pc state name')
LsmodEntry = namedtuple('LsmodEntry', ['name', 'size', 'use_count', 'used_by'])


class Cpuinfo(object):

    @property
    @memoized
    def architecture(self):
        for section in self.sections:
            if 'CPU architecture' in section:
                return section['CPU architecture']
            if 'architecture' in section:
                return section['architecture']

    @property
    @memoized
    def cpu_names(self):
        cpu_names = []
        global_name = None
        for section in self.sections:
            if 'processor' in section:
                if 'CPU part' in section:
                    cpu_names.append(_get_part_name(section))
                elif 'model name' in section:
                    cpu_names.append(_get_model_name(section))
                else:
                    cpu_names.append(None)
            elif 'CPU part' in section:
                global_name = _get_part_name(section)
        return [caseless_string(c or global_name) for c in cpu_names]

    def __init__(self, text):
        self.sections = None
        self.text = None
        self.parse(text)

    @memoized
    def get_cpu_features(self, cpuid=0):
        global_features = []
        for section in self.sections:
            if 'processor' in section:
                if int(section.get('processor')) != cpuid:
                    continue
                if 'Features' in section:
                    return section.get('Features').split()
                elif 'flags' in section:
                    return section.get('flags').split()
            elif 'Features' in section:
                global_features = section.get('Features').split()
            elif 'flags' in section:
                global_features = section.get('flags').split()
        return global_features

    def parse(self, text):
        self.sections = []
        current_section = {}
        self.text = text.strip()
        for line in self.text.split('\n'):
            line = line.strip()
            if line:
                key, value = line.split(':', 1)
                current_section[key.strip()] = value.strip()
            else:  # not line
                self.sections.append(current_section)
                current_section = {}
        self.sections.append(current_section)

    def __str__(self):
        return 'CpuInfo({})'.format(self.cpu_names)

    __repr__ = __str__


class KernelVersion(object):
    """
    Class representing the version of a target kernel

    Not expected to work for very old (pre-3.0) kernel version numbers.

    :ivar release: Version number/revision string. Typical output of
                   ``uname -r``
    :type release: str
    :ivar version: Extra version info (aside from ``release``) reported by
                   ``uname``
    :type version: str
    :ivar version_number: Main version number (e.g. 3 for Linux 3.18)
    :type version_number: int
    :ivar major: Major version number (e.g. 18 for Linux 3.18)
    :type major: int
    :ivar minor: Minor version number for stable kernels (e.g. 9 for 4.9.9). May
                 be None
    :type minor: int
    :ivar rc: Release candidate number (e.g. 3 for Linux 4.9-rc3). May be None.
    :type rc: int
    :ivar commits: Number of additional commits on the branch. May be None.
    :type commits: int
    :ivar sha1: Kernel git revision hash, if available (otherwise None)
    :type sha1: str

    :ivar parts: Tuple of version number components. Can be used for
                 lexicographically comparing kernel versions.
    :type parts: tuple(int)
    """
    def __init__(self, version_string):
        if ' #' in version_string:
            release, version = version_string.split(' #')
            self.release = release
            self.version = version
        elif version_string.startswith('#'):
            self.release = ''
            self.version = version_string
        else:
            self.release = version_string
            self.version = ''

        self.version_number = None
        self.major = None
        self.minor = None
        self.sha1 = None
        self.rc = None
        self.commits = None
        match = KVERSION_REGEX.match(version_string)
        if match:
            groups = match.groupdict()
            self.version_number = int(groups['version'])
            self.major = int(groups['major'])
            if groups['minor'] is not None:
                self.minor = int(groups['minor'])
            if groups['rc'] is not None:
                self.rc = int(groups['rc'])
            if groups['commits'] is not None:
                self.commits = int(groups['commits'])
            if groups['sha1'] is not None:
                self.sha1 = match.group('sha1')

        self.parts = (self.version_number, self.major, self.minor)

    def __str__(self):
        return '{} {}'.format(self.release, self.version)

    __repr__ = __str__


class HexInt(long):
    """
    Subclass of :class:`int` that uses hexadecimal formatting by default.
    """

    def __new__(cls, val=0, base=16):
        super_new = super(HexInt, cls).__new__
        if isinstance(val, Number):
            return super_new(cls, val)
        else:
            return super_new(cls, val, base=base)

    def __str__(self):
        return hex(self).strip('L')


class KernelConfigTristate(Enum):
    YES = 'y'
    NO = 'n'
    MODULE = 'm'

    def __bool__(self):
        """
        Allow using this enum to represent bool Kconfig type, although it is
        technically different from tristate.
        """
        return self in (self.YES, self.MODULE)

    def __nonzero__(self):
        """
        For Python 2.x compatibility.
        """
        return self.__bool__()

    @classmethod
    def from_str(cls, str_):
        for state in cls:
            if state.value == str_:
                return state
        raise ValueError('No kernel config tristate value matches "{}"'.format(str_))


class TypedKernelConfig(Mapping):
    """
    Mapping-like typed version of :class:`KernelConfig`.

    Values are either :class:`str`, :class:`int`,
    :class:`KernelConfigTristate`, or :class:`HexInt`. ``hex`` Kconfig type is
    mapped to :class:`HexInt` and ``bool`` to :class:`KernelConfigTristate`.
    """
    not_set_regex = re.compile(r'# (\S+) is not set')

    @staticmethod
    def get_config_name(name):
        name = name.upper()
        if not name.startswith('CONFIG_'):
            name = 'CONFIG_' + name
        return name

    def __init__(self, mapping=None):
        mapping = mapping if mapping is not None else {}
        self._config = {
            # Ensure we use the canonical name of the config keys for internal
            # representation
            self.get_config_name(k): v
            for k, v in dict(mapping).items()
        }

    @classmethod
    def from_str(cls, text):
        """
        Build a :class:`TypedKernelConfig` out of the string content of a
        Kconfig file.
        """
        return cls(cls._parse_text(text))

    @staticmethod
    def _val_to_str(val):
        "Convert back values to Kconfig-style string value"
        # Special case the gracefully handle the output of get()
        if val is None:
            return None
        elif isinstance(val, KernelConfigTristate):
            return val.value
        elif isinstance(val, basestring):
            return '"{}"'.format(val.strip('"'))
        else:
            return str(val)

    def __str__(self):
        return '\n'.join(
            '{}={}'.format(k, self._val_to_str(v))
            for k, v in self.items()
        )

    @staticmethod
    def _parse_val(k, v):
        """
        Parse a value of types handled by Kconfig:
            * string
            * bool
            * tristate
            * hex
            * int

        Since bool cannot be distinguished from tristate, tristate is
        always used. :meth:`KernelConfigTristate.__bool__` will allow using
        it as a bool though, so it should not impact user code.
        """
        if not v:
            return None

        # Handle "string" type
        if v.startswith('"'):
            # Strip enclosing "
            return v[1:-1]

        else:
            try:
                # Handles "bool" and "tristate" types
                return KernelConfigTristate.from_str(v)
            except ValueError:
                pass

            try:
                # Handles "int" type
                return int(v)
            except ValueError:
                pass

            try:
                # Handles "hex" type
                return HexInt(v)
            except ValueError:
                pass

            # If no type could be parsed
            raise ValueError('Could not parse Kconfig key: {}={}'.format(
                    k, v
                ), k, v
            )

    @classmethod
    def _parse_text(cls, text):
        config = {}
        for line in text.splitlines():
            line = line.strip()

            # skip empty lines
            if not line:
                continue

            if line.startswith('#'):
                match = cls.not_set_regex.search(line)
                if match:
                    value = 'n'
                    name = match.group(1)
                else:
                    continue
            else:
                name, value = line.split('=', 1)

            name = cls.get_config_name(name.strip())
            value = cls._parse_val(name, value.strip())
            config[name] = value
        return config

    def __getitem__(self, name):
        name = self.get_config_name(name)
        try:
            return self._config[name]
        except KeyError:
            raise KernelConfigKeyError(
                "{} is not exposed in kernel config".format(name),
                name
            )

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

    def __contains__(self, name):
        name = self.get_config_name(name)
        return name in self._config

    def like(self, name):
        regex = re.compile(name, re.I)
        return {
            k: v for k, v in self.items()
            if regex.search(k)
        }

    def is_enabled(self, name):
        return self.get(name) is KernelConfigTristate.YES

    def is_module(self, name):
        return self.get(name) is KernelConfigTristate.MODULE

    def is_not_set(self, name):
        return self.get(name) is KernelConfigTristate.NO

    def has(self, name):
        return self.is_enabled(name) or self.is_module(name)


class KernelConfig(object):
    """
    Backward compatibility shim on top of :class:`TypedKernelConfig`.

    This class does not provide a Mapping API and only return string values.
    """
    @staticmethod
    def get_config_name(name):
        return TypedKernelConfig.get_config_name(name)

    def __init__(self, text):
        # Expose typed_config as a non-private attribute, so that user code
        # needing it can get it from any existing producer of KernelConfig.
        self.typed_config = TypedKernelConfig.from_str(text)
        # Expose the original text for backward compatibility
        self.text = text

    def __bool__(self):
        return bool(self.typed_config)

    not_set_regex = TypedKernelConfig.not_set_regex

    def iteritems(self):
        for k, v in self.typed_config.items():
            yield (k, self.typed_config._val_to_str(v))

    items = iteritems

    def get(self, name, strict=False):
        if strict:
            val = self.typed_config[name]
        else:
            val = self.typed_config.get(name)

        return self.typed_config._val_to_str(val)

    def like(self, name):
        return {
            k: self.typed_config._val_to_str(v)
            for k, v in self.typed_config.like(name).items()
        }

    def is_enabled(self, name):
        return self.typed_config.is_enabled(name)

    def is_module(self, name):
        return self.typed_config.is_module(name)

    def is_not_set(self, name):
        return self.typed_config.is_not_set(name)

    def has(self, name):
        return self.typed_config.has(name)


class LocalLinuxTarget(LinuxTarget):

    def __init__(self,
                 connection_settings=None,
                 platform=None,
                 working_directory=None,
                 executables_directory=None,
                 connect=True,
                 modules=None,
                 load_default_modules=True,
                 shell_prompt=DEFAULT_SHELL_PROMPT,
                 conn_cls=LocalConnection,
                 is_container=False,
                 ):
        super(LocalLinuxTarget, self).__init__(connection_settings=connection_settings,
                                               platform=platform,
                                               working_directory=working_directory,
                                               executables_directory=executables_directory,
                                               connect=connect,
                                               modules=modules,
                                               load_default_modules=load_default_modules,
                                               shell_prompt=shell_prompt,
                                               conn_cls=conn_cls,
                                               is_container=is_container)

    def _resolve_paths(self):
        if self.working_directory is None:
            self.working_directory = '/tmp/devlib-target'
        self._file_transfer_cache = self.path.join(self.working_directory, '.file-cache')
        if self.executables_directory is None:
            self.executables_directory = '/tmp/devlib-target/bin'


def _get_model_name(section):
    name_string = section['model name']
    parts = name_string.split('@')[0].strip().split()
    return ' '.join([p for p in parts
                     if '(' not in p and p != 'CPU'])


def _get_part_name(section):
    implementer = section.get('CPU implementer', '0x0')
    part = section['CPU part']
    variant = section.get('CPU variant', '0x0')
    name = get_cpu_name(*list(map(integer, [implementer, part, variant])))
    if name is None:
        name = '{}/{}/{}'.format(implementer, part, variant)
    return name


def _build_path_tree(path_map, basepath, sep=os.path.sep, dictcls=dict):
    """
    Convert a flat mapping of paths to values into a nested structure of
    dict-line object (``dict``'s by default), mirroring the directory hierarchy
    represented by the paths relative to ``basepath``.

    """
    def process_node(node, path, value):
        parts = path.split(sep, 1)
        if len(parts) == 1:   # leaf
            node[parts[0]] = value
        else:  # branch
            if parts[0] not in node:
                node[parts[0]] = dictcls()
            process_node(node[parts[0]], parts[1], value)

    relpath_map = {os.path.relpath(p, basepath): v
                   for p, v in path_map.items()}

    if len(relpath_map) == 1 and list(relpath_map.keys())[0] == '.':
        result = list(relpath_map.values())[0]
    else:
        result = dictcls()
        for path, value in relpath_map.items():
            process_node(result, path, value)

    return result


class ChromeOsTarget(LinuxTarget):

    os = 'chromeos'

    # pylint: disable=too-many-locals
    def __init__(self,
                 connection_settings=None,
                 platform=None,
                 working_directory=None,
                 executables_directory=None,
                 android_working_directory=None,
                 android_executables_directory=None,
                 connect=True,
                 modules=None,
                 load_default_modules=True,
                 shell_prompt=DEFAULT_SHELL_PROMPT,
                 package_data_directory="/data/data",
                 is_container=False
                 ):

        self.supports_android = None
        self.android_container = None

        # Pull out ssh connection settings
        ssh_conn_params = ['host', 'username', 'password', 'keyfile',
                           'port', 'timeout', 'sudo_cmd',
                           'strict_host_check', 'use_scp',
                           'total_timeout', 'poll_transfers',
                           'start_transfer_poll_delay']
        self.ssh_connection_settings = {}
        for setting in ssh_conn_params:
            if connection_settings.get(setting, None):
                self.ssh_connection_settings[setting] = connection_settings[setting]

        super(ChromeOsTarget, self).__init__(connection_settings=self.ssh_connection_settings,
                                             platform=platform,
                                             working_directory=working_directory,
                                             executables_directory=executables_directory,
                                             connect=False,
                                             modules=modules,
                                             load_default_modules=load_default_modules,
                                             shell_prompt=shell_prompt,
                                             conn_cls=SshConnection,
                                             is_container=is_container)

        # We can't determine if the target supports android until connected to the linux host so
        # create unconditionally.
        # Pull out adb connection settings
        adb_conn_params = ['device', 'adb_server', 'timeout']
        self.android_connection_settings = {}
        for setting in adb_conn_params:
            if connection_settings.get(setting, None):
                self.android_connection_settings[setting] = connection_settings[setting]

        # If adb device is not explicitly specified use same as ssh host
        if not connection_settings.get('device', None):
            self.android_connection_settings['device'] = connection_settings.get('host', None)

        self.android_container = AndroidTarget(connection_settings=self.android_connection_settings,
                                               platform=platform,
                                               working_directory=android_working_directory,
                                               executables_directory=android_executables_directory,
                                               connect=False,
                                               modules=[], # Only use modules with linux target
                                               load_default_modules=False,
                                               shell_prompt=shell_prompt,
                                               conn_cls=AdbConnection,
                                               package_data_directory=package_data_directory,
                                               is_container=True)
        if connect:
            self.connect()

    def __getattr__(self, attr):
        """
        By default use the linux target methods and attributes however,
        if not present, use android implementation if available.
        """
        try:
            return super(ChromeOsTarget, self).__getattribute__(attr)
        except AttributeError:
            if hasattr(self.android_container, attr):
                return getattr(self.android_container, attr)
            else:
                raise

    def connect(self, timeout=30, check_boot_completed=True):
        super(ChromeOsTarget, self).connect(timeout, check_boot_completed)

        # Assume device supports android apps if container directory is present
        if self.supports_android is None:
            self.supports_android = self.directory_exists('/opt/google/containers/android/')

        if self.supports_android:
            self.android_container.connect(timeout)
        else:
            self.android_container = None

    def _resolve_paths(self):
        if self.working_directory is None:
            self.working_directory = '/mnt/stateful_partition/devlib-target'
        self._file_transfer_cache = self.path.join(self.working_directory, '.file-cache')
        if self.executables_directory is None:
            self.executables_directory = self.path.join(self.working_directory, 'bin')
