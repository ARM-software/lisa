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
Utility functions for working with Android devices through adb.

"""
# pylint: disable=E1103
import os
import time
import subprocess
import logging
import re
from collections import defaultdict

from devlib.exception import TargetError, HostError, DevlibError
from devlib.utils.misc import check_output, which, memoized
from devlib.utils.misc import escape_single_quotes, escape_double_quotes


logger = logging.getLogger('android')

MAX_ATTEMPTS = 5
AM_START_ERROR = re.compile(r"Error: Activity class {[\w|.|/]*} does not exist")

# See:
# http://developer.android.com/guide/topics/manifest/uses-sdk-element.html#ApiLevels
ANDROID_VERSION_MAP = {
    23: 'MARSHMALLOW',
    22: 'LOLLYPOP_MR1',
    21: 'LOLLYPOP',
    20: 'KITKAT_WATCH',
    19: 'KITKAT',
    18: 'JELLY_BEAN_MR2',
    17: 'JELLY_BEAN_MR1',
    16: 'JELLY_BEAN',
    15: 'ICE_CREAM_SANDWICH_MR1',
    14: 'ICE_CREAM_SANDWICH',
    13: 'HONEYCOMB_MR2',
    12: 'HONEYCOMB_MR1',
    11: 'HONEYCOMB',
    10: 'GINGERBREAD_MR1',
    9: 'GINGERBREAD',
    8: 'FROYO',
    7: 'ECLAIR_MR1',
    6: 'ECLAIR_0_1',
    5: 'ECLAIR',
    4: 'DONUT',
    3: 'CUPCAKE',
    2: 'BASE_1_1',
    1: 'BASE',
}


# Initialized in functions near the botton of the file
android_home = None
platform_tools = None
adb = None
aapt = None
fastboot = None


class AndroidProperties(object):

    def __init__(self, text):
        self._properties = {}
        self.parse(text)

    def parse(self, text):
        self._properties = dict(re.findall(r'\[(.*?)\]:\s+\[(.*?)\]', text))

    def iteritems(self):
        return self._properties.iteritems()

    def __iter__(self):
        return iter(self._properties)

    def __getattr__(self, name):
        return self._properties.get(name)

    __getitem__ = __getattr__


class AdbDevice(object):

    def __init__(self, name, status):
        self.name = name
        self.status = status

    def __cmp__(self, other):
        if isinstance(other, AdbDevice):
            return cmp(self.name, other.name)
        else:
            return cmp(self.name, other)

    def __str__(self):
        return 'AdbDevice({}, {})'.format(self.name, self.status)

    __repr__ = __str__


class ApkInfo(object):

    version_regex = re.compile(r"name='(?P<name>[^']+)' versionCode='(?P<vcode>[^']+)' versionName='(?P<vname>[^']+)'")
    name_regex = re.compile(r"name='(?P<name>[^']+)'")

    def __init__(self, path=None):
        self.path = path
        self.package = None
        self.activity = None
        self.label = None
        self.version_name = None
        self.version_code = None
        self.parse(path)

    def parse(self, apk_path):
        _check_env()
        command = [aapt, 'dump', 'badging', apk_path]
        logger.debug(' '.join(command))
        output = subprocess.check_output(command)
        for line in output.split('\n'):
            if line.startswith('application-label:'):
                self.label = line.split(':')[1].strip().replace('\'', '')
            elif line.startswith('package:'):
                match = self.version_regex.search(line)
                if match:
                    self.package = match.group('name')
                    self.version_code = match.group('vcode')
                    self.version_name = match.group('vname')
            elif line.startswith('launchable-activity:'):
                match = self.name_regex.search(line)
                self.activity = match.group('name')
            else:
                pass  # not interested


class AdbConnection(object):

    # maintains the count of parallel active connections to a device, so that
    # adb disconnect is not invoked untill all connections are closed
    active_connections = defaultdict(int)
    default_timeout = 10
    ls_command = 'ls'

    @property
    def name(self):
        return self.device

    @property
    @memoized
    def newline_separator(self):
        output = adb_command(self.device,
                             "shell '({}); echo \"\n$?\"'".format(self.ls_command))
        if output.endswith('\r\n'):
            return '\r\n'
        elif output.endswith('\n'):
            return '\n'
        else:
            raise DevlibError("Unknown line ending")

    # Again, we need to handle boards where the default output format from ls is
    # single column *and* boards where the default output is multi-column.
    # We need to do this purely because the '-1' option causes errors on older
    # versions of the ls tool in Android pre-v7.
    def _setup_ls(self):
        command = "shell '(ls -1); echo \"\n$?\"'"
        output = adb_command(self.device, command, timeout=self.timeout)
        lines = output.splitlines()
        retval = lines[-1].strip()
        if int(retval) == 0:
            self.ls_command = 'ls -1'
        else:
            self.ls_command = 'ls'
        logger.info("ls command is set to {}".format(self.ls_command))

    def __init__(self, device=None, timeout=None):
        self.timeout = timeout if timeout is not None else self.default_timeout
        if device is None:
            device = adb_get_device(timeout=timeout)
        self.device = device
        adb_connect(self.device)
        AdbConnection.active_connections[self.device] += 1
        self._setup_ls()

    def push(self, source, dest, timeout=None):
        if timeout is None:
            timeout = self.timeout
        command = "push '{}' '{}'".format(source, dest)
        return adb_command(self.device, command, timeout=timeout)

    def pull(self, source, dest, timeout=None):
        if timeout is None:
            timeout = self.timeout
        # Pull all files matching a wildcard expression
        if os.path.isdir(dest) and \
           ('*' in source or '?' in source):
            command = 'shell {} {}'.format(self.ls_command, source)
            output = adb_command(self.device, command, timeout=timeout)
            for line in output.splitlines():
                command = "pull '{}' '{}'".format(line, dest)
                adb_command(self.device, command, timeout=timeout)
            return
        command = "pull '{}' '{}'".format(source, dest)
        return adb_command(self.device, command, timeout=timeout)

    def execute(self, command, timeout=None, check_exit_code=False, as_root=False):
        return adb_shell(self.device, command, timeout, check_exit_code,
                         as_root, self.newline_separator)

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        return adb_background_shell(self.device, command, stdout, stderr, as_root)

    def close(self):
        AdbConnection.active_connections[self.device] -= 1
        if AdbConnection.active_connections[self.device] <= 0:
            adb_disconnect(self.device)
            del AdbConnection.active_connections[self.device]

    def cancel_running_command(self):
        # adbd multiplexes commands so that they don't interfer with each
        # other, so there is no need to explicitly cancel a running command
        # before the next one can be issued.
        pass


def fastboot_command(command, timeout=None):
    _check_env()
    full_command = "fastboot {}".format(command)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


def fastboot_flash_partition(partition, path_to_image):
    command = 'flash {} {}'.format(partition, path_to_image)
    fastboot_command(command)


def adb_get_device(timeout=None):
    """
    Returns the serial number of a connected android device.

    If there are more than one device connected to the machine, or it could not
    find any device connected, :class:`devlib.exceptions.HostError` is raised.
    """
    # TODO this is a hacky way to issue a adb command to all listed devices

    # The output of calling adb devices consists of a heading line then
    # a list of the devices sperated by new line
    # The last line is a blank new line. in otherwords, if there is a device found
    # then the output length is 2 + (1 for each device)
    start = time.time()
    while True:
        output = adb_command(None, "devices").splitlines()  # pylint: disable=E1103
        output_length = len(output)
        if output_length == 3:
            # output[1] is the 2nd line in the output which has the device name
            # Splitting the line by '\t' gives a list of two indexes, which has
            # device serial in 0 number and device type in 1.
            return output[1].split('\t')[0]
        elif output_length > 3:
            message = '{} Android devices found; either explicitly specify ' +\
                      'the device you want, or make sure only one is connected.'
            raise HostError(message.format(output_length - 2))
        else:
            if timeout < time.time() - start:
                raise HostError('No device is connected and available')
            time.sleep(1)


def adb_connect(device, timeout=None, attempts=MAX_ATTEMPTS):
    _check_env()
    # Connect is required only for ADB-over-IP
    if "." not in device:
        logger.debug('Device connected via USB, connect not required')
        return
    tries = 0
    output = None
    while tries <= attempts:
        tries += 1
        if device:
            command = 'adb connect {}'.format(device)
            logger.debug(command)
            output, _ = check_output(command, shell=True, timeout=timeout)
        if _ping(device):
            break
        time.sleep(10)
    else:  # did not connect to the device
        message = 'Could not connect to {}'.format(device or 'a device')
        if output:
            message += '; got: "{}"'.format(output)
        raise HostError(message)


def adb_disconnect(device):
    _check_env()
    if not device:
        return
    if ":" in device and device in adb_list_devices():
        command = "adb disconnect " + device
        logger.debug(command)
        retval = subprocess.call(command, stdout=open(os.devnull, 'wb'), shell=True)
        if retval:
            raise TargetError('"{}" returned {}'.format(command, retval))


def _ping(device):
    _check_env()
    device_string = ' -s {}'.format(device) if device else ''
    command = "adb{} shell \"ls / > /dev/null\"".format(device_string)
    logger.debug(command)
    result = subprocess.call(command, stderr=subprocess.PIPE, shell=True)
    if not result:
        return True
    else:
        return False


def adb_shell(device, command, timeout=None, check_exit_code=False,
              as_root=False, newline_separator='\r\n'):  # NOQA
    _check_env()
    if as_root:
        command = 'echo \'{}\' | su'.format(escape_single_quotes(command))
    device_part = ['-s', device] if device else []
    device_string = ' {} {}'.format(*device_part) if device_part else ''
    full_command = 'adb{} shell "{}"'.format(device_string,
                                             escape_double_quotes(command))
    logger.debug(full_command)
    if check_exit_code:
        adb_shell_command = '({}); echo \"\n$?\"'.format(command)
        actual_command = ['adb'] + device_part + ['shell', adb_shell_command]
        raw_output, error = check_output(actual_command, timeout, shell=False)
        if raw_output:
            try:
                output, exit_code, _ = raw_output.rsplit(newline_separator, 2)
            except ValueError:
                exit_code, _ = raw_output.rsplit(newline_separator, 1)
                output = ''
        else:  # raw_output is empty
            exit_code = '969696'  # just because
            output = ''

        exit_code = exit_code.strip()
        if exit_code.isdigit():
            if int(exit_code):
                message = 'Got exit code {}\nfrom: {}\nSTDOUT: {}\nSTDERR: {}'
                raise TargetError(message.format(exit_code, full_command, output, error))
            elif AM_START_ERROR.findall(output):
                message = 'Could not start activity; got the following:'
                message += '\n{}'.format(AM_START_ERROR.findall(output)[0])
                raise TargetError(message)
        else:  # not all digits
            if AM_START_ERROR.findall(output):
                message = 'Could not start activity; got the following:\n{}'
                raise TargetError(message.format(AM_START_ERROR.findall(output)[0]))
            else:
                message = 'adb has returned early; did not get an exit code. '\
                          'Was kill-server invoked?'
                raise TargetError(message)
    else:  # do not check exit code
        output, _ = check_output(full_command, timeout, shell=True)
    return output


def adb_background_shell(device, command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         as_root=False):
    """Runs the sepcified command in a subprocess, returning the the Popen object."""
    _check_env()
    if as_root:
        command = 'echo \'{}\' | su'.format(escape_single_quotes(command))
    device_string = ' -s {}'.format(device) if device else ''
    full_command = 'adb{} shell "{}"'.format(device_string, escape_double_quotes(command))
    logger.debug(full_command)
    return subprocess.Popen(full_command, stdout=stdout, stderr=stderr, shell=True)


def adb_list_devices():
    output = adb_command(None, 'devices')
    devices = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split()]
        if len(parts) == 2:
            devices.append(AdbDevice(*parts))
    return devices


def adb_command(device, command, timeout=None):
    _check_env()
    device_string = ' -s {}'.format(device) if device else ''
    full_command = "adb{} {}".format(device_string, command)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


# Messy environment initialisation stuff...

class _AndroidEnvironment(object):

    def __init__(self):
        self.android_home = None
        self.platform_tools = None
        self.adb = None
        self.aapt = None
        self.fastboot = None


def _initialize_with_android_home(env):
    logger.debug('Using ANDROID_HOME from the environment.')
    env.android_home = android_home
    env.platform_tools = os.path.join(android_home, 'platform-tools')
    os.environ['PATH'] += os.pathsep + env.platform_tools
    _init_common(env)
    return env


def _initialize_without_android_home(env):
    adb_full_path = which('adb')
    if adb_full_path:
        env.adb = 'adb'
    else:
        raise HostError('ANDROID_HOME is not set and adb is not in PATH. '
                        'Have you installed Android SDK?')
    logger.debug('Discovering ANDROID_HOME from adb path.')
    env.platform_tools = os.path.dirname(adb_full_path)
    env.android_home = os.path.dirname(env.platform_tools)
    _init_common(env)
    return env


def _init_common(env):
    logger.debug('ANDROID_HOME: {}'.format(env.android_home))
    build_tools_directory = os.path.join(env.android_home, 'build-tools')
    if not os.path.isdir(build_tools_directory):
        msg = '''ANDROID_HOME ({}) does not appear to have valid Android SDK install
                 (cannot find build-tools)'''
        raise HostError(msg.format(env.android_home))
    versions = os.listdir(build_tools_directory)
    for version in reversed(sorted(versions)):
        aapt_path = os.path.join(build_tools_directory, version, 'aapt')
        if os.path.isfile(aapt_path):
            logger.debug('Using aapt for version {}'.format(version))
            env.aapt = aapt_path
            break
    else:
        raise HostError('aapt not found. Please make sure at least one Android '
                        'platform is installed.')


def _check_env():
    global android_home, platform_tools, adb, aapt  # pylint: disable=W0603
    if not android_home:
        android_home = os.getenv('ANDROID_HOME')
        if android_home:
            _env = _initialize_with_android_home(_AndroidEnvironment())
        else:
            _env = _initialize_without_android_home(_AndroidEnvironment())
        android_home = _env.android_home
        platform_tools = _env.platform_tools
        adb = _env.adb
        aapt = _env.aapt
