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

from wlauto.exceptions import DeviceError, ConfigError, HostError
from wlauto.utils.misc import check_output, escape_single_quotes, escape_double_quotes, get_null


MAX_TRIES = 5

logger = logging.getLogger('android')

# See:
# http://developer.android.com/guide/topics/manifest/uses-sdk-element.html#ApiLevels
ANDROID_VERSION_MAP = {
    22: 'LOLLIPOP_MR1',
    21: 'LOLLIPOP',
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

# TODO: these are set to their actual values near the bottom of the file. There
# is some HACKery  involved to ensure that ANDROID_HOME does not need to be set
# or adb added to path for root when installing as root, and the whole
# implemenationt is kinda clunky and messier than I'd like. The only file that
# rivals this one in levels of mess is bootstrap.py (for very much the same
# reasons). There must be a neater way to ensure that enviromental dependencies
# are met when they are needed, and are not imposed when they are not.
android_home = None
platform_tools = None
adb = None
aapt = None
fastboot = None


class _AndroidEnvironment(object):

    def __init__(self):
        self.android_home = None
        self.platform_tools = None
        self.adb = None
        self.aapt = None
        self.fastboot = None


class AndroidProperties(object):

    def __init__(self, text):
        self._properties = {}
        self.parse(text)

    def parse(self, text):
        self._properties = dict(re.findall(r'\[(.*?)\]:\s+\[(.*?)\]', text))

    def __iter__(self):
        return iter(self._properties)

    def __getattr__(self, name):
        return self._properties.get(name)

    __getitem__ = __getattr__


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


def fastboot_command(command, timeout=None):
    _check_env()
    full_command = "fastboot {}".format(command)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


def fastboot_flash_partition(partition, path_to_image):
    command = 'flash {} {}'.format(partition, path_to_image)
    fastboot_command(command)


def adb_get_device():
    """
    Returns the serial number of a connected android device.

    If there are more than one device connected to the machine, or it could not
    find any device connected, :class:`wlauto.exceptions.ConfigError` is raised.
    """
    _check_env()
    # TODO this is a hacky way to issue a adb command to all listed devices

    # The output of calling adb devices consists of a heading line then
    # a list of the devices sperated by new line
    # The last line is a blank new line. in otherwords, if there is a device found
    # then the output length is 2 + (1 for each device)
    output = adb_command('0', "devices").splitlines()  # pylint: disable=E1103
    output_length = len(output)
    if output_length == 3:
        # output[1] is the 2nd line in the output which has the device name
        # Splitting the line by '\t' gives a list of two indexes, which has
        # device serial in 0 number and device type in 1.
        return output[1].split('\t')[0]
    elif output_length > 3:
        raise ConfigError('Number of discovered devices is {}, it should be 1'.format(output_length - 2))
    else:
        raise ConfigError('No device is connected and available')


def adb_connect(device, timeout=None):
    _check_env()
    command = "adb connect " + device
    if ":" in device:
        port = device.split(':')[-1]
        logger.debug(command)

        output, _ = check_output(command, shell=True, timeout=timeout)
        logger.debug(output)
        #### due to a rare adb bug sometimes an extra :5555 is appended to the IP address
        if output.find('{}:{}'.format(port, port)) != -1:
            logger.debug('ADB BUG with extra port')
            command = "adb connect " + device.replace(':{}'.format(port), '')

    tries = 0
    output = None
    while not poll_for_file(device, "/proc/cpuinfo"):
        logger.debug("adb connect failed, retrying now...")
        tries += 1
        if tries > MAX_TRIES:
            raise DeviceError('Cannot connect to adb server on the device.')
        logger.debug(command)
        output, _ = check_output(command, shell=True, timeout=timeout)
        time.sleep(10)

    if tries and output.find('connected to') == -1:
        raise DeviceError('Could not connect to {}'.format(device))


def adb_disconnect(device):
    _check_env()
    if ":5555" in device:
        command = "adb disconnect " + device
        logger.debug(command)
        retval = subprocess.call(command, stdout=open(os.devnull, 'wb'), shell=True)
        if retval:
            raise DeviceError('"{}" returned {}'.format(command, retval))


def poll_for_file(device, dfile):
    _check_env()
    device_string = '-s {}'.format(device) if device else ''
    command = "adb " + device_string + " shell \" if [ -f " + dfile + " ] ; then true ; else false ; fi\" "
    logger.debug(command)
    result = subprocess.call(command, stderr=subprocess.PIPE, shell=True)
    if not result:
        return True
    else:
        return False


am_start_error = re.compile(r"Error: Activity class {[\w|.|/]*} does not exist")


def adb_shell(device, command, timeout=None, check_exit_code=False, as_root=False):  # NOQA
    _check_env()
    if as_root:
        command = 'echo "{}" | su'.format(escape_double_quotes(command))
    device_string = '-s {}'.format(device) if device else ''
    full_command = 'adb {} shell "{}"'.format(device_string, escape_double_quotes(command))
    logger.debug(full_command)
    if check_exit_code:
        actual_command = "adb {} shell '({}); echo; echo $?'".format(device_string, escape_single_quotes(command))
        raw_output, error = check_output(actual_command, timeout, shell=True)
        if raw_output:
            try:
                output, exit_code, _ = raw_output.rsplit('\r\n', 2)
            except ValueError:
                exit_code, _ = raw_output.rsplit('\r\n', 1)
                output = ''
        else:  # raw_output is empty
            exit_code = '969696'  # just because
            output = ''

        exit_code = exit_code.strip()
        if exit_code.isdigit():
            if int(exit_code):
                message = 'Got exit code {}\nfrom: {}\nSTDOUT: {}\nSTDERR: {}'.format(exit_code, full_command,
                                                                                      output, error)
                raise DeviceError(message)
            elif am_start_error.findall(output):
                message = 'Could not start activity; got the following:'
                message += '\n{}'.format(am_start_error.findall(output)[0])
                raise DeviceError(message)
        else:  # not all digits
            if am_start_error.findall(output):
                message = 'Could not start activity; got the following:'
                message += '\n{}'.format(am_start_error.findall(output)[0])
                raise DeviceError(message)
            else:
                raise DeviceError('adb has returned early; did not get an exit code. Was kill-server invoked?')
    else:  # do not check exit code
        output, _ = check_output(full_command, timeout, shell=True)
    return output


def adb_background_shell(device, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
    """Runs the sepcified command in a subprocess, returning the the Popen object."""
    _check_env()
    if as_root:
        command = 'echo \'{}\' | su'.format(escape_single_quotes(command))
    device_string = '-s {}'.format(device) if device else ''
    full_command = 'adb {} shell "{}"'.format(device_string, escape_double_quotes(command))
    logger.debug(full_command)
    return subprocess.Popen(full_command, stdout=stdout, stderr=stderr, shell=True)


class AdbDevice(object):

    def __init__(self, name, status):
        self.name = name
        self.status = status

    def __cmp__(self, other):
        if isinstance(other, AdbDevice):
            return cmp(self.name, other.name)
        else:
            return cmp(self.name, other)


def adb_list_devices():
    _check_env()
    output = adb_command(None, 'devices')
    devices = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split()]
        if len(parts) == 2:
            devices.append(AdbDevice(*parts))
    return devices


def adb_command(device, command, timeout=None):
    _check_env()
    device_string = '-s {}'.format(device) if device else ''
    full_command = "adb {} {}".format(device_string, command)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


# Messy environment initialisation stuff...


def _initialize_with_android_home(env):
    logger.debug('Using ANDROID_HOME from the environment.')
    env.android_home = android_home
    env.platform_tools = os.path.join(android_home, 'platform-tools')
    os.environ['PATH'] += os.pathsep + env.platform_tools
    _init_common(env)
    return env


def _initialize_without_android_home(env):
    if os.name == 'nt':
        raise HostError('Please set ANDROID_HOME to point to the location of the Android SDK.')
    # Assuming Unix in what follows.
    if subprocess.call('adb version >{}'.format(get_null()), shell=True):
        raise HostError('ANDROID_HOME is not set and adb is not in PATH. Have you installed Android SDK?')
    logger.debug('Discovering ANDROID_HOME from adb path.')
    env.platform_tools = os.path.dirname(subprocess.check_output('which adb', shell=True))
    env.android_home = os.path.dirname(env.platform_tools)
    _init_common(env)
    return env


def _init_common(env):
    logger.debug('ANDROID_HOME: {}'.format(env.android_home))
    build_tools_directory = os.path.join(env.android_home, 'build-tools')
    if not os.path.isdir(build_tools_directory):
        msg = 'ANDROID_HOME ({}) does not appear to have valid Android SDK install (cannot find build-tools)'
        raise HostError(msg.format(env.android_home))
    versions = os.listdir(build_tools_directory)
    for version in reversed(sorted(versions)):
        aapt_path = os.path.join(build_tools_directory, version, 'aapt')
        if os.path.isfile(aapt_path):
            logger.debug('Using aapt for version {}'.format(version))
            env.aapt = aapt_path
            break
    else:
        raise HostError('aapt not found. Please make sure at least one Android platform is installed.')


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
