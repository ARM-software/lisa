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
Utility functions for working with Android devices through adb.

"""
# pylint: disable=E1103
import os
import re
import sys
import time
import logging
import tempfile
import subprocess
from collections import defaultdict
import pexpect
import xml.etree.ElementTree
import zipfile

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from devlib.exception import TargetTransientError, TargetStableError, HostError
from devlib.utils.misc import check_output, which, ABI_MAP


logger = logging.getLogger('android')

MAX_ATTEMPTS = 5
AM_START_ERROR = re.compile(r"Error: Activity.*")

# See:
# http://developer.android.com/guide/topics/manifest/uses-sdk-element.html#ApiLevels
ANDROID_VERSION_MAP = {
    29: 'Q',
    28: 'PIE',
    27: 'OREO_MR1',
    26: 'OREO',
    25: 'NOUGAT_MR1',
    24: 'NOUGAT',
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

# See https://developer.android.com/reference/android/content/Intent.html#setFlags(int)
INTENT_FLAGS = {
    'ACTIVITY_NEW_TASK' : 0x10000000,
    'ACTIVITY_CLEAR_TASK' : 0x00008000
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
        return iter(self._properties.items())

    def __iter__(self):
        return iter(self._properties)

    def __getattr__(self, name):
        return self._properties.get(name)

    __getitem__ = __getattr__


class AdbDevice(object):

    def __init__(self, name, status):
        self.name = name
        self.status = status

    # pylint: disable=undefined-variable
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
    permission_regex = re.compile(r"name='(?P<permission>[^']+)'")
    activity_regex = re.compile(r'\s*A:\s*android:name\(0x\d+\)=".(?P<name>\w+)"')

    def __init__(self, path=None):
        self.path = path
        self.package = None
        self.activity = None
        self.label = None
        self.version_name = None
        self.version_code = None
        self.native_code = None
        self.permissions = []
        self.parse(path)

    # pylint: disable=too-many-branches
    def parse(self, apk_path):
        _check_env()
        output = self._run([aapt, 'dump', 'badging', apk_path])
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
            elif line.startswith('native-code'):
                apk_abis = [entry.strip() for entry in line.split(':')[1].split("'") if entry.strip()]
                mapped_abis = []
                for apk_abi in apk_abis:
                    found = False
                    for abi, architectures in ABI_MAP.items():
                        if apk_abi in architectures:
                            mapped_abis.append(abi)
                            found = True
                            break
                    if not found:
                        mapped_abis.append(apk_abi)
                self.native_code = mapped_abis
            elif line.startswith('uses-permission:'):
                match = self.permission_regex.search(line)
                if match:
                    self.permissions.append(match.group('permission'))
            else:
                pass  # not interested

        self._apk_path = apk_path
        self._activities = None
        self._methods = None

    @property
    def activities(self):
        if self._activities is None:
            cmd = [aapt, 'dump', 'xmltree', self._apk_path,
                   'AndroidManifest.xml']
            matched_activities = self.activity_regex.finditer(self._run(cmd))
            self._activities = [m.group('name') for m in matched_activities]
        return self._activities

    @property
    def methods(self):
        if self._methods is None:
            with zipfile.ZipFile(self._apk_path, 'r') as z:
                extracted = z.extract('classes.dex', tempfile.gettempdir())

            dexdump = os.path.join(os.path.dirname(aapt), 'dexdump')
            command = [dexdump, '-l', 'xml', extracted]
            dump = self._run(command)

            xml_tree = xml.etree.ElementTree.fromstring(dump)

            package = next(i for i in xml_tree.iter('package')
                           if i.attrib['name'] == self.package)

            self._methods = [(meth.attrib['name'], klass.attrib['name'])
                             for klass in package.iter('class')
                             for meth in klass.iter('method')]
        return self._methods

    def _run(self, command):
        logger.debug(' '.join(command))
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
            if sys.version_info[0] == 3:
                output = output.decode(sys.stdout.encoding or 'utf-8', 'replace')
        except subprocess.CalledProcessError as e:
            raise HostError('Error while running "{}":\n{}'
                            .format(command, e.output))
        return output


class AdbConnection(object):

    # maintains the count of parallel active connections to a device, so that
    # adb disconnect is not invoked untill all connections are closed
    active_connections = defaultdict(int)
    # Track connected as root status per device
    _connected_as_root = defaultdict(lambda: None)
    default_timeout = 10
    ls_command = 'ls'
    su_cmd = 'su -c {}'

    @property
    def name(self):
        return self.device

    @property
    def connected_as_root(self):
        if self._connected_as_root[self.device] is None:
                result = self.execute('id')
                self._connected_as_root[self.device] = 'uid=0(' in result
        return self._connected_as_root[self.device]

    @connected_as_root.setter
    def connected_as_root(self, state):
        self._connected_as_root[self.device] = state


    # pylint: disable=unused-argument
    def __init__(self, device=None, timeout=None, platform=None, adb_server=None,
                 adb_as_root=False):
        self.timeout = timeout if timeout is not None else self.default_timeout
        if device is None:
            device = adb_get_device(timeout=timeout, adb_server=adb_server)
        self.device = device
        self.adb_server = adb_server
        self.adb_as_root = adb_as_root
        if self.adb_as_root:
            self.adb_root(enable=True)
        adb_connect(self.device, adb_server=self.adb_server)
        AdbConnection.active_connections[self.device] += 1
        self._setup_ls()
        self._setup_su()

    def push(self, source, dest, timeout=None):
        if timeout is None:
            timeout = self.timeout
        command = "push {} {}".format(quote(source), quote(dest))
        if not os.path.exists(source):
            raise HostError('No such file "{}"'.format(source))
        return adb_command(self.device, command, timeout=timeout, adb_server=self.adb_server)

    def pull(self, source, dest, timeout=None):
        if timeout is None:
            timeout = self.timeout
        # Pull all files matching a wildcard expression
        if os.path.isdir(dest) and \
           ('*' in source or '?' in source):
            command = 'shell {} {}'.format(self.ls_command, source)
            output = adb_command(self.device, command, timeout=timeout, adb_server=self.adb_server)
            for line in output.splitlines():
                command = "pull {} {}".format(quote(line.strip()), quote(dest))
                adb_command(self.device, command, timeout=timeout, adb_server=self.adb_server)
            return
        command = "pull {} {}".format(quote(source), quote(dest))
        return adb_command(self.device, command, timeout=timeout, adb_server=self.adb_server)

    # pylint: disable=unused-argument
    def execute(self, command, timeout=None, check_exit_code=False,
                as_root=False, strip_colors=True, will_succeed=False):
        try:
            return adb_shell(self.device, command, timeout, check_exit_code,
                             as_root, adb_server=self.adb_server, su_cmd=self.su_cmd)
        except TargetStableError as e:
            if will_succeed:
                raise TargetTransientError(e)
            else:
                raise

    def background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False):
        return adb_background_shell(self.device, command, stdout, stderr, as_root, adb_server=self.adb_server)

    def close(self):
        AdbConnection.active_connections[self.device] -= 1
        if AdbConnection.active_connections[self.device] <= 0:
            if self.adb_as_root:
                self.adb_root(self.device, enable=False)
            adb_disconnect(self.device, self.adb_server)
            del AdbConnection.active_connections[self.device]

    def cancel_running_command(self):
        # adbd multiplexes commands so that they don't interfer with each
        # other, so there is no need to explicitly cancel a running command
        # before the next one can be issued.
        pass

    def adb_root(self, enable=True):
        cmd = 'root' if enable else 'unroot'
        output = adb_command(self.device, cmd, timeout=30, adb_server=self.adb_server)
        if 'cannot run as root in production builds' in output:
            raise TargetStableError(output)
        AdbConnection._connected_as_root[self.device] = enable

    def wait_for_device(self, timeout=30):
        adb_command(self.device, 'wait-for-device', timeout, self.adb_server)

    def reboot_bootloader(self, timeout=30):
        adb_command(self.device, 'reboot-bootloader', timeout, self.adb_server)

    # Again, we need to handle boards where the default output format from ls is
    # single column *and* boards where the default output is multi-column.
    # We need to do this purely because the '-1' option causes errors on older
    # versions of the ls tool in Android pre-v7.
    def _setup_ls(self):
        command = "shell '(ls -1); echo \"\n$?\"'"
        try:
            output = adb_command(self.device, command, timeout=self.timeout, adb_server=self.adb_server)
        except subprocess.CalledProcessError as e:
            raise HostError(
                'Failed to set up ls command on Android device. Output:\n'
                + e.output)
        lines = output.splitlines()
        retval = lines[-1].strip()
        if int(retval) == 0:
            self.ls_command = 'ls -1'
        else:
            self.ls_command = 'ls'
        logger.debug("ls command is set to {}".format(self.ls_command))

    def _setup_su(self):
        try:
            # Try the new style of invoking `su`
            self.execute('ls', timeout=self.timeout, as_root=True,
                         check_exit_code=True)
        # If failure assume either old style or unrooted. Here we will assume
        # old style and root status will be verified later.
        except (TargetStableError, TargetTransientError, TimeoutError):
            self.su_cmd = 'echo {} | su'
        logger.debug("su command is set to {}".format(quote(self.su_cmd)))


def fastboot_command(command, timeout=None, device=None):
    _check_env()
    target = '-s {}'.format(quote(device)) if device else ''
    full_command = 'fastboot {} {}'.format(target, command)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output


def fastboot_flash_partition(partition, path_to_image):
    command = 'flash {} {}'.format(quote(partition), quote(path_to_image))
    fastboot_command(command)


def adb_get_device(timeout=None, adb_server=None):
    """
    Returns the serial number of a connected android device.

    If there are more than one device connected to the machine, or it could not
    find any device connected, :class:`devlib.exceptions.HostError` is raised.
    """
    # TODO this is a hacky way to issue a adb command to all listed devices

    # Ensure server is started so the 'daemon started successfully' message
    # doesn't confuse the parsing below
    adb_command(None, 'start-server', adb_server=adb_server)

    # The output of calling adb devices consists of a heading line then
    # a list of the devices sperated by new line
    # The last line is a blank new line. in otherwords, if there is a device found
    # then the output length is 2 + (1 for each device)
    start = time.time()
    while True:
        output = adb_command(None, "devices", adb_server=adb_server).splitlines()  # pylint: disable=E1103
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


def adb_connect(device, timeout=None, attempts=MAX_ATTEMPTS, adb_server=None):
    _check_env()
    tries = 0
    output = None
    while tries <= attempts:
        tries += 1
        if device:
            if "." in device: # Connect is required only for ADB-over-IP
                # ADB does not automatically remove a network device from it's
                # devices list when the connection is broken by the remote, so the
                # adb connection may have gone "stale", resulting in adb blocking
                # indefinitely when making calls to the device. To avoid this,
                # always disconnect first.
                adb_disconnect(device, adb_server)
                adb_cmd = get_adb_command(None, 'connect', adb_server)
                command = '{} {}'.format(adb_cmd, quote(device))
                logger.debug(command)
                output, _ = check_output(command, shell=True, timeout=timeout)
        if _ping(device, adb_server):
            break
        time.sleep(10)
    else:  # did not connect to the device
        message = 'Could not connect to {}'.format(device or 'a device')
        if output:
            message += '; got: "{}"'.format(output)
        raise HostError(message)


def adb_disconnect(device, adb_server=None):
    _check_env()
    if not device:
        return
    if ":" in device and device in adb_list_devices(adb_server):
        adb_cmd = get_adb_command(None, 'disconnect', adb_server)
        command = "{} {}".format(adb_cmd, device)
        logger.debug(command)
        retval = subprocess.call(command, stdout=open(os.devnull, 'wb'), shell=True)
        if retval:
            raise TargetTransientError('"{}" returned {}'.format(command, retval))


def _ping(device, adb_server=None):
    _check_env()
    adb_cmd = get_adb_command(device, 'shell', adb_server)
    command = "{} {}".format(adb_cmd, quote('ls /data/local/tmp > /dev/null'))
    logger.debug(command)
    result = subprocess.call(command, stderr=subprocess.PIPE, shell=True)
    if not result:  # pylint: disable=simplifiable-if-statement
        return True
    else:
        return False


# pylint: disable=too-many-locals
def adb_shell(device, command, timeout=None, check_exit_code=False,
              as_root=False, adb_server=None, su_cmd='su -c {}'):  # NOQA
    _check_env()

    # On older combinations of ADB/Android versions, the adb host command always
    # exits with 0 if it was able to run the command on the target, even if the
    # command failed (https://code.google.com/p/android/issues/detail?id=3254).
    # Homogenise this behaviour by running the command then echoing the exit
    # code of the executed command itself.
    command = r'({}); echo "\n$?"'.format(command)

    parts = ['adb']
    if adb_server is not None:
        parts += ['-H', adb_server]
    if device is not None:
        parts += ['-s', device]
    parts += ['shell',
              command if not as_root else su_cmd.format(quote(command))]

    logger.debug(' '.join(quote(part) for part in parts))
    try:
        raw_output, _ = check_output(parts, timeout, shell=False, combined_output=True)
    except subprocess.CalledProcessError as e:
        raise TargetStableError(str(e))

    if raw_output:
        try:
            output, exit_code, _ = raw_output.replace('\r\n', '\n').replace('\r', '\n').rsplit('\n', 2)
        except ValueError:
            exit_code, _ = raw_output.replace('\r\n', '\n').replace('\r', '\n').rsplit('\n', 1)
            output = ''
    else:  # raw_output is empty
        exit_code = '969696'  # just because
        output = ''

    if check_exit_code:
        exit_code = exit_code.strip()
        re_search = AM_START_ERROR.findall(output)
        if exit_code.isdigit():
            if int(exit_code):
                message = ('Got exit code {}\nfrom target command: {}\n'
                           'OUTPUT: {}')
                raise TargetStableError(message.format(exit_code, command, output))
            elif re_search:
                message = 'Could not start activity; got the following:\n{}'
                raise TargetStableError(message.format(re_search[0]))
        else:  # not all digits
            if re_search:
                message = 'Could not start activity; got the following:\n{}'
                raise TargetStableError(message.format(re_search[0]))
            else:
                message = 'adb has returned early; did not get an exit code. '\
                          'Was kill-server invoked?\nOUTPUT:\n-----\n{}\n'\
                          '-----'
                raise TargetTransientError(message.format(raw_output))

    return output


def adb_background_shell(device, command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         as_root=False,
                         adb_server=None):
    """Runs the sepcified command in a subprocess, returning the the Popen object."""
    _check_env()
    if as_root:
        command = 'echo {} | su'.format(quote(command))

    adb_cmd = get_adb_command(None, 'shell', adb_server)
    full_command = '{} {}'.format(adb_cmd, quote(command))
    logger.debug(full_command)
    return subprocess.Popen(full_command, stdout=stdout, stderr=stderr, shell=True)

def adb_kill_server(timeout=30, adb_server=None):
    adb_command(None, 'kill-server', timeout, adb_server)

def adb_list_devices(adb_server=None):
    output = adb_command(None, 'devices', adb_server=adb_server)
    devices = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split()]
        if len(parts) == 2:
            devices.append(AdbDevice(*parts))
    return devices


def get_adb_command(device, command, adb_server=None):
    _check_env()
    device_string = ""
    if adb_server != None:
        device_string = ' -H {}'.format(adb_server)
    device_string += ' -s {}'.format(device) if device else ''
    return "adb{} {}".format(device_string, command)

def adb_command(device, command, timeout=None, adb_server=None):
    full_command = get_adb_command(device, command, adb_server)
    logger.debug(full_command)
    output, _ = check_output(full_command, timeout, shell=True)
    return output

def grant_app_permissions(target, package):
    """
    Grant an app all the permissions it may ask for
    """
    dumpsys = target.execute('dumpsys package {}'.format(package))

    permissions = re.search(
        'requested permissions:\s*(?P<permissions>(android.permission.+\s*)+)', dumpsys
    )
    if permissions is None:
        return
    permissions = permissions.group('permissions').replace(" ", "").splitlines()

    for permission in permissions:
        try:
            target.execute('pm grant {} {}'.format(package, permission))
        except TargetStableError:
            logger.debug('Cannot grant {}'.format(permission))


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
    os.environ['PATH'] = env.platform_tools + os.pathsep + os.environ['PATH']
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

class LogcatMonitor(object):
    """
    Helper class for monitoring Anroid's logcat

    :param target: Android target to monitor
    :type target: :class:`AndroidTarget`

    :param regexps: List of uncompiled regular expressions to filter on the
                    device. Logcat entries that don't match any will not be
                    seen. If omitted, all entries will be sent to host.
    :type regexps: list(str)
    """

    @property
    def logfile(self):
        return self._logfile

    def __init__(self, target, regexps=None):
        super(LogcatMonitor, self).__init__()

        self.target = target
        self._regexps = regexps
        self._logcat = None
        self._logfile = None

    def start(self, outfile=None):
        """
        Start logcat and begin monitoring

        :param outfile: Optional path to file to store all logcat entries
        :type outfile: str
        """
        if outfile:
            self._logfile = open(outfile, 'w')
        else:
            self._logfile = tempfile.NamedTemporaryFile()

        self.target.clear_logcat()

        logcat_cmd = 'logcat'

        # Join all requested regexps with an 'or'
        if self._regexps:
            regexp = '{}'.format('|'.join(self._regexps))
            if len(self._regexps) > 1:
                regexp = '({})'.format(regexp)
            # Logcat on older version of android do not support the -e argument
            # so fall back to using grep.
            if self.target.get_sdk_version() > 23:
                logcat_cmd = '{} -e {}'.format(logcat_cmd, quote(regexp))
            else:
                logcat_cmd = '{} | grep {}'.format(logcat_cmd, quote(regexp))

        logcat_cmd = get_adb_command(self.target.conn.device, logcat_cmd)

        logger.debug('logcat command ="{}"'.format(logcat_cmd))
        self._logcat = pexpect.spawn(logcat_cmd, logfile=self._logfile)

    def stop(self):
        self._logcat.terminate()
        self._logfile.close()

    def get_log(self):
        """
        Return the list of lines found by the monitor
        """
        # Unless we tell pexect to 'expect' something, it won't read from
        # logcat's buffer or write into our logfile. We'll need to force it to
        # read any pending logcat output.
        while True:
            try:
                read_size = 1024 * 8
                # This will read up to read_size bytes, but only those that are
                # already ready (i.e. it won't block). If there aren't any bytes
                # already available it raises pexpect.TIMEOUT.
                buf = self._logcat.read_nonblocking(read_size, timeout=0)

                # We can't just keep calling read_nonblocking until we get a
                # pexpect.TIMEOUT (i.e. until we don't find any available
                # bytes), because logcat might be writing bytes the whole time -
                # in that case we might never return from this function. In
                # fact, we only care about bytes that were written before we
                # entered this function. So, if we read read_size bytes (as many
                # as we were allowed to), then we'll assume there are more bytes
                # that have already been sitting in the output buffer of the
                # logcat command. If not, we'll assume we read everything that
                # had already been written.
                if len(buf) == read_size:
                    continue
                else:
                    break
            except pexpect.TIMEOUT:
                # No available bytes to read. No prob, logcat just hasn't
                # printed anything since pexpect last read from its buffer.
                break

        with open(self._logfile.name) as fh:
            return [line for line in fh]

    def clear_log(self):
        with open(self._logfile.name, 'w') as _:
            pass

    def search(self, regexp):
        """
        Search a line that matches a regexp in the logcat log
        Return immediatly
        """
        return [line for line in self.get_log() if re.match(regexp, line)]

    def wait_for(self, regexp, timeout=30):
        """
        Search a line that matches a regexp in the logcat log
        Wait for it to appear if it's not found

        :param regexp: regexp to search
        :type regexp: str

        :param timeout: Timeout in seconds, before rasing RuntimeError.
                        ``None`` means wait indefinitely
        :type timeout: number

        :returns: List of matched strings
        """
        log = self.get_log()
        res = [line for line in log if re.match(regexp, line)]

        # Found some matches, return them
        if res:
            return res

        # Store the number of lines we've searched already, so we don't have to
        # re-grep them after 'expect' returns
        next_line_num = len(log)

        try:
            self._logcat.expect(regexp, timeout=timeout)
        except pexpect.TIMEOUT:
            raise RuntimeError('Logcat monitor timeout ({}s)'.format(timeout))

        return [line for line in self.get_log()[next_line_num:]
                if re.match(regexp, line)]
