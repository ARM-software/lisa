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

# pylint: disable=E1101
import os
import sys
import re
import time
import tempfile
import shutil
import threading
from subprocess import CalledProcessError

from wlauto.core.extension import Parameter
from wlauto.common.linux.device import BaseLinuxDevice, PsEntry
from wlauto.exceptions import DeviceError, WorkerThreadError, TimeoutError, DeviceNotRespondingError
from wlauto.utils.misc import convert_new_lines
from wlauto.utils.types import boolean, regex
from wlauto.utils.android import (adb_shell, adb_background_shell, adb_list_devices,
                                  adb_command, AndroidProperties, ANDROID_VERSION_MAP)


SCREEN_STATE_REGEX = re.compile('(?:mPowerState|mScreenOn)=([0-9]+|true|false)', re.I)


class AndroidDevice(BaseLinuxDevice):  # pylint: disable=W0223
    """
    Device running Android OS.

    """

    platform = 'android'

    parameters = [
        Parameter('adb_name',
                  description='The unique ID of the device as output by "adb devices".'),
        Parameter('android_prompt', kind=regex, default=re.compile('^.*(shell|root)@.*:/\S* [#$] ', re.MULTILINE),
                  description='The format  of matching the shell prompt in Android.'),
        Parameter('working_directory', default='/sdcard/wa-working',
                  description='Directory that will be used WA on the device for output files etc.'),
        Parameter('binaries_directory', default='/system/bin',
                  description='Location of binaries on the device.'),
        Parameter('package_data_directory', default='/data/data',
                  description='Location of of data for an installed package (APK).'),
        Parameter('external_storage_directory', default='/sdcard',
                  description='Mount point for external storage.'),
        Parameter('connection', default='usb', allowed_values=['usb', 'ethernet'],
                  description='Specified the nature of adb connection.'),
        Parameter('logcat_poll_period', kind=int,
                  description="""
                  If specified and is not ``0``, logcat will be polled every
                  ``logcat_poll_period`` seconds, and buffered on the host. This
                  can be used if a lot of output is expected in logcat and the fixed
                  logcat buffer on the device is not big enough. The trade off is that
                  this introduces some minor runtime overhead. Not set by default.
                  """),
        Parameter('enable_screen_check', kind=boolean, default=False,
                  description="""
                  Specified whether the device should make sure that the screen is on
                  during initialization.
                  """),
    ]

    default_timeout = 30
    delay = 2
    long_delay = 3 * delay
    ready_timeout = 60

    # Overwritten from Device. For documentation, see corresponding method in
    # Device.

    @property
    def is_rooted(self):
        if self._is_rooted is None:
            try:
                result = adb_shell(self.adb_name, 'su', timeout=1)
                if 'not found' in result:
                    self._is_rooted = False
                else:
                    self._is_rooted = True
            except TimeoutError:
                self._is_rooted = True
            except DeviceError:
                self._is_rooted = False
        return self._is_rooted

    @property
    def abi(self):
        return self.getprop()['ro.product.cpu.abi'].split('-')[0]

    @property
    def supported_eabi(self):
        props = self.getprop()
        result = [props['ro.product.cpu.abi']]
        if 'ro.product.cpu.abi2' in props:
            result.append(props['ro.product.cpu.abi2'])
        if 'ro.product.cpu.abilist' in props:
            for eabi in props['ro.product.cpu.abilist'].split(','):
                if eabi not in result:
                    result.append(eabi)
        return result

    def __init__(self, **kwargs):
        super(AndroidDevice, self).__init__(**kwargs)
        self._logcat_poller = None

    def reset(self):
        self._is_ready = False
        self._just_rebooted = True
        adb_command(self.adb_name, 'reboot', timeout=self.default_timeout)

    def hard_reset(self):
        super(AndroidDevice, self).hard_reset()
        self._is_ready = False
        self._just_rebooted = True

    def boot(self, **kwargs):
        self.reset()

    def connect(self):  # NOQA pylint: disable=R0912
        iteration_number = 0
        max_iterations = self.ready_timeout / self.delay
        available = False
        self.logger.debug('Polling for device {}...'.format(self.adb_name))
        while iteration_number < max_iterations:
            devices = adb_list_devices()
            if self.adb_name:
                for device in devices:
                    if device.name == self.adb_name and device.status != 'offline':
                        available = True
            else:  # adb_name not set
                if len(devices) == 1:
                    available = True
                elif len(devices) > 1:
                    raise DeviceError('More than one device is connected and adb_name is not set.')

            if available:
                break
            else:
                time.sleep(self.delay)
                iteration_number += 1
        else:
            raise DeviceError('Could not boot {} ({}).'.format(self.name, self.adb_name))

        while iteration_number < max_iterations:
            available = (1 == int('0' + adb_shell(self.adb_name, 'getprop sys.boot_completed', timeout=self.default_timeout)))
            if available:
                break
            else:
                time.sleep(self.delay)
                iteration_number += 1
        else:
            raise DeviceError('Could not boot {} ({}).'.format(self.name, self.adb_name))

        if self._just_rebooted:
            self.logger.debug('Waiting for boot to complete...')
            # On some devices, adb connection gets reset some time after booting.
            # This  causes errors during execution. To prevent this, open a shell
            # session and wait for it to be killed. Once its killed, give adb
            # enough time to restart, and then the device should be ready.
            # TODO: This is more of a work-around rather than an actual solution.
            #       Need to figure out what is going on the "proper" way of handling it.
            try:
                adb_shell(self.adb_name, '', timeout=20)
                time.sleep(5)  # give adb time to re-initialize
            except TimeoutError:
                pass  # timed out waiting for the session to be killed -- assume not going to be.

            self.logger.debug('Boot completed.')
            self._just_rebooted = False
        self._is_ready = True

    def initialize(self, context):
        self.execute('mkdir -p {}'.format(self.working_directory))
        if self.is_rooted:
            if not self.executable_is_installed('busybox'):
                self.busybox = self.deploy_busybox(context)
            else:
                self.busybox = self.path.join(self.binaries_directory, 'busybox')
            self.disable_screen_lock()
            self.disable_selinux()
        if self.enable_screen_check:
            self.ensure_screen_is_on()

    def disconnect(self):
        if self._logcat_poller:
            self._logcat_poller.close()

    def ping(self):
        try:
            # May be triggered inside initialize()
            adb_shell(self.adb_name, 'ls /', timeout=10)
        except (TimeoutError, CalledProcessError):
            raise DeviceNotRespondingError(self.adb_name or self.name)

    def start(self):
        if self.logcat_poll_period:
            if self._logcat_poller:
                self._logcat_poller.close()
            self._logcat_poller = _LogcatPoller(self, self.logcat_poll_period, timeout=self.default_timeout)
            self._logcat_poller.start()

    def stop(self):
        if self._logcat_poller:
            self._logcat_poller.stop()

    def get_android_version(self):
        return ANDROID_VERSION_MAP.get(self.get_sdk_version(), None)

    def get_android_id(self):
        """
        Get the device's ANDROID_ID. Which is

            "A 64-bit number (as a hex string) that is randomly generated when the user
            first sets up the device and should remain constant for the lifetime of the
            user's device."

        .. note:: This will get reset on userdata erasure.

        """
        return self.execute('settings get secure android_id').strip()

    def get_sdk_version(self):
        try:
            return int(self.getprop('ro.build.version.sdk'))
        except (ValueError, TypeError):
            return None

    def get_installed_package_version(self, package):
        """
        Returns the version (versionName) of the specified package if it is installed
        on the device, or ``None`` otherwise.

        Added in version 2.1.4

        """
        output = self.execute('dumpsys package {}'.format(package))
        for line in convert_new_lines(output).split('\n'):
            if 'versionName' in line:
                return line.split('=', 1)[1]
        return None

    def list_packages(self):
        """
        List packages installed on the device.

        Added in version 2.1.4

        """
        output = self.execute('pm list packages')
        output = output.replace('package:', '')
        return output.split()

    def package_is_installed(self, package_name):
        """
        Returns ``True`` the if a package with the specified name is installed on
        the device, and ``False`` otherwise.

        Added in version 2.1.4

        """
        return package_name in self.list_packages()

    def executable_is_installed(self, executable_name):
        return executable_name in self.listdir(self.binaries_directory)

    def is_installed(self, name):
        return self.executable_is_installed(name) or self.package_is_installed(name)

    def listdir(self, path, as_root=False, **kwargs):
        contents = self.execute('ls {}'.format(path), as_root=as_root)
        return [x.strip() for x in contents.split()]

    def push_file(self, source, dest, as_root=False, timeout=default_timeout):  # pylint: disable=W0221
        """
        Modified in version 2.1.4: added  ``as_root`` parameter.

        """
        self._check_ready()
        try:
            if not as_root:
                adb_command(self.adb_name, "push '{}' '{}'".format(source, dest), timeout=timeout)
            else:
                device_tempfile = self.path.join(self.file_transfer_cache, source.lstrip(self.path.sep))
                self.execute('mkdir -p {}'.format(self.path.dirname(device_tempfile)))
                adb_command(self.adb_name, "push '{}' '{}'".format(source, device_tempfile), timeout=timeout)
                self.execute('cp {} {}'.format(device_tempfile, dest), as_root=True)
        except CalledProcessError as e:
            raise DeviceError(e)

    def pull_file(self, source, dest, as_root=False, timeout=default_timeout):  # pylint: disable=W0221
        """
        Modified in version 2.1.4: added  ``as_root`` parameter.

        """
        self._check_ready()
        try:
            if not as_root:
                adb_command(self.adb_name, "pull '{}' '{}'".format(source, dest), timeout=timeout)
            else:
                device_tempfile = self.path.join(self.file_transfer_cache, source.lstrip(self.path.sep))
                self.execute('mkdir -p {}'.format(self.path.dirname(device_tempfile)))
                self.execute('cp {} {}'.format(source, device_tempfile), as_root=True)
                adb_command(self.adb_name, "pull '{}' '{}'".format(device_tempfile, dest), timeout=timeout)
        except CalledProcessError as e:
            raise DeviceError(e)

    def delete_file(self, filepath, as_root=False):  # pylint: disable=W0221
        self._check_ready()
        adb_shell(self.adb_name, "rm '{}'".format(filepath), as_root=as_root, timeout=self.default_timeout)

    def file_exists(self, filepath):
        self._check_ready()
        output = adb_shell(self.adb_name, 'if [ -e \'{}\' ]; then echo 1; else echo 0; fi'.format(filepath),
                           timeout=self.default_timeout)
        if int(output):
            return True
        else:
            return False

    def install(self, filepath, timeout=default_timeout, with_name=None):  # pylint: disable=W0221
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            return self.install_apk(filepath, timeout)
        else:
            return self.install_executable(filepath, with_name)

    def install_apk(self, filepath, timeout=default_timeout):  # pylint: disable=W0221
        self._check_ready()
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            return adb_command(self.adb_name, "install {}".format(filepath), timeout=timeout)
        else:
            raise DeviceError('Can\'t install {}: unsupported format.'.format(filepath))

    def install_executable(self, filepath, with_name=None):
        """
        Installs a binary executable on device. Requires root access. Returns
        the path to the installed binary, or ``None`` if the installation has failed.
        Optionally, ``with_name`` parameter may be used to specify a different name under
        which the executable will be installed.

        Added in version 2.1.3.
        Updated in version 2.1.5 with ``with_name`` parameter.

        """
        self._ensure_binaries_directory_is_writable()
        executable_name = with_name or os.path.basename(filepath)
        on_device_file = self.path.join(self.working_directory, executable_name)
        on_device_executable = self.path.join(self.binaries_directory, executable_name)
        self.push_file(filepath, on_device_file)
        self.execute('cp {} {}'.format(on_device_file, on_device_executable), as_root=self.is_rooted)
        self.execute('chmod 0777 {}'.format(on_device_executable), as_root=self.is_rooted)
        return on_device_executable

    def uninstall(self, package):
        self._check_ready()
        adb_command(self.adb_name, "uninstall {}".format(package), timeout=self.default_timeout)

    def uninstall_executable(self, executable_name):
        """
        Requires root access.

        Added in version 2.1.3.

        """
        on_device_executable = self.path.join(self.binaries_directory, executable_name)
        self._ensure_binaries_directory_is_writable()
        self.delete_file(on_device_executable, as_root=self.is_rooted)

    def execute(self, command, timeout=default_timeout, check_exit_code=True, background=False,
                as_root=False, busybox=False, **kwargs):
        """
        Execute the specified command on the device using adb.

        Parameters:

            :param command: The command to be executed. It should appear exactly
                            as if you were typing it into a shell.
            :param timeout: Time, in seconds, to wait for adb to return before aborting
                            and raising an error. Defaults to ``AndroidDevice.default_timeout``.
            :param check_exit_code: If ``True``, the return code of the command on the Device will
                                    be check and exception will be raised if it is not 0.
                                    Defaults to ``True``.
            :param background: If ``True``, will execute adb in a subprocess, and will return
                               immediately, not waiting for adb to return. Defaults to ``False``
            :param busybox: If ``True``, will use busybox to execute the command. Defaults to ``False``.

                            Added in version 2.1.3

                            .. note:: The device must be rooted to be able to use busybox.

            :param as_root: If ``True``, will attempt to execute command in privileged mode. The device
                            must be rooted, otherwise an error will be raised. Defaults to ``False``.

                            Added in version 2.1.3

        :returns: If ``background`` parameter is set to ``True``, the subprocess object will
                  be returned; otherwise, the contents of STDOUT from the device will be returned.

        :raises: DeviceError if adb timed out  or if the command returned non-zero exit
                 code on the device, or if attempting to execute a command in privileged mode on an
                 unrooted device.

        """
        self._check_ready()
        if as_root and not self.is_rooted:
            raise DeviceError('Attempting to execute "{}" as root on unrooted device.'.format(command))
        if busybox:
            if not self.is_rooted:
                DeviceError('Attempting to execute "{}" with busybox. '.format(command) +
                            'Busybox can only be deployed to rooted devices.')
            command = ' '.join([self.busybox, command])
        if background:
            return adb_background_shell(self.adb_name, command, as_root=as_root)
        else:
            return adb_shell(self.adb_name, command, timeout, check_exit_code, as_root)

    def kick_off(self, command):
        """
        Like execute but closes adb session and returns immediately, leaving the command running on the
        device (this is different from execute(background=True) which keeps adb connection open and returns
        a subprocess object).

        .. note:: This relies on busybox's nohup applet and so won't work on unrooted devices.

        Added in version 2.1.4

        """
        if not self.is_rooted:
            raise DeviceError('kick_off uses busybox\'s nohup applet and so can only be run a rooted device.')
        try:
            command = 'cd {} && busybox nohup {}'.format(self.working_directory, command)
            output = self.execute(command, timeout=1, as_root=True)
        except TimeoutError:
            pass
        else:
            raise ValueError('Background command exited before timeout; got "{}"'.format(output))

    def get_pids_of(self, process_name):
        """Returns a list of PIDs of all processes with the specified name."""
        result = self.execute('ps {}'.format(process_name[-15:]), check_exit_code=False).strip()
        if result and 'not found' not in result:
            return [int(x.split()[1]) for x in result.split('\n')[1:]]
        else:
            return []

    def ps(self, **kwargs):
        """
        Returns the list of running processes on the device. Keyword arguments may
        be used to specify simple filters for columns.

        Added in version 2.1.4

        """
        lines = iter(convert_new_lines(self.execute('ps')).split('\n'))
        lines.next()  # header
        result = []
        for line in lines:
            parts = line.split()
            if parts:
                result.append(PsEntry(*(parts[0:1] + map(int, parts[1:5]) + parts[5:])))
        if not kwargs:
            return result
        else:
            filtered_result = []
            for entry in result:
                if all(getattr(entry, k) == v for k, v in kwargs.iteritems()):
                    filtered_result.append(entry)
            return filtered_result

    def get_properties(self, context):
        """Captures and saves the information from /system/build.prop and /proc/version"""
        props = super(AndroidDevice, self).get_properties(context)
        props['android_id'] = self.get_android_id()
        buildprop_file = os.path.join(context.host_working_directory, 'build.prop')
        if not os.path.isfile(buildprop_file):
            self.pull_file('/system/build.prop', context.host_working_directory)
        self._update_build_properties(buildprop_file, props)
        context.add_run_artifact('build_properties', buildprop_file, 'export')

        dumpsys_window_file = os.path.join(context.host_working_directory, 'window.dumpsys')
        dumpsys_window_output = self.execute('dumpsys window')
        with open(dumpsys_window_file, 'w') as wfh:
            wfh.write(dumpsys_window_output)
        context.add_run_artifact('dumpsys_window', dumpsys_window_file, 'meta')
        return props

    def getprop(self, prop=None):
        """Returns parsed output of Android getprop command. If a property is
        specified, only the value for that property will be returned (with
        ``None`` returned if the property doesn't exist. Otherwise,
        ``wlauto.utils.android.AndroidProperties`` will be returned, which is
        a dict-like object."""
        props = AndroidProperties(self.execute('getprop'))
        if prop:
            return props[prop]
        return props

    # Android-specific methods. These either rely on specifics of adb or other
    # Android-only concepts in their interface and/or implementation.

    def forward_port(self, from_port, to_port):
        """
        Forward a port on the device to a port on localhost.

        :param from_port: Port on the device which to forward.
        :param to_port: Port on the localhost to which the device port will be forwarded.

        Ports should be specified using adb spec. See the "adb forward" section in "adb help".

        """
        adb_command(self.adb_name, 'forward {} {}'.format(from_port, to_port), timeout=self.default_timeout)

    def dump_logcat(self, outfile, filter_spec=None):
        """
        Dump the contents of logcat, for the specified filter spec to the
        specified output file.
        See http://developer.android.com/tools/help/logcat.html

        :param outfile: Output file on the host into which the contents of the
                        log will be written.
        :param filter_spec: Logcat filter specification.
                            see http://developer.android.com/tools/debugging/debugging-log.html#filteringOutput

        """
        if self._logcat_poller:
            return self._logcat_poller.write_log(outfile)
        else:
            if filter_spec:
                command = 'logcat -d -s {} > {}'.format(filter_spec, outfile)
            else:
                command = 'logcat -d > {}'.format(outfile)
            return adb_command(self.adb_name, command, timeout=self.default_timeout)

    def clear_logcat(self):
        """Clear (flush) logcat log."""
        if self._logcat_poller:
            return self._logcat_poller.clear_buffer()
        else:
            return adb_shell(self.adb_name, 'logcat -c', timeout=self.default_timeout)

    def capture_screen(self, filepath):
        """Caputers the current device screen into the specified file in a PNG format."""
        on_device_file = self.path.join(self.working_directory, 'screen_capture.png')
        self.execute('screencap -p  {}'.format(on_device_file))
        self.pull_file(on_device_file, filepath)
        self.delete_file(on_device_file)

    def is_screen_on(self):
        """Returns ``True`` if the device screen is currently on, ``False`` otherwise."""
        output = self.execute('dumpsys power')
        match = SCREEN_STATE_REGEX.search(output)
        if match:
            return boolean(match.group(1))
        else:
            raise DeviceError('Could not establish screen state.')

    def ensure_screen_is_on(self):
        if not self.is_screen_on():
            self.execute('input keyevent 26')

    def disable_screen_lock(self):
        """
        Attempts to disable he screen lock on the device.

        .. note:: This does not always work...

        Added inversion 2.1.4

        """
        lockdb = '/data/system/locksettings.db'
        sqlcommand = "update locksettings set value=\\'0\\' where name=\\'screenlock.disabled\\';"
        self.execute('sqlite3 {} "{}"'.format(lockdb, sqlcommand), as_root=True)

    def disable_selinux(self):
        # This may be invoked from intialize() so we can't use execute() or the
        # standard API for doing this.
        api_level = int(adb_shell(self.adb_name, 'getprop ro.build.version.sdk',
                                  timeout=self.default_timeout).strip())
        # SELinux was added in Android 4.3 (API level 18). Trying to
        # 'getenforce' in earlier versions will produce an error.
        if api_level >= 18:
            se_status = self.execute('getenforce', as_root=True).strip()
            if se_status == 'Enforcing':
                self.execute('setenforce 0', as_root=True)

    # Internal methods: do not use outside of the class.

    def _update_build_properties(self, filepath, props):
        try:
            with open(filepath) as fh:
                for line in fh:
                    line = re.sub(r'#.*', '', line).strip()
                    if not line:
                        continue
                    key, value = line.split('=', 1)
                    props[key] = value
        except ValueError:
            self.logger.warning('Could not parse build.prop.')

    def _update_versions(self, filepath, props):
        with open(filepath) as fh:
            text = fh.read()
            props['version'] = text
            text = re.sub(r'#.*', '', text).strip()
            match = re.search(r'^(Linux version .*?)\s*\((gcc version .*)\)$', text)
            if match:
                props['linux_version'] = match.group(1).strip()
                props['gcc_version'] = match.group(2).strip()
            else:
                self.logger.warning('Could not parse version string.')

    def _ensure_binaries_directory_is_writable(self):
        matched = []
        for entry in self.list_file_systems():
            if self.binaries_directory.rstrip('/').startswith(entry.mount_point):
                matched.append(entry)
        if matched:
            entry = sorted(matched, key=lambda x: len(x.mount_point))[-1]
            if 'rw' not in entry.options:
                self.execute('mount -o rw,remount {} {}'.format(entry.device, entry.mount_point), as_root=True)
        else:
            raise DeviceError('Could not find mount point for binaries directory {}'.format(self.binaries_directory))


class _LogcatPoller(threading.Thread):

    join_timeout = 5

    def __init__(self, device, period, timeout=None):
        super(_LogcatPoller, self).__init__()
        self.adb_device = device.adb_name
        self.logger = device.logger
        self.period = period
        self.timeout = timeout
        self.stop_signal = threading.Event()
        self.lock = threading.RLock()
        self.buffer_file = tempfile.mktemp()
        self.last_poll = 0
        self.daemon = True
        self.exc = None

    def run(self):
        self.logger.debug('Starting logcat polling.')
        try:
            while True:
                if self.stop_signal.is_set():
                    break
                with self.lock:
                    current_time = time.time()
                    if (current_time - self.last_poll) >= self.period:
                        self._poll()
                time.sleep(0.5)
        except Exception:  # pylint: disable=W0703
            self.exc = WorkerThreadError(self.name, sys.exc_info())
        self.logger.debug('Logcat polling stopped.')

    def stop(self):
        self.logger.debug('Stopping logcat polling.')
        self.stop_signal.set()
        self.join(self.join_timeout)
        if self.is_alive():
            self.logger.error('Could not join logcat poller thread.')
        if self.exc:
            raise self.exc  # pylint: disable=E0702

    def clear_buffer(self):
        self.logger.debug('Clearing logcat buffer.')
        with self.lock:
            adb_shell(self.adb_device, 'logcat -c', timeout=self.timeout)
            with open(self.buffer_file, 'w') as _:  # NOQA
                pass

    def write_log(self, outfile):
        self.logger.debug('Writing logbuffer to {}.'.format(outfile))
        with self.lock:
            self._poll()
            if os.path.isfile(self.buffer_file):
                shutil.copy(self.buffer_file, outfile)
            else:  # there was no logcat trace at this time
                with open(outfile, 'w') as _:  # NOQA
                    pass

    def close(self):
        self.logger.debug('Closing logcat poller.')
        if os.path.isfile(self.buffer_file):
            os.remove(self.buffer_file)

    def _poll(self):
        with self.lock:
            self.last_poll = time.time()
            adb_command(self.adb_device, 'logcat -d >> {}'.format(self.buffer_file), timeout=self.timeout)
            adb_command(self.adb_device, 'logcat -c', timeout=self.timeout)


class BigLittleDevice(AndroidDevice):  # pylint: disable=W0223

    parameters = [
        Parameter('scheduler', default='hmp', override=True),
    ]

