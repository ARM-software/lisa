import os
import sys
import re
import time
import tempfile
import shutil
import threading

from wlauto.core.device_manager import DeviceManager
from wlauto import Parameter, Alias
from wlauto.utils.types import boolean, regex
from wlauto.utils.android import adb_command
from wlauto.exceptions import WorkerThreadError

from devlib.target import AndroidTarget

SCREEN_STATE_REGEX = re.compile('(?:mPowerState|mScreenOn|Display Power: state)=([0-9]+|true|false|ON|OFF)', re.I)
SCREEN_SIZE_REGEX = re.compile(r'mUnrestrictedScreen=\(\d+,\d+\)\s+(?P<width>\d+)x(?P<height>\d+)')


class AndroidDevice(DeviceManager):

    name = "android"
    target_type = AndroidTarget

    aliases = [
        Alias('generic_android'),
    ]

    parameters = [
        Parameter('adb_name', default=None, kind=str,
                  description='The unique ID of the device as output by "adb devices".'),
        Parameter('android_prompt', kind=regex, default=re.compile('^.*(shell|root)@.*:/\S* [#$] ', re.MULTILINE),  # ##
                  description='The format  of matching the shell prompt in Android.'),
        Parameter('working_directory', default='/sdcard/wa-working', override=True),
        Parameter('binaries_directory', default='/data/local/tmp', override=True),
        Parameter('package_data_directory', default='/data/data',
                  description='Location of of data for an installed package (APK).'),
        Parameter('external_storage_directory', default='/sdcard',
                  description='Mount point for external storage.'),
        Parameter('logcat_poll_period', kind=int,
                  description="""
                  If specified and is not ``0``, logcat will be polled every
                  ``logcat_poll_period`` seconds, and buffered on the host. This
                  can be used if a lot of output is expected in logcat and the fixed
                  logcat buffer on the device is not big enough. The trade off is that
                  this introduces some minor runtime overhead. Not set by default.
                  """),   # ##
        Parameter('enable_screen_check', kind=boolean, default=False,
                  description="""
                  Specified whether the device should make sure that the screen is on
                  during initialization.
                  """),
        Parameter('swipe_to_unlock', kind=str, default=None,
                  allowed_values=[None, "horizontal", "vertical"],
                  description="""
                  If set a swipe of the specified direction will be performed.
                  This should unlock the screen.
                  """),  # ##
    ]

    def __init__(self, **kwargs):
        super(AndroidDevice, self).__init__(**kwargs)
        self.connection_settings = self._make_connection_settings()

        self.platform = self.platform_type(core_names=self.core_names,  # pylint: disable=E1102
                                           core_clusters=self.core_clusters)

        self.target = self.target_type(connection_settings=self.connection_settings,
                                       connect=False,
                                       platform=self.platform,
                                       working_directory=self.working_directory,
                                       executables_directory=self.binaries_directory,)
        self._logcat_poller = None

    def connect(self):
        self.target.connect()

    def initialize(self, context):
        super(AndroidDevice, self).initialize(context)
        if self.enable_screen_check:
            self.target.ensure_screen_is_on()
            if self.swipe_to_unlock:
                self.target.swipe_to_unlock(direction=self.swipe_to_unlock)

    def start(self):
        if self.logcat_poll_period:
            if self._logcat_poller:
                self._logcat_poller.close()
            self._logcat_poller = _LogcatPoller(self, self.logcat_poll_period,
                                                timeout=self.default_timeout)
            self._logcat_poller.start()
        else:
            self.target.clear_logcat()

    def _make_connection_settings(self):
        connection_settings = {}
        connection_settings['device'] = self.adb_name
        return connection_settings

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
            return adb_command(self.adb_name, command)


class _LogcatPoller(threading.Thread):

    join_timeout = 5

    def __init__(self, target, period, timeout=None):
        super(_LogcatPoller, self).__init__()
        self.target = target
        self.logger = target.logger
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
            self.target.clear_logcat()
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
            self.target.dump_logcat(self.buffer_file, append=True)
