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

import logging
import os
import shutil
import sys
import tempfile
import threading
import time

from wa.framework.plugin import Parameter
from wa.framework.exception import WorkerThreadError
from wa.utils.misc import touch


class LinuxAssistant(object):

    parameters = []

    def __init__(self, target):
        self.target = target

    def start(self):
        pass

    def extract_results(self, context):
        pass

    def stop(self):
        pass


class AndroidAssistant(object):

    parameters = [
        Parameter('disable_selinux', kind=bool, default=True,
                  description="""
                  If ``True``, the default, and the target is rooted, an attempt will
                  be made to disable SELinux by running ``setenforce 0`` on the target
                  at the beginning of the run.
                  """),
        Parameter('logcat_poll_period', kind=int,
                  constraint=lambda x: x > 0,
                  description="""
                  Polling period for logcat in seconds. If not specified,
                  no polling will be used.

                  Logcat buffer on android is of limited size and it cannot be
                  adjusted at run time. Depending on the amount of logging activity,
                  the buffer may not be enought to capture comlete trace for a
                  workload execution. For those situations, logcat may be polled
                  periodically during the course of the run and stored in a
                  temporary locaiton on the host. Setting the value of the poll
                  period enables this behavior.
                  """),
    ]

    def __init__(self, target, logcat_poll_period=None, disable_selinux=True):
        self.target = target
        self.logcat_poll_period = logcat_poll_period
        self.disable_selinux = disable_selinux
        self.logcat_poller = None
        if self.target.is_rooted and self.disable_selinux:
            self.do_disable_selinux()

    def start(self):
        if self.logcat_poll_period:
            self.logcat_poller = LogcatPoller(self.target, self.logcat_poll_period)
            self.logcat_poller.start()

    def stop(self):
        if self.logcat_poller:
            self.logcat_poller.stop()

    def extract_results(self, context):
        logcat_file = os.path.join(context.output_directory, 'logcat.log')
        self.dump_logcat(logcat_file)
        context.add_artifact('logcat', logcat_file, kind='log')
        self.clear_logcat()

    def dump_logcat(self, outfile):
        if self.logcat_poller:
            self.logcat_poller.write_log(outfile)
        else:
            self.target.dump_logcat(outfile)

    def clear_logcat(self):
        if self.logcat_poller:
            self.logcat_poller.clear_buffer()

    def do_disable_selinux(self):
        # SELinux was added in Android 4.3 (API level 18). Trying to
        # 'getenforce' in earlier versions will produce an error.
        if self.target.get_sdk_version() >= 18:
            se_status = self.target.execute('getenforce', as_root=True).strip()
            if se_status == 'Enforcing':
                self.target.execute('setenforce 0', as_root=True, check_exit_code=False)


class LogcatPoller(threading.Thread):

    def __init__(self, target, period=60, timeout=30):
        super(LogcatPoller, self).__init__()
        self.target = target
        self.logger = logging.getLogger('logcat')
        self.period = period
        self.timeout = timeout
        self.stop_signal = threading.Event()
        self.lock = threading.Lock()
        self.buffer_file = tempfile.mktemp()
        self.last_poll = 0
        self.daemon = True
        self.exc = None

    def run(self):
        self.logger.debug('Starting polling')
        try:
            while True:
                if self.stop_signal.is_set():
                    break
                with self.lock:
                    current_time = time.time()
                    if (current_time - self.last_poll) >= self.period:
                        self.poll()
                time.sleep(0.5)
        except Exception:  # pylint: disable=W0703
            self.exc = WorkerThreadError(self.name, sys.exc_info())
        self.logger.debug('Polling stopped')

    def stop(self):
        self.logger.debug('Stopping logcat polling')
        self.stop_signal.set()
        self.join(self.timeout)
        if self.is_alive():
            self.logger.error('Could not join logcat poller thread.')
        if self.exc:
            raise self.exc  # pylint: disable=E0702

    def clear_buffer(self):
        self.logger.debug('Clearing logcat buffer')
        with self.lock:
            self.target.clear_logcat()
            touch(self.buffer_file)

    def write_log(self, outfile):
        with self.lock:
            self.poll()
            if os.path.isfile(self.buffer_file):
                shutil.copy(self.buffer_file, outfile)
            else:  # there was no logcat trace at this time
                touch(outfile)

    def close(self):
        self.logger.debug('Closing poller')
        if os.path.isfile(self.buffer_file):
            os.remove(self.buffer_file)

    def poll(self):
        self.last_poll = time.time()
        self.target.dump_logcat(self.buffer_file, append=True, timeout=self.timeout)
        self.target.clear_logcat()


class ChromeOsAssistant(LinuxAssistant):

    parameters = LinuxAssistant.parameters + AndroidAssistant.parameters

    def __init__(self, target, logcat_poll_period=None, disable_selinux=True):
        super(ChromeOsAssistant, self).__init__(target)
        if target.supports_android:
            self.android_assistant = AndroidAssistant(target.android_container,
                                                      logcat_poll_period, disable_selinux)
        else:
            self.android_assistant = None

    def start(self):
        super(ChromeOsAssistant, self).start()
        if self.android_assistant:
            self.android_assistant.start()

    def extract_results(self, context):
        super(ChromeOsAssistant, self).extract_results(context)
        if self.android_assistant:
            self.android_assistant.extract_results(context)

    def stop(self):
        super(ChromeOsAssistant, self).stop()
        if self.android_assistant:
            self.android_assistant.stop()
