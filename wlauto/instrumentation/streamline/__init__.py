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


# pylint: disable=W0613,E1101
import os
import signal
import shutil
import subprocess
import logging
import re

from wlauto import settings, Instrument, Parameter, ResourceGetter, GetterPriority, File
from wlauto.exceptions import InstrumentError, DeviceError, ResourceError
from wlauto.utils.misc import ensure_file_directory_exists as _f, which
from wlauto.utils.types import boolean
from wlauto.utils.log import StreamLogger, LogWriter, LineLogWriter


SESSION_TEXT_TEMPLATE = ('<?xml version="1.0" encoding="US-ASCII" ?>'
                         '<session'
                         '    version="1"'
                         '    output_path="x"'
                         '    call_stack_unwinding="no"'
                         '    parse_debug_info="no"'
                         '    high_resolution="no"'
                         '    buffer_mode="streaming"'
                         '    sample_rate="none"'
                         '    duration="0"'
                         '    target_host="{}"'
                         '    target_port="{}"'
                         '    energy_cmd_line="{}">'
                         '</session>')

VERSION_REGEX = re.compile(r'Streamline (.*?) ')


class StreamlineResourceGetter(ResourceGetter):

    name = 'streamline_resource'
    resource_type = 'file'
    priority = GetterPriority.environment + 1  # run before standard enviroment resolvers.

    dependencies_directory = os.path.join(settings.dependencies_directory, 'streamline')
    old_dependencies_directory = os.path.join(settings.environment_root, 'streamline')  # backwards compatibility

    def get(self, resource, **kwargs):
        if resource.owner.name != 'streamline':
            return None
        test_path = _f(os.path.join(self.dependencies_directory, resource.path))
        if os.path.isfile(test_path):
            return test_path
        test_path = _f(os.path.join(self.old_dependencies_directory, resource.path))
        if os.path.isfile(test_path):
            return test_path


def _instantiate(resolver):
    return StreamlineResourceGetter(resolver)


class StreamlineInstrument(Instrument):

    name = 'streamline'
    description = """
    Collect Streamline traces from the device.

    .. note:: This instrument supports streamline that comes with DS-5 5.17 and later
              earlier versions of streamline  may not work correctly (or at all).

    This Instrument allows collecting streamline traces (such as PMU counter values) from
    the device. It assumes you have DS-5 (which Streamline is part of) installed on your
    system, and that streamline command is somewhere in PATH.

    Streamline works by connecting to gator service on the device. gator comes in two parts
    a driver (gator.ko) and daemon (gatord). The driver needs to be compiled against your
    kernel and both driver and daemon need to be compatible with your version of Streamline.
    The best way to ensure compatibility is to build them from source which came with your
    DS-5. gator source can be found in ::

        /usr/local/DS-5/arm/gator

    (the exact path may vary depending of where you have installed DS-5.) Please refer to the
    README the accompanies the source for instructions on how to build it.

    Once you have built the driver and the daemon, place the binaries into your
    ~/.workload_automation/streamline/ directory (if you haven't tried running WA with
    this instrument before, the streamline/ subdirectory might not exist, in which
    case you will need to create it.

    In order to specify which events should be captured, you need to provide a
    configuration.xml for the gator. The easiest way to obtain this file is to export it
    from event configuration dialog in DS-5 streamline GUI. The file should be called
    "configuration.xml" and it be placed in the same directory as the gator binaries.
    """
    parameters = [
        Parameter('port', default='8080',
                  description='Specifies the port on which streamline will connect to gator'),
        Parameter('configxml', default=None,
                  description='streamline configuration XML file to be used. This must be '
                              'an absolute path, though it may count the user home symbol (~)'),
        Parameter('report', kind=boolean, default=False, global_alias='streamline_report_csv',
                  description='Specifies whether a report should be generated from streamline data.'),
        Parameter('report_options', kind=str, default='-format csv',
                  description='A string with options that will be added to streamline -report command.'),
    ]

    daemon = 'gatord'
    driver = 'gator.ko'
    configuration_file_name = 'configuration.xml'

    def __init__(self, device, **kwargs):
        super(StreamlineInstrument, self).__init__(device, **kwargs)
        self.streamline = None
        self.session_file = None
        self.capture_file = None
        self.analysis_file = None
        self.report_file = None
        self.configuration_file = None
        self.on_device_config = None
        self.daemon_process = None
        self.resource_getter = None

        self.host_daemon_file = None
        self.host_driver_file = None
        self.device_driver_file = None

        self._check_has_valid_display()

    def validate(self):
        if not which('streamline'):
            raise InstrumentError('streamline not in PATH. Cannot enable Streamline tracing.')
        p = subprocess.Popen('streamline --version 2>&1', stdout=subprocess.PIPE, shell=True)
        out, _ = p.communicate()
        match = VERSION_REGEX.search(out)
        if not match:
            raise InstrumentError('Could not find streamline version.')
        version_tuple = tuple(map(int, match.group(1).split('.')))
        if version_tuple < (5, 17):
            raise InstrumentError('Need DS-5 v5.17 or greater; found v{}'.format(match.group(1)))

    def initialize(self, context):
        self.resource_getter = _instantiate(context.resolver)
        self.resource_getter.register()

        try:
            self.host_daemon_file = context.resolver.get(File(self, self.daemon))
            self.logger.debug('Using daemon from {}.'.format(self.host_daemon_file))
            self.device.killall(self.daemon)  # in case a version is already running
            self.device.install(self.host_daemon_file)
        except ResourceError:
            self.logger.debug('Using on-device daemon.')

        try:
            self.host_driver_file = context.resolver.get(File(self, self.driver))
            self.logger.debug('Using driver from {}.'.format(self.host_driver_file))
            self.device_driver_file = self.device.install(self.host_driver_file)
        except ResourceError:
            self.logger.debug('Using on-device driver.')

        try:
            self.configuration_file = (os.path.expanduser(self.configxml or '') or
                                       context.resolver.get(File(self, self.configuration_file_name)))
            self.logger.debug('Using {}'.format(self.configuration_file))
            self.on_device_config = self.device.path.join(self.device.working_directory, 'configuration.xml')
            shutil.copy(self.configuration_file, settings.meta_directory)
        except ResourceError:
            self.logger.debug('No configuration file was specfied.')

        caiman_path = subprocess.check_output('which caiman', shell=True).strip()  # pylint: disable=E1103
        self.session_file = os.path.join(context.host_working_directory, 'streamline_session.xml')
        with open(self.session_file, 'w') as wfh:
            if self.device.platform == "android":
                wfh.write(SESSION_TEXT_TEMPLATE.format('127.0.0.1', self.port, caiman_path))
            else:
                wfh.write(SESSION_TEXT_TEMPLATE.format(self.device.host, self.port, caiman_path))

        if self.configuration_file:
            self.device.push_file(self.configuration_file, self.on_device_config)
        self._initialize_daemon()

    def setup(self, context):
        self.capture_file = _f(os.path.join(context.output_directory, 'streamline', 'capture.apc'))
        self.report_file = _f(os.path.join(context.output_directory, 'streamline', 'streamline.csv'))

    def start(self, context):
        command = ['streamline', '-capture', self.session_file, '-output', self.capture_file]
        self.streamline = subprocess.Popen(command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           stdin=subprocess.PIPE,
                                           preexec_fn=os.setpgrp)
        outlogger = StreamLogger('streamline', self.streamline.stdout, klass=LineLogWriter)
        errlogger = StreamLogger('streamline', self.streamline.stderr, klass=LineLogWriter)
        outlogger.start()
        errlogger.start()

    def stop(self, context):
        os.killpg(self.streamline.pid, signal.SIGTERM)

    def update_result(self, context):
        if self.report:
            self.logger.debug('Creating report...')
            command = ['streamline', '-report', self.capture_file, '-output', self.report_file]
            command += self.report_options.split()
            _run_streamline_command(command)
            context.add_artifact('streamlinecsv', self.report_file, 'data')

    def teardown(self, context):
        self._kill_daemon()
        self.device.delete_file(self.on_device_config)

    def _check_has_valid_display(self):  # pylint: disable=R0201
        reason = None
        if os.name == 'posix' and not os.getenv('DISPLAY'):
            reason = 'DISPLAY is not set.'
        else:
            p = subprocess.Popen('xhost', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, error = p.communicate()
            if p.returncode:
                reason = 'Invalid DISPLAY; xhost returned: "{}".'.format(error.strip())  # pylint: disable=E1103
        if reason:
            raise InstrumentError('{}\nstreamline binary requires a valid display server to be running.'.format(reason))

    def _initialize_daemon(self):
        if self.device_driver_file:
            try:
                self.device.execute('insmod {}'.format(self.device_driver_file))
            except DeviceError, e:
                if 'File exists' not in e.message:
                    raise
                self.logger.debug('Driver was already installed.')
        self._start_daemon()
        if self.device.platform == "android":
            port_spec = 'tcp:{}'.format(self.port)
            self.device.forward_port(port_spec, port_spec)

    def _start_daemon(self):
        self.logger.debug('Starting gatord')
        self.device.killall('gatord', as_root=True)
        if self.configuration_file:
            command = '{} -c {}'.format(self.daemon, self.on_device_config)
        else:
            command = '{}'.format(self.daemon)

        self.daemon_process = self.device.execute(command, as_root=True, background=True)
        outlogger = StreamLogger('gatord', self.daemon_process.stdout)
        errlogger = StreamLogger('gatord', self.daemon_process.stderr, logging.ERROR)
        outlogger.start()
        errlogger.start()
        if self.daemon_process.poll() is not None:
            # If adb returned, something went wrong.
            raise InstrumentError('Could not start gatord.')

    def _kill_daemon(self):
        self.logger.debug('Killing daemon process.')
        self.daemon_process.kill()


def _run_streamline_command(command):
    streamline = subprocess.Popen(command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  stdin=subprocess.PIPE)
    output, error = streamline.communicate()
    LogWriter('streamline').write(output).close()
    LogWriter('streamline').write(error).close()
