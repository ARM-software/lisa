#    Copyright 2014-2015 ARM Limited
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


# pylint: disable=E1101,W0613
from __future__ import division
import os
import sys
import argparse
import shutil
import socket
import time
from datetime import datetime, timedelta

from zope.interface import implements
from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory, Protocol
from twisted.internet import reactor, interfaces
from twisted.internet.error import ConnectionLost, ConnectionDone


if __name__ == "__main__":  # for debugging
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from daqpower import log
from daqpower.config import DeviceConfiguration
from daqpower.common import DaqServerRequest, DaqServerResponse, Status
try:
    from daqpower.daq import DaqRunner, list_available_devices, CAN_ENUMERATE_DEVICES
    __import_error = None
except ImportError as e:
    # May be using debug mode.
    __import_error = e
    DaqRunner = None
    list_available_devices = lambda: ['Dev1']


class ProtocolError(Exception):
    pass


class DummyDaqRunner(object):
    """Dummy stub used when running in debug mode."""

    num_rows = 200

    @property
    def number_of_ports(self):
        return self.config.number_of_ports

    def __init__(self, config, output_directory):
        log.info('Creating runner with {} {}'.format(config, output_directory))
        self.config = config
        self.output_directory = output_directory
        self.is_running = False

    def start(self):
        import csv, random
        log.info('runner started')
        for i in xrange(self.config.number_of_ports):
            rows = [['power', 'voltage']] + [[random.gauss(1.0, 1.0), random.gauss(1.0, 0.1)]
                                             for _ in xrange(self.num_rows)]
            with open(self.get_port_file_path(self.config.labels[i]), 'wb') as wfh:
                writer = csv.writer(wfh)
                writer.writerows(rows)

        self.is_running = True

    def stop(self):
        self.is_running = False
        log.info('runner stopped')

    def get_port_file_path(self, port_id):
        if port_id in self.config.labels:
            return os.path.join(self.output_directory, '{}.csv'.format(port_id))
        else:
            raise Exception('Invalid port id: {}'.format(port_id))


class DaqServer(object):

    def __init__(self, base_output_directory):
        self.base_output_directory = os.path.abspath(base_output_directory)
        if os.path.isdir(self.base_output_directory):
            log.info('Using output directory: {}'.format(self.base_output_directory))
        else:
            log.info('Creating new output directory: {}'.format(self.base_output_directory))
            os.makedirs(self.base_output_directory)
        self.runner = None
        self.output_directory = None
        self.labels = None

    def configure(self, config_string):
        message = None
        if self.runner:
            message = 'Configuring a new session before previous session has been terminated.'
            log.warning(message)
            if self.runner.is_running:
                self.runner.stop()
        config = DeviceConfiguration.deserialize(config_string)
        config.validate()
        self.output_directory = self._create_output_directory()
        self.labels = config.labels
        log.info('Writing port files to {}'.format(self.output_directory))
        self.runner = DaqRunner(config, self.output_directory)
        return message

    def start(self):
        if self.runner:
            if not self.runner.is_running:
                self.runner.start()
            else:
                message = 'Calling start() before stop() has been called. Data up to this point will be lost.'
                log.warning(message)
                self.runner.stop()
                self.runner.start()
                return message
        else:
            raise ProtocolError('Start called before a session has been configured.')

    def stop(self):
        if self.runner:
            if self.runner.is_running:
                self.runner.stop()
            else:
                message = 'Attempting to stop() before start() was invoked.'
                log.warning(message)
                self.runner.stop()
                return message
        else:
            raise ProtocolError('Stop called before a session has been configured.')

    def list_devices(self):  # pylint: disable=no-self-use
        return list_available_devices()

    def list_ports(self):
        return self.labels

    def list_port_files(self):
        if not self.runner:
            raise ProtocolError('Attempting to list port files before session has been configured.')
        ports_with_files = []
        for port_id in self.labels:
            path = self.get_port_file_path(port_id)
            if os.path.isfile(path):
                ports_with_files.append(port_id)
        return ports_with_files

    def get_port_file_path(self, port_id):
        if not self.runner:
            raise ProtocolError('Attepting to get port file path before session has been configured.')
        return self.runner.get_port_file_path(port_id)

    def terminate(self):
        message = None
        if self.runner:
            if self.runner.is_running:
                message = 'Terminating session before runner has been stopped.'
                log.warning(message)
                self.runner.stop()
            self.runner = None
            if self.output_directory and os.path.isdir(self.output_directory):
                shutil.rmtree(self.output_directory)
            self.output_directory = None
            log.info('Session terminated.')
        else:  # Runner has not been created.
            message = 'Attempting to close session before it has been configured.'
            log.warning(message)
        return message

    def _create_output_directory(self):
        basename = datetime.now().strftime('%Y-%m-%d_%H%M%S%f')
        dirname = os.path.join(self.base_output_directory, basename)
        os.makedirs(dirname)
        return dirname

    def __del__(self):
        if self.runner:
            self.runner.stop()

    def __str__(self):
        return '({})'.format(self.base_output_directory)

    __repr__ = __str__


class DaqControlProtocol(LineReceiver):  # pylint: disable=W0223

    def __init__(self, daq_server):
        self.daq_server = daq_server
        self.factory = None

    def lineReceived(self, line):
        line = line.strip()
        log.info('Received: {}'.format(line))
        try:
            request = DaqServerRequest.deserialize(line)
        except Exception, e:  # pylint: disable=W0703
            # PyDAQmx exceptions use "mess" rather than the standard "message"
            # to pass errors...
            message = getattr(e, 'mess', e.message)
            self.sendError('Received bad request ({}: {})'.format(e.__class__.__name__, message))
        else:
            self.processRequest(request)

    def processRequest(self, request):
        try:
            if request.command == 'configure':
                self.configure(request)
            elif request.command == 'start':
                self.start(request)
            elif request.command == 'stop':
                self.stop(request)
            elif request.command == 'list_devices':
                self.list_devices(request)
            elif request.command == 'list_ports':
                self.list_ports(request)
            elif request.command == 'list_port_files':
                self.list_port_files(request)
            elif request.command == 'pull':
                self.pull_port_data(request)
            elif request.command == 'close':
                self.terminate(request)
            else:
                self.sendError('Received unknown command: {}'.format(request.command))
        except Exception, e:  # pylint: disable=W0703
            message = getattr(e, 'mess', e.message)
            self.sendError('{}: {}'.format(e.__class__.__name__, message))

    def configure(self, request):
        if 'config' in request.params:
            result = self.daq_server.configure(request.params['config'])
            if not result:
                self.sendResponse(Status.OK)
            else:
                self.sendResponse(Status.OKISH, message=result)
        else:
            self.sendError('Invalid config; config string not provided.')

    def start(self, request):
        result = self.daq_server.start()
        if not result:
            self.sendResponse(Status.OK)
        else:
            self.sendResponse(Status.OKISH, message=result)

    def stop(self, request):
        result = self.daq_server.stop()
        if not result:
            self.sendResponse(Status.OK)
        else:
            self.sendResponse(Status.OKISH, message=result)

    def pull_port_data(self, request):
        if 'port_id' in request.params:
            port_id = request.params['port_id']
            port_file = self.daq_server.get_port_file_path(port_id)
            if os.path.isfile(port_file):
                port = self._initiate_file_transfer(port_file)
                self.sendResponse(Status.OK, data={'port_number': port})
            else:
                self.sendError('File for port {} does not exist.'.format(port_id))
        else:
            self.sendError('Invalid pull request; port id not provided.')

    def list_devices(self, request):
        if CAN_ENUMERATE_DEVICES:
            devices = self.daq_server.list_devices()
            self.sendResponse(Status.OK, data={'devices': devices})
        else:
            message = "Server does not support DAQ device enumration"
            self.sendResponse(Status.OKISH, message=message)

    def list_ports(self, request):
        port_labels = self.daq_server.list_ports()
        self.sendResponse(Status.OK, data={'ports': port_labels})

    def list_port_files(self, request):
        port_labels = self.daq_server.list_port_files()
        self.sendResponse(Status.OK, data={'ports': port_labels})

    def terminate(self, request):
        status = Status.OK
        message = ''
        if self.factory.transfer_sessions:
            message = 'Terminating with file tranfer sessions in progress. '
            log.warning(message)
            for session in self.factory.transfer_sessions:
                self.factory.transferComplete(session)
        message += self.daq_server.terminate() or ''
        if message:
            status = Status.OKISH
        self.sendResponse(status, message)

    def sendError(self, message):
        log.error(message)
        self.sendResponse(Status.ERROR, message)

    def sendResponse(self, status, message=None, data=None):
        response = DaqServerResponse(status, message=message, data=data)
        self.sendLine(response.serialize())

    def sendLine(self, line):
        log.info('Responding: {}'.format(line))
        LineReceiver.sendLine(self, line.replace('\r\n', ''))

    def _initiate_file_transfer(self, filepath):
        sender_factory = FileSenderFactory(filepath, self.factory)
        connector = reactor.listenTCP(0, sender_factory)
        self.factory.transferInitiated(sender_factory, connector)
        return connector.getHost().port


class DaqFactory(Factory):

    protocol = DaqControlProtocol
    check_alive_period = 5 * 60
    max_transfer_lifetime = 30 * 60

    def __init__(self, server, cleanup_period=24 * 60 * 60, cleanup_after_days=5):
        self.server = server
        self.cleanup_period = cleanup_period
        self.cleanup_threshold = timedelta(cleanup_after_days)
        self.transfer_sessions = {}

    def buildProtocol(self, addr):
        proto = DaqControlProtocol(self.server)
        proto.factory = self
        reactor.callLater(self.check_alive_period, self.pulse)
        reactor.callLater(self.cleanup_period, self.perform_cleanup)
        return proto

    def clientConnectionLost(self, connector, reason):
        log.msg('client connection lost: {}.'.format(reason))
        if not isinstance(reason, ConnectionLost):
            log.msg('ERROR: Client terminated connection mid-transfer.')
            for session in self.transfer_sessions:
                self.transferComplete(session)

    def transferInitiated(self, session, connector):
        self.transfer_sessions[session] = (time.time(), connector)

    def transferComplete(self, session, reason='OK'):
        if reason != 'OK':
            log.error(reason)
        self.transfer_sessions[session][1].stopListening()
        del self.transfer_sessions[session]

    def pulse(self):
        """Close down any file tranfer sessions that have been open for too long."""
        current_time = time.time()
        for session in self.transfer_sessions:
            start_time, conn = self.transfer_sessions[session]
            if (current_time - start_time) > self.max_transfer_lifetime:
                message = '{} session on port {} timed out'
                self.transferComplete(session, message.format(session, conn.getHost().port))
        if self.transfer_sessions:
            reactor.callLater(self.check_alive_period, self.pulse)

    def perform_cleanup(self):
        """
        Cleanup and old uncollected data files to recover disk space.

        """
        log.msg('Performing cleanup of the output directory...')
        base_directory = self.server.base_output_directory
        current_time = datetime.now()
        for entry in os.listdir(base_directory):
            entry_path = os.path.join(base_directory, entry)
            entry_ctime = datetime.fromtimestamp(os.path.getctime(entry_path))
            existence_time = current_time - entry_ctime
            if existence_time > self.cleanup_threshold:
                log.debug('Removing {} (existed for {})'.format(entry, existence_time))
                shutil.rmtree(entry_path)
            else:
                log.debug('Keeping {} (existed for {})'.format(entry, existence_time))
        log.msg('Cleanup complete.')

    def __str__(self):
        return '<DAQ {}>'.format(self.server)

    __repr__ = __str__


class FileReader(object):

    implements(interfaces.IPushProducer)

    def __init__(self, filepath):
        self.fh = open(filepath)
        self.proto = None
        self.done = False
        self._paused = True

    def setProtocol(self, proto):
        self.proto = proto

    def resumeProducing(self):
        if not self.proto:
            raise ProtocolError('resumeProducing called with no protocol set.')
        self._paused = False
        try:
            while not self._paused:
                line = self.fh.next().rstrip('\n') + '\r\n'
                self.proto.transport.write(line)
        except StopIteration:
            log.debug('Sent everything.')
            self.stopProducing()

    def pauseProducing(self):
        self._paused = True

    def stopProducing(self):
        self.done = True
        self.fh.close()
        self.proto.transport.unregisterProducer()
        self.proto.transport.loseConnection()


class FileSenderProtocol(Protocol):

    def __init__(self, reader):
        self.reader = reader
        self.factory = None

    def connectionMade(self):
        self.transport.registerProducer(self.reader, True)
        self.reader.resumeProducing()

    def connectionLost(self, reason=ConnectionDone):
        if self.reader.done:
            self.factory.transferComplete()
        else:
            self.reader.pauseProducing()
            self.transport.unregisterProducer()


class FileSenderFactory(Factory):

    @property
    def done(self):
        if self.reader:
            return self.reader.done
        else:
            return None

    def __init__(self, path, owner):
        self.path = os.path.abspath(path)
        self.reader = None
        self.owner = owner

    def buildProtocol(self, addr):
        if not self.reader:
            self.reader = FileReader(self.path)
        proto = FileSenderProtocol(self.reader)
        proto.factory = self
        self.reader.setProtocol(proto)
        return proto

    def transferComplete(self):
        self.owner.transferComplete(self)

    def __hash__(self):
        return hash(self.path)

    def __str__(self):
        return '<FileSender {}>'.format(self.path)

    __repr__ = __str__


def run_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='Working directory', metavar='DIR', default='.')
    parser.add_argument('-p', '--port', help='port the server will listen on.',
                        metavar='PORT', default=45677, type=int)
    parser.add_argument('-c', '--cleanup-after', type=int, default=5, metavar='DAYS',
                        help="""
                        Sever will perodically clean up data files that are older than the number of
                        days specfied by this parameter.
                        """)
    parser.add_argument('--cleanup-period', type=int, default=1, metavar='DAYS',
                        help='Specifies how ofte the server will attempt to clean up old files.')
    parser.add_argument('--debug', help='Run in debug mode (no DAQ connected).',
                        action='store_true', default=False)
    parser.add_argument('--verbose', help='Produce verobose output.', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        global DaqRunner  # pylint: disable=W0603
        DaqRunner = DummyDaqRunner
    else:
        if not DaqRunner:
            raise __import_error  # pylint: disable=raising-bad-type
    if args.verbose or args.debug:
        log.start_logging('DEBUG')
    else:
        log.start_logging('INFO')

    # days to seconds
    cleanup_period = args.cleanup_period * 24 * 60 * 60

    server = DaqServer(args.directory)
    factory = DaqFactory(server, cleanup_period, args.cleanup_after)
    reactor.listenTCP(args.port, factory).getHost()
    try:
        hostname = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        hostname = 'localhost'
    log.info('Listening on {}:{}'.format(hostname, args.port))
    reactor.run()


if __name__ == "__main__":
    run_server()
