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


# pylint: disable=E1101,E1103
import os
import sys

from twisted.internet import reactor
from twisted.internet.protocol import Protocol, ClientFactory, ReconnectingClientFactory
from twisted.internet.error import ConnectionLost, ConnectionDone
from twisted.protocols.basic import LineReceiver

if __name__ == '__main__':  # for debugging
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from daqpower import log
from daqpower.common import DaqServerRequest, DaqServerResponse, Status
from daqpower.config import get_config_parser


__all__ = ['execute_command', 'run_send_command', 'Status']


class Command(object):

    def __init__(self, name, **params):
        self.name = name
        self.params = params


class CommandResult(object):

    def __init__(self):
        self.status = None
        self.message = None
        self.data = None

    def __str__(self):
        return '{} {}'.format(self.status, self.message)


class CommandExecutorProtocol(Protocol):

    def __init__(self, command, timeout=10, retries=1):
        self.command = command
        self.sent_request = None
        self.waiting_for_response = False
        self.keep_going = None
        self.ports_to_pull = None
        self.factory = None
        self.timeoutCallback = None
        self.timeout = timeout
        self.retries = retries
        self.retry_count = 0

    def connectionMade(self):
        if self.command.name == 'get_data':
            self.sendRequest('list_port_files')
        else:
            self.sendRequest(self.command.name, **self.command.params)

    def connectionLost(self, reason=ConnectionDone):
        if isinstance(reason, ConnectionLost):
            self.errorOut('connection lost: {}'.format(reason))
        elif self.waiting_for_response:
            self.errorOut('Server closed connection without sending a response.')
        else:
            log.debug('connection terminated.')

    def sendRequest(self, command, **params):
        self.sent_request = DaqServerRequest(command, params)
        request_string = self.sent_request.serialize()
        log.debug('sending request: {}'.format(request_string))
        self.transport.write(''.join([request_string, '\r\n']))
        self.timeoutCallback = reactor.callLater(self.timeout, self.requestTimedOut)
        self.waiting_for_response = True

    def dataReceived(self, data):
        self.keep_going = False
        if self.waiting_for_response:
            self.waiting_for_response = False
            self.timeoutCallback.cancel()
            try:
                response = DaqServerResponse.deserialize(data)
            except Exception, e:  # pylint: disable=W0703
                self.errorOut('Invalid response: {} ({})'.format(data, e))
            else:
                if response.status != Status.ERROR:
                    self.processResponse(response)  # may set self.keep_going
                    if not self.keep_going:
                        self.commandCompleted(response.status, response.message, response.data)
                else:
                    self.errorOut(response.message)
        else:
            self.errorOut('unexpected data received: {}\n'.format(data))

    def processResponse(self, response):
        if self.sent_request.command in ['list_ports', 'list_port_files']:
            self.processPortsResponse(response)
        elif self.sent_request.command == 'list_devices':
            self.processDevicesResponse(response)
        elif self.sent_request.command == 'pull':
            self.processPullResponse(response)

    def processPortsResponse(self, response):
        if 'ports' not in response.data:
            self.errorOut('Response did not containt ports data: {} ({}).'.format(response, response.data))
        ports = response.data['ports']
        response.data = ports
        if self.command.name == 'get_data':
            if ports:
                self.ports_to_pull = ports
                self.sendPullRequest(self.ports_to_pull.pop())
            else:
                response.status = Status.OKISH
                response.message = 'No ports were returned.'

    def processDevicesResponse(self, response):
        if response.status == Status.OK:
            if 'devices' not in response.data:
                self.errorOut('Response did not containt devices data: {} ({}).'.format(response, response.data))
            devices = response.data['devices']
            response.data = devices

    def sendPullRequest(self, port_id):
        self.sendRequest('pull', port_id=port_id)
        self.keep_going = True

    def processPullResponse(self, response):
        if 'port_number' not in response.data:
            self.errorOut('Response does not contain port number: {} ({}).'.format(response, response.data))
        port_number = response.data.pop('port_number')
        filename = self.sent_request.params['port_id'] + '.csv'
        self.factory.initiateFileTransfer(filename, port_number)
        if self.ports_to_pull:
            self.sendPullRequest(self.ports_to_pull.pop())

    def commandCompleted(self, status, message=None, data=None):
        self.factory.result.status = status
        self.factory.result.message = message
        self.factory.result.data = data
        self.transport.loseConnection()

    def requestTimedOut(self):
        self.retry_count += 1
        if self.retry_count > self.retries:
            self.errorOut("Request timed out; server failed to respond.")
        else:
            log.debug('Retrying...')
            self.connectionMade()

    def errorOut(self, message):
        self.factory.errorOut(message)


class CommandExecutorFactory(ClientFactory):

    protocol = CommandExecutorProtocol
    wait_delay = 1

    def __init__(self, config, command, timeout=10, retries=1):
        self.config = config
        self.command = command
        self.timeout = timeout
        self.retries = retries
        self.result = CommandResult()
        self.done = False
        self.transfers_in_progress = {}
        if command.name == 'get_data':
            if 'output_directory' not in command.params:
                self.errorOut('output_directory not specifed for get_data command.')
            self.output_directory = command.params['output_directory']
            if not os.path.isdir(self.output_directory):
                log.debug('Creating output directory {}'.format(self.output_directory))
                os.makedirs(self.output_directory)

    def buildProtocol(self, addr):
        protocol = CommandExecutorProtocol(self.command, self.timeout, self.retries)
        protocol.factory = self
        return protocol

    def initiateFileTransfer(self, filename, port):
        log.debug('Downloading {} from port {}'.format(filename, port))
        filepath = os.path.join(self.output_directory, filename)
        session = FileReceiverFactory(filepath, self)
        connector = reactor.connectTCP(self.config.host, port, session)
        self.transfers_in_progress[session] = connector

    def transferComplete(self, session):
        connector = self.transfers_in_progress[session]
        log.debug('Transfer on port {} complete.'.format(connector.port))
        del self.transfers_in_progress[session]

    def clientConnectionLost(self, connector, reason):
        if self.transfers_in_progress:
            log.debug('Waiting for the transfer(s) to complete.')
        self.waitForTransfersToCompleteAndExit()

    def clientConnectionFailed(self, connector, reason):
        self.result.status = Status.ERROR
        self.result.message = 'Could not connect to server.'
        self.waitForTransfersToCompleteAndExit()

    def waitForTransfersToCompleteAndExit(self):
        if self.transfers_in_progress:
            reactor.callLater(self.wait_delay, self.waitForTransfersToCompleteAndExit)
        else:
            log.debug('Stopping the reactor.')
            reactor.stop()

    def errorOut(self, message):
        self.result.status = Status.ERROR
        self.result.message = message
        reactor.crash()

    def __str__(self):
        return '<CommandExecutorProtocol {}>'.format(self.command.name)

    __repr__ = __str__


class FileReceiver(LineReceiver):  # pylint: disable=W0223

    def __init__(self, path):
        self.path = path
        self.fh = None
        self.factory = None

    def connectionMade(self):
        if os.path.isfile(self.path):
            log.warning('overriding existing file.')
            os.remove(self.path)
        self.fh = open(self.path, 'w')

    def connectionLost(self, reason=ConnectionDone):
        if self.fh:
            self.fh.close()

    def lineReceived(self, line):
        line = line.rstrip('\r\n') + '\n'
        self.fh.write(line)


class FileReceiverFactory(ReconnectingClientFactory):

    def __init__(self, path, owner):
        self.path = path
        self.owner = owner

    def buildProtocol(self, addr):
        protocol = FileReceiver(self.path)
        protocol.factory = self
        self.resetDelay()
        return protocol

    def clientConnectionLost(self, conector, reason):
        if isinstance(reason, ConnectionLost):
            log.error('Connection lost: {}'.format(reason))
            ReconnectingClientFactory.clientConnectionLost(self, conector, reason)
        else:
            self.owner.transferComplete(self)

    def clientConnectionFailed(self, conector, reason):
        if isinstance(reason, ConnectionLost):
            log.error('Connection failed: {}'.format(reason))
            ReconnectingClientFactory.clientConnectionFailed(self, conector, reason)

    def __str__(self):
        return '<FileReceiver {}>'.format(self.path)

    __repr__ = __str__


def execute_command(server_config, command, **kwargs):
    before_fds = _get_open_fds()  # see the comment in the finally clause below
    if isinstance(command, basestring):
        command = Command(command, **kwargs)
    timeout = 300 if command.name in ['stop', 'pull'] else 10
    factory = CommandExecutorFactory(server_config, command, timeout)

    # reactors aren't designed to be re-startable. In order to be
    # able to call execute_command multiple times, we need to froce
    # re-installation of the reactor; hence this hackery.
    # TODO: look into implementing restartable reactors. According to the
    #       Twisted FAQ, there is no good reason why there isn't one:
    #       http://twistedmatrix.com/trac/wiki/FrequentlyAskedQuestions#WhycanttheTwistedsreactorberestarted
    from twisted.internet import default
    del sys.modules['twisted.internet.reactor']
    default.install()
    global reactor  # pylint: disable=W0603
    reactor = sys.modules['twisted.internet.reactor']

    try:
        reactor.connectTCP(server_config.host, server_config.port, factory)
        reactor.run()
        return factory.result
    finally:
        # re-startable reactor hack part 2.
        # twisted hijacks SIGINT and doesn't bother to un-hijack it when the reactor
        # stops. So we have to do it for it *rolls eye*.
        import signal
        signal.signal(signal.SIGINT, signal.default_int_handler)
        # OK, the reactor is also leaking file descriptors. Tracking down all
        # of them is non trivial, so instead we're just comparing the before
        # and after lists of open FDs for the current process, and closing all
        # new ones, as execute_command should never leave anything open after
        # it exits (even when downloading data files from the server).
        # TODO: This is way too hacky even compared to the rest of this function.
        #       Additionally, the current implementation ties this to UNIX,
        #       so in the long run, we need to do this properly and get the FDs
        #       from the reactor.
        after_fds = _get_open_fds()
        for fd in after_fds - before_fds:
            try:
                os.close(int(fd[1:]))
            except OSError:
                pass
        # Below is the alternative code that gets FDs from the reactor, however
        # at the moment it doesn't seem to get everything, which is why code
        # above is used instead.
        #for fd in readtor._selectables:
        #    os.close(fd)
        #reactor._poller.close()


def _get_open_fds():
    if os.name == 'posix':
        import subprocess
        pid = os.getpid()
        procs = subprocess.check_output(["lsof", '-w', '-Ff', "-p", str(pid)])
        return set(procs.split())
    else:
        # TODO: Implement the Windows equivalent.
        return []


def run_send_command():
    """Main entry point when running as a script -- should not be invoked form another module."""
    parser = get_config_parser()
    parser.add_argument('command')
    parser.add_argument('-o', '--output-directory', metavar='DIR', default='.',
                        help='Directory used to output data files (defaults to the current directory).')
    parser.add_argument('--verbose', help='Produce verobose output.', action='store_true', default=False)
    args = parser.parse_args()
    if not args.device_config.labels:
        args.device_config.labels = ['PORT_{}'.format(i) for i in xrange(len(args.device_config.resistor_values))]

    if args.verbose:
        log.start_logging('DEBUG')
    else:
        log.start_logging('INFO', fmt='%(levelname)-8s %(message)s')

    if args.command == 'configure':
        args.device_config.validate()
        command = Command(args.command, config=args.device_config)
    elif args.command == 'get_data':
        command = Command(args.command, output_directory=args.output_directory)
    else:
        command = Command(args.command)

    result = execute_command(args.server_config, command)
    print result
    if result.data:
        print result.data


if __name__ == '__main__':
    run_send_command()
