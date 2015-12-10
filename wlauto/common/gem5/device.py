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

# Original implementation by Rene de Jong. Updated by Sascha Bischoff.

# pylint: disable=E1101

import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pexpect import EOF, TIMEOUT, pxssh

from wlauto import settings, Parameter
from wlauto.core.resource import NO_ONE
from wlauto.common.resources import Executable
from wlauto.core import signal as sig
from wlauto.exceptions import DeviceError
from wlauto.utils import ssh, types


class BaseGem5Device(object):
    """
    Base implementation for a gem5-based device

    This class is used as the base class for OS-specific devices such as the
    G3m5LinuxDevice and the Gem5AndroidDevice. The majority of the gem5-specific
    functionality is included here.

    Note: When inheriting from this class, make sure to inherit from this class
    prior to inheriting from the OS-specific class, i.e. LinuxDevice, to ensure
    that the methods are correctly overridden.
    """
    # gem5 can be very slow. Hence, we use some very long timeouts!
    delay = 3600
    long_delay = 3 * delay
    ready_timeout = long_delay
    default_timeout = delay

    platform = None
    path_module = 'posixpath'

    parameters = [
        Parameter('gem5_binary', kind=str, default='./build/ARM/gem5.fast',
                  mandatory=False, description="Command used to execute gem5. "
                  "Adjust according to needs."),
        Parameter('gem5_args', kind=types.arguments, mandatory=True,
                  description="Command line passed to the gem5 simulation. This"
                  " command line is used to set up the simulated system, and "
                  "should be the same as used for a standard gem5 simulation "
                  "without workload automation. Note that this is simulation "
                  "script specific and will hence need to be tailored to each "
                  "particular use case."),
        Parameter('gem5_vio_args', kind=types.arguments, mandatory=True,
                  constraint=lambda x: "{}" in str(x),
                  description="gem5 VirtIO command line used to enable the "
                  "VirtIO device in the simulated system. At the very least, "
                  "the root parameter of the VirtIO9PDiod device must be "
                  "exposed on the command line. Please set this root mount to "
                  "{}, as it will be replaced with the directory used by "
                  "Workload Automation at runtime."),
        Parameter('temp_dir', kind=str, default='/tmp',
                  description="Temporary directory used to pass files into the "
                  "gem5 simulation. Workload Automation will automatically "
                  "create a directory in this folder, and will remove it again "
                  "once the simulation completes."),
        Parameter('checkpoint', kind=bool, default=False,
                  mandatory=False, description="This parameter "
                  "tells Workload Automation to create a checkpoint of the "
                  "simulated system once the guest system has finished booting."
                  " This checkpoint can then be used at a later stage by other "
                  "WA runs to avoid booting the guest system a second time. Set"
                  " to True to take a checkpoint of the simulated system post "
                  "boot."),
        Parameter('run_delay', kind=int, default=0, mandatory=False,
                  constraint=lambda x: x >= 0,
                  description="This sets the time that the "
                  "system should sleep in the simulated system prior to "
                  "running and workloads or taking checkpoints. This allows "
                  "the system to quieten down prior to running the workloads. "
                  "When this is combined with the checkpoint_post_boot"
                  " option, it allows the checkpoint to be created post-sleep,"
                  " and therefore the set of workloads resuming from this "
                  "checkpoint will not be required to sleep.")
    ]

    @property
    def is_rooted(self):  # pylint: disable=R0201
        # gem5 is always rooted
        return True

    # pylint: disable=E0203
    def __init__(self):
        self.logger = logging.getLogger('gem5Device')

        # The gem5 subprocess
        self.gem5 = None
        self.gem5_port = -1
        self.gem5outdir = os.path.join(settings.output_directory, "gem5")
        self.m5_path = 'm5'
        self.stdout_file = None
        self.stderr_file = None
        self.stderr_filename = None
        self.sckt = None

        # Find the first one that does not exist. Ensures that we do not re-use
        # the directory used by someone else.
        for i in xrange(sys.maxint):
            directory = os.path.join(self.temp_dir, "wa_{}".format(i))
            try:
                os.stat(directory)
                continue
            except OSError:
                break
        self.temp_dir = directory
        self.logger.debug("Using {} as the temporary directory.".format(self.temp_dir))

        # Start the gem5 simulation when WA starts a run using a signal.
        sig.connect(self.init_gem5, sig.RUN_START)

    def validate(self):
        # Assemble the virtio args
        self.gem5_vio_args = str(self.gem5_vio_args).format(self.temp_dir)  # pylint: disable=W0201
        self.logger.debug("gem5 VirtIO command: {}".format(self.gem5_vio_args))

    def init_gem5(self, _):
        """
        Start gem5, find out the telnet port and connect to the simulation.

        We first create the temporary directory used by VirtIO to pass files
        into the simulation, as well as the gem5 output directory.We then create
        files for the standard output and error for the gem5 process. The gem5
        process then is started.
        """
        self.logger.info("Creating temporary directory: {}".format(self.temp_dir))
        os.mkdir(self.temp_dir)
        os.mkdir(self.gem5outdir)

        # We need to redirect the standard output and standard error for the
        # gem5 process to a file so that we can debug when things go wrong.
        f = os.path.join(self.gem5outdir, 'stdout')
        self.stdout_file = open(f, 'w')
        f = os.path.join(self.gem5outdir, 'stderr')
        self.stderr_file = open(f, 'w')
        # We need to keep this so we can check which port to use for the telnet
        # connection.
        self.stderr_filename = f

        self.start_gem5()

    def start_gem5(self):
        """
        Starts the gem5 simulator, and parses the output to get the telnet port.
        """
        self.logger.info("Starting the gem5 simulator")

        command_line = "{} --outdir={}/gem5 {} {}".format(self.gem5_binary,
                                                          settings.output_directory,
                                                          self.gem5_args,
                                                          self.gem5_vio_args)
        self.logger.debug("gem5 command line: {}".format(command_line))
        self.gem5 = subprocess.Popen(command_line.split(),
                                     stdout=self.stdout_file,
                                     stderr=self.stderr_file)

        while self.gem5_port == -1:
            # Check that gem5 is running!
            if self.gem5.poll():
                raise DeviceError("The gem5 process has crashed with error code {}!".format(self.gem5.poll()))

            # Open the stderr file
            f = open(self.stderr_filename, 'r')
            for line in f:
                m = re.search(r"Listening\ for\ system\ connection\ on\ port\ (?P<port>\d+)", line)
                if m:
                    port = int(m.group('port'))
                    if port >= 3456 and port < 5900:
                        self.gem5_port = port
                        f.close()
                        break
            else:
                time.sleep(1)
            f.close()

    def connect(self):  # pylint: disable=R0912,W0201
        """
        Connect to the gem5 simulation and wait for Android to boot. Then,
        create checkpoints, and mount the VirtIO device.
        """
        self.connect_gem5()

        self.wait_for_boot()

        if self.run_delay:
            self.logger.info("Sleeping for {} seconds in the guest".format(self.run_delay))
            self.gem5_shell("sleep {}".format(self.run_delay))

        if self.checkpoint:
            self.checkpoint_gem5()

        self.mount_virtio()
        self.logger.info("Creating the working directory in the simulated system")
        self.gem5_shell('mkdir -p {}'.format(self.working_directory))
        self._is_ready = True  # pylint: disable=W0201

    def wait_for_boot(self):
        pass

    def connect_gem5(self):  # pylint: disable=R0912
        """
        Connect to the telnet port of the gem5 simulation.

        We connect, and wait for the prompt to be found. We do not use a timeout
        for this, and wait for the prompt in a while loop as the gem5 simulation
        can take many hours to reach a prompt when booting the system. We also
        inject some newlines periodically to try and force gem5 to show a
        prompt. Once the prompt has been found, we replace it with a unique
        prompt to ensure that we are able to match it properly. We also disable
        the echo as this simplifies parsing the output when executing commands
        on the device.
        """
        self.logger.info("Connecting to the gem5 simulation on port {}".format(self.gem5_port))
        host = socket.gethostname()
        port = self.gem5_port

        # Connect to the gem5 telnet port. Use a short timeout here.
        attempts = 0
        while attempts < 10:
            attempts += 1
            try:
                self.sckt = ssh.TelnetConnection()
                self.sckt.login(host, 'None', port=port, auto_prompt_reset=False,
                                login_timeout=10)
                break
            except pxssh.ExceptionPxssh:
                pass
        else:
            self.gem5.kill()
            raise DeviceError("Failed to connect to the gem5 telnet session.")

        self.logger.info("Connected! Waiting for prompt...")

        # We need to find the prompt. It might be different if we are resuming
        # from a checkpoint. Therefore, we test multiple options here.
        prompt_found = False
        while not prompt_found:
            try:
                self.login_to_device()
            except TIMEOUT:
                pass
            try:
                # Try and force a prompt to be shown
                self.sckt.send('\n')
                self.sckt.expect([r'# ', self.sckt.UNIQUE_PROMPT, r'\[PEXPECT\][\\\$\#]+ '], timeout=60)
                prompt_found = True
            except TIMEOUT:
                pass

        self.logger.info("Setting unique prompt...")

        self.sckt.set_unique_prompt()
        self.sckt.prompt()
        self.logger.info("Prompt found and replaced with a unique string")

        # We check that the prompt is what we think it should be. If not, we
        # need to update the regex we use to match.
        self.find_prompt()

        self.sckt.setecho(False)
        self.sync_gem5_shell()
        self.resize_shell()

    def get_properties(self, context):  # pylint: disable=R0801
        """ Get the property files from the device """
        for propfile in self.property_files:
            try:
                normname = propfile.lstrip(self.path.sep).replace(self.path.sep, '.')
                outfile = os.path.join(context.host_working_directory, normname)
                if self.is_file(propfile):
                    self.execute('cat {} > {}'.format(propfile, normname))
                    self.pull_file(normname, outfile)
                elif self.is_directory(propfile):
                    self.get_directory(context, propfile)
                    continue
                else:
                    continue
            except DeviceError:
                # We pull these files "opportunistically", so if a pull fails
                # (e.g. we don't have permissions to read the file), just note
                # it quietly (not as an error/warning) and move on.
                self.logger.debug('Could not pull property file "{}"'.format(propfile))
        return {}

    def get_directory(self, context, directory):
        """ Pull a directory from the device """
        normname = directory.lstrip(self.path.sep).replace(self.path.sep, '.')
        outdir = os.path.join(context.host_working_directory, normname)
        temp_file = os.path.join(context.host_working_directory, "{}.tar".format(normname))
        # Check that the folder exists
        self.gem5_shell("ls -la {}".format(directory))
        # Compress the folder
        try:
            self.gem5_shell("{} tar -cvf {}.tar {}".format(self.busybox, normname, directory))
        except DeviceError:
            self.logger.debug("Failed to run tar command on device! Not pulling {}".format(directory))
            return
        self.pull_file(normname, temp_file)
        f = tarfile.open(temp_file, 'r')
        os.mkdir(outdir)
        f.extractall(outdir)
        os.remove(temp_file)

    def get_pids_of(self, process_name):
        """ Returns a list of PIDs of all processes with the specified name. """
        result = self.gem5_shell('ps | {} grep {}'.format(self.busybox, process_name),
                                 check_exit_code=False).strip()
        if result and 'not found' not in result and len(result.split('\n')) > 2:
            return [int(x.split()[1]) for x in result.split('\n')]
        else:
            return []

    def find_prompt(self):
        prompt = r'\[PEXPECT\][\\\$\#]+ '
        synced = False
        while not synced:
            self.sckt.send('\n')
            i = self.sckt.expect([prompt, self.sckt.UNIQUE_PROMPT, r'[\$\#] '], timeout=self.delay)
            if i == 0:
                synced = True
            elif i == 1:
                prompt = self.sckt.UNIQUE_PROMPT
                synced = True
            else:
                prompt = re.sub(r'\$', r'\\\$', self.sckt.before.strip() + self.sckt.after.strip())
                prompt = re.sub(r'\#', r'\\\#', prompt)
                prompt = re.sub(r'\[', r'\[', prompt)
                prompt = re.sub(r'\]', r'\]', prompt)

        self.sckt.PROMPT = prompt

    def close(self):
        if self._logcat_poller:
            self._logcat_poller.stop()

    def reset(self):
        self.logger.warn("Attempt to restart the gem5 device. This is not "
                         "supported!")

    # pylint: disable=unused-argument
    def push_file(self, source, dest, **kwargs):
        """
        Push a file to the gem5 device using VirtIO

        The file to push to the device is copied to the temporary directory on
        the host, before being copied within the simulation to the destination.
        Checks, in the form of 'ls' with error code checking, are performed to
        ensure that the file is copied to the destination.
        """
        filename = os.path.basename(source)
        self.logger.debug("Pushing {} to device.".format(source))
        self.logger.debug("temp_dir: {}".format(self.temp_dir))
        self.logger.debug("dest: {}".format(dest))
        self.logger.debug("filename: {}".format(filename))

        # We need to copy the file to copy to the temporary directory
        self.move_to_temp_dir(source)

        # Back to the gem5 world
        self.gem5_shell("ls -al /mnt/obb/{}".format(filename))
        if self.busybox:
            self.gem5_shell("{} cp /mnt/obb/{} {}".format(self.busybox, filename, dest))
        else:
            self.gem5_shell("cat /mnt/obb/{} > {}".format(filename, dest))
        self.gem5_shell("sync")
        self.gem5_shell("ls -al {}".format(dest))
        self.gem5_shell("ls -al /mnt/obb/")
        self.logger.debug("Push complete.")

    # pylint: disable=unused-argument
    def pull_file(self, source, dest, **kwargs):
        """
        Pull a file from the gem5 device using m5 writefile

        The file is copied to the local directory within the guest as the m5
        writefile command assumes that the file is local. The file is then
        written out to the host system using writefile, prior to being moved to
        the destination on the host.
        """
        filename = os.path.basename(source)

        self.logger.debug("pull_file {} {}".format(source, filename))
        # We don't check the exit code here because it is non-zero if the source
        # and destination are the same. The ls below will cause an error if the
        # file was not where we expected it to be.
        self.gem5_shell("{} cp {} {}".format(self.busybox, source, filename),
                        check_exit_code=False)
        self.gem5_shell("sync")
        self.gem5_shell("ls -la {}".format(filename))
        self.logger.debug('Finished the copy in the simulator')
        self.gem5_util("writefile {}".format(filename))

        if 'cpu' not in filename:
            while not os.path.exists(os.path.join(self.gem5outdir, filename)):
                time.sleep(1)

        # Perform the local move
        shutil.move(os.path.join(self.gem5outdir, filename), dest)
        self.logger.debug("Pull complete.")

    # pylint: disable=unused-argument
    def delete_file(self, filepath, **kwargs):
        """ Delete a file on the device """
        self._check_ready()
        self.gem5_shell("rm '{}'".format(filepath))

    def file_exists(self, filepath):
        """ Check if a file exists """
        self._check_ready()
        output = self.gem5_shell('if [ -e \'{}\' ]; then echo 1; else echo 0; fi'.format(filepath))
        try:
            if int(output):
                return True
        except ValueError:
            # If we cannot process the output, assume that there is no file
            pass
        return False

    def disconnect(self):
        """
        Close and disconnect from the gem5 simulation. Additionally, we remove
        the temporary directory used to pass files into the simulation.
        """
        self.logger.info("Gracefully terminating the gem5 simulation.")
        try:
            self.gem5_util("exit")
            self.gem5.wait()
        except EOF:
            pass
        self.logger.info("Removing the temporary directory")
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            self.logger.warn("Failed to remove the temporary directory!")

    # gem5 might be slow. Hence, we need to make the ping timeout very long.
    def ping(self):
        self.logger.debug("Pinging gem5 to see if it is still alive")
        self.gem5_shell('ls /', timeout=self.longdelay)

    # Additional Android-specific methods.
    def forward_port(self, _):  # pylint: disable=R0201
        raise DeviceError('we do not need forwarding')

    # gem5 should dump out a framebuffer. We can use this if it exists. Failing
    # that, fall back to the parent class implementation.
    def capture_screen(self, filepath):
        file_list = os.listdir(self.gem5outdir)
        screen_caps = []
        for f in file_list:
            if '.bmp' in f:
                screen_caps.append(f)

        if len(screen_caps) == 1:
            # Bail out if we do not have image, and resort to the slower, built
            # in method.
            try:
                import Image
                gem5_image = os.path.join(self.gem5outdir, screen_caps[0])
                temp_image = os.path.join(self.gem5outdir, "file.png")
                im = Image.open(gem5_image)
                im.save(temp_image, "PNG")
                shutil.copy(temp_image, filepath)
                os.remove(temp_image)
                self.logger.debug("capture_screen: using gem5 screencap")
                return True
            except (shutil.Error, ImportError, IOError):
                pass
        return False

    # pylint: disable=W0613
    def execute(self, command, timeout=1000, check_exit_code=True, background=False,
                as_root=False, busybox=False, **kwargs):
        self._check_ready()
        if as_root and not self.is_rooted:
            raise DeviceError('Attempting to execute "{}" as root on unrooted device.'.format(command))
        if busybox:
            if not self.is_rooted:
                raise DeviceError('Attempting to execute "{}" with busybox. '.format(command) +
                                  'Busybox can only be deployed to rooted devices.')
            command = ' '.join([self.busybox, command])
        if background:
            self.logger.debug("Attempt to execute in background. Not supported "
                              "in gem5, hence ignored.")
        return self.gem5_shell(command, as_root=as_root)

    # Internal methods: do not use outside of the class.

    def _check_ready(self):
        """
        Check if the device is ready.

        As this is gem5, we just assume that the device is ready once we have
        connected to the gem5 simulation, and updated the prompt.
        """
        if not self._is_ready:
            raise DeviceError('Device not ready.')

    def gem5_shell(self, command, as_root=False, timeout=None, check_exit_code=True, sync=True):  # pylint: disable=R0912
        """
        Execute a command in the gem5 shell

        This wraps the telnet connection to gem5 and processes the raw output.

        This method waits for the shell to return, and then will try and
        separate the output from the command from the command itself. If this
        fails, warn, but continue with the potentially wrong output.

        The exit code is also checked by default, and non-zero exit codes will
        raise a DeviceError.
        """
        conn = self.sckt
        if sync:
            self.sync_gem5_shell()

        self.logger.debug("gem5_shell command: {}".format(command))

        # Send the actual command
        conn.send("{}\n".format(command))

        # Wait for the response. We just sit here and wait for the prompt to
        # appear, as gem5 might take a long time to provide the output. This
        # avoids timeout issues.
        command_index = -1
        while command_index == -1:
            if conn.prompt():
                output = re.sub(r' \r([^\n])', r'\1', conn.before)
                output = re.sub(r'[\b]', r'', output)
                # Deal with line wrapping
                output = re.sub(r'[\r].+?<', r'', output)
                command_index = output.find(command)

                # If we have -1, then we cannot match the command, but the
                # prompt has returned. Hence, we have a bit of an issue. We
                # warn, and return the whole output.
                if command_index == -1:
                    self.logger.warn("gem5_shell: Unable to match command in "
                                     "command output. Expect parsing errors!")
                    command_index = 0

        output = output[command_index + len(command):].strip()

        # It is possible that gem5 will echo the command. Therefore, we need to
        # remove that too!
        command_index = output.find(command)
        if command_index != -1:
            output = output[command_index + len(command):].strip()

        self.logger.debug("gem5_shell output: {}".format(output))

        # We get a second prompt. Hence, we need to eat one to make sure that we
        # stay in sync. If we do not do this, we risk getting out of sync for
        # slower simulations.
        self.sckt.expect([self.sckt.UNIQUE_PROMPT, self.sckt.PROMPT], timeout=self.delay)

        if check_exit_code:
            exit_code_text = self.gem5_shell('echo $?', as_root=as_root,
                                             timeout=timeout, check_exit_code=False,
                                             sync=False)
            try:
                exit_code = int(exit_code_text.split()[0])
                if exit_code:
                    message = 'Got exit code {}\nfrom: {}\nOUTPUT: {}'
                    raise DeviceError(message.format(exit_code, command, output))
            except (ValueError, IndexError):
                self.logger.warning('Could not get exit code for "{}",\ngot: "{}"'.format(command, exit_code_text))

        return output

    def gem5_util(self, command):
        """ Execute a gem5 utility command using the m5 binary on the device """
        self.gem5_shell('{} {}'.format(self.m5_path, command))

    def sync_gem5_shell(self):
        """
        Synchronise with the gem5 shell.

        Write some unique text to the gem5 device to allow us to synchronise
        with the shell output. We actually get two prompts so we need to match
        both of these.
        """
        self.logger.debug("Sending Sync")
        self.sckt.send("echo \*\*sync\*\*\n")
        self.sckt.expect(r"\*\*sync\*\*", timeout=self.delay)
        self.sckt.expect([self.sckt.UNIQUE_PROMPT, self.sckt.PROMPT], timeout=self.delay)
        self.sckt.expect([self.sckt.UNIQUE_PROMPT, self.sckt.PROMPT], timeout=self.delay)

    def resize_shell(self):
        """
        Resize the shell to avoid line wrapping issues.

        """
        # Try and avoid line wrapping as much as possible. Don't check the error
        # codes from these command because some of them WILL fail.
        self.gem5_shell('stty columns 1024', check_exit_code=False)
        self.gem5_shell('{} stty columns 1024'.format(self.busybox), check_exit_code=False)
        self.gem5_shell('stty cols 1024', check_exit_code=False)
        self.gem5_shell('{} stty cols 1024'.format(self.busybox), check_exit_code=False)
        self.gem5_shell('reset', check_exit_code=False)

    def move_to_temp_dir(self, source):
        """
        Move a file to the temporary directory on the host for copying to the
        gem5 device
        """
        command = "cp {} {}".format(source, self.temp_dir)
        self.logger.debug("Local copy command: {}".format(command))
        subprocess.call(command.split())
        subprocess.call("sync".split())

    def checkpoint_gem5(self, end_simulation=False):
        """ Checkpoint the gem5 simulation, storing all system state """
        self.logger.info("Taking a post-boot checkpoint")
        self.gem5_util("checkpoint")
        if end_simulation:
            self.disconnect()

    def mount_virtio(self):
        """
        Mount the VirtIO device in the simulated system.
        """
        self.logger.info("Mounting VirtIO device in simulated system")

        self.gem5_shell('mkdir -p /mnt/obb')

        mount_command = "mount -t 9p -o trans=virtio,version=9p2000.L,aname={} gem5 /mnt/obb".format(self.temp_dir)
        self.gem5_shell(mount_command)

    def deploy_m5(self, context, force=False):
        """
        Deploys the m5 binary to the device and returns the path to the binary
        on the device.

        :param force: by default, if the binary is already present on the
                    device, it will not be deployed again. Setting force to
                    ``True`` overrides that behaviour and ensures that the
                    binary is always copied. Defaults to ``False``.

        :returns: The on-device path to the m5 binary.

        """
        on_device_executable = self.path.join(self.binaries_directory, 'm5')
        if not force and self.file_exists(on_device_executable):
            # We want to check the version of the binary. We cannot directly
            # check this because the m5 binary itself is unversioned. We also
            # need to make sure not to check the error code as "m5 --help"
            # returns a non-zero error code.
            output = self.gem5_shell('m5 --help', check_exit_code=False)
            if "writefile" in output:
                self.logger.debug("Using the m5 binary on the device...")
                self.m5_path = on_device_executable
                return on_device_executable
            else:
                self.logger.debug("m5 on device does not support writefile!")
        host_file = context.resolver.get(Executable(NO_ONE, self.abi, 'm5'))
        self.logger.info("Installing the m5 binary to the device...")
        self.m5_path = self.install(host_file)
        return self.m5_path
