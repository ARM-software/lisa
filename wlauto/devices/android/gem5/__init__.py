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

import logging
import os
import time

from wlauto import AndroidDevice, Parameter
from wlauto.common.gem5.device import BaseGem5Device
from wlauto.exceptions import DeviceError


class Gem5AndroidDevice(BaseGem5Device, AndroidDevice):
    """
    Implements gem5 Android device.

    This class allows a user to connect WA to a simulation using gem5. The
    connection to the device is made using the telnet connection of the
    simulator, and is used for all commands. The simulator does not have ADB
    support, and therefore we need to fall back to using standard shell
    commands.

    Files are copied into the simulation using a VirtIO 9P device in gem5. Files
    are copied out of the simulated environment using the m5 writefile command
    within the simulated system.

    When starting the workload run, the simulator is automatically started by
    Workload Automation, and a connection to the simulator is established. WA
    will then wait for Android to boot on the simulated system (which can take
    hours), prior to executing any other commands on the device. It is also
    possible to resume from a checkpoint when starting the simulation. To do
    this, please append the relevant checkpoint commands from the gem5
    simulation script to the gem5_discription argument in the agenda.

    Host system requirements:
        * VirtIO support. We rely on diod on the host system. This can be
          installed on ubuntu using the following command:

                sudo apt-get install diod

    Guest requirements:
        * VirtIO support. We rely on VirtIO to move files into the simulation.
          Please make sure that the following are set in the kernel
          configuration:

                CONFIG_NET_9P=y

                CONFIG_NET_9P_VIRTIO=y

                CONFIG_9P_FS=y

                CONFIG_9P_FS_POSIX_ACL=y

                CONFIG_9P_FS_SECURITY=y

                CONFIG_VIRTIO_BLK=y

        * m5 binary. Please make sure that the m5 binary is on the device and
          can by found in the path.
        * Busybox. Due to restrictions, we assume that busybox is installed in
          the guest system, and can be found in the path.
    """

    name = 'gem5_android'
    platform = 'android'

    parameters = [
        Parameter('core_names', default=[], override=True),
        Parameter('core_clusters', default=[], override=True),
    ]

    # Overwritten from Device. For documentation, see corresponding method in
    # Device.

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('Gem5AndroidDevice')
        AndroidDevice.__init__(self, **kwargs)
        BaseGem5Device.__init__(self)

    def login_to_device(self):
        pass

    def wait_for_boot(self):
        self.logger.info("Waiting for Android to boot...")
        while True:
            try:
                booted = (1 == int('0' + self.gem5_shell('getprop sys.boot_completed', check_exit_code=False)))
                anim_finished = (1 == int('0' + self.gem5_shell('getprop service.bootanim.exit', check_exit_code=False)))
                if booted and anim_finished:
                    break
            except (DeviceError, ValueError):
                pass
            time.sleep(60)

        self.logger.info("Android booted")

    def install(self, filepath, timeout=3 * 3600):  # pylint: disable=W0221
        """ Install an APK or a normal executable """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            return self.install_apk(filepath, timeout)
        else:
            return self.install_executable(filepath)

    def install_apk(self, filepath, timeout=3 * 3600):  # pylint: disable=W0221
        """
        Install an APK on the gem5 device

        The APK is pushed to the device. Then the file and folder permissions
        are changed to ensure that the APK can be correctly installed. The APK
        is then installed on the device using 'pm'.
        """
        self._check_ready()
        self.logger.info("Installing {}".format(filepath))
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.apk':
            filename = os.path.basename(filepath)
            on_device_path = os.path.join('/data/local/tmp', filename)
            self.push_file(filepath, on_device_path)
            # We need to make sure that the folder permissions are set
            # correctly, else the APK does not install correctly.
            self.gem5_shell('busybox chmod 775 /data/local/tmp')
            self.gem5_shell('busybox chmod 774 {}'.format(on_device_path))
            self.logger.debug("Actually installing the APK: {}".format(on_device_path))
            return self.gem5_shell("pm install {}".format(on_device_path))
        else:
            raise DeviceError('Can\'t install {}: unsupported format.'.format(filepath))

    def install_executable(self, filepath, with_name=None):
        """ Install an executable """
        executable_name = os.path.basename(filepath)
        on_device_file = self.path.join(self.working_directory, executable_name)
        on_device_executable = self.path.join(self.binaries_directory, executable_name)
        self.push_file(filepath, on_device_file)
        self.execute('busybox cp {} {}'.format(on_device_file, on_device_executable))
        self.execute('busybox chmod 0777 {}'.format(on_device_executable))
        return on_device_executable

    def uninstall(self, package):
        self._check_ready()
        self.gem5_shell("pm uninstall {}".format(package))

    def dump_logcat(self, outfile, filter_spec=None):
        """ Extract logcat from simulation """
        self.logger.info("Extracting logcat from the simulated system")
        filename = outfile.split('/')[-1]
        command = 'logcat -d > {}'.format(filename)
        self.gem5_shell(command)
        self.pull_file("{}".format(filename), outfile)

    def clear_logcat(self):
        """Clear (flush) logcat log."""
        if self._logcat_poller:
            return self._logcat_poller.clear_buffer()
        else:
            return self.gem5_shell('logcat -c')

    def disable_selinux(self):
        """ Disable SELinux. Overridden as parent implementation uses ADB """
        api_level = int(self.gem5_shell('getprop ro.build.version.sdk').strip())

        # SELinux was added in Android 4.3 (API level 18). Trying to
        # 'getenforce' in earlier versions will produce an error.
        if api_level >= 18:
            se_status = self.execute('getenforce', as_root=True).strip()
            if se_status == 'Enforcing':
                self.execute('setenforce 0', as_root=True)

    def get_properties(self, context):  # pylint: disable=R0801
        """ Get the property files from the device """
        BaseGem5Device.get_properties(self, context)
        props = self._get_android_properties(context)
        return props

    def disable_screen_lock(self):
        """
        Attempts to disable he screen lock on the device.

        Overridden here as otherwise we have issues with too many backslashes.
        """
        lockdb = '/data/system/locksettings.db'
        sqlcommand = "update locksettings set value=\'0\' where name=\'screenlock.disabled\';"
        self.execute('sqlite3 {} "{}"'.format(lockdb, sqlcommand), as_root=True)

    def capture_screen(self, filepath):
        if BaseGem5Device.capture_screen(self, filepath):
            return

        # If we didn't manage to do the above, call the parent class.
        self.logger.warning("capture_screen: falling back to parent class implementation")
        AndroidDevice.capture_screen(self, filepath)
