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


import os
import sys
import re
import string
import shutil
import time
from collections import Counter

import pexpect

from wlauto import BigLittleDevice, RuntimeParameter, Parameter, settings
from wlauto.exceptions import ConfigError, DeviceError
from wlauto.utils.android import adb_connect, adb_disconnect, adb_list_devices
from wlauto.utils.serial_port import open_serial_connection
from wlauto.utils.misc import merge_dicts
from wlauto.utils.types import boolean


BOOT_FIRMWARE = {
    'uefi': {
        'SCC_0x010': '0x000003E0',
        'reboot_attempts': 0,
    },
    'bootmon': {
        'SCC_0x010': '0x000003D0',
        'reboot_attempts': 2,
    },
}

MODES = {
    'mp_a7_only': {
        'images_file': 'images_mp.txt',
        'dtb': 'mp_a7',
        'initrd': 'init_mp',
        'kernel': 'kern_mp',
        'SCC_0x700': '0x1032F003',
        'cpus': ['a7', 'a7', 'a7'],
    },
    'mp_a7_bootcluster': {
        'images_file': 'images_mp.txt',
        'dtb': 'mp_a7bc',
        'initrd': 'init_mp',
        'kernel': 'kern_mp',
        'SCC_0x700': '0x1032F003',
        'cpus': ['a7', 'a7', 'a7', 'a15', 'a15'],
    },
    'mp_a15_only': {
        'images_file': 'images_mp.txt',
        'dtb': 'mp_a15',
        'initrd': 'init_mp',
        'kernel': 'kern_mp',
        'SCC_0x700': '0x0032F003',
        'cpus': ['a15', 'a15'],
    },
    'mp_a15_bootcluster': {
        'images_file': 'images_mp.txt',
        'dtb': 'mp_a15bc',
        'initrd': 'init_mp',
        'kernel': 'kern_mp',
        'SCC_0x700': '0x0032F003',
        'cpus': ['a15', 'a15', 'a7', 'a7', 'a7'],
    },
    'iks_cpu': {
        'images_file': 'images_iks.txt',
        'dtb': 'iks',
        'initrd': 'init_iks',
        'kernel': 'kern_iks',
        'SCC_0x700': '0x1032F003',
        'cpus': ['a7', 'a7'],
    },
    'iks_a15': {
        'images_file': 'images_iks.txt',
        'dtb': 'iks',
        'initrd': 'init_iks',
        'kernel': 'kern_iks',
        'SCC_0x700': '0x0032F003',
        'cpus': ['a15', 'a15'],
    },
    'iks_a7': {
        'images_file': 'images_iks.txt',
        'dtb': 'iks',
        'initrd': 'init_iks',
        'kernel': 'kern_iks',
        'SCC_0x700': '0x0032F003',
        'cpus': ['a7', 'a7'],
    },
    'iks_ns_a15': {
        'images_file': 'images_iks.txt',
        'dtb': 'iks',
        'initrd': 'init_iks',
        'kernel': 'kern_iks',
        'SCC_0x700': '0x0032F003',
        'cpus': ['a7', 'a7', 'a7', 'a15', 'a15'],
    },
    'iks_ns_a7': {
        'images_file': 'images_iks.txt',
        'dtb': 'iks',
        'initrd': 'init_iks',
        'kernel': 'kern_iks',
        'SCC_0x700': '0x0032F003',
        'cpus': ['a7', 'a7', 'a7', 'a15', 'a15'],
    },
}

A7_ONLY_MODES = ['mp_a7_only', 'iks_a7', 'iks_cpu']
A15_ONLY_MODES = ['mp_a15_only', 'iks_a15']

DEFAULT_A7_GOVERNOR_TUNABLES = {
    'interactive': {
        'above_hispeed_delay': 80000,
        'go_hispeed_load': 85,
        'hispeed_freq': 800000,
        'min_sample_time': 80000,
        'timer_rate': 20000,
    },
    'ondemand': {
        'sampling_rate': 50000,
    },
}

DEFAULT_A15_GOVERNOR_TUNABLES = {
    'interactive': {
        'above_hispeed_delay': 80000,
        'go_hispeed_load': 85,
        'hispeed_freq': 1000000,
        'min_sample_time': 80000,
        'timer_rate': 20000,
    },
    'ondemand': {
        'sampling_rate': 50000,
    },
}

ADB_SHELL_TIMEOUT = 30


class _TC2DeviceConfig(object):

    name = 'TC2 Configuration'
    device_name = 'TC2'

    def __init__(self,  # pylint: disable=R0914,W0613
                 root_mount='/media/VEMSD',

                 disable_boot_configuration=False,
                 boot_firmware=None,
                 mode=None,

                 fs_medium='usb',

                 device_working_directory='/data/local/usecase',

                 bm_image='bm_v519r.axf',

                 serial_device='/dev/ttyS0',
                 serial_baud=38400,
                 serial_max_timeout=600,
                 serial_log=sys.stdout,

                 init_timeout=120,

                 always_delete_uefi_entry=True,
                 psci_enable=True,

                 host_working_directory=None,

                 a7_governor_tunables=None,
                 a15_governor_tunables=None,

                 adb_name=None,
                 # Compatibility with other android devices.
                 enable_screen_check=None,  # pylint: disable=W0613
                 **kwargs
                 ):
        self.root_mount = root_mount
        self.disable_boot_configuration = disable_boot_configuration
        if not disable_boot_configuration:
            self.boot_firmware = boot_firmware or 'uefi'
            self.default_mode = mode or 'mp_a7_bootcluster'
        elif boot_firmware or mode:
            raise ConfigError('boot_firmware and/or mode cannot be specified when disable_boot_configuration is enabled.')

        self.mode = self.default_mode
        self.working_directory = device_working_directory
        self.serial_device = serial_device
        self.serial_baud = serial_baud
        self.serial_max_timeout = serial_max_timeout
        self.serial_log = serial_log
        self.bootmon_prompt = re.compile('^([KLM]:\\\)?>', re.MULTILINE)

        self.fs_medium = fs_medium.lower()

        self.bm_image = bm_image

        self.init_timeout = init_timeout

        self.always_delete_uefi_entry = always_delete_uefi_entry
        self.psci_enable = psci_enable

        self.resource_dir = os.path.join(os.path.dirname(__file__), 'resources')
        self.board_dir = os.path.join(self.root_mount, 'SITE1', 'HBI0249A')
        self.board_file = 'board.txt'
        self.board_file_bak = 'board.bak'
        self.images_file = 'images.txt'

        self.host_working_directory = host_working_directory or settings.meta_directory

        if not a7_governor_tunables:
            self.a7_governor_tunables = DEFAULT_A7_GOVERNOR_TUNABLES
        else:
            self.a7_governor_tunables = merge_dicts(DEFAULT_A7_GOVERNOR_TUNABLES, a7_governor_tunables)

        if not a15_governor_tunables:
            self.a15_governor_tunables = DEFAULT_A15_GOVERNOR_TUNABLES
        else:
            self.a15_governor_tunables = merge_dicts(DEFAULT_A15_GOVERNOR_TUNABLES, a15_governor_tunables)

        self.adb_name = adb_name

    @property
    def src_images_template_file(self):
        return os.path.join(self.resource_dir, MODES[self.mode]['images_file'])

    @property
    def src_images_file(self):
        return os.path.join(self.host_working_directory, 'images.txt')

    @property
    def src_board_template_file(self):
        return os.path.join(self.resource_dir, 'board_template.txt')

    @property
    def src_board_file(self):
        return os.path.join(self.host_working_directory, 'board.txt')

    @property
    def kernel_arguments(self):
        kernel_args = ' console=ttyAMA0,38400 androidboot.console=ttyAMA0 selinux=0'
        if self.fs_medium == 'usb':
            kernel_args += ' androidboot.hardware=arm-versatileexpress-usb'
        if 'iks' in self.mode:
            kernel_args += ' no_bL_switcher=0'
        return kernel_args

    @property
    def kernel(self):
        return MODES[self.mode]['kernel']

    @property
    def initrd(self):
        return MODES[self.mode]['initrd']

    @property
    def dtb(self):
        return MODES[self.mode]['dtb']

    @property
    def SCC_0x700(self):
        return MODES[self.mode]['SCC_0x700']

    @property
    def SCC_0x010(self):
        return BOOT_FIRMWARE[self.boot_firmware]['SCC_0x010']

    @property
    def reboot_attempts(self):
        return BOOT_FIRMWARE[self.boot_firmware]['reboot_attempts']

    def validate(self):
        valid_modes = MODES.keys()
        if self.mode not in valid_modes:
            message = 'Invalid mode: {}; must be in {}'.format(
                self.mode, valid_modes)
            raise ConfigError(message)

        valid_boot_firmware = BOOT_FIRMWARE.keys()
        if self.boot_firmware not in valid_boot_firmware:
            message = 'Invalid boot_firmware: {}; must be in {}'.format(
                self.boot_firmware,
                valid_boot_firmware)
            raise ConfigError(message)

        if self.fs_medium not in ['usb', 'sdcard']:
            message = 'Invalid filesystem medium: {}  allowed values : usb, sdcard '.format(self.fs_medium)
            raise ConfigError(message)


class TC2Device(BigLittleDevice):

    name = 'TC2'
    description = """
    TC2 is a development board, which has three A7 cores and two A15 cores.

    TC2 has a number of boot parameters which are:

        :root_mount: Defaults to '/media/VEMSD'
        :boot_firmware: It has only two boot firmware options, which are
                        uefi and bootmon. Defaults to 'uefi'.
        :fs_medium: Defaults to 'usb'.
        :device_working_directory: The direcitory that WA will be using to copy
                                   files to. Defaults to 'data/local/usecase'
        :serial_device: The serial device which TC2 is connected to. Defaults to
                        '/dev/ttyS0'.
        :serial_baud: Defaults to 38400.
        :serial_max_timeout: Serial timeout value in seconds. Defaults to 600.
        :serial_log: Defaults to standard output.
        :init_timeout: The timeout in seconds to init the device. Defaults set
                       to 30.
        :always_delete_uefi_entry: If true, it will delete the ufi entry.
                                   Defaults to True.
        :psci_enable: Enabling the psci. Defaults to True.
        :host_working_directory: The host working directory. Defaults to None.
        :disable_boot_configuration: Disables boot configuration through images.txt and board.txt. When
                                     this is ``True``, those two files will not be overwritten in VEMSD.
                                     This option may be necessary if the firmware version in the ``TC2``
                                     is not compatible with the templates in WA. Please note that enabling
                                     this will prevent you form being able to set ``boot_firmware`` and
                                     ``mode`` parameters. Defaults to ``False``.

    TC2 can also have a number of different booting mode, which are:

        :mp_a7_only: Only the A7 cluster.
        :mp_a7_bootcluster: Both A7 and A15 clusters, but it boots on A7
                            cluster.
        :mp_a15_only: Only the A15 cluster.
        :mp_a15_bootcluster: Both A7 and A15 clusters, but it boots on A15
                             clusters.
        :iks_cpu: Only A7 cluster with only 2 cpus.
        :iks_a15: Only A15 cluster.
        :iks_a7: Same as iks_cpu
        :iks_ns_a15: Both A7 and A15 clusters.
        :iks_ns_a7: Both A7 and A15 clusters.

    The difference between mp and iks is the scheduling policy.

    TC2 takes the following runtime parameters

        :a7_cores: Number of active A7 cores.
        :a15_cores: Number of active A15 cores.
        :a7_governor: CPUFreq governor for the A7 cluster.
        :a15_governor: CPUFreq governor for the A15 cluster.
        :a7_min_frequency: Minimum CPU frequency for the A7 cluster.
        :a15_min_frequency: Minimum CPU frequency for the A15 cluster.
        :a7_max_frequency: Maximum CPU frequency for the A7 cluster.
        :a15_max_frequency: Maximum CPU frequency for the A7 cluster.
        :irq_affinity: lambda x: Which cluster will receive IRQs.
        :cpuidle: Whether idle states should be enabled.
        :sysfile_values: A dict mapping a complete file path to the value that
                         should be echo'd into it. By default, the file will be
                         subsequently read to verify that the value was written
                         into it with DeviceError raised otherwise. For write-only
                         files, this check can be disabled by appending a ``!`` to
                         the end of the file path.

    """

    has_gpu = False
    a15_only_modes = A15_ONLY_MODES
    a7_only_modes = A7_ONLY_MODES
    not_configurable_modes = ['iks_a7', 'iks_cpu', 'iks_a15']

    parameters = [
        Parameter('core_names', mandatory=False, override=True,
                  description='This parameter will be ignored for TC2'),
        Parameter('core_clusters', mandatory=False, override=True,
                  description='This parameter will be ignored for TC2'),
    ]

    runtime_parameters = [
        RuntimeParameter('irq_affinity', lambda d, x: d.set_irq_affinity(x.lower()), lambda: None),
        RuntimeParameter('cpuidle', lambda d, x: d.enable_idle_states() if boolean(x) else d.disable_idle_states(),
                         lambda d: d.get_cpuidle())
    ]

    def get_mode(self):
        return self.config.mode

    def set_mode(self, mode):
        if self._has_booted:
            raise DeviceError('Attempting to set boot mode when already booted.')
        valid_modes = MODES.keys()
        if mode is None:
            mode = self.config.default_mode
        if mode not in valid_modes:
            message = 'Invalid mode: {}; must be in {}'.format(mode, valid_modes)
            raise ConfigError(message)
        self.config.mode = mode

    mode = property(get_mode, set_mode)

    def _get_core_names(self):
        return MODES[self.mode]['cpus']

    def _set_core_names(self, value):
        pass

    core_names = property(_get_core_names, _set_core_names)

    def _get_core_clusters(self):
        seen = set([])
        core_clusters = []
        cluster_id = -1
        for core in MODES[self.mode]['cpus']:
            if core not in seen:
                seen.add(core)
                cluster_id += 1
            core_clusters.append(cluster_id)
        return core_clusters

    def _set_core_clusters(self, value):
        pass

    core_clusters = property(_get_core_clusters, _set_core_clusters)

    @property
    def cpu_cores(self):
        return MODES[self.mode]['cpus']

    @property
    def max_a7_cores(self):
        return Counter(MODES[self.mode]['cpus'])['a7']

    @property
    def max_a15_cores(self):
        return Counter(MODES[self.mode]['cpus'])['a15']

    @property
    def a7_governor_tunables(self):
        return self.config.a7_governor_tunables

    @property
    def a15_governor_tunables(self):
        return self.config.a15_governor_tunables

    def __init__(self, **kwargs):
        super(TC2Device, self).__init__()
        self.config = _TC2DeviceConfig(**kwargs)
        self.working_directory = self.config.working_directory
        self._serial = None
        self._has_booted = None

    def boot(self, **kwargs):  # NOQA
        mode = kwargs.get('os_mode', None)
        self._is_ready = False
        self._has_booted = False

        self.mode = mode
        self.logger.debug('Booting in {} mode'.format(self.mode))

        with open_serial_connection(timeout=self.config.serial_max_timeout,
                                    port=self.config.serial_device,
                                    baudrate=self.config.serial_baud) as target:
            if self.config.boot_firmware == 'bootmon':
                self._boot_using_bootmon(target)
            elif self.config.boot_firmware == 'uefi':
                self._boot_using_uefi(target)
            else:
                message = 'Unexpected boot firmware: {}'.format(self.config.boot_firmware)
                raise ConfigError(message)

            try:
                target.sendline('')
                self.logger.debug('Waiting for the Android prompt.')
                target.expect(self.android_prompt, timeout=40)  # pylint: disable=E1101
            except pexpect.TIMEOUT:
                # Try a second time before giving up.
                self.logger.debug('Did not get Android prompt, retrying...')
                target.sendline('')
                target.expect(self.android_prompt, timeout=10)  # pylint: disable=E1101

            self.logger.debug('Waiting for OS to initialize...')
            started_waiting_time = time.time()
            time.sleep(20)  # we know it's not going to to take less time than this.
            boot_completed, got_ip_address = False, False
            while True:
                try:
                    if not boot_completed:
                        target.sendline('getprop sys.boot_completed')
                        boot_completed = target.expect(['0.*', '1.*'], timeout=10)
                    if not got_ip_address:
                        target.sendline('getprop dhcp.eth0.ipaddress')
                        # regexes  are processed in order, so ip regex has to
                        # come first (as we only want to match new line if we
                        # don't match the IP). We do a "not" make the logic
                        # consistent with boot_completed.
                        got_ip_address = not target.expect(['[1-9]\d*.\d+.\d+.\d+', '\n'], timeout=10)
                except pexpect.TIMEOUT:
                    pass  # We have our own timeout -- see below.
                if boot_completed and got_ip_address:
                    break
                time.sleep(5)
                if (time.time() - started_waiting_time) > self.config.init_timeout:
                    raise DeviceError('Timed out waiting for the device to initialize.')

        self._has_booted = True

    def connect(self):
        if not self._is_ready:
            if self.config.adb_name:
                self.adb_name = self.config.adb_name  # pylint: disable=attribute-defined-outside-init
            else:
                with open_serial_connection(timeout=self.config.serial_max_timeout,
                                            port=self.config.serial_device,
                                            baudrate=self.config.serial_baud) as target:
                    # Get IP address and push the Gator and PMU logger.
                    target.sendline('su')  # as of Android v5.0.2, Linux does not boot into root shell
                    target.sendline('netcfg')
                    ipaddr_re = re.compile('eth0 +UP +(.+)/.+', re.MULTILINE)
                    target.expect(ipaddr_re)
                    output = target.after
                    match = re.search('eth0 +UP +(.+)/.+', output)
                    if not match:
                        raise DeviceError('Could not get adb IP address.')
                    ipaddr = match.group(1)

                    # Connect to device using adb.
                    target.expect(self.android_prompt)  # pylint: disable=E1101
                    self.adb_name = ipaddr + ":5555"  # pylint: disable=W0201

            if self.adb_name in adb_list_devices():
                adb_disconnect(self.adb_name)
            adb_connect(self.adb_name)
            self._is_ready = True
            self.execute("input keyevent 82", timeout=ADB_SHELL_TIMEOUT)
            self.execute("svc power stayon true", timeout=ADB_SHELL_TIMEOUT)

    def disconnect(self):
        adb_disconnect(self.adb_name)
        self._is_ready = False

    # TC2-specific methods. You should avoid calling these in
    # Workloads/Instruments as that would tie them to TC2 (and if that is
    # the case, then you should set the supported_devices parameter in the
    # Workload/Instrument accordingly). Most of these can be replace with a
    # call to set_runtime_parameters.

    def get_cpuidle(self):
        return self.get_sysfile_value('/sys/devices/system/cpu/cpu0/cpuidle/state1/disable')

    def enable_idle_states(self):
        """
        Fully enables idle states on TC2.
        See http://wiki.arm.com/Research/TC2SetupAndUsage ("Enabling Idle Modes" section)
        and http://wiki.arm.com/ASD/ControllingPowerManagementInLinaroKernels

        """
        # Enable C1 (cluster shutdown).
        self.set_sysfile_value('/sys/devices/system/cpu/cpu0/cpuidle/state1/disable', 0, verify=False)
        # Enable C0 on A15 cluster.
        self.set_sysfile_value('/sys/kernel/debug/idle_debug/enable_idle', 0, verify=False)
        # Enable C0 on A7 cluster.
        self.set_sysfile_value('/sys/kernel/debug/idle_debug/enable_idle', 1, verify=False)

    def disable_idle_states(self):
        """
        Disable idle states on TC2.
        See http://wiki.arm.com/Research/TC2SetupAndUsage ("Enabling Idle Modes" section)
        and http://wiki.arm.com/ASD/ControllingPowerManagementInLinaroKernels

        """
        # Disable C1 (cluster shutdown).
        self.set_sysfile_value('/sys/devices/system/cpu/cpu0/cpuidle/state1/disable', 1, verify=False)
        # Disable C0.
        self.set_sysfile_value('/sys/kernel/debug/idle_debug/enable_idle', 0xFF, verify=False)

    def set_irq_affinity(self, cluster):
        """
        Set's IRQ affinity to the specified cluster.

        This method will only work if the device mode is mp_a7_bootcluster or
        mp_a15_bootcluster. This operation does not make sense if there is only one
        cluster active (all IRQs will obviously go to that), and it will not work for
        IKS kernel because clusters are not exposed to sysfs.

        :param cluster: must be either 'a15' or 'a7'.

        """
        if self.config.mode not in ('mp_a7_bootcluster', 'mp_a15_bootcluster'):
            raise ConfigError('Cannot set IRQ affinity with mode {}'.format(self.config.mode))
        if cluster == 'a7':
            self.execute('/sbin/set_irq_affinity.sh 0xc07', check_exit_code=False)
        elif cluster == 'a15':
            self.execute('/sbin/set_irq_affinity.sh 0xc0f', check_exit_code=False)
        else:
            raise ConfigError('cluster must either "a15" or "a7"; got {}'.format(cluster))

    def _boot_using_uefi(self, target):
        self.logger.debug('Booting using UEFI.')
        self._wait_for_vemsd_mount(target)
        self._setup_before_reboot()
        self._perform_uefi_reboot(target)

        # Get to the UEFI menu.
        self.logger.debug('Waiting for UEFI default selection.')
        target.sendline('reboot')
        target.expect('The default boot selection will start in'.rstrip())
        time.sleep(1)
        target.sendline(''.rstrip())

        # If delete every time is specified, try to delete entry.
        if self.config.always_delete_uefi_entry:
            self._delete_uefi_entry(target, entry='workload_automation_MP')
            self.config.always_delete_uefi_entry = False

        # Specify argument to be passed specifying that psci is (or is not) enabled
        if self.config.psci_enable:
            psci_enable = ' psci=enable'
        else:
            psci_enable = ''

        # Identify the workload automation entry.
        selection_pattern = r'\[([0-9]*)\] '

        try:
            target.expect(re.compile(selection_pattern + 'workload_automation_MP'), timeout=5)
            wl_menu_item = target.match.group(1)
        except pexpect.TIMEOUT:
            self._create_uefi_entry(target, psci_enable, entry_name='workload_automation_MP')
            # At this point the board should be rebooted so we need to retry to boot
            self._boot_using_uefi(target)
        else:  # Did not time out.
            try:
                #Identify the boot manager menu item
                target.expect(re.compile(selection_pattern + 'Boot Manager'))
                boot_manager_menu_item = target.match.group(1)

                #Update FDT
                target.sendline(boot_manager_menu_item)
                target.expect(re.compile(selection_pattern + 'Update FDT path'), timeout=15)
                update_fdt_menu_item = target.match.group(1)
                target.sendline(update_fdt_menu_item)
                target.expect(re.compile(selection_pattern + 'NOR Flash .*'), timeout=15)
                bootmonfs_menu_item = target.match.group(1)
                target.sendline(bootmonfs_menu_item)
                target.expect('File path of the FDT blob:')
                target.sendline(self.config.dtb)

                #Return to main manu and boot from wl automation
                target.expect(re.compile(selection_pattern + 'Return to main menu'), timeout=15)
                return_to_main_menu_item = target.match.group(1)
                target.sendline(return_to_main_menu_item)
                target.sendline(wl_menu_item)
            except pexpect.TIMEOUT:
                raise DeviceError('Timed out')

    def _setup_before_reboot(self):
        if not self.config.disable_boot_configuration:
            self.logger.debug('Performing pre-boot setup.')
            substitution = {
                'SCC_0x010': self.config.SCC_0x010,
                'SCC_0x700': self.config.SCC_0x700,
            }
            with open(self.config.src_board_template_file, 'r') as fh:
                template_board_txt = string.Template(fh.read())
                with open(self.config.src_board_file, 'w') as wfh:
                    wfh.write(template_board_txt.substitute(substitution))

            with open(self.config.src_images_template_file, 'r') as fh:
                template_images_txt = string.Template(fh.read())
                with open(self.config.src_images_file, 'w') as wfh:
                    wfh.write(template_images_txt.substitute({'bm_image': self.config.bm_image}))

            shutil.copyfile(self.config.src_board_file,
                            os.path.join(self.config.board_dir, self.config.board_file))
            shutil.copyfile(self.config.src_images_file,
                            os.path.join(self.config.board_dir, self.config.images_file))
            os.system('sync')  # make sure everything is flushed to microSD
        else:
            self.logger.debug('Boot configuration disabled proceeding with existing board.txt and images.txt.')

    def _delete_uefi_entry(self, target, entry):  # pylint: disable=R0201
        """
        this method deletes the entry specified as parameter
        as a precondition serial port input needs to be parsed AT MOST up to
        the point BEFORE recognizing this entry (both entry and boot manager has
        not yet been parsed)

        """
        try:
            selection_pattern = r'\[([0-9]+)\] *'

            try:
                target.expect(re.compile(selection_pattern + entry), timeout=5)
                wl_menu_item = target.match.group(1)
            except pexpect.TIMEOUT:
                return  # Entry does not exist, nothing to delete here...

            # Identify and select boot manager menu item
            target.expect(selection_pattern + 'Boot Manager', timeout=15)
            bootmanager_item = target.match.group(1)
            target.sendline(bootmanager_item)

            # Identify and select 'Remove entry'
            target.expect(selection_pattern + 'Remove Boot Device Entry', timeout=15)
            new_entry_item = target.match.group(1)
            target.sendline(new_entry_item)

            # Delete entry
            target.expect(re.compile(selection_pattern + entry), timeout=5)
            wl_menu_item = target.match.group(1)
            target.sendline(wl_menu_item)

            # Return to main manu
            target.expect(re.compile(selection_pattern + 'Return to main menu'), timeout=15)
            return_to_main_menu_item = target.match.group(1)
            target.sendline(return_to_main_menu_item)
        except pexpect.TIMEOUT:
            raise DeviceError('Timed out while deleting UEFI entry.')

    def _create_uefi_entry(self, target, psci_enable, entry_name):
        """
        Creates the default boot entry that is expected when booting in uefi mode.

        """
        self._wait_for_vemsd_mount(target)
        try:
            selection_pattern = '\[([0-9]+)\] *'

            # Identify and select boot manager menu item.
            target.expect(selection_pattern + 'Boot Manager', timeout=15)
            bootmanager_item = target.match.group(1)
            target.sendline(bootmanager_item)

            # Identify and select 'add new entry'.
            target.expect(selection_pattern + 'Add Boot Device Entry', timeout=15)
            new_entry_item = target.match.group(1)
            target.sendline(new_entry_item)

            # Identify and select BootMonFs.
            target.expect(selection_pattern + 'NOR Flash .*', timeout=15)
            BootMonFs_item = target.match.group(1)
            target.sendline(BootMonFs_item)

            # Specify the parameters of the new entry.
            target.expect('.+the kernel', timeout=5)
            target.sendline(self.config.kernel)  # kernel path
            target.expect('Has FDT support\?.*\[y\/n\].*', timeout=5)
            time.sleep(0.5)
            target.sendline('y')   # Has Fdt support? -> y
            target.expect('Add an initrd.*\[y\/n\].*', timeout=5)
            time.sleep(0.5)
            target.sendline('y')   # add an initrd? -> y
            target.expect('.+the initrd.*', timeout=5)
            time.sleep(0.5)
            target.sendline(self.config.initrd)  # initrd path
            target.expect('.+to the binary.*', timeout=5)
            time.sleep(0.5)
            _slow_sendline(target, self.config.kernel_arguments + psci_enable)  # arguments to pass to binary
            time.sleep(0.5)
            target.expect('.+new Entry.+', timeout=5)
            _slow_sendline(target, entry_name)  # Entry name
            target.expect('Choice.+', timeout=15)
            time.sleep(2)
        except pexpect.TIMEOUT:
            raise DeviceError('Timed out while creating UEFI entry.')
        self._perform_uefi_reboot(target)

    def _perform_uefi_reboot(self, target):
        self._wait_for_vemsd_mount(target)
        open(os.path.join(self.config.root_mount, 'reboot.txt'), 'a').close()

    def _wait_for_vemsd_mount(self, target, timeout=100):
        attempts = 1 + self.config.reboot_attempts
        if os.path.exists(os.path.join(self.config.root_mount, 'config.txt')):
            return

        self.logger.debug('Waiting for VEMSD to mount...')
        for i in xrange(attempts):
            if i:  # Do not reboot on the first attempt.
                target.sendline('reboot')
            target.sendline('usb_on')
            for _ in xrange(timeout):
                time.sleep(1)
                if os.path.exists(os.path.join(self.config.root_mount, 'config.txt')):
                    return

        raise DeviceError('Timed out waiting for VEMSD to mount.')

    def _boot_using_bootmon(self, target):
        """
        This method Boots TC2 using the bootmon interface.
        """
        self.logger.debug('Booting using bootmon.')

        try:
            self._wait_for_vemsd_mount(target, timeout=20)
        except DeviceError:
            # OK, something's wrong. Reboot the board and try again.
            self.logger.debug('VEMSD not mounted, attempting to power cycle device.')
            target.sendline(' ')
            state = target.expect(['Cmd> ', self.config.bootmon_prompt, self.android_prompt])  # pylint: disable=E1101

            if state == 0 or state == 1:
                # Reboot - Bootmon
                target.sendline('reboot')
                target.expect('Powering up system...')
            elif state == 2:
                target.sendline('reboot -n')
                target.expect('Powering up system...')
            else:
                raise DeviceError('Unexpected board state {}; should be 0, 1 or 2'.format(state))

            self._wait_for_vemsd_mount(target)

        self._setup_before_reboot()

        # Reboot - Bootmon
        self.logger.debug('Rebooting into bootloader...')
        open(os.path.join(self.config.root_mount, 'reboot.txt'), 'a').close()
        target.expect('Powering up system...')
        target.expect(self.config.bootmon_prompt)

        # Wait for VEMSD to mount
        self._wait_for_vemsd_mount(target)

        #Boot Linux - Bootmon
        target.sendline('fl linux fdt ' + self.config.dtb)
        target.expect(self.config.bootmon_prompt)
        target.sendline('fl linux initrd ' + self.config.initrd)
        target.expect(self.config.bootmon_prompt)
        #Workaround TC2 bootmon serial issue for loading large initrd blob
        target.sendline(' ')
        target.expect(self.config.bootmon_prompt)
        target.sendline('fl linux boot ' + self.config.kernel + self.config.kernel_arguments)


# Utility functions.

def _slow_sendline(target, line):
    for c in line:
        target.send(c)
        time.sleep(0.1)
    target.sendline('')

