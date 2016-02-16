from wlauto.core.device_manager import DeviceManager
from wlauto import Parameter, Alias
from wlauto.utils.types import boolean
from wlauto.exceptions import ConfigError

from devlib.target import LinuxTarget


class LinuxManager(DeviceManager):

    name = "linux"
    target_type = LinuxTarget

    aliases = [
        Alias('generic_linux'),
    ]

    parameters = [
        Parameter('host', mandatory=True, description='Host name or IP address for the device.'),
        Parameter('username', mandatory=True, description='User name for the account on the device.'),
        Parameter('password', description='Password for the account on the device (for password-based auth).'),
        Parameter('keyfile', description='Keyfile to be used for key-based authentication.'),
        Parameter('port', kind=int, default=22, description='SSH port number on the device.'),
        Parameter('password_prompt', default='[sudo] password',
                  description='Prompt presented by sudo when requesting the password.'),
        Parameter('use_telnet', kind=boolean, default=False,
                  description='Optionally, telnet may be used instead of ssh, though this is discouraged.'),
        Parameter('boot_timeout', kind=int, default=120,
                  description='How long to try to connect to the device after a reboot.'),
        Parameter('working_directory', default="/root/wa", override=True),
        Parameter('binaries_directory', default="/root/wa/bin", override=True),
    ]

    def __init__(self, **kwargs):
        super(LinuxManager, self).__init__(**kwargs)
        self.platform = self.platform_type(core_names=self.core_names,  # pylint: disable=E1102
                                           core_clusters=self.core_clusters,
                                           modules=self.modules)
        self.target = self.target_type(connection_settings=self._make_connection_settings(),
                                       connect=False,
                                       platform=self.platform,
                                       working_directory=self.working_directory,
                                       executables_directory=self.binaries_directory,)

    def validate(self):
        if self.password and self.keyfile:
            raise ConfigError("Either `password` or `keyfile` must be given but not both")

    def connect(self):
        self.target.connect(self.boot_timeout)

    def _make_connection_settings(self):
        connection_settings = {}
        connection_settings['host'] = self.host
        connection_settings['username'] = self.username
        connection_settings['port'] = self.port
        connection_settings['telnet'] = self.use_telnet
        connection_settings['password_prompt'] = self.password_prompt

        if self.keyfile:
            connection_settings['keyfile'] = self.keyfilehollis
        elif self.password:
            connection_settings['password'] = self.password

        return connection_settings
