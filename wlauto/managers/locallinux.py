from wlauto.core.device_manager import DeviceManager
from wlauto import Parameter

from devlib.target import LocalLinuxTarget


class LocalLinuxManager(DeviceManager):

    name = "local_linux"
    target_type = LocalLinuxTarget

    parameters = [
        Parameter('password',
                  description='Password for the user.'),
    ]

    def __init__(self, **kwargs):
        super(LocalLinuxManager, self).__init__(**kwargs)
        self.platform = self.platform_type(core_names=self.core_names,  # pylint: disable=E1102
                                           core_clusters=self.core_clusters,
                                           modules=self.modules)
        self.target = self.target_type(connection_settings=self._make_connection_settings())

    def connect(self):
        self.target.connect()

    def _make_connection_settings(self):
        connection_settings = {}
        connection_settings['password'] = self.password
        return connection_settings
