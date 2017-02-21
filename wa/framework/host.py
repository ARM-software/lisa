import os

from wa.framework.configuration import settings
from wa.framework.exception import ConfigError
from wa.utils.misc import ensure_directory_exists


class HostRunConfig(object):
    """
    Host-side configuration for a run.
    """

    def __init__(self, output_directory, 
                 run_info_directory=None,
                 run_config_directory=None):
        self.output_directory = output_directory
        self.run_info_directory = run_info_directory or os.path.join(self.output_directory, '_info')
        self.run_config_directory = run_config_directory or os.path.join(self.output_directory, '_config')

    def initialize(self):
        ensure_directory_exists(self.output_directory)
        ensure_directory_exists(self.run_info_directory)
        ensure_directory_exists(self.run_config_directory)
