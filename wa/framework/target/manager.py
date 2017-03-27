import logging
import tempfile
import threading
import os
import time
import shutil
import sys

from wa.framework import signal
from wa.framework.exception import WorkerThreadError, ConfigError
from wa.framework.plugin import Parameter
from wa.framework.target.descriptor import (get_target_descriptions,
                                            instantiate_target,
                                            instantiate_assistant)
from wa.framework.target.info import TargetInfo
from wa.framework.target.runtime_config import (SysfileValuesRuntimeConfig,
                                                HotplugRuntimeConfig,
                                                CpufreqRuntimeConfig,
                                                CpuidleRuntimeConfig)
from wa.utils.misc import isiterable
from wa.utils.serializer import json


from devlib import LocalLinuxTarget, LinuxTarget, AndroidTarget
from devlib.utils.types import identifier
from devlib.utils.misc import memoized


class TargetManager(object):

    name = 'target-manager'

    description = """
    Instanciated the required target and performs configuration and validation
    of the device.

    """

    parameters = [
        Parameter('disconnect', kind=bool, default=False,
                  description="""
                  Specifies whether the target should be disconnected from
                  at the end of the run.
                  """),
    ]

    runtime_config_cls = [
        # order matters
        SysfileValuesRuntimeConfig,
        HotplugRuntimeConfig,
        CpufreqRuntimeConfig,
        CpuidleRuntimeConfig,
    ]

    def __init__(self, name, parameters):
        self.logger = logging.getLogger('tm')
        self.target_name = name
        self.target = None
        self.assistant = None
        self.platform_name = None
        self.parameters = parameters
        self.disconnect = parameters.get('disconnect')
        self.info = TargetInfo()

        self._init_target()
        self.runtime_configs = [cls(self.target) for cls in self.runtime_config_cls]

    def finalize(self):
        # self.logger.info('Disconnecting from the device')
        if self.disconnect:
            with signal.wrap('TARGET_DISCONNECT'):
                self.target.disconnect()

    def add_parameters(self, parameters=None):
        if parameters:
            self.parameters = parameters
        if not self.parameters:
            raise ConfigError('No Configuration Provided')

        for name in self.parameters.keys():
            for cfg in self.runtime_configs:
                # if name in cfg.supported_parameters:
                if any(parameter in name for parameter in cfg.supported_parameters):
                    cfg.add(name, self.parameters.pop(name))

    def start(self):
        self.assistant.start()

    def stop(self):
        self.assistant.stop()

    def extract_results(self, context):
        self.assistant.extract_results(context)

    @memoized
    def get_target_info(self):
        return TargetInfo(self.target)

    def validate_runtime_parameters(self, params):
        for cfg in self.runtime_configs:
            cfg.validate()

    def merge_runtime_parameters(self, params):
        pass

    def set_parameters(self):
        for cfg in self.runtime_configs:
            cfg.set()

    def clear_parameters(self):
        for cfg in self.runtime_configs:
            cfg.clear()

    def _init_target(self):
        target_map = {td.name: td for td in get_target_descriptions()}
        if self.target_name not in target_map:
            raise ValueError('Unknown Target: {}'.format(self.target_name))
        tdesc = target_map[self.target_name]

        self.target = instantiate_target(tdesc, self.parameters, connect=False)

        with signal.wrap('TARGET_CONNECT'):
            self.target.connect()
        self.logger.info('Setting up target')
        self.target.setup()

        self.assistant = instantiate_assistant(tdesc, self.parameters, self.target)
