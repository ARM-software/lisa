#    Copyright 2018 ARM Limited
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

import logging

from devlib import Gem5SimulationPlatform
from devlib.utils.misc import memoized

from wa.framework import signal
from wa.framework.exception import ExecutionError, TargetError, TargetNotRespondingError
from wa.framework.plugin import Parameter
from wa.framework.target.descriptor import (get_target_description,
                                            instantiate_target,
                                            instantiate_assistant)
from wa.framework.target.info import (get_target_info, get_target_info_from_cache,
                                      cache_target_info, read_target_info_cache)
from wa.framework.target.runtime_parameter_manager import RuntimeParameterManager
from wa.utils.types import module_name_set


class TargetManager(object):
    """
    Instantiate the required target and perform configuration and validation of the device.
    """

    parameters = [
        Parameter('disconnect', kind=bool, default=False,
                  description="""
                  Specifies whether the target should be disconnected from
                  at the end of the run.
                  """),
    ]

    def __init__(self, name, parameters, outdir):
        self.outdir = outdir
        self.logger = logging.getLogger('tm')
        self.target_name = name
        self.target = None
        self.assistant = None
        self.platform_name = None
        self.is_responsive = None
        self.rpm = None
        self.parameters = parameters
        self.disconnect = parameters.get('disconnect')

    def initialize(self):
        self._init_target()
        self.assistant.initialize()

        # If target supports hotplugging, online all cpus before perform discovery
        # and restore original configuration after completed.
        if self.target.has('hotplug'):
            online_cpus = self.target.list_online_cpus()
            try:
                self.target.hotplug.online_all()
            except TargetError:
                msg = 'Failed to online all CPUS - some information may not be '\
                      'able to be retrieved.'
                self.logger.debug(msg)
            self.rpm = RuntimeParameterManager(self.target)
            all_cpus = set(range(self.target.number_of_cpus))
            self.target.hotplug.offline(*all_cpus.difference(online_cpus))
        else:
            self.rpm = RuntimeParameterManager(self.target)

    def finalize(self):
        if not self.target:
            return
        if self.assistant:
            self.assistant.finalize()
        if self.disconnect or isinstance(self.target.platform, Gem5SimulationPlatform):
            self.logger.info('Disconnecting from the device')
            with signal.wrap('TARGET_DISCONNECT'):
                self.target.disconnect()

    def start(self):
        self.assistant.start()

    def stop(self):
        self.assistant.stop()

    def extract_results(self, context):
        self.assistant.extract_results(context)

    @memoized
    def get_target_info(self):
        cache = read_target_info_cache()
        info = get_target_info_from_cache(self.target.system_id, cache=cache)

        if info is None:
            info = get_target_info(self.target)
            cache_target_info(info, cache=cache)
        else:
            # If module configuration has changed form when the target info
            # was previously cached, it is possible additional info will be
            # available, so should re-generate the cache.
            if module_name_set(info.modules) != module_name_set(self.target.modules):
                info = get_target_info(self.target)
                cache_target_info(info, overwrite=True, cache=cache)

        return info

    def reboot(self, context, hard=False):
        with signal.wrap('REBOOT', self, context):
            self.target.reboot(hard)

    def merge_runtime_parameters(self, parameters):
        return self.rpm.merge_runtime_parameters(parameters)

    def validate_runtime_parameters(self, parameters):
        self.rpm.validate_runtime_parameters(parameters)

    def commit_runtime_parameters(self, parameters):
        self.rpm.commit_runtime_parameters(parameters)

    def verify_target_responsive(self, context):
        can_reboot = context.reboot_policy.can_reboot
        if not self.target.check_responsive(explode=False):
            self.is_responsive = False
            if not can_reboot:
                raise TargetNotRespondingError('Target unresponsive and is not allowed to reboot.')
            elif self.target.has('hard_reset'):
                self.logger.info('Target unresponsive; performing hard reset')
                self.reboot(context, hard=True)
                self.is_responsive = True
                raise ExecutionError('Target became unresponsive but was recovered.')
            else:
                raise TargetNotRespondingError('Target unresponsive and hard reset not supported; bailing.')

    def _init_target(self):
        tdesc = get_target_description(self.target_name)

        extra_plat_params = {}
        if tdesc.platform is Gem5SimulationPlatform:
            extra_plat_params['host_output_dir'] = self.outdir

        self.logger.debug('Creating {} target'.format(self.target_name))
        self.target = instantiate_target(tdesc, self.parameters, connect=False,
                                         extra_platform_params=extra_plat_params)

        self.is_responsive = True

        with signal.wrap('TARGET_CONNECT'):
            self.target.connect()
        self.logger.info('Setting up target')
        self.target.setup()

        self.assistant = instantiate_assistant(tdesc, self.parameters, self.target)
