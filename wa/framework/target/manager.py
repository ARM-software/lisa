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
                                            instantiate_target)
from wa.framework.target.info import TargetInfo
from wa.framework.target.runtime_config import (SysfileValuesRuntimeConfig,
                                                HotplugRuntimeConfig,
                                                CpufreqRuntimeConfig,
                                                CpuidleRuntimeConfig)
from wa.utils.misc import isiterable
from wa.utils.serializer import json


from devlib import LocalLinuxTarget, LinuxTarget, AndroidTarget
from devlib.utils.types import identifier
# from wa.target.manager import AndroidTargetManager, LinuxTargetManager


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
        self.target_name = name
        self.target = None
        self.assistant = None
        self.platform_name = None
        self.parameters = parameters
        self.disconnect = parameters.get('disconnect')
        self.info = TargetInfo()

        self._init_target()
        self._init_assistant()
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
        self.target.setup()

    def _init_assistant(self):
        # Create a corresponding target and target-assistant to help with
        # platformy stuff?
        if self.target.os == 'android':
            self.assistant = AndroidAssistant(self.target)
        elif self.target.os == 'linux':
            self.assistant = LinuxAssistant(self.target)  # pylint: disable=redefined-variable-type
        else:
            raise ValueError('Unknown Target OS: {}'.format(self.target.os))


class LinuxAssistant(object):

    name = 'linux-assistant'

    description = """
    Performs configuration, instrumentation, etc. during runs on Linux targets.
    """

    def __init__(self, target, **kwargs):
        self.target = target
    # parameters = [

    #     Parameter('disconnect', kind=bool, default=False,
    #               description="""
    #               Specifies whether the target should be disconnected from
    #               at the end of the run.
    #               """),
    # ]

    # runtime_config_cls = [
    #     # order matters
    #     SysfileValuesRuntimeConfig,
    #     HotplugRuntimeConfig,
    #     CpufreqRuntimeConfig,
    #     CpuidleRuntimeConfig,
    # ]

    # def __init__(self, target, context, **kwargs):
    #     # super(LinuxTargetManager, self).__init__(target, context, **kwargs)
    #     self.target = target
    #     self.context = context
    #     self.info = TargetInfo()
    #     self.runtime_configs = [cls(target) for cls in self.runtime_config_cls]

    # def __init__(self):
    #     # super(LinuxTargetManager, self).__init__(target, context, **kwargs)
    #     self.target = target
    #     self.info = TargetInfo()
    #     self.parameters = parameters

        # self.info = TargetInfo()
        # self.runtime_configs = [cls(target) for cls in self.runtime_config_cls]

    # def initialize(self):
    #     # self.runtime_configs = [cls(self.target) for cls in self.runtime_config_cls]
    #     # if self.parameters:
    #     self.logger.info('Connecting to the device')
    #     with signal.wrap('TARGET_CONNECT'):
    #         self.target.connect()
    #         self.info.load(self.target)
    #         # info_file = os.path.join(self.context.info_directory, 'target.json')
    #         # with open(info_file, 'w') as wfh:
    #         #     json.dump(self.info.to_pod(), wfh)

    # def finalize(self, runner):
    #     self.logger.info('Disconnecting from the device')
    #     if self.disconnect:
    #         with signal.wrap('TARGET_DISCONNECT'):
    #             self.target.disconnect()

    # def _add_parameters(self):
    #     for name, value in self.parameters.iteritems():
    #         self.add_parameter(name, value)

    # def validate_runtime_parameters(self, parameters):
    #     self.clear()
    #     for  name, value in parameters.iteritems():
    #         self.add_parameter(name, value)
    #     self.validate_parameters()

    # def set_runtime_parameters(self, parameters):
    #     self.clear()
    #     for  name, value in parameters.iteritems():
    #         self.add_parameter(name, value)
    #     self.set_parameters()

    # def clear_parameters(self):
    #     for cfg in self.runtime_configs:
    #         cfg.clear()

    # def add_parameter(self, name, value):
    #     for cfg in self.runtime_configs:
    #         if name in cfg.supported_parameters:
    #             cfg.add(name, value)
    #             return
    #     raise ConfigError('Unexpected runtime parameter "{}".'.format(name))

    # def validate_parameters(self):
    #     for cfg in self.runtime_configs:
    #         cfg.validate()

    # def set_parameters(self):
    #     for cfg in self.runtime_configs:
    #         cfg.set()


class AndroidAssistant(LinuxAssistant):

    name = 'android-assistant'
    description = """
    Extends ``LinuxTargetManager`` with Android-specific operations.
    """

    parameters = [
        Parameter('logcat_poll_period', kind=int,
                  description="""
                  If specified, logcat will cached in a temporary file on the 
                  host every ``logcat_poll_period`` seconds. This is useful for
                  longer job executions, where on-device logcat buffer may not be 
                  big enough to capture output for the entire execution.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(AndroidAssistant, self).__init__(target)
        self.logcat_poll_period = kwargs.get('logcat_poll_period', None)
        if self.logcat_poll_period:
            self.logcat_poller = LogcatPoller(target, self.logcat_poll_period)
        else:
            self.logcat_poller = None

    # def __init__(self, target, context, **kwargs):
    #     super(AndroidAssistant, self).__init__(target, context, **kwargs)
    #     self.logcat_poll_period = kwargs.get('logcat_poll_period', None)
    #     if self.logcat_poll_period:
    #         self.logcat_poller = LogcatPoller(target, self.logcat_poll_period)
    #     else:
    #         self.logcat_poller = None

    # def next_job(self, job):
    #     super(AndroidAssistant, self).next_job(job)
    #     if self.logcat_poller:
    #         self.logcat_poller.start()

    # def job_done(self, job):
    #     super(AndroidAssistant, self).job_done(job)
    #     if self.logcat_poller:
    #         self.logcat_poller.stop()
    #     outfile = os.path.join(self.context.output_directory, 'logcat.log')
    #     self.logger.debug('Dumping logcat to {}'.format(outfile))
    #     self.dump_logcat(outfile)
    #     self.clear()

    def dump_logcat(self, outfile):
        if self.logcat_poller:
            self.logcat_poller.write_log(outfile)
        else:
            self.target.dump_logcat(outfile)

    def clear_logcat(self):
        if self.logcat_poller:
            self.logcat_poller.clear_buffer()


class LogcatPoller(threading.Thread):

    def __init__(self, target, period=60, timeout=30):
        super(LogcatPoller, self).__init__()
        self.target = target
        self.logger = logging.getLogger('logcat')
        self.period = period
        self.timeout = timeout
        self.stop_signal = threading.Event()
        self.lock = threading.Lock()
        self.buffer_file = tempfile.mktemp()
        self.last_poll = 0
        self.daemon = True
        self.exc = None

    def start(self):
        self.logger.debug('starting polling')
        try:
            while True:
                if self.stop_signal.is_set():
                    break
                with self.lock:
                    current_time = time.time()
                    if (current_time - self.last_poll) >= self.period:
                        self.poll()
                time.sleep(0.5)
        except Exception:  # pylint: disable=W0703
            self.exc = WorkerThreadError(self.name, sys.exc_info())
        self.logger.debug('polling stopped')

    def stop(self):
        self.logger.debug('Stopping logcat polling')
        self.stop_signal.set()
        self.join(self.timeout)
        if self.is_alive():
            self.logger.error('Could not join logcat poller thread.')
        if self.exc:
            raise self.exc  # pylint: disable=E0702

    def clear_buffer(self):
        self.logger.debug('clearing logcat buffer')
        with self.lock:
            self.target.clear_logcat()
            with open(self.buffer_file, 'w') as _:  # NOQA
                pass

    def write_log(self, outfile):
        with self.lock:
            self.poll()
            if os.path.isfile(self.buffer_file):
                shutil.copy(self.buffer_file, outfile)
            else:  # there was no logcat trace at this time
                with open(outfile, 'w') as _:  # NOQA
                    pass

    def close(self):
        self.logger.debug('closing poller')
        if os.path.isfile(self.buffer_file):
            os.remove(self.buffer_file)

    def poll(self):
        self.last_poll = time.time()
        self.target.dump_logcat(self.buffer_file, append=True, timeout=self.timeout)
        self.target.clear_logcat()
