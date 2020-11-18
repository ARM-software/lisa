#    Copyright 2013-2018 ARM Limited
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


# pylint: disable=W0613,E1101,E0203,W0201
import time

from wa import Instrument, Parameter
from wa.framework.exception import ConfigError, InstrumentError
from wa.framework.instrument import extremely_slow
from wa.utils.types import identifier


class DelayInstrument(Instrument):

    name = 'delay'
    description = """
    This instrument introduces a delay before beginning a new
    spec, a new job or before the main execution of a workload.

    The delay may be specified as either a fixed period or a temperature
    threshold that must be reached.

    Optionally, if an active cooling solution is available on the device to
    speed up temperature drop between runs, it may be controlled using this
    instrument.

    """

    parameters = [
        Parameter('temperature_file', default='/sys/devices/virtual/thermal/thermal_zone0/temp',
                  global_alias='thermal_temp_file',
                  description="""
                  Full path to the sysfile on the target that
                  contains the target's temperature.
                  """),
        Parameter('temperature_timeout', kind=int, default=600,
                  global_alias='thermal_timeout',
                  description="""
                  The timeout after which the instrument will
                  stop waiting even if the specified threshold temperature is
                  not reached. If this timeout is hit, then a warning will be
                  logged stating the actual temperature at which the timeout has
                  ended.
                  """),
        Parameter('temperature_poll_period', kind=int, default=5,
                  global_alias='thermal_sleep_time',
                  description="""
                  How long to sleep (in seconds) between polling
                  current target temperature.
                  """),
        Parameter('temperature_between_specs', kind=int, default=None,
                  global_alias='thermal_threshold_between_specs',
                  description="""
                  Temperature (in target-specific units) the
                  target must cool down to before the iteration spec will be
                  run.

                  If this is set to ``0`` then the devices initial temperature will
                  used as the threshold.

                  .. note:: This cannot be specified at the same time as
                            ``fixed_between_specs``
                  """),
        Parameter('fixed_between_specs', kind=int, default=None,
                  global_alias='fixed_delay_between_specs',
                  description="""
                  How long to sleep (in seconds) before starting
                  a new workload spec.

                  .. note:: This cannot be specified at the same time as
                            ``temperature_between_specs``
                  """),
        Parameter('temperature_between_jobs', kind=int, default=None,
                  global_alias='thermal_threshold_between_jobs',
                  aliases=['temperature_between_iterations'],
                  description="""
                  Temperature (in target-specific units) the
                  target must cool down to before the next job will be run.

                  If this is set to ``0`` then the devices initial temperature will
                  used as the threshold.

                  .. note:: This cannot be specified at the same time as
                            ``fixed_between_jobs``
                  """),
        Parameter('fixed_between_jobs', kind=int, default=None,
                  global_alias='fixed_delay_between_jobs',
                  aliases=['fixed_between_iterations'],
                  description="""
                  How long to sleep (in seconds) before starting each
                  new job.

                  .. note:: This cannot be specified at the same time as
                            ``temperature_between_jobs``
                  """),
        Parameter('fixed_before_start', kind=int, default=None,
                  global_alias='fixed_delay_before_start',
                  description="""
                  How long to sleep (in seconds) after setup for
                  an iteration has been performed but before running the
                  workload.

                  .. note:: This cannot be specified at the same time as
                            ``temperature_before_start``
                  """),
        Parameter('temperature_before_start', kind=int, default=None,
                  global_alias='thermal_threshold_before_start',
                  description="""
                  Temperature (in device-specific units) the
                  device must cool down to just before the actual workload
                  execution (after setup has been performed).

                  .. note:: This cannot be specified at the same time as
                            ``fixed_between_jobs``
                  """),
        Parameter('active_cooling', kind=bool, default=False,
                  description="""
                  This instrument supports an active cooling
                  solution while waiting for the device temperature to drop to
                  the threshold. If you wish to use this feature please ensure
                  the relevant module is installed on the device.
                  """),
    ]

    active_cooling_modules = ['mbed-fan', 'odroidxu3-fan']

    def initialize(self, context):
        if self.active_cooling:
            self.cooling = self._discover_cooling_module()
            if not self.cooling:
                msg = 'Cooling module not found on target. Please install one of the following modules: {}'
                raise InstrumentError(msg.format(self.active_cooling_modules))

        if self.temperature_between_jobs == 0:
            temp = self.target.read_int(self.temperature_file)
            self.logger.debug('Setting temperature threshold between jobs to {}'.format(temp))
            self.temperature_between_jobs = temp
        if self.temperature_between_specs == 0:
            temp = self.target.read_int(self.temperature_file)
            msg = 'Setting temperature threshold between workload specs to {}'
            self.logger.debug(msg.format(temp))
            self.temperature_between_specs = temp

    @extremely_slow
    def start(self, context):
        if self.fixed_before_start:
            msg = 'Waiting for {}s before running workload...'
            self.logger.info(msg.format(self.fixed_before_start))
            time.sleep(self.fixed_before_start)
        elif self.temperature_before_start:
            self.logger.info('Waiting for temperature drop before running workload...')
            self.wait_for_temperature(self.temperature_before_start)

    @extremely_slow
    def before_job(self, context):
        if self.fixed_between_specs and context.spec_changed:
            msg = 'Waiting for {}s before starting new spec...'
            self.logger.info(msg.format(self.fixed_between_specs))
            time.sleep(self.fixed_between_specs)
        elif self.temperature_between_jobs and context.spec_changed:
            self.logger.info('Waiting for temperature drop before starting new spec...')
            self.wait_for_temperature(self.temperature_between_jobs)
        elif self.fixed_between_jobs:
            msg = 'Waiting for {}s before starting new job...'
            self.logger.info(msg.format(self.fixed_between_jobs))
            time.sleep(self.fixed_between_jobs)
        elif self.temperature_between_jobs:
            self.logger.info('Waiting for temperature drop before starting new job...')
            self.wait_for_temperature(self.temperature_between_jobs)

    def wait_for_temperature(self, temperature):
        if self.active_cooling:
            self.cooling.start()
            self.do_wait_for_temperature(temperature)
            self.cooling.stop()
        else:
            self.do_wait_for_temperature(temperature)

    def do_wait_for_temperature(self, temperature):
        reading = self.target.read_int(self.temperature_file)
        waiting_start_time = time.time()
        while reading > temperature:
            self.logger.debug('target temperature: {}'.format(reading))
            if time.time() - waiting_start_time > self.temperature_timeout:
                self.logger.warning('Reached timeout; current temperature: {}'.format(reading))
                break
            time.sleep(self.temperature_poll_period)
            reading = self.target.read_int(self.temperature_file)

    def validate(self):
        if (self.temperature_between_specs is not None
                and self.fixed_between_specs is not None):
            raise ConfigError('Both fixed delay and thermal threshold specified for specs.')

        if (self.temperature_between_jobs is not None
                and self.fixed_between_jobs is not None):
            raise ConfigError('Both fixed delay and thermal threshold specified for jobs.')

        if (self.temperature_before_start is not None
                and self.fixed_before_start is not None):
            raise ConfigError('Both fixed delay and thermal threshold specified before start.')

        if not any([self.temperature_between_specs, self.fixed_between_specs,
                    self.temperature_between_jobs, self.fixed_between_jobs,
                    self.temperature_before_start, self.fixed_before_start]):
            raise ConfigError('Delay instrument is enabled, but no delay is specified.')

    def _discover_cooling_module(self):
        cooling_module = None
        for module in self.active_cooling_modules:
            if self.target.has(module):
                if not cooling_module:
                    cooling_module = getattr(self.target, identifier(module))
                else:
                    msg = 'Multiple cooling modules found "{}" "{}".'
                    raise InstrumentError(msg.format(cooling_module.name, module))
        return cooling_module
