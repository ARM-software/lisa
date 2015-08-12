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


#pylint: disable=W0613,E1101,E0203,W0201
import time

from wlauto import Instrument, Parameter
from wlauto.exceptions import ConfigError, InstrumentError
from wlauto.utils.types import boolean


class DelayInstrument(Instrument):

    name = 'delay'
    description = """
    This instrument introduces a delay before executing either an iteration
    or all iterations for a spec.

    The delay may be specified as either a fixed period or a temperature
    threshold that must be reached.

    Optionally, if an active cooling solution is employed to speed up temperature drop between
    runs, it may be controlled using this instrument.

    """

    parameters = [
        Parameter('temperature_file', default='/sys/devices/virtual/thermal/thermal_zone0/temp',
                  global_alias='thermal_temp_file',
                  description="""Full path to the sysfile on the device that contains the device's
                  temperature."""),
        Parameter('temperature_timeout', kind=int, default=600,
                  global_alias='thermal_timeout',
                  description="""
                  The timeout after which the instrument will stop waiting even if the specified threshold
                  temperature is not reached. If this timeout is hit, then a warning will be logged stating
                  the actual temperature at which the timeout has ended.
                  """),
        Parameter('temperature_poll_period', kind=int, default=5,
                  global_alias='thermal_sleep_time',
                  description="""How long to sleep (in seconds) between polling current device temperature."""),
        Parameter('temperature_between_specs', kind=int, default=None,
                  global_alias='thermal_threshold_between_specs',
                  description="""
                  Temperature (in device-specific units) the device must cool down to before
                  the iteration spec will be run.

                  .. note:: This cannot be specified at the same time as ``fixed_between_specs``

                  """),
        Parameter('temperature_between_iterations', kind=int, default=None,
                  global_alias='thermal_threshold_between_iterations',
                  description="""
                  Temperature (in device-specific units) the device must cool down to before
                  the next spec will be run.

                  .. note:: This cannot be specified at the same time as ``fixed_between_iterations``

                  """),
        Parameter('temperature_before_start', kind=int, default=None,
                  global_alias='thermal_threshold_before_start',
                  description="""
                  Temperature (in device-specific units) the device must cool down to just before
                  the actual workload execution (after setup has been performed).

                  .. note:: This cannot be specified at the same time as ``fixed_between_iterations``

                  """),
        Parameter('fixed_between_specs', kind=int, default=None,
                  global_alias='fixed_delay_between_specs',
                  description="""
                  How long to sleep (in seconds) after all iterations for a workload spec have
                  executed.

                  .. note:: This cannot be specified at the same time as ``temperature_between_specs``

                  """),
        Parameter('fixed_between_iterations', kind=int, default=None,
                  global_alias='fixed_delay_between_iterations',
                  description="""
                  How long to sleep (in seconds) after each iterations for a workload spec has
                  executed.

                  .. note:: This cannot be specified at the same time as ``temperature_between_iterations``

                  """),
        Parameter('fixed_before_start', kind=int, default=None,
                  global_alias='fixed_delay_before_start',
                  description="""

                  How long to sleep (in seconds) after setup for an iteration has been perfromed but
                  before running the workload.

                  .. note:: This cannot be specified at the same time as ``temperature_before_start``

                  """),
        Parameter('active_cooling', kind=boolean, default=False,
                  global_alias='thermal_active_cooling',
                  description="""
                  This instrument supports an active cooling solution while waiting for the device temperature
                  to drop to the threshold. The solution involves an mbed controlling a fan. The mbed is signaled
                  over a serial port. If this solution is present in the setup, this should be set to ``True``.
                  """),
    ]

    def initialize(self, context):
        if self.temperature_between_iterations == 0:
            temp = self.device.get_sysfile_value(self.temperature_file, int)
            self.logger.debug('Setting temperature threshold between iterations to {}'.format(temp))
            self.temperature_between_iterations = temp
        if self.temperature_between_specs == 0:
            temp = self.device.get_sysfile_value(self.temperature_file, int)
            self.logger.debug('Setting temperature threshold between workload specs to {}'.format(temp))
            self.temperature_between_specs = temp

    def very_slow_on_iteration_start(self, context):
        if self.active_cooling:
            self.device.stop_active_cooling()
        if self.fixed_between_iterations:
            self.logger.debug('Waiting for a fixed period after iteration...')
            time.sleep(self.fixed_between_iterations)
        elif self.temperature_between_iterations:
            self.logger.debug('Waiting for temperature drop before iteration...')
            self.wait_for_temperature(self.temperature_between_iterations)

    def very_slow_on_spec_start(self, context):
        if self.active_cooling:
            self.device.stop_active_cooling()
        if self.fixed_between_specs:
            self.logger.debug('Waiting for a fixed period after spec execution...')
            time.sleep(self.fixed_between_specs)
        elif self.temperature_between_specs:
            self.logger.debug('Waiting for temperature drop before spec execution...')
            self.wait_for_temperature(self.temperature_between_specs)

    def very_slow_start(self, context):
        if self.active_cooling:
            self.device.stop_active_cooling()
        if self.fixed_before_start:
            self.logger.debug('Waiting for a fixed period after iteration...')
            time.sleep(self.fixed_before_start)
        elif self.temperature_before_start:
            self.logger.debug('Waiting for temperature drop before commencing execution...')
            self.wait_for_temperature(self.temperature_before_start)

    def wait_for_temperature(self, temperature):
        if self.active_cooling:
            self.device.start_active_cooling()
            self.do_wait_for_temperature(temperature)
            self.device.stop_active_cooling()
        else:
            self.do_wait_for_temperature(temperature)

    def do_wait_for_temperature(self, temperature):
        reading = self.device.get_sysfile_value(self.temperature_file, int)
        waiting_start_time = time.time()
        while reading > temperature:
            self.logger.debug('Device temperature: {}'.format(reading))
            if time.time() - waiting_start_time > self.temperature_timeout:
                self.logger.warning('Reached timeout; current temperature: {}'.format(reading))
                break
            time.sleep(self.temperature_poll_period)
            reading = self.device.get_sysfile_value(self.temperature_file, int)

    def validate(self):
        if (self.temperature_between_specs is not None and
                self.fixed_between_specs is not None):
            raise ConfigError('Both fixed delay and thermal threshold specified for specs.')

        if (self.temperature_between_iterations is not None and
                self.fixed_between_iterations is not None):
            raise ConfigError('Both fixed delay and thermal threshold specified for iterations.')

        if (self.temperature_before_start is not None and
                self.fixed_before_start is not None):
            raise ConfigError('Both fixed delay and thermal threshold specified before start.')

        if not any([self.temperature_between_specs, self.fixed_between_specs, self.temperature_before_start,
                    self.temperature_between_iterations, self.fixed_between_iterations,
                    self.fixed_before_start]):
            raise ConfigError('delay instrument is enabled, but no delay is specified.')

        if self.active_cooling and not self.device.has('active_cooling'):
            message = 'Your device does not support active cooling. Did you configure it with an approprite module?'
            raise InstrumentError(message)

