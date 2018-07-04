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

import os

from devlib import SerialTraceCollector

from wa import Instrument, Parameter, InstrumentError, hostside


class SerialMon(Instrument):

    name = 'serialmon'
    description = """
    Records the traffic on a serial connection

    The traffic on a serial connection is monitored and logged to a
    file. In the event that the device is reset, the instrument will
    stop monitoring during the reset, and will reconnect once the
    reset has completed. This is to account for devices (i.e., the
    Juno) which utilise the serial connection to reset the board.
    """

    parameters = [
        Parameter('serial_port', kind=str, default="/dev/ttyS0",
                  description="""
                  The serial device to monitor.
                  """),
        Parameter('baudrate', kind=int, default=115200,
                  description="""
                  The baud-rate to use when connecting to the serial connection.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(SerialMon, self).__init__(target, **kwargs)
        self._collector = SerialTraceCollector(target, self.serial_port, self.baudrate)
        self._collector.reset()

    def start_logging(self, context):
        self.logger.debug("Acquiring serial port ({})".format(self.serial_port))
        if self._collector.collecting:
            self.stop_logging(context)
        self._collector.start()

    def stop_logging(self, context, filename="serial.log", identifier="job"):
        self.logger.debug("Releasing serial port ({})".format(self.serial_port))
        if self._collector.collecting:
            self._collector.stop()

            outpath = os.path.join(context.output_directory, filename)
            self._collector.get_trace(outpath)
            context.add_artifact("{}_serial_log".format(identifier),
                                 outpath, kind="log")

    def on_run_start(self, context):
        self.start_logging(context)

    def before_job_queue_execution(self, context):
        self.stop_logging(context, "preamble_serial.log", "preamble")

    def after_job_queue_execution(self, context):
        self.start_logging(context)

    def on_run_end(self, context):
        self.stop_logging(context, "postamble_serial.log", "postamble")

    def on_job_start(self, context):
        self.start_logging(context)

    def on_job_end(self, context):
        self.stop_logging(context)

    @hostside
    def before_reboot(self, context):
        self.stop_logging(context)
