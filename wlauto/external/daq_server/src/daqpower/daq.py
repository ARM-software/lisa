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

"""
Creates a new DAQ device class. This class assumes that there is a
DAQ connected and mapped as Dev1. It assumes a specific syndesmology on the DAQ (it is not
meant to be a generic DAQ interface). The following diagram shows the wiring for one DaqDevice
port::

Port 0
========
|   A0+ <--- Vr -------------------------|
|                                        |
|   A0- <--- GND -------------------//   |
|                                        |
|   A1+ <--- V+ ------------|-------V+   |
|                   r       |            |
|   A1- <--- Vr --/\/\/\----|            |
|             |                          |
|             |                          |
|             |--------------------------|
========

:number_of_ports: The number of ports connected on the DAQ. Each port requires 2 DAQ Channels
                    one for the source voltage and one for the Voltage drop over the
                    resistor r (V+ - Vr) allows us to detect the current.
:resistor_value: The resistance of r. Typically a few milliOhm
:downsample: The number of samples combined to create one Power point. If set to one
                each sample corresponds to one reported power point.
:sampling_rate: The rate at which DAQ takes a sample from each channel.

"""
# pylint: disable=F0401,E1101,W0621
import os
import sys
import csv
import time
import threading
from Queue import Queue, Empty

import numpy

from PyDAQmx import Task
from PyDAQmx.DAQmxFunctions import DAQmxGetSysDevNames
from PyDAQmx.DAQmxTypes import int32, byref, create_string_buffer
from PyDAQmx.DAQmxConstants import (DAQmx_Val_Diff, DAQmx_Val_Volts, DAQmx_Val_GroupByScanNumber, DAQmx_Val_Auto,
                                    DAQmx_Val_Acquired_Into_Buffer, DAQmx_Val_Rising, DAQmx_Val_ContSamps)

from daqpower import log

def list_available_devices():
    """Returns the list of DAQ devices visible to the driver."""
    bufsize = 2048  # Should be plenty for all but the most pathalogical of situations.
    buf = create_string_buffer('\000' * bufsize)
    DAQmxGetSysDevNames(buf, bufsize)
    return buf.value.split(',')


class ReadSamplesTask(Task):

    def __init__(self, config, consumer):
        Task.__init__(self)
        self.config = config
        self.consumer = consumer
        self.sample_buffer_size = (self.config.sampling_rate + 1) * self.config.number_of_ports * 2
        self.samples_read = int32()
        self.remainder = []
        # create voltage channels
        for i in xrange(0, 2 * self.config.number_of_ports, 2):
            self.CreateAIVoltageChan('{}/ai{}'.format(config.device_id, config.channel_map[i]),
                                     '', DAQmx_Val_Diff,
                                     -config.v_range, config.v_range,
                                     DAQmx_Val_Volts, None)
            self.CreateAIVoltageChan('{}/ai{}'.format(config.device_id, config.channel_map[i + 1]),
                                     '', DAQmx_Val_Diff,
                                     -config.dv_range, config.dv_range,
                                     DAQmx_Val_Volts, None)
        # configure sampling rate
        self.CfgSampClkTiming('',
                              self.config.sampling_rate,
                              DAQmx_Val_Rising,
                              DAQmx_Val_ContSamps,
                              self.config.sampling_rate)
        # register callbacks
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.config.sampling_rate // 2, 0)
        self.AutoRegisterDoneEvent(0)

    def EveryNCallback(self):
        samples_buffer = numpy.zeros((self.sample_buffer_size,), dtype=numpy.float64)
        self.ReadAnalogF64(DAQmx_Val_Auto, 0.0, DAQmx_Val_GroupByScanNumber, samples_buffer,
                           self.sample_buffer_size, byref(self.samples_read), None)
        self.consumer.write((samples_buffer, self.samples_read.value))

    def DoneCallback(self, status):  # pylint: disable=W0613,R0201
        return 0  # The function should return an integer


class AsyncWriter(threading.Thread):

    def __init__(self, wait_period=1):
        super(AsyncWriter, self).__init__()
        self.daemon = True
        self.wait_period = wait_period
        self.running = threading.Event()
        self._stop_signal = threading.Event()
        self._queue = Queue()

    def write(self, stuff):
        if self._stop_signal.is_set():
            raise IOError('Attempting to writer to {} after it has been closed.'.format(self.__class__.__name__))
        self._queue.put(stuff)

    def do_write(self, stuff):
        raise NotImplementedError()

    def run(self):
        self.running.set()
        while True:
            if self._stop_signal.is_set() and self._queue.empty():
                break
            try:
                self.do_write(self._queue.get(block=True, timeout=self.wait_period))
            except Empty:
                pass  # carry on
        self.running.clear()

    def stop(self):
        self._stop_signal.set()

    def wait(self):
        while self.running.is_set():
            time.sleep(self.wait_period)


class PortWriter(object):

    def __init__(self, path):
        self.path = path
        self.fh = open(path, 'w', 0)
        self.writer = csv.writer(self.fh)
        self.writer.writerow(['power', 'voltage'])

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.fh.close()

    def __del__(self):
        self.close()


class SamplePorcessorError(Exception):
    pass


class SampleProcessor(AsyncWriter):

    def __init__(self, resistor_values, output_directory, labels):
        super(SampleProcessor, self).__init__()
        self.resistor_values = resistor_values
        self.output_directory = output_directory
        self.labels = labels
        self.number_of_ports = len(resistor_values)
        if len(self.labels) != self.number_of_ports:
            message = 'Number of labels ({}) does not match number of ports ({}).'
            raise SamplePorcessorError(message.format(len(self.labels), self.number_of_ports))
        self.port_writers = []

    def do_write(self, sample_tuple):
        samples, number_of_samples = sample_tuple
        for i in xrange(0, number_of_samples * self.number_of_ports * 2, self.number_of_ports * 2):
            for j in xrange(self.number_of_ports):
                V = float(samples[i + 2 * j])
                DV = float(samples[i + 2 * j + 1])
                P = V * (DV / self.resistor_values[j])
                self.port_writers[j].write([P, V])

    def start(self):
        for label in self.labels:
            port_file = self.get_port_file_path(label)
            writer = PortWriter(port_file)
            self.port_writers.append(writer)
        super(SampleProcessor, self).start()

    def stop(self):
        super(SampleProcessor, self).stop()
        self.wait()
        for writer in self.port_writers:
            writer.close()

    def get_port_file_path(self, port_id):
        if port_id in self.labels:
            return os.path.join(self.output_directory, port_id + '.csv')
        else:
            raise SamplePorcessorError('Invalid port ID: {}'.format(port_id))

    def __del__(self):
        self.stop()


class DaqRunner(object):

    @property
    def number_of_ports(self):
        return self.config.number_of_ports

    def __init__(self, config, output_directory):
        self.config = config
        self.processor = SampleProcessor(config.resistor_values, output_directory, config.labels)
        self.task = ReadSamplesTask(config, self.processor)
        self.is_running = False

    def start(self):
        log.debug('Starting sample processor.')
        self.processor.start()
        log.debug('Starting DAQ Task.')
        self.task.StartTask()
        self.is_running = True
        log.debug('Runner started.')

    def stop(self):
        self.is_running = False
        log.debug('Stopping DAQ Task.')
        self.task.StopTask()
        log.debug('Stopping sample processor.')
        self.processor.stop()
        log.debug('Runner stopped.')

    def get_port_file_path(self, port_id):
        return self.processor.get_port_file_path(port_id)


if __name__ == '__main__':
    from collections import namedtuple
    DeviceConfig = namedtuple('DeviceConfig', ['device_id', 'channel_map', 'resistor_values',
                                               'v_range', 'dv_range', 'sampling_rate',
                                               'number_of_ports', 'labels'])
    channel_map = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
    resistor_values = [0.005]
    labels = ['PORT_0']
    dev_config = DeviceConfig('Dev1', channel_map, resistor_values, 2.5, 0.2, 10000, len(resistor_values), labels)
    if not len(sys.argv) == 3:
        print 'Usage: {} OUTDIR DURATION'.format(os.path.basename(__file__))
        sys.exit(1)
    output_directory = sys.argv[1]
    duration = float(sys.argv[2])

    print "Avialable devices:", list_availabe_devices()
    runner = DaqRunner(dev_config, output_directory)
    runner.start()
    time.sleep(duration)
    runner.stop()
