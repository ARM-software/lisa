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
# pylint: disable=W0613,attribute-defined-outside-init
import os
import subprocess
import shutil

from wlauto import Instrument, Parameter
from wlauto.utils.types import list_of_strings, boolean
from wlauto.utils.misc import check_output
from wlauto.exceptions import ConfigError, InstrumentError


class systrace(Instrument):
    name = 'systrace'
    description = """
      This instrument uses systrace.py from the android SDK to dump atrace
      output.

      Note: This is unlikely to work on devices that have an android build built
            before 15-May-2015. Before this date there was a bug with running
            atrace asynchronously.

      From developer.android.com:
      The Systrace tool helps analyze the performance of your application by
      capturing and displaying execution times of your applications processes
      and other Android system processes. The tool combines data from the
      Android kernel such as the CPU scheduler, disk activity, and application
      threads to generate an HTML report that shows an overall picture of an
      Android device's system processes for a given period of time.
    """
    parameters = [
        Parameter('buffer_size', kind=int, default=1024,
                  description="""
                  Use a trace buffer size of N kilobytes. This option lets you
                  limit the total size of the data collected during a trace.
                    """),
        Parameter('use_circular_buffer', kind=boolean, default=False,
                  description="""
                  When true trace data will be put into a circular buffer such
                  that when it overflows it will start overwriting the beginning
                  of the buffer.
                  """),
        Parameter('kernel_functions', kind=list_of_strings,
                  description="""
                  Specify the names of kernel functions to trace.
                  """),
        Parameter('categories', kind=list_of_strings,
                  default=["freq", "sched"],
                  description="""
                  A list of the categories you wish to trace.
                  """),
        Parameter('app_names', kind=list_of_strings,
                  description="""
                  Enable tracing for applications, specified as a
                  comma-separated list of package names. The apps must contain
                  tracing instrumentation calls from the Trace class. For more
                  information, see
                  http://developer.android.com/tools/debugging/systrace.html#app-trace
                  """),
        Parameter("ignore_signals", kind=boolean, default=False,
                  description="""
                  This will cause atrace to ignore ``SIGHUP``, ``SIGINT``,
                  ``SIGQUIT`` and ``SIGTERM``.
                  """),
        Parameter("compress_trace", kind=boolean, default=True,
                  description="""
                  Compresses atrace output. This *greatly* decreases the time
                  it takes to pull results from a device but the resulting txt
                  file is not human readable.
                  """)
    ]

    def initialize(self, context):
        cmd_options = {}
        if context.device.get_sdk_version() >= 23:
            # Set up command line options
            if self.app_names:
                cmd_options["-a"] = ",".join(self.app_names)
            if self.buffer_size:
                cmd_options["-b"] = self.buffer_size
            if self.use_circular_buffer:
                cmd_options["-c"] = None
            if self.kernel_functions:
                cmd_options["-k"] = ",".join(self.kernel_functions)
            if self.ignore_signals:
                cmd_options["-n"] = None

            # Generate commands
            opt_string = ''.join(['{} {} '.format(name, value or "")
                                  for name, value in cmd_options.iteritems()])
            self.start_cmd = "atrace --async_start {} {}".format(opt_string,
                                                                 " ".join(self.categories))
            self.output_file = os.path.join(self.device.working_directory, "atrace.txt")
            self.stop_cmd = "atrace --async_stop {} > {}".format("-z" if self.compress_trace else "",
                                                                 self.output_file)

            # Check if provided categories are available on the device
            available_categories = [cat.strip().split(" - ")[0] for cat in
                                    context.device.execute("atrace --list_categories").splitlines()]
            for category in self.categories:
                if category not in available_categories:
                    raise ConfigError("Unknown category '{}'; Must be one of: {}"
                                      .format(category, available_categories))
        else:
            raise InstrumentError("Only android devices with an API level >= 23 can use systrace properly")

    def setup(self, context):
        self.device.execute("atrace --async_dump")

    def start(self, context):
        result = self.device.execute(self.start_cmd)
        if "error" in result:
            raise InstrumentError(result)

    def stop(self, context):
        self.p = self.device.execute(self.stop_cmd, background=True)

    def update_result(self, context):  # pylint: disable=r0201
        self.logger.debug("Waiting for atrace to finish dumping data")
        self.p.wait()
        context.device.pull_file(self.output_file, context.output_directory)
        cmd = "python {} --from-file={} -o {}"
        cmd = cmd.format(os.path.join(os.environ['ANDROID_HOME'],
                                      "platform-tools/systrace/systrace.py"),
                         os.path.join(context.output_directory, "atrace.txt"),
                         os.path.join(context.output_directory, "systrace.html"))
        self.logger.debug(cmd)
        _, error = check_output(cmd.split(" "), timeout=10)
        if error:
            raise InstrumentError(error)

        context.add_iteration_artifact('atrace.txt',
                                       path=os.path.join(context.output_directory,
                                                         "atace.txt"),
                                       kind='data',
                                       description='atrace dump.')
        context.add_iteration_artifact('systrace.html',
                                       path=os.path.join(context.output_directory,
                                                         "systrace.html"),
                                       kind='data',
                                       description='Systrace HTML report.')
