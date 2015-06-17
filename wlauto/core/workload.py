#    Copyright 2014-2015 ARM Limited
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
A workload is the unit of execution. It represents a set of activities are are performed
and measured together, as well as the necessary setup and teardown procedures. A single
execution of a workload produces one :class:`wlauto.core.result.WorkloadResult` that is populated with zero or more
:class:`wlauto.core.result.WorkloadMetric`\ s and/or
:class:`wlauto.core.result.Artifact`\s by the workload and active instrumentation.

"""
from wlauto.core.extension import Extension
from wlauto.exceptions import WorkloadError


class Workload(Extension):
    """
    This is the base class for the workloads executed by the framework.
    Each of the methods throwing NotImplementedError *must* be implemented
    by the derived classes.

    """

    supported_devices = []
    supported_platforms = []
    summary_metrics = []

    def __init__(self, device, **kwargs):
        """
        Creates a new Workload.

        :param device: the Device on which the workload will be executed.
        """
        super(Workload, self).__init__(**kwargs)
        if self.supported_devices and device.name not in self.supported_devices:
            raise WorkloadError('Workload {} does not support device {}'.format(self.name, device.name))
        if self.supported_platforms and device.platform not in self.supported_platforms:
            raise WorkloadError('Workload {} does not support platform {}'.format(self.name, device.platform))
        self.device = device

    def init_resources(self, context):
        """
        This method may be used to perform early resource discovery and initialization. This is invoked
        during the initial loading stage and before the device is ready, so cannot be used for any
        device-dependent initialization. This method is invoked before the workload instance is
        validated.

        """
        pass

    def initialize(self, context):
        """
        This method should be used to perform once-per-run initialization of a workload instance, i.e.,
        unlike ``setup()`` it will not be invoked on each iteration.

        """
        pass

    def setup(self, context):
        """
        Perform the setup necessary to run the workload, such as copying the necessary files
        to the device, configuring the environments, etc.

        This is also the place to perform any on-device checks prior to attempting to execute
        the workload.

        """
        pass

    def run(self, context):
        """Execute the workload. This is the method that performs the actual "work" of the"""
        pass

    def update_result(self, context):
        """
        Update the result within the specified execution context with the metrics
        form this workload iteration.

        """
        pass

    def teardown(self, context):
        """ Perform any final clean up for the Workload. """
        pass

    def finalize(self, context):
        pass

    def __str__(self):
        return '<Workload {}>'.format(self.name)

