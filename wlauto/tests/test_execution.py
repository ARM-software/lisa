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


# pylint: disable=E0611
# pylint: disable=R0201
# pylint: disable=protected-access
# pylint: disable=abstract-method
# pylint: disable=attribute-defined-outside-init
# pylint: disable=no-member
from unittest import TestCase
from nose.tools import assert_equal, assert_raises, raises

from wlauto.core.execution import BySpecRunner, ByIterationRunner
from wlauto.exceptions import DeviceError
from wlauto.core.configuration import WorkloadRunSpec, RebootPolicy
from wlauto.core.instrumentation import Instrument
from wlauto.core.device import Device, DeviceMeta
from wlauto.core import instrumentation, signal
from wlauto.core.workload import Workload
from wlauto.core.result import IterationResult
from wlauto.core.signal import Signal


class SignalCatcher(Instrument):
    name = 'Signal Catcher'

    def __init__(self):
        Instrument.__init__(self, None)
        self.signals_received = []
        for sig in signal.__dict__.values():
            if isinstance(sig, Signal):
                signal.connect(self.handler, sig)

    def handler(self, *_, **kwargs):
        self.signals_received.append(kwargs.pop('signal').name)


class Mock(object):
    def __init__(self):
        self.__members = {}

    def __getattr__(self, name):
        if name not in self.__members:
            self.__members[name] = Mock()
        return self.__members[name]

    def __call__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter([])


class BadDeviceMeta(DeviceMeta):

    @classmethod
    def _implement_virtual(mcs, cls, bases):
        """
        This version of _implement_virtual does not inforce "call global virutals only once"
        policy, so that intialize() and finalize() my be invoked multiple times to test that
        the errors they generated are handled correctly.

        """
        # pylint: disable=cell-var-from-loop,unused-argument
        methods = {}
        for vmname in mcs.virtual_methods:
            clsmethod = getattr(cls, vmname, None)
            if clsmethod:
                basemethods = [getattr(b, vmname) for b in bases if hasattr(b, vmname)]
                methods[vmname] = [bm for bm in basemethods if bm != clsmethod]
                methods[vmname].append(clsmethod)

                def generate_method_wrapper(vname):
                    name__ = vmname

                    def wrapper(self, *args, **kwargs):
                        for dm in methods[name__]:
                            dm(self, *args, **kwargs)
                    return wrapper
                setattr(cls, vmname, generate_method_wrapper(vmname))


class BadDevice(Device):

    __metaclass__ = BadDeviceMeta

    def __init__(self, when_to_fail, exception=DeviceError):
        #pylint: disable=super-init-not-called
        self.when_to_fail = when_to_fail
        self.exception = exception

    def connect(self):
        if self.when_to_fail == 'connect':
            raise self.exception("Connection failure")

    def initialize(self, _):
        if self.when_to_fail == 'initialize':
            raise self.exception("Initialisation failure")

    def get_properties(self, _):
        if self.when_to_fail == 'get_properties':
            raise self.exception("Failure getting propeties")

    def start(self):
        if self.when_to_fail == 'start':
            raise self.exception("Start failure")

    def set_device_parameters(self, **_):
        if self.when_to_fail == 'set_device_parameters':
            raise self.exception("Failure setting parameter")

    def stop(self):
        if self.when_to_fail == 'stop':
            raise self.exception("Stop failure")

    def disconnect(self):
        if self.when_to_fail == 'disconnect':
            raise self.exception("Disconnection failure")

    def ping(self):
        return True


class BadWorkload(Workload):

    def __init__(self, exception, when_to_fail):
        #pylint: disable=super-init-not-called
        self.exception = exception
        self.when_to_fail = when_to_fail

    def setup(self, _):
        if "setup" in self.when_to_fail:
            raise self.exception("Setup failed")

    def run(self, _):
        if "run" in self.when_to_fail:
            raise self.exception("Run failed")

    def update_result(self, _):
        if "update_result" in self.when_to_fail:
            raise self.exception("Result update failed")

    def teardown(self, _):
        if "teardown" in self.when_to_fail:
            raise self.exception("Teardown failed")


class RunnerTest(TestCase):

    errors = 0

    def signal_check(self, expected_signals, workloads, reboot_policy="never", runner_class=BySpecRunner):
        context = Mock()
        context.reboot_policy = RebootPolicy(reboot_policy)
        context.config.workload_specs = workloads
        context.config.retry_on_status = []

        instrument = _instantiate(SignalCatcher)
        instrumentation.install(instrument)

        runner = runner_class(Mock(), context, Mock())
        runner.init_queue(context.config.workload_specs)

        try:
            runner.run()
        finally:
            instrumentation.uninstall(instrument)

        assert_equal(instrument.signals_received, expected_signals)

    def test_single_run(self):
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]
        workloads = [WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=['Signal Catcher'])]
        workloads[0]._workload = Mock()

        self.signal_check(expected_signals, workloads)

    def test_multiple_run_byspec(self):
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]
        workloads = [
            WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='2', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='3', number_of_iterations=3, instrumentation=['Signal Catcher'])
        ]
        workloads[0]._workload = Mock()
        workloads[1]._workload = Mock()
        workloads[2]._workload = Mock()

        self.signal_check(expected_signals, workloads)

    def test_multiple_run_byiteration(self):
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]
        workloads = [
            WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='2', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='3', number_of_iterations=3, instrumentation=['Signal Catcher']),
        ]
        workloads[0]._workload = Mock()
        workloads[1]._workload = Mock()
        workloads[2]._workload = Mock()

        self.signal_check(expected_signals, workloads, runner_class=ByIterationRunner)

    def test_reboot_policies(self):
        expected_never = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        expected_initial = [
            signal.RUN_START.name,
            signal.BEFORE_INITIAL_BOOT.name,
            signal.BEFORE_BOOT.name,
            signal.SUCCESSFUL_BOOT.name,
            signal.AFTER_BOOT.name,
            signal.SUCCESSFUL_INITIAL_BOOT.name,
            signal.AFTER_INITIAL_BOOT.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        expected_each_spec = [
            signal.RUN_START.name,
            signal.BEFORE_INITIAL_BOOT.name,
            signal.BEFORE_BOOT.name,
            signal.SUCCESSFUL_BOOT.name,
            signal.AFTER_BOOT.name,
            signal.SUCCESSFUL_INITIAL_BOOT.name,
            signal.AFTER_INITIAL_BOOT.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.BEFORE_BOOT.name,
            signal.SUCCESSFUL_BOOT.name,
            signal.AFTER_BOOT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        expected_each_iteration = [
            signal.RUN_START.name,
            signal.BEFORE_INITIAL_BOOT.name,
            signal.BEFORE_BOOT.name,
            signal.SUCCESSFUL_BOOT.name,
            signal.AFTER_BOOT.name,
            signal.SUCCESSFUL_INITIAL_BOOT.name,
            signal.AFTER_INITIAL_BOOT.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.BEFORE_BOOT.name,
            signal.SUCCESSFUL_BOOT.name,
            signal.AFTER_BOOT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.BEFORE_BOOT.name,
                signal.SUCCESSFUL_BOOT.name,
                signal.AFTER_BOOT.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        workloads = [
            WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='2', number_of_iterations=1, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='3', number_of_iterations=2, instrumentation=['Signal Catcher'])
        ]
        workloads[0]._workload = Mock()
        workloads[1]._workload = Mock()
        workloads[2]._workload = Mock()

        self.signal_check(expected_never, workloads[0:1], reboot_policy="never")
        self.signal_check(expected_initial, workloads[0:1], reboot_policy="initial")
        self.signal_check(expected_each_spec, workloads[0:2], reboot_policy="each_spec")
        self.signal_check(expected_each_iteration, workloads[1:3], reboot_policy="each_iteration")

    def test_spec_skipping(self):
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        workloads = [
            WorkloadRunSpec(id='1', number_of_iterations=5, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='2', number_of_iterations=1, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='3', number_of_iterations=4, instrumentation=['Signal Catcher'])
        ]

        workloads[0]._workload = Mock()
        workloads[1]._workload = Mock()
        workloads[2]._workload = Mock()
        workloads[0].enabled = False
        workloads[2].enabled = False

        self.signal_check(expected_signals, workloads)

    def test_bad_workload_status(self):
        workloads = [
            WorkloadRunSpec(id='1', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='2', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='3', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='4', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='5', number_of_iterations=2, instrumentation=['Signal Catcher'])
        ]

        workloads[0]._workload = BadWorkload(Exception, ["setup"])
        workloads[1]._workload = BadWorkload(Exception, ["run"])
        workloads[2]._workload = BadWorkload(Exception, ["update_result"])
        workloads[3]._workload = BadWorkload(Exception, ["teardown"])
        workloads[4]._workload = Mock()

        context = Mock()
        context.reboot_policy = RebootPolicy("never")
        context.config.workload_specs = workloads

        runner = BySpecRunner(Mock(), context, Mock())
        runner.init_queue(context.config.workload_specs)

        instrument = _instantiate(SignalCatcher)
        instrumentation.install(instrument)

        try:
            runner.run()
        finally:
            instrumentation.uninstall(instrument)

        #Check queue was handled correctly
        assert_equal(len(runner.completed_jobs), 10)
        assert_equal(len(runner.job_queue), 0)

        #Check job status'
        expected_status = [
            IterationResult.FAILED, IterationResult.SKIPPED,
            IterationResult.FAILED, IterationResult.FAILED,
            IterationResult.PARTIAL, IterationResult.PARTIAL,
            IterationResult.NONCRITICAL, IterationResult.NONCRITICAL,
            IterationResult.OK, IterationResult.OK
        ]
        for i in range(0, len(runner.completed_jobs)):
            assert_equal(runner.completed_jobs[i].result.status, expected_status[i])

        #Check signals were sent correctly
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,  # Fail Setup
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                signal.ITERATION_END.name,
                #Skipped iteration
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,  # Fail Run
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    #signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name, - not sent because run failed
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    #signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name, - not sent because run failed
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,  # Fail Result Update
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,  # Fail Teardown
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.WORKLOAD_SPEC_START.name,  # OK
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        assert_equal(expected_signals, instrument.signals_received)

    def test_CTRL_C(self):
        workloads = [
            WorkloadRunSpec(id='1', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='2', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='3', number_of_iterations=2, instrumentation=['Signal Catcher']),
            WorkloadRunSpec(id='4', number_of_iterations=2, instrumentation=['Signal Catcher']),
        ]

        workloads[0]._workload = BadWorkload(KeyboardInterrupt, ["setup"])
        workloads[1]._workload = BadWorkload(KeyboardInterrupt, ["run"])
        workloads[2]._workload = BadWorkload(KeyboardInterrupt, ["update_result"])
        workloads[3]._workload = BadWorkload(KeyboardInterrupt, ["teardown"])

        expected_status = [IterationResult.ABORTED, IterationResult.ABORTED]

        expected_signals = [
            [
                signal.RUN_START.name,
                signal.RUN_INIT.name,
                signal.WORKLOAD_SPEC_START.name,
                    signal.ITERATION_START.name,
                        signal.BEFORE_WORKLOAD_SETUP.name,
                        signal.AFTER_WORKLOAD_SETUP.name,
                    signal.ITERATION_END.name,
                signal.WORKLOAD_SPEC_END.name,
                signal.RUN_FIN.name,
                signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
                signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
                signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
                signal.RUN_END.name
            ],
            [
                signal.RUN_START.name,
                signal.RUN_INIT.name,
                signal.WORKLOAD_SPEC_START.name,
                    signal.ITERATION_START.name,
                        signal.BEFORE_WORKLOAD_SETUP.name,
                        signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                        signal.AFTER_WORKLOAD_SETUP.name,
                        signal.BEFORE_WORKLOAD_EXECUTION.name,
                        signal.AFTER_WORKLOAD_EXECUTION.name,
                        signal.BEFORE_WORKLOAD_TEARDOWN.name,
                        signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                        signal.AFTER_WORKLOAD_TEARDOWN.name,
                    signal.ITERATION_END.name,
                signal.WORKLOAD_SPEC_END.name,
                signal.RUN_FIN.name,
                signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
                signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
                signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
                signal.RUN_END.name
            ],
            [
                signal.RUN_START.name,
                signal.RUN_INIT.name,
                signal.WORKLOAD_SPEC_START.name,
                    signal.ITERATION_START.name,
                        signal.BEFORE_WORKLOAD_SETUP.name,
                        signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                        signal.AFTER_WORKLOAD_SETUP.name,
                        signal.BEFORE_WORKLOAD_EXECUTION.name,
                        signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                        signal.AFTER_WORKLOAD_EXECUTION.name,
                        signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                        signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                        signal.BEFORE_WORKLOAD_TEARDOWN.name,
                        signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                        signal.AFTER_WORKLOAD_TEARDOWN.name,
                    signal.ITERATION_END.name,
                signal.WORKLOAD_SPEC_END.name,
                signal.RUN_FIN.name,
                signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
                signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
                signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
                signal.RUN_END.name
            ],
            [
                signal.RUN_START.name,
                signal.RUN_INIT.name,
                signal.WORKLOAD_SPEC_START.name,
                    signal.ITERATION_START.name,
                        signal.BEFORE_WORKLOAD_SETUP.name,
                        signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                        signal.AFTER_WORKLOAD_SETUP.name,
                        signal.BEFORE_WORKLOAD_EXECUTION.name,
                        signal.SUCCESSFUL_WORKLOAD_EXECUTION.name,
                        signal.AFTER_WORKLOAD_EXECUTION.name,
                        signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                        signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE.name,
                        signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                        signal.BEFORE_WORKLOAD_TEARDOWN.name,
                        signal.AFTER_WORKLOAD_TEARDOWN.name,
                    signal.ITERATION_END.name,
                signal.WORKLOAD_SPEC_END.name,
                signal.RUN_FIN.name,
                signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
                signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
                signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
                signal.RUN_END.name
            ],
        ]

        for i in xrange(0, len(workloads)):
            context = Mock()
            context.reboot_policy = RebootPolicy("never")
            context.config.workload_specs = [workloads[i]]

            runner = BySpecRunner(Mock(), context, Mock())
            runner.init_queue(context.config.workload_specs)

            instrument = _instantiate(SignalCatcher)
            instrumentation.install(instrument)

            try:
                runner.run()
            finally:
                instrumentation.uninstall(instrument)

            #Check queue was handled correctly
            assert_equal(len(runner.completed_jobs), 2)
            assert_equal(len(runner.job_queue), 0)

            #check correct signals were sent
            assert_equal(expected_signals[i], instrument.signals_received)

            #Check job status'
            for j in range(0, len(runner.completed_jobs)):
                assert_equal(runner.completed_jobs[j].result.status, expected_status[j])

    def test_no_teardown_after_setup_fail(self):
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]

        workloads = [WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=['Signal Catcher'])]
        workloads[0]._workload = BadWorkload(Exception, ["setup"])

        self.signal_check(expected_signals, workloads)

    def test_teardown_on_run_and_result_update_fail(self):
        expected_signals = [
            signal.RUN_START.name,
            signal.RUN_INIT.name,
            signal.WORKLOAD_SPEC_START.name,
                signal.ITERATION_START.name,
                    signal.BEFORE_WORKLOAD_SETUP.name,
                    signal.SUCCESSFUL_WORKLOAD_SETUP.name,
                    signal.AFTER_WORKLOAD_SETUP.name,
                    signal.BEFORE_WORKLOAD_EXECUTION.name,
                    signal.AFTER_WORKLOAD_EXECUTION.name,
                    signal.BEFORE_WORKLOAD_RESULT_UPDATE.name,
                    signal.AFTER_WORKLOAD_RESULT_UPDATE.name,
                    signal.BEFORE_WORKLOAD_TEARDOWN.name,
                    signal.SUCCESSFUL_WORKLOAD_TEARDOWN.name,
                    signal.AFTER_WORKLOAD_TEARDOWN.name,
                signal.ITERATION_END.name,
            signal.WORKLOAD_SPEC_END.name,
            signal.RUN_FIN.name,
            signal.BEFORE_OVERALL_RESULTS_PROCESSING.name,
            signal.SUCCESSFUL_OVERALL_RESULTS_PROCESSING.name,
            signal.AFTER_OVERALL_RESULTS_PROCESSING.name,
            signal.RUN_END.name
        ]
        workloads = [WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=['Signal Catcher'])]
        workloads[0]._workload = BadWorkload(Exception, ["run", "update_result"])

        self.signal_check(expected_signals, workloads)

    def bad_device(self, method):
        workloads = [WorkloadRunSpec(id='1', number_of_iterations=1, instrumentation=[])]
        workloads[0]._workload = Mock()

        context = Mock()
        context.reboot_policy = RebootPolicy("never")
        context.config.workload_specs = workloads

        runner = BySpecRunner(BadDevice(method), context, Mock())
        runner.init_queue(context.config.workload_specs)
        runner.run()

    @raises(DeviceError)
    def test_bad_connect(self):
        assert_raises(DeviceError, self.bad_device('connect'))

    @raises(DeviceError)
    def test_bad_initialize(self):
        assert_raises(DeviceError, self.bad_device('initialize'))

    def test_bad_start(self):
        self.bad_device('start')  # error must not propagate

    def test_bad_stop(self):
        self.bad_device('stop')  # error must not propagate

    def test_bad_disconnect(self):
        self.bad_device('disconnect')  # error must not propagate

    @raises(DeviceError)
    def test_bad_get_properties(self):
        assert_raises(DeviceError, self.bad_device('get_properties'))


def _instantiate(cls, *args, **kwargs):
    # Needed to get around Extension's __init__ checks
    return cls(*args, **kwargs)
