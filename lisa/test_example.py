# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from lisa.utils import ArtifactPath, show_doc
from lisa.datautils import df_filter_task_ids
from lisa.trace import FtraceCollector, requires_events
from lisa.wlgen.rta import Periodic
from lisa.tests.base import RTATestBundle, ResultBundle
from lisa.target import Target
from lisa.analysis.load_tracking import LoadTrackingAnalysis

"""
This module provides a LISA synthetic test example, heavily commented to show
how to use the main APIs.
"""

################################################################################
# It's a good idea to open the online doc in your browser when reading
# this example:
# https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/
#
# Also, lisa.utils.show_doc() can be called on any class/function to open the
# corresponding documentation in a browser.
################################################################################


class ExampleTestBundle(RTATestBundle):
    """
    The test bundle contains the data the test will work on. See
    :class:`lisa.tests.base.TestBundle` for design notes.

    This example derives from :class:`lisa.tests.base.RTATestBundle`, so it
    gains some ``rt-app``-specific and ftrace capabilities.
    """

    task_prefix = 'exmpl'
    "Prefix used for rt-app task names"

    # res_dir and plat_info are "mandatory" parameters of all TestBundle, but
    # the other ones are specific to a given use case.
    def __init__(self, res_dir, plat_info, shell_output):
        # This must be called, don't set res_dir or plat_info yourself
        super().__init__(res_dir, plat_info)

        self.shell_output = shell_output

    @classmethod
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath, ftrace_coll: FtraceCollector = None) -> 'ExampleTestBundle':
        """
        This class method is the main way of creating a :class:`ExampleTestBundle`.

        It takes a first (positional) ``target`` parameter, which is a live
        :class:`lisa.target.Target` object. It can be used to manipulate a
        remote device such as a development board, to run workloads on it,
        manipulate sysfs entries and so on.

        **All other parameters are keyword-only**
        This means they must appear after the lone ``*`` in the parameter list.

        ``res_dir`` stands for "result directory" and is a location where the
        bundle can store some artifacts collected from the target. The bundle
        can rely on that folder being populated by this method.

        The "'ExampleTestBundle'" return annotation tells the test runner that
        this class method acts as a factory of :class:`ExampleTestBundle`, so it
        will be used to assemble the test case.

        .. seealso:: The class :class:`lisa.platforms.platinfo.PlatformInfo`
            provides information about a device that are usually needed in
            tests.

        .. seealso: This methods provides an easy way of running an rt-app
            workload on the target device
            :meth:`lisa.tests.base.RTATestBundle.run_rtapp`
        """
        # PlatformInfo
        # https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/target.html#lisa.platforms.platinfo.PlatformInfo
        #
        # It's a central piece of LISA: it holds all the information about a
        # given device. Use it to access any data it contains rather than
        # fetching them yourselves, as the final user will have ways of
        # providing values in case auto-detection fails, and logging of all the
        # data it contains is provided out of the box.
        plat_info = target.plat_info

        # The rt-app profile defines the rt-app workload that will be run
        # note: If None is given to run_rtapp(), it will default to calling
        # get_rtapp_profile()
        rtapp_profile = cls.get_rtapp_profile(plat_info)

        # Here, we wanted to make sure the cpufreq governor is schedutil, since
        # that's what we want to test. This is achieved through the used of
        # devlib modules:
        # https://devlib.readthedocs.io/en/latest/modules.html
        with target.cpufreq.use_governor("schedutil"):
            # RTATestBundle.run_rtapp()
            # https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/kernel_tests.html#lisa.tests.base.RTATestBundle.run_rtapp
            #
            # It allows running the rt-app profile on the target. ftrace_coll
            # is the object used to control the recording of the trace, and is
            # setup by the test runner. This allows the final user to extend
            # the list of ftrace events collected. If no collector is provided,
            # a default one will be created by run_rtapp() based on the
            # @requires_events() decorators used on method of that
            # ExampleTestBundle. Note that it will also freeze all the tasks on
            # the target device, so that the scheduler signals are not
            # disturbed. Some critical tasks are not frozen though.
            cls.run_rtapp(target, res_dir, rtapp_profile, ftrace_coll=ftrace_coll)

        # Execute a silly shell command on the target device as well
        output = target.execute('echo $((21+21))').split()

        # Logging must be done through the provided logger, so it integrates well in LISA.
        cls.get_logger().info('Finished doing stuff')

        # Actually create a ExampleTestBundle by calling the class.
        return cls(res_dir, plat_info, output)

    @classmethod
    def get_rtapp_profile(cls, plat_info):
        """
        This class method is in charge of generating an rt-app profile, to
        configure the workload that will be run using
        :meth:`lisa.tests.base.RTATestBundle.run_rtapp`.

        It can access any information in the given
        :class:`lisa.platforms.PlatformInfo` in order to obtain a workload
        tailored to the capacity of the CPUs of the target, the available
        frequencies and so on.
        """

        # Build a list of the CPU IDs that are available
        cpus = list(range(plat_info['cpus-count']))

        # The profile is a dictionary of task names (keys) to
        # lisa.wlgen.RTATask instances
        # https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/workloads.html
        profile = {}

        for cpu in cpus:
            # Compute a utilization needed to fill 50% of ``cpu`` capacity.
            util = cls.unscaled_utilization(plat_info, cpu, 50)

            # A Periodic task has a period, and a duty_cycle (which really is a
            # target utilization). LISA will run rt-app calibration if needed
            # (it can be provided by the user in the platform information)
            profile[f"{cls.task_prefix}_{cpu}"] = Periodic(
                duty_cycle_pct=util,
                duration_s=1,
                period_ms=cls.TASK_PERIOD_MS,
                cpus=[cpu]
            )

        return profile

    # ftrace events necessary for that test method to run must be specified here.
    # This information will be used in a number of places:
    # * To build the ExampleTestBundle.ftrace_conf attribute, which is then used by RTATestBundle.run_rtapp()
    # * To parse the ftrace trace
    # * In the Sphinx documentation.
    # * To check that the events are available in the trace. A clear exception
    #   is raised if an even is missing.
    # Note: Other decorators can be used to express optional events or
    # alternatives, see lisa.trace module.
    @requires_events('sched_switch', 'sched_wakeup')
    # This allows referencing the @requires_events() of
    # LoadTrackingAnalysis.df_tasks_signal(), so we don't duplicate that
    # information here in case it changes in the future. Use that when you
    # don't use the events directly in your code.
    @LoadTrackingAnalysis.df_tasks_signal.used_events
    # This decorator allows checking that there was no background noise (other
    # tasks executing) while running the workload. If that was the case, the
    # returned result_bundle.result will be set to Result.UNDECIDED, expressing
    # that the data don't allow drawing a pass/fail conclusion.
    @RTATestBundle.test_noisy_tasks.undecided_filter(noise_threshold_pct=1)
    def test_output(self, util_margin=50) -> ResultBundle:
        """
        Actual test method that looks at the collected data and draws a
        conclusion based on it.

        The return annotation "'ResultBundle'" is used by the test runner to
        assemble the test cases, since it's driven by types and what function
        can produce them.

        .. seealso:: :class:`lisa.tests.base.ResultBundle`
        """

        # Get the pandas DataFrame of tasks utilisation.
        #
        # self.trace: This is a lisa.trace.Trace object, with all the events
        # specified using @requires_events() on methods of this class. For
        # subclasses of RTATestBundle, self.trace is actually a TraceView
        # object, restricting the time range to when the rt-app tasks were
        # executing. The test methods can therefore work on minimal and
        # hopefully clean/relevant data.
        #
        # self.trace.analyis: A number of analysis objects are available,
        # giving df_* methods that return various dataframes, and plot_*
        # functions that can do various plots.
        # https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/trace_analysis.html
        df = self.trace.analysis.load_tracking.df_tasks_signal('util')

        # "resolve" the task names into (pid, comm) tuples. If there is any
        # ambiguity because of the same name is reused in different PIDs, an
        # exception will be raised.
        # self.rtapp_tasks gives the list of task names as defined in
        # get_rtapp_profile().
        task_ids = [self.trace.get_task_id(task) for task in self.rtapp_tasks]

        util_means = {}

        # Example test that checks the tasks' average utilisation is as expected
        def check_task_util(task_id):
            # Only keep the data about the tasks we care about.
            _df = df_filter_task_ids(df, [task_id])
            avg = _df['util'].mean()
            util_means[task_id.comm] = avg
            # Util is not supposed to be higher than 512 given what we asked for in get_rtapp_profile()
            return avg < (512 + util_margin)

        # Will be True if all check_task_util() calls are True
        ok = all(check_task_util(task_id) for task_id in task_ids)

        # Create a pass/fail ResultBundle.
        res_bundle = ResultBundle.from_bool(ok)

        # Named metrics (with a optional unit) can be attached to the
        # ResultBundle, and will be reported to whoever runs the test. Good
        # practice for threshold-based tests is to add one metric for the
        # computed value, and one for the threshold.
        # Extra metrics can be very helpful when doing initial investigations
        # on a test failure, so it's better to be more verbose than not.
        res_bundle.add_metric('expected util', 512)
        for task, util_mean in util_means.items():
            res_bundle.add_metric(f'{task} util', util_mean)

        return res_bundle
