# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

import os

from lisa.trace import Trace
from lisa.wlgen.rta import RTA
from lisa.perf_analysis import PerfAnalysis

from lisa.serialization import YAMLSerializable

class TestMetric(object):
    """
    A storage class for metrics used by tests

    :param data: The data to store
    :param units: The data units
    """

    def __init__(self, data, units=None):
        self.data = data
        self.units = units

    def __str__(self):
        result = '{}'.format(self.data)
        if self.units:
            result += ' ' + self.units
        return result

    def __repr__(self):
        return '<{}>'.format(self.__str__())

class ResultBundle(object):
    """
    Bundle for storing test results

    :param passed: Indicates whether the associated test passed.
      It will also be used as the truth-value of a ResultBundle.
    :type passed: boolean

    :class:`TestMetric` can be added to an instance of this class. This can
    make it easier for users of your tests to understand why a certain test
    passed or failed. For instance::

        def test_is_noon():
            now = time.localtime().tm_hour
            res = ResultBundle(now == 12)
            res.add_metric("current time", now)

            return res

        >>> res_bundle = test_is_noon()
        >>> if res_bundle:
        >>>     print "PASSED"
        >>> else:
        >>>     print "FAILED"
        FAILED

        # At this point, the user can wonder why the test failed.
        # Metrics are here to help.
        >>> print res_bundle
        current time = 11
    """
    def __init__(self, passed=True):
        self.passed = passed
        self.metrics = {}

    def __nonzero__(self):
        return self.passed

    def __str__(self):
        res = ''
        if self.metrics:
            metrics_str = ', '.join(
                ['{} = {}'.format(key, val) for key, val in self.metrics.items()])
            res = metrics_str

        return res

    def add_metric(self, name, data, units=None):
        """
        Lets you append several test :class:`TestMetric` to the bundle.

        :Parameters: :class:`TestMetric` parameters
        """
        self.metrics[name] = TestMetric(data, units)

class TestBundle(YAMLSerializable):
    """
    A LISA test bundle.

    :param res_dir: Directory in which the target execution artifacts reside.
        This will also be used to dump any artifact generated in the test code.
    :type res_dir: str

    **Design notes:**

    * :meth:`from_target` will collect whatever artifacts are required
      from a given target, and will then return a :class:`TestBundle`.
    * :meth:`from_path` will use whatever artifacts are available in a
      given directory (which should have been created by an earlier call
      to :meth:`from_target` and then :meth:`to_path`), and will then return
      a :class:`TestBundle`.
    * :attr:`verify_serialization` is there to ensure both above methods remain
      operationnal at all times.

    The point of a TestBundle is to bundle in a single object all of the
    required data to run some test assertion (hence the name). When inheriting
    from this class, you can define test methods that use this data, and return
    a :class:`ResultBundle`.

    Thanks to :class:`YAMLSerializable`, instances of this class can be
    serialized with minimal effort. As long as some information is stored
    within an object's member, it will be automagically handled.

    Please refrain from monkey-patching the object in :meth:`from_target`.
    Data required by the object to run test assertions should be exposed as
    :meth:`__init__` parameters.

    **Implementation example**::

        class DummyTestBundle(TestBundle):

            def __init__(self, res_dir, shell_output):
                super(DummyTestBundle, self).__init__(res_dir)

                self.shell_output = shell_output

            @classmethod
            def _from_target(cls, te, res_dir):
                output = te.target.execute('echo $((21+21))').split()
                return cls(res_dir, output)

            def test_output(self):
                passed = False
                for line in self.shell_output:
                    if '42' in line:
                        passed = True
                        break

                return ResultBundle(passed)

    **Usage example**::

        # Creating a Bundle from a live target
        bundle = TestBundle.from_target(test_env, "/my/res/dir")
        # Running some test on the bundle
        res_bundle = bundle.test_foo()

        # Saving the bundle on the disk
        bundle.to_path(test_env, "/my/res/dir")

        # Reloading the bundle from the disk
        bundle = TestBundle.from_path("/my/res/dir")
        res_bundle = bundle.test_foo()
    """

    verify_serialization = True
    """
    When True, this enforces a serialization/deserialization step in :meth:`from_target`.
    Although it hinders performance (we end up creating two :class:`TestBundle`
    instances), it's very valuable to ensure :meth:`from_path` does not get broken
    for some particular class.
    """

    def __init__(self, res_dir):
        self.res_dir = res_dir

    @classmethod
    def _from_target(cls, te, res_dir):
        """
        Internals of the target factory method.
        """
        raise NotImplementedError()

    @classmethod
    def from_target(cls, te, res_dir=None, **kwargs):
        """
        Factory method to create a bundle using a live target

        This is mostly boiler-plate code around :meth:`_from_target`,
        which lets us introduce common functionalities for daughter classes.
        Unless you know what you are doing, you should not override this method,
        but the internal :meth:`_from_target` instead.
        """
        if not res_dir:
            res_dir = te.get_res_dir()

        # Logger stuff?

        bundle = cls._from_target(te, res_dir, **kwargs)

        # We've created the bundle from the target, and have all of
        # the information we need to execute the test code. However,
        # we enforce the use of the offline reloading path to ensure
        # it does not get broken.
        if cls.verify_serialization:
            bundle.to_path(res_dir)
            bundle = cls.from_path(res_dir)

        return bundle

    @classmethod
    def _filepath(cls, res_dir):
        return os.path.join(res_dir, "{}.yaml".format(cls.__name__))

    @classmethod
    def from_path(cls, res_dir):
        """
        See :meth:`YAMLSerializable.from_path`
        """
        bundle = YAMLSerializable.from_path(cls._filepath(res_dir))
        # We need to update the res_dir to the one we were given
        bundle.res_dir = res_dir

        return bundle

    def to_path(self, res_dir):
        """
        See :meth:`YAMLSerializable.to_path`
        """
        super(TestBundle, self).to_path(self._filepath(res_dir))

class RTATestBundle(TestBundle):
    """
    "Abstract" class for :class:`wlgen.rta.RTA`-powered TestBundles

    :param rtapp_profile: The rtapp parameters used to create the synthetic
      workload. That happens to be what is returned by :meth:`create_rtapp_profile`
    :type rtapp_profile: dict
    """

    ftrace_conf = {
        "events" : ["sched_switch"],
    }
    """
    The FTrace configuration used to record a trace while the synthetic workload
    is being run.
    """

    TASK_PERIOD_MS=16
    """
    A task period you can re-use for your :class:`wlgen.rta.RTATask`
    definitions.
    """

    @property
    def trace(self):
        """
        :returns: a Trace

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`to_path`
        """
        if not self._trace:
            self._trace = Trace(self.res_dir, events=self.ftrace_conf["events"])

        return self._trace

    def __init__(self, res_dir, rtapp_profile):
        super(RTATestBundle, self).__init__(res_dir)

        self._trace = None
        self.rtapp_profile = rtapp_profile

    @classmethod
    def create_rtapp_profile(cls, te):
        """
        :returns: a :class:`dict` with task names as keys and :class:`RTATask` as values

        This is the method you want to override to specify what is your
        synthetic workload.
        """
        raise NotImplementedError()

    @classmethod
    def _run_rtapp(cls, te, res_dir, profile):
        wload = RTA.by_profile(te, "rta_{}".format(cls.__name__.lower()),
                               profile, res_dir=res_dir,
                               calibration=te.get_rtapp_calibration())

        trace_path = os.path.join(res_dir, "trace.dat")
        te.configure_ftrace(**cls.ftrace_conf)

        with te.record_ftrace(trace_path):
            with te.freeze_userspace():
                wload.run()

    @classmethod
    def _from_target(cls, te, res_dir):
        rtapp_profile = cls.create_rtapp_profile(te)
        cls._run_rtapp(te, res_dir, rtapp_profile)

        return cls(res_dir, rtapp_profile)

    def test_slack(self, negative_slack_allowed_pct=15):
        """
        Assert that the RTApp workload was given enough performance

        :param negative_slack_allowed_pct: Allowed percentage of RT-app task
            activations with negative slack.
        :type negative_slack_allowed_pct: int

        Use :class:`PerfAnalysis` to find instances where the RT-App workload
        wasn't able to complete its activations (i.e. its reported "slack"
        was negative). Assert that this happened less than
        :attr:`negative_slack_allowed_pct` percent of the time.
        """
        pa = PerfAnalysis(self.res_dir)

        slacks = {}
        res = ResultBundle()

        # Data is only collected for rt-app tasks, so it's safe to iterate over
        # all of them
        for task in pa.tasks():
            slack = pa.df(task)["Slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100. / len(slack)
            if bad_activations_pct > negative_slack_allowed_pct:
                res.passed = False

            slacks[task] = bad_activations_pct

        for task, slack in slacks.iteritems():
            res.add_metric("slack_{}".format(task), slack, '%')

        return res


################################################################################
################################################################################

# class AndroidWorkload(LisaWorkload):

#     def _setup_wload(self):
#         self.target.set_auto_brightness(0)
#         self.target.set_brightness(0)

#         self.target.ensure_screen_is_on()
#         self.target.swipe_to_unlock()

#         self.target.set_auto_rotation(0)
#         self.target.set_rotation(1)

#     def _run_wload(self):
#         pass

#     def _teardown_wload(self):
#         self.target.set_auto_rotation(1)
#         self.target.set_auto_brightness(1)

#     def run(self, trace_tool):
#         if trace_tool == "ftrace":
#             pass
#         elif trace_tool == "systrace":
#             pass

#         self._setup_wload()

#         with self.te.record_ftrace():
#             self._run_wload()

#         self._teardown_wload()

# from target_script import TargetScript
# from devlib.target import AndroidTarget

# class GmapsWorkload(AndroidWorkload):

#     def _setup_wload(self):
#         super(GmapsWorkload, self)._setup_wload()

#         self.script = TargetScript(self.te, "gmaps_swiper.sh")

#         for i in range(self.swipe_count):
#             # Swipe right
#             self.script.input_swipe_pct(40, 50, 60, 60)
#             #AndroidTarget.input_swipe_pct(self.script, 40, 50, 60, 60)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe down
#             AndroidTarget.input_swipe_pct(self.script, 50, 60, 50, 40)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe left
#             AndroidTarget.input_swipe_pct(self.script, 60, 50, 40, 50)
#             AndroidTarget.sleep(self.script, 1)
#             # Swipe up
#             AndroidTarget.input_swipe_pct(self.script, 50, 40, 50, 60)
#             AndroidTarget.sleep(self.script, 1)

#         # Push script to the target
#         self.script.push()

#     def _run_wload(self):
#         self.script.run()

#     def run(self, swipe_count=10):
#         self.swipe_count = swipe_count

#         super(GmapsWorkload, self).run("ftrace")
