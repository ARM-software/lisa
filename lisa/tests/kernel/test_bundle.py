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

import enum
import os
import os.path
import abc

from collections.abc import Mapping

from lisa.trace import Trace
from lisa.wlgen.rta import RTA
from lisa.perf_analysis import PerfAnalysis

from lisa.utils import Serializable, memoized
from lisa.env import TestEnv, ArtifactPath

class TestMetric:
    """
    A storage class for metrics used by tests

    :param data: The data to store
    :type data: Any base type or dict(TestMetric)

    :param units: The data units
    :type units: str
    """
    def __init__(self, data, units=None):
        self.data = data
        self.units = units

    def __str__(self):
        if isinstance(self.data, Mapping):
            return '{{{}}}'.format(', '.join(
                ["{}={}".format(name, data) for name, data in self.data.items()]))

        result = str(self.data)
        if self.units:
            result += ' ' + self.units

        return result

    def __repr__(self):
        return '{cls}({self.data}, {self.units})'.format(
            cls=type(self).__name__, self=self)

@enum.unique
class Result(enum.Enum):
    """
    A classification of a test result
    """
    PASSED = 1
    """
    The test has passed
    """

    FAILED = 2
    """
    The test has failed
    """

    UNDECIDED = 3
    """
    The test data could not be used to decide between :attr:`PASSED` or :attr:`FAILED`
    """

    @property
    def lower_name(self):
        """Return the name in lower case"""
        return self.name.lower()

class ResultBundle:
    """
    Bundle for storing test results

    :param result: Indicates whether the associated test passed.
      It will also be used as the truth-value of a ResultBundle.
    :type result: :class:`Result`

    :class:`TestMetric` can be added to an instance of this class. This can
    make it easier for users of your tests to understand why a certain test
    passed or failed. For instance::

        def test_is_noon():
            now = time.localtime().tm_hour
            res = ResultBundle(Result.PASSED if now == 12 else Result.FAILED)
            res.add_metric("current time", now)

            return res

        >>> res_bundle = test_is_noon()
        >>> print(res_bundle.result.name)
        FAILED

        # At this point, the user can wonder why the test failed.
        # Metrics are here to help, and are printed along with the result:
        >>> print(res_bundle)
        FAILED: current time=11
    """
    def __init__(self, result):
        self.result = result
        self.metrics = {}

    @classmethod
    def from_bool(cls, cond, *args, **kwargs):
        """
        Alternate constructor where :attr:`result` is determined from a bool
        """
        result = Result.PASSED if cond else Result.FAILED
        return cls(result, *args, **kwargs)

    def __bool__(self):
        return self.result == Result.PASSED

    def __str__(self):
        return self.result.name + ': ' + ', '.join(
                '{}={}'.format(key, val)
                for key, val in self.metrics.items())

    def add_metric(self, name, data, units=None):
        """
        Lets you append several test :class:`TestMetric` to the bundle.

        :Parameters: :class:`TestMetric` parameters
        """
        self.metrics[name] = TestMetric(data, units)

class CannotCreateError(RuntimeError):
    """
    Something prevented the creation of a :class:`TestBundle` instance
    """
    pass

class TestBundle(Serializable, abc.ABC):
    """
    A LISA test bundle.

    :param res_dir: Directory in which the target execution artifacts reside.
        This will also be used to dump any artifact generated in the test code.
    :type res_dir: str

    The point of a TestBundle is to bundle in a single object all of the
    required data to run some test assertion (hence the name). When inheriting
    from this class, you can define test methods that use this data, and return
    a :class:`ResultBundle`.

    Thanks to :class:`~lisa.utils.Serializable`, instances of this class
    can be serialized with minimal effort. As long as some information is stored
    within an object's member, it will be automagically handled.

    Please refrain from monkey-patching the object in :meth:`from_testenv`.
    Data required by the object to run test assertions should be exposed as
    :meth:`__init__` parameters.

    **Design notes:**

      * :meth:`from_testenv` will collect whatever artifacts are required
        from a given target, and will then return a :class:`TestBundle`.
      * :meth:`from_dir` will use whatever artifacts are available in a
        given directory (which should have been created by an earlier call
        to :meth:`from_testenv` and then :meth:`to_dir`), and will then return
        a :class:`TestBundle`.
      * :attr:`verify_serialization` is there to ensure both above methods remain
        operationnal at all times.

    **Implementation example**::

        class DummyTestBundle(TestBundle):

            def __init__(self, res_dir, shell_output):
                super(DummyTestBundle, self).__init__(res_dir)

                self.shell_output = shell_output

            @classmethod
            def _from_testenv(cls, te, res_dir):
                output = te.target.execute('echo $((21+21))').split()
                return cls(res_dir, output)

            def test_output(self):
                passed = False
                for line in self.shell_output:
                    if '42' in line:
                        passed = True
                        break

                return ResultBundle.from_bool(passed)

    **Usage example**::

        # Creating a Bundle from a live target
        bundle = TestBundle.from_testenv(test_env, "/my/res/dir")
        # Running some test on the bundle
        res_bundle = bundle.test_foo()

        # Saving the bundle on the disk
        bundle.to_dir(test_env, "/my/res/dir")

        # Reloading the bundle from the disk
        bundle = TestBundle.from_dir("/my/res/dir")
        res_bundle = bundle.test_foo()
    """

    verify_serialization = True
    """
    When True, this enforces a serialization/deserialization step in :meth:`from_testenv`.
    Although it hinders performance (we end up creating two :class:`TestBundle`
    instances), it's very valuable to ensure :meth:`from_dir` does not get broken
    for some particular class.
    """

    def __init__(self, res_dir):
        # It is important that res_dir is directly stored as an attribute, so
        # it can be replaced by a relocated res_dir after the object is
        # deserialized on another host.
        # See exekall_customization.LISAAdaptor.load_db
        self.res_dir = res_dir

    @classmethod
    @abc.abstractmethod
    def _from_testenv(cls, te, res_dir):
        """
        Internals of the target factory method.
        """
        pass

    @classmethod
    def check_from_testenv(cls, te):
        """
        Check whether the given target can be used to create an instance of this class

        :raises: CannotCreateError if the check fails

        This method should be overriden to check your implementation requirements
        """
        pass

    @classmethod
    def can_create_from_testenv(cls, te):
        """
        :returns: Whether the given target can be used to create an instance of this class
        :rtype: bool

        :meth:`check_from_testenv` is used internally, so there shouldn't be any
          need to override this.
        """
        try:
            cls.check_from_testenv(te)
            return True
        except:
            return False

    @classmethod
    def from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None, **kwargs) -> 'TestBundle':
        """
        Factory method to create a bundle using a live target

        This is mostly boiler-plate code around :meth:`_from_testenv`,
        which lets us introduce common functionalities for daughter classes.
        Unless you know what you are doing, you should not override this method,
        but the internal :meth:`_from_testenv` instead.
        """
        cls.check_from_testenv(te)

        if not res_dir:
            res_dir = te.get_res_dir()

        bundle = cls._from_testenv(te, res_dir, **kwargs)

        # We've created the bundle from the target, and have all of
        # the information we need to execute the test code. However,
        # we enforce the use of the offline reloading path to ensure
        # it does not get broken.
        if cls.verify_serialization:
            bundle.to_dir(res_dir)
            bundle = cls.from_dir(res_dir)

        return bundle

    @classmethod
    def _filepath(cls, res_dir):
        return os.path.join(res_dir, "{}.yaml".format(cls.__qualname__))

    @classmethod
    def from_dir(cls, res_dir):
        """
        See :meth:`Serializable.from_path`
        """
        bundle = super().from_path(cls._filepath(res_dir))
        # We need to update the res_dir to the one we were given
        bundle.res_dir = res_dir

        return bundle

    def to_dir(self, res_dir):
        """
        See :meth:`Serializable.to_path`
        """
        super().to_path(self._filepath(res_dir))

class RTATestBundle(TestBundle, abc.ABC):
    """
    "Abstract" class for :class:`lisa.wlgen.rta.RTA`-powered TestBundles

    :param rtapp_profile: The rtapp parameters used to create the synthetic
      workload. That happens to be what is returned by :meth:`get_rtapp_profile`
    :type rtapp_profile: dict
    """

    ftrace_conf = {
        "events" : ["sched_switch"],
    }
    """
    The FTrace configuration used to record a trace while the synthetic workload
    is being run. Items are arguments to :meth:`lisa.env.TestEnv.configure_ftrace`.
    """

    TASK_PERIOD_MS = 16
    """
    A task period you can re-use for your :class:`lisa.wlgen.rta.RTATask`
    definitions.
    """

    @property
    @memoized
    def trace(self):
        """
        :returns: a Trace

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`to_path`
        """
        return Trace(self.res_dir, events=self.ftrace_conf["events"])

    def __init__(self, res_dir, rtapp_profile):
        super().__init__(res_dir)
        self.rtapp_profile = rtapp_profile

    @classmethod
    @abc.abstractmethod
    def get_rtapp_profile(cls, te):
        """
        :returns: a :class:`dict` with task names as keys and :class:`RTATask` as values

        This is the method you want to override to specify what is your
        synthetic workload.
        """
        pass

    @classmethod
    def _run_rtapp(cls, te, res_dir, profile):
        wload = RTA.by_profile(te, "rta_{}".format(cls.__name__.lower()),
                               profile, res_dir=res_dir,
                               calibration=te.get_rtapp_calibration())

        trace_path = os.path.join(res_dir, "trace.dat")
        te.configure_ftrace(**cls.ftrace_conf)

        with te.record_ftrace(trace_path), te.freeze_userspace():
            wload.run()

    @classmethod
    def _from_testenv(cls, te, res_dir):
        rtapp_profile = cls.get_rtapp_profile(te)
        cls._run_rtapp(te, res_dir, rtapp_profile)

        return cls(res_dir, rtapp_profile)

    @classmethod
    def from_testenv(cls, te:TestEnv, res_dir:ArtifactPath=None) -> 'RTATestBundle':
        """
        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in :meth:`get_rtapp_profile`
        """
        return super().from_testenv(te, res_dir)

    def test_slack(self, negative_slack_allowed_pct=15) -> ResultBundle:
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

        # Data is only collected for rt-app tasks, so it's safe to iterate over
        # all of them
        passed = True
        for task in pa.tasks():
            slack = pa.df(task)["Slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100 / len(slack)
            if bad_activations_pct > negative_slack_allowed_pct:
                passed = False

            slacks[task] = bad_activations_pct

        res = ResultBundle.from_bool(passed)

        for task, slack in slacks.items():
            res.add_metric("{} slack".format(task), slack, '%')

        return res

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
