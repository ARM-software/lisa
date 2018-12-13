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
import sys
from collections.abc import Mapping

from devlib.target import KernelVersion


from lisa.trace import Trace
from lisa.wlgen.rta import RTA

from lisa.utils import Serializable, memoized, ArtifactPath
from lisa.env import TestEnv
from lisa.platforms.platinfo import PlatformInfo

class TestMetric:
    """
    A storage class for metrics used by tests

    :param data: The data to store. Can be any base type or dict(TestMetric)

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
        Alternate constructor where ``ResultBundle.result`` is determined from a bool
        """
        result = Result.PASSED if cond else Result.FAILED
        return cls(result, *args, **kwargs)

    def __bool__(self):
        return self.result is Result.PASSED

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

    def display_and_exit(self) -> type(None):
        print("Test result: {}".format(self))
        if self:
            sys.exit(0)
        else:
            sys.exit(1)

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

    :param plat_info: Various informations about the platform, that is available
        to all tests.
    :type plat_info: :class:`lisa.platforms.platinfo.PlatformInfo`

    The point of a TestBundle is to bundle in a single object all of the
    required data to run some test assertion (hence the name). When inheriting
    from this class, you can define test methods that use this data, and return
    a :class:`ResultBundle`.

    Thanks to :class:`~lisa.utils.Serializable`, instances of this class
    can be serialized with minimal effort. As long as some information is stored
    within an object's member, it will be automagically handled.

    Please refrain from monkey-patching the object in :meth:`from_testenv`.
    Data required by the object to run test assertions should be exposed as
    ``__init__`` parameters.

    **Design notes:**

      * :meth:`from_testenv` will collect whatever artifacts are required
        from a given target, and will then return a :class:`TestBundle`.
      * :meth:`from_dir` will use whatever artifacts are available in a
        given directory (which should have been created by an earlier call
        to :meth:`from_testenv` and then :meth:`to_dir`), and will then return
        a :class:`TestBundle`.
      * :attr:`VERIFY_SERIALIZATION` is there to ensure both above methods remain
        operationnal at all times.
      * `res_dir` parameter of `__init__` must be stored as an attribute
        without further processing, in order to support result directory
        relocation.
      * Test methodes should have a return annotation for the
        :class:`ResultBundle` to be picked up by the test runners.

    **Implementation example**::

        class DummyTestBundle(TestBundle):

            def __init__(self, res_dir, plat_info, shell_output):
                super(DummyTestBundle, self).__init__(res_dir, plat_info)

                self.shell_output = shell_output

            @classmethod
            def _from_testenv(cls, te, plat_info, res_dir):
                output = te.target.execute('echo $((21+21))').split()
                return cls(res_dir, plat_info, output)

            def test_output(self) -> ResultBundle:
                return ResultBundle.from_bool(
                    any(
                        '42' in line
                        for line in self.shell_output
                    )
                )

    **Usage example**::

        # Creating a Bundle from a live target
        bundle = TestBundle.from_testenv(test_env, plat_info, "/my/res/dir")
        # Running some test on the bundle
        res_bundle = bundle.test_foo()

        # Saving the bundle on the disk
        bundle.to_dir("/my/res/dir")

        # Reloading the bundle from the disk
        bundle = TestBundle.from_dir("/my/res/dir")
        # The reloaded object can be used just like the original one.
        # Keep in mind that serializing/deserializing this way will have a
        # similar effect than a deepcopy.
        res_bundle = bundle.test_foo()
    """

    VERIFY_SERIALIZATION = True
    """
    When True, this enforces a serialization/deserialization step in
    :meth:`from_testenv`. Although it adds an extra step (we end up creating
    two :class:`TestBundle` instances), it's very valuable to ensure
    :meth:`TestBundle.from_dir` does not get broken for some particular
    class.
    """

    def __init__(self, res_dir, plat_info):
        # It is important that res_dir is directly stored as an attribute, so
        # it can be replaced by a relocated res_dir after the object is
        # deserialized on another host.
        # See exekall_customization.LISAAdaptor.load_db
        self.res_dir = res_dir
        self.plat_info = plat_info

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
        except CannotCreateError:
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
            res_dir = te.get_res_dir(cls.__qualname__)

        bundle = cls._from_testenv(te, res_dir, **kwargs)

        # We've created the bundle from the target, and have all of
        # the information we need to execute the test code. However,
        # we enforce the use of the offline reloading path to ensure
        # it does not get broken.
        if cls.VERIFY_SERIALIZATION:
            bundle.to_dir(res_dir)
            # Updating the res_dir breaks deserialization for some use cases
            bundle = cls.from_dir(res_dir, update_res_dir=False)

        return bundle

    @classmethod
    def _filepath(cls, res_dir):
        return os.path.join(res_dir, "{}.yaml".format(cls.__qualname__))

    @classmethod
    def from_dir(cls, res_dir, update_res_dir=True):
        """
        Wrapper around :meth:`lisa.utils.Serializable.from_path`.

        It uses :meth:`_filepath` to get the name of the serialized file to
        reload.
        """
        res_dir = ArtifactPath(root=res_dir, relative='')

        bundle = super().from_path(cls._filepath(res_dir))
        # We need to update the res_dir to the one we were given
        if update_res_dir:
            bundle.res_dir = res_dir

        return bundle

    def to_dir(self, res_dir):
        """
        See :meth:`lisa.utils.Serializable.to_path`
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
    is being run. Items are arguments to :meth:`lisa.env.TestEnv.collect_ftrace`.
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
        serialized when calling :meth:`lisa.utils.Serializable.to_path` and
        allows updating the underlying path before it is actually loaded to
        match a different folder structure.
        """
        return Trace(self.res_dir, self.plat_info, events=self.ftrace_conf["events"])

    def __init__(self, res_dir, plat_info, rtapp_profile):
        super().__init__(res_dir, plat_info)
        self.rtapp_profile = rtapp_profile

    @classmethod
    def unscaled_utilization(cls, te, cpu, utilization_pct):
        """
        Convert utilization scaled to a CPU to a 'raw', unscaled one.

        :param capacity: The CPU against which ``utilization_pct``` is scaled
        :type capacity: int

        :param utilization_pct: The scaled utilization in %
        :type utilization_pct: int
        """
        if "nrg-model" in te.plat_info:
            capacity_scale = te.plat_info["nrg-model"].capacity_scale
        else:
            capacity_scale = 1024

        return int((te.plat_info["cpu-capacities"][cpu] / capacity_scale) * utilization_pct)

    @classmethod
    @abc.abstractmethod
    def get_rtapp_profile(cls, te):
        """
        :returns: a :class:`dict` with task names as keys and
          :class:`lisa.wlgen.rta.RTATask` as values

        This is the method you want to override to specify what is your
        synthetic workload.
        """
        pass

    @classmethod
    def _run_rtapp(cls, te, res_dir, profile):
        wload = RTA.by_profile(te, "rta_{}".format(cls.__name__.lower()),
                               profile, res_dir=res_dir)

        trace_path = os.path.join(res_dir, "trace.dat")

        with te.collect_ftrace(trace_path, **cls.ftrace_conf), te.freeze_userspace():
            wload.run()

    @classmethod
    def _from_testenv(cls, te, res_dir):
        rtapp_profile = cls.get_rtapp_profile(te)
        cls._run_rtapp(te, res_dir, rtapp_profile)

        return cls(res_dir, te.plat_info, rtapp_profile)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
