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
import functools
import os
import os.path
import abc
import sys
import textwrap
from collections.abc import Mapping
from inspect import signature, Parameter

from devlib.target import KernelVersion

from lisa.analysis.tasks import TasksAnalysis
from lisa.trace import Trace, requires_events
from lisa.wlgen.rta import RTA

from lisa.utils import Serializable, memoized, ArtifactPath
from lisa.platforms.platinfo import PlatformInfo
from lisa.target import Target
from lisa.trace import FtraceCollector, FtraceConf

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

    Please refrain from monkey-patching the object in :meth:`from_target`.
    Data required by the object to run test assertions should be exposed as
    ``__init__`` parameters.

    **Design notes:**

      * :meth:`from_target` will collect whatever artifacts are required
        from a given target, and will then return a :class:`TestBundle`.
      * :meth:`from_dir` will use whatever artifacts are available in a
        given directory (which should have been created by an earlier call
        to :meth:`from_target` and then :meth:`to_dir`), and will then return
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
            def _from_target(cls, target, plat_info, res_dir):
                output = target.execute('echo $((21+21))').split()
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
        bundle = TestBundle.from_target(target, plat_info, "/my/res/dir")
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
    :meth:`from_target`. Although it adds an extra step (we end up creating
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
    def _from_target(cls, target, res_dir):
        """
        Internals of the target factory method.
        """
        pass

    @classmethod
    def check_from_target(cls, target):
        """
        Check whether the given target can be used to create an instance of this class

        :raises: CannotCreateError if the check fails

        This method should be overriden to check your implementation requirements
        """
        pass

    @classmethod
    def can_create_from_target(cls, target):
        """
        :returns: Whether the given target can be used to create an instance of this class
        :rtype: bool

        :meth:`check_from_target` is used internally, so there shouldn't be any
          need to override this.
        """
        try:
            cls.check_from_target(target)
            return True
        except CannotCreateError:
            return False

    @classmethod
    def from_target(cls, target, res_dir=None, **kwargs):
        """
        Factory method to create a bundle using a live target

        This is mostly boiler-plate code around :meth:`_from_target`,
        which lets us introduce common functionalities for daughter classes.
        Unless you know what you are doing, you should not override this method,
        but the internal :meth:`_from_target` instead.
        """
        cls.check_from_target(target)

        res_dir = res_dir or target.get_res_dir(
            name=cls.__qualname__,
            symlink=True,
        )

        bundle = cls._from_target(target, res_dir, **kwargs)

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

    ftrace_conf = FtraceConf({
        "events" : [
            "sched_switch",
            "sched_wakeup"
        ],
    }, __qualname__)
    """
    The FTrace configuration used to record a trace while the synthetic workload
    is being run.
    """

    TASK_PERIOD_MS = 16
    """
    A task period you can re-use for your :class:`lisa.wlgen.rta.RTATask`
    definitions.
    """

    NOISE_IGNORED_PIDS = [0]
    """
    PIDs to ignore in :meth:`test_noisy_tasks`.

    PID 0 is the idle task, don't count it as noise.
    """

    def trace_window(self, trace):
        """
        The time window to consider for this :class:`RTATestBundle`

        :returns: a (start, stop) tuple

        Since we're using rt-app profiles, we know the name of tasks we are
        interested in, so we can trim our trace scope to filter out the
        setup/teardown events we don't care about.

        Override this method if you need a different trace trimming.

        .. warning::

          Don't call ``self.trace`` in here unless you like infinite recursion.
        """
        sdf = trace.df_events('sched_switch')

        # Find when the first task starts running
        rta_start = sdf[sdf.next_comm.isin(self.rtapp_profile.keys())].index[0]
        # Find when the last task stops running
        rta_stop = sdf[sdf.prev_comm.isin(self.rtapp_profile.keys())].index[-1]

        return (rta_start, rta_stop)

    @property
    @memoized
    def trace(self):
        """
        :returns: a :class:`lisa.trace.TraceView`

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`lisa.utils.Serializable.to_path` and
        allows updating the underlying path before it is actually loaded to
        match a different folder structure.
        """
        path = os.path.join(self.res_dir, 'trace.dat')

        trace = Trace(path, self.plat_info, events=self.ftrace_conf["events"])
        return trace.get_view(self.trace_window(trace))

    def __init__(self, res_dir, plat_info, rtapp_profile):
        super().__init__(res_dir, plat_info)
        self.rtapp_profile = rtapp_profile

    @TasksAnalysis.df_tasks_runtime.used_events
    def test_noisy_tasks(self, noise_threshold_pct=None, noise_threshold_ms=None):
        """
        Test that no non-rtapp ("noisy") task ran for longer than the specified thresholds

        :param noise_threshold_pct: The maximum allowed runtime for noisy tasks in
          percentage of the total rt-app execution time
        :type noise_threshold_pct: float

        :param noise_threshold_ms: The maximum allowed runtime for noisy tasks in ms
        :type noise_threshold_ms: float

        If both are specified, the smallest threshold (in seconds) will be used.
        """
        if noise_threshold_pct is None and noise_threshold_ms is None:
            raise ValueError('Both "{}" and "{}" cannot be None'.format(
                "noise_threshold_pct", "noise_threshold_ms"))

        # No task can run longer than the recorded duration
        threshold_s = self.trace.time_range

        if noise_threshold_pct is not None:
            threshold_s = noise_threshold_pct * self.trace.time_range / 100

        if noise_threshold_ms is not None:
            threshold_s = min(threshold_s, noise_threshold_ms * 1e3)

        df = self.trace.analysis.tasks.df_tasks_runtime()
        test_tasks = list(map(self.trace.get_task_pid, self.rtapp_profile.keys()))
        df_noise = df[~df.index.isin(test_tasks + self.NOISE_IGNORED_PIDS)]

        if df_noise.empty:
            return ResultBundle.from_bool(True)

        pid = df_noise.index[0]
        comm = df_noise.comm.values[0]
        duration_s = df_noise.runtime.values[0]
        duration_pct = duration_s * 100 / self.trace.time_range

        res = ResultBundle.from_bool(duration_s < threshold_s)
        metric = {"pid" : pid,
                  "comm": comm,
                  "duration (abs)": TestMetric(duration_s, "s"),
                  "duration (rel)" : TestMetric(duration_pct, "%")}
        res.add_metric("noisiest task", metric)

        return res

    @classmethod
    #pylint: disable=unused-argument
    def check_noisy_tasks(cls, noise_threshold_pct=None, noise_threshold_ms=None):
        """
        Decorator that applies :meth:`test_noisy_tasks` to the trace of the
        :class:`TestBundle` returned by the underlying method. The :class:`Result`
        will be changed to :attr:`Result.UNDECIDED` if that test fails.

        We also expose :meth:`test_noisy_tasks` parameters to the decorated
        function.
        """
        def decorator(func):
            @cls.test_noisy_tasks.used_events
            @functools.wraps(func)
            def wrapper(self, *args,
                        noise_threshold_pct=noise_threshold_pct,
                        noise_threshold_ms=noise_threshold_ms,
                        **kwargs):
                res = func(self, *args, **kwargs)

                noise_res = self.test_noisy_tasks(
                    noise_threshold_pct, noise_threshold_ms)
                res.metrics.update(noise_res.metrics)

                if not noise_res:
                    res.result = Result.UNDECIDED

                return res

            # https://stackoverflow.com/a/33112180
            # The wrapper has all of `func`'s parameters plus `test_noisy_tasks`',
            # but since we use `wraps(func)` we'll only get doc/autocompletion for
            # `func`'s. Expose the extra parameters to the decorated function to
            # make it more user friendly.
            func_sig = signature(func)
            dec_params = signature(cls.check_noisy_tasks).parameters

            # We want the default values of the new parameters for the
            # *decorated* function to be the values passed to the decorator,
            # which aren't the default values of the decorator.
            new_params = [
                dec_params["noise_threshold_pct"].replace(default=noise_threshold_pct),
                dec_params["noise_threshold_ms"].replace(default=noise_threshold_ms)
            ]
            wrapper.__signature__ = func_sig.replace(
                parameters=list(func_sig.parameters.values()) + new_params
            )

            # Make it obvious in the doc where the extra parameters come from
            merged_doc = textwrap.dedent(cls.test_noisy_tasks.__doc__).splitlines()
            # Replace the one-liner func description
            merged_doc[1] = textwrap.dedent(
                """
                **Added by** :meth:`~{}.{}.{}`:

                The returned ``ResultBundle.result`` will be changed to
                :attr:`~lisa.tests.base.Result.UNDECIDED` if the environment was
                too noisy:
                """.format(cls.__module__, cls.__name__, cls.check_noisy_tasks.__name__)
            )

            #pylint: disable=no-member
            wrapper.__doc__ = textwrap.dedent(wrapper.__doc__) + "\n".join(merged_doc)

            return wrapper
        return decorator

    @classmethod
    def unscaled_utilization(cls, plat_info, cpu, utilization_pct):
        """
        Convert utilization scaled to a CPU to a 'raw', unscaled one.

        :param capacity: The CPU against which ``utilization_pct``` is scaled
        :type capacity: int

        :param utilization_pct: The scaled utilization in %
        :type utilization_pct: int
        """
        if "nrg-model" in plat_info:
            capacity_scale = plat_info["nrg-model"].capacity_scale
        else:
            capacity_scale = 1024

        return int((plat_info["cpu-capacities"][cpu] / capacity_scale) * utilization_pct)

    @classmethod
    @abc.abstractmethod
    def get_rtapp_profile(cls, plat_info):
        """
        :returns: a :class:`dict` with task names as keys and
          :class:`lisa.wlgen.rta.RTATask` as values

        This is the method you want to override to specify what is your
        synthetic workload.
        """
        pass

    @classmethod
    def _run_rtapp(cls, target, res_dir, profile, ftrace_coll=None):
        wload = RTA.by_profile(target, "rta_{}".format(cls.__name__.lower()),
                               profile, res_dir=res_dir)

        trace_path = os.path.join(res_dir, "trace.dat")
        ftrace_coll = ftrace_coll or FtraceCollector.from_conf(target, cls.ftrace_conf)

        with ftrace_coll, target.freeze_userspace():
            wload.run()
        ftrace_coll.get_trace(trace_path)
        return trace_path

    @classmethod
    def _from_target(cls, target, res_dir, ftrace_coll=None):
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info)
        cls._run_rtapp(target, res_dir, rtapp_profile, ftrace_coll)

        return cls(res_dir, plat_info, rtapp_profile)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
