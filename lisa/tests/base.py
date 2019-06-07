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
import re

from collections.abc import Mapping
from inspect import signature
import inspect
import copy

from devlib.trace.dmesg import DmesgCollector

from lisa.analysis.tasks import TasksAnalysis
from lisa.trace import Trace, requires_events
from lisa.wlgen.rta import RTA

from lisa.utils import (
    Serializable, memoized, ArtifactPath, non_recursive_property,
    LayeredMapping
)
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


class ResultBundleBase:
    """
    Base class for all result bundles.

    .. note:: ``__init__`` is not provided as some classes uses properties to
        provide some of the attributes.
    """

    def __bool__(self):
        return self.result is Result.PASSED

    def __str__(self):

        def format_val(val):
            # Handle recursive mappings, like metrics of AggregatedResultBundle
            if isinstance(val, Mapping):
                return '{' + ', '.join(
                    '{}={}'.format(key, format_val(val))
                    for key, val in val.items()
                ) + '}'
            else:
                return str(val)

        return self.result.name + ': ' + format_val(self.metrics)

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

class ResultBundle(ResultBundleBase):
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

class AggregatedResultBundle(ResultBundleBase):
    """
    Aggregates many :class:`ResultBundle` into one.

    :param result_bundles: List of :class:`ResultBundle` to aggregate.
    :type result_bundles: list(ResultBundle)

    :param name_metric: Metric to use as the "name" of each result bundle.
        The value of that metric will be used as top-level key in the
        aggregated metrics. If not provided, the index in the
        ``result_bundles`` list will be used.
    :type name_metric: str

    :param result: Optionally, force the ``self.result`` attribute to that
        value. This is useful when the way of combining the result bundles is
        not the default one, without having to make a whole new subclass.
    :type result: Result

    This is useful for some tests that are naturally decomposed in subtests.

    .. note:: Metrics of aggregated bundles will always be shown, but can be
        augmented with new metrics using the usual API.
    """
    def __init__(self, result_bundles, name_metric=None, result=None):
        self.result_bundles = result_bundles
        self.name_metric = name_metric
        self.extra_metrics = {}
        self._forced_result = result

    @property
    def result(self):
        forced_result = self._forced_result
        if forced_result is not None:
            return forced_result

        def predicate(combinator, result):
            return combinator(
                res_bundle.result is result
                for res_bundle in self.result_bundles
            )

        if predicate(all, Result.UNDECIDED):
            return Result.UNDECIDED
        elif predicate(any, Result.FAILED):
            return Result.FAILED
        elif predicate(any, Result.PASSED):
            return Result.PASSED
        else:
            return Result.UNDECIDED

    @result.setter
    def _(self, result):
        self._forced_result = result

    @property
    def metrics(self):
        def get_name(res_bundle, i):
            if self.name_metric:
                return res_bundle.metrics[self.name_metric]
            else:
                return str(i)

        names = {
            res_bundle: get_name(res_bundle, i)
            for i, res_bundle in enumerate(self.result_bundles)
        }

        def get_metrics(res_bundle):
            metrics = copy.copy(res_bundle.metrics)
            # Since we already show it at the top-level, we can remove it from
            # the nested level to remove some clutter
            metrics.pop(self.name_metric, None)
            return metrics

        base = {
            names[res_bundle]: get_metrics(res_bundle)
            for res_bundle in self.result_bundles
        }

        if 'failed' not in base:
            base['failed'] = TestMetric([
                names[res_bundle]
                for res_bundle in self.result_bundles
                if res_bundle.result is Result.FAILED
            ])
        top = self.extra_metrics
        return LayeredMapping(base, top)


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
      * ``res_dir`` parameter of ``__init__`` must be stored as an attribute
        without further processing, in order to support result directory
        relocation.
      * Test methods should have a return annotation for the
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


class RTATestBundleMeta(abc.ABCMeta):
    """
    Metaclass of :class:`RTATestBundle`.

    This metaclass ensures that each class will get its own copy of
    ``ftrace_conf`` attribute, and that the events specified in that
    configuration are a superset of what is needed by methods using the
    decorator :func:`lisa.trace.requires_events`. This makes sure that the
    default set of events is always enough to run all defined methods, without
    duplicating that information.

    .. note:: An existing ``ftrace_conf`` attribute is used, with extra
        detected events merged-in.
    """

    def __new__(metacls, name, bases, dct, **kwargs):
        new_cls = super().__new__(metacls, name, bases, dct, **kwargs)

        # Collect all the events that can be used by all methods available on
        # that class.
        ftrace_events = set()
        for name, obj in inspect.getmembers(new_cls, callable):
            try:
                used_events = obj.used_events
            except AttributeError:
                continue
            else:
                ftrace_events.update(used_events.get_all_events())

        # Get the ftrace_conf attribute of the class, and make sure it is
        # unique to that class (i.e. not shared with any other parent or
        # sibling classes)
        try:
            ftrace_conf = new_cls.ftrace_conf
        except AttributeError:
            ftrace_conf = FtraceConf(src=new_cls.__qualname__)
        else:
            # If the ftrace_conf attribute has been defined in a base class,
            # make sure that class gets its own copy since we are going to
            # modify it
            if 'ftrace_conf' not in dct:
                ftrace_conf = copy.copy(ftrace_conf)

        new_cls.ftrace_conf = ftrace_conf

        # Merge-in a new source to FtraceConf that contains the events we
        # collected
        ftrace_conf.add_merged_src(
            src='{}(required)'.format(new_cls.__qualname__),
            conf={
                'events': sorted(ftrace_events),
            },
        )

        return new_cls


class RTATestBundle(TestBundle, metaclass=RTATestBundleMeta):
    """
    Abstract Base Class for :class:`lisa.wlgen.rta.RTA`-powered TestBundles

    Optionally, an ``ftrace_conf`` class attribute can be defined to hold
    additional FTrace configuration used to record a trace while the synthetic
    workload is being run. By default, the required events are extracted from
    decorated test methods.

    .. seealso: :class:`lisa.tests.base.RTATestBundleMeta` for default
        ``ftrace_conf`` content.
    """

    TRACE_PATH = 'trace.dat'
    """
    Path to the ``trace-cmd`` trace.dat file in the result directory.
    """
    DMESG_PATH = 'dmesg.log'
    """
    Path to the dmesg log in the result directory.
    """

    TASK_PERIOD_MS = 16
    """
    A task period you can re-use for your :class:`lisa.wlgen.rta.RTATask`
    definitions.
    """

    NOISE_ACCOUNTING_THRESHOLDS = {
        # Idle task - ignore completely
        0 : 100,
        # Feeble boards like Juno/TC2 spend a while in sugov
        r"^sugov:\d+$" : 5,
    }
    """
    PID/comm specific tuning for :meth:`test_noisy_tasks`

    * **keys** can be PIDs, comms, or regexps for comms.

    * **values** are noisiness thresholds (%), IOW below that runtime threshold
      the associated task will be ignored in the noise accounting.
    """

    @requires_events('sched_switch')
    def trace_window(self, trace):
        """
        The time window to consider for this :class:`RTATestBundle`

        :returns: a (start, stop) tuple

        Since we're using rt-app profiles, we know the name of tasks we are
        interested in, so we can trim our trace scope to filter out the
        setup/teardown events we don't care about.

        Override this method if you need a different trace trimming.

        .. warning::

          Calling ``self.trace`` here will raise an :exc:`AttributeError`
          exception, to avoid entering infinite recursion.
        """
        sdf = trace.df_events('sched_switch')

        # Find when the first task starts running
        rta_start = sdf[sdf.next_comm.isin(self.rtapp_profile.keys())].index[0]
        # Find when the last task stops running
        rta_stop = sdf[sdf.prev_comm.isin(self.rtapp_profile.keys())].index[-1]

        return (rta_start, rta_stop)

    @property
    def trace_path(self):
        """
        Path to the ``trace-cmd report`` trace.dat file.
        """
        return os.path.join(self.res_dir, self.TRACE_PATH)

    # Guard before the cache, so we don't accidentally start depending on the
    # LRU cache for functionnal correctness.
    @non_recursive_property
    # Use LRU cache instead of memoized, to avoid caching the trace forever, in
    # case the thread is manipulating a large number of TestBundles without
    # deleting them.
    @functools.lru_cache(maxsize=30, typed=True)
    def trace(self):
        """
        :returns: a :class:`lisa.trace.TraceView`

        All events specified in ``ftrace_conf`` are parsed from the trace,
        so it is suitable for direct use in methods.

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`lisa.utils.Serializable.to_path` and
        allows updating the underlying path before it is actually loaded to
        match a different folder structure.
        """
        return self.get_trace(events=self.ftrace_conf["events"])

    def get_trace(self, **kwargs):
        """
        :returns: a :class:`lisa.trace.TraceView` cropped to fit the ``rt-app``
            tasks.

        :Keyword arguments: forwarded to :class:`lisa.trace.Trace`.
        """
        trace = Trace(self.trace_path, self.plat_info, **kwargs)
        return trace.get_view(self.trace_window(trace))

    @property
    def rtapp_profile(self):
        """
        Compute the RTapp profile based on ``plat_info``.
        """
        return self.get_rtapp_profile(self.plat_info)

    @property
    def cgroup_configuration(self):
        """
        Compute the cgroup configuration based on ``plat_info``
        """
        return self.get_cgroup_configuration(self.plat_info)

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

        # We don't want to account the test tasks
        ignored_pids = list(map(self.trace.get_task_pid, self.rtapp_profile.keys()))

        def compute_duration_pct(row):
            return row.runtime * 100 / self.trace.time_range

        df["runtime_pct"] = df.apply(compute_duration_pct, axis=1)

        # Figure out which PIDs to exclude from the thresholds
        for key, threshold in self.NOISE_ACCOUNTING_THRESHOLDS.items():
            # Find out which task(s) this threshold is about
            if isinstance(key, int):
                pids = [key]
            elif isinstance(key, str):
                comms = [comm for comm in df.comm.values if re.match(key, comm)]
                pids = df[df.comm.isin(comms)].index.values
            else:
                pids = []

            # For those tasks, check the threshold
            ignored_pids += [pid for pid in pids if df.loc[pid].runtime_pct <= threshold]

        log_pids = ["{} ({})".format(pid, df.loc[pid].comm) for pid in ignored_pids]
        self.get_logger().info(
            "Ignored PIDs for noise contribution: %s", ", ".join(log_pids))

        # Filter out unwanted tasks (rt-app tasks + thresholds)
        df_noise = df[~df.index.isin(ignored_pids)]

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
            noise_doc = inspect.getdoc(cls.test_noisy_tasks).splitlines()
            # Replace the one-liner func description
            noise_doc[1] = textwrap.dedent(
                """
                **Added by** :meth:`~{}.{}.{}`:

                The returned ``ResultBundle.result`` will be changed to
                :attr:`~lisa.tests.base.Result.UNDECIDED` if the environment was
                too noisy:
                """.format(cls.__module__, cls.__name__, cls.check_noisy_tasks.__name__)
            )
            noise_doc = '\n'.join(noise_doc)

            wrapper_doc = inspect.getdoc(wrapper) or ''
            wrapper.__doc__ = wrapper_doc + "\n".join(noise_doc)

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
    def get_cgroup_configuration(cls, plat_info):
        """
        :returns: a :class:`dict` representing the configuration of a
          particular cgroup.

        This is a method you may optionally override to configure a cgroup for
        the synthetic workload.

        Example of return value::

          {
              'name': 'lisa_test',
              'controller': 'schedtune',
              'attributes' : {
                  'prefer_idle' : 1,
                  'boost': 50
              }
          }

        """
        return None

    @classmethod
    def _target_configure_cgroup(cls, target, cfg):
        if not cfg:
            return None

        kind = cfg['controller']
        if kind not in target.cgroups.controllers:
            raise CannotCreateError('"{}" cgroup controller unavailable'.format(kind))
        ctrl = target.cgroups.controllers[kind]

        cg = ctrl.cgroup(cfg['name'])
        cg.set(**cfg['attributes'])

        return '/' + cg.name

    @classmethod
    def _run_rtapp(cls, target, res_dir, profile, ftrace_coll=None, cg_cfg=None):
        wload = RTA.by_profile(target, "rta_{}".format(cls.__name__.lower()),
                               profile, res_dir=res_dir)

        trace_path = os.path.join(res_dir, cls.TRACE_PATH)
        dmesg_path = os.path.join(res_dir, cls.DMESG_PATH)
        ftrace_coll = ftrace_coll or FtraceCollector.from_conf(target, cls.ftrace_conf)
        dmesg_coll = DmesgCollector(target)

        cgroup = cls._target_configure_cgroup(target, cg_cfg)
        as_root = cgroup is not None

        with dmesg_coll, ftrace_coll, target.freeze_userspace():
            wload.run(cgroup=cgroup, as_root=as_root)

        ftrace_coll.get_trace(trace_path)
        dmesg_coll.get_trace(dmesg_path)
        return trace_path

    @classmethod
    def _from_target(cls, target, res_dir, ftrace_coll=None):
        plat_info = target.plat_info
        rtapp_profile = cls.get_rtapp_profile(plat_info)
        cgroup_config = cls.get_cgroup_configuration(plat_info)
        cls._run_rtapp(target, res_dir, rtapp_profile, ftrace_coll, cgroup_config)

        return cls(res_dir, plat_info)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
