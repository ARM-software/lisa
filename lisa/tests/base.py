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
import inspect
import copy
import contextlib
import itertools

from datetime import datetime
from collections import OrderedDict, ChainMap
from collections.abc import Mapping
from inspect import signature

from devlib.trace.dmesg import KernelLogEntry
from devlib import TargetStableError

from lisa.analysis.tasks import TasksAnalysis
from lisa.trace import Trace, requires_events, TaskID
from lisa.wlgen.rta import RTA
from lisa.target import Target

from lisa.utils import (
    Serializable, memoized, ArtifactPath, non_recursive_property,
    update_wrapper_doc, ExekallTaggable, annotations_from_signature,
    HideExekallID,
)
from lisa.datautils import df_filter_task_ids
from lisa.trace import FtraceCollector, FtraceConf, DmesgCollector
from lisa.conf import (
    SimpleMultiSrcConf, KeyDesc, TopLevelKeyDesc,
    StrList,
)

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
            result = '{{{}}}'.format(', '.join(
                "{}={}".format(name, data) for name, data in self.data.items()))
        else:
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

    :param utc_datetime: UTC time at which the result was collected, or
        ``None`` to record the current datetime.
    :type utc_datetime: datetime.datetime

    :param context: Contextual information to attach to the bundle.
        Keep the content small, as size of :class:`ResultBundle` instances
        matters a lot for storing long test sessions results.
    :type context: dict(str, object)

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
    def __init__(self, result, utc_datetime=None, context=None):
        self.result = result
        self.metrics = {}
        self.utc_datetime = utc_datetime or datetime.utcnow()
        self.context = context if context is not None else {}

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

    :param context: Contextual information to attach to the bundle.
        Keep the content small, as size of :class:`ResultBundle` instances
        matters a lot for storing long test sessions results.
    :type context: dict(str, object)

    This is useful for some tests that are naturally decomposed in subtests.

    .. note:: Metrics of aggregated bundles will always be shown, but can be
        augmented with new metrics using the usual API.
    """
    def __init__(self, result_bundles, name_metric=None, result=None, context=None):
        self.result_bundles = result_bundles
        self.name_metric = name_metric
        self.extra_metrics = {}
        self.extra_context = context if context is not None else {}
        self._forced_result = result

    @property
    def utc_datetime(self):
        """
        Use the earliest ``utc_datetime`` among the aggregated bundles.
        """
        return min(
            result_bundle.utc_datetime
            for result_bundle in self.result_bundles
        )

    @property
    def context(self):
        """
        Merge the context of all the aggregated bundles, with priority given to
        last in the list.
        """
        # All writes will be done in that first layer
        bases = [self.extra_context]
        bases.extend(
            result_bundle.context
            for result_bundle in self.result_bundles
        )

        return ChainMap(*bases)

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
        return ChainMap(top, base)


class CannotCreateError(RuntimeError):
    """
    Something prevented the creation of a :class:`TestBundle` instance
    """
    pass


class TestBundleMeta(abc.ABCMeta):
    """
    Metaclass of :class:`TestBundle`.

    Method with a return annotation of :class:`ResultBundleBase` are wrapped to
    update the ``context`` attribute of a returned :class:`ResultBundleBase`.

    If ``_from_target`` is defined in the class but ``from_target`` is not, a
    stub is created and the annotation of ``_from_target`` is copied to the
    stub. The annotation is then removed from ``_from_target`` so that it is
    not picked up by exekall.

    The signature of ``from_target`` is the result of merging
    ``super().from_target`` parameters with the ones defined in
    ``_from_target``.
    """
    @staticmethod
    def test_method(func):
        """
        Decorator to intercept returned :class:`ResultBundle` and attach some contextual information.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            if isinstance(res, ResultBundleBase):
                plat_info = self.plat_info
                # Map context keys to PlatformInfo nested keys
                keys = {
                    'board-name': ['name'],
                    'kernel-version': ['kernel', 'version']
                }
                context = {}
                for context_key, plat_info_key in keys.items():
                    try:
                        val = plat_info.get_nested_key(plat_info_key)
                    except KeyError:
                        continue
                    else:
                        context[context_key] = val

                # Only update what is strictly necessary here, so that
                # AggregatedResultBundle ends up with a minimal context state.
                res_context = res.context
                for key, val in context.items():
                    if key not in res_context:
                        res_context[key] = val

            return res

        return wrapper

    def __new__(metacls, cls_name, bases, dct, **kwargs):
        new_cls = super().__new__(metacls, cls_name, bases, dct, **kwargs)

        # Wrap the test methods to add contextual information
        for name, f in dct.items():
            try:
                sig = signature(f)
            except TypeError:
                continue

            annotation = sig.return_annotation
            if isinstance(annotation, type) and issubclass(annotation, ResultBundleBase):
                f = metacls.test_method(f)
                setattr(new_cls, name, f)


        # If that class defines _from_target but not from_target, we create a
        # stub from_target and move the annotations of _from_target to
        # from_target
        if '_from_target' in dct and 'from_target' not in dct:
            assert isinstance(dct['_from_target'], classmethod)
            _from_target = new_cls._from_target

            # Sanity check on _from_target signature
            for name, param in signature(_from_target).parameters.items():
                if name != 'target' and param.kind is not inspect.Parameter.KEYWORD_ONLY:
                    raise TypeError('Non keyword parameters "{}" are not allowed in {} signature'.format(
                        _from_target.__qualname__, name))

            def get_keyword_only_names(f):
                return {
                    param.name
                    for param in signature(f).parameters.values()
                    if param.kind is inspect.Parameter.KEYWORD_ONLY
                }

            try:
                missing_params = (
                    get_keyword_only_names(super(bases[0], new_cls)._from_target)
                    - get_keyword_only_names(_from_target)
                )
            except AttributeError:
                pass
            else:
                if missing_params:
                    raise TypeError('{}._from_target() must at least implement all the parameters of {}._from_target(). Missing parameters: {}'.format(
                        new_cls.__qualname__,
                        bases[0].__qualname__,
                        ', '.join(sorted(missing_params))

                    ))

            def merge_signatures(sig1, sig2):
                parameters = list(sig1.parameters.values())
                sig1_param_names = {param.name for param in parameters}
                parameters.extend(
                    param
                    for param in sig2.parameters.values()
                    if (
                        param.kind is inspect.Parameter.KEYWORD_ONLY
                        and not param.name in sig1_param_names
                    )
                )
                parameters = [
                    param
                    for param in parameters
                    if param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                ]
                return sig1.replace(
                    parameters=parameters,
                    return_annotation=sig2.return_annotation
                )

            # Make a stub that we can freely update
            @functools.wraps(_from_target.__func__)
            def from_target(cls, *args, **kwargs):
                return super(new_cls, cls).from_target(*args, **kwargs)

            # Hide the fact that we wrapped the function, so exekall does not
            # get confused
            del from_target.__wrapped__

            # Fixup the names, so it is not displayed as `_from_target`
            from_target.__name__ = 'from_target'
            from_target.__qualname__ = new_cls.__qualname__ + '.' + from_target.__name__

            # Merge the signatures to get the base signature of super().from_target,
            # and add the keyword-only and return annotation of _from_target.
            from_target.__signature__ = merge_signatures(
                signature(new_cls.from_target.__func__),
                signature(_from_target.__func__),
            )

            # Stich the relevant docstrings
            func = new_cls.from_target.__func__
            from_target_doc = inspect.cleandoc(func.__doc__ or '')
            _from_target_doc = inspect.cleandoc(_from_target.__doc__ or '')
            if _from_target_doc:
                doc = '{}\n\n(**above inherited from** :meth:`{}.{}`)\n\n{}\n'.format(
                    from_target_doc,
                    func.__module__, func.__qualname__,
                    _from_target_doc,
                )
            else:
                doc = from_target_doc

            from_target.__doc__ = doc

            # Make sure the annotation points to an actual class object if it
            # was set, as most of the time they will be strings for factories.
            # Since the wrapper's __globals__ (read-only) attribute is not
            # going to contain the necessary keys to resolve that string, we
            # take care of it here.

            # Only update the annotation if there was one.
            if 'return' in from_target.__annotations__:
                # since we set the signature manually, we also need to update
                # the annotations in it
                sig = from_target.__signature__
                assert sig.return_annotation == cls_name
                from_target.__signature__ = sig.replace(return_annotation=new_cls)

            # Keep the annotations and the signature in sync
            from_target.__annotations__ = annotations_from_signature(from_target.__signature__)

            # De-annotate the _from_target function so it is not picked up by exekall
            del _from_target.__func__.__annotations__

            new_cls.from_target = classmethod(from_target)

        return new_cls

class TestBundle(Serializable, ExekallTaggable, abc.ABC, metaclass=TestBundleMeta):
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
        Note that a default implementation is provided out of ``_from_target``.
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

        from lisa.target import Target
        from lisa.platforms.platinfo import PlatformInfo
        from lisa.utils import ArtifactPath

        class DummyTestBundle(TestBundle):

            def __init__(self, res_dir, plat_info, shell_output):
                super().__init__(res_dir, plat_info)

                self.shell_output = shell_output

            @classmethod
            def _from_target(cls, target:Target, *, res_dir:ArtifactPath) -> 'DummyTestBundle':
                output = target.execute('echo $((21+21))').split()
                return cls(res_dir, target.plat_info, output)

            def test_output(self) -> ResultBundle:
                return ResultBundle.from_bool(
                    any(
                        '42' in line
                        for line in self.shell_output
                    )
                )

    **Usage example**::

        # Creating a Bundle from a live target
        bundle = TestBundle.from_target(target, plat_info=plat_info, res_dir="/my/res/dir")
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

    def get_tags(self):
        try:
            return {'board': self.plat_info['name']}
        except KeyError:
            return {}

    @classmethod
    @abc.abstractmethod
    def _from_target(cls, target, *, res_dir):
        """
        Internals of the target factory method.

        .. note:: This must be a classmethod, and all parameters except
            ``target`` must be keyword-only, i.e. appearing after `args*` or a
            lonely `*`::

                @classmethod
                def _from_target(cls, target, *, foo=33, bar):
                    ...
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
    def from_target(cls, target:Target, *, res_dir:ArtifactPath=None, **kwargs):
        """
        Factory method to create a bundle using a live target

        :param target: Target to connect to.
        :type target: lisa.target.Target

        :param res_dir: Host result directory holding artifacts.
        :type res_dir: str or lisa.utils.ArtifactPath

        This is mostly boiler-plate code around
        :meth:`~lisa.tests.base.TestBundle._from_target`, which lets us
        introduce common functionalities for daughter classes. Unless you know
        what you are doing, you should not override this method, but the
        internal :meth:`lisa.tests.base.TestBundle._from_target` instead.
        """
        cls.check_from_target(target)

        res_dir = res_dir or target.get_res_dir(
            name=cls.__qualname__,
            symlink=True,
        )

        # Make sure that all the relevant dmesg warnings will fire when running
        # things on the target, even if we already hit some warn_once warnings.
        with contextlib.suppress(TargetStableError):
            target.write_value('/sys/kernel/debug/clear_warn_once', '1', verify=False)

        bundle = cls._from_target(target, res_dir=res_dir, **kwargs)

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
        return ArtifactPath.join(res_dir, "{}.yaml".format(cls.__qualname__))

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


class FtraceTestBundleMeta(TestBundleMeta):
    """
    Metaclass of :class:`FtraceTestBundle`.

    This metaclass ensures that each class will get its own copy of
    ``ftrace_conf`` attribute, and that the events specified in that
    configuration are a superset of what is needed by methods using the family
    of decorators :func:`lisa.trace.requires_events`. This makes sure that the
    default set of events is always enough to run all defined methods, without
    duplicating that information. That means that trace events are "inherited"
    at the same time as the methods that need them.

    The ``ftrace_conf`` attribute is typically built by merging these sources:

        * Existing ``ftrace_conf`` class attribute on the
          :class:`RTATestBundle` subclass

        * Events required by methods using :func:`lisa.trace.requires_events`
          decorator (and equivalents).

        * :class:`lisa.trace.FtraceConf` specified by the user and passed to
          :meth:`lisa.trace.FtraceCollector.from_user_conf`
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


class FtraceTestBundle(TestBundle, metaclass=FtraceTestBundleMeta):
    """
    Abstract Base Class for :class:`lisa.wlgen.rta.RTA`-powered TestBundles

    Optionally, an ``ftrace_conf`` class attribute can be defined to hold
    additional FTrace configuration used to record a trace while the synthetic
    workload is being run. By default, the required events are extracted from
    decorated test methods.

    .. seealso: :class:`lisa.tests.base.FtraceTestBundleMeta` for default
        ``ftrace_conf`` content.
    """

    TRACE_PATH = 'trace.dat'
    """
    Path to the ``trace-cmd`` trace.dat file in the result directory.
    """

    @property
    def trace_path(self):
        """
        Path to the ``trace-cmd report`` trace.dat file.
        """
        return ArtifactPath.join(self.res_dir, self.TRACE_PATH)

    # Guard before the cache, so we don't accidentally start depending on the
    # LRU cache for functionnal correctness.
    @non_recursive_property
    @memoized
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
        :returns: a :class:`lisa.trace.Trace` collected in the standard location.

        :Variable keyword arguments: Forwarded to :class:`lisa.trace.Trace`.
        """
        return Trace(self.trace_path, self.plat_info, **kwargs)


class DmesgIgnoredPatterns(StrList, HideExekallID):
    pass

class DmesgTestConf(SimpleMultiSrcConf):
    """
    Configuration class for :meth:`lisa.tests.base.DmesgTestBundle.test_dmesg`.

    {generated_help}
    """
    STRUCTURE = TopLevelKeyDesc('dmesg-test-conf', 'Dmesg test configuration', (
        KeyDesc('ignored-patterns', 'List of Python regex matching dmesg entries content to be ignored', [DmesgIgnoredPatterns]),
    ))

    def get_ignored_patterns(self) -> DmesgIgnorePatterns:
        return self.get('ignored-patterns', [])


class DmesgTestBundle(TestBundle):
    """
    Abstract Base Class for TestBundles based on dmesg output.
    """

    DMESG_PATH = 'dmesg.log'
    """
    Path to the dmesg log in the result directory.
    """

    @property
    def dmesg_path(self):
        """
        Path to the dmesg output log file
        """
        return ArtifactPath.join(self.res_dir, self.DMESG_PATH)

    @property
    def dmesg_entries(self):
        """
        List of parsed dmesg output entries
        :class:`devlib.trace.dmesg.KernelLogEntry`.
        """
        with open(self.dmesg_path) as f:
            return [
                KernelLogEntry.from_str(line)
                for line in f.read().splitlines()
                if line.strip()
            ]

    def test_dmesg(self, level='warn', facility=None, ignored_patterns:DmesgIgnoredPatterns=[]) -> ResultBundle:
        """
        Basic test on kernel dmesg output.

        :param level: Any dmesg entr with a level more critical than (and
            including) that will make the test fail.
        :type level: str

        :param facility: Only select entries emitted by the given dmesg
            facility like `kern`. Note that not all versions of `dmesg` are
            able to print it, so specifying it may lead to no entry being
            inspected at all. If ``None``, the facility is ignored.
        :type facility: str or None

        :param ignored_patterns: List of regexes to ignore some messages.
        :type ignored_patterns: list or None
        """
        levels = DmesgCollector.LOG_LEVELS
        # Consider as an issue all levels more critical than `level`
        issue_levels = levels[:levels.index(level) + 1]

        logger = self.get_logger()
        if ignored_patterns:
            logger.info('Will ignore patterns in dmesg output: {}'.format(ignored_patterns))

        ignored_regex = [
            re.compile(pattern)
            for pattern in ignored_patterns
        ]

        issues = [
            entry
            for entry in self.dmesg_entries
            if (
                (entry.facility == facility if facility else True)
                and (entry.level in issue_levels)
                and not any(regex.match(entry.msg) for regex in ignored_regex)
            )
        ]

        res = ResultBundle.from_bool(not issues)
        multiline = len(issues) > 1
        res.add_metric('dmesg output', ('\n' if multiline else '') + '\n'.join(str(entry) for entry in issues))
        return res


class RTATestBundle(FtraceTestBundle, DmesgTestBundle):
    """
    Abstract Base Class for :class:`lisa.wlgen.rta.RTA`-powered TestBundles

    .. seealso: :class:`lisa.tests.base.FtraceTestBundleMeta` for default
        ``ftrace_conf`` content.
    """

    TASK_PERIOD_MS = 16
    """
    A task period you can re-use for your :class:`lisa.wlgen.rta.RTATask`
    definitions.
    """

    NOISE_ACCOUNTING_THRESHOLDS = {
        # Idle task - ignore completely
        # note: since it has multiple comms, we need to ignore them
        TaskID(pid=0, comm=None) : 100,
        # Feeble boards like Juno/TC2 spend a while in sugov
        r"^sugov:\d+$" : 5,
        # The mailbox controller (MHU), now threaded, creates work that sometimes
        # exceeds the 1% threshold.
        r"^irq/\d+-mhu_link$": 1.5
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
        rta_start = sdf[sdf.next_comm.isin(self.rtapp_tasks)].index[0]
        # Find when the last task stops running
        rta_stop = sdf[sdf.prev_comm.isin(self.rtapp_tasks)].index[-1]

        return (rta_start, rta_stop)

    @property
    def rtapp_profile(self):
        """
        Compute the RTapp profile based on ``plat_info``.
        """
        return self.get_rtapp_profile(self.plat_info)

    @property
    def rtapp_tasks(self):
        """
        The rtapp task names as found from the trace in this bundle.

        :return: the list of actual trace task names
        """
        return sorted(itertools.chain.from_iterable(self.rtapp_tasks_map.values()))

    @property
    @requires_events('sched_switch')
    @memoized
    def rtapp_tasks_map(self):
        """
        Mapping of task names as specified in the rtapp profile to list of task
        names found in the trace.

        If the task forked, the list will contain more than one item.
        """
        trace = self.get_trace(events=['sched_switch'])

        prefix_regexps = {
            prefix: re.compile(r"^{}(-[0-9]+)*$".format(re.escape(prefix)))
            for prefix in self.rtapp_profile.keys()
        }

        comms = set(itertools.chain.from_iterable(trace.get_tasks().values()))
        task_map = {
            prefix: sorted(
                comm
                for comm in comms
                if re.match(regexp, comm)
            )
            for prefix, regexp in prefix_regexps.items()
        }

        missing = set(self.rtapp_profile.keys()) - task_map.keys()
        if missing:
            raise RuntimeError("Missing tasks matching the following rt-app profile names: {}"
                                .format(', '.join(missing)))
        return task_map

    @property
    def cgroup_configuration(self):
        """
        Compute the cgroup configuration based on ``plat_info``
        """
        return self.get_cgroup_configuration(self.plat_info)

    @non_recursive_property
    @memoized
    def trace(self):
        """
        :returns: a :class:`lisa.trace.TraceView` cropped to fit the ``rt-app``
            tasks.

        All events specified in ``ftrace_conf`` are parsed from the trace,
        so it is suitable for direct use in methods.

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`lisa.utils.Serializable.to_path` and
        allows updating the underlying path before it is actually loaded to
        match a different folder structure.
        """
        trace = self.get_trace(events=self.ftrace_conf["events"])
        return trace.get_view(self.trace_window(trace))

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
        ignored_ids = list(map(self.trace.get_task_id, self.rtapp_tasks))

        def compute_duration_pct(row):
            return row.runtime * 100 / self.trace.time_range

        df["runtime_pct"] = df.apply(compute_duration_pct, axis=1)
        df['pid'] = df.index

        # Figure out which PIDs to exclude from the thresholds
        for key, threshold in self.NOISE_ACCOUNTING_THRESHOLDS.items():
            # Find out which task(s) this threshold is about
            if isinstance(key, str):
                comms = [comm for comm in df.comm.values if re.match(key, comm)]
                task_ids = [self.trace.get_task_id(comm) for comm in comms]
            else:
                # Use update=False to let None fields propagate, as they are
                # used to indicate a "dont care" value
                task_ids = [self.trace.get_task_id(key, update=False)]

            # For those tasks, check the threshold
            ignored_ids.extend(
                task_id
                for task_id in task_ids
                if df_filter_task_ids(df, [task_id]).iloc[0].runtime_pct <= threshold
            )

        self.get_logger().info("Ignored PIDs for noise contribution: {}".format(
            ", ".join(map(str, ignored_ids))
        ))

        # Filter out unwanted tasks (rt-app tasks + thresholds)
        df_noise = df_filter_task_ids(df, ignored_ids, invert=True)

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
            @update_wrapper_doc(
                func,
                added_by=':meth:`lisa.tests.base.RTATestBundle.test_noisy_tasks`',
                description=textwrap.dedent(
                """
                The returned ``ResultBundle.result`` will be changed to
                :attr:`~lisa.tests.base.Result.UNDECIDED` if the environment was
                too noisy:
                {}
                """).strip().format(
                    inspect.getdoc(cls.test_noisy_tasks)
                )
            )
            @cls.test_noisy_tasks.used_events
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
        return {}

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
    def run_rtapp(cls, target, res_dir, profile=None, ftrace_coll=None, cg_cfg=None):
        """
        Run the given RTA profile on the target, and collect an ftrace trace.

        :param target: target to execute the workload on.
        :type target: lisa.target.Target

        :param res_dir: Artifact folder where the artifacts will be stored.
        :type res_dir: str or lisa.utils.ArtifactPath

        :param profile: ``rt-app`` profile, as a dictionary of
            ``dict(task_name, RTATask)``. If ``None``,
            :meth:`~lisa.tests.base.RTATestBundle.get_rtapp_profile` is called
            with ``target.plat_info``.
        :type profile: dict(str, lisa.wlgen.rta.RTATask)

        :param ftrace_coll: Ftrace collector to use to record the trace. This
            allows recording extra events compared to the default one, which is
            based on the ``ftrace_conf`` class attribute.
        :type ftrace_coll: lisa.trace.FtraceCollector

        :param cg_cfg: CGroup configuration dictionary. If ``None``,
            :meth:`lisa.tests.base.RTATestBundle.get_cgroup_configuration` is
            called with ``target.plat_info``.
        :type cg_cfg: dict
        """

        trace_path = ArtifactPath.join(res_dir, cls.TRACE_PATH)
        dmesg_path = ArtifactPath.join(res_dir, cls.DMESG_PATH)
        ftrace_coll = ftrace_coll or FtraceCollector.from_conf(target, cls.ftrace_conf)
        dmesg_coll = DmesgCollector(target)

        profile = profile or cls.get_rtapp_profile(target.plat_info)
        cg_cfg = cg_cfg or cls.get_cgroup_configuration(target.plat_info)

        wload = RTA.by_profile(target, "rta_{}".format(cls.__name__.lower()),
                               profile, res_dir=res_dir)
        cgroup = cls._target_configure_cgroup(target, cg_cfg)
        as_root = cgroup is not None

        # Pre-hit the calibration information, in case this is a lazy value.
        # This avoids polluting the trace and the dmesg output with the
        # calibration tasks. Since we know that rt-app will always need it for
        # anything useful, it's reasonable to do it here.
        target.plat_info['rtapp']['calib']

        with dmesg_coll, ftrace_coll, target.freeze_userspace():
            wload.run(cgroup=cgroup, as_root=as_root)

        ftrace_coll.get_trace(trace_path)
        dmesg_coll.get_trace(dmesg_path)
        return trace_path

    # Keep compat with existing code
    @classmethod
    def _run_rtapp(cls, *args, **kwargs):
        """
        Has been renamed to :meth:`~lisa.tests.base.RTATestBundle.run_rtapp`, as it really is part of the public API.
        """
        return cls.run_rtapp(*args, **kwargs)

    @classmethod
    def _from_target(cls, target:Target, *, res_dir:ArtifactPath, ftrace_coll:FtraceCollector=None) -> 'RTATestBundle':
        """
        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in
        :meth:`~lisa.tests.base.RTATestBundle.get_rtapp_profile`
        """
        cls.run_rtapp(target, res_dir, ftrace_coll=ftrace_coll)
        plat_info = target.plat_info
        return cls(res_dir, plat_info)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
