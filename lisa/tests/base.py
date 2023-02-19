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

import gc
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
import types
import warnings
from operator import attrgetter
import typing

from datetime import datetime
from collections import OrderedDict, ChainMap, defaultdict, Counter
from collections.abc import Mapping
from inspect import signature

import pandas as pd
import IPython.display

from devlib.collector.dmesg import KernelLogEntry
from devlib import TargetStableError

from lisa.analysis.tasks import TasksAnalysis
from lisa.analysis.rta import RTAEventsAnalysis
from lisa.trace import requires_events, TraceEventCheckerBase, AndTraceEventChecker
from lisa.trace import Trace, TaskID
from lisa.wlgen.rta import RTA, PeriodicWload, RTAPhase, leaf_precedence
from lisa.target import Target

from lisa.utils import (
    Serializable, memoized, lru_memoized, ArtifactPath, non_recursive_property,
    update_wrapper_doc, ExekallTaggable, annotations_from_signature,
    get_sphinx_name, optional_kwargs, group_by_value, kwargs_dispatcher,
    dispatch_kwargs, Loggable, kwargs_forwarded_to, docstring_update,
    is_running_ipython,
)

from lisa.datautils import df_filter_task_ids, df_window

from lisa.trace import FtraceCollector, FtraceConf, DmesgCollector, ComposedCollector
from lisa.conf import (
    SimpleMultiSrcConf, KeyDesc, TopLevelKeyDesc,
)
from lisa.pelt import pelt_settling_time


def _nested_formatter(multiline):
    def sort_mapping(data):
        if isinstance(data, Mapping):
            # Ensure stable ordering of keys if possible
            try:
                data = OrderedDict(sorted(data.items()))
            except TypeError:
                data = data

        return data

    if multiline:
        def format_data(data, level=0):
            idt = '\n' + ' ' * 4 * level

            def indent(s):
                stripped = s.strip()
                if '\n' in stripped:
                    return idt + stripped.replace('\n', idt)
                else:
                    return stripped

            if isinstance(data, TestMetric):
                out = data.pretty_format(multiline=multiline)
                out = indent(out) if '\n' in out else out

            elif isinstance(data, Mapping):
                data = sort_mapping(data)
                body = '\n'.join(
                    f'{key}: {format_data(data, level + 1)}'
                    for key, data in data.items()
                )
                out = indent(body)

            else:
                out = str(data)

            return out
    else:
        def format_data(data):
            # Handle recursive mappings, like metrics of AggregatedResultBundle
            if isinstance(data, Mapping):
                data = sort_mapping(data)
                return '{' + ', '.join(
                    f'{key}={format_data(data)}'
                    for key, data in data.items()
                ) + '}'

            else:
                return str(data)

    return format_data


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
        return self.pretty_format(multiline=False)

    def pretty_format(self, multiline=True):
        """
        Pretty print the metrics.

        :param multiline: If ``True``, use a multiline format.
        :type multiline: bool
        """
        format_data = _nested_formatter(multiline=multiline)
        result = format_data(self.data)

        if self.units:
            result += ' ' + self.units

        return result

    def __repr__(self):
        return f'{type(self).__name__}({self.data}, {self.units})'


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

    SKIPPED = 4
    """
    The test does not make sense on this platform and should therefore be skipped.

    .. note:: :attr:`UNDECIDED` should be used when the data are inconclusive
        but the test still makes sense on the target.
    """

    @property
    def lower_name(self):
        """Return the name in lower case"""
        return self.name.lower()


class ResultBundleBase(Exception):
    """
    Base class for all result bundles.

    .. note:: ``__init__`` is not provided as some classes uses properties to
        provide some of the attributes.
    """

    def __bool__(self):
        """
        ``True`` if the ``result`` is :attr:`Result.PASSED`, ``False``
        otherwise.
        """
        return self.result is Result.PASSED

    def __str__(self):
        return self.pretty_format(multiline=False)

    def pretty_format(self, multiline=True):
        format_data = _nested_formatter(multiline=multiline)
        metrics_str = format_data(self.metrics)
        if '\n' in metrics_str:
            idt = '\n' + ' ' * 4
            metrics_str = metrics_str.replace('\n', idt)
        else:
            metrics_str = ': ' + metrics_str

        return self.result.name + metrics_str

    def _repr_pretty_(self, p, cycle):
        "Pretty print instances in Jupyter notebooks"
        p.text(self.pretty_format())

    def add_metric(self, name, data, units=None):
        """
        Lets you append several test :class:`TestMetric` to the bundle.

        :Parameters: :class:`TestMetric` parameters
        """
        self.metrics[name] = TestMetric(data, units)

    def display_and_exit(self) -> type(None):
        print(f"Test result: {self}")
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

    @classmethod
    def raise_skip(cls, msg, from_=None, **kwargs):
        """
        Raise an :class:`ResultBundle` with the :attr:`Result.SKIPPED` result,
        thereby short-circuiting the rest of the test.

        :param msg: Reason why the test is skipped
        :type msg: str

        :param from_: Other exception that lead to the test being skipped. It
            will be used as the ``Y`` in ``raise X from Y``.
        :type from_: Exception or None

        This is typically used as a way to bail out while indicating to the user
        that the test has essentially been skipped because the target does not
        support what the test is testing.
        """
        res = cls(Result.SKIPPED, **kwargs)
        res.add_metric('skipped-reason', msg)
        raise res from from_


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


class TestBundleMeta(abc.ABCMeta):
    """
    Metaclass of :class:`TestBundleBase`.

    Method with a return annotation of :class:`ResultBundleBase` are wrapped to:

        * Update the ``context`` attribute of a returned
          :class:`ResultBundleBase`

        * Add an ``undecided_filter`` attribute, with
          :meth:`add_undecided_filter` decorator, so that any test method can
          be used as a pre-filter for another one right away.

        * Wrap ``_from_target`` to provide a single ``collector`` parameter,
          built from the composition of the collectors provided by
          ``_make_collector`` methods in the base class tree.

    If ``_from_target`` is defined in the class but ``from_target`` is not, a
    stub is created and the annotation of ``_from_target`` is copied to the
    stub. The annotation is then removed from ``_from_target`` so that it is
    not picked up by exekall.

    The signature of ``from_target`` is the result of merging the original
    ``cls.from_target`` parameters with the ones defined in ``_from_target``.
    """
    @classmethod
    def test_method(metacls, func):
        """
        Decorator to intercept returned :class:`ResultBundle` and attach some contextual information.
        """
        def update_res(test_bundle, res):
            plat_info = test_bundle.plat_info
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

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                res = func(self, *args, **kwargs)
            except ResultBundleBase as res:
                update_res(self, res)
                raise
            else:
                if isinstance(res, ResultBundleBase):
                    update_res(self, res)
                return res

        wrapper = metacls.add_undecided_filter(wrapper)
        return wrapper

    @classmethod
    def collector_factory(cls, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        wrapper._COLLECTOR_FACTORY = True
        return wrapper

    @staticmethod
    def add_undecided_filter(func):
        """
        Turn any method returning a :class:`ResultBundleBase` into a decorator
        that can be used as a test method filter.

        The filter decorator is accessible as the ``undecided_filter``
        attribute of the decorated method.

        Once a test is decorated, the filter method will be run in addition to
        the wrapped test, and if the filter does not succeed, the
        :class:`ResultBundleBase` result will be set to
        :attr:`Result.UNDECIDED`.

        :Example:

        .. code-block:: python

            class Foo(TestBundle):
                @TestBundle.add_undecided_filter
                def test_foo(self, xxx=42, ...):
                    ...

                # Alternatively, ResultBundle return annotation will
                # automatically decorate the method with TestBundleMeta
                # metaclass.
                def test_foo(self, xxx=42, ...) -> ResultBundle:
                    ...

            class Bar(Foo):
                # Set xxx=55 as default, but this can be overriden when
                # test_bar() is called.
                @Foo.test_foo.undecided_filter(xxx=77)
                def test_bar(self, yyy=43, ...) -> ResultBundle:
                    ...

        The resulting decorated method can take the union of keyword
        parameters::

            bar = Bar()
            bar.test_bar(xxx=33, yyy=55)
            # Same as
            bar.test_bar(33, yyy=55)
            # But this fails, since only keyword arguments can be passed to the
            # wrapping pre-test
            bar.test_bar(33, 55)

        If there is a parameter conflict, it is detected at import time and will
        result in a :exc:`TypeError`.

        .. note:: Even if the pre-test does not succeed, the wrapped test is
            still executed, so that the ResultBundle metrics are updated and
            the artifacts still produced. This can be important in order to
            manually analyse results in case the pre-filter was overly
            conservative and marked a usable result as UNDECIDED.

        """
        @optional_kwargs
        def decorator(wrapped_test, **preset_kwargs):
            # Propagate the events used by the filter
            try:
                used_events = func.used_events
            except AttributeError:
                used_events = lambda x: x

            @used_events
            @update_wrapper_doc(
                wrapped_test,
                added_by=func,
                sig_from=func,
                description=textwrap.dedent(
                    """
                    The returned ``ResultBundle.result`` will be changed to
                    :attr:`~lisa.tests.base.Result.UNDECIDED` if {} does not
                    succeed (i.e. either
                    :attr:`~lisa.tests.base.Result.UNDECIDED` or
                    :attr:`~lisa.tests.base.Result.FAILED`).

                    {}
                    """).strip().format(
                        get_sphinx_name(func, style='rst', abbrev=True),
                        inspect.getdoc(func),
                    ),
            )
            @kwargs_dispatcher(
                {
                    func.__get__(0): 'filter_kwargs',
                },
                # Better safe than sorry, there is no guarantee that the tests
                # won't step on each other's toes
                allow_overlap=False,
            )
            @functools.wraps(wrapped_test)
            def filter_wrapper(self, *args, filter_kwargs=None, **kwargs):
                # Merge-in the presets
                filter_kwargs = {
                    **preset_kwargs,
                    **filter_kwargs,
                }

                # Run the wrapped test no matter what, so we get the metrics
                # and also the artifacts
                res = wrapped_test(self, *args, **kwargs)
                filter_res = func(self, **filter_kwargs)
                res.metrics.update(filter_res.metrics)

                if not filter_res:
                    res.result = Result.UNDECIDED
                    res.add_metric('undecided-reason', f'{func.__qualname__} failed')

                return res

            return filter_wrapper

        func.undecided_filter = decorator
        return func

    @classmethod
    def __prepare__(metacls, cls_name, bases, **kwargs):
        # Decorate each method when it is bound to its name in the class'
        # namespace, so that other methods can use e.g. undecided_filter
        # If we do that from __new__, the decoration will happen after all
        # methods are defined, just before the class object is created.
        class NS(dict):
            def __setitem__(self, name, f):
                if isinstance(f, types.FunctionType):
                    # Wrap the test methods to add contextual information
                    sig = signature(f)
                    annotation = sig.return_annotation
                    if isinstance(annotation, type) and issubclass(annotation, ResultBundleBase):
                        f = metacls.test_method(f)

                super().__setitem__(name, f)

        return NS()

    @staticmethod
    def _make_collector_cm_factory(cls):
        """
        Create the method in charge of creating the collector for the test.

        This method is created by aggregating the ``_make_collector`` of all
        base classes into one :class:`lisa.trace.ComposedCollector`.

        The resulting method is then used to consume the user-level parameters
        exposed by each ``_make_collector`` and turn it into a single
        ``collector`` parameter passed to :meth:`_from_target`.
        """

        def find_factories(cls):
            def predicate(f):
                if isinstance(f, (classmethod, staticmethod)):
                    _f = f.__func__
                else:
                    _f = f

                return (
                    getattr(_f, '_COLLECTOR_FACTORY', False) or
                    (
                        hasattr(_f, '__wrapped__') and
                        find_factories(_f.__wrapped__)
                    )
                )

            factories = inspect.getmembers(cls, predicate)
            return list(map(
                # Unbind the method and turn it again into an unbound
                # classmethod
                lambda member: classmethod(member[1].__func__),
                factories
            ))

        factories_f = find_factories(cls)

        # Bind the classmethods to remove the first parameter from their
        # signature
        factories = [
            f.__get__(None, cls)
            for f in factories_f
        ]

        params = {
            param: param.name
            for f in factories
            for param in inspect.signature(f).parameters.values()
            if param.kind == param.KEYWORD_ONLY
        }
        for _name, _params in group_by_value(params, key_sort=attrgetter('name')).items():
            if len(_params) > 1:
                _params = ', '.join(map(str, _params))
                raise TypeError(f'Conflicting parameters for {cls.__qualname__} collectors factory: {_params}')

        params = sorted(params.keys(), key=attrgetter('name'))

        @classmethod
        def factory(cls, **kwargs):
            factories = [
                f.__get__(None, cls)
                for f in factories_f
            ]

            dispatched = dispatch_kwargs(
                factories,
                kwargs,
                call=True,
                allow_overlap=True,
            )
            cms = [
                cm
                for cm in dispatched.values()
                if cm is not None
            ]

            cms = sorted(
                cms,
                key=attrgetter('_COMPOSITION_ORDER'),
                reverse=True,
            )
            cm = ComposedCollector(cms)
            return cm

        first_param = list(inspect.signature(factory.__func__).parameters.values())[0]

        factory.__func__.__signature__ = inspect.Signature(
            parameters=[first_param] + params,
        )
        factory.__name__ = '_make_collector_cm'
        factory.__qualname__ = f'{cls.__qualname__}.{factory.__name__}'
        factory.__module__ = cls.__module__
        return factory

    def __new__(metacls, cls_name, bases, dct, **kwargs):
        new_cls = super().__new__(metacls, cls_name, bases, dct, **kwargs)

        # Merge the collectors available for that class and pass the
        # composed collector to _from_target
        new_cls._make_collector_cm = metacls._make_collector_cm_factory(new_cls)

        # If that class defines _from_target, stub from_target and move the
        # annotations of _from_target to from_target. If from_target was
        # already defined on that class, it's wrapped by the stub, otherwise
        # super().from_target is used.
        if '_from_target' in dct and not getattr(new_cls._from_target, '__isabstractmethod__', False):
            assert isinstance(dct['_from_target'], classmethod)
            _from_target = new_cls._from_target

            # Sanity check on _from_target signature
            for name, param in signature(_from_target).parameters.items():
                if name != 'target' and param.kind is not inspect.Parameter.KEYWORD_ONLY:
                    raise TypeError(f'Non keyword parameters "{name}" are not allowed in {_from_target.__qualname__} signature')

            # This is necessary since _from_target is then reassigned, and the
            # closure refers to it by name
            _real_from_target = _from_target

            @classmethod
            @kwargs_dispatcher(
                {
                    _from_target: 'from_target_kwargs',
                    new_cls._make_collector_cm: 'collector_kwargs',
                },
                ignore=['collector'],
            )
            def wrapper(cls, target, from_target_kwargs, collector_kwargs):
                cm = cls._make_collector_cm(**collector_kwargs)
                return _real_from_target.__func__(cls, collector=cm, **from_target_kwargs)

            # Make sure to get the return annotation from _real_from_target
            wrapper.__func__.__signature__ = inspect.signature(wrapper.__func__).replace(
                return_annotation=inspect.signature(_real_from_target.__func__).return_annotation
            )
            wrapper.__func__.__annotations__ = annotations_from_signature(wrapper.__func__.__signature__)

            new_cls._from_target = wrapper
            _from_target = new_cls._from_target

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

            if 'from_target' in dct:
                # Bind the classmethod object to the class
                orig_from_target = dct['from_target']
                def get_orig_from_target(cls):
                    return orig_from_target.__get__(cls, cls)
            else:
                def get_orig_from_target(cls):
                    return super(new_cls, cls).from_target

            # Make a stub that we can freely update
            # Merge the signatures to get the base signature of
            # super().from_target.
            @kwargs_forwarded_to(_from_target.__func__)
            @functools.wraps(new_cls.from_target.__func__)
            def from_target(cls, *args, **kwargs):
                from_target = get_orig_from_target(cls)
                return from_target(*args, **kwargs)

            # Hide the fact that we wrapped the function, so exekall does not
            # get confused
            del from_target.__wrapped__

            # Fixup the names, so it is not displayed as `_from_target`
            from_target.__name__ = 'from_target'
            from_target.__qualname__ = new_cls.__qualname__ + '.' + from_target.__name__

            # Stich the relevant docstrings
            func = new_cls.from_target.__func__
            from_target_doc = inspect.cleandoc(func.__doc__ or '')
            _from_target_doc = inspect.cleandoc(_from_target.__doc__ or '')
            if _from_target_doc:
                doc = f'{from_target_doc}\n\n(**above inherited from** :meth:`{func.__module__}.{func.__qualname__}`)\n\n{_from_target_doc}\n'
            else:
                doc = from_target_doc

            from_target.__doc__ = doc

            # Make sure the annotation points to an actual class object if it
            # was set, as most of the time they will be strings for factories.
            # Since the wrapper's __globals__ (read-only) attribute is not
            # going to contain the necessary keys to resolve that string, we
            # take care of it here.
            if inspect.signature(_from_target).return_annotation != inspect.Signature.empty:
                # Since we set the signature manually, we also need to update
                # the annotations in it
                from_target.__signature__ = from_target.__signature__.replace(return_annotation=new_cls)

            # Keep the annotations and the signature in sync
            from_target.__annotations__ = annotations_from_signature(from_target.__signature__)

            # De-annotate the _from_target function so it is not picked up by exekall
            del _from_target.__func__.__annotations__

            new_cls.from_target = classmethod(from_target)

        return new_cls


class TestBundleBase(
    Serializable,
    ExekallTaggable,
    abc.ABC,
    docstring_update('.. note:: As a subclass of :class:`lisa.tests.base.TestBundleBase`, this class is considered as "application" and its API is therefore more subject to change than other parts of :mod:`lisa`.'),
    metaclass=TestBundleMeta
):
    """
    A LISA test bundle.

    :param res_dir: Directory in which the target execution artifacts reside.
        This will also be used to dump any artifact generated in the test code.
    :type res_dir: str

    :param plat_info: Various informations about the platform, that is available
        to all tests.
    :type plat_info: :class:`lisa.platforms.platinfo.PlatformInfo`

    The point of a :class:`TestBundleBase` is to bundle in a single object all of the
    required data to run some test assertion (hence the name). When inheriting
    from this class, you can define test methods that use this data, and return
    a :class:`ResultBundle`.

    Thanks to :class:`~lisa.utils.Serializable`, instances of this class
    can be serialized with minimal effort. As long as some information is stored
    within an object's member, it will be automagically handled.

    Please refrain from monkey-patching the object in :meth:`from_target`.
    Data required by the object to run test assertions should be exposed as
    ``__init__`` parameters.

    .. note:: All subclasses are considered as "application" code, as opposed
        to most of the rest of :mod:`lisa` which is treated as a library. This
        means that the classes and their API is subject to change when needs
        evolve, which is not always backward compatible. It's rarely an issue
        since these classes are used "manually" mostly for debugging, which is
        a version-specific activity. Likewise, the set of tests will evolve as
        existing tests are replaced by more general implementations, that could
        be organized and named differently.

    **Design notes:**

      * :meth:`from_target` will collect whatever artifacts are required
        from a given target, and will then return a :class:`TestBundleBase`.
        Note that a default implementation is provided out of ``_from_target``.
      * :meth:`from_dir` will use whatever artifacts are available in a
        given directory (which should have been created by an earlier call
        to :meth:`from_target` and then :meth:`to_dir`), and will then return
        a :class:`TestBundleBase`.
      * :attr:`VERIFY_SERIALIZATION` is there to ensure the instances can
        serialized and deserialized without error.
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
    :meth:`from_target`.

    .. note:: The deserialized instance is thrown away in order to avoid using
        what is in effect a deepcopy of the original bundle. Using that
        deepcopy greatly increases the memory consumption of long running
        processes.
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
        :meta public:

        Internals of the target factory method.

        .. note:: This must be a classmethod, and all parameters except
            ``target`` must be keyword-only, i.e. appearing after `args*` or a
            lonely `*`::

                @classmethod
                def _from_target(cls, target, *, foo=33, bar):
                    ...
        """

    @classmethod
    def check_from_target(cls, target):
        """
        Check whether the given target can be used to create an instance of this class

        :raises: :class:`lisa.tests.base.ResultBundleBase` with ``result`` as
            :attr:`lisa.tests.base.Result.SKIPPED` if the check fails

        This method should be overriden to check your implementation requirements
        """

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
        except ResultBundleBase:
            return False

    @classmethod
    def from_target(cls, target: Target, *, res_dir: ArtifactPath = None, **kwargs):
        """
        Factory method to create a bundle using a live target

        :param target: Target to connect to.
        :type target: lisa.target.Target

        :param res_dir: Host result directory holding artifacts.
        :type res_dir: str or lisa.utils.ArtifactPath

        :param custom_collector: Custom collector that will be used as a
            context manager when calling the workload.
        :type custom_collector: lisa.trace.CollectorBase

        This is mostly boiler-plate code around
        :meth:`~lisa.tests.base.TestBundleBase._from_target`, which lets us
        introduce common functionalities for daughter classes. Unless you know
        what you are doing, you should not override this method, but the
        internal :meth:`lisa.tests.base.TestBundleBase._from_target` instead.
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
            cls.from_dir(res_dir, update_res_dir=False)

        return bundle

    @classmethod
    @TestBundleMeta.collector_factory
    def _make_custom_collector(cls, *, custom_collector=None):
        return custom_collector

    @classmethod
    def _get_filepath(cls, res_dir):
        """
        :meta public:

        Returns the path of the file containing the serialized object in
        ``res_dir`` folder.
        """
        return ArtifactPath.join(res_dir, f"{cls.__qualname__}.yaml")

    def _save_debug_plot(self, fig, name):
        """
        Save a holoviews debug plot using the bokeh backend and show it in the
        notebook cell.
        """
        self.trace.ana.notebook.save_plot(
            fig,
            filepath=ArtifactPath.join(
                self.res_dir,
                f'{name}.html',
            ),
            backend='bokeh',
        )

        # Check before calling display(), as running it outside a notebook will
        # just print the structure of the element, which is useless
        #
        # TODO: See if we can capture this side effect and re-run it when a
        # memoized test method is called again.
        if is_running_ipython():
            IPython.display.display(fig)

        return fig

    @classmethod
    def _get_referred_objs(cls, obj, predicate=lambda x: True):
        visited = set()
        objs = []

        def update_refs(obj):
            obj_id = id(obj)
            # Avoid cycles. Use the id() of the objects directly since the
            # inclusion check is orders of magnitude faster than checking for
            # inclusing on the object directly. It also handles well non hashable
            # objects and broken __eq__ implementations.
            if obj_id in visited:
                return
            else:
                visited.add(obj_id)
                # Filter-out weird objects that end up in the list and that can
                # trigger a coredump on the interpreter
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    has_class = hasattr(obj, '__class__')

                if has_class and predicate(obj):
                    objs.append(obj)

                for sub in gc.get_referents(obj):
                    update_refs(sub)

        update_refs(obj)
        return objs

    @property
    def _children_test_bundles(self):
        """
        :meta public:

        List of references to :class:`TestBundleBase` instances ``self`` relies on
        (directly *and* indirectly).

        This is used for some post-deserialization fixup that need to walk the
        whole graph of :class:`TestBundleBase`.
        """
        # Work around:
        # https://github.com/pallets/werkzeug/issues/2188
        def predicate(x):
            try:
                return isinstance(x, TestBundleBase)
            except Exception:
                return False

        objs = set(self._get_referred_objs(self, predicate))

        objs.discard(self)
        return objs

    def _fixup_res_dir(self, new):
        orig_root = self.res_dir

        def fixup(obj):
            rel = os.path.relpath(obj.res_dir, orig_root)
            absolute = os.path.abspath(os.path.join(new, rel))
            obj.res_dir = absolute

        for child in self._children_test_bundles | {self}:
            fixup(child)

    @classmethod
    def from_dir(cls, res_dir, update_res_dir=True):
        """
        Wrapper around :meth:`lisa.utils.Serializable.from_path`.

        It uses :meth:`_get_filepath` to get the name of the serialized file to
        reload.
        """
        res_dir = ArtifactPath(root=res_dir, relative='')

        bundle = super().from_path(cls._get_filepath(res_dir))
        # We need to update the res_dir to the one we were given
        if update_res_dir:
            bundle._fixup_res_dir(res_dir)

        return bundle

    def to_dir(self, res_dir):
        """
        See :meth:`lisa.utils.Serializable.to_path`
        """
        super().to_path(self._get_filepath(res_dir))


class FtraceTestBundleBase(TestBundleBase):
    """
    Base class for test bundles needing ftrace traces.

    Optionally, an ``FTRACE_CONF`` class attribute can be defined to hold
    additional FTrace configuration used to record a trace while the synthetic
    workload is being run. By default, the required events are extracted from
    decorated test methods.

    This base class ensures that each subclass will get its own copy of
    ``FTRACE_CONF`` attribute, and that the events specified in that
    configuration are a superset of what is needed by methods using the family
    of decorators :func:`lisa.trace.requires_events`. This makes sure that the
    default set of events is always enough to run all defined methods, without
    duplicating that information. That means that trace events are "inherited"
    at the same time as the methods that need them.

    The ``FTRACE_CONF`` attribute is typically built by merging these sources:

        * Existing ``FTRACE_CONF`` class attribute on the
          :class:`RTATestBundle` subclass

        * Events required by methods using :func:`lisa.trace.requires_events`
          decorator (and equivalents).

        * :class:`lisa.trace.FtraceConf` specified by the user and passed to
          :meth:`lisa.tests.base.TestBundleBase.from_target` as ``ftrace_conf``
          parameter.
    """

    TRACE_PATH = 'trace.dat'
    """
    Path to the ``trace-cmd`` trace.dat file in the result directory.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Collect all the events that can be used by all methods available on
        # that class.
        ftrace_events = []
        for name, obj in inspect.getmembers(cls, callable):
            try:
                used_events = obj.used_events
            except AttributeError:
                continue
            else:
                ftrace_events.append(used_events)

        ftrace_events = AndTraceEventChecker(ftrace_events)

        # Get the ftrace_conf attribute of the class, and make sure it is
        # unique to that class (i.e. not shared with any other parent or
        # sibling classes)
        try:
            ftrace_conf = cls.FTRACE_CONF
        except AttributeError:
            ftrace_conf = None
        else:
            # If the ftrace_conf attribute has been defined in a base
            # class, make sure that class gets its own copy since we are
            # going to modify it
            if 'ftrace_conf' not in cls.__dict__:
                ftrace_conf = copy.copy(ftrace_conf)

        # Re-wrap into an FtraceConf so we get a change to set a correct source
        # name.
        ftrace_conf = FtraceConf(
            conf=ftrace_conf or None,
            src=cls.__qualname__,
            # Let the original object decide of that.
            add_default_src=False,
        )

        # Merge-in a new source to FtraceConf that contains the events we
        # collected
        ftrace_conf.add_merged_src(
            src=f'{cls.__qualname__}(required)',
            conf={
                'events': ftrace_events,
            },
        )

        cls.FTRACE_CONF = ftrace_conf

        # Deprecated, for backward compat only, all new code uses the
        # capitalized version
        cls.ftrace_conf = ftrace_conf

    @classmethod
    @TestBundleBase.collector_factory
    def _make_ftrace_collector(cls, *, target: Target, res_dir: ArtifactPath = None, ftrace_conf: FtraceConf = None):
        cls_conf = cls.FTRACE_CONF or FtraceConf()
        user_conf = ftrace_conf or FtraceConf()

        # Make a copy of the conf, since it may be shared by multiple classes
        conf = copy.copy(cls_conf)

        # Merge user configuration with the test's configuration
        conf.add_merged_src(
            src=f'user+{cls.__qualname__}',
            conf=user_conf,
            optional_events=True,
        )

        # If there is no event, do not collect the trace unless the user asked
        # for it. This can happen for classes that inherit from
        # FtraceTestBundle as a convenience to users without actually needing
        # it internally
        if conf.get('events'):
            path = ArtifactPath.join(res_dir, cls.TRACE_PATH)
            return FtraceCollector.from_conf(
                target=target,
                conf=conf,
                output_path=path,
            )
        else:
            return None

    @property
    def trace_path(self):
        """
        Path to the ``trace-cmd report`` trace.dat file.
        """
        return ArtifactPath.join(self.res_dir, self.TRACE_PATH)

    # Guard before the cache, so we don't accidentally start depending on the
    # LRU cache for functionnal correctness.
    @non_recursive_property
    # Only cache the trace of N bundles at a time, to avoid running out of memory.
    # This should not really impact the test when ran with exekall, since they
    # are sequenced one after another. It would have some speed impact on
    # scripts/notebooks that try to do something with a bunch of
    # FtraceTestBundle.
    @lru_memoized(first_param_maxsize=5)
    def trace(self):
        """
        :returns: a :class:`lisa.trace.TraceView`

        All events specified in ``FTRACE_CONF`` are parsed from the trace,
        so it is suitable for direct use in methods.

        Having the trace as a property lets us defer the loading of the actual
        trace to when it is first used. Also, this prevents it from being
        serialized when calling :meth:`lisa.utils.Serializable.to_path` and
        allows updating the underlying path before it is actually loaded to
        match a different folder structure.
        """
        return self.get_trace(
            events=self.FTRACE_CONF["events"],
            normalize_time=True,
            # Soft limit on the amount of memory used by dataframes kept around
            # in memory by Trace, so that we don't blow up the memory when we
            # have a large-ish number of FTraceTestBundle alive at the same
            # time.
            max_mem_size=500e6,
            # TODO: revisit that. As of pyarrow 2.0.0 and pandas 1.1.4, reading
            # (and maybe writing) parquet fils seem to leak memory. This can
            # take the consumption in the order of tens of gigabytes for a few
            # iterations of the tests with exekall, leading to crashes.
            # Therefore, disable the on-disk swap.
            enable_swap=False,
        )

    def get_trace(self, events=None, **kwargs):
        """
        :returns: a :class:`lisa.trace.Trace` collected in the standard location.

        :Variable keyword arguments: Forwarded to :class:`lisa.trace.Trace`.
        """
        return Trace(self.trace_path, self.plat_info, events=events, **kwargs)


class FtraceTestBundle(FtraceTestBundleBase):
    """
    Dummy subclass of :class:`FtraceTestBundleBase` to be inherited from to
    override :class:`OptionalFtraceTestBundle` in the inheritance tree.
    """
    _make_ftrace_collector = FtraceTestBundleBase._make_ftrace_collector


class OptionalFtraceTestBundle(FtraceTestBundleBase, Loggable):
    @classmethod
    @TestBundleBase.collector_factory
    @kwargs_forwarded_to(FtraceTestBundleBase._make_ftrace_collector)
    def _make_ftrace_collector(cls, **kwargs):
        try:
            return super()._make_ftrace_collector(**kwargs)
        except Exception as e:
            cls.get_logger().warning(f'Could not create ftrace collector: {e}')
            return None


class TestConfBase(SimpleMultiSrcConf):
    """
    Base class for test configurations.

    This class will add a ``test-conf`` top-level key above the level specified
    by the class, so that if the class specifies a ``TopLevelKeyDesc('foo')``,
    the actual top-level key will be ``test-conf/foo``.
    """
    def __init_subclass__(cls, **kwargs):
        structure = copy.copy(cls.STRUCTURE)
        structure.levels = ['test-conf', *structure.levels]
        cls.STRUCTURE = structure
        super().__init_subclass__(**kwargs)


class DmesgTestConf(TestConfBase):
    """
    Configuration class for :meth:`lisa.tests.base.DmesgTestBundle.test_dmesg`.

    {generated_help}
    {yaml_example}
    """
    STRUCTURE = TopLevelKeyDesc('dmesg', 'Dmesg test configuration', (
        KeyDesc('ignored-patterns', 'List of Python regex matching dmesg entries *content* to be ignored (see :class:`devlib.collector.dmesg.KernelLogEntry` for how the message is split)', [typing.Sequence[str]]),
    ))


class DmesgTestBundleBase(TestBundleBase):
    """
    Abstract Base Class for TestBundles based on dmesg output.

    .. seealso: Test subclasses should inherit from :class:`DmesgTestBundle` in
        order to require the features.
    """

    DMESG_PATH = 'dmesg.log'
    """
    Path to the dmesg log in the result directory.
    """

    CANNED_DMESG_IGNORED_PATTERNS = {
        'EAS-schedutil': 'Disabling EAS, schedutil is mandatory',
        # On kernel >= 5.6, executable stack will trigger this issue:
    	# kern: warn: [555.927466] process 'root/devlib-target/bin/busybox' started with executable stack
        'executable-stack': 'started with executable stack',
    }
    """
    Mapping of canned patterns to avoid repetition while defining
    :attr:`lisa.tests.base.DmesgTestBundleBase.DMESG_IGNORED_PATTERNS` in
    subclasses.
    """

    DMESG_IGNORED_PATTERNS = [
        CANNED_DMESG_IGNORED_PATTERNS['executable-stack'],
    ]
    """
    List of patterns to ignore in addition to the ones passed to
    :meth:`~lisa.tests.base.DmesgTestBundle.test_dmesg`.
    """

    @classmethod
    @TestBundleBase.collector_factory
    def _make_dmesg_collector(cls, *, target: Target, res_dir: ArtifactPath = None):
        path = ArtifactPath.join(res_dir, cls.DMESG_PATH)
        return DmesgCollector(
            target,
            output_path=path,
        )

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
        :class:`devlib.collector.dmesg.KernelLogEntry`.
        """
        with open(self.dmesg_path) as f:
            return list(KernelLogEntry.from_dmesg_output(f.read()))

    def test_dmesg(self, level='warn', facility=None, ignored_patterns: DmesgTestConf.IgnoredPatterns = None) -> ResultBundle:
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

        :param ignored_patterns: List of regexes to ignore some messages. The
            pattern list is combined with
            :attr:`~lisa.tests.base.DmesgTestBundleBase.DMESG_IGNORED_PATTERNS`
            class attribute.
        :type ignored_patterns: list or None
        """
        levels = DmesgCollector.LOG_LEVELS
        # Consider as an issue all levels more critical than `level`
        issue_levels = levels[:levels.index(level) + 1]
        ignored_patterns = (
            (ignored_patterns or []) +
            (self.DMESG_IGNORED_PATTERNS or [])
        )

        logger = self.logger

        if ignored_patterns:
            logger.info(f'Will ignore patterns in dmesg output: {ignored_patterns}')
            ignored_regex = [
                re.compile(pattern)
                for pattern in ignored_patterns
            ]
        else:
            ignored_regex = []

        issues = [
            entry
            for entry in self.dmesg_entries
            if (
                entry.msg.strip()
                and (entry.facility == facility if facility else True)
                and (entry.level in issue_levels)
                and not any(regex.search(entry.msg.strip()) for regex in ignored_regex)
            )
        ]

        res = ResultBundle.from_bool(not issues)
        multiline = len(issues) > 1
        res.add_metric('dmesg output', ('\n' if multiline else '') + '\n'.join(str(entry) for entry in issues))
        return res


class DmesgTestBundle(DmesgTestBundleBase):
    """
    Dummy subclass of :class:`DmesgTestBundleBase` to be inherited from to
    override :class:`OptionalDmesgTestBundle` in the inheritance tree.
    """
    test_dmesg = DmesgTestBundleBase.test_dmesg
    _make_dmesg_collector = DmesgTestBundleBase._make_dmesg_collector


class OptionalDmesgTestBundle(DmesgTestBundleBase, Loggable):
    @functools.wraps(DmesgTestBundleBase.test_dmesg)
    def test_dmesg(self, *args, **kwargs):
        try:
            return super().test_dmesg(*args, **kwargs)
        except FileNotFoundError:
            self.logger.warning('Could not check dmesg content, as it was not collected')
            return ResultBundle(result=Result.UNDECIDED)

    @classmethod
    @TestBundleBase.collector_factory
    @kwargs_forwarded_to(DmesgTestBundleBase._make_dmesg_collector)
    def _make_dmesg_collector(cls, **kwargs):
        try:
            return super()._make_dmesg_collector(**kwargs)
        except Exception as e:
            cls.get_logger().warning(f'Could not create dmesg collector: {e}')
            return None


class RTATestBundle(FtraceTestBundle, DmesgTestBundle):
    """
    Abstract Base Class for :class:`lisa.wlgen.rta.RTA`-powered TestBundles

    :param rtapp_profile_kwargs: Keyword arguments to pass to
        :meth:`lisa.tests.base.RTATestBundle._get_rtapp_profile` when called from
        the :meth:`lisa.tests.base.RTATestBundle._get_rtapp_profile` property.
    :type rtapp_profile_kwargs: collections.abc.Mapping or None

    .. seealso: :class:`lisa.tests.base.FtraceTestBundle` for default
        ``FTRACE_CONF`` content.
    """

    TASK_PERIOD = 16e-3
    """
    A task period in seconds you can re-use for your
    :class:`lisa.wlgen.rta.RTATask` definitions.
    """

    NOISE_ACCOUNTING_THRESHOLDS = {
        # Idle task - ignore completely
        # note: since it has multiple comms, we need to ignore them
        TaskID(pid=0, comm=None): 100,
        # Feeble boards like Juno/TC2 spend a while in sugov
        r"^sugov:\d+$": 5,
        # Some boards like Hikey960 have noisy threaded IRQs (thermal sensor
        # mailbox ...)
        r"^irq/\d+-.*$": 1.5,
    }
    """
    PID/comm specific tuning for :meth:`test_noisy_tasks`

    * **keys** can be PIDs, comms, or regexps for comms.

    * **values** are noisiness thresholds (%), IOW below that runtime threshold
      the associated task will be ignored in the noise accounting.
    """

    # Roughly 330*2 ms for PELT half life~=32ms
    # This allows enough time for scheduler signals to converge.
    _BUFFER_PHASE_DURATION_S = pelt_settling_time() * 2
    """
    Duration of the initial buffer phase; this is a phase that copies the first
    phase of each task, and that is prepended to the relevant task - this means
    all task in the profile get a buffer phase.
    """

    _BUFFER_PHASE_PROPERTIES = {
        'name': 'buffer',
    }
    """
    Properties of the buffer phase, see :attr:`_BUFFER_PHASE_DURATION_S`
    """

    def __init__(self, res_dir, plat_info, rtapp_profile_kwargs=None):
        super().__init__(res_dir, plat_info)
        self._rtapp_profile_kwargs = dict(rtapp_profile_kwargs or {})

    @RTAEventsAnalysis.df_rtapp_phases_start.used_events
    @RTAEventsAnalysis.df_rtapp_phases_end.used_events
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
        swdf = trace.df_event('sched_switch')

        def get_first_switch(row):
            comm, pid, _ = row.name
            start_time = row['Time']
            task = TaskID(comm=comm, pid=pid)
            start_swdf = df_filter_task_ids(swdf, [task], pid_col='next_pid', comm_col='next_comm')
            pre_phase_swdf = start_swdf[start_swdf.index < start_time]
            # The task with that comm and PID was never switched-in, which
            # means it was still on the current CPU when it was renamed, so we
            # just report phase-start.
            if pre_phase_swdf.empty:
                return start_time
            # Otherwise, we return the timestamp of the switch
            else:
                return pre_phase_swdf.index[-1]

        profile = self.rtapp_profile

        # Find when the first rtapp phase starts, and take the associated
        # sched_switch that is immediately preceding
        phase_start_df = trace.ana.rta.df_rtapp_phases_start(
            wlgen_profile=profile,
        )

        # Get rid of the buffer phase we don't care about
        phase_start_df = phase_start_df[
            phase_start_df['properties'].transform(lambda props: props['meta']['from_test'])
        ]

        rta_start = phase_start_df.apply(get_first_switch, axis=1).min()

        # Find when the last rtapp phase ends
        rta_stop = trace.ana.rta.df_rtapp_phases_end()['Time'].max()

        return (rta_start, rta_stop)

    @property
    def rtapp_profile(self):
        """
        Compute the RTapp profile based on ``plat_info``.
        """

        return self.get_rtapp_profile(
            self.plat_info,
            **self._rtapp_profile_kwargs,
        )

    _rtapp_tasks_events = requires_events('sched_switch')

    @property
    @_rtapp_tasks_events
    @memoized
    def rtapp_task_ids_map(self):
        """
        Mapping of task names as specified in the rtapp profile to list of
        :class:`lisa.trace.TaskID` names found in the trace.

        If the task forked, the list will contain more than one item.
        """
        trace = self.get_trace(events=['sched_switch'])
        names = self.rtapp_profile.keys()
        return {
            name: task_ids
            for name, task_ids in RTA.resolve_trace_task_names(trace, names).items()
        }

    @property
    @_rtapp_tasks_events
    def rtapp_task_ids(self):
        """
        The rtapp task :class:`lisa.trace.TaskID` as found from the trace in
        this bundle.

        :return: the list of actual trace task :class:`lisa.trace.TaskID`
        """
        return sorted(itertools.chain.from_iterable(self.rtapp_task_ids_map.values()))

    @property
    @_rtapp_tasks_events
    def rtapp_tasks_map(self):
        """
        Same as :func:`rtapp_task_ids_map` but with list of strings for values.
        """
        return {
            name: [task_id.comm for task_id in task_ids]
            for name, task_ids in self.rtapp_task_ids_map.items()
        }

    @property
    @_rtapp_tasks_events
    def rtapp_tasks(self):
        """
        Same as :func:`rtapp_task_ids` but as a list of string.

        :return: the list of actual trace task names
        """
        return [task_id.comm for task_id in self.rtapp_task_ids]

    @property
    def cgroup_configuration(self):
        """
        Compute the cgroup configuration based on ``plat_info``
        """
        return self.get_cgroup_configuration(self.plat_info)

    @non_recursive_property
    @lru_memoized(first_param_maxsize=5)
    def trace(self):
        """
        :returns: a :class:`lisa.trace.TraceView` cropped to the window given
            by :meth:`trace_window`.

        .. seealso:: :attr:`FtraceTestBundleBase.trace`
        """
        trace = super().trace
        return trace.get_view(self.trace_window(trace), clear_base_cache=True)

    def df_noisy_tasks(self, with_threshold_exclusion=True):
        """
        :returns: a DataFrame containing all tasks that participate to the test
          noise. i.e. all non rt-app tasks.

        :param with_threshold_exclusion: When set to True, known noisy services
          will be ignored.
        """
        df = self.trace.ana.tasks.df_tasks_runtime()
        df = df.copy(deep=False)

        # We don't want to account the test tasks
        ignored_ids = copy.copy(self.rtapp_task_ids)

        df['runtime_pct'] = df['runtime'] * (100 / self.trace.time_range)
        df['pid'] = df.index

        threshold_exclusion = self.NOISE_ACCOUNTING_THRESHOLDS if with_threshold_exclusion else {}

        # Figure out which PIDs to exclude from the thresholds
        for key, threshold in threshold_exclusion.items():
            # Find out which task(s) this threshold is about
            if isinstance(key, str):
                comms = df.loc[df['comm'].str.match(key), 'comm']
                task_ids = comms.apply(self.trace.get_task_id)
            else:
                # Use update=False to let None fields propagate, as they are
                # used to indicate a "dont care" value
                task_ids = [self.trace.get_task_id(key, update=False)]

            # For those tasks, check the cumulative threshold
            runtime_pct_sum = df_filter_task_ids(df,
                task_ids)['runtime_pct'].sum()
            if runtime_pct_sum <= threshold:
                 ignored_ids.extend(task_ids)


        self.logger.info(f"Ignored PIDs for noise contribution: {', '.join(map(str, ignored_ids))}")

        # Filter out unwanted tasks (rt-app tasks + thresholds)
        df = df_filter_task_ids(df, ignored_ids, invert=True)

        return df.loc[df['runtime'] > 0]

    @TestBundleBase.add_undecided_filter
    @TasksAnalysis.df_tasks_runtime.used_events
    def test_noisy_tasks(self, *, noise_threshold_pct=None, noise_threshold_ms=None):
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
            raise ValueError('Both "noise_threshold_pct" and "noise_threshold_ms" cannot be None')

        # No task can run longer than the recorded duration
        threshold_s = self.trace.time_range

        if noise_threshold_pct is not None:
            threshold_s = noise_threshold_pct * self.trace.time_range / 100

        if noise_threshold_ms is not None:
            threshold_s = min(threshold_s, noise_threshold_ms * 1e3)

        df_noise = self.df_noisy_tasks()

        if df_noise.empty:
            return ResultBundle.from_bool(True)

        res = ResultBundle.from_bool(df_noise['runtime'].sum() < threshold_s)

        pid = df_noise.index[0]
        comm = df_noise['comm'].iloc[0]
        duration_s = df_noise['runtime'].iloc[0]
        duration_pct = df_noise['runtime_pct'].iloc[0]

        metric = {"pid": pid,
                  "comm": comm,
                  "duration (abs)": TestMetric(duration_s, "s"),
                  "duration (rel)": TestMetric(duration_pct, "%")}
        res.add_metric("noisiest task", metric)

        return res

    def df_estimated_freq(self, tasks, window=None):
        """
        Provide an estimated CPU(s) frequency

        :pram tasks: Set of tasks to take into account when providing the
            estimates
        type tasks: list(lisa.trace.TaskID)

        :param window: Optional, restrict the data to given window only
        :type window: tuple(float, float)

        :returns: a :class:`pandas.DataFrame` with

            * CPU id as index
            * A ``runtime`` column with total runtime reported on given CPU
            * A ``counter`` column with CPU_CYCLES event counter
            * A ``freq``column with the estimated frequency
        """
        try:
            df_perf = self.trace.df_event('perf_counter')
        except:
            return pd.DataFrame()

        # CPU_CYCLES counter:
        # ARMV8_PMUV3_PERFCTR_CPU_CYCLES	0x0011
        # ARMV[6/7]_PERFCTR_CPU_CYCLES          0xFF
        df_perf = df_perf.query('counter_id == 17 or counter_id == 255').copy(deep=False)

        if df_perf.empty:
            self.logger.warning("CPU_CYCLES event counter missing")
            return df_perf


        d_perf = defaultdict(Counter)

        for task in tasks:

            def skip_task(task):
                return not task.pid != 0

            if skip_task(task):
                continue

            try:
                df_act = self.trace.ana.tasks.df_task_activation(task)
                df_act = df_act.query('active == 1').copy(deep=False)
            except:
                continue

            if window is not None:
                df_act = df_window(df_act, window, method='inclusive')

            def __map_perf_events(entry, df_perf):
                # Find corresponding events for sched_switch ones (activation)
                df = df_perf.query('cpu == @entry.cpu')

                __loc = df.index.get_indexer(
                        [entry.name + entry.duration],
                        method='nearest'
                )

                value = df.iloc[__loc[0]].value
                __loc = df.index.get_indexer(
                        [entry.name],
                        method='nearest'
                )
                value -= df.iloc[__loc[0]].value
                return value

            df_act['counter'] = df_act.apply(
                    lambda x: __map_perf_events(x, df_perf), axis = 1
            )

            for cpu, group_df in df_act.groupby('cpu'):
                d_perf[cpu]['runtime'] = group_df.duration.sum()
                d_perf[cpu]['counter'] = group_df.counter.sum()

        df_freq = pd.DataFrame(d_perf).T
        df_freq.index.name = 'cpu'
        df_freq['freq'] = df_freq['counter'] / df_freq['runtime'] / 1000
        return df_freq

    @requires_events('perf_counter')
    @TestBundleBase.add_undecided_filter
    @TasksAnalysis.df_task_activation.used_events
    def test_estiamted_freq(self, skip_verification=False, freq=None):
        """
        Verify expected frequency for given CPUs

        :param skip_verification: Do not perform any validation
        :type skip_verification: bool

        :param freq: Expected frequency to validate estimated one against
        :type freq: float

        If freq is not specified, the estimated frequency is validated against
        the maximum one for given CPUs.
        """
        df = self.df_estimated_freq(self.rtapp_task_ids, self.trace.window)

        if skip_verification:
            res = ResultBundle.from_bool(True)
            res.add_metric("estimated frequencies", df.T.to_dict())
            return res

        if df.empty:
            # Do not compromise the test if smth went wrong with setting up
            # the counters
            res = ResultBundle.from_bool(True)
            self.logger.warning("Unable to estimate frequency")
            return res

        cpus = df.index.values

        def __get_cpu_freq(cpu, freq):
            return cpu, freq if freq is not None else self.plat_info['freqs'][cpu][-1]

        df_expected = pd.DataFrame({
            __get_cpu_freq(cpu, freq) for cpu in cpus
            }, columns=['cpu', 'freq']).set_index('cpu')

        df = df[['freq']]
        df_expected = df / df_expected

        df.reset_index(inplace=True)
        res = ResultBundle.from_bool(df_expected.query('freq < 0.9').empty)
        res.add_metric("estimated frequencies", df.T.to_dict())
        return res

    @classmethod
    def unscaled_utilization(cls, plat_info, cpu, utilization_pct):
        """
        Convert utilization scaled to a CPU to a 'raw', unscaled one.

        :param capacity: The CPU against which ``utilization_pct``` is scaled
        :type capacity: int

        :param utilization_pct: The scaled utilization in %
        :type utilization_pct: int

        .. seealso: In most cases,
            `PeriodicWload(scale_for_cpu=..., scale_for_freq=...)` is easier to
            use and leads to clearer code.
        """
        return PeriodicWload(
            duty_cycle_pct=utilization_pct,
            scale_for_cpu=cpu,
        ).unscaled_duty_cycle_pct(plat_info)

    @classmethod
    def get_rtapp_profile(cls, plat_info, **kwargs):
        """
        Returns a :class:`dict` with task names as keys and
        :class:`lisa.wlgen.rta.RTATask` as values.

        The following modifications are done on the profile returned by
        :meth:`_get_rtapp_profile`:

            * A buffer phase may be inserted at the beginning of each task in order
              to stabilize some kernel signals.
            * A ``from_test`` meta key is added to each
              :class:`lisa.wlgen.rta.RTAPhase` with a boolean value that is
              ``True`` if the phase comes from the test itself and ``False`` if
              it was added here (e.g. the buffer phase). This allows
              future-proof filtering of phases in the test code when inspecting
              the profile by looking at ``phase['meta']['from_test']``.

        .. note:: If you want to override the method in a subclass, override
            :meth:`_get_rtapp_profile` instead.
        """

        def add_buffer(task):
            template_phase = task.phases[0]
            wload = template_phase['wload']
            task = task.with_props(meta=leaf_precedence({'from_test': True}))
            if 'name' not in task:
                task = task.with_props(name='test')

            # Don't add the buffer phase if it has a nil duration
            if not cls._BUFFER_PHASE_DURATION_S:
                return task
            elif isinstance(wload, PeriodicWload):
                # Notes:
                #
                # Using a small period to allow the util_avg to be very close
                # to duty_cycle, but that also makes the duty_cycle converge to
                # a wrong value (rtapp looses fidelity with small periods,
                # maybe due to tracing overhead). Therefore we just replicate
                # the period.
                ref_wload = PeriodicWload(
                    duration=cls._BUFFER_PHASE_DURATION_S,
                )

                buffer_phase = RTAPhase(
                    # Override some parameters with the reference ones
                    prop_wload=ref_wload & wload,
                    # Pin to the same CPUs and NUMA nodes if any, so that we
                    # also let the runqueue signals converge and things like
                    # that, if it's going to matter later.
                    prop_cpus=template_phase.get('cpus'),
                    prop_numa_nodes_membind=template_phase.get('numa_nodes_membind'),
                    prop_meta={'from_test': False},
                    properties=cls._BUFFER_PHASE_PROPERTIES,
                )

                # Prepend the buffer task
                return buffer_phase + task
            else:
                return task

        profile = cls._get_rtapp_profile(plat_info, **kwargs)
        return {
            name: add_buffer(task)
            for name, task in profile.items()
        }

    @classmethod
    @abc.abstractmethod
    def _get_rtapp_profile(cls, plat_info):
        """
        :meta public:

        :returns: a :class:`dict` with task names as keys and
          :class:`lisa.wlgen.rta.RTATask` as values

        This is the method you want to override to specify what is your
        synthetic workload.
        """

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

        try:
            cgroups = target.cgroups
        except AttributeError:
            ResultBundle.raise_skip('cgroups are not available on this target')

        kind = cfg['controller']
        try:
            ctrl = cgroups.controllers[kind]
        except KeyError:
            ResultBundle.raise_skip(f'"{kind}" cgroup controller unavailable')

        cg = ctrl.cgroup(cfg['name'])
        cg.set(**cfg['attributes'])

        return '/' + cg.name

    @classmethod
    def run_rtapp(cls, target, res_dir, profile=None, collector=None, cg_cfg=None, wipe_run_dir=True, update_cpu_capacities=None):
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

        :param collector: Context manager collector to use while running rt-app.
        :type collector: lisa.trace.ComposedCollector

        :param cg_cfg: CGroup configuration dictionary. If ``None``,
            :meth:`lisa.tests.base.RTATestBundle.get_cgroup_configuration` is
            called with ``target.plat_info``.
        :type cg_cfg: dict

        :param wipe_run_dir: Remove the run directory on the target after
            execution of the workload.
        :type wipe_run_dir: bool

        :param update_cpu_capacities: Attempt to update the CPU capacities
            based on the calibration values of rtapp to get the most accurate
            reproduction of duty cycles.
        :type update_cpu_capacities: bool
        """
        logger = cls.get_logger()
        trace_path = ArtifactPath.join(res_dir, cls.TRACE_PATH)

        profile = profile or cls.get_rtapp_profile(target.plat_info)
        cg_cfg = cg_cfg or cls.get_cgroup_configuration(target.plat_info)

        try:
            ftrace_coll = collector['ftrace']
        except KeyError:
            trace_events = []
        else:
            trace_events = [
                event.replace('userspace@rtapp_', '')
                for event in ftrace_coll.events
                if event.startswith('userspace@rtapp_')
            ]


        wload = RTA.from_profile(
            target=target,
            profile=profile,
            res_dir=res_dir,
            name=f"rta_{cls.__name__.casefold()}",
            trace_events=trace_events,
            # Force the default value for all settings so that the test does
            # not depend on the environment setup.
            force_defaults=True,
            no_force_default_keys=[
                # Since "taskgroup" cannot be always expected to work in case
                # cgroupfs is not mounted at all, we will not force a default
                # value for it.
                'taskgroup'
            ],
        )

        profile_str = '\n'.join(
            'Task {}:\n{}'.format(
                task,
                textwrap.indent(str(phase), ' ' * 4)
            )
            for task, phase in profile.items()
        )

        logger.info(f'rt-app workload:\n{profile_str}')
        logger.debug(f'rt-app JSON:\n{wload.conf.json}')
        cgroup = cls._target_configure_cgroup(target, cg_cfg)
        as_root = bool(
            cgroup is not None or trace_events
        )

        wload = wload(
            wipe_run_dir=wipe_run_dir,
            cgroup=cgroup,
            as_root=as_root,
            update_cpu_capacities=update_cpu_capacities,
        )

        with target.freeze_userspace(), wload, collector:
            wload.run()

        return collector

    # Keep compat with existing code
    @classmethod
    def _run_rtapp(cls, *args, **kwargs):
        """
        :meta public:

        Has been renamed to :meth:`~lisa.tests.base.RTATestBundle.run_rtapp`, as it really is part of the public API.
        """
        return cls.run_rtapp(*args, **kwargs)

    @classmethod
    def _from_target(cls, target: Target, *, res_dir: ArtifactPath, collector=None) -> 'RTATestBundle':
        """
        :meta public:

        Factory method to create a bundle using a live target

        This will execute the rt-app workload described in
        :meth:`~lisa.tests.base.RTATestBundle.get_rtapp_profile`
        """
        cls.run_rtapp(target, res_dir, collector=collector)
        plat_info = target.plat_info
        return cls(res_dir, plat_info)


class TestBundle(OptionalFtraceTestBundle, OptionalDmesgTestBundle, TestBundleBase):
    """
    Dummy class used as a base class for all tests.
    """
    @classmethod
    def check_from_target(cls, target):
        super().check_from_target(target)
        online = set(target.list_online_cpus())
        cpus = set(range(target.plat_info['cpus-count']))
        if not online <= cpus:
            raise ValueError('Online CPUs ({online}) are not a subset of detected CPUs ({cpus})')
        elif online != cpus:
            offline = sorted(cpus - online)
            raise ResultBundle.raise_skip(f'All CPUs must be online (aka not hotplugged) before creating a TestBundle. Offline CPUs: {offline}')


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
