# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

""" Helper module for registering Analysis classes methods """

import contextlib
import inspect
import itertools
import functools
import warnings

from lisa.analysis.base import TraceAnalysisBase
from lisa.utils import Loggable, sig_bind


class _AnalysisPreset:
    def __init__(self, instance, params):
        self._instance = instance
        self._params = params

    def __getattr__(self, attr):
        if attr == '_instance':
            raise AttributeError
        else:
            x = getattr(self._instance, attr)
            try:
                sig = inspect.signature(x)
            except Exception:
                return x
            else:
                extra = {
                    k: v
                    for k, v in self._params.items()
                    if k in sig.parameters
                }

                @functools.wraps(x)
                def wrapper(*args, **kwargs):
                    kwargs = {
                        **extra,
                        **sig_bind(
                            sig,
                            args=args,
                            kwargs=kwargs,
                            include_defaults=False
                        )[0],
                    }
                    return x(**kwargs)

                # Update the signature so it shows the effective default value
                def update_default(param):
                    # Make it keyword-only if it does not have a default value,
                    # otherwise we might end up setting a parameter without a
                    # default after one with a default, which is unfortunately
                    # illegal.
                    if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        kind = param.kind
                    else:
                        kind = param.KEYWORD_ONLY

                    try:
                        default = extra[param.name]
                    except KeyError:
                        default = param.default

                    return param.replace(
                        default=default,
                        kind=kind
                    )

                wrapper.__signature__ = sig.replace(
                    parameters=list(
                        map(
                            update_default,
                            sig.parameters.values()
                        )
                    )
                )

                return wrapper


class AnalysisProxy(Loggable):
    """
    Entry point to call analysis methods on :class:`~lisa.trace.Trace` objects.

    **Example**

    # Call lisa.analysis.LoadTrackingAnalysis.df_task_signal() on a trace::

        df = trace.ana.load_tracking.df_task_signal(task='foo', signal='util')

    The proxy can also be called like a function to define default values for
    analysis methods::

        ana = trace.ana(task='big_0-3')
        ana.load_tracking.df_task_signal(signal='util')

        # Equivalent to:
        ana.load_tracking.df_task_signal(task='big_0-3', signal='util')

        # The proxy can be called again to override the value given to some
        # parameters, and the the value can also be overridden when calling the
        # method:
        ana(task='foo').df_task_signal(signal='util')
        ana.df_task_signal(task='foo', signal='util')

    :param trace: input Trace object
    :type trace: lisa.trace.Trace
    """

    def __init__(self, trace, params=None):
        self._preset_params = params or {}
        self.trace = trace
        # Get the list once when the proxy is built, since we know all classes
        # will have had a chance to get registered at that point
        self._class_map = TraceAnalysisBase.get_analysis_classes()
        self._instance_map = {}

    def __call__(self, **kwargs):
        return self._with_params(
            {
                **self._preset_params,
                **kwargs,
            }
        )

    def _with_params(self, params):
        return self.__class__(
            trace=self.trace,
            params=params,
        )

    @classmethod
    def get_all_events(cls):
        """
        Returns the set of all events used by any of the registered analysis.
        """
        return set(itertools.chain.from_iterable(
            cls.get_all_events()
            for cls in TraceAnalysisBase.get_analysis_classes().values()
        ))

    def __dir__(self):
        """Provide better completion support for interactive notebook usage"""
        return itertools.chain(super().__dir__(), self._class_map.keys())

    def __getattr__(self, attr):
        # dunder name lookup would have succeeded by now, like __setstate__
        if attr.startswith('__') and attr.endswith('__'):
            return super().__getattribute__(attr)

        logger = self.logger

        # First, try to get the instance of the Analysis that was built if we
        # used it already on that proxy.
        try:
            return self._instance_map[attr]
        except KeyError:
            # If that is the first use, we get the analysis class and build an
            # instance of it
            try:
                analysis_cls = self._class_map[attr]
            except KeyError:
                # No analysis class matching "attr", so we log the ones that
                # are available and let an AttributeError bubble up
                try:
                    analysis_cls = super().__getattribute__(attr)
                except Exception:
                    logger.debug(f'{attr} not found. Registered analysis:')
                    for name, cls in list(self._class_map.items()):
                        src_file = '<unknown source>'
                        with contextlib.suppress(TypeError):
                            src_file = inspect.getsourcefile(cls) or src_file

                        logger.debug(f'{name} ({cls}) defined in {src_file}')

                    raise
            else:
                # Allows straightforward composition of plot methods by
                # ensuring that inside an analysis method, self.ana.foo.bar()
                # will call bar with no extra implicit value for bar()
                # parameters.
                proxy = self._with_params({})

                instance = analysis_cls(trace=self.trace, proxy=proxy)
                preset = _AnalysisPreset(
                    instance=instance,
                    params=self._preset_params
                )
                self._instance_map[attr] = preset
                return preset


class _DeprecatedAnalysisProxy(AnalysisProxy):
    def __init__(self, trace, params=None):
        params = {
            # Enable the old behaviour of returning a matplotlib axis when
            # matplotlib backend is in use, otherwise return holoviews
            # objects (unless output='render')
            '_compat_render': True,
            **(params or {})
        }
        super().__init__(trace=trace, params=params)

    def __getattr__(self, attr):
        # Do not catch dunder names
        if not attr.startswith('__'):
            warnings.warn(
                'trace.analysis is deprecated, use trace.ana instead. Note that plot method will return holoviews objects, use output="render" to render them as matplotlib figure to get legacy behaviour',
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getattr__(attr)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
