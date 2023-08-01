#    Copyright 2022 ARM Limited
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

import functools
from weakref import WeakKeyDictionary
from contextlib import contextmanager, nullcontext
import threading

from wa import Instrument, Parameter
from wa.framework.instrument import very_slow, very_fast
from wa.utils.types import list_of_strings

from lisa.target import Target as LISATarget
from lisa._kmod import LISAFtraceDynamicKmod
from lisa.utils import get_nested_key


class _Default:
    pass


_DEFAULT = _Default()


class _AttrStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._store = WeakKeyDictionary()

    def get(self, obj, attr, setdefault=_DEFAULT):
        with self._lock:
            attrs = self._store.setdefault(obj, {})
            try:
                return attrs[attr]
            except KeyError:
                if setdefault is _DEFAULT:
                    raise
                else:
                    attrs[attr] = setdefault
                    return setdefault

    def set(self, obj, attr, val):
        with self._lock:
            self._store.setdefault(obj, {})[attr] = val


class _AttrProxy:
    """
    Proxy to another object that redirects ``__setattr__`` to a different
    backing storage so that the original object is not modified.
    """
    _STORE = _AttrStore()

    def __init__(self, obj, subkey=None):
        super().__setattr__('_obj', obj)
        super().__setattr__('_subkey', subkey)

    def __getattr__(self, attr):
        try:
            return self._STORE.get(self._obj, self._subkey)[attr]
        except KeyError:
            return getattr(self._obj, attr)

    def __setattr__(self, attr, val):
        attrs = self._STORE.get(self._obj, self._subkey, setdefault={})
        attrs[attr] = val


class LisaKmodInstrument(Instrument):
    name = 'lisa-kmod'
    description = """
    Compile and load LISA kernel module when the trace-cmd instrument is used.

    The events to enable are taken from the trace-cmd configuration. Disabling
    the trace-cmd instrument will also disable that instrument.

    Example config:

    .. code:: yaml

        lisa-kmod:
            kernel_src: /path/to/linux/kernel/tree/sources
            build_env:
                # Using "build-env: alpine" will use an Alpine Linux chroot,
                # removing the need to have your own toolchain installed.
                build-env: host
                build-env-settings:
                    host:
                        # Extra entry to PATH when running the toolchain in the "host" build-env
                        toolchain-path: /foobar
    """

    parameters = [
        Parameter('kernel_src', kind=str, default=None,
                  description="""
                  Path to kernel sources
                  """),

        Parameter('build_env', kind=dict, default={'build-env': 'host'},
                  description="""
                  Configuration of the build environment
                  """),

        Parameter('ftrace_events', kind=list_of_strings, default=[],
                  description="""
                  List of ftrace events that should be enabled in the kernel
                  module. Events specified in the trace-cmd instrument config
                  will also be used.
                  """),
    ]

    def __init__(self, target, kernel_src, build_env, ftrace_events, **kwargs):
        super().__init__(target, **kwargs)
        self._lisa_target = LISATarget._from_devlib_target(
            target=target,
            lazy_platinfo=True,
            kernel_src=kernel_src,
            kmod_build_env=build_env,
        )
        self._ftrace_events = set(ftrace_events)
        self._kmod = None
        self._cm = None
        self._features = set()

        # Add a new attribute to the devlib target so we can find ourselves
        # from the monkey-patched methods.
        _AttrProxy(self.target).kmod_instrument = self

    @classmethod
    def _monkey_patch(cls, instrument):
        patch = dict(
            initialize=cls._initialize_cm,
            setup=cls._setup_cm,
            start=cls._start_cm,
            stop=cls._stop_cm,
        )

        def make_wrapper(orig, f):
            @very_slow
            @functools.wraps(orig)
            def wrapper(self, context, *args, **kwargs):
                target = _AttrProxy(self.target)
                def bind():
                    try:
                        instr = target.kmod_instrument
                    except AttributeError:
                        return nullcontext
                    else:
                        return f.__get__(instr, type(instr))

                # Only enable the instrument for jobs that use our
                # augmentation.
                job = context.current_job
                if job is None:
                    _f = bind()
                else:
                    # Track the state per job and per patched function name.
                    # This way, we will only run the code once per job at each
                    # stage of the setup, even if multiple monkey patched
                    # instruments are used simultaneously
                    job = _AttrProxy(job, subkey=orig.__name__)

                    # Only run the kmod code once per job and per patched
                    # function. If multiple instruments are monkey patched, we
                    # will be triggered more than once per job.
                    has_run = getattr(job, 'kmod_has_run', False)
                    job.kmod_has_run = True

                    if has_run:
                        _f = nullcontext
                    elif cls.name in job.spec.augmentations:
                        _f = bind()
                    else:
                        _f = nullcontext

                with _f(context, *args, **kwargs):
                    return orig.__get__(self, type(self))(context, *args, **kwargs)

            return wrapper

        for attr, f in patch.items():
            setattr(
                instrument,
                attr,
                make_wrapper(
                    getattr(
                        instrument,
                        attr,
                        # No-op method, since getattr() already crawls the MRO
                        # there is truly nothing to do if the method does not
                        # exist.
                        lambda *args, **kwargs: None
                    ),
                    f
                )
            )

    def _all_ftrace_events(self, context):
        try:
            trace_cmd_events = get_nested_key(
                context.cm.run_config.augmentations,
                ['trace-cmd', 'events'],
            )
        except KeyError:
            trace_cmd_events = []

        return set(trace_cmd_events) | set(self._ftrace_events)

    def _run(self):
        features = sorted(self._features)
        self.logger.info(f'Enabling LISA kmod features {", ".join(features)}')
        return self._kmod.run(
            kmod_params={
                'features': features,
            }
        )

    @contextmanager
    def _initialize_cm(self, context):
        # Note that this function will be ran for each monkey patched
        # instrument, unlike the other methods ran in job context.
        events = self._all_ftrace_events(context)
        kmod = self._lisa_target.get_kmod(LISAFtraceDynamicKmod)
        self._features = set(kmod._event_features(events))
        self._kmod = kmod

        # Load the module while running the instrument's initialize so that the
        # events are visible in the kernel at that point.
        with self._run():
            yield

    @contextmanager
    def _setup_cm(self, context):
        self._cm = self._run()
        yield

    @contextmanager
    def _start_cm(self, context):
        self.logger.info(f'Loading LISA kmod')
        self._cm.__enter__()
        yield

    @contextmanager
    def _stop_cm(self, context):
        self.logger.info(f'Unloading LISA kmod')
        self._cm.__exit__(None, None, None)
        yield


# Monkey-patch the trace-cmd instrument in order to reliably load the
# module before the detection of kernel events. Otherwise, the random
# ordering would lead to ftrace events being detected by the
# FtraceCollector before loading the kernel module, and the events
# would therefore be disabled by devlib with a warning.
from wa.instruments.trace_cmd import TraceCmdInstrument
LisaKmodInstrument._monkey_patch(TraceCmdInstrument)
