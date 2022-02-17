# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, Arm Limited and contributors.
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

"""
This module implements an `rt-app <https://github.com/scheduler-tools/rt-app>`_
JSON programmatic configuration file generator, along the class to run it
directly on a :class:`~lisa.target.Target`. This is the backbone of our
scheduler tests, allowing to easily run custom workloads.

The most important classes are:

    * :class:`RTA`: Subclass of :class:`lisa.wlgen.workload.Workload` that can
      run rt-app on a given :class:`~lisa.target.Target`.

    * :class:`RTAConf`: An rt-app configuration file. It can be created either
      from a template JSON or using a programmatic API.

    * :class:`RTAPhase`: The entry point of the programmatic API to build
      rt-app configuration, phase by phase.

    * :class:`WloadPropertyBase`: The base class of all workloads that can be
      given to an :class:`RTAPhase`. It has the following subclasses:

        .. exec::
            from lisa._doc.helpers import get_subclasses_bullets
            from lisa.wlgen.rta import WloadPropertyBase

            print(
                get_subclasses_bullets(
                    WloadPropertyBase,
                    abbrev=True,
                    only_leaves=True,
                    style='rst',
                )
            )

A typical workload would be created this way::

    from lisa.wlgen.rta import RTA, RTAPhase, PeriodicWload, SleepWload

    task = (
        # Phases can be added together so they will be executed in order
        RTAPhase(
            prop_name='first',
            # The workload of a phase is a subclass of WloadPropertyBase
            prop_wload=RunWload(1),
            prop_uclamp=(256, 512),
        ) +
        RTAPhase(
            prop_name='second',
            prop_wload=(
                # Workloads can be added together too
                SleepWload(5) +
                PeriodicWload(
                    duty_cycle_pct=20,
                    period=16e-3,
                    duration=2,
                )
            )
        )
    )

    # Important note: all the classes in this module are immutable. Modifying
    # attributes is not allowed, use the RTAPhase.with_props() if you want to
    # get a new object with updated properties.

    # You can create a modified RTAPhase using with_props(), and the property
    # will be combined with the existing ones.
    # For util clamp, it means taking the toughest restrictions, which in this
    # case are (300, 512).
    task = task.with_props(uclamp=(300, 800))

    # If you want to set the clamp and override any existing value rather than
    # combining it, you can use the override() function
    task = task.with_props(uclamp=override((300, 800)))

    # Similarly, you can delete any property that was already set with delete()
    task = task.with_props(uclamp=delete())

    # Connect to a target
    target = Target.from_default_conf()

    # Mapping of rt-app task names to the phases they will execute
    profile = {'task1': task}

    # Create the RTA object that configures the profile for the given target
    wload = RTA.from_profile(target, profile=profile)

    # Workloads are context managers to do some cleanup on exit
    with wload:
        wload.run()

"""

import abc
import copy
import functools
import inspect
import json
import math
import operator
import os
import re
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Callable
from itertools import chain, product, starmap
from operator import itemgetter
from shlex import quote
from statistics import mean
import contextlib

from devlib import TargetStableError
from devlib.target import KernelConfigTristate

from lisa.pelt import PELT_SCALE
from lisa.utils import (
    TASK_COMM_MAX_LEN,
    ArtifactPath,
    FrozenDict,
    Loggable,
    SimpleHash,
    deprecate,
    fixedpoint,
    fold,
    get_cls_name,
    get_subclasses,
    group_by_value,
    groupby,
    loopify,
    memoized,
    nullcontext,
    order_as,
    value_range,
    get_cls_name,
    get_short_doc,
    kwargs_dispatcher,
    kwargs_forwarded_to,
    PartialInit,
)
from lisa.wlgen.workload import Workload
from lisa.conf import DeferredValueComputationError


def _to_us(x):
    """
    Convert seconds to microseconds.
    """
    return math.ceil(x * 1e6)


def _make_dict(items):
    """
    Make an OrderedDict out of the provided items, checking for any duplicated
    key.
    """
    # Deduplicate any iterable, even if it contains non-hashable types
    def dedup(xs):
        res = []
        for x in xs:
            if x not in res:
                res.append(x)
        return res

    def check(key, vals):
        vals = dedup(map(itemgetter(1), vals))
        if len(vals) > 1:
            vals = ', '.join(map(str, vals))
            raise KeyError(f'Value for key "{key}" was set multiple times: {vals}')
        else:
            val, = vals
            return (key, val)

    key = itemgetter(0)
    items = list(items)
    order = map(key, items)

    # OrderedDict is important here, even if dict() preserves insertion order
    # for Python >= 3.6. OrderedDict.__eq__ takes order into account unlike
    # dict.__eq__
    return OrderedDict(
        starmap(check, order_as(list(groupby(items, key=key)), order, key=key))
    )


class CalibrationError(RuntimeError):
    """
    Exception raised when the ``rt-app`` calibration is not consistent with the
    CPU capacities in a way or another.
    """


class RTAConf(Loggable, Mapping):
    """
    JSON configuration for rt-app.

    :param conf: Python object graph with the JSON content.
    :type conf: object
    """

    ALLOWED_TASK_NAME_REGEX = r'^[a-zA-Z0-9_]+$'

    def __init__(self, conf):
        self.conf = conf

    def __str__(self):
        return str(self.conf)

    def __getitem__(self, key):
        return self.conf[key]

    def __len__(self):
        return len(self.conf)

    def __iter__(self):
        return iter(self.conf)

    @property
    def json(self):
        """
        rt-app configuration file content as a JSON string.
        """
        return json.dumps(self.conf, indent=4, separators=(',', ': ')) + '\n'

    @staticmethod
    def _process_calibration(plat_info, calibration):
        """
        Select CPU or pload value for task calibration
        """
        # This is done at init time rather than at run time, because the
        # calibration value lives in the file
        if isinstance(calibration, int):
            pass
        elif isinstance(calibration, str):
            calibration = calibration.upper()
        elif calibration is None:
            calib_map = plat_info['rtapp']['calib']
            calibration = min(calib_map.values())
        else:
            raise ValueError(f'Calibration value "{calibration}" is cannot be handled')

        return calibration

    @classmethod
    def from_profile(cls, profile,
        *,
        plat_info,
        force_defaults=False,
        max_duration_s=None,
        calibration=None,
        log_stats=False,
        trace_events=None,
        run_dir=None,
        lock_pages=False,
        no_force_default_keys=None,
    ):
        """
        Create an rt-app workload using :class:`RTAPhase` instances

        :param profile: The workload description in a {task_name : :class:`RTATask`}
          shape
        :type profile: dict

        :param plat_info: Platform information used to tweak the configuration
            file according to the target.
        :type plat_info: lisa.platforms.platinfo.PlatformInfo

        :param force_defaults: If ``True``, default values for all settings
            will be set in the first phase (unless they are set by the profile).
            If ``False``, defaults will be removed from the file (even if they
            were explicitly set by the user).
        :type force_defaults: bool

        :param no_force_default_keys: JSON keys for which no default will be
            forced by ``force_defaults=True``.
        :type no_force_default_keys: list(str) or None

        :param max_duration_s: Maximum duration of the workload. Will be determined
          by the longest running task if not specified.
        :type max_duration_s: int

        :param calibration: The calibration value to be used by rt-app. This can
          be an integer value or a CPU string (e.g. "CPU0").
        :type calibration: int or str

        :param log_stats: Generate a log file with stats for each task
        :type log_stats: bool

        :param lock_pages: Lock the pages to memory to avoid jitter. Requires running as root.
        :type lock_pages: bool

        :param trace_events: A list of trace events to generate.
            For a full list of trace events which can be generated by rt-app,
            refer to the tool documentation:
            https://github.com/scheduler-tools/rt-app/blob/master/doc/tutorial.txt
            By default, no events are generated.
        :type trace_events: list(str)
        """
        logger = cls.get_logger()

        # Sanity check for task names rt-app uses pthread_setname_np(3) which
        # limits the task name to 16 characters including the terminal '\0' and
        # the rt-app suffix.
        max_size = TASK_COMM_MAX_LEN - len('-XX-XXXX')
        too_long_tids = sorted(
            tid for tid in profile.keys()
            if len(tid) > max_size
        )
        if too_long_tids:
            raise ValueError(
                f'Task names too long, please configure your tasks with names shorter than {max_size} characters: {too_long_tids}')

        invalid_tids = sorted(
            tid for tid in profile.keys()
            if not re.match(cls.ALLOWED_TASK_NAME_REGEX, tid)
        )
        if invalid_tids:
            raise ValueError(
                f'Task names not matching "{cls.ALLOWED_TASK_NAME_REGEX}": {invalid_tids}')

        if max_duration_s:
            logger.warning(f'Limiting workload duration to {max_duration_s} [s]')

        calibration = cls._process_calibration(plat_info, calibration)

        def make_phases(task_name, task):
            return task.get_rtapp_repr(
                task_name=task_name,
                plat_info=plat_info,
                force_defaults=force_defaults,
                no_force_default_keys=no_force_default_keys,
            )

        conf = OrderedDict((
            (
                'global',
                {
                    'duration': -1 if not max_duration_s else max_duration_s,
                    'calibration': calibration,
                    'lock_pages': lock_pages,
                    'log_size': 'file' if log_stats else 'disable',
                    'ftrace': ','.join(trace_events or []),
                    'logdir': run_dir or './',
                }
            ),
            (
                'tasks',
                OrderedDict(
                    (task_name, make_phases(task_name, task))
                    for task_name, task in sorted(profile.items(), key=itemgetter(0))
                )
            )

        ))
        return cls(conf)

    @classmethod
    def _process_template(cls,
        template,
        duration=None,
        pload=None,
        log_dir=None,
    ):
        """
        :meta public:

        Turn a raw string rt-app description into a JSON dict.
        Also, process some tokens and replace them.

        :param template: The raw string to process
        :type template: str

        :param duration: The value to replace ``__DURATION__`` with
        :type duration: int

        :param pload: The value to replace ``__PVALUE__`` with
        :type pload: int or str

        :param log_dir: The value to replace ``__LOGDIR__`` and ``__WORKDIR__``
            with.
        :type log_dir: str

        :returns: a JSON dict
        """

        replacements = {
            '__DURATION__': duration,
            '__PVALUE__': pload,
            '__LOGDIR__': log_dir,
            '__WORKDIR__': log_dir,
        }

        json_str = template
        for placeholder, value in replacements.items():
            if placeholder in template and placeholder is None:
                raise ValueError(f'Missing value for {placeholder} placeholder')
            else:
                json_str = json_str.replace(placeholder, json.dumps(value))

        return json.loads(json_str, object_pairs_hook=OrderedDict)

    @classmethod
    def from_path(cls, path, **kwargs):
        """
        Same as :meth:`from_str` but with a file path instead.
        """
        with open(path) as f:
            content = f.read()
        return cls.from_str(content, **kwargs)

    @classmethod
    def from_str(cls, str_conf, plat_info, run_dir,
            max_duration_s=None,
            calibration=None,
        ):
        """
        Create an rt-app workload using a pure string description

        :param str_conf: The raw string description. This must be a valid json
          description, with the exception of some tokens (see
          :meth:`_process_template`) that will be replaced automagically.
        :type str_conf: str

        :param plat_info: Platform information used to tweak the configuration
            file according to the target.
        :type plat_info: lisa.platforms.platinfo.PlatformInfo

        :param run_dir: Directory used by rt-app to produce artifacts
        :type run_dir: str

        :param max_duration_s: Maximum duration of the workload.
        :type max_duration_s: int

        :param calibration: The calibration value to be used by rt-app. This can
          be an integer value or a CPU string (e.g. "CPU0").
        :type calibration: int or str
        """

        calibration = cls._process_calibration(plat_info, calibration)
        conf = cls._process_template(
            template=str_conf,
            duration=max_duration_s,
            pload=calibration,
            log_dir=run_dir,
        )
        return cls(conf)


class RTA(Workload):
    """
    An rt-app workload

    :param json_file: Path to the rt-app json description
    :type json_file: str

    .. warning::
      The class constructor only deals with pre-constructed json files.
      For creating rt-app workloads through other means, see :meth:`from_profile`
      and :meth:`by_str`.

    For more information about rt-app itself, see
    https://github.com/scheduler-tools/rt-app
    """

    REQUIRED_TOOLS = ['rt-app']

    @kwargs_forwarded_to(Workload.__init__, ignore=['command'])
    def _early_init(self, *, log_stats=False, update_cpu_capacities=None, **kwargs):
        """
        Initialize everything that is not related to the contents of the json file
        """
        super().__init__(**kwargs)

        self.log_stats = log_stats
        self.update_cpu_capacities = update_cpu_capacities

        json_file = f'{self.name}.json'
        self.local_json = ArtifactPath.join(self.res_dir, json_file)
        self.remote_json = self.target.path.join(self.run_dir, json_file)
        self._settings['command'] = f'rt-app {quote(self.remote_json)} 2>&1'


    def _late_init(self, conf):
        """
        Complete initialization with a ready json file

        :parameters: Attributes that have been pre-computed and ended up
          in the json file. Passing them can prevent a needless file read.
        """
        self.tasks = sorted(conf['tasks'].keys())
        self.conf = conf
        # Ensure we stay aligned with what folder rt-app will use
        assert self.run_dir == conf['global']['logdir']

    @kwargs_dispatcher(
        (
            _early_init,
            RTAConf.from_path
        ),
        ignore=[
            'plat_info',
            'path',
        ]
    )
    # The only reason we keep positional parameters is for backward
    # compatibility, so that __init__ can be called with positional parameters
    # as it used to.
    def __init__(self, target, name=None, res_dir=None, json_file=None, _early_init_kwargs=None, from_path_kwargs=None):
        # Don't add code here, use the early/late init methods instead.
        # This lets us factorize some code for the class methods that serve as
        # alternate constructors.
        self._early_init(**_early_init_kwargs)

        from_path_kwargs.update(
            path=json_file,
            plat_info=target.plat_info,
            run_dir=self.run_dir,
        )
        conf = RTAConf.from_path(**from_path_kwargs)
        self._late_init(conf=conf)

    def __str__(self):
        return self.conf.json

    @PartialInit.factory
    @classmethod
    @kwargs_dispatcher(
        (
            _early_init,
            RTAConf.from_profile
        ),
        ignore=[
            'plat_info',
            'lock_pages',
        ]
    )
    def from_profile(cls, target, profile, name=None, res_dir=None, log_stats=False, *, as_root=False, _early_init_kwargs=None, from_profile_kwargs=None):
        """
        Create an rt-app workload using :class:`RTATask` instances

        :param target: Target that the workload will run on.
        :type target: lisa.target.Target

        :param name: Name of the workload.
        :type name: str or None

        :param res_dir: Host folder to store artifacts in.
        :type res_dir: str or None

        :Variable keyword arguments: Forwarded to :meth:`RTAConf.from_profile`
        """
        logger = cls.get_logger()
        self = cls.__new__(cls)

        self._early_init(**_early_init_kwargs)

        from_profile_kwargs.update(
            plat_info=target.plat_info,
            run_dir=self.run_dir,
            lock_pages=as_root,
        )
        conf = RTAConf.from_profile(**from_profile_kwargs)
        self._late_init(conf=conf)
        return self

    @deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=from_profile)
    @classmethod
    def by_profile(cls, *args, **kwargs):
        return cls.from_profile(*args, **kwargs)

    @PartialInit.factory
    @classmethod
    @kwargs_dispatcher(
        (
            _early_init,
            RTAConf.from_str
        ),
        ignore=[
            'plat_info',
        ]
    )
    def from_str(cls, target, str_conf,
        name=None,
        res_dir=None,
        _early_init_kwargs=None,
        from_str_kwargs=None,
    ):
        """
        Create an rt-app workload using a pure string description

        :param target: Target that the workload will run on.
        :type target: lisa.target.Target

        :param str_conf: The raw string description. This must be a valid json
          description, with the exception of some tokens (see
          :meth:`RTAConf.from_str`) that will be replaced automagically.
        :type str_conf: str

        :param name: Name of the workload.
        :type name: str or None

        :param res_dir: Host folder to store artifacts in.
        :type res_dir: str or None

        :Variable keyword arguments: Forwarded to :meth:`RTAConf.from_profile`
        """
        self = cls.__new__(cls)
        self._early_init(**_early_init_kwargs)

        from_str_kwargs.update(
            plat_info=target.plat_info,
            run_dir=self.run_dir,
        )
        conf = RTAConf.from_str(**from_str_kwargs)
        self._late_init(conf=conf)
        return self

    @deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=from_str)
    @classmethod
    def by_str(cls, *args, **kwargs):
        return cls.from_profile(*args, **kwargs)

    @PartialInit.factory
    @classmethod
    @kwargs_forwarded_to(_early_init)
    def from_conf(cls, target, conf,
        name=None,
        res_dir=None,
        **kwargs
    ):
        """
        Create an rt-app workload using a :class:`RTAConf`.

        :param target: Target that the workload will run on.
        :type target: lisa.target.Target

        :param conf: Configuration object.
        :type conf: RTAConf

        :param name: Name of the workload.
        :type name: str or None

        :param res_dir: Host folder to store artifacts in.
        :type res_dir: str or None
        """
        logdir = conf.get('global', {}).get('logdir')

        kwargs.update(
            target=target,
            name=name,
            res_dir=res_dir
        )

        self = cls.__new__(cls)
        self._early_init(**kwargs)
        self._late_init(conf=conf)
        return self

    @contextlib.contextmanager
    def _setup(self):
        logger = self.logger
        plat_info = self.target.plat_info
        writeable_capacities = plat_info['cpu-capacities']['writeable']
        update_cpu_capacities = self.update_cpu_capacities
        target = self.target

        with super()._setup():
            # Generate JSON configuration on local file
            with open(self.local_json, 'w') as f:
                f.write(self.conf.json)

            # Move configuration file to target
            target.push(self.local_json, self.remote_json)

            # Pre-hit the calibration information, in case this is a lazy value.
            # This avoids polluting the trace and the dmesg output with the
            # calibration tasks. Since we know that rt-app will always need it for
            # anything useful, it's reasonable to do it here.
            try:
                plat_info['rtapp']['calib']
            # We will get this exception if we are currently trying to compute the calibration
            except (DeferredValueComputationError, RecursionError):
                pass

            if update_cpu_capacities:
                if not writeable_capacities:
                    raise ValueError('CPU capacities are not writeable on this target, please use update_cpu_capacities=False or None')
            # If left to None, we update if possible
            elif update_cpu_capacities is None:
                update_cpu_capacities = writeable_capacities
                if not writeable_capacities:
                    logger.warning('CPU capacities will not be updated on this platform')

            if update_cpu_capacities:
                rtapp_capacities = plat_info['cpu-capacities']['rtapp']
                logger.info(f'Will update CPU capacities in sysfs: {rtapp_capacities}')

                write_kwargs = [
                    dict(
                        path=f'/sys/devices/system/cpu/cpu{cpu}/cpu_capacity',
                        value=capa,
                        verify=True,
                    )
                    for cpu, capa in sorted(rtapp_capacities.items())
                ]
                capa_cm = target.batch_revertable_write_value(write_kwargs)
            else:
                # There might not be any rtapp calibration available, specifically
                # when we are being called to run the calibration workload.
                try:
                    rtapp_capacities = plat_info['cpu-capacities']['rtapp']
                    orig_capacities = plat_info['cpu-capacities']['orig']
                except KeyError:
                    pass
                else:
                    # Spit out some warning in case we are not going to update the
                    # capacities, so we know what to expect
                    self.warn_capacities_mismatch(orig_capacities, rtapp_capacities)

                capa_cm = nullcontext()

            try:
                with capa_cm:
                    yield
            finally:
                target.remove(self.remote_json)

                if self.log_stats:
                    logger.debug(f'Pulling logfiles to: {self.res_dir}')
                    for task in self.tasks:
                        # RT-app appends some number to the logs, so we can't predict the
                        # exact filename
                        logfile = target.path.join(self.run_dir, f'*{task}*.log')
                        target.pull(logfile, self.res_dir, globbing=True)

    def _run(self):
        out = yield self._basic_run()

        # Extract calibration information from stdout

        # Match lines like this one:
        # [rt-app] <notice> pLoad = 643ns : calib_cpu 5
        pload_regex = re.compile(rb'pLoad\s*=\s*(?P<pload>[0-9]+).*calib_cpu\s(?P<cpu>[0-9]+)')

        def parse(line):
            m = pload_regex.search(line)
            if m:
                return (int(m.group('cpu')), int(m.group('pload')))
            else:
                return None

        pload = dict(filter(bool, map(parse, out['stdout'].splitlines())))

        return {'calib': pload}

    _ONGOING_CALIBRATION = weakref.WeakKeyDictionary()
    @classmethod
    def _calibrate(cls, target, res_dir):
        if target in cls._ONGOING_CALIBRATION:
            raise RecursionError('Trying to calibrate rt-app while calibrating rt-app')
        else:
            try:
                cls._ONGOING_CALIBRATION[target] = True
                return cls._do_calibrate(target, res_dir)
            finally:
                cls._ONGOING_CALIBRATION.pop(target, None)

    @classmethod
    def _do_calibrate(cls, target, res_dir):
        res_dir = res_dir if res_dir else target .get_res_dir(
            "rta_calib", symlink=False
        )
        logger = cls.get_logger()

        # Create calibration task
        if target.is_rooted:
            try:
                max_rtprio = int(target.execute('ulimit -Hr').splitlines()[0])
            # Some variants of ulimit (which is a shell builtin) will not
            # accept -r, notably on Ubuntu 20.04:
            # https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=975326
            except TargetStableError as e:
                out = target.execute('ulimit -a')
                for line in out.splitlines():
                    m = re.search(r'rtprio *([0-9]+)', line)
                    if m:
                        max_rtprio = int(m.group(1))
                        break
                # If we could not find anything, re-raise the initial exception
                else:
                    raise e

            logger.debug(f'Max RT prio: {max_rtprio}')

            priority = max_rtprio + 1 if max_rtprio <= 10 else 10
            sched_policy = 'SCHED_FIFO'
        else:
            logger.warning('Will use default scheduler class instead of RT since the target is not rooted')
            priority = None
            sched_policy = None

        pload = {}
        for cpu in target.list_online_cpus():
            logger.debug(f'Starting CPU{cpu} calibration...')

            # RT-app will run a calibration for us, so we just need to
            # run a dummy task and read the output
            calib_task = RTAPhase(
                prop_wload=PeriodicWload(
                    duty_cycle_pct=100,
                    duration=0.001,
                    period=1e-3,
                ),
                prop_priority=priority,
                prop_policy=sched_policy,
            )
            rta = cls.from_profile(target,
                name=f"rta_calib_cpu{cpu}",
                profile={'task1': calib_task},
                calibration=f"CPU{cpu}",
                res_dir=os.path.join(res_dir, f'CPU{cpu}'),
                as_root=target.is_rooted,
                # Disable CPU capacities update, since that leads to infinite
                # recursion
                update_cpu_capacities=False,
            )

            with rta, target.freeze_userspace():
                output = rta.run()
            calib = output['calib']

            logger.info(f'CPU{cpu} calibration={calib[cpu]}')
            pload.update(calib)

        # Avoid circular import issue
        from lisa.platforms.platinfo import PlatformInfo
        snippet_plat_info = PlatformInfo({
            'rtapp': {
                'calib': pload,
            },
        })
        logger.info(f'Platform info rt-app calibration configuration:\n{snippet_plat_info.to_yaml_map_str()}')

        zero_pload = sorted(
            cpu
            for cpu, load in pload.items()
            if load == 0
        )
        if zero_pload:
            raise ValueError(f'The pload for the following CPUs is 0, which means the CPU is infinitely fast: {zero_pload}')

        plat_info = target.plat_info

        # Sanity check calibration values for asymmetric systems if we have
        # access to capacities
        try:
            orig_capacities = plat_info['cpu-capacities']['orig']
        except KeyError:
            return pload
        else:
            capa_ploads = {
                capacity: {cpu: pload[cpu] for cpu in cpus}
                for capacity, cpus in group_by_value(orig_capacities).items()
            }

            # Find the min pload per capacity level, i.e. the fastest detected CPU.
            # It is more likely to represent the right pload, as it has suffered
            # from less IRQ slowdown or similar disturbances that might be random.
            capa_pload = {
                capacity: min(ploads.values())
                for capacity, ploads in capa_ploads.items()
            }

            # Sort by capacity
            capa_pload_list = sorted(capa_pload.items())
            # unzip the list of tuples
            _, pload_list = zip(*capa_pload_list)

            # If sorting according to capa was not equivalent to reverse sorting
            # according to pload (small pload=fast cpu)
            if list(pload_list) != sorted(pload_list, reverse=True):
                raise CalibrationError('Calibration values reports big cores less capable than LITTLE cores')

            # Check that the CPU capacities seen by rt-app are similar to the one
            # the kernel uses
            orig_capacities = plat_info['cpu-capacities']['orig']
            true_capacities = cls.get_cpu_capacities_from_calibrations(orig_capacities, pload)
            cls.warn_capacities_mismatch(orig_capacities, true_capacities)

            return pload

    @classmethod
    def warn_capacities_mismatch(cls, orig_capacities, new_capacities):
        """
        Compare ``orig_capacities`` and ``new_capacities`` and log warnings if
        they are not consistent.

        :param orig_capacities: Original CPU capacities, as a map of CPU to capacity.
        :type orig_capacities: dict(int, int)

        :param new_capacities: New CPU capacities, as a map of CPU to capacity.
        :type new_capacities: dict(int, int)
        """
        logger = cls.get_logger()
        capacities = {
            cpu: (orig_capacities[cpu], new_capacities[cpu])
            for cpu in orig_capacities.keys() & new_capacities.keys()
        }
        logger.info(f'CPU capacities according to rt-app workload: {new_capacities}')

        capa_factors_pct = {
            cpu: new / orig * 100
            for cpu, (orig, new) in capacities.items()
        }
        dispersion_pct = max(abs(100 - factor) for factor in capa_factors_pct.values())

        if dispersion_pct > 2:
            logger.warning(f'The calibration values are not inversely proportional to the CPU capacities, the duty cycles will be up to {dispersion_pct:.2f}% off on some CPUs: {capa_factors_pct}')

        if dispersion_pct > 20:
            logger.warning(f'The calibration values are not inversely proportional to the CPU capacities. Either rt-app calibration failed, or the rt-app busy loops has a very different instruction mix compared to the workload used to establish the CPU capacities: {capa_factors_pct}')

        # Map of CPUs X to list of CPUs Ys that are faster than it although CPUs
        # of Ys have a smaller orig capacity than X
        if len(capacities) > 1:
            faster_than_map = {
                cpu1: sorted(
                    cpu2
                    for cpu2, (orig2, new2) in capacities.items()
                    if new2 > new1 and orig2 < orig1
                )
                for cpu1, (orig1, new1) in capacities.items()
            }
        else:
            faster_than_map = {}

        # Remove empty lists
        faster_than_map = {
            cpu: faster_cpus
            for cpu, faster_cpus in faster_than_map.items()
            if faster_cpus
        }

        if faster_than_map:
            raise CalibrationError(f'Some CPUs of higher capacities are slower than other CPUs of smaller capacities: {faster_than_map}')

    @classmethod
    def get_cpu_capacities_from_calibrations(cls, orig_capacities, calibrations):
        """
        Compute the CPU capacities out of the rt-app calibration values.

        :returns: A mapping of CPU to capacity.

        :param orig_capacities: Original capacities as a mapping of CPU ID to
            capacity.
        :type orig_capacities: dict(int, int)

        :param calibrations: Mapping of CPU to pload value.
        :type calibrations: dict
        """

        # calibration values are inversely proportional to the CPU capacities
        inverse_calib = {cpu: 1 / calib for cpu, calib in calibrations.items()}

        def compute_capa(cpu):
            # True CPU capacity for the rt-app workload, rather than for the
            # whatever workload was used to compute the CPU capacities exposed by
            # the kernel
            return inverse_calib[cpu] / max(inverse_calib.values()) * PELT_SCALE

        rtapp_capacities =  {cpu: compute_capa(cpu) for cpu in calibrations.keys()}

        # Average in a capacity class, since the kernel will only use one
        # value for the whole class anyway
        new_capacities = {}
        # Group the CPUs by original capacity
        for capa, capa_class in group_by_value(orig_capacities).items():
            avg_capa = mean(
                capa
                for cpu, capa in rtapp_capacities.items()
                if cpu in capa_class
            )
            new_capacities.update({cpu: avg_capa for cpu in capa_class})

        # Make sure that the max cap is 1024 and that we use integer values
        new_max_cap = max(new_capacities.values())
        new_capacities = {
            # Make sure the max cap will be 1024 and not 1023 due to rounding
            # errors
            cpu: math.ceil(capa / new_max_cap * 1024)
            for cpu, capa in new_capacities.items()
        }
        return new_capacities


    @classmethod
    def get_cpu_calibrations(cls, target, res_dir=None):
        """
        Get the rt-ap calibration value for all CPUs.

        :param target: Target to run calibration on.
        :type target: lisa.target.Target

        :returns: Dict mapping CPU numbers to rt-app calibration values.
        """

        if not target.is_module_available('cpufreq'):
            cls.get_logger().warning(
                'cpufreq module not loaded, skipping setting frequency to max')
            cm = nullcontext()
        else:
            cm = target.cpufreq.use_governor('performance')

        with cm, target.disable_idle_states():
            return cls._calibrate(target, res_dir)

    @classmethod
    def _compute_task_map(cls, trace, names):
        prefix_regexps = {
            prefix: re.compile(rf"^{re.escape(prefix)}(-[0-9]+)*$")
            for prefix in names
        }

        task_map = {
            prefix: sorted(
                task_id
                for task_id in trace.task_ids
                if re.match(regexp, task_id.comm)
            )
            for prefix, regexp in prefix_regexps.items()
        }

        missing = sorted(prefix for prefix, task_ids in task_map.items() if not task_ids)
        if missing:
            raise RuntimeError(f"Missing tasks matching the following rt-app profile names: {', '.join(missing)}")
        return task_map

    # Mapping of Trace objects to their task map.
    # We don't want to keep traces alive just for this cache, so we use weak
    # references for the keys.
    _traces_task_map = weakref.WeakKeyDictionary()

    @classmethod
    def resolve_trace_task_names(cls, trace, names):
        """
        Translate an RTA profile task name to a list of
        :class:`lisa.trace.TaskID` as found in a :class:`lisa.trace.Trace`.

        :returns: A dictionnary of ``rt-app`` profile names to list of
            :class:`lisa.trace.TaskID` The list will contain more than one item
            if the task forked.

        :param trace: Trace to look at.
        :type trace: lisa.trace.Trace

        :param names: ``rt-app`` task names as specified in profile keys
        :type names: list(str)
        """

        task_map = cls._traces_task_map.setdefault(trace, {})
        # Update with the names that have not been discovered yet
        not_computed_yet = set(names) - task_map.keys()
        if not_computed_yet:
            task_map.update(cls._compute_task_map(trace, not_computed_yet))

        # Only return what we were asked for, so the client code does not
        # accidentally starts depending on whatever was requested in earlier
        # calls
        return {
            name: task_ids
            for name, task_ids in task_map.items()
            if name in names
        }

    def get_trace_task_names(self, trace):
        """
        Get a dictionnary of :class:`lisa.trace.TaskID` used in the given trace
        for this task.
        """
        return self.resolve_trace_task_names(trace, self.tasks)


class PropertyMeta(abc.ABCMeta):
    """
    Metaclass for properties.

    It overrides ``__instancecheck__`` so that instances of
    :class:`PropertyWrapper` can be recognized as instances of the type they
    wrap, in order to make them as transparent as possible.
    """
    def __instancecheck__(cls, obj):
        # If we have a PropertyWrapper object, treat it as if it was an
        # instance of the wrapped object. This is mostly safe since
        # PropertWrapper implements __getattr__ and allows uniform handling of
        # ConcretePropertyBase and MetaPropertyBase, even for wrapped values

        # Scary super() call: we want to get the __instancecheck__
        # implementation of our base class (abc.ABCMeta), but we want to call
        # it on PropertyWrapper. If we don't do that,
        # PropertyWrapper.__instancecheck__ is the current function and we have
        # an infinite recursion.
        if super(PropertyMeta, cls).__instancecheck__.__func__(PropertyWrapper, obj):
            # Still do the regular check, so that we can detect instances of
            # PropertyWrapper
            is_wrapper_subcls = super().__instancecheck__(obj)
            obj = obj.__wrapped__
        else:
            is_wrapper_subcls = False

        return is_wrapper_subcls or super().__instancecheck__(obj)


class PropertyBase(SimpleHash, metaclass=PropertyMeta):
    """
    Base class of all properties.
    """

    KEY = None
    """
    Subclasses can override this attribute so that
    :meth:`PropertyBase.from_key` knows that it can call their
    :meth:`~PropertyBase._from_key` method for that key.

    .. note:: This class attribute will not be inherited automatically so that
        each class can be uniquely identified by its key. Subclass that do not
        override the value explicitly will get ``None``.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        dct = cls.__dict__

        # Ensure KEY class attribute is not inherited
        if 'KEY' not in dct:
            cls.KEY = None

        try:
            and_ = dct['__and__']
        except KeyError:
            pass
        else:
            @functools.wraps(and_)
            def wrapper(self, other):
                # If allow the right operand to define the operation if it's a
                # contaminating one
                if isinstance(other, ContaminatingProperty):
                    return other.__rand__(self)
                else:
                    return and_(self, other)

            cls.__and__ = wrapper

        return super().__init_subclass__(**kwargs)


    @property
    def key(self):
        """
        Key of the instance.

        This property will default to getting the value of the :attr:`KEY`
        class attribute of the first of its ancestor defining it. This way,
        :meth:`_from_key` can be shared in a base class between multiple
        subclasses if needed.
        """
        try:
            return self.__dict__['key']
        except KeyError:
            # Look for the KEY in base classes, since it could be managing the
            # creation of its subclasses in _from_keys(). Therefore, it is
            # relevant to "pretend" that we have the same key. At the same
            # time, KEY cannot be inherited in order to have a 1-1 mapping
            # between keys and classes when looking for the right
            # implementation of _from_keys().
            for base in inspect.getmro(self.__class__):
                try:
                    key = base.KEY
                except AttributeError:
                    continue
                else:
                    if key is not None:
                        return key
                    else:
                        continue

            raise AttributeError(f'No "key" attribute on {self.__class__}')

    @key.setter
    def key(self, val):
        if val is None:
            try:
                del self.__dict__['key']
            except KeyError:
                pass
        else:
            self.__dict__['key'] = val

    @property
    @abc.abstractmethod
    def val(self):
        """
        Value "payload" of the property.

        Ideally, it should be a valid value that can be given to
        :meth:`~PropertyBase.from_key`, but it's not mandatory. For complex
        properties that are not isomorphic to a Python basic type (int, tuple
        etc.), ``self`` should be returned.
        """

    @classmethod
    def _from_key(cls, key, val):
        """
        :meta public:

        Build an instance out of ``key`` and ``val``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __and__(self, other):
        """
        Combine two instances of the same property together.

        This is used to combine properties at the various levels of the
        :class:`RTAPhaseTree` tree, on each path from the root to the
        leaves. It is guaranteed that the instance closer to the root will be
        ``self`` and the one closer to the leaves will be ``other``.

        If the property is a constraint, a good implementation should combine
        two instances by applying the strongest constraint. For example, CPU
        affinity are combined by taking the intersection of allowed CPUs.

        If the property is some sort of "dummy" structure, it can make sense to
        allow the instance closer to the root of the tree to override the set
        members in the instance closer to the leaves.

        Otherwise, it's probably best to just bail with :exc:`ValueError` with
        a message explaining that there is a conflict.
        """

    @classmethod
    def _check_key(cls, key):
        """
        :meta public:

        Check that the ``key`` is allowed for this class.
        """
        if cls.KEY is not None and key != cls.KEY:
            raise ValueError(f'Using wrong key name "{key}" to build {cls.__qualname__}')

    @classmethod
    def from_key(cls, key, val):
        """
        Alternative constructor that is available with the same signature for
        all properties.

        :param key: Key passed by the user. It will be checked with
            :meth:`~PropertyBase._check_key` before building an instance.
        :type key: str

        :param val: Value passed by the user.
        :type val: object
        """
        subcls = cls.find_cls(key)
        subcls._check_key(key)
        return subcls._from_key(key, val)

    @classmethod
    def find_cls(cls, key):
        """
        Find which subclass can handle ``key``.

        It is best called on :class:`PropertyBase` in order to allow any
        property to be built.
        """
        subclasses = [
            subcls
            for subcls in chain([cls], get_subclasses(cls))
            if subcls.KEY == key
        ]
        if subclasses:
            assert len(subclasses) == 1
            return subclasses[0]
        else:
            raise ValueError(f'Property "{key}" not handled')


    @classmethod
    def _get_cls_doc(cls):
        def type_of_param(name, x):
            doc = inspect.getdoc(x) or ''
            m = re.search(rf':type {name}:(.*)', doc)
            if m:
                return m.group(1).strip()
            else:
                return None

        doc = get_short_doc(cls)
        type_ = ''

        # Read the format accepted by _from_key and use it here
        from_key_doc = inspect.getdoc(cls._from_key) or ''
        sig = inspect.signature(cls)

        parsed_type = None

        from_from_key = type_of_param('val', cls._from_key)
        if from_from_key:
            parsed_type = from_from_key
        # Constructor with only one parameter in addition to "self"
        elif len(sig.parameters) == 1:
            param, = sig.parameters.values()
            from__init__ = type_of_param(param.name, cls)
            if from__init__:
                parsed_type = from__init__

        if parsed_type:
            type_ += f'{parsed_type} or '

        type_ += get_cls_name(cls)

        return (doc, type_)


class ConcretePropertyBase(PropertyBase, Loggable):
    """
    Base class for concrete properties.

    Concrete properties are properties that will ultimately translated into
    JSON, as opposed to meta properties that will never make it to the final
    configuration file.
    """
    OPTIMIZE_JSON_KEYS = {}
    """
    Configuration driving the JSON optimization, as a ``dict(str, set(str))``.

    This is a dictionary mapping JSON key names to set of "barrier" JSON keys.
    When successive phases of a given task share the same value for the keys of
    that dictionary, they will be removed in the later phases since rt-app
    settings are persistent across phases. When any of the barrier key listed in the
    set has a change in its value, it will be considered as an optimization
    barrier and the value will be set again, even if it means repeating the
    same value as earlier.
    """
    REQUIRED_KCONFIG_KEYS = []
    """
    List of KCONFIG keys that need to be =Y on the target kernel for this
    property to be usable.
    """


    @classmethod
    def check_kconfig(cls, plat_info, strict=True):
        """
        Check whether ``plat_info`` contains the kernel KCONFIG keys contained
        in :attr:`~ConcretePropertyBase.REQUIRED_KCONFIG_KEYS`.

        :param keys: Kernel config keys to check, e.g. ['CONFIG_FOO_BAR'].
        :type keys: list(str)

        :param strict: If True, raise an exception if any key is missing.
                       If False, log if any key is missing.
        :type strict: bool
        """
        def raise_err(msg, exc):
            if strict:
                raise exc
            else:
                cls.get_logger().debug(msg)
            return False

        for key in cls.REQUIRED_KCONFIG_KEYS:
            try:
                val = plat_info['kernel']['config'][key]
            except KeyError as e:
                return raise_err(f'Kernel config does not have key: {key}', e)

            # Dependency could be built as a module, in which case we'd
            # need to check whether it is loaded. Assert against Y for
            # now.
            if val is not KernelConfigTristate.YES:
                msg = f'Kernel config {key}={val}, expected Y'
                return raise_err(msg, TargetStableError(msg))
        return True

    def to_json(self, plat_info, **kwargs):
        """
        Snippet of JSON content for that property as Python objects.

        :param plat_info: Platform information that can be used to generate the
            default value .
        :type plat_info: lisa.platforms.platinfo.PlatformInfo
        """
        # Raising an exception at this point is *mandatory*, otherwise the value
        # of the property in following phases is basically undefined.
        self.check_kconfig(plat_info, True)

        kwargs['plat_info'] = plat_info
        return self._to_json(**kwargs)

    @abc.abstractmethod
    def _to_json(self, plat_info):
        pass

    @classmethod
    def to_default_json(cls, plat_info, properties):
        """
        Similar to :meth:`~ConcretePropertyBase.to_json` but returns the
        default values for the keys set in
        :meth:`~ConcretePropertyBase.to_json`.

        :param plat_info: Platform information that can be used to generate the
            default value .
        :type plat_info: lisa.platforms.platinfo.PlatformInfo

        :param properties: :class:`collections.OrderedDict` of JSON properties for the
            current phase. This can be used if the default value is context
            dependent. For example, if two properties depend on each other,
            they can get the value of the other key from ``properties``. The
            property might not have been set yet so abscence of the key has to
            be handled. The calling code will look for a fixpoint for the
            default properties, so this method will be called iteratively until
            the result is stable, allowing for arbitrary dependency between
            keys.
        :type properties: collections.OrderedDict(str, object)
        """
        if not cls.check_kconfig(plat_info, False):
            return {}
        return cls._to_default_json(plat_info, properties)

    @classmethod
    @abc.abstractmethod
    def _to_default_json(cls, plat_info, properties):
        # Implementation still matters for classmethods, as being an ABC with
        # unimplemented abstractmethod only prevents instances from being
        # created. Classmethods can still be called on the class and are
        # inherited the usual way.
        return {}


class MetaPropertyBase(PropertyBase):
    """
    Base class for meta properties.

    Meta properties are properties that will not be translated into JSON, as
    opposed to concrete properties.
    """


class ContaminatingProperty(PropertyBase):
    """
    Base class for properties that will "contaminate" other instances when
    combined with them, even if they are closer to the leaf.
    """
    @classmethod
    @abc.abstractmethod
    def __rand__(self, other):
        pass


class PropertyWrapper(ContaminatingProperty):
    """
    Base class for properties that are merely wrapper around another property
    instance.

    It is advised that subclasses use name mangling for attributes (name
    starting with ``__``), so that the wrapper's attribute will not conflict
    with the attributes of the wrapped property, so that the wrapper is as
    transparent as possible.
    """
    def __init__(self, prop):
        self.__wrapped__ = prop

    def __eq__(self, other):
        return self.__wrapped__ == other

    def __hash__(self):
        return hash(self.__wrapped__)

    def __rand__(self, other):
        return self.with_wrapped(
            other & self.__wrapped__
        )

    def with_wrapped(self, wrapped):
        """
        Build a new instance with modified wrapped property.
        """
        new = copy.copy(self)
        new.__wrapped__ = wrapped
        return new

    @classmethod
    def from_key(cls, key, val, **kwargs):
        # Explicit reference to PropertyBase instead of super(), otherwise the
        # "cls" parameter will be (wrongly in our case) forwarded
        prop = PropertyBase.from_key(key, val)
        return cls(prop=prop, **kwargs)

    @property
    def val(self):
        # To satisfy the ABC even though __getattr__ would take care of this
        return self.__wrapped__.val

    def __copy__(self):
        # Necessary because of __getattr__, otherwise we get some infinite
        # recursion
        new = self.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def __getattr__(self, attr):
        """
        Be as transparent as possible, so that this sort of call would work:
        ``self.__prop.__and__(self)``
        """
        return getattr(self.__wrapped__, attr)


class PlaceHolderValue:
    """
    Placeholder value to be passed to
    :meth:`RTAPhaseProperties.from_polymorphic` to hijack the usual
    :meth:`PropertyBase.from_key` path and allow ``PROPERTY_CLS.from_key`` to
    be called instead.

    :param val: "payload" value.
    :type val: object

    :Variable keyword arguments: Saved for later consumption.
    """
    PROPERTY_CLS = None
    """
    Class on which ``from_key`` will be used, rather than looking up the class
    based on the key name.
    """

    def __init__(self, val=None, **kwargs):
        self.val = val
        self.kwargs = kwargs

    def __str__(self):
        return str(self.val)


class OverridenProperty(PropertyWrapper):
    """
    Forcefully override the value of another property.

    .. seealso:: :func:`override`
    """
    def __init__(self, prop, deep=True):
        super().__init__(prop)
        # Use mangled names (starting with "__") to minimize conflicts with the
        # wrapped prop, so that __getattr__ is as transparent as possible
        self.__deep = deep

    def __and__(self, other):
        """
        We only want to override properties "downards", which means:

            * ``OverridenProperty(root) & leaf = OverridenProperty(root)``
            * ``root & OverridenProperty(leaf) = OverridenProperty(root & leaf)``

        .. note:: When ``deep=False`` is used, this class does not form a
            semigroup, i.e. the result will depend on the order of the tree
            traversal.
        """
        prop = self if self.__deep else self.__wrapped__
        return copy.copy(prop)


class LeafPrecedenceProperty(PropertyWrapper):
    """
    Invert the usual combination of order (root to leaf) of property values.

    .. seealso:: :func:`leaf_precedence`
    """
    def __and__(self, other):
        # Return __wrapped__ so the effect does not propagate down the tree
        # more than one level.
        return other.__and__(self).__wrapped__


class _LeafPrecedenceValue(PlaceHolderValue):
    """
    Placeholder value for :class:`LeafPrecedenceProperty`.
    """
    PROPERTY_CLS = LeafPrecedenceProperty


def leaf_precedence(val, **kwargs):
    """
    Give precedence to the leaf values when combining with ``&``::

        phase = phase.with_props(meta=({'hello': 'leaf'})
        phase = phase.with_props(meta=leaf_precedence({'hello': 'root'})
        assert phase['meta'] == {'hello': 'leaf'}

    This allows setting a property with some kind of default value on a root
    node in the phase tree while still allowing it to be combined with the
    children values in such a way that the children values take precedence.

    .. note:: In order to make the result more predictable, the effect will not
        propagate beyond one level where the property is set. This means it
        will work best for properties that are only set in leaves only.
    """
    return _LeafPrecedenceValue(val, **kwargs)


class _OverridingValue(PlaceHolderValue):
    """
    Placeholder value for :class:`OverridenProperty`.
    """
    PROPERTY_CLS = OverridenProperty


def override(val, **kwargs):
    """
    Override a property with the given value, rather than combining it with the
    property-specific ``&`` implementation::

        phase = phase.with_props(cpus=override({1,2}))
    """
    return _OverridingValue(val, **kwargs)


class DeletedProperty(ContaminatingProperty):
    """
    Forcefully delete the given property, recursively for all subtrees.

    :param key: Key of the property to delete.
    :type key: str

    .. seealso:: :func:`delete`

    .. note:: The property is not actually deleted but just replaced by an
        instance of this class, which will have specific handling in relevant
        parts of the code.
    """
    def __init__(self, key):
        self.key = key

    def __bool__(self):
        return False

    @property
    def val(self):
        return None

    def _to_json(self, **kwargs):
        return {}

    @classmethod
    def from_key(cls, key, val):
        return cls(key=key)

    def __and__(self, other):
        return copy.copy(self)

    def __rand__(self, other):
        # After a property has been deleted, we can set it again to any value
        # we want.
        return copy.copy(other)


class _DeletingValue(PlaceHolderValue):
    """
    Placeholder value for :class:`DeletedProperty`.
    """
    PROPERTY_CLS = DeletedProperty

def delete():
    """
    Remove the given property from the phase::

        phase = phase.with_props(cpus=delete())
    """
    return _DeletingValue()


class SimpleProperty(PropertyBase):
    """
    Simple property with dynamic ``key``.
    """
    def __init__(self, key, val):
        self.key = key
        self.val = val

    def __str__(self):
        if self.key == self.KEY:
            key = ''
        else:
            key = f'{self.key}='

        return f'{key}{self.val}'

    @property
    def val(self):
        return self.__dict__['val']

    @val.setter
    def val(self, val):
        self.__dict__['val'] = val

    @classmethod
    def _from_key(cls, key, val):
        return cls(key, val)

    def __and__(self, other):
        if self.val != other.val:
            raise ValueError(f'Conflicting values for key "{self.key}": "{self.val}" and "{other.val}"')
        else:
            return self.__class__(key=self.key, val=self.val)


class SimpleConcreteProperty(SimpleProperty, ConcretePropertyBase):
    """
    Base class for simple properties that maps to JSON.
    """

    JSON_KEY = None
    """
    Name of the JSON key the property will set.

    .. note:: If it needs to be dynamically chose, see
        :attr:`~SimpleConcreteProperty.json_key`.
    """

    DEFAULT_JSON = None
    """
    JSON value to use as a default.

    If ``None``, nothing will be output.

    .. note:: If the default value is context-dependent, it should override
        :meth:`~ConcretePropertyBase.to_default_json` instead.
    """

    FILTER_NONE = True
    """
    If ``True``, no JSON content will be generated when the property value is
    ``None``
    """

    @property
    def json_key(self):
        """
        Name of the JSON key that will be set.

        Defaults to :attr:`JSON_KEY` if it is not ``None``, otherwise ``key``
        instance attribute will be used.
        """
        return self.JSON_KEY or self.key

    def _to_json(self, plat_info):
        val = self.val
        if val is None and self.FILTER_NONE:
            return {}
        else:
            return {self.json_key: val}

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        default = cls.DEFAULT_JSON
        if default is None:
            return {}
        else:
            key = cls.JSON_KEY or cls.KEY
            return {key: default}


class _SemigroupProperty(PropertyBase):
    """
    :meta public:

    Base class for properties forming a semigroup with respect to their
    ``__and__`` method.

    This implies ``__and__`` is associative, i.e. this must hold:
    ``(a & b) & c == a & (b & c)``
    """
    @staticmethod
    @abc.abstractmethod
    def _SEMIGROUP_OP(x, y):
        """
        :meta public:

        Function used to combine two non-None values.
        """

    def __and__(self, other):
        """
        Combine values of the properties using
        :meth:`~_SemigroupProperty._SEMIGROUP_OP`, except when one of the value
        is ``None``, in which case the other value is used as is and wrapped
        into an instance using :meth:`~PropertyBase.from_key`.
        """
        if (self.val, other.val) == (None, None):
            val = None
        elif self.val is None:
            val = other.val
        elif other.val is None:
            val = self.val
        else:
            val = self._SEMIGROUP_OP(self.val, other.val)

        return self.__class__.from_key(key=self.key, val=val)


class MinProperty(_SemigroupProperty):
    """
    Semigroup property with the :func:`min` operation.
    """
    _SEMIGROUP_OP = min


class MaxProperty(_SemigroupProperty):
    """
    Semigroup property with the :func:`max` operation.
    """
    _SEMIGROUP_OP = max


class AndProperty(_SemigroupProperty):
    """
    Semigroup property with the ``&`` operation.
    """
    _SEMIGROUP_OP = operator.and_


class OrProperty(_SemigroupProperty):
    """
    Semigroup property with the ``|`` operation.
    """
    _SEMIGROUP_OP = operator.or_


class NameProperty(SimpleProperty, MetaPropertyBase):
    """
    Name the phase.
    """
    KEY = 'name'
    SEPARATOR = '/'

    def __init__(self, name, _from_merge=False):
        sep = self.SEPARATOR
        if not _from_merge and sep in name:
            raise ValueError(f'"{sep}" not allowed in phase name "{name}"')
        super().__init__(key=self.KEY, val=name)

    @classmethod
    def _from_key(cls, key, val):
        """
        :param val: Name of the phase
        :type val: str
        """
        return cls(val)

    def __and__(self, other):
        """
        Names are combined with ``/`` along the path to each leaf to reflect names
        of all levels from the root to the leaves.
        """
        return self.__class__(
            f'{self.val}{self.SEPARATOR}{other.val}',
            _from_merge=True,
        )


class MetaStoreProperty(SimpleProperty, MetaPropertyBase):
    """
    Plain key-value storage to be used as the user see fit.

    :param mapping: Dictionary of user-defined keys.
    :type mapping: dict

    Since this is a meta property, it will not influence the generation of the
    JSON and can be used to hold any sort of custom metadata needing to be
    attached to the phases.
    """
    KEY = 'meta'

    def __init__(self, mapping):
        super().__init__(key=self.KEY, val=mapping)

    @classmethod
    def _from_key(cls, key, val):
        return cls(val)

    def __and__(self, other):
        """
        Combine the key value pairs together. In case of conflict, the value on
        closer to the root is chosen.
        """
        return self.__class__(
            mapping={
                **other.val,
                # Root takes precedence
                **self.val,
            },
        )


class PolicyProperty(SimpleConcreteProperty):
    """
    Scheduler policy property.

    :param policy: Scheduling policy:
            * ``SCHED_OTHER`` for CFS task
            * ``SCHED_FIFO`` for FIFO realtime task
            * ``SCHED_RR`` for round-robin realtime task
            * ``SCHED_DEADLINE`` for deadline task
            * ``SCHED_BATCH`` for batch task
            * ``SCHED_IDLE`` for idle task
    :type policy: str
    """
    KEY = 'policy'
    JSON_KEY = KEY
    DEFAULT_JSON = 'SCHED_OTHER'
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: {},
    }
    def __init__(self, policy):
        if policy is not None:
            policy = policy.upper()
            policy = policy if policy.startswith('SCHED_') else f'SCHED_{policy}'

        super().__init__(key=self.KEY, val=policy)

    @classmethod
    def _from_key(cls, key, val):
        return cls(val)

class TaskGroupProperty(SimpleConcreteProperty):
    """
    Task group property.

    :param path: Path of the taskgroup.
    :type path: str

    Only supported by rt-app for ``SCHED_OTHER`` and ``SCHED_IDLE`` for now.
    """
    KEY = 'taskgroup'
    JSON_KEY = KEY
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: {'policy'},
    }
    REQUIRED_KCONFIG_KEYS = ['CONFIG_CGROUP_SCHED']

    def __init__(self, path):
        super().__init__(key=self.KEY, val=path)

    @classmethod
    def _from_key(cls, key, val):
        return cls(val)

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        policy = properties.get('policy')
        # rt-app only supports taskgroup for some policies
        if policy in ('SCHED_OTHER', 'SCHED_IDLE'):
            return {cls.JSON_KEY: '/'}
        else:
            return {}


class PriorityProperty(SimpleConcreteProperty):
    """
    Task scheduler priority property.

    :param priority: Priority of the task.
    :type priority: int
    """
    KEY = 'priority'
    JSON_KEY = KEY
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: {'policy'},
    }

    def __init__(self, priority):
        super().__init__(key=self.KEY, val=priority)

    @classmethod
    def _from_key(cls, key, val):
        return cls(val)

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        """
        The default value depends on the ``policy`` that is in use.
        """
        # The default is context-sensitive
        defaults = {
            'SCHED_OTHER': 0,
            'SCHED_IDLE': 0,
            'SCHED_RR': 10,
            'SCHED_FIFO': 10,
            'SCHED_DEADLINE': 10,
        }
        policy = properties.get('policy', 'SCHED_OTHER')
        val = defaults[policy]
        return {cls.JSON_KEY: val}


class _UsecSimpleConcreteProperty(SimpleConcreteProperty):
    """
    :meta public:

    Simple property that converts its value from seconds to microseconds for
    the JSON file.
    """
    def __init__(self, val):
        super().__init__(key=self.KEY, val=val)

    @classmethod
    def _from_key(cls, key, val):
        return cls(val)

    def _to_json(self, **kwargs):
        val = self.val
        if val is None:
            return {}
        else:
            return {self.json_key: _to_us(self.val)}


class DeadlineRuntimeProperty(_UsecSimpleConcreteProperty):
    """
    ``SCHED_DEADLINE`` scheduler policy's runtime property.

    :param val: runtime in seconds
    :type val: int
    """
    KEY = 'dl_runtime'
    JSON_KEY = 'dl-runtime'
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: {'policy'},
    }

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        if properties.get('policy') != 'SCHED_DEADLINE':
            return {}
        else:
            return {cls.JSON_KEY: 0}


class DeadlinePeriodProperty(_UsecSimpleConcreteProperty):
    """
    ``SCHED_DEADLINE`` scheduler policy's period property.

    :param val: period in seconds
    :type val: int
    """
    KEY = 'dl_period'
    JSON_KEY = 'dl-period'
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: {'policy'}
    }

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        if properties.get('policy') != 'SCHED_DEADLINE':
            return {}

        try:
            val = properties['dl-runtime']
        except KeyError:
            return {}
        else:
            return {cls.JSON_KEY: val}


class DeadlineDeadlineProperty(_UsecSimpleConcreteProperty):
    """
    ``SCHED_DEADLINE`` scheduler policy's deadline property.

    :param val: deadline in seconds
    :type val: int
    """
    KEY = 'dl_deadline'
    JSON_KEY = 'dl-deadline'
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: {'policy'}
    }

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        if properties.get('policy') != 'SCHED_DEADLINE':
            return {}

        try:
            val = properties['dl-period']
        except KeyError:
            return {}
        else:
            return {cls.JSON_KEY: val}


class _AndSetConcreteProperty(AndProperty, SimpleConcreteProperty):
    def __init__(self, items):
        if items is None:
            items = None
        elif isinstance(items, Iterable):
            items = set(items)
        else:
            items = {items}
        super().__init__(key=self.KEY, val=items)

    def _check_val(self, val):
        pass

    def _to_json(self, plat_info):
        val = self.val
        if val is None:
            return {}
        elif not val:
            raise ValueError(f'Empty {self.key} set')
        else:
            self._check_val(val, plat_info)
            return {self.json_key: sorted(val)}

    @classmethod
    def _from_key(cls, key, val):
        return cls(val)


class CPUProperty(_AndSetConcreteProperty):
    """
    CPU affinity property.

    :param cpus: Set of CPUs the task will be bound to.
    :type cpus: set(int) or None
    """
    KEY = 'cpus'
    JSON_KEY = KEY
    # Do not optimize out the cpus settings: unlike every single else setting,
    # rt-app *will not* leave the cpu affinity alone if the user has not set it
    # in a given phase. Instead, it's gonna reset it to some default value.
    OPTIMIZE_JSON_KEYS = {}

    def __init__(self, cpus):
        super().__init__(items=cpus)

    def _check_val(self, val, plat_info):
        plat_cpus = set(range(plat_info['cpus-count']))
        if not (val <= plat_cpus):
            raise ValueError(f'CPUs {val} outside of allowed range of CPUs {plat_cpus}')

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        # If no CPU set is given, set the affinity to all CPUs in the system,
        # i.e. do not set any affinity. This is necessary to decouple the phase
        # from any CPU set in a previous phase

        # Sorting is important for deduplication of JSON keys
        cpus = sorted(range(plat_info['cpus-count']))
        return {cls.JSON_KEY: cpus}


class NUMAMembindProperty(_AndSetConcreteProperty):
    """
    NUMA node membind property.

    :param nodes: Set of NUMA nodes the task will be bound to.
    :type nodes: set(int) or None
    """
    KEY = 'numa_nodes_membind'
    JSON_KEY = 'nodes_membind'
    OPTIMIZE_JSON_KEYS = {
        JSON_KEY: set(),
    }

    def __init__(self, nodes):
        super().__init__(items=nodes)

    def _check_val(self, val, plat_info):
        plat_nodes = set(range(plat_info['numa-nodes-count']))
        if not (val <= plat_nodes):
            raise ValueError(f'NUMA nodes {val} outside of allowed range of NUMA nodes {plat_nodes}')

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        # Sorting is important for deduplication of JSON keys
        nodes = sorted(range(plat_info['numa-nodes-count']))
        return {cls.JSON_KEY: nodes}


class MultiProperty(PropertyBase):
    """
    Base class for properties setting multiple JSON keys at once.
    """


class MultiConcreteProperty(MultiProperty, ConcretePropertyBase):
    DEFAULT_JSON = None

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        return cls.DEFAULT_JSON or {}


class ComposableMultiConcretePropertyBase(MultiConcreteProperty):
    """
    Base class for properties that are a collection of values.

    :Variable keyword arguments: attributes to set on the instance.
    """

    _ATTRIBUTES = {}
    """
    :meta public:

    Dictionary of allowed attributes where each value is in the format
    ``dict(doc=..., type_=...)``. This extra information is used to patch the
    docstrings (see :meth:`__init_subclass__`).
    """

    _ATTRIBUTE_DEFAULT = None
    """
    Default value given to attributes that have not been set by the user.
    """

    def __init__(self, **kwargs):
        def check(key, val):
            if key in self._ATTRIBUTES:
                return val
            else:
                raise TypeError(f'Unknown parameter "{key}". Only {sorted(self._ATTRIBUTES)} are allowed')

        attrs = {
            key: check(key, val)
            for key, val in kwargs.items()
        }
        self._attrs = FrozenDict({
            **dict.fromkeys(self._ATTRIBUTES, self._ATTRIBUTE_DEFAULT),
            **attrs
        })

    def __str__(self):
        key = itemgetter(0)
        attrs = ', '.join(
            f'{k}={v}'
            # Order the attrs as keys of _ATTRIBUTES, and alphanumeric sort for
            # the others if there is any.
            for k, v in order_as(
                sorted(self._attrs.items(), key=key),
                self._ATTRIBUTES.keys(),
                key=key
            )
            if v != self._ATTRIBUTE_DEFAULT
        )
        return f'{self.__class__.__qualname__}({attrs})'

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Update the docstring used as a :meth:`str.format` template with the
        following keys:

            * ``{params}``: replaced by the Sphinx-friendly list of attributes
        """
        docstring = inspect.getdoc(cls)
        if docstring:
            cls.__doc__ = docstring.format(
                params=cls._get_rst_param_doc()
            )

        super().__init_subclass__(**kwargs)

    @classmethod
    def _get_rst_param_doc(cls):
        def make(param, desc):
            fst = f':param {param}: {desc["doc"]}'
            snd = f':type {param}: {get_cls_name(desc["type_"])} or None'
            return f'{fst}\n{snd}'

        return '\n\n'.join(starmap(make, cls._ATTRIBUTES.items()))

    __repr__ = __str__

    def __getattr__(self, attr):
        # TODO: refer to :attr:`_ATTRIBUTES` once this bug is fixed:
        # https://github.com/sphinx-doc/sphinx/issues/8922
        """
        Lookup the attributes values defined in ``_ATTRIBUTES``.
        """
        # Prevent infinite recursion on deepcopy
        if attr == '_attrs':
            raise AttributeError()

        try:
            return self._attrs[attr]
        except KeyError:
            raise AttributeError(f'Attribute "{attr}" not available')

    @classmethod
    def _from_key(cls, key, val):
        if not isinstance(val, cls):
            raise TypeError(f'"{cls.KEY}" key needs a value of a subclass of {cls.__qualname__}, not {val.__class__.__qualname__}')

        return val

    @classmethod
    def from_product(cls, **kwargs):
        """
        To be called the same way as the class itself, except that all values
        are expected to be iterables and the class will be called with all
        combinations, returning a list of instances.
        """
        names, values = zip(*kwargs.items())
        return [
            cls(**dict(zip(names, combi)))
            for combi in product(*values)
        ]

    @property
    def val(self):
        return self

    def __and__(self, other):
        # Since each subclass describes a totally different kind of workload,
        # adding them together does not make sense
        if self.__class__ is other.__class__:
            return self._and(other)
        else:
            raise TypeError(f'Cannot add {self.__class__.__qualname__} instance with {other.__class__.__qualname__}')

    def _and(self, other):
        """
        :meta public:

        Combine together two instances by taking the non-default values for
        each attribute, and giving priority to ``self``.
        """
        default = self._ATTRIBUTE_DEFAULT
        def and_(name, x, y):
            # Give priority to the left operand
            if x == default:
                return y
            else:
                return x

        kwargs = {
            attr: and_(attr, val, other._attrs[attr])
            for attr, val in self._attrs.items()
        }
        return self.__class__(**kwargs)


class UclampProperty(ComposableMultiConcretePropertyBase):
    """
    Set util clamp (uclamp) values.

    {params}
    """
    KEY = 'uclamp'
    OPTIMIZE_JSON_KEYS = dict.fromkeys(
        ('util_min', 'util_max'),
        {'policy', 'priority'}
    )
    REQUIRED_KCONFIG_KEYS = ['CONFIG_UCLAMP_TASK']

    _ATTRIBUTES = {
        'min_': dict(
            type_=int,
            doc='Minimum value that util can take for this task. If ``None``, min clamp is removed.',
        ),
        'max_': dict(
            type_=int,
            doc='Maximum value that util can take for this task. If ``None``, max clamp is removed.',
        ),
    }

    @classmethod
    def _get_default(cls, plat_info):
        # Old kernels don't support removing the clamp, so we have to fall back
        # on setting a "real" clamp. This can result in different behaviors
        # when mixed with cgroups but it's the best we can do on such systems.
        ref_version = (5, 11)
        version = plat_info['kernel']['version'].parts[:len(ref_version)]
        if None not in version and version < ref_version:
            return (0, 1024)
        else:
            return (-1, -1)

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        min_, max_ = cls._get_default(plat_info)
        return {
            'util_min': min_,
            'util_max': max_,
        }

    def _to_json(self, plat_info):
        min_ = self.min_
        max_ = self.max_
        if None not in (min_, max_) and min_ > max_:
            raise ValueError(f'{self.__class__.__qualname__}: min={min_} cannot be higher than max={max_}')

        def_min, def_max = self._get_default(plat_info)

        min_ = min_ if min_ is not None else def_min
        max_ = max_ if max_ is not None else def_max

        return {
            'util_min': min_,
            'util_max': max_,
        }

    @classmethod
    def _from_key(cls, key, val):
        """
        :param val: Clamp for the utilization.
        :type val: tuple(int or None, int or None) or UclampProperty
        """
        if isinstance(val, cls):
            return super()._from_key(key, val)
        else:
            min_, max_ = val
            return cls(min_=min_, max_=max_)

    def _and(self, other):
        """
        :meta public:

        Combine clamps by taking the most constraining solution.
        """
        def none_shortcircuit(f, x, y):
            if (x, y) == (None, None):
                return None
            elif x is None:
                return y
            elif y is None:
                return x
            else:
                return f(x, y)

        return self.__class__(
            # Use max() to combine min and min() to combine max, so that we end
            # up with pick the strongest constraints.
            min_=none_shortcircuit(max, self.min_, other.min_),
            max_=none_shortcircuit(min, self.max_, other.max_),
        )


class WloadPropertyBase(ConcretePropertyBase):
    """
    Phase workload.

    Workloads are a sequence of rt-app events such as ``run``.
    """
    KEY = 'wload'
    # TODO: remove that depending on outcome of:
    # https://github.com/scheduler-tools/rt-app/pull/108
    JSON_KEY = 'events'

    @classmethod
    def _from_key(cls, key, val):
        if not isinstance(val, cls):
            raise TypeError(f'"{cls.KEY}" key needs a value of a subclass of {cls.__qualname__}, not {val.__class__.__qualname__}')

        return val

    @property
    def val(self):
        return self

    def __add__(self, other):
        """
        Adding two workloads together concatenates them.

        .. note:: Any subclass implementation of ``__add__`` must be
            associative, i.e. ``a+(b+c) == (a+b)+c``. This property is relied on.
        """
        return WloadSequence(wloads=[self, other])

    def __mul__(self, n):
        """
        Replicate the given workload ``n`` times.
        """
        if n == 1:
            return copy.copy(self)
        else:
            return WloadSequence(wloads=[self] * n)

    @abc.abstractmethod
    def to_events(self, **kwargs):
        pass

    def _to_json(self, **kwargs):
        """
        Deduplicate the event names and turn the whole workload into a loop if
        possible.
        """
        events = list(self.to_events(**kwargs))
        loop, events = loopify(events)

        # add unique stable suffix to duplicated events, after loopification.
        # Note: this suffix should be ignored by rt-app, even if it's not
        # currently explicilty documented as such. rt-app/doc/workgen script
        # also relies on that.
        def dedup(state, item):
            items, seen = state

            key, val = item
            nr = seen.setdefault(key, -1) + 1
            seen[key] = nr

            if nr:
                key = f'{key}-{nr}'

            return (items + [(key, val)], seen)

        if events:
            events, _ = fold(dedup, events, init=([], {}))

        # Keep "loop" at the beginning for readability
        json_ = OrderedDict([('loop', loop)])
        json_.update(events)
        assert json_['loop'] == loop
        return json_

    @classmethod
    def _to_default_json(cls, plat_info, properties):
        return {}


class WloadSequence(WloadPropertyBase, SimpleConcreteProperty):
    """
    Sequence of workloads, to be executed one after another.

    .. note:: Adding together two :class:`WloadPropertyBase` with ``+``
        operator will give a :class:`WloadSequence`, so there is usually no
        need to create one explicitly.
    """
    def __init__(self, wloads):
        self.wloads = list(wloads)

    @property
    # @memoized
    def _expanded_wloads(self):
        # Deep first expansion of the WloadSequence tree followed by combining
        # the workloads.
        def topo_sort(wload):
            if isinstance(wload, WloadSequence):
                return chain.from_iterable(map(topo_sort, wload.wloads))
            else:
                return [wload]

        def add(wloads, wload):
            if wloads:
                last = wloads[-1]
                # This relies on associativity of the __add__ definition for
                # all workloads, since we work on the topological sort of the
                # tree, thereby loosing the information on the exact internal
                # stucture of the "+" expression tree
                new = last + wload
                # If the workload has a special __add__, use the result
                if not isinstance(new, WloadSequence):
                    wload = new
                    wloads = wloads[:-1]

            return wloads + [wload]

        return list(functools.reduce(add, topo_sort(self), []))

    def __str__(self):
        return ' -> '.join(map(str, self._expanded_wloads))

    def __bool__(self):
        return any(map(bool, self.wloads))

    def to_events(self, **kwargs):
        return list(chain.from_iterable(
            wload.to_events(**kwargs)
            for wload in self._expanded_wloads
        ))


class _SingleWloadBase(WloadPropertyBase):
    """
    :meta public:

    Execute a single rt-app event.
    """

    _ACTION = None
    """
    Name of the rt-app JSON event to execute.
    """

    def __init__(self, action=None):
        self._action = action or self._ACTION

    def __str__(self):
        return f'{self._action}({self.json_value})'

    def __and__(self):
        if self != other:
            raise ValueError(f'Conflicting properties "{self}" and "{other}')
        else:
            return copy.copy(self)

    @property
    @abc.abstractmethod
    def json_value(self):
        """
        Value to pass to JSON.
        """

    def to_events(self, **kwargs):
        return [(self._action, self.json_value)]


class DurationWload(WloadPropertyBase):
    """
    Workload parametrized by a duration.
    """
    def __init__(self, duration, **kwargs):
        self.duration = duration
        super().__init__(**kwargs)

    @classmethod
    def from_duration(cls, duration):
        """
        Build a workload from the given ``duration`` in seconds.
        """
        return cls(duration=duration)

    def __str__(self):
        return f'{self._action}({self.duration})'

    @property
    def json_value(self):
        return _to_us(self.duration)


class DurationWload(DurationWload, _SingleWloadBase):
    pass


class RunWload(DurationWload):
    """
    Workload for the ``run`` event.

    :param duration: Duration of the run in seconds.
    :type duration: float
    """
    _ACTION = 'run'


class RunForTimeWload(DurationWload):
    """
    Workload for the ``runtime`` event.

    :param duration: Duration of the run in seconds.
    :type duration: float
    """
    _ACTION = 'runtime'


class SleepWload(DurationWload):
    """
    Workload for the ``sleep`` event.

    :param duration: Duration of the sleep in seconds.
    :type duration: float
    """
    _ACTION = 'sleep'


class TimerWload(DurationWload):
    """
    Workload for the ``timer`` event.

    :param duration: Duration of the timer period in seconds.
    :type duration: float
    """
    _ACTION = 'timer'

    @property
    def json_value(self):
        return {
            # This special reference ensures each thread get their own timer
            'ref': 'unique',
            'period': _to_us(self.duration)
        }

class BarrierWload(_SingleWloadBase):
    """
    Workload for the ``barrier`` event.

    :param barrier: Name of the barrier
    :type barrier: str
    """
    _ACTION = 'barrier'

    def __init__(self, barrier, **kwargs):
        self.barrier = barrier
        super().__init__(**kwargs)

    @property
    def json_value(self):
        return self.barrier


class LockWload(_SingleWloadBase):
    """
    Workload for the ``lock`` and ``unlock`` event.

    :param lock: Name of the lock
    :type lock: str

    :param action: One of ``lock`` or ``unlock``.
    :type action: str
    """
    def __init__(self, lock, action='lock', **kwargs):
        self.lock = lock
        if action not in ('lock', 'unlock'):
            raise ValueError(f'Unknown action: {action}')
        super().__init__(action=action, **kwargs)

    @property
    def json_value(self):
        return self.lock


class YieldWload(_SingleWloadBase):
    """
    Workload for the ``yield`` event.
    """
    _ACTION = 'yield'

    @property
    def json_value(self):
        return ''


class WaitWload(_SingleWloadBase):
    """
    Workload for the ``wait``, ``signal`` and ``broad`` events.

    :param lock: Name of the lock
    :type lock: str

    :param action: One of ``wait``, ``signal`` or ``broad``.
    :type action: str
    """
    def __init__(self, resource, action='wait', **kwargs):
        self.resource = resource
        # Action can also be set to "sync" directly by __add__, so it bypasses
        # the check
        if action not in ('wait', 'signal', 'broad'):
            raise ValueError(f'Unknown action: {action}')
        super().__init__(action=action, **kwargs)

    @property
    def json_value(self):
        return self.resource

    def __add__(self, other):
        """
        Combine a ``signal`` :class:`WaitWload` with a ``wait``
        :class:`WaitWload` into a ``sync`` workload.
        """
        if (
            isinstance(other, self.__class__) and
            self._action == 'signal' and
            other._action == 'wait' and
            self.resource == other.resource
        ):
            new = copy.copy(self)
            new._action = 'sync'
            return new
        else:
            return super().__add__(other)


class _SizeSingleWload(_SingleWloadBase):
    def __init__(self, size, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def __str__(self):
        return f'{self._action}({self.size})'

    @property
    def json_value(self):
        return self.size


class MemWload(_SizeSingleWload):
    """
    Workload for the ``mem`` event.

    :param size: Size in bytes to be written to the buffer.
    :type size: int
    """
    _ACTION = 'mem'


class IOWload(_SizeSingleWload):
    """
    Workload for the ``iorun`` event.

    :param size: Size in bytes to be written to the file.
    :type size: int
    """
    _ACTION = 'iorun'
    # TODO: add an "io_device" global key to optionally change the file to
    # write to (defaults to /dev/null)


class PeriodicWload(WloadPropertyBase, ComposableMultiConcretePropertyBase):
    """
    Periodic task workload.

    The task runs to complete a given amount of work, then sleeps until the end
    of the period.
    {params}
    """
    _ATTRIBUTES = {
        'duty_cycle_pct': dict(
            doc="Duty cycle of the task in percents (when executing on the fastest CPU at max frequency). This is effectively equivalent to an amount of work.",
            type_=float,
        ),
        'period': dict(
            doc="Period of the activation pattern in seconds",
            type_=float,
        ),
        'duration': dict(
            doc="Duration of the workload in seconds. If ``None``, keep running forever",
            type_=float,
        ),
        'scale_for_cpu': dict(
            doc='CPU ID used to scale the ``duty_cycle_pct`` value on asymmetric systems. If ``None``, it will be assumed to be the fastest CPU on the system.',
            type_=int,
        ),
        'scale_for_freq': dict(
            doc='Frequency used to scale ``duty_cycle_pct`` in a similar way to ``scale_for_cpu``. This is only valid in conjunction with ``scale_for_cpu``.',
            type_=int,
        ),
        'run_wload': dict(
            doc="Workload factory callback used for the running part. It will be called with a single ``duration`` parameter (in seconds) and must return a :class:`WloadPropertyBase`. Note that the passed duration is scaled according to ``scale_for_cpu`` and ``scale_for_freq``",
            type_=type,
        ),
        'sleep_wload': dict(
            doc="Workload factory callback used for the sleeping part. It will be called with a ``duration`` parameter and ``period`` parameter (in seconds) and must return a :class:`WloadPropertyBase`. Note that the passed duration is scaled according to ``scale_for_cpu`` and ``scale_for_freq``",
            type_=Callable,
        ),
        'guaranteed_time': dict(
            doc="Chooses the default 'sleep_wload'. Can be 'period' (guarantee the period is fixed regardless of preemption) or 'sleep' (guarantee the time spent sleeping, stretching the period if the task is preempted)",
            type_=str,
        ),
    }

    def unscaled_duty_cycle_pct(self, plat_info):
        cpu = self.scale_for_cpu
        freq = self.scale_for_freq
        dc = self.duty_cycle_pct

        if cpu is None:
            capa = PELT_SCALE
        else:
            capa = plat_info.get_nested_key(['cpu-capacities', 'rtapp'], quiet=True)[cpu]

        if freq is not None:
            freqs = plat_info.get_key(['freqs'], quiet=True)[cpu]
            capa *= freq / max(freqs)

        capa /= PELT_SCALE
        return dc * capa

    def to_events(self, plat_info):
        duty_cycle_pct = self.duty_cycle_pct
        duration = self.duration
        period = self.period
        scale_cpu = self.scale_for_cpu
        scale_freq = self.scale_for_freq
        guaranteed_time = self.guaranteed_time or 'period'

        if period is None or period <= 0:
            raise ValueError(f'Period outside ]0,+inf[ : {period}')
        if duty_cycle_pct is None:
            raise ValueError('duty_cycle_pct cannot be None')
        if duration is None:
            raise ValueError('duration cannot be None')

        if scale_cpu is not None:
            duty_cycle_pct = self.unscaled_duty_cycle_pct(plat_info)
        elif scale_freq:
            raise ValueError(f'scale_for_freq is ignored if scale_for_cpu is None')

        if not (0 <= duty_cycle_pct <= 100):
            raise ValueError(f'duty_cycle_pct={duty_cycle_pct} outside of [0, 100]')

        if period > duration:
            raise ValueError(f'period={period} cannot be higher than duration={duration}')

        def get_run(duration, period, make_wload=None):
            make_wload = make_wload or self.run_wload or RunWload.from_duration
            wload = make_wload(duration)
            return list(wload.to_events(plat_info=plat_info))

        def get_sleep(duration, period):
            if self.sleep_wload:
                make_wload = self.sleep_wload
            elif guaranteed_time == 'period':
                make_wload = lambda _, period: TimerWload.from_duration(period)
            elif guaranteed_time == 'sleep':
                make_wload = lambda duration, _: SleepWload.from_duration(duration)
            else:
                raise ValueError(f'Invalid value for guaranteed_time: {guaranteed_time}')

            wload = make_wload(duration, period)
            return list(wload.to_events(plat_info=plat_info))

        if duty_cycle_pct == 0:
            events = get_run(duration, period, SleepWload.from_duration)
        elif duty_cycle_pct == 100:
            events = get_run(duration, period)
        else:
            run = duty_cycle_pct * period / 100
            # Use math.floor() so we never exceed "duration"
            loop = math.floor(duration / period)

            run_events = get_run(run, period)
            sleep_events = get_sleep(period - run, period)
            # run events have to come before "timer" as events are processed in
            # order
            events = loop * (run_events + sleep_events)

        return events


class RTAPhaseProperties(SimpleHash, Mapping):
    """
    Hold the properties of an :class:`RTAPhaseBase`.

    :param properties: List of properties.
    :type properties: list(PropertyBase)
    """
    def __init__(self, properties):
        properties = [
            (prop.key, prop)
            for prop in (properties or [])
        ]
        self.properties = FrozenDict(_make_dict(properties), deepcopy=False, type_=lambda x: x)

    @classmethod
    def from_polymorphic(cls, obj):
        """
        Alternative constructor with polymorphic input:

            * ``None``: equivalent to an empty list.
            * :class:`RTAPhaseProperties`: taken as-is.
            * :class:`~collections.abc.Mapping`: each key/value pair is either:

                * the value is a :class:`PropertyBase`: it's taken as-is
                * the value is a :class:`PlaceHolderValue`: the property is
                  created using its ``PROPERTY_CLS.from_key`` method.
                * otherwise, an instance of the appropriate class is built by
                  :meth:`PropertyBase.from_key`.
        """
        if obj is None:
            return cls(properties=[])
        elif isinstance(obj, cls):
            return obj
        elif isinstance(obj, Mapping):
            def from_key(key, val):
                # Allow Property to be used as values in the dict directly, in
                # case just one value needs some specific setting and the rest
                # is using the simple API
                if isinstance(val, PropertyBase):
                    return val
                elif isinstance(val, PlaceHolderValue):
                    return val.PROPERTY_CLS.from_key(
                        key=key,
                        val=val.val,
                        **val.kwargs
                    )
                else:
                    return PropertyBase.from_key(key, val)

            properties = list(starmap(from_key, obj.items()))
            return cls(properties)
        else:
            raise TypeError(f'Unsupported type: {obj.__class__}')

    def to_json(self, plat_info, **kwargs):
        """
        Output a JSON object with the values of all properties, including
        defaults if a given property is not set.
        """
        kwargs['plat_info'] = plat_info
        properties = _make_dict(chain.from_iterable(
            prop.to_json(**kwargs).items()
            for prop in self.properties.values()
            if isinstance(prop, ConcretePropertyBase)
        ))

        return OrderedDict(chain(
            self.get_defaults(plat_info, properties).items(),
            # Keep ordering of properties as they were output
            properties.items()
        ))

    @classmethod
    def get_defaults(cls, plat_info, properties=None, trim_defaults=True):
        """
        Get the default JSON object for the phase with the given user-derived
        JSON ``properties``.

        :param plat_info: Platform information used to compute some defaults
            values, such as the default CPU affinity set based on the number of
            CPUs.
        :type plat_info: lisa.platforms.platinfo.PlatformInfo

        :param properties: JSON object derived from user-provided properties.
            It is used to compute some context-sensitive defaults, such as the
            ``priority`` that depends on the ``policy``.
        :type properties: dict(str, object)

        :param trim_defaults: If ``True``, default values that are already set
            in ``properties`` will be omitted.
        :type trim_defaults: bool
        """
        properties = properties or {}

        def get_defaults(defaults):
            return _make_dict(chain.from_iterable(
                subcls.to_default_json(
                    plat_info=plat_info,
                    properties={
                        **defaults,
                        **properties
                    },
                ).items()
                for subcls in get_subclasses(ConcretePropertyBase)
            ))

        # Compute the defaults until they are stable, to take into account any
        # dependency between keys
        defaults = fixedpoint(get_defaults, {}, limit=1000)
        return OrderedDict(
            (key, val)
            # sort the defaults to get stable output.
            for key, val in sorted(defaults.items())
            # Remove the keys that are set in properties from the defaults,
            # otherwise it will mess up the order in the final OrderedDict.
            # Keys that appear twice would be combined such that the latest
            # value is inserted at the position of the first key, so the first
            # key cannot be in "defaults", otherwise the order set in
            # "properties" will be broken
            if not (trim_defaults and key in properties)
        )

    def __and__(self, other):
        """
        Combine two instances.

        Properties are merged according to the following rules:

            * Take the value as-is for all the keys that only appear in one of
              them.
            * For values set in both properties, combine them with ``&``
              operator. The value coming from ``self`` will be the left
              operand.
        """
        common = self.properties.keys() & other.properties.keys()
        merged = [
            # Order of operand matters, "&" is not expected to be commutative
            # or associative (for some of the properties). The order is chosen so
            # that the left operand is closer to the root of the tree.
            self.properties[key] & other.properties[key]
            # Preserve the key order that can be important
            for key in order_as(
                common,
                order_as=self.properties.keys(),
            )
        ]
        for properties in (self.properties, other.properties):
            merged.extend(
                prop
                # It is important that order is preserved for the properties
                # coming from any given mapping, since correctness depends on
                # it for rt-app events like run and barriers.
                for key, prop in properties.items()
                if key not in common
            )

        return self.__class__(properties=merged)

    @property
    def existing_properties(self):
        """
        Trim the properties to only contain the "public" ones, i.e. the ones
        that have not been deleted.
        """
        return OrderedDict(
            (key, val)
            for key, val in self.properties.items()
            if not isinstance(val, DeletedProperty)
        )

    def __getitem__(self, key):
        return self.existing_properties[key].val

    def __iter__(self):
        return iter(self.existing_properties)

    def __len__(self):
        return len(self.existing_properties)

    def __bool__(self):
        # Do not use existing_properties here, since checking for emptiness is
        # used to know if the properties will have any effect when combined
        # with another.
        return bool(self.properties)

    def __str__(self):
        return str(dict(
            (k, str(v))
            for k, v in self.existing_properties.items()
        ))


class _RTAPhaseBase:
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Update the docstring used as a :meth:`str.format` template with the
        following keys:

            * ``{prop_kwargs}``: replaced by the Sphinx-friendly list of
              "prop_*" keyword arguments
        """
        docstring = inspect.getdoc(cls)
        if docstring:
            cls.__doc__ = docstring.format(
                prop_kwargs=cls._get_rst_prop_kwargs_doc()
            )

        super().__init_subclass__(**kwargs)

    @classmethod
    def _get_rst_prop_kwargs_doc(cls):
        def make(key, cls):
            param = f'prop_{key}'
            doc, type_ = cls._get_cls_doc()
            fst = f':param {param}: {doc}'
            snd = f':type {param}: {type_}'
            return f'{fst}\n{snd}'

        properties = {
            cls.KEY: cls
            for cls in get_subclasses(PropertyBase)
            if cls.KEY is not None
        }

        return '\n\n'.join(starmap(make, sorted(properties.items())))


class RTAPhaseBase(_RTAPhaseBase, SimpleHash, Mapping, abc.ABC):
    """
    Base class for rt-app phase modelisation.

    :param properties: Properties mapping to set on that phase. See
        :meth:`RTAPhaseProperties.from_polymorphic` for the accepted formats.
        Alternatively, keyword arguments ``prop_*`` can be used.
    :type properties: object

    {prop_kwargs}
    """
    def __init__(self, properties=None, **kwargs):
        properties, other_kwargs = self.split_prop_kwargs(kwargs, properties)

        if other_kwargs:
            illegal = ', '.join(sorted(other_kwargs.keys()))
            raise TypeError(f'TypeError: got an unexpected keyword arguments: {illegal}')

        self.properties = RTAPhaseProperties.from_polymorphic(properties)

    def __str__(self):
        sep = '\n' + ' ' * 4
        props = sep.join(
            f'{key}={val}'
            for key, val in sorted(self.properties.items(), key=itemgetter(0))
            if key != 'name'
        )
        try:
            name = self['name']
        except KeyError:
            name = 'Phase'
        else:
            name = f'Phase {name}'

        return f'{name}:{sep}{props}'

    def with_phase_properties(self, properties):
        """
        Return a cloned instance with the properties combined with the given
        ``properties`` using :meth:`RTAPhaseProperties.__and__` (``&``). The
        ``properties`` parameter is the left operand.
        """
        new = copy.copy(self)
        new.properties = properties & new.properties
        return new

    def with_properties_map(self, properties, **kwargs):
        """
        Same as :meth:`with_phase_properties` but with ``properties`` passed to
        :meth:`RTAPhaseProperties.from_polymorphic` first.
        """
        return self.with_phase_properties(
            RTAPhaseProperties.from_polymorphic(properties),
            **kwargs,
        )

    def with_props(self, **kwargs):
        """
        Same as :meth:`with_phase_properties` but using keyword arguments to
        set each property. The resulting dictionary is passed to
        :meth:`RTAPhaseProperties.from_polymorphic` first.
        """
        return self.with_properties_map(kwargs)

    def with_delete_props(self, properties):
        """
        Delete all the given property names, equivalent to
        `with_props(foo=delete())``
        """
        return self.with_properties_map(
            dict.fromkeys(properties, delete())
        )

    @abc.abstractmethod
    def get_rtapp_repr(self, *, task_name, plat_info, force_defaults=False, no_force_default_keys=None, **kwargs):
        """
        rt-app JSON representation of the phase.

        :param task_name: Name of the task this phase will be attached to.
        :type task_name: str

        :param plat_info: Platform information used to compute default
            properties and validate them.
        :type plat_info: lisa.platforms.platinfo.PlatformInfo

        :param force_defaults: If ``True``, a default value will be provided
            for all properties that are not set. If ``False``, the defaults
            will not be provided if the user-provided properties don't touch a
            given JSON key.
        :type force_defaults: bool

        :param no_force_default_keys: List of JSON keys for which no default
            will be emitted when ``force_defaults=True``.
        :type no_force_default_keys: list(str) or None

        :Variable keyword arguments: Forwarded to
            :meth:`RTAPhase.to_json`
        """

    def __add__(self, other):
        """
        Compose two phases together by running one after the other.

        Since this operation returns an :class:`RTAPhaseTree`, it is possible
        to set properties on it that will only apply to its children.
        """
        return RTAPhaseTree(children=[self, other])

    def __mul__(self, n):
        """
        Multiply the phase by ``n``, in order to repeat it.
        """
        if n == 1:
            return copy.copy(self)
        else:
            return RTAPhaseTree(children=[self] * n)

    def __rmul__(self, n):
        return self.__mul__(n)

    def __getitem__(self, key):
        """
        Lookup the value of the given property on that phase.
        """
        return self.properties[key]

    def __len__(self):
        return len(self.properties)

    def __iter__(self):
        return iter(self.properties)

    @staticmethod
    def split_prop_kwargs(kwargs, properties=None):
        """
        Split the ``kwargs`` into two categories:

            * Arguments with a name starting with ``prop_``. They are then
              merged with the optional ``properties``.
            * The others

        Returns a tuple ``(properties, other_kwargs)``.
        """
        def dispatch(item):
            key, val = item
            if key.startswith('prop_'):
                return 'properties'
            else:
                return 'others'

        kwargs = dict(groupby(kwargs.items(), key=dispatch))
        kwargs['properties'] = {
            key[len('prop_'):]: val
            for key, val in kwargs.get('properties', [])
        }

        for cat in ('properties', 'others'):
            kwargs[cat] = dict(kwargs.get(cat, {}))

        properties = _make_dict(chain(
            (properties or {}).items(),
            kwargs['properties'].items()
        ))

        return (properties, kwargs['others'])



class _RTAPhaseTreeBase(RTAPhaseBase, abc.ABC):
    """
    :meta public:

    Base class for phases laid out as a tree.
    """
    @abc.abstractmethod
    def topo_sort(self):
        """
        Topological sort of the subtree.

        :rtype: list(RTAPhase)

        The merge of :class`PhaseProperties` object is done from root to leaf
        (pre-order traversal). This is important for some classes that are not
        semigroup like :class:`OverridenProperty`.
        """

    @property
    @abc.abstractmethod
    def is_empty(self):
        """
        ``True`` if the phase has no content and will result in an empty JSON
        phase(s).
        """

    @property
    def phases(self):
        """
        Topological sort of the phases in the tree, with the properties merged
        along each path from the root to the leaves.
        """
        return self.topo_sort()

    def get_rtapp_repr(self, task_name, plat_info, force_defaults=False, no_force_default_keys=None, **kwargs):
        phases = self.phases

        # to_json is expected to apply the defaults itself
        json_phases = [
            phase.to_json(
                plat_info=plat_info,
                **kwargs
            )
            for phase in phases
        ]

        defaults = [
            (
                json_phase,
                RTAPhaseProperties.get_defaults(
                    plat_info=plat_info,
                    properties=json_phase,
                    trim_defaults=False,
                )
            )
            for json_phase in json_phases
        ]

        # All the keys that have a default value somewhere are potentially
        # removable
        removable_keys = set(chain.from_iterable(
            default.keys()
            for _, default in defaults
        ))

        # If we want to force the defaults, we restrict the set of removable
        # keys to the ones for which we are not going to force the default.
        if force_defaults:
            removable_keys &= set(no_force_default_keys or [])

        keys_to_remove = set(
            key
            for key in removable_keys
            # Remove the key if it is not present at all or set to its
            # default value in all phases
            if all(
                (
                    # If the key is neither in the defaults of that phase
                    # nor in the phase itself, it won't matter if we
                    # attempt to remove it or not
                    (
                        key not in phase and
                        key not in phase_default
                    ) or
                    (
                        # If the key is in phase and not phase_default or
                        # the opposite, we treat it as a non-default
                        # setting.
                        key in phase and
                        key in phase_default and
                        phase[key] == phase_default[key]
                    )
                )
                for phase, phase_default in defaults
            )
        )

        def remove_keys(dct, keys):
            if keys:
                return OrderedDict(
                    (key, val)
                    for key, val in dct.items()
                    if key not in keys
                )
            else:
                return dct

        json_phases = [
            remove_keys(phase, keys_to_remove)
            for phase in json_phases
        ]

        # All the JSON properties that need to be considered to optimize-away
        # redundant values between phases, except when one of their
        # optimization barrier key changes.
        optimize_barriers = list(chain.from_iterable(
            subcls.OPTIMIZE_JSON_KEYS.items()
            for subcls in get_subclasses(ConcretePropertyBase)
        ))
        optimize_barriers = {
            key: set(chain.from_iterable(map(itemgetter(1), item)))
            for key, item in groupby(optimize_barriers, key=itemgetter(0))
        }
        to_dedup = optimize_barriers.keys()

        def _dedup(fold_state, properties):
            state, processed = fold_state

            for key, val in properties.items():
                barriers = optimize_barriers.get(key, set())
                # For each key in the currently inspected properties, check if
                # any other key acting as a barrier for it had a change of
                # value. If so, we remove the value of the key from the current
                # state so it will not be optimized out in the inspected
                # properties, even if it has the same value as the one in the
                # state.
                if any(
                    state.get(barrier) != properties.get(barrier)
                    for barrier in barriers
                ):
                    try:
                        del state[key]
                    except KeyError:
                        pass

            properties = OrderedDict(
                (key, val)
                for key, val in properties.items()
                # Filter out settings that are equal to the current state
                if not (key in to_dedup and key in state and val == state[key])
            )

            # Update the state for the next round
            state = {
                **state,
                **properties,
            }
            # Build the list of processed properties
            processed = processed + [properties]

            return (state, processed)

        def dedup(properties_list):
            properties_list = list(properties_list)
            if properties_list:
                _, properties_list = fold(_dedup, properties_list, init=({}, []))
                return properties_list
            else:
                return []

        json_phases = dedup(json_phases)

        _json_phases = json_phases
        loop, json_phases = loopify(json_phases)
        # Check loopify gave a prefix of json_phases, since we rely on that
        # with zip() to associate the phase object
        assert json_phases == _json_phases[:len(json_phases)]

        return {
            'loop': loop,
            'phases': OrderedDict(
                # Some phases might not have a name. Only phases of accessed
                # via the "phases" property in a RTAPhaseTree have this
                # guarantee
                (phase.get('name', str(i)), json_phase)
                for i, (phase, json_phase) in enumerate(zip(phases, json_phases))
            )
        }


class RTAPhase(_RTAPhaseTreeBase):
    """
    Leaf in a tree of :class:`RTAPhaseTree`.

    {prop_kwargs}
    """
    def to_json(self, **kwargs):
        """
        JSON content of the properties of the phase.
        """
        properties = self.properties

        # rt-app errors on phases without any events, so provide a dummy one
        # that will do nothing
        if not properties.get('wload'):
            dummy_wload = RunWload(0)
            # Make sure the dummy wload is the left operand, to override any
            # DeletedProperty
            properties = RTAPhaseProperties(
                [OverridenProperty(dummy_wload)]
            ) & properties

        return properties.to_json(**kwargs)

    def topo_sort(self):
        return [self]

    @property
    def is_empty(self):
        return not self.properties


class RTAPhaseTreeChildren(SimpleHash, Mapping):
    """
    Proxy object used by :class:`RTAPhaseTree` to store the children list.

    It provides a mapping interface where children can be looked up by name if
    they have one.

    :param children: List of the children.
    :type children: list(RTAPhaseTree)
    """

    def __init__(self, children):
        self.children = list(children)

    def __getitem__(self, key):
        names = [
            (child.get('name', ''), child)
            for child in self.children
        ]

        grouped = dict(groupby(names, key=itemgetter(0)))
        grouped.pop('', None)
        children = list(grouped[key])

        if not children:
            raise KeyError(f'No child named "{key}"')
        if len(children) > 1:
            raise ValueError(f'Multiple children have the same name: {key}')
        else:
            (_, child), = children
            return child

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)


class RTAPhaseTree(_RTAPhaseTreeBase):
    """
    Tree node in an :class:`_RTAPhaseTreeBase`.

    :param children: List of children phases.
    :type children: list(_RTAPhaseTreeBase)

    :param properties: Forwarded to base class.
    :Variable keyword arguments: Forwarded to base class.

    {prop_kwargs}

    The properties set on this node will be combined of the properties of the
    children in :meth:`topo_sort`.
    """

    def __init__(self, properties=None, children=None, **kwargs):
        children = tuple(children or [])
        self._children = children
        super().__init__(properties=properties, **kwargs)

        # pre-compute the memoized property ahead of time, as it will
        # recursively compute its grand-children. This can lead to
        # RecursionError if it's not done when the object is created.
        self.children

    def __str__(self):
        sep = '\n'
        try:
            name = self['name']
        except KeyError:
            name = ''
            idt = ''
        else:
            idt = ' ' * 4
            name = f'Phase {name}:\n'

        sep += idt

        children = ('\n' + sep).join(
            str(child).replace('\n', sep)
            for child in self._renamed_children
        )
        return f'{name}{idt}{children}'

    @property
    def is_empty(self):
        return not self.children

    def _update_children(self, children):
        return [
            child.with_phase_properties(self.properties)
            for child in children
        ]

    @property
    @memoized
    def children(self):
        """
        Tree levels are transparent and their children expanded directly in
        their parent, as long as they have no properties on their own that
        could change the output of :meth:`topo_sort()`. This allows nested
        :class:`RTAPhaseTree` to act as if it was just a flat node, which is
        useful since repeated composition with ``+`` operator will give nested
        binary trees like that.
        """
        def expand(phase):
            # We can expand the children of the phase into their grandparent if
            # and only if the phase has no impact on its children (apart from
            # deleted/overriden properties)
            if isinstance(phase, self.__class__) and not phase.properties.existing_properties:
                # Still apply the properties, as there could be some
                # properties to override or delete
                return phase._update_children(phase.children)
            # Hide completely empty children here, so that they don't even
            # appear in RTAPhaseBase.phases property. This ensures consistency
            # with the JSON content
            elif phase.is_empty:
                return []
            else:
                return [phase]

        return RTAPhaseTreeChildren(
            children=chain.from_iterable(
                expand(child)
                for child in self._children
            )
        )

    @property
    @memoized
    def _renamed_children(self):
        children = self.children

        one_child = len(children) == 1
        def update_name(state, child):
            children, i, names = state

            # Add a default name, to avoid inheriting from the parent the exact
            # same name
            try:
                name = child['name']
            except KeyError:
                # If we only have one child, it's safe for it to inherit
                # from the name of it's parent. This avoids having a
                # trailing ".../0" in all names, since leaves will be
                # considered "children of themselves".
                if one_child:
                    name = self.get('name')
                else:
                    name = str(i)
                    child = child.with_props(name=name)
                    i += 1

            if name in names:
                raise ValueError(f'Two children cannot have the same name "{name}" and share the same parent')
            names.add(name)

            return (children + [child], i, names)

        if children:
            children, *_ = fold(update_name, children, init=([], 0, set()))

        return self._update_children(children)

    def topo_sort(self):
        """
        Topological sort of the tree, and combine the properties along each
        path from root to leaves at the same time.
        """
        # Update the properties before recursing, so that the order of aggregation is:
        # (((root) & child) & subchild)
        # Instead of:
        # (root & (child & (subchild)))

        # Only assign a number to unnamed phases, so that adding a named phase
        # to the mix does not change the named of unnamed ones.
        return list(chain.from_iterable(
            child.topo_sort()
            for child in self._renamed_children
        ))


class ParametricPhase(RTAPhaseTree):
    """
    Base class for phases with special behavior beyond their properties.

    :param template: Template phase used to create children.
    :type template: RTAPhaseBase

    :param properties: Properties to set for that phase.
    :type properties: dict(str, object)

    {prop_kwargs}

    :Variable keyword arguments: Forwarded to :meth:`_make_children`.

    Extra behaviour is enabled by allowing this phase to have multiple children
    created based on the parameters.
    """

    DEFAULT_PHASE_CLS = RTAPhase
    """
    If no template is passed, an instance of this class will be used as
    template.
    """

    def __init__(self, template=None, properties=None, **kwargs):
        properties, other_kwargs = self.split_prop_kwargs(kwargs, properties)

        template = self.DEFAULT_PHASE_CLS() if template is None else template
        children = self._make_children(
            template=template,
            **other_kwargs
        )
        super().__init__(
            properties=properties,
            children=children,
        )

    @classmethod
    @abc.abstractmethod
    def _make_children(cls, template, **kwargs):
        """
        :meta public:

        Create a list of children :class:`RTAPhaseBase` based on the parameters
        passed from the constructor.
        """


class SweepPhase(ParametricPhase):
    """
    Parametric phase creating children by setting the property ``key`` to
    values found in ``values``, in order.

    :param key: Property to set.
    :type key: str

    :param values: Values to set the property to.
    :type values: list(object)

    {prop_kwargs}
    """
    @classmethod
    def _make_children(cls, template, *, key, values):
        return [
            template.with_properties_map({
                key: i,
            })
            for i in values
        ]


class DutyCycleSweepPhase(SweepPhase):
    """
    Sweep on the ``duty_cycle_pct`` parameter of a :class:`PeriodicWload`.

    :param template: Template phase to use.
    :type template: RTAPhaseBase

    :param period: See :class:`PeriodicWload`
    :param duration: See :class:`PeriodicWload`

    :param duration_of: If ``"total"``, the ``duration`` will be used as the
        total duration of the sweep. If ``"step"``, it will be the duration of
        a single step of the sweep.
    :type duration_of: str

    {prop_kwargs}

    :Variable keyword arguments: Forwarded to :func:`lisa.utils.value_range` to
        generate the ``duty_cycle_pct`` values.
    """
    @classmethod
    def _make_children(cls, template, *, period, duration, duration_of=None, **kwargs):

        dc_values = list(value_range(**kwargs, inclusive=True, clip=True))

        duration_of = duration_of or 'total'
        if duration_of == 'step':
            phase_duration = duration
        elif duration_of == 'total':
            phase_duration = duration / len(dc_values)
        else:
            raise ValueError(f'Illegal value "{duration_of}" for "duration_of"')

        values = PeriodicWload.from_product(
            duty_cycle_pct=dc_values,
            period=[period],
            duration=[phase_duration],
        )
        return super()._make_children(template, key='wload', values=values)


################################################################################
# Deprecated classes
################################################################################


class Phase(RTAPhase):
    """
    Descriptor for an rt-app load phase

    :param duration_s: the phase duration in [s].
    :type duration_s: float

    :param period_ms: the phase period in [ms].
    :type period_ms: float

    :param duty_cycle_pct: the generated load in percents.
    :type duty_cycle_pct: float

    :param cpus: the CPUs on which task execution is restricted during this phase.
        If unspecified, that phase will be allowed to run on any CPU,
        regardless of the affinity of the previous phases.
    :type cpus: list(int) or None

    :param barrier_after: if provided, the name of the barrier to sync against
                          when reaching the end of this phase. Currently only
                          supported when duty_cycle_pct=100
    :type barrier_after: str

    :param uclamp_min: the task uclamp.min value to set for the task for the
      duration of the phase.
    :type uclamp_min: int

    :param uclamp_max: the task uclamp.max value to set for the task for the
      duration of the phase.
    :type uclamp_max: int

    :param numa_nodes_membind: the list of NUMA Nodes.
        Task will only allocate memory from these nodes during this phase.
        If unspecified, that phase will be allowed to allocate memory from any
        NUMA node, regardless of the previous phase settings.
    :type numa_nodes_membind: list(int) or None
    """

    def __init__(self, duration_s, period_ms, duty_cycle_pct, cpus=None, barrier_after=None,
                 uclamp_min=None, uclamp_max=None, numa_nodes_membind=None, **kwargs):
        if barrier_after and duty_cycle_pct != 100:
            # This could be implemented but currently don't foresee any use.
            raise ValueError('Barriers only supported when duty_cycle_pct=100')

        # Since Phase used to be the kitchen sink class, it is sometimes used
        # with parameter values that are in themselves invalid, but sort of ok
        # if you make assumptions on what is generated exactly.
        if duty_cycle_pct in (0, 100):
            # The value won't matter as it will be translated to either a pure
            # "run" or "sleep" event
            if period_ms is None:
                period_ms = duration_s * 1e3
            # Avoid triggering an exception because of invalid period
            elif period_ms > duration_s:
                period_ms = duration_s

        wload = PeriodicWload(
            duration=duration_s,
            period=period_ms * 1e-3,
            duty_cycle_pct=duty_cycle_pct,
        )

        if barrier_after:
            wload = wload + BarrierWload(barrier_after)

        super().__init__(
            prop_uclamp=(uclamp_min, uclamp_max),
            prop_wload=wload,
            prop_cpus=cpus,
            prop_numa_nodes_membind=numa_nodes_membind,
            **kwargs,
        )

        self.cpus = cpus
        self.duration_s = duration_s
        self.period_ms = period_ms
        self.duty_cycle_pct = duty_cycle_pct
        self.barrier_after = barrier_after
        self.uclamp_min = uclamp_min
        self.uclamp_max = uclamp_max
        self.numa_nodes_membind = numa_nodes_membind


class _RTATask(RTAPhaseTree):
    def __init__(self, delay_s=0, loops=1, sched_policy=None, priority=None, children=None, **kwargs):
        if loops < 0:
            raise ValueError(f'loops={loops} is not supported anymore, only positive values can be used')

        # Add some attributes for the show. They are only there for client code
        # to inspect, but don't have any actual effect.
        self._delay_s = delay_s
        self._loops = loops
        self._sched_policy = sched_policy
        self._priority = priority

        children = loops * list(children or [])

        sched_policy = f'SCHED_{sched_policy}' if sched_policy else None
        if delay_s:
            delay_phase = RTAPhase(
                prop_wload=SleepWload(delay_s),
                prop_name='delay',
            )
            children = [delay_phase] + children

        super().__init__(
            children=children,
            prop_policy=sched_policy,
            prop_priority=priority,
            **kwargs,
        )

    # Use property() so that the attributes are read-only. This is needed to
    # catch client code expecting to get a different JSON by mutating the
    # attributes, which is not the case.
    @property
    def delay_s(self):
        return self._delay_s

    @property
    def loops(self):
        return self._loops

    @property
    def sched_policy(self):
        return self._sched_policy

    @property
    def priority(self):
        return self._priority


@deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=RTAPhase)
class RTATask(_RTATask):
    """
    Base class for conveniently constructing params to :meth:`RTA.from_profile`

    :param delay_s: the delay in seconds before starting.
    :type delay_s: float

    :param loops: Number of times to repeat the described task.
    :type loops: int

    :param sched_policy: the scheduler policy for this task. Defaults to
      ``SCHED_OTHER``, see :manpage:`sched` for information on scheduler policies.
    :type sched_policy: str or None

    :param priority: the scheduler priority for this task. See :manpage:`sched`
      for information on scheduler priorities.
    :type priority: int or None

    This class represents an rt-app task which may contain multiple :class:`Phase`.
    It implements ``__add__`` so that using ``+`` on two tasks concatenates their
    phases. For example ``Ramp() + Periodic()`` would yield an ``RTATask`` that
    executes the default phases for :class:`Ramp` followed by the default phases for
    :class:`Periodic`.
    """


class _Ramp(_RTATask):
    def __init__(self, start_pct=0, end_pct=100, delta_pct=10, time_s=1,
                 period_ms=100, delay_s=0, loops=1, sched_policy=None,
                 priority=None, cpus=None, uclamp_min=None, uclamp_max=None,
                 numa_nodes_membind=None, **kwargs):

        if not (0 <= start_pct <= 100 and 0 <= end_pct <= 100):
            raise ValueError('start_pct and end_pct must be in [0..100] range')

        children = [
            Phase(
                duration_s=time_s,
                period_ms=0 if load == 0 else period_ms,
                duty_cycle_pct=load,
                uclamp_min=uclamp_min,
                uclamp_max=uclamp_max,
                numa_nodes_membind=numa_nodes_membind,
                cpus=cpus,
            )
            for load in value_range(
                start=start_pct,
                stop=end_pct,
                step=delta_pct,
                clip=True,
                inclusive=True,
            )
        ]

        if not children:
            raise ValueError('No phase created')

        super().__init__(
            children=children,
            delay_s=delay_s,
            loops=loops,
            sched_policy=sched_policy,
            priority=priority,
            **kwargs,
        )


@deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=DutyCycleSweepPhase)
class Ramp(_Ramp):
    """
    Configure a ramp load.

    This class defines a task which load is a ramp with a configured number
    of steps according to the input parameters.

    :param start_pct: the initial load percentage.
    :type start_pct: float

    :param end_pct: the final load percentage.
    :type end_pct: float

    :param delta_pct: the load increase/decrease at each step, in percentage
      points.
    :type delta_pct: float

    :param time_s: the duration in seconds of each load step.
    :type time_s: float

    :param period_ms: the period used to define the load in [ms].
    :type period_ms: float

    .. seealso:: See :class:`RTATask` for the documentation of the following
      parameters:

      * **delay_s**
      * **loops**
      * **sched_policy**
      * **priority**

    .. seealso:: See :class:`Phase` for the documentation of the following
      parameters:

      * **cpus**
      * **uclamp_min**
      * **uclamp_max**
      * **numa_nodes_membind**
    """


@deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=DutyCycleSweepPhase)
class Step(_Ramp):
    """
    Configure a step load.

    This class defines a task which load is a step with a configured initial and
    final load. Using the ``loops`` param, this can be used to create a workload
    that alternates between two load values.

    :param start_pct: the initial load percentage.
    :type start_pct: float

    :param end_pct: the final load percentage.
    :type end_pct: float

    :param time_s: the duration in seconds of each load step.
    :type time_s: float

    :param period_ms: the period used to define the load in [ms].
    :type period_ms: float

    .. seealso:: See :class:`RTATask` for the documentation of the following
      parameters:

      * **delay_s**
      * **loops**
      * **sched_policy**
      * **priority**

    .. seealso:: See :class:`Phase` for the documentation of the following
      parameters:

      * **cpus**
      * **uclamp_min**
      * **uclamp_max**
      * **numa_nodes_membind**
    """

    def __init__(self, start_pct=0, end_pct=100, time_s=1, period_ms=100,
                 delay_s=0, loops=1, sched_policy=None, priority=None, cpus=None,
                 uclamp_min=None, uclamp_max=None, numa_nodes_membind=None, **kwargs):
        delta_pct = abs(end_pct - start_pct)
        super().__init__(
            start_pct=start_pct,
            end_pct=end_pct,
            delta_pct=delta_pct,
            time_s=time_s,
            period_ms=period_ms,
            delay_s=delay_s,
            loops=loops,
            sched_policy=sched_policy,
            priority=priority,
            cpus=cpus,
            uclamp_min=uclamp_min,
            uclamp_max=uclamp_max,
            numa_nodes_membind=numa_nodes_membind,
            **kwargs,
        )


class _Pulse(_RTATask):
    def __init__(self, start_pct=100, end_pct=0, time_s=1, period_ms=100,
                 delay_s=0, loops=1, sched_policy=None, priority=None, cpus=None,
                 uclamp_min=None, uclamp_max=None, numa_nodes_membind=None, **kwargs):

        if end_pct > start_pct:
            raise ValueError('end_pct must be lower than start_pct')

        if not (0 <= start_pct <= 100 and 0 <= end_pct <= 100):
            raise ValueError('end_pct and start_pct must be in [0..100] range')

        loads = [start_pct]
        if end_pct:
            loads += [end_pct]

        children = [
            Phase(
                duration_s=time_s,
                period_ms=period_ms,
                duty_cycle_pct=load,
                uclamp_min=uclamp_min,
                uclamp_max=uclamp_max,
                numa_nodes_membind=numa_nodes_membind,
                cpus=cpus,
            )
            for load in loads
        ]

        super().__init__(
            children=children,
            delay_s=delay_s,
            loops=loops,
            sched_policy=sched_policy,
            priority=priority,
            **kwargs,
        )


@deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=RTAPhase)
class Pulse(_Pulse):
    """
    Configure a pulse load.

    This class defines a task which load is a pulse with a configured
    initial and final load.

    The main difference with the 'step' class is that a pulse workload is
    by definition a 'step down', i.e. the workload switch from an initial
    load to a final one which is always lower than the initial one.
    Moreover, a pulse load does not generate a sleep phase in case of 0[%]
    load, i.e. the task ends as soon as the non null initial load has
    completed.

    :param start_pct: the initial load percentage.
    :type start_pct: float

    :param end_pct: the final load percentage.
    :type end_pct: float

    :param time_s: the duration in seconds of each load step.
    :type time_s: float

    :param period_ms: the period used to define the load in [ms].
    :type period_ms: float

    .. seealso:: See :class:`RTATask` for the documentation of the following
      parameters:

      * **delay_s**
      * **loops**
      * **sched_policy**
      * **priority**

    .. seealso:: See :class:`Phase` for the documentation of the following
      parameters:

      * **cpus**
      * **uclamp_min**
      * **uclamp_max**
      * **numa_nodes_membind**
    """


@deprecate('Replaced by :class:`lisa.wlgen.rta.RTAPhase` along with :class:`lisa.wlgen.rta.PeriodicWload` workload', deprecated_in='2.0', removed_in='3.0', replaced_by=RTAPhase)
class Periodic(_Pulse):
    """
    Configure a periodic load. This is the simplest type of RTA task.

    This class defines a task which load is periodic with a configured
    period and duty-cycle.

    :param duty_cycle_pct: the generated load in percents.
    :type duty_cycle_pct: float

    :param duration_s: the phase duration in [s].
    :type duration_s: float

    :param period_ms: the period used to define the load in [ms].
    :type period_ms: float

    .. seealso:: See :class:`RTATask` for the documentation of the following
      parameters:

      * **delay_s**
      * **loops**
      * **sched_policy**
      * **priority**

    .. seealso:: See :class:`Phase` for the documentation of the following
      parameters:

      * **cpus**
      * **uclamp_min**
      * **uclamp_max**
      * **numa_nodes_membind**
    """

    def __init__(self, duty_cycle_pct=50, duration_s=1, period_ms=100,
                 delay_s=0, sched_policy=None, priority=None, cpus=None,
                 uclamp_min=None, uclamp_max=None, numa_nodes_membind=None,
                 **kwargs):
        super().__init__(
            start_pct=duty_cycle_pct,
            end_pct=0,
            time_s=duration_s,
            period_ms=period_ms,
            delay_s=delay_s,
            loops=1,
            sched_policy=sched_policy,
            priority=priority,
            cpus=cpus,
            uclamp_min=uclamp_min,
            uclamp_max=uclamp_max,
            numa_nodes_membind=numa_nodes_membind,
            **kwargs
        )


@deprecate('Replaced by :class:`lisa.wlgen.rta.RTAPhase` along with :class:`lisa.wlgen.rta.RunWload` and :class:`lisa.wlgen.rta.BarrierWload` workloads', deprecated_in='2.0', removed_in='3.0', replaced_by=RTAPhase)
class RunAndSync(_RTATask):
    """
    Configure a task that runs 100% then waits on a barrier

    :param barrier: name of barrier to wait for. Sleeps until any other tasks
      that refer to this barrier have reached the barrier too.
    :type barrier: str

    :param time_s: time to run for in [s]
    :type time_s: float

    .. seealso:: See :class:`RTATask` for the documentation of the following
      parameters:

      * **delay_s**
      * **loops**
      * **sched_policy**
      * **priority**

    .. seealso:: See :class:`Phase` for the documentation of the following
      parameters:

      * **cpus**
      * **uclamp_min**
      * **uclamp_max**
      * **numa_nodes_membind**
    """

    def __init__(self, barrier, time_s=1, delay_s=0, loops=1,
                 sched_policy=None, priority=None, cpus=None, uclamp_min=None,
                 uclamp_max=None, numa_nodes_membind=None, **kwargs):

        # This should translate into a phase containing a 'run' event and a
        # 'barrier' event
        children = [
            Phase(
                duration_s=time_s,
                period_ms=None,
                duty_cycle_pct=100,
                cpus=cpus,
                barrier_after=barrier,
                uclamp_min=uclamp_min,
                uclamp_max=uclamp_max,
                numa_nodes_membind=numa_nodes_membind,
            )
        ]

        super().__init__(
            children=children,
            delay_s=delay_s,
            loops=loops,
            sched_policy=sched_policy,
            priority=priority,
            **kwargs,
        )

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
