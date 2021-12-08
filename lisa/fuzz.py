# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, ARM Limited and contributors.
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

"""
Fuzzing API to build random constrained values.

**Example**::

    import operator
    import functools

    from lisa.platforms.platinfo import PlatformInfo
    from lisa.wlgen.rta import RTAPhase, RTAConf, PeriodicWload
    from lisa.fuzz import Gen, Choice, Int, Float, retry_until

    # The function must be decorated with Gen.lift() so that "await" gains its
    # special meaning. In addition to that, parameters are automatically awaited if
    # they are an instance of Gen, and the return value is automatically promoted
    # to an instance of Gen if it is not already.
    @Gen.lift
    async def make_task(duration=None):
        # Draw a value from an iterable.
        period = await Choice([16e-3, 8e-3])
        nr = await Choice(range(1, 4))
        duration = duration or (await Float(1, 2))

        # Arbitrary properties can be enforced. If they are not satisfied, the
        # function will run again until the condition is true.
        retry_until(0 < nr <= 2)

        phase = functools.reduce(
            operator.add,
            [
                RTAPhase(
                    prop_wload=PeriodicWload(
                        duty_cycle_pct=await Choice(range(100)),
                        period=period,
                        duration=duration,
                    ),
                )
                for i in range(nr)
            ]
        )

        return phase

    @Gen.lift
    async def make_profile(plat_info, **kwargs):
        nr_tasks = await Int(1, plat_info['cpus-count'])

        profile = {}
        for i in range(nr_tasks):
            profile[f'task{i}'] = await make_task(**kwargs)
        return profile


    def main():
        plat_info = PlatformInfo.from_yaml_map('./doc/traces/plat_info.yml')

        # When called, profile_gen() will create a random profiles
        profile_gen = make_profile(plat_info, duration=1)

        # Display a few randomly generated tasks
        for _ in range(2):
            # seed (or rng) can be fixed for reproducible results
            # profile = profile_gen(seed=1)
            profile = profile_gen(seed=None)

            conf = RTAConf.from_profile(profile, plat_info=plat_info)
            print(conf.json)

    main()
"""

import random
import functools
import itertools
import inspect
import logging
from operator import attrgetter
from collections.abc import Iterable, Mapping

from lisa.monad import StateMonad
from lisa.utils import Loggable


class RetryException(Exception):
    """
    Exception raised to signify to :class:`lisa.fuzz.Gen` to retry the random
    draw.

    .. seealso:: :func:`lisa.fuzz.retry_until`
    """
    pass


def retry_until(cond):
    """
    If ``cond`` is ``True``, signify to the :class:`lisa.fuzz.Gen` monad to
    retry the computation. This is used to enforce constraints on the output.

    .. note:: If possible, it's a better idea to generate the data in a way
        that satisfy the constraints, as retrying can happen an arbitrary
        number of time and thus become quite costly.
    """
    if not cond:
        raise RetryException()


class Gen(StateMonad, Loggable):
    """
    Random generator monad inspired by Haskell's QuickCheck.
    """
    def __init__(self, f, name=None):
        log_level = logging.DEBUG
        logger = self.logger
        if logger.isEnabledFor(log_level):
            caller_info = inspect.stack()[2]
        else:
            caller_info = None

        @functools.wraps(f)
        def wrapper(state):
            for i in itertools.count(1):
                try:
                    x = f(state)
                except RetryException:
                    continue
                else:
                    trials = f'after {i} trials ' if i > 1 else ''
                    if caller_info:
                        info =  f' ({caller_info.filename}:{caller_info.lineno})'
                    else:
                        info = ''
                    val, _ = x
                    val = str(val)
                    sep = '\n' + ' ' * 4
                    val = sep + val.replace('\n', sep) + '\n' if '\n' in val else val + ' '
                    self.logger.log(log_level, f'Drawn {val}{trials}from {self}{info}')
                    return x

        self.name = name
        super().__init__(wrapper)

    class _STATE:
        def __init__(self, rng):
            self.rng = rng

    @classmethod
    def make_state(cls, *, rng=None, seed=None):
        return cls._STATE(
            rng=rng or random.Random(seed),
        )

    def __str__(self):
        name = self.name or self._f.__qualname__
        return f'{self.__class__.__qualname__}({name})'


class Choices(Gen):
    """
    Randomly choose ``n`` values among ``xs``.

    :param n: Number of values to yield every time.
    :type n: int

    :param xs: Finite iterable of values to choose from.
    :type xs: collections.abc.Iterable

    :param typ: Callable used to build the output from an iterable.
    :type typ: type
    """
    _TYP = list
    _RANDOM_METH = attrgetter('choices')

    def __init__(self, n, xs, typ=None):
        self._xs_str = str(xs)
        typ = typ or self._TYP
        xs = list(xs)
        if not n or n <= 0:
            raise ValueError(f'n must be > 0: {n}')

        super().__init__(
            lambda state: (typ(self._RANDOM_METH(state.rng)(xs, k=n)), state),
        )

    def __str__(self):
        return f'{self.__class__.__qualname__}({self._xs_str})'

class Set(Choices):
    """
    Same as :class:`lisa.fuzz.Choices` but returns a set.

    .. note:: The values are drawn without replacement to ensure the set is of
        the correct size, assuming the input contained no duplicate.
    """
    _TYP = set
    _RANDOM_METH = attrgetter('sample')


class Tuple(Choices):
    """
    Same as :class:`lisa.fuzz.Choices` but returns a tuple.
    """
    _TYP = tuple


class SortedList(Choices):
    """
    Same as :class:`lisa.fuzz.Choices` but returns a sorted list.
    """
    _TYP = sorted


class Shuffle(Choices):
    """
    Randomly shuffle the given sequence.

    :param xs: Finite sequence of values to shuffle.
    :type xs: collections.abc.Sequence
    """
    _RANDOM_METH = attrgetter('sample')

    def __init__(self, xs):
        typ = type(xs)
        xs = list(xs)

        super().__init__(
            xs=xs,
            typ=typ,
            n=len(xs),
        )


class Bool(Gen):
    """
    Draw a random bool.
    """
    def __init__(self):
        super().__init__(
            lambda state: (bool(state.rng.randint(0, 1)), state),
        )

    def __str__(self):
        return f'{self.__class__.__qualname__}()'


class Int(Gen):
    """
    Draw a random int fitting within the ``[min_, max_]`` range.
    """
    def __init__(self, min_=0, max_=0):
        self.min_ = min_
        self.max_ = max_
        super().__init__(
            lambda state: (state.rng.randint(min_, max_), state),
        )

    def __str__(self):
        return f'{self.__class__.__qualname__}({self.min_} <= x <= {self.max_})'


class Float(Gen):
    """
    Draw a random float fitting within the ``[min_, max_]`` range.
    """
    def __init__(self, min_=0, max_=0):
        self.min_ = min_
        self.max_ = max_
        super().__init__(
            lambda state: (state.rng.uniform(min_, max_), state),
        )

    def __str__(self):
        return f'{self.__class__.__qualname__}({self.min_} <= x <= {self.max_})'


class Dict(Choices):
    """
    Same as :class:`lisa.fuzz.Choices` but returns a dictionary.

    .. note:: The input must be an iterable of ``tuple(key, value)``.

    .. note:: The values are drawn without replacement to ensure the dict is of
        the correct size, assuming the input contained no duplicate.
    """
    _TYP = dict
    _RANDOM_METH = attrgetter('sample')

    def __init__(self, n, xs, typ=None):
        if isinstance(xs, Mapping):
            xs = xs.items()

        super().__init__(
            n=n,
            xs=xs,
            typ=typ,
        )


class Choice(Gen):
    """
    Randomly choose one values among ``xs``.

    :param xs: Finite iterable of values to choose from.
    :type xs: collections.abc.Iterable
    """
    def __init__(self, xs):
        self._xs_str = str(xs)
        xs = list(xs)
        super().__init__(
            lambda state: (state.rng.choice(xs), state),
        )

    def __str__(self):
        return f'{self.__class__.__qualname__}({self._xs_str})'

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
