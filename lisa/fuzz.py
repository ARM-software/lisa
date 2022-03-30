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

.. note:: The following example shows a direct use of the :class:`Gen` monad,
    but be aware that :mod:`lisa.wlgen.rta` API allows mixing both :class:`Gen`
    and RTA DSL into the same coroutine function.

**Example**::

    import operator
    import functools

    from lisa.platforms.platinfo import PlatformInfo
    from lisa.wlgen.rta import RTAPhase, RTAConf, PeriodicWload
    from lisa.fuzz import GenMonad, Choice, Int, Float, retry_until

    # The function must be decorated with GenMonad.do() so that "await" gains its
    # special meaning.
    @GenMonad.do
    async def make_task(duration=None):
        # Draw a value from an iterable.
        period = await Choice([16e-3, 8e-3])
        nr = await Choice(range(1, 4))
        duration = duration or (await Float(1, 2))

        # Arbitrary properties can be enforced. If they are not satisfied, the
        # function will run again until the condition is true.
        await retry_until(0 < nr <= 2)

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

    @GenMonad.do
    async def make_profile(plat_info, **kwargs):
        nr_tasks = await Int(1, plat_info['cpus-count'])

        profile = {}
        for i in range(nr_tasks):
            profile[f'task{i}'] = await make_task(**kwargs)
        return profile


    def main():
        plat_info = PlatformInfo.from_yaml_map('./doc/traces/plat_info.yml')

        # Display a few randomly generated tasks
        for _ in range(2):
            # When called, profile_gen() will create a random profiles
            profile_gen = make_profile(plat_info, duration=1)

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
from operator import attrgetter, itemgetter
from collections.abc import Iterable, Mapping

from lisa.monad import StateDiscard
from lisa.utils import Loggable, deprecate


class RetryException(Exception):
    """
    Exception raised to signify to :class:`lisa.fuzz.Gen` to retry the random
    draw.

    .. seealso:: :func:`lisa.fuzz.retry_until`
    """
    pass


class _Retrier:
    def __init__(self, cond):
        self.cond = cond

    def __await__(self):
        if self.cond:
            return
        else:
            raise RetryException()
        # Ensures __await__ is a generator function
        yield


def retry_until(cond):
    """
    Returns an awaitable that will signify to the :class:`lisa.fuzz.Gen` monad
    to retry the computation until ``cond`` is ``True``. This is used to
    enforce arbitrary constraints on generated data.

    .. note:: If possible, it's a better idea to generate the data in a way
        that satisfy the constraints, as retrying can happen an arbitrary
        number of time and thus become quite costly.
    """
    return _Retrier(cond)


class GenMonad(StateDiscard, Loggable):
    """
    Random generator monad inspired by Haskell's QuickCheck.
    """
    def __init__(self, f, name=None):
        self.name = name or f.__qualname__
        super().__init__(f)

    class _State:
        def __init__(self, rng):
            self.rng = rng

    @classmethod
    def make_state(cls, *, rng=None, seed=None):
        """
        Initialize the RNG state with either an rng or a seed.

        :param seed: Seed to initialize the :class:`random.Random` instance.
        :type seed: object

        :param rng: Instance of RNG.
        :type rng: random.Random
        """
        return cls._State(
            rng=rng or random.Random(seed),
        )

    def __str__(self):
        name = self.name or self._f.__qualname__
        return f'{self.__class__.__qualname__}({name})'

    @classmethod
    def _decorate_coroutine_function(cls, f):
        _f = super()._decorate_coroutine_function(f)

        @functools.wraps(_f)
        async def wrapper(*args, **kwargs):
            for i in itertools.count(1):
                try:
                    x = await _f(*args, **kwargs)
                except RetryException:
                    continue
                else:
                    trials = f'after {i} trials ' if i > 1 else ''
                    val = str(x)
                    sep = '\n' + ' ' * 4
                    val = sep + val.replace('\n', sep) + '\n' if '\n' in val else val + ' '
                    cls.get_logger().debug(f'Drawn {val}{trials}from {_f.__qualname__}')
                    return x

        return wrapper


class Gen:
    def __init__(self, *args, **kwargs):
        self._action = GenMonad(*args, **kwargs)

    def __await__(self):
        return (yield from self._action.__await__())

    @classmethod
    @deprecate(deprecated_in='2.0', removed_in='3.0', replaced_by=GenMonad.do,
        msg='Note that GenMonad.do() will not automatically await on arguments if they are Gen instances, this must be done manually.',
    )
    def lift(cls, f):

        @GenMonad.do
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            args = [
                (await arg) if isinstance(arg, cls) else arg
                for arg in args
            ]
            kwargs = {
                k: (await v) if isinstance(v, cls) else v
                for k, v in kwargs.items()
            }
            return await f(*args, **kwargs)

        return wrapper


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
