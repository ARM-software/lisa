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
Monads with syntactic sugar.

All monads share the following API::

    # for a given monad Monad

    # Turns a regular function into a function returning an instance of Monad,
    # and able to consume monadic values. Similar to liftM in Haskell.
    @Monad.lift
    async def foo(x, y):
        # Inside a decorated function, await can be used to "extract" the value
        # contained in the monad, like ``<-`` in Haskell.
        z = await Monad.something()
        return x

    # Equivalent to
    @Monad.lift
    async def foo(x, y):
        # note: await is used automatically if x is an instance of Monad
        x_ = await x
        y_ = await y
        # Monad.pure() is called if x_ is not an instance of Monad
        return Monad.pure(x_)

This allow composing lifted functions easily

.. note:: There currently is no overridable ``bind`` operation, since nothing
    in Python currently allows getting a proper continuation without explicit
    manipulations of lambdas. The closest thing that is used is coroutine
    functions, where ``await`` somewhat provides a continuation using
    ``coroutine.send()``. The limitation comes from that it can only be called
    at most once (preventing anything like the list monad). Early-return
    control flow such as the maybe monad are typically not necessary as Python
    has exceptions already.

.. note:: ``async/await`` is used as syntactic sugar instead of ``yield`` since
    the grammar works better for ``await``. ``yield`` cannot be used in
    comprehensions, which prevents some idiomatic functional patterns based on
    generator expressions.
"""

import abc
import functools
import inspect


class StateMonad(abc.ABC):
    """
    The state monad.

    :param f: Callable that takes the state as parameter and returns an
        instance of the monad.
    :type f: collections.abc.Callable

    """

    def __init__(self, f):
        self._f = f

    def __await__(self):
        # What happens here is executed as if the code was inlined in the
        # coroutine's body ("await x" is actually equivalent to
        # "yield from x"):
        # 1. "yield self" allows to relinquish control back to the loop,
        #    providing the monadic value that was awaited on by the user code.
        # 2. Returning the result of the yield allows the loop to inject any
        #    value it sees fit using coro.send().
        return (yield self)

    def __call__(self, *args, **kwargs):
        state = self.make_state(*args, **kwargs)
        x, _ = self._f(state)
        return x

    def __init_subclass__(cls, **kwargs):
        # The one inheriting directly from StateMonad is the base of the
        # hierarchy
        if StateMonad in cls.__bases__:
            cls._MONAD_BASE = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_f(cls, *args, **kwargs):
        """
        Build an instance of the monad from a state transformation function.
        The callback takes the current state as parameter and returns
        ``tuple(value, new_state)``.
        """
        return cls._MONAD_BASE(*args, **kwargs)

    @abc.abstractclassmethod
    def make_state(cls, *args, **kwargs):
        """
        Create the state from user-defined parameters. This is used by
        :meth:`lisa.monad.StateMonad.__call__` in order to initialize the
        state.
        """
        pass

    @classmethod
    def pure(cls, x):
        """
        Lift a value in the state monad, i.e. create a monad instance with a
        function that returns the value and the state unchanged.
        """
        return cls.from_f(lambda state: (x, state))

    @classmethod
    def lift(cls, f):
        """
        Decorator used to lift a function into the monad, such that it can take
        monadic parameters that will be evaluated in the current state, and
        returns a monadic value as well.
        """

        cls = cls._MONAD_BASE

        def run(_f, args, kwargs):
            call = lambda: _f(*args, **kwargs)
            x = call()
            if inspect.iscoroutine(x):
                def body(state):
                    if inspect.getcoroutinestate(x) == inspect.CORO_CLOSED:
                        _x = call()
                    else:
                        _x = x

                    next_ = lambda: _x.send(None)
                    while True:
                        try:
                            future = next_()
                        except StopIteration as e:
                            val = e.value
                            break
                        else:
                            assert isinstance(future, cls)
                            try:
                                val, state = future._f(state)
                            except Exception as e:
                                # We need an intermediate variable here, since
                                # "e" is not really bound in this scope.
                                excep = e
                                next_ = lambda: _x.throw(excep)
                            else:
                                next_ = lambda: _x.send(val)

                    if isinstance(val, cls):
                        return val._f(state)
                    else:
                        return (val, state)

                val = cls.from_f(body, name=f.__qualname__)
            else:
                if isinstance(x, cls):
                    val = x
                else:
                    val = cls.pure(x)

            return val

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            async def _f(*args, **kwargs):
                args = [
                    (await arg) if isinstance(arg, cls) else arg
                    for arg in args
                ]
                kwargs = {
                    k: (await v) if isinstance(v, cls) else v
                    for k, v in kwargs.items()
                }
                return run(f, args, kwargs)
            return run(_f, args, kwargs)

        return wrapper

    @classmethod
    def get_state(cls):
        """
        Returns a monadic value making the current state available.
        To be used inside a lifted function using::

            state = await StateMonad.get_state()
        """
        return cls.from_f(lambda state: (state, state))

    @classmethod
    def set_state(cls, new_state):
        """
        Returns a monadic value to set the current state.
        To be used inside a lifted function using::

            await StateMonad.set_state(new_state)
        """
        return cls.from_f(lambda state: (state, new_state))

    @classmethod
    def modify_state(cls, f):
        """
        Returns a monadic value to modify the current state.
        To be used inside a lifted function using::

            await StateMonad.modify_state(lambda state: new_state)
        """
        def _f(state):
            new_state = f(state)
            return (new_state, new_state)
        return cls.from_f(_f)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
