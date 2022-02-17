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
import contextlib

from lisa.utils import compose, nullcontext


class _StateInitializer:
    """
    Wrapper for a state-initializing function, along with the underlying
    non-lifted coroutine function so that lifted functions can be composed
    naturally with await.
    """
    def __init__(self, f, coro_f):
        self.f = f
        self.coro_f = coro_f
        functools.update_wrapper(wrapper=self, wrapped=f)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __await__(self):
        return (yield from self.coro_f().__await__())

    def state_init_decorator(self, f):
        """
        Decorator used to decorate wrapper that are initializing the state
        (i.e. calling :class:`_StateInitializer` instances).

        This is necessary in order for resulting values to be awaitable, so
        that composition is preserved.
        """
        return self.__class__(
            f,
            self.coro_f,
        )


def _consume(coro):
    try:
        action = coro.send(None)
    except StopIteration as e:
        return e.value
    else:
        if isinstance(action, StateMonad):
            extra = f'. The top-level function should be decorated with @{action._MONAD_BASE.__qualname__}.lift'
        else:
            extra = ''
        raise TypeError(f'The coroutine could not be consumed as it contains unhandled action: {action}{extra}')
    finally:
        coro.close()


class _RestartableCoro:
    def __init__(self, factory):
        self._factory = factory

    @property
    def coro(self):
        return self._factory()


class StateMonad(abc.ABC):
    """
    The state monad.

    :param f: Callable that takes the state as parameter and returns a tuple
        ``(value, new_state)``.
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

    def __init_subclass__(cls, **kwargs):
        # The one inheriting directly from StateMonad is the base of the
        # hierarchy
        if StateMonad in cls.__bases__:
            cls._MONAD_BASE = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def _process_coroutine_val(cls, val, state):
        """
        Subclasses can override this method to customize the return value of
        the user-defined lifted coroutine function.

        This allows subclasses to use the current state to override the value
        returned by the user.

        :param val: The value actually returned in the user-defined lifted
            coroutine function.
        :type val: object

        :param state: The current state.
        :type state: object
        """
        return val

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


    @staticmethod
    def _loop(_coro, *, state, cls, consume):
        async def factory():
            if isinstance(_coro, _RestartableCoro):
                coro = _coro.coro
            else:
                coro = _coro

            _state = state
            next_ = lambda: coro.send(None)
            while True:
                try:
                    action = next_()
                except StopIteration as e:
                    val = cls._process_coroutine_val(e.value, state)
                    break
                else:
                    is_cls = isinstance(action, cls)
                    try:
                        if is_cls:
                            val, _state = action._f(_state)
                        else:
                            val = await action
                    except Exception as e:
                        # We need an intermediate variable here, since
                        # "e" is not really bound in this scope.
                        excep = e
                        next_ = lambda: coro.throw(excep)
                    else:
                        next_ = lambda: coro.send(val)

            if isinstance(val, cls):
                val, _ = val._f(_state)

            return val

        # Wrap the coroutine in something that can be called to consume it
        # entirely
        if consume:
            return _consume(factory())
        else:
            return _RestartableCoro(factory)

    @classmethod
    def _wrap_coroutine_f(cls, f):
        """
        Decorator used to wrap user-defined coroutine-functions.

        This allows subclasses of :class:`StateMonad` to handle exceptions
        inside user-defined coroutine functions, or do arbitrary other
        processing.
        """
        return f

    @classmethod
    def lift(cls, f):
        """
        Decorator used to lift a coroutine function into the monad.


        The decorated coroutine function can be called to set its parameters
        values, and will return another callable. This callable will take the
        :meth:`StateMon.make_state` method to initialize the state, and will
        then run the computation.

        .. note:: If a coroutine function is decorated with
            :meth:`StateMonad.lift` multiple times for various subclasses, each
            state-initializing callable will return the state-initializing
            callable of the next level in the decorator stack, starting from
            the top.
        """
        cls = cls._MONAD_BASE

        @functools.wraps(f)
        def wrapper(*fargs, **fkwargs):
            @functools.wraps(cls.make_state)
            def make_state_wrapper(*sargs, _state_monad_private_wrap_coro=None, **skwargs):
                _loop = functools.partial(
                    cls._loop,
                    cls=cls,
                    state=cls.make_state(*sargs, **skwargs),
                    # Only ask _loop to consume the coroutine if we are the
                    # top-level state monad in the stack
                    consume=_state_monad_private_wrap_coro is None,
                )

                if  _state_monad_private_wrap_coro is None:
                    _state_monad_private_wrap_coro = lambda x: x

                wrap_coro = compose(cls._wrap_coroutine_f, _state_monad_private_wrap_coro)

                # We found the inner user-defined coroutine, so we just wrap it
                # with the loop
                if wrapper._state_monad_is_bottom:
                    return _loop(
                        _RestartableCoro(
                            lambda: wrap_coro(f)(*fargs, **fkwargs)
                        ),
                    )
                # If we are lifting an already-lifted function, we wrap with
                # our loop
                else:
                    def loop_wrapper(*args, **kwargs):
                        return _loop(
                            f(*fargs, **fkwargs)(
                                *args,
                                **kwargs,
                                _state_monad_private_wrap_coro=wrap_coro,
                            ),
                        )
                    return loop_wrapper

            return _StateInitializer(
                make_state_wrapper,
                # Provide the top-most non lifted function in the decorator
                # stack, so we can use it to await from it directly when
                # composing lifted functions.
                functools.partial(
                    wrapper._state_monad_coro_f,
                    *fargs,
                    **fkwargs,
                )
            )

        def find_user_f(f):
            """
            Find the top-most non lifted function in the decorator stack.
            """
            _f = f
            while True:
                # If we find a lifted function, we just pick it from there
                try:
                    return (_f._state_monad_coro_f, False)
                except AttributeError:
                    pass

                try:
                    _f = _f.__wrapped__
                except AttributeError:
                    break

            # If we could not find any lifted function, it means we are the
            # bottom-most decorator in the stack and we can just take what we
            # are given directly
            return (f, True)

        # We wrap the coroutine function so that layers will accumulate and no
        # _wrap_coroutine_f() will be missed
        user_f, is_bottom = find_user_f(f)
        wrapper._state_monad_coro_f = cls._wrap_coroutine_f(user_f)
        wrapper._state_monad_is_bottom = is_bottom
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
