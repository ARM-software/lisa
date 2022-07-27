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
    # and able to await on monadic values. Similar to the "do notation" in
    # Haskell.
    @Monad.do
    async def foo(x, y):
        # Inside a decorated function, await can be used to "extract" the value
        # contained in the monad, like ``<-`` in Haskell.
        z = await Monad.something()
        return x

    # Equivalent to
    @Monad.do
    async def foo(x, y):
        # note: await is used automatically if x is an instance of Monad
        x_ = await x
        y_ = await y
        # Monad.pure() is called if x_ is not an instance of Monad
        return Monad.pure(x_)

This allow composing decorated functions easily

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
import itertools
import functools
import inspect
import operator
import asyncio
from operator import attrgetter
from functools import partial
from weakref import WeakKeyDictionary

import nest_asyncio
nest_asyncio.apply()

from lisa.utils import memoized, foldr, instancemethod


class _MonadBase:
    """
    Abstract Base Class parametrized by a type T,
    like a container of some sort.
    """

    @abc.abstractclassmethod
    def pure(cls, x):
        """
        Takes a value of type T and turns it into a "monadic value" of type Monad[T].
        """
        pass

    @abc.abstractmethod
    def bind(self, continuation):
        """
        Takes a monadic value Monad[A], a function that takes an A and returns Monad[B], and returns a Monad[B].

        .. note:: It is allowed to return a :class:`_TailCall` instance.
        """
        pass

    @abc.abstractmethod
    def map(self, f):
        """
        Takes a monadic value Monad[A], a function that takes an A and returns B, and returns a Monad[B].
        """
        pass

    @abc.abstractmethod
    def join(self):
        """
        Takes a monadic value Monad[Monad[A]], and returns a Monad[A].
        """
        pass

    def __await__(self):
        return (yield self)


class Monad(_MonadBase):
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        def wrap_bind(f):
            @functools.wraps(f)
            def wrapper(self, continuation):
                # Ensure that all the parameters are resolved before use, so
                # that implementations don't have to be sprinkled with
                # _TailCall.run()
                run = _TailCall.run
                return f(
                    run(self),
                    _TailCall._trampoline(run(continuation))
                )
            return wrapper

        cls.bind = wrap_bind(cls.bind)

    @instancemethod
    def map(cls, self, f):
        return cls.bind(self, lambda x: self.pure(f(x)))

    @instancemethod
    def join(cls, self):
        return cls.bind(self, lambda x: x)


# Inherit from _MonadBase directly to avoid the automatic trampoline on bind()
# that would lead to infinite recursion
class _Identity(_MonadBase):
    """
    This monad is private as it is not "proper".

    It does not actually wrap the value, which means that the only way e.g.
    bind() can be called is when referring to the class directly, since no
    instance of :class:`_Identity` will be created.

    This does have the advantage of avoiding unnecessary boxing, which improves
    performance and also avoids a layer of indirection when inspecting the
    resulting values.
    """
    def __new__(cls, x):
        return x

    def bind(self, continuation):
        return _TailCall(_TailCall.run(continuation), _TailCall.run(self))

    @staticmethod
    def pure(x):
        return x

    def map(self, f):
        return f(self)

    def join(self):
        return self


class _TailCall:
    """
    Represents a function call, to be returned so that an enclosing loop
    can execute the call, rather than using code itself::

        return f(x)
        # becomes
        return _TailCall(f, x)
    """
    __slots__ = ('f', '__weakref__')

    def __init__(self, f, *args, **kwargs):
        self.f = functools.partial(f, *args, **kwargs)

    def run(x):
        """
        Evaluates a chain of _TailCall until it's all evaluated.
        """
        # Do not use self.__class__ so that _TailCall.run() can be called on
        # arbitrary objects.
        while isinstance(x, _TailCall):
            x = x.f()

        # Recursive implementation, sometimes useful for debugging
        # if isinstance(x, _TailCall):
        #     x = _TailCall.run(x.f())
        return x

    @classmethod
    def trampoline(cls, f):
        """
        Decorator to allow a function to return :class:`_TailCall`.
        """
        return functools.wraps(f)(cls._trampoline(f))

    @classmethod
    def _trampoline(cls, f):
        run = cls.run
        def wrapper(*args, **kwargs):
            return run(
                f(*args, **kwargs)
            )
        return wrapper


class AlreadyCalledError(Exception):
    """
    Exception raised by :class:`_CallOnce` when the wrapped function has already
    been called once.
    """
    pass


class _CallOnce:
    """
    Only allow calling the wrapped function once.

    This prevents accidentally calling ``coro.send(...)`` multiple times,
    wrongly expecting it to resume twice from the same point.
    """
    __slots__ = ('f', '__weakref__')

    def __init__(self, f):
        self.f = f

    @staticmethod
    def _raise(*args, **kwargs):
        raise AlreadyCalledError(f'This function cannot be called more than once')

    def __call__(self, *args, **kwargs):
        x = self.f(*args, **kwargs)
        self.f = self._raise
        return x


def _build_stack(stack):
    *stack, init = tuple(stack)
    if stack:
        return foldr(
            lambda trans, base: trans._rebase_on(base),
            stack,
            init,
        )
    else:
        return init


class MonadTrans(Monad, abc.ABC):
    """
    Base class for monad transformers.

    Heavily inspired by transformers as defined by:
    https://hackage.haskell.org/package/transformers

    And stack manipulation inspired by:
    https://hackage.haskell.org/package/mmorph
    """
    _BASE = _Identity

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls._UNAPPLIED = cls
        """
        Unapplied monad transformer, i.e. applied to (a subclass of) :class:`_Identity`.
        """

        # Stacks of more than one transformer will override that
        cls._TRANSFORMER_STACK = (cls,)
        """
        Stack of transformers.
        """

        cls._TRANSFORM_CACHE = WeakKeyDictionary()
        """
        Cache of the transformer applied to other base monads.

        .. note:: This cache is necessary for functional purposes, so that
            rebasing a transformer on the same monad twice gives back the same
            class, which can therefore be used as key in mappings.
        """

    @abc.abstractclassmethod
    def lift(cls, m):
        """
        Lift a monadic value ``m`` by one level in the stack, i.e.:
        Given a stack for 3 transformers ``T1(T2(T3(Identity)))``,
        a value ``m = T3(Identity).pure(42)``. we have
        ``T2.lift(m) == T2(T3(Identity)).pure(42)``.

        .. seealso:: ``lift`` as defined in https://hackage.haskell.org/package/transformers
        """
        pass

    @abc.abstractclassmethod
    def hoist(cls, self, nat):
        """
        Lift a monadic value ``m`` by one level in the stack, i.e.:
        Given a stack for 3 transformers ``T1(T2(T3(Identity)))``,
        a value ``m = T2(Identity).pure(42)``. we have
        ``T2.hoist(m, T3.pure) == T2(T3(Identity)).pure(42)``.

        In other words, it allows adding a level "from below", whereas ``lift``
        adds a level "from above". It's similar to ``map``, except that instead
        of traversing all the nested functor layers, it stops at the first one.

        :param self: Monadic value to hoist.
        :type self: lisa.monad.Monad

        :param nat: Natural transform. i.e. a morphism from ``Monad1[A]`` to
            ``Monad2[A]`` that obeys certain laws.
        :type nat: collections.abc.Callable

        .. seealso:: ``hoist`` as defined in https://hackage.haskell.org/package/mmorph

        .. note:: Note for implementers: A monad transformers ``t m a`` (``t``
            is the transformer HKT, ``m`` is the base monad and ``a`` is the
            "contained type) usually ends up containing an "m (f a)" (``f`` being
            some kind of functor). For example, ``MaybeT`` in Haskell
            (:class:`Option` here) is more or less defined as ``data MaybeT m a =
            MaybeT (m (Maybe a))``. What the ``hoist`` implementation must do is to
            "rebuild" a value with a call to ``nat()`` around the ``m (...)`` part.
            For ``MaybeT``, this gives
            ``hoist nat (MaybeT (m (Maybe a))) = MaybeT(nat(m (Maybe a)))``.
        """
        pass

    @classmethod
    def pure(cls, x):
        """
        Turn a regular value of type ``A`` into a monadic value of type ``Monad[A]``.
        """
        return cls.lift(cls._BASE.pure(x))

    @classmethod
    def _rebase_on(cls, monad):
        """
        Rebase the transformer onto a new base monad and returns it.
        """
        key = monad
        # "unapply" the transformer, since we are rebuilding the stack from the
        # bottom to the top
        cls = cls._UNAPPLIED

        # This cache is necessary for functional purposes, it's not only an
        # optimisation. It guarantees that rebasing the same transformer on the
        # same monad will give the same class, so it can be used as key in
        # dictionaries.
        try:
            return cls._TRANSFORM_CACHE[key]
        except KeyError:
            transformed = cls._do_rebase_on(monad)
            cls._TRANSFORM_CACHE[key] = transformed
            return transformed

    @classmethod
    def _do_rebase_on(cls, base):
        class _Monad(cls):
            _BASE = base

        _Monad._UNAPPLIED = cls
        _Monad.__name__ = cls.__name__
        _Monad.__qualname__ = cls.__qualname__
        return _Monad

    @staticmethod
    def _decorate_coroutine_function(f):
        """
        Called by :meth:`MonadTrans.do` to wrap the user-provided coroutine function, i.e. ``async def`` functions.
        """
        return f

    @classmethod
    def do(cls, f):
        """
        Decorate a coroutine function so that ``awaits`` gains the powers of the monad.

        .. seealso:: This decorator is very similar to the do-notation in Haskell.
        """
        if not inspect.iscoroutinefunction(f):
            raise TypeError(f'{cls.__qualname__}.do() can only decorate generator functions, i.e. defined with "async def"')

        do_cls = cls
        stack = cls._TRANSFORMER_STACK

        # The direct "await" at our level are expected to yield instances of
        # the "bare" transformers (i.e. applied to _Identity), or an instance
        # of a substack
        substacks = {
            cls._UNAPPLIED_TOP: cls
            for cls in stack
            if issubclass(cls, _MonadTransStack)
        }

        transformers = {
            cls._UNAPPLIED: cls
            for cls in stack
        }

        def resolve(cls):
            try:
                return substacks[cls]
            except KeyError:
                try:
                    return transformers[cls._UNAPPLIED]
                except KeyError:
                    _stack = ', '.join(
                        cls.__qualname__
                        for cls in stack
                    )
                    raise TypeError(f'Could not find the transformer {cls.__qualname__} in the stack {do_cls.__name__}. Only the following transformers are allowed at that level: {_stack}')

        lifters = {
            cls: tuple(reversed(tuple(itertools.takewhile(
                lambda x: x is not cls,
                stack
            ))))
            for cls in stack
        }

        def make_hoister(stack):
            try:
                cls1, cls2 = stack
            except ValueError:
                return lambda x: x
            else:
                return lambda x: cls1.hoist(x, cls2.pure)

        hoisters = {
            cls: make_hoister(
                itertools.islice(
                    itertools.dropwhile(
                        lambda x: x is not cls,
                        stack
                    ),
                    None,
                    2,
                )
            )
            for cls in stack
        }

        def convert(action):
            if action.__class__ is stack[0]:
                return action
            else:
                action_cls = resolve(action.__class__)
                hoisted = hoisters[action_cls](action)

                lifted = hoisted
                for _cls in lifters[action_cls]:
                    lifted = _cls.lift(lifted)

                return lifted

        def decorate(f):
            for trans in reversed(stack):
                f = trans._decorate_coroutine_function(f)
            return f

        _f = decorate(f)

        @functools.wraps(_f)
        def wrapper(*args, **kwargs):
            coro = _f(*args, **kwargs)

            def next_(x):
                run = functools.partial(coro.send, x)

                while True:
                    try:
                        action = run()
                    except StopIteration as e:
                        x = e.value
                        if isinstance(x, Monad):
                            return x
                        else:
                            return cls.pure(x)
                    else:
                        try:
                            # Automatically lift actions based on their class,
                            # which indicates their position in the transformer
                            # stack.
                            action = convert(action)
                        except Exception as e:
                            run = functools.partial(coro.throw, e)
                        else:
                            return action.bind(_CallOnce(next_))

            return _TailCall.run(next_(None))

        wrapper.monad = cls
        return wrapper


class _CloneIdentity(_Identity):
    @classmethod
    def clone(cls):
        # Derive from _CloneIdentity to avoid long chains of inheritance, which
        # would prevent garbage collection classes and also slow down attribute
        # resolution.
        class new(_CloneIdentity):
            pass
        return new


class _MonadTransStack(MonadTrans):
    """
    Stack of monad transformers.

    * The ``_TOP`` attribute points at the top of the stack, which can the be
      unfolded by accessing ``_BASE``.

    * ``_UNAPPLIED_TOP`` is the "original" ``_TOP``, i.e. before the stack is
      rebased using :meth:`MonadTrans.rebase_on`.

    .. note:: Each stack is based on its own copy of :class:`_CloneIdentity`,
        so that all the transformers composing the stack are independent from
        any other transformer. This allows natural mixing of stacks, as there
        is never any ambiguity on what stack a given monadic value belongs to.
    """
    _BASE = _CloneIdentity

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        def get_stack(trans):
            if issubclass(trans, MonadTrans):
                yield trans
                yield from get_stack(trans._BASE)
            else:
                return

        _base = cls._BASE

        # Update the base we will build _TOP on, but do not update _BASE
        # itself. This allows subclasses of non-rebased classes to all have
        # _BASE==_CloneIdentity, so they all get independent _TOP stack.
        if _base is _CloneIdentity:
            # Ensure each stack is independent by providing a separate
            # _Identity base = _base.clone() if _base is _CloneIdentity else
            # _base
            base = _base.clone()
        else:
            base = _base

        # Rebuild the stack on top of the new base
        top = _build_stack(
            itertools.chain(
                get_stack(cls._UNAPPLIED_TOP),
                (base,)
            )
        )

        # We are defining a new stack, not just rebasing an existing stack.
        if _base is _CloneIdentity:
            cls._UNAPPLIED_TOP = top

        cls._TOP = top
        cls._TRANSFORMER_STACK = tuple(get_stack(top))

    def bind(self, continuation):
        # This looks like a dummy implementation, but is actually useful since
        # it can be called as T.bind(m, f), regardless of the precise type of
        # m.
        return self.bind(continuation)

    @classmethod
    def lift(cls, m):
        # Lift a monadic value on the whole original stack (i.e. disregarding
        # any other stack this transformer might be rebased on).
        def lift(monad, unapplied, x):
            if issubclass(unapplied, MonadTrans):
                return monad.lift(lift(monad._BASE, unapplied._BASE, x))
            else:
                return x

        return lift(cls._TOP, cls._UNAPPLIED_TOP, m)

    @instancemethod
    def hoist(cls, self, nat):
        def hoist(monad, unapplied, x):
            return monad.hoist(
                x,
                (
                    functools.partial(
                        hoist,
                        monad._BASE, unapplied._BASE,
                    )
                    if issubclass(unapplied._BASE, MonadTrans) else
                    nat
                )
            )

        # Traverse the structure using the _UNAPPLIED_TOP, but build values of the
        # correct type that reflects the actual structure of the value.
        return hoist(cls._TOP, cls._UNAPPLIED_TOP, self)


def TransformerStack(*stack):
    """
    Allows stacking together multiple :class:`MonadTrans`, e.g.::

        class Stack(TransformerStack(T1, T2, T3)):
            pass

        @Stack.do
        async def foo():

            # Any monadic value from the stack's direct components can be used.
            await T1.pure(42)
            await T2.pure(42)
            await T3.pure(42)

    """

    if len(set(stack)) != len(stack):
        raise ValueError(f'Monad transformers can only appear once in any given stack')
    else:
        class Stack(_MonadTransStack):
            _UNAPPLIED_TOP = _build_stack(stack)

        return Stack


class _Optional:
    """
    Instances of this class represents either the absence of a value, or a
    value.
    """
    pass


class Some(_Optional):
    """
    Wraps an arbitrary value to indicate its presence.
    """
    __slots__ = ('x',)

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.x})'

    __str__ = __repr__

    def __eq__(self, other):
        assert type(self) is type(other)
        return self.x == other.x


class _Nothing(_Optional):
    """
    Do not make your own instances, use the ``Nothing`` singleton.
    """
    def __repr__(self):
        return 'Nothing'

    __str__ = __repr__

    def __eq__(self, other):
        return self is other


Nothing = _Nothing()
"""
Similar to :class:`Some` but indicating the absence of value.
"""


# Do that in a base class so that Option itself gets __init_subclass__ ran on
class _AddPureNothing:
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # Specialize the _OPTION_NOTHING to the correct right _BASE.
        # Pre-computing it allows sharing a single instance for all the cases
        # where it is needed.
        cls._PURE_NOTHING = cls._BASE.pure(Nothing)


class Option(MonadTrans, _AddPureNothing):
    """
    Monad transformer that manipulates :class:`Some` and :attr:`Nothing`.

    :meth:`Option.bind` will short-circuit if :attr:`Nothing` is passed, much
    like the Rust or Javascript ``?`` operator, or the ``MaybeT`` monad
    transformer in Haskell.
    """
    __slots__ = ('_x',)

    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        """
        Wrapped value, of type ``Base[_Optional[A]]`` with ``Base`` the base
        monad of the transformer.
        """
        # Ensure we only run the _TailCall once, since it will run a
        # continuation
        x = _TailCall.run(self._x)
        self._x = x
        return x

    def __repr__(self):
        return f'Option({self.x})'

    __str__ = __repr__

    def bind(self, continuation):
        cls = self.__class__
        base = self._BASE

        def k(v):
            if v is Nothing:
                # Use pre-computed object
                return cls._PURE_NOTHING
            else:
                return continuation(v.x)._x

        return cls(base.bind(self.x, k))

    @classmethod
    def lift(cls, m):
        base = cls._BASE
        return cls(base.map(m, Some))

    @instancemethod
    def hoist(cls, self, nat):
        return cls(nat(self.x))


class State(MonadTrans):
    """
    Monad transformer analogous to Haskell's ``StateT`` transformer.

    It manipulates state-transforming functions of type ``state -> (value,
    new_state)``. This allows simulating a global state, without actually
    requiring one.
    """
    __slots__ = ('_f',)

    def __init__(self, f):
        self._f = f

    @property
    def f(self):
        """
        State-transforming function of type ``state -> (value, new_state)``
        """
        return _TailCall._trampoline(self._f)

    @classmethod
    def make_state(cls, x):
        """
        Create an initial state. All the parameters of :meth:`State.__call__`
        are passed to :meth:`State.make_state`.
        """
        return x

    def __call__(self, *args, **kwargs):
        """
        Allow calling monadic values to run the state-transforming function,
        with the initial state provided by :meth:`State.make_state`.
        """
        init = self.make_state(*args, **kwargs)
        return self.f(init)

    def bind(self, continuation):
        base = self._BASE
        def k(res):
            x, state = res
            return continuation(x)._f(state)

        return self.__class__(
            lambda state: base.bind(self.f(state), k)
        )

    @classmethod
    def lift(cls, m):
        base = cls._BASE
        return cls(
            lambda state: base.bind(
                m,
                lambda a: base.pure((a, state))
            )
        )

    @instancemethod
    def hoist(cls, self, nat):
        return cls(lambda state: nat(self.f(state)))

    @classmethod
    def from_f(cls, f):
        """
        Build a monadic value out of a state modifying function of type
        ``state -> (value, new_state)``.
        """
        base = cls._BASE
        return cls(lambda state: base.pure(f(state)))

    @classmethod
    def get_state(cls):
        """
        Returns a monadic value returning the current state.
        """
        return cls.from_f(lambda state: (state, state))

    @classmethod
    def set_state(cls, new):
        """
        Returns a monadic value setting the current state and returning the old
        one.
        """
        return cls.from_f(lambda state: (state, new))

    @classmethod
    def modify_state(cls, f):
        """
        Returns a monadic value applying ``f`` on the current state, setting
        the new state and then returning it.
        """
        def _f(state):
            new_state = f(state)
            return (new_state, new_state)
        return cls.from_f(_f)


class StateDiscard(State):
    """
    Same as :class:`State` except that calling monadic values will return the
    computed value instead of a tuple ``(value, state)``.

    This is useful for APIs where the final state is of no interest to the
    user.
    """
    def __call__(self, *args, **kwargs):
        # We are part of a monad transformer stack, so we need to apply
        # the function at the right level.
        # super().__call__() returns a "m (a, s)", so we need to map on that
        # monadic value to turn it into "m a".
        return self._BASE.map(
            super().__call__(*args, **kwargs),
            operator.itemgetter(0),
        )


def _wrap_in_coro(x):
    """
    Wrap ``x`` in a coroutine that will simply return it.
    """
    async def f():
        return x
    return f()


class _AwaitWrapper:
    """
    Dummy wrapper that allows transparently forwarding an "await" from a
    wrapped coroutine
    """
    __slots__ = ('x', '__weakref__')

    def __init__(self, x):
        self.x = x

    def __await__(self):
        return (yield self.x)


class _StopIteration(Exception):
    """
    Custom :exc:`StopIteration` that is allowed to be raised by an async
    generator.
    """
    pass


class Async(MonadTrans):
    """
    Monad transformer allowing the decorated coroutine function to await on
    non-monadic values. This is useful to mix any monad transformer defined in
    this module with other async APIs, such as :mod:`asyncio`.
    """
    __slots__ = ('_coro',)

    def __init__(self, coro):
        self._coro = coro

    @property
    def coro(self):
        """
        Coroutine that will only yield non-monadic values. All the monadic
        values will be processed by the monad transformer stack as expected and
        will stay hidden.
        """
        async def _f():
            return _TailCall.run(await self._coro)
        return _f()

    @property
    @memoized
    def x(self):
        """
        Run the coroutine to completion in its event loop.
        """
        return self._run(self.coro)

    @staticmethod
    def _run(coro):
        """
        Run a coroutine to completion, handling any non-monadic value along the
        way. This is typically the entry point of an event loop, such as
        :func:`asyncio.run`.

        .. note:: This function must be re-entrant. :func:`asyncio.run` is not
            re-entrant by default, but can be made to with
            https://pypi.org/project/nest-asyncio/
        """
        try:
            action = coro.send(None)
        except StopIteration as e:
            return e.value
        else:
            raise RuntimeError(f'Did not expect coroutine to await on {action.__class__.__qualname__}, please subclass Async and override the _run() method with e.g. asyncio.run')

    @classmethod
    def _decorate_coroutine_function(cls, f):
        """
        Wrap the coroutine function into an async generator that will:

            1. yield any monadic action
            2. await on everything else, e.g. asyncio actions

        A simpler design would simply ``cls._run()`` or await each action
        individually, but it is unfortunately not enough for ``asyncio``.
        Indeed, the coroutine itself must be run by ``asyncio.run()``, otherwise
        creating the action itself (e.g. ``asyncio.sleep(1)``) will fail as it will
        not find the global event loop.
        """
        async def _genf(*args, **kwargs):
            coro = f(*args, **kwargs)
            run = functools.partial(coro.send, None)
            while True:
                try:
                    action = run()
                # Convert to a custom type, as an async generator is not
                # allowed to raise a StopIteration (if it does, it is punished
                # by a RuntimeError). We also cannot simply "return e.value"
                # since async generators are not allowed to return anything.
                except StopIteration as e:
                    raise _StopIteration(e.value)
                else:
                    is_monad = isinstance(action, Monad)
                    try:
                        if is_monad:
                            x = yield action
                        else:
                            x = await _AwaitWrapper(action)
                    except Exception as e:
                        run = functools.partial(coro.throw, e)
                    else:
                        run = functools.partial(coro.send, x)

        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            gen = _genf(*args, **kwargs)

            # All the action the generator yields is to be processed by our
            # monadic stack, the others are already handled by the _run() call
            x = None
            while True:
                next_ = gen.asend(x)
                try:
                    action = cls._run(next_)
                except _StopIteration as e:
                    return e.args[0]
                else:
                    x = await action

        return wrapper

    @classmethod
    def lift(cls, x):
        return cls(_wrap_in_coro(x))

    @instancemethod
    def hoist(cls, self, nat):
        async def f():
            # We could use "await self.coro" but expand it to save on a layer
            # of coroutine
            x = _TailCall.run(await self._coro)
            return nat(x)
        return cls(f())

    def bind(self, continuation):
        cls = self.__class__
        base = cls._BASE

        async def f():
            return base.bind(
                await self._coro,
                lambda x: _TailCall(cls._run, continuation(x)._coro)
            )

        return cls(f())


class AsyncIO(Async):
    """
    Specialization of :class:`lisa.monad.Async` to :mod:`asyncio` event loop.
    """
    # Note that this only works properly with nest_asyncio package. Otherwise,
    # asyncio.run() is not re-entrant.
    _run = staticmethod(asyncio.run)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
