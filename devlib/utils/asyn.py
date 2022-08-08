#    Copyright 2013-2018 ARM Limited
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


"""
Async-related utilities
"""

import abc
import asyncio
import functools
import itertools
import contextlib
import pathlib
import os.path
import inspect

# Allow nesting asyncio loops, which is necessary for:
# * Being able to call the blocking variant of a function from an async
#   function for backward compat
# * Critically, run the blocking variant of a function in a Jupyter notebook
#   environment, since it also uses asyncio.
#
# Maybe there is still hope for future versions of Python though:
# https://bugs.python.org/issue22239
import nest_asyncio
nest_asyncio.apply()


def create_task(awaitable, name=None):
    if isinstance(awaitable, asyncio.Task):
        task = awaitable
    else:
        task = asyncio.create_task(awaitable)
    if name is None:
        name = getattr(awaitable, '__qualname__', None)
    task.name = name
    return task


class AsyncManager:
    def __init__(self):
        self.task_tree = dict()
        self.resources = dict()

    def track_access(self, access):
        """
        Register the given ``access`` to have been handled by the current
        async task.

        :param access: Access that were done.
        :type access: ConcurrentAccessBase

        This allows :func:`concurrently` to check that concurrent tasks did not
        step on each other's toes.
        """
        try:
            task = asyncio.current_task()
        except RuntimeError:
            pass
        else:
            self.resources.setdefault(task, set()).add(access)

    async def concurrently(self, awaitables):
        """
        Await concurrently for the given awaitables, and cancel them as soon as
        one raises an exception.
        """
        awaitables = list(awaitables)

        # Avoid creating asyncio.Tasks when it's not necessary, as it will
        # disable a the blocking path optimization of Target._execute_async()
        # that uses blocking calls as long as there is only one asyncio.Task
        # running on the event loop.
        if len(awaitables) == 1:
            return [await awaitables[0]]

        tasks = list(map(create_task, awaitables))

        current_task = asyncio.current_task()
        task_tree = self.task_tree

        try:
            node = task_tree[current_task]
        except KeyError:
            is_root_task = True
            node = set()
        else:
            is_root_task = False
        task_tree[current_task] = node

        task_tree.update({
            child: set()
            for child in tasks
        })
        node.update(tasks)

        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        finally:

            def get_children(task):
                immediate_children = task_tree[task]
                return frozenset(
                    itertools.chain(
                        [task],
                        immediate_children,
                        itertools.chain.from_iterable(
                            map(get_children, immediate_children)
                        )
                    )
                )

            # Get the resources created during the execution of each subtask
            # (directly or indirectly)
            resources = {
                task: frozenset(
                    itertools.chain.from_iterable(
                        self.resources.get(child, [])
                        for child in get_children(task)
                    )
                )
                for task in tasks
            }
            for (task1, resources1), (task2, resources2) in itertools.combinations(resources.items(), 2):
                for res1, res2 in itertools.product(resources1, resources2):
                    if issubclass(res2.__class__, res1.__class__) and res1.overlap_with(res2):
                        raise RuntimeError(
                            'Overlapping resources manipulated in concurrent async tasks: {} (task {}) and {} (task {})'.format(res1, task1.name, res2, task2.name)
                        )

            if is_root_task:
                self.resources.clear()
                task_tree.clear()

    async def map_concurrently(self, f, keys):
        """
        Similar to :meth:`concurrently`,
        but maps the given function ``f`` on the given ``keys``.

        :return: A dictionary with ``keys`` as keys, and function result as
            values.
        """
        keys = list(keys)
        return dict(zip(
            keys,
            await self.concurrently(map(f, keys))
        ))


def compose(*coros):
    """
    Compose coroutines, feeding the output of each as the input of the next
    one.

    ``await compose(f, g)(x)`` is equivalent to ``await f(await g(x))``

    .. note:: In Haskell, ``compose f g h`` would be equivalent to ``f <=< g <=< h``
    """
    async def f(*args, **kwargs):
        empty_dict = {}
        for coro in reversed(coros):
            x = coro(*args, **kwargs)
            # Allow mixing corountines and regular functions
            if asyncio.isfuture(x):
                x = await x
            args = [x]
            kwargs = empty_dict

        return x
    return f


class _AsyncPolymorphicFunction:
    """
    A callable that allows exposing both a synchronous and asynchronous API.

    When called, the blocking synchronous operation is called. The ```asyn``
    attribute gives access to the asynchronous version of the function, and all
    the other attribute access will be redirected to the async function.
    """
    def __init__(self, asyn, blocking):
        self.asyn = asyn
        self.blocking = blocking

    def __get__(self, *args, **kwargs):
        return self.__class__(
            asyn=self.asyn.__get__(*args, **kwargs),
            blocking=self.blocking.__get__(*args, **kwargs),
        )

    def __call__(self, *args, **kwargs):
        return self.blocking(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.asyn, attr)


class memoized_method:
    """
    Decorator to memmoize a method.

    It works for:

        * async methods (coroutine functions)
        * non-async methods
        * method already decorated with :func:`devlib.asyn.asyncf`.

    .. note:: This decorator does not rely on hacks to hash unhashable data. If
        such input is required, it will either have to be coerced to a hashable
        first (e.g. converting a list to a tuple), or the code of
        :func:`devlib.asyn.memoized_method` will have to be updated to do so.
    """
    def __init__(self, f):
        memo = self

        sig = inspect.signature(f)

        def bind(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            key = (bound.args[1:], tuple(sorted(bound.kwargs.items())))

            return (key, bound.args, bound.kwargs)

        def get_cache(self):
            try:
                cache = self.__dict__[memo.name]
            except KeyError:
                cache = {}
                self.__dict__[memo.name] = cache
            return cache


        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def wrapper(self, *args, **kwargs):
                cache = get_cache(self)
                key, args, kwargs = bind(self, *args, **kwargs)
                try:
                    return cache[key]
                except KeyError:
                    x = await f(*args, **kwargs)
                    cache[key] = x
                    return x
        else:
            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):
                cache = get_cache(self)
                key, args, kwargs = bind(self, *args, **kwargs)
                try:
                    return cache[key]
                except KeyError:
                    x = f(*args, **kwargs)
                    cache[key] = x
                    return x


        self.f = wrapper
        self._name = f.__name__

    @property
    def name(self):
        return '__memoization_cache_of_' + self._name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __get__(self, obj, owner=None):
        return self.f.__get__(obj, owner)

    def __set__(self, obj, value):
        raise RuntimeError("Cannot monkey-patch a memoized function")

    def __set_name__(self, owner, name):
        self.name = name


def asyncf(f):
    """
    Decorator used to turn a coroutine into a blocking function, with an
    optional asynchronous API.

    **Example**::

        @asyncf
        async def foo(x):
            await do_some_async_things(x)
            return x

        # Blocking call, just as if the function was synchronous, except it may
        # use asynchronous code inside, e.g. to do concurrent operations.
        foo(42)

        # Asynchronous API, foo.asyn being a corountine
        await foo.asyn(42)

    This allows the same implementation to be both used as blocking for ease of
    use and backward compatibility, or exposed as a corountine for callers that
    can deal with awaitables.
    """
    @functools.wraps(f)
    def blocking(*args, **kwargs):
        # Since run() needs a corountine, make sure we provide one
        async def wrapper():
            x = f(*args, **kwargs)
            # Async generators have to be consumed and accumulated in a list
            # before crossing a blocking boundary.
            if inspect.isasyncgen(x):

                def genf():
                    asyncgen = x.__aiter__()
                    while True:
                        try:
                            yield asyncio.run(asyncgen.__anext__())
                        except StopAsyncIteration:
                            return

                return genf()
            else:
                return await x
        return asyncio.run(wrapper())

    return _AsyncPolymorphicFunction(
        asyn=f,
        blocking=blocking,
    )


class _AsyncPolymorphicCM:
    """
    Wrap an async context manager such that it exposes a synchronous API as
    well for backward compatibility.
    """
    def __init__(self, async_cm):
        self.cm = async_cm

    def __aenter__(self, *args, **kwargs):
        return self.cm.__aenter__(*args, **kwargs)

    def __aexit__(self, *args, **kwargs):
        return self.cm.__aexit__(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        return asyncio.run(self.cm.__aenter__(*args, **kwargs))

    def __exit__(self, *args, **kwargs):
        return asyncio.run(self.cm.__aexit__(*args, **kwargs))


def asynccontextmanager(f):
    """
    Same as :func:`contextlib.asynccontextmanager` except that it can also be
    used with a regular ``with`` statement for backward compatibility.
    """
    f = contextlib.asynccontextmanager(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        cm = f(*args, **kwargs)
        return _AsyncPolymorphicCM(cm)

    return wrapper


class ConcurrentAccessBase(abc.ABC):
    """
    Abstract Base Class for resources tracked by :func:`concurrently`.
    """
    @abc.abstractmethod
    def overlap_with(self, other):
        """
        Return ``True`` if the resource overlaps with the given one.

        :param other: Resources that should not overlap with ``self``.
        :type other: devlib.utils.asym.ConcurrentAccessBase

        .. note:: It is guaranteed that ``other`` will be a subclass of our
            class.
        """

class PathAccess(ConcurrentAccessBase):
    """
    Concurrent resource representing a file access.

    :param namespace: Identifier of the namespace of the path. One of "target" or "host".
    :type namespace: str

    :param path: Normalized path to the file.
    :type path: str

    :param mode: Opening mode of the file. Can be ``"r"`` for read and ``"w"``
        for writing.
    :type mode: str
    """
    def __init__(self, namespace, path, mode):
        assert namespace in ('host', 'target')
        self.namespace = namespace
        assert mode in ('r', 'w')
        self.mode = mode
        self.path = os.path.abspath(path) if namespace == 'host' else os.path.normpath(path)

    def overlap_with(self, other):
        path1 = pathlib.Path(self.path).resolve()
        path2 = pathlib.Path(other.path).resolve()
        return (
            self.namespace == other.namespace and
            'w' in (self.mode, other.mode) and
            (
                path1 == path2 or
                path1 in path2.parents or
                path2 in path1.parents
            )
        )

    def __str__(self):
        mode = {
            'r': 'read',
            'w': 'write',
        }[self.mode]
        return '{} ({})'.format(self.path, mode)
