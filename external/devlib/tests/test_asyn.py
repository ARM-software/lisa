#
#    Copyright 2024 ARM Limited
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

import sys
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from pytest import skip, raises

from devlib.utils.asyn import run, asynccontextmanager


class AsynTestExcep(Exception):
    pass


class Awaitable:
    def __await__(self):
        return (yield self)


@contextmanager
def raises_and_bubble(cls):
    try:
        yield
    except BaseException as e:
        if isinstance(e, cls):
            raise
        else:
            raise AssertionError(f'Did not raise instance of {cls}')
    else:
        raise AssertionError(f'Did not raise any exception')


@contextmanager
def coro_stop_iteration(x):
    try:
        yield
    except StopIteration as e:
        assert e.value == x
    except BaseException:
        raise
    else:
        raise AssertionError('Coroutine did not finish')


def _do_test_run(top_run):

    async def test_run_basic():

        async def f():
            return 42

        assert run(f()) == 42

    top_run(test_run_basic())


    async def test_run_basic_raise():

        async def f():
            raise AsynTestExcep

        with raises(AsynTestExcep):
            run(f())

    top_run(test_run_basic_raise())


    async def test_run_basic_await():
        async def nested():
            return 42

        async def f():
            return await nested()

        assert run(f()) == 42

    top_run(test_run_basic_await())


    async def test_run_basic_await_raise():
        async def nested():
            raise AsynTestExcep

        async def f():
            with raises_and_bubble(AsynTestExcep):
                return await nested()

        with raises(AsynTestExcep):
            run(f())

    top_run(test_run_basic_await_raise())


    async def test_run_nested1():
        async def nested():
            return 42

        async def f():
            return run(nested())

        assert run(f()) == 42

    top_run(test_run_nested1())


    async def test_run_nested1_raise():
        async def nested():
            raise AsynTestExcep

        async def f():
            with raises_and_bubble(AsynTestExcep):
                return run(nested())

        with raises(AsynTestExcep):
            run(f())

    top_run(test_run_nested1_raise())


    async def test_run_nested2():
        async def nested2():
            return 42

        async def nested1():
            return run(nested2())

        async def f():
            return run(nested1())

        assert run(f()) == 42

    top_run(test_run_nested2())


    async def test_run_nested2_raise():
        async def nested2():
            raise AsynTestExcep

        async def nested1():
            with raises_and_bubble(AsynTestExcep):
                return run(nested2())

        async def f():
            with raises_and_bubble(AsynTestExcep):
                return run(nested1())

        with raises(AsynTestExcep):
            run(f())

    top_run(test_run_nested2_raise())


    async def test_run_nested2_block():
        async def nested2():
            return 42

        def nested1():
            return run(nested2())

        async def f():
            return nested1()

        assert run(f()) == 42

    top_run(test_run_nested2_block())


    async def test_run_nested2_block_raise():
        async def nested2():
            raise AsynTestExcep

        def nested1():
            with raises_and_bubble(AsynTestExcep):
                return run(nested2())

        async def f():
            with raises_and_bubble(AsynTestExcep):
                return nested1()

        with raises(AsynTestExcep):
            run(f())

    top_run(test_run_nested2_block_raise())



    async def test_coro_send():
        async def f():
            return await Awaitable()

        coro = f()
        coro.send(None)

        with coro_stop_iteration(42):
            coro.send(42)

    top_run(test_coro_send())


    async def test_coro_nested_send():
        async def nested():
            return await Awaitable()

        async def f():
            return await nested()

        coro = f()
        coro.send(None)

        with coro_stop_iteration(42):
            coro.send(42)

    top_run(test_coro_nested_send())


    async def test_coro_nested_send2():
        future = asyncio.Future()
        future.set_result(42)

        async def nested():
            return await future

        async def f():
            return run(nested())

        assert run(f()) == 42

    top_run(test_coro_nested_send2())


    async def test_coro_nested_send3():
        future = asyncio.Future()
        future.set_result(42)

        async def nested2():
            return await future

        async def nested():
            return run(nested2())

        async def f():
            return run(nested())

        assert run(f()) == 42

    top_run(test_coro_nested_send3())


    async def test_coro_throw():
        async def f():
            try:
                await Awaitable()
            except AsynTestExcep:
                return 42

        coro = f()
        coro.send(None)

        with coro_stop_iteration(42):
            coro.throw(AsynTestExcep)

    top_run(test_coro_throw())


    async def test_coro_throw2():
        async def f():
            await Awaitable()

        coro = f()
        coro.send(None)

        with raises(AsynTestExcep):
            coro.throw(AsynTestExcep)

    top_run(test_coro_throw2())


    async def test_coro_nested_throw():
        async def nested():
            try:
                await Awaitable()
            except AsynTestExcep:
                return 42

        async def f():
            return await nested()

        coro = f()
        coro.send(None)

        with coro_stop_iteration(42):
            coro.throw(AsynTestExcep)

    top_run(test_coro_nested_throw())


    async def test_coro_nested_throw2():
        async def nested():
            await Awaitable()

        async def f():
            with raises_and_bubble(AsynTestExcep):
                await nested()

        coro = f()
        coro.send(None)

        with raises(AsynTestExcep):
            coro.throw(AsynTestExcep)

    top_run(test_coro_nested_throw2())


    async def test_coro_nested_throw3():
        future = asyncio.Future()
        future.set_exception(AsynTestExcep())

        async def nested():
            await future

        async def f():
            with raises_and_bubble(AsynTestExcep):
                run(nested())

        with raises(AsynTestExcep):
            run(f())

    top_run(test_coro_nested_throw3())


    async def test_coro_nested_throw4():
        future = asyncio.Future()
        future.set_exception(AsynTestExcep())

        async def nested2():
            await future

        async def nested():
            return run(nested2())

        async def f():
            with raises_and_bubble(AsynTestExcep):
                run(nested())

        with raises(AsynTestExcep):
            run(f())

    top_run(test_coro_nested_throw4())

    async def test_async_cm():
        state = None

        async def f():
            return 43

        @asynccontextmanager
        async def cm():
            nonlocal state
            state = 'started'
            await f()
            try:
                yield 42
            finally:
                await f()
                state = 'finished'

        async with cm() as x:
            assert state == 'started'
            assert x == 42

        assert state == 'finished'

    top_run(test_async_cm())

    async def test_async_cm2():
        state = None

        async def f():
            return 43

        @asynccontextmanager
        async def cm():
            nonlocal state
            state = 'started'
            await f()
            try:
                await f()
                yield 42
                await f()
            except AsynTestExcep:
                await f()
                # Swallow the exception
                pass
            finally:
                await f()
                state = 'finished'

        async with cm() as x:
            assert state == 'started'
            raise AsynTestExcep()

        assert state == 'finished'

    top_run(test_async_cm2())

    async def test_async_cm3():
        state = None

        async def f():
            return 43

        @asynccontextmanager
        async def cm():
            nonlocal state
            state = 'started'
            await f()
            try:
                yield 42
            finally:
                await f()
                state = 'finished'

        with cm() as x:
            assert state == 'started'
            assert x == 42

        assert state == 'finished'

    top_run(test_async_cm3())

    def test_async_cm4():
        state = None

        async def f():
            return 43

        @asynccontextmanager
        async def cm():
            nonlocal state
            state = 'started'
            await f()
            try:
                yield 42
            finally:
                await f()
                state = 'finished'

        with cm() as x:
            assert state == 'started'
            assert x == 42

        assert state == 'finished'

    test_async_cm4()

    def test_async_cm5():
        @asynccontextmanager
        async def cm_f():
            yield 42

        cm = cm_f()
        assert top_run(cm.__aenter__()) == 42
        assert not top_run(cm.__aexit__(None, None, None))

    test_async_cm5()

    def test_async_gen1():
        async def agen_f():
            for i in range(2):
                yield i

        agen = agen_f()
        assert top_run(anext(agen)) == 0
        assert top_run(anext(agen)) == 1

    test_async_gen1()


def _test_in_thread(setup, test):
    def f():
        with setup() as run:
            return test()

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(f).result()


def _test_run_with_setup(setup):
    def run_with_existing_loop(coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Simulate case where devlib is ran in a context where the main app has
        # set an event loop at some point
        try:
            return asyncio.run(coro)
        finally:
            loop.close()

    def run_with_existing_loop2(coro):
        # This is similar to how things are executed on IPython/jupyterlab
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def run_with_to_thread(top_run, coro):
        # Add a layer of asyncio.to_thread(), to simulate a case where users
        # would be using the blocking API along with asyncio.to_thread() (code
        # written before devlib gained async capabilities or wishing to
        # preserve compat with older devlib versions)
        async def wrapper():
            return await asyncio.to_thread(
                top_run, coro
            )
        return top_run(wrapper())


    runners = [
        run,
        asyncio.run,
        run_with_existing_loop,
        run_with_existing_loop2,

        partial(run_with_to_thread, run),
        partial(run_with_to_thread, asyncio.run),
        partial(run_with_to_thread, run_with_existing_loop),
        partial(run_with_to_thread, run_with_existing_loop2),
    ]

    for top_run in runners:
        _test_in_thread(
            setup,
            partial(_do_test_run, top_run),
        )


def test_run_stdlib():
    @contextmanager
    def setup():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield asyncio.run
        finally:
            loop.close()

    _test_run_with_setup(setup)


def test_run_uvloop():
    try:
        import uvloop
    except ImportError:
        skip('uvloop not installed')
    else:
        @contextmanager
        def setup():
            if sys.version_info >= (3, 11):
                with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
                    yield runner.run
            else:
                uvloop.install()
                yield asyncio.run

        _test_run_with_setup(setup)
