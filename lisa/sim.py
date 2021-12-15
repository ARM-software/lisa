# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022, ARM Limited and contributors.
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
Discrete event simulation framework
"""

import abc
import copy
import uuid
import functools
import itertools
import inspect
from weakref import WeakKeyDictionary
from operator import itemgetter

from lisa.utils import order_as, once_per_instance, OrderedSet, compose

class ActionBase(abc.ABC):
    def __init__(self):
        self._run_excep = None

    def __await__(self):
        return (yield self)

    def run(self, coro):
        try:
            self._run(coro)
        except Exception as e:
            self._run_excep = e

    def _run(self, coro):
        return

    def wakeup(self, coro):
        excep = self._run_excep
        if excep is None:
            return self._wakeup(coro)
        else:
            raise excep

    def _wakeup(self, coro):
        return None

    def cancel(self, coro, *args, **kwargs):
        return self._cancel(coro, *args, *kwargs)

    def _cancel(self, coro):
        return


class OnBehalfOfCoroutine(ActionBase):
    def __init__(self, coro, action):
        self.action = action
        self.coro = coro

    def run(self, _, *args, **kwargs):
        return self.action.run(self.coro, *args, **kwargs)

    def wakeup(self, _, *args, **kwargs):
        return self.action.wakeup(self.coro, *args, **kwargs)

    def cancel(self, _, *args, **kwargs):
        return self.action.cancel(self.coro, *args, **kwargs)


class RunOnceAction(ActionBase):
    def __init__(self):
        super().__init__()
        self._has_run = False

    def run(self, coro):
        if self._has_run:
            raise ValueError('Action {self} has already been used once')
        else:
            self._has_run = True
            return super().run(coro)


class NoOpAction(ActionBase):
    pass


class Raise(ActionBase):
    def __init__(self, excep):
        super().__init__()
        self._run_excep = excep

    def __str__(self):
        return f'{self.__class__.__qualname__}({self._run_excep} ({self._run_excep.__class__.__qualname__}))'


class Return(ActionBase):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def _wakeup(self, coro):
        return self.x


class Magic:
    pass


class Blocked(Magic):
    pass


class Yield(ActionBase):
    _YIELDED = WeakKeyDictionary()
    _INJECT = WeakKeyDictionary()

    def __init__(self, x):
        super().__init__()
        self.x = x

    def _run(self, coro):
        # Ugly workaround: this allows "traversing" an arbitrary number of
        # layers of Spawn and combinators and still reach the top-level
        # Coroutine.step()
        self._YIELDED[coro] = self.x

    def _wakeup(self, coro):
        return self._INJECT.pop(coro, None)


class Block(Return):
    def __init__(self):
        super().__init__(
            Blocked()
        )

    def __str__(self):
        return f'{self.__class__.__qualname__}()'


class CoroutineBase(ActionBase):
    def __init__(self, coro):
        super().__init__()
        self.coro = coro

    def step(self, **kwargs):
        try:
            return (yield from self._step(**kwargs))
        finally:
            self.coro.close()

    def _step(self):
        coro = self.coro
        action = NoOpAction()

        while True:
            try:
                x = action.wakeup(coro)
            # We take StopIteration as a signal to return immediately from the
            # calling coroutine
            except StopIteration as e:
                return e.value
            except BaseException as e:
                x = None
                run = False
                excep = e
            else:
                excep = None
                run = not isinstance(x, Blocked)

            has_excep = excep is not None

            if run or has_excep:
                if has_excep and inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED:
                    raise excep
                else:
                    try:
                        if has_excep:
                            action = coro.throw(excep)
                        else:
                            action = coro.send(x)
                    except StopIteration as e:
                        return e.value
                    else:
                        action.run(coro)

            try:
                yield
            except GeneratorExit:
                action.cancel(coro)
                raise


class CoroutineAction(CoroutineBase, ActionBase):
    def __init__(self, coro):
        super().__init__(coro)
        self._stepper = self.step()

    def _wakeup(self, coro):
        try:
            next(self._stepper)
        except StopIteration as e:
            x = e.value
        else:
            x = Blocked()

        return x

    def _cancel(self, coro):
        self._stepper.close()


class Coroutine(CoroutineBase):
    def step(self, *args, **kwargs):
        coro = self.coro
        gen = super().step(*args, **kwargs)
        while True:
            try:
                next(gen)
            except StopIteration as e:
                return e.value
            else:
                if coro in Yield._YIELDED:
                    Yield._INJECT[coro] = yield Yield._YIELDED.pop(coro)

    def run(self, callback=None):
        callback = callback or (lambda _: None)
        generator = self.step()
        while True:
            try:
                x = next(generator)
            except StopIteration as e:
                result = e.value
                callback(result)
                break
            else:
                callback(x)
        return result


def simulation(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return Coroutine(f(*args, **kwargs))
    return wrapper


class Logger:
    class _Log(ActionBase):
        def __init__(self, logger, x):
            super().__init__()
            self.x = x
            self.logger = logger

        def _run(self, coro):
            self.logger.trace.append(x)

    def __init__(self):
        self.trace = []

    def __str__(self):
        return '\n'.join(map(str, self.trace))

    def log(self, x):
        return self._Log(self, x)


class SignalClosedException(Exception):
    def __init__(self, signal):
        self.signal = signal

    def __str__(self):
        return f'The signal {self.signal} has been closed'



class Signal:
    class _Wait(ActionBase):
        def __init__(self, signal):
            super().__init__()
            self.signal = signal
            self._canceled = False

        def _run(self, coro):
            # If this coroutine has already observed this signal's value, we
            # register it so that it will be added to the waiters when this
            # round is finished.
            observed = self.signal._waiters.setdefault(coro, False)
            if observed:
                self.signal._next_waiters.add(coro)

        def _wakeup(self, coro):
            signal = self.signal
            if signal._phase == 'read':
                if signal.excep:
                    self._cancel(coro)
                    raise signal.excep
                else:
                    # If we already have observed this value, we just ignore it
                    if signal._waiters[coro]:
                        return Blocked()
                    else:
                        self._cancel(coro)
                        return signal.value
            else:
                return Blocked()

        def _cancel(self, coro):
            # This guarantees we won't destroy the state if the action is
            # canceled at an arbitrary point in the future, since _wakeup()
            # uses _cancel() in the path that unblocks.
            if self._canceled:
                return
            else:
                self._canceled = True
                signal = self.signal
                waiters = signal._waiters

                waiters[coro] = True
                # We are the last waiter to block on that write
                if all(waiters.values()):
                    # Load the next set of waiters that arrived too late for this
                    # value
                    signal._waiters = dict.fromkeys(signal._next_waiters, False)
                    signal._next_waiters = set()
                    # Unblock the current writer since all the waiters waiting for
                    # it has consumed the value.
                    if signal._phase == 'read':
                        try:
                            signal._curr_writer._consumed = True
                        # The writer might have been canceled already
                        except IndexError:
                            pass
                    # Allow the next writer to write its value
                    signal._phase = 'write'

    class _Set(ActionBase):
        def __init__(self, signal, value, excep, override_excep, allow_read=True):
            super().__init__()
            self.signal = signal
            self.value = value
            self.excep = excep
            self.override_excep = override_excep
            self.allow_read = allow_read

        def _run(self, coro):
            self._consumed = False
            # Order the writes according to who got its _run() executed first.
            self.signal._writers.append(self)

        def _cancel(self, coro):
            signal = self.signal
            writers = signal._writers
            writers.remove(self)

        def _wakeup(self, coro):
            signal = self.signal
            # When all the waiters have consumed this writer, we de-register
            # ourselves and enable the next writer in line
            if self._consumed:
                self._cancel(coro)
                return None
            elif signal.excep and not self.override_excep:
                self._cancel(coro)
                raise signal.excep
            elif signal._curr_writer is self and signal._phase == 'write':
                signal.value = self.value
                signal.excep = self.excep
                if self.allow_read:
                    signal._phase = 'read'
                    return Blocked()
                else:
                    signal._phase = 'write'
                    return None
            else:
                return Blocked()

    def __init__(self, value=None, name=None):
        self.name = name or uuid.uuid4().hex
        self.value = value
        self.excep = None
        self._orig_value = value
        self._phase = 'write'
        self._waiters = dict()
        self._next_waiters = set()
        # List of writers by order of arrival
        self._writers = []

    @property
    def _curr_writer(self):
        return self._writers[0]

    def wait(self):
        return self._Wait(self)

    def set(self, value=None):
        return self._Set(self, value=value, excep=None, override_excep=False)

    def raise_(self, excep):
        return self._Set(self, value=None, excep=excep, override_excep=True)

    def close(self):
        return self._Set(self, value=None, excep=SignalClosedException(self), override_excep=False)

    def reset(self):
        return self._Set(
            self,
            value=self._orig_value,
            excep=None,
            override_excep=True,
            # Do not allow waiters to observe this write, as we don't want to
            # wake them up.
            allow_read=False,
        )


class Map(ActionBase):
    def __init__(self, f, action, excepf=None):
        super().__init__()
        self.f = f
        self.excepf = excepf or (lambda x: x)
        self.action = action

    def _wakeup(self, coro):
        try:
            x = self.action.wakeup(coro)
        except BaseException as e:
            raise self.excepf(e)
        else:
            return x if isinstance(x, Magic) else self.f(x)

    def _run(self, coro):
        return self.action.run(coro)

    def _cancel(self, coro):
        return self.action.cancel(coro)


class Spawn(RunOnceAction):
    def __init__(self, coros):
        super().__init__()
        self._coros = list(coros)

    def _run(self, coro):
        self._coros.append(coro)
        self._action = All(self._coros)

    def _wakeup(self, coro):
        x = self._action.wakeup(coro)
        if isinstance(x, Blocked):
            return x
        else:
            # Since we added the calling coroutine at the end of the list, we
            # propagate the StopIteration into our calling coroutine (via a
            # cooperating Coroutine.step()), so that Coroutine.step() do not
            # keep running the coroutine.
            raise StopIteration(x[-1])

    def _cancel(self, coro):
        return self._action.cancel(coro)


class Lock:
    class _Lock(ActionBase):
        def __init__(self, lock, meth):
            super().__init__()
            self.lock = lock
            self._meth = meth

        def __str__(self):
            return f'Lock({self._meth.__name__} {self.lock.name})'

        def _wakeup(self, coro):
            return self._meth(self, coro)

        def _cancel(self, coro):
            try:
                self.release(coro)
            except ValueError:
                pass

        def acquire(self, coro):
            lock = self.lock
            if lock.owner is None:
                lock.owner = coro
                return True
            else:
                return Blocked()

        def release(self, coro):
            lock = self.lock
            owner = lock.owner
            if owner is None:
                raise ValueError(f'Could not release non-acquired lock {self.name} in {coro.__name__}')
            elif owner is coro:
                lock.owner = None
                return True
            else:
                raise ValueError(f'Could not release lock {self.name} in {coro.__name__} acquired by {self.owner.__name__}')

    def __init__(self, name=None):
        self.name = name or uuid.uuid4().hex
        self.owner = None

    def __str__(self):
        return f'Lock({self.name})'

    def acquire(self):
        return self._Lock(self, self._Lock.acquire)

    def release(self):
        return self._Lock(self, self._Lock.release)


class _All(RunOnceAction):
    def __init__(self, actions, raise_=True, transparent_excep=False):
        super().__init__()
        self._action = MultiAction(
            actions,
            cancel_on_raise=True,
            # We need to be able to distinguish the source of the exception
            # internally, but we will remove the MultiActionException layer
            # ourselves.
            transparent_excep=False,
        )
        self.actions = list(self._action.remaining)
        self._unblocked = {}
        self._raise = raise_
        self.__transparent_excep = transparent_excep

    def _wakeup(self, coro):
        while self._action.remaining:
            try:
                res = self._action.wakeup(coro)
            except MultiActionException as e:
                res = (e.action, None)
                excep = e.excep
                # Since we used MultiAction(..., cancel_on_raise=True), we know
                # that the other actions have been canceled already.
                if self._raise:
                    if self.__transparent_excep:
                        raise excep
                    else:
                        raise e
            else:
                excep = None

            if isinstance(res, Blocked):
                return res
            else:
                action, x = res
                self._unblocked[action] = x if self._raise else (x, excep)

        return tuple(
            self._unblocked[action]
            for action in self.actions
        )

    def _run(self, *args, **kwargs):
        return self._action.run(*args, **kwargs)

    def _cancel(self, *args, **kwargs):
        return self._action.cancel(*args, **kwargs)


class All(_All):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, raise_=True)


class AllWithExcep(_All):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, raise_=False)


class First(ActionBase):
    def __init__(self, actions, transparent_excep=False):
        super().__init__()
        self._action = MultiAction(
            actions,
            cancel_on_raise=True,
            transparent_excep=transparent_excep,
        )
        self.actions = list(self._action.remaining)

    def _wakeup(self, coro):
        res = self._action.wakeup(coro)
        if isinstance(res, Blocked):
            return res
        else:
            action, x = res
            self.cancel(coro, skip={action})
            return (action, x)

    def _run(self, *args, **kwargs):
        return self._action.run(*args, **kwargs)

    def _cancel(self, *args, **kwargs):
        return self._action.cancel(*args, **kwargs)


class AllFinishedError(Exception):
    pass


class MultiActionException(Exception):
    def __init__(self, action, excep):
        self._action = action
        self._excep = excep

    @property
    def action(self):
        if isinstance(self._action, self.__class__):
            return self._action.action
        else:
            return self._action

    @property
    def excep(self):
        if isinstance(self._excep, self.__class__):
            return self._excep.excep
        else:
            return self._excep


class MultiAction(ActionBase):

    class _Cancel(ActionBase):
        def __init__(self, action):
            super().__init__()
            self.action = action

        def _run(self, coro):
            return self.action.cancel(coro)

    @staticmethod
    def _ensure_action(x):
        if inspect.iscoroutine(x):
            return CoroutineAction(x)
        else:
            assert isinstance(x, ActionBase)
            return x

    def __init__(self, actions=(), cancel_on_raise=True, transparent_excep=False):
        super().__init__()
        actions = list(map(self._ensure_action, actions))
        self._to_run = actions
        self.remaining = OrderedSet(actions)
        self._cancel_on_raise = cancel_on_raise
        self._transparent_excep = transparent_excep

    def cancel_remaining(self):
        return self._Cancel(self)

    def register(self, action):
        if action not in self.remaining:
            self._to_run.append(action)
            self.remaining.add(action)

    def _wakeup(self, coro):
        for action in self.remaining:
            x = None
            try:
                x = action.wakeup(coro)
            except Exception as e:
                self._excep_cancel(coro, skip={action})
                if self._transparent_excep:
                    raise e
                else:
                    raise MultiActionException(action, e)

            if isinstance(x, Blocked):
                continue
            else:
                self.remaining.discard(action)
                return (action, x)
        else:
            if self.remaining:
                return Blocked()
            else:
                raise AllFinishedError()

    def _run(self, coro):
        while self._to_run:
            action = self._to_run.pop(0)
            try:
                action.run(coro)
            except Exception:
                self._excep_cancel(coro, skip={action})
                raise

    def _excep_cancel(self, coro, **kwargs):
        if self._cancel_on_raise:
            return self._cancel(coro, **kwargs)

    def _cancel(self, coro, skip=None):
        skip = set(skip or [])
        excep = None
        to_cancel = self.remaining - skip
        for action in to_cancel:
            try:
                action.cancel(coro)
            except Exception as e:
                # Only save the first exception, as we need something to raise
                # later. The first one is probably the most interesting and the
                # least likely to be the result of a broken state.
                excep = e if excep is None else excep
            finally:
                self.remaining.remove(action)

        if excep is None:
            return
        else:
            raise excep


class MissedTimerException(Exception):
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual

    def __str__(self):
        return f'{self.__class__.__qualname__}(expected={self.expected}, actual={self.actual})'


class Clock:
    class _InstantAction(ActionBase):
        def __init__(self, clock, action):
            super().__init__()
            self.clock = clock
            self.action = action

        def _unregister(self):
            try:
                del self.clock._instant_actions[self]
            except KeyError:
                pass

        def _run(self, coro):
            self.clock._instant_actions[self] = coro
            return self.action.run(coro)

        def _wakeup(self, coro):
            clock = self.clock
            woken = clock._instant_wakeup
            if woken is None:
                self._unregister()
                return self.action.wakeup(coro)
            else:
                action, x = woken
                if action is self:
                    clock._instant_wakeup = None
                    self._unregister()
                    return x
                else:
                    return Blocked()

        def _cancel(self, coro):
            self._unregister()
            return self.action.cancel(coro)

    class _TimerBase(ActionBase, abc.ABC):
        def __init__(self, clock):
            super().__init__()
            self.clock = clock

        @abc.abstractproperty
        def timestamp(self):
            pass

        def _run(self, coro):
            timestamp = self.timestamp
            self.clock._blocked_timers[self] = timestamp

        def _wakeup(self, coro):
            clock = self.clock
            timers = clock._blocked_timers
            wakeup_ts = timers[self]

            instantaneous = clock._instant_actions
            if clock._instant_wakeup is None:
                for _action, _coro, in list(instantaneous.items()):
                    x = _action.wakeup(_coro)
                    if not isinstance(x, Blocked):
                        clock._instant_wakeup = (_action, x)
                        return Blocked()
            else:
                return Blocked()

            ts = max(
                clock.value,
                min(timers.values()),
            )

            if wakeup_ts > ts:
                res = Blocked()
            elif wakeup_ts == ts:
                clock.value = ts
                self.cancel(coro)
                res = ts
            else:
                self.cancel(coro)
                raise MissedTimerException(
                    wakeup_ts,
                    ts,
                )

            return res

        def _cancel(self, coro):
            try:
                del self.clock._blocked_timers[self]
            except KeyError:
                pass

    class _Timer(_TimerBase):
        def __init__(self, clock, timestamp):
            super().__init__(clock)
            self._timestamp = timestamp

        @property
        def timestamp(self):
            return self._timestamp

    class _Sleep(_TimerBase):
        def __init__(self, clock, timeout):
            super().__init__(clock)
            self.timeout = timeout

        @property
        def timestamp(self):
            return self.clock.value + self.timeout


    def __init__(self, name=None, value=0):
        self.name = name or uuid.uuid4().hex
        self.value = value
        self._blocked_timers = {}
        self._instant_actions = {}
        self._instant_wakeup = None

    def instantaneous(self, action):
        return self._InstantAction(self, action)

    def timer(self, timestamp):
        return self._Timer(self, timestamp)

    def sleep(self, timeout):
        return self._Sleep(self, timeout)


class Counter:
    class _Counter(ActionBase):

        def __init__(self, counter, f):
            super().__init__()
            self.counter = counter
            self.f = f

        def _wakeup(self, coro):
            counter = self.counter
            f = self.f
            x = f(counter.x)
            counter.x = x
            return x


    def __init__(self, init=0, name=None):
        self.init = init
        self.x = init
        self.name = name or uuid.uuid4().hex

    def modify(self, f):
        return self._Counter(self, f)

    def inc(self):
        return self.modify(lambda x: x + 1)

    def dec(self):
        return self.modify(lambda x: x - 1)

    def set(self, x):
        return self.modify(lambda _: x)

    def reset(self):
        return self.set(self.init)


async def map_action(f, coro):
    @coro_scheduler
    async def scheduler():
        x = None
        excep = None
        while True:
            action = yield (coro, x, excep)
            try:
                x = await f(action)
            except Exception as e:
                excep = e
                x = None
            else:
                excep = None
    return await scheduler()


class ClosedCoroutine(Exception):
    def __init__(self, value):
        self.value = value


async def schedule_coro(gen):
    try:
        coro, x, excep = await gen.asend(None)
        while True:
            try:
                if excep is None:
                    action = coro.send(x)
                elif isinstance(excep, GeneratorExit):
                    coro.close()
                    # As if the coroutine returned None
                    raise StopIteration(None)
                else:
                    action = coro.throw(excep)
            except StopIteration as e:
                coro, x, excep = await gen.athrow(ClosedCoroutine(e.value))
            except Exception as e:
                coro, x, excep = await gen.athrow(e)
            else:
                coro, x, excep = await gen.asend(action)
    except ClosedCoroutine as e:
        return e.value
    except StopAsyncIteration:
        return


def coro_scheduler(f):
    @functools.wraps(f)
    async def wrapper(*args, **kwargs):
        return await schedule_coro(f(*args, **kwargs))
    return wrapper


class Interrupt(Exception):
    pass


class Interrupter:
    def __init__(self):
        self._signal = Signal()

    async def interrupt(self, excep=None):
        sig = self._signal
        excep = excep or Interrupt()
        await sig.raise_(excep)
        # Reset the signal, so that it can be reused again and will not raise
        # until then.
        await sig.reset()

    def wrap_action(self, action):
        terminate = self._signal.wait()

        def extract(res):
            _action, _x = res
            assert _action is not terminate
            return _x

        return Map(
            f=extract,
            action=First(
                (terminate, action),
                transparent_excep=True,
            ),
        )

    def wrap_func(self, f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            try:
                x = await map_action(
                    self.wrap_action,
                    f(*args, **kwargs),
                )
            # Catch the exception marking the end of the execution and simply return
            except Interrupt:
                return
            else:
                # The function is not allowed to return anything as it
                # could be forcefully stopped at some point
                assert x is None
                return

        return wrapper


import enum

class TaskState(enum.Enum):
    CREATED = enum.auto()
    WORKING = enum.auto()
    SLEEPING = enum.auto()
    FROZEN = enum.auto()
    DEAD = enum.auto()


class SysCall:
    def __await__(self):
        return (yield self)

class SleepSysCall(SysCall):
    def __init__(self, timeout):
        self.timeout = timeout

class WorkSysCall(SysCall):
    def __init__(self, amount):
        self.amount = amount

class SuspendedSyscall(Exception):
    def __init__(self, syscall):
        self.syscall = syscall

class InterruptedSyscall(Exception):
    pass


class Runqueue:
    def __init__(self, cpu):
        self.cpu = cpu

    async def run_task(self, task):
        work_clk = Clock('work')
        capa_signal = Signal(value=1)

        rq_interrupter = Interrupter()
        syscall_interrupter = Interrupter()
        async def interrupt_syscall():
            await syscall_interrupter.interrupt(InterruptedSyscall())


        async def user_handler():
            ts = await work_clk.sleep(0.008)
            print('changing capa', ts)
            await capa_signal.set(2)
            ts = await work_clk.sleep(0.009)
            print('interrupting syscall @', ts)
            await interrupt_syscall()

        async def pelt_handler():
            ts = work_clk.value
            while True:
            # while task.state != TaskState.DEAD:
                print('pelt', ts)
                ts = await work_clk.sleep(1e-3)

        async def tick_handler():
            ts = work_clk.value
            while True:
                print('tick', ts)
                ts = await work_clk.sleep(4e-3)

        async def handle_syscall(task, syscall):
            start_ts = work_clk.value

            if isinstance(syscall, SleepSysCall):
                task.state = TaskState.SLEEPING
                try:
                    ts = await work_clk.sleep(syscall.timeout)
                except InterruptedSyscall as e:
                    ts = work_clk.value
                    print('interrupted syscall', syscall, '@', ts)
                    elapsed = ts - start_ts
                    remaining = syscall.timeout - elapsed
                    if remaining:
                        raise SuspendedSyscall(SleepSysCall(remaining))
                    else:
                        return start_ts
                else:
                    return ts

            elif isinstance(syscall, WorkSysCall):
                task.state = TaskState.WORKING

                work = syscall.amount
                capa = capa_signal.value

                def remaining():
                    ts = work_clk.value
                    remaining_work = work - (ts - start_ts) * capa
                    if remaining_work:
                        raise SuspendedSyscall(WorkSysCall(remaining_work))
                    else:
                        return ts

                do_work = work_clk.sleep(work / capa)
                capa_change = capa_signal.wait()
                try:
                    action, x = await First((do_work, capa_change))
                except InterruptedSyscall:
                    print('interrupted syscall', syscall, '@', work_clk.value)
                    return remaining()
                else:
                    if action is capa_change:
                        return remaining()
                    else:
                        ts = x
                        return ts
            else:
                raise ValueError(f'Unknown syscall: {syscall}')


        # import copy
        # task2 = copy.deepcopy(task)
        tasks = {
            task: task.run(),
            # task2: task2.run(),
        }
        def pick_next(tasks):
            scheduled = set()
            while True:
                runnable = [
                    task
                    for task in tasks
                    if (
                        task not in scheduled and
                        task.state != TaskState.DEAD
                    )
                ]
                if runnable:
                    task = runnable[0]
                    scheduled.add(task)
                    yield task
                elif scheduled:
                    scheduled.clear()
                else:
                    return

        async def task_handler():
            @coro_scheduler
            async def scheduler(coros):
                state = dict.fromkeys(tasks.values())
                for task in pick_next(tasks.keys()):
                    next_syscall = None
                    if task is None:
                        return
                    else:
                        coro = tasks[task]
                        syscall = state[coro]
                        if syscall is None:
                            x = None
                            excep = None
                        else:
                            action = map_action(
                                compose(
                                    # Make all the syscall's action
                                    # instantaneous wrt to the clock, to avoid
                                    # a range of subtle bugs.
                                    work_clk.instantaneous,
                                    # Make the actions of the syscall handler
                                    # executed on behalf of the task's
                                    # coroutine, so that any state attached to
                                    # the coroutine "ownning" the action is
                                    # properly attached to the task
                                    functools.partial(OnBehalfOfCoroutine, coro),
                                    # Allow interrupting the ongoing syscall by
                                    # injecting an exception.
                                    syscall_interrupter.wrap_action,
                                ),
                                handle_syscall(task, syscall)
                            )
                            try:
                                x = await action
                            except GeneratorExit:
                                raise
                            except SuspendedSyscall as e:
                                next_syscall = e.syscall
                            except BaseException as e:
                                x = None
                                excep = e
                            else:
                                excep = None

                        try:
                            if next_syscall is None:
                                syscall = yield (coro, x, excep)
                            else:
                                syscall = next_syscall
                        except ClosedCoroutine as e:
                            task.state = TaskState.DEAD
                            del state[coro]
                        else:
                            state[coro] = syscall

            await scheduler(tasks)
            await rq_interrupter.interrupt()

        async def test():
            while True:
                capa = await capa_signal.wait()
                print('log (1) capa=', capa)

        async def test2():
            while True:
                capa = await capa_signal.wait()
                print('log (2) capa=', capa)

        return await All(
            map(
                lambda f: rq_interrupter.wrap_func(f)(),
                (
                    tick_handler,
                    pelt_handler,
                    user_handler,
                    task_handler,
                    test,
                    test2,
                )
            )
        )


class TaskBase(abc.ABC):
    def __init__(self):
        self.state = TaskState.CREATED

    async def run(self, *args, **kwargs):
        try:
            return await self._run(*args, **kwargs)
        finally:
            self.state = TaskState.DEAD

    @abc.abstractmethod
    async def _run(self):
        pass


class Task(TaskBase):
    def __init__(self, f, name=None):
        super().__init__()
        self.f = f
        self.name = name or str(f)

    async def _run(self):
        return await self.f()

    def __str__(self):
        return f'{self.__class__.__qualname__}({self.name})'


def task_factory(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return Task(functools.partial(f, *args, **kwargs))
    return wrapper


def main():
    @task_factory
    async def taskf(duration=10, name=None):
        ts = 0
        while ts < duration:
            print('task start working @', ts)
            ts = await WorkSysCall(16e-3)
            print('task start sleeping @', ts)
            ts = await SleepSysCall(16e-3)


    @simulation
    async def simmain():
        rq0 = Runqueue(cpu=0)
        rq1 = Runqueue(cpu=1)
        task = taskf(0.052)

        # await rq0.run_task(task)
        await Spawn([rq0.run_task(task)])
        # for i in range(6):
        #     await NoOpAction()
        #     print(f'aaaaaaa {i}')
        for i in range(10500):
            await Yield(f'XXX{i}')
        return 55

    sim = simmain()

    def cb(state):
        print('.'*40, state)
    cb=None

    import io
    from contextlib import redirect_stdout
    cm = redirect_stdout(io.StringIO())
    from lisa.utils import measure_time
    with measure_time() as m:#, cm:
        result = sim.run(callback=cb)
    print()
    print('simulation wall clock duration', m.delta, '(s)')
    # result = sim.run()

    # for state in sim.step():
    #     print(state)

    # print('='*80)
    # print(state['log'])
    # print()

    # print('counters')
    # for cnt in state['counters']:
    #     print(f'{cnt.name}: {cnt.x}')
    # print('='*80)
    # print(result)

if __name__ == '__main__':
    main()
