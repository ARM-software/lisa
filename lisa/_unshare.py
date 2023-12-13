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
This is a separate module for the sole reason that it must be importable
without pulling other imports.

This is because modules like pyarrow will spawn a background thread (for
jemalloc) that will prevent the unshare(CLONE_NEWUSER) syscall from succeeding,
since it cannot work on a multithreaded application.
"""

import sys
import os
import pickle
import errno
import functools
import contextlib
import multiprocessing
import threading
import logging
import logging.handlers
import queue
from importlib.util import module_from_spec
from importlib.machinery import ModuleSpec

from cffi import FFI

def _do_unshare():
    CLONE_NEWUSER = 0x10000000
    CLONE_NEWNS = 0x00020000

    ffi = FFI()

    ffi.cdef('''
    int unshare(int flags);
    int mount(const char *source, const char *target, const char *filesystemtype, unsigned long mountflags, const void *data);
    ''')

    libc = ffi.dlopen(None)

    # Get UID and GID before the unshare call
    euid = os.geteuid()
    egid = os.getegid()

    # Unshare the mount namespace and the user namespace. This allows us to
    # become root and mount whatever we need, all without being actually root
    # in the parent namespace.
    ret = libc.unshare(CLONE_NEWUSER | CLONE_NEWNS)
    if ret:
        raise RuntimeError(f'unshare syscall failed with ret={ret} (errno={errno.errorcode[ffi.errno]})')

    # This is required before writing to gid_map
    with open('/proc/self/setgroups', 'wb') as f:
        f.write(b'deny\n')

    # Map the current UID on root inside the new namespace
    for id_, name in (
        (euid, 'uid'),
        (egid, 'gid'),
    ):
        # We can unfortunately only remap our own user as we wish. This means
        # that from within the namespace, we can only create files with that
        # owner and group, since other owners would not be mapped on anything
        # in the parent namespace.
        mapping = f'0 {id_} 1\n'.encode('ascii')
        with open(f'/proc/self/{name}_map', 'wb') as f:
            f.write(mapping)

    # Attempt to "mount --make-private /". This will prevent mount events
    # created under this process to propagate outside of the mount namespace.
    # That said, this call might fail if "/" is not a mount point (e.g. inside
    # a chroot or a container).
    MS_REC = 0x4000
    MS_PRIVATE = 0x40000
    mount_flags = MS_REC | MS_PRIVATE
    libc.mount(b"none", b"/", ffi.NULL, mount_flags, ffi.NULL);


def _unshare_wrapper(configure, f):
    # If we are already root, we don't need to do anything. This will increase
    # the odds of all that working in a CI environment inside an existing
    # container.
    if os.geteuid() != 0:
        _do_unshare()

    # We get all the parameters pickled manually, so that they are not
    # unpickled before we had a chance to _do_unshare(). If we did not do that,
    # pickle would import all the necessary modules to deserialize the objects,
    # leading to importing modules like pyarrow that create a background
    # thread, preventing the unshare(CLONE_NEWUSER) syscall from working.
    configure = pickle.loads(configure)

    # Configure logging module to get the records back in the parent thread
    # where they will be processed as usual. We need to do this before
    # unpickling the function and its data, as unpickling will trigger some
    # __new__/__init__ calls that Otherwise would run with the wrong logging
    # setup
    configure()

    f = pickle.loads(f)
    return f()


@contextlib.contextmanager
def _empty_main():
    """
    Context manager removing the ``__main__`` module from ``sys.modules`` and
    then putting it back.
    """
    mod = sys.modules['__main__']
    sys.modules['__main__'] = module_from_spec(
        ModuleSpec(
            name=mod.__name__,
            loader=None,
        )
    )
    try:
        yield
    finally:
        sys.modules['__main__'] = mod


def _logging_consumer(log_queue, stop_event):
    while not (stop_event.is_set() and log_queue.empty()):
        try:
            record = log_queue.get(timeout=0.01)
        except queue.Empty:
            pass
        else:
            logger = logging.getLogger(record.name)
            logger.handle(record)


def _logging_child_conf(log_queue, level):
    """
    Configure logging in a child Python process so that records are fed back to
    the parent process for handling.
    """
    handler = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


@contextlib.contextmanager
def _with_mp_logging():
    with multiprocessing.Manager() as manager:
        log_queue = manager.Queue(-1)
        conf = functools.partial(
            _logging_child_conf,
            log_queue,
            logging.getLogger().getEffectiveLevel()
        )

        consumer_thread = None
        try:
            stop_event = threading.Event()
            consumer_thread = threading.Thread(
                target=_logging_consumer,
                args=(log_queue, stop_event),
                daemon=True,
            )
            consumer_thread.start()
            yield conf
        finally:
            if consumer_thread:
                stop_event.set()
                consumer_thread.join()


def _with_unshare(f, args=tuple(), kwargs={}):
    # Setup a thread in the parent process to relay logging output
    with _with_mp_logging() as configure:
        # Terrible terrible hack:
        # New Python processes re-import __main__, and unpickling args will also
        # trigger modules import. The problem is, some modules such as pyarrow will
        # create a background thread upon import, making it impossible to create a
        # user namespace since it does not support multithreaded processes.
        #
        # We work around that problem by delaying the point at which the objects
        # are unpickled (triggering imports) by pickling them ourselves.
        ctx = multiprocessing.get_context('spawn')
        with _empty_main(), ctx.Pool(processes=1) as pool:
            configure = pickle.dumps(configure)
            f = pickle.dumps(functools.partial(f, *args, **kwargs))
            return pool.apply(
                _unshare_wrapper,
                args=(configure, f),
            )


def ensure_root(f, inline=False):
    """
    Decorator to ensure that the function is ran as root user.

    This works by spawning a new process, and running the function inside it
    after having entered a new user and mount namespaces unless the current
    user is already root. This has a pretty high cost and not everything will
    work as expected, but is good enough to setup e.g. a chroot as regular
    user.

    :param inline: If ``False``, assumes ``@ensure_root`` is used as a
        decorator. Otherwise, assume it's used inline as
        ``f2 = ensure_root(f1)``
    :type inline: bool

    .. note:: This decorator needs to be used alone, as the decorated function
        needs to be pickleable. This means the inner function cannot be a
        closure. If more decorators are needed, you can just make an "inner"
        function that has only one applied and put the other ones on a manually
        written wrapper.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return _with_unshare(f, args=args, kwargs=kwargs)

    if not inline:
        # Set the function name to be looked up as the wrapper.__wrapped__ by
        # pickle, since it's the only way it will find the correct function
        f.__qualname__ += '.__wrapped__'
    return wrapper

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
