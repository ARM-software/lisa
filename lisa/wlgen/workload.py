# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, Arm Limited and contributors.
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

import os
from pathlib import Path
from shlex import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from subprocess import CalledProcessError
import functools
import uuid
import warnings
import contextlib
import shutil
import inspect

from devlib.utils.misc import list_to_mask

from lisa.utils import Loggable, ArtifactPath, deprecate, PartialInit, sig_bind


class _WorkloadRunCMDecorator:
    def __init__(self, gen_f):
        self.gen_f = gen_f
        # Set __doc__ and the likes
        functools.update_wrapper(self, gen_f)

    def __call__(self, wload, *args, **kwargs):
        return _WorkloadRunCM(wload, self.gen_f, args, kwargs)

    def __get__(self, instance, owner=None):
        """
        Behave like a regular method
        """
        if instance is None:
            return self
        else:
            return functools.partial(self, instance)


class _NotSet:
    def __init__(self, excep):
        self.excep = excep


class _WorkloadRunCM(Loggable):
    def __init__(self, wload, f, args, kwargs):
        self.gen = f(wload, *args, **kwargs)
        self._bg_cmd = None
        self._futures = None
        self._output = _NotSet(RuntimeError('The output cannot be used inside the "with" statement that computes it'))
        self.name = f.__qualname__.split('.')[0]
        self.wload = wload

    @property
    def output(self):
        out = self._output
        if isinstance(out, _NotSet):
            raise out.excep
        else:
            return out

    def __getattr__(self, attr):
        """
        Make the context manager look like the
        :class:`devlib.connection.BackgroundCommand` it wraps.
        """
        if attr in ('stdout', 'stderr'):
            raise AttributeError(f'Attribute "{attr}" is not available, use "output" attribute outside the "with" block to get the postprocessed output')
        return getattr(self._bg_cmd, attr)

    @staticmethod
    def _read_streams(bg_cmd):

        def read(pipe):
            return pipe.read()

        pipes = dict(
            stderr=bg_cmd.stderr,
            stdout=bg_cmd.stdout,
        )
        executor = None
        try:
            executor = ThreadPoolExecutor(max_workers=len(pipes))
            futures = {
                name: executor.submit(read, pipe)
                for name, pipe in pipes.items()
            }
        except BaseException:
            # If something bad happened, ensure the executor is correctly
            # shutdown and all futures canceled to avoid leaking any thread
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            # Ask the executor to free the resources as soon as all the futures
            # are completed
            if executor is not None:
                executor.shutdown(wait=False)

        return futures

    def __enter__(self):
        try:
            bg_cmd = next(self.gen)
        except StopIteration:
            raise RuntimeError('Generator did not yield')

        bg_cmd = bg_cmd.__enter__()
        self._bg_cmd = bg_cmd
        futures = self._read_streams(bg_cmd)
        self._futures = futures
        return self

    def __exit__(self, *exc_info):
        inner_exit = self._bg_cmd.__exit__
        wload = self.wload
        logger = self.get_logger()

        try:
            suppress = inner_exit(*exc_info)
        except BaseException as e:
            exc_info = (type(e), e, e.__traceback__)
        else:
            if suppress:
                exc_info = (None, None, None)

        type_, value, traceback = exc_info

        returncode = self._bg_cmd.poll()

        if exc_info[0] is not None:
            try:
                self.gen.throw(*exc_info)
            except StopIteration as e:
                if e is value:
                    self._output = _NotSet(e)
                    return False
                else:
                    self._output = e.value
                    return True
            except BaseException as e:
                # Place a "bomb" value: if the user tries to access
                # "self.output", the exception will be raised again
                self._output = _NotSet(e)
                # __exit__ is not expected to re-raise the exception it was
                # given, instead it returns a falsy value to indicate it should
                # not be swallowed
                if e is value:
                    return False
                else:
                    raise
            # This cannot happen: throw() has to raise the exception or swallow
            # it and then later raise a StopIteration because it is finished
            else:
                assert False
        else:
            try:
                futures = self._futures
            except ValueError:
                results = dict(stdout=None, stderr=None)
            else:
                results = {
                    name: future.result()
                    for name, future in futures.items()
                }
                if wload._settings['log_std_streams']:
                    # Dump the stdout/stderr content to log files for easier
                    # debugging
                    for name, content in results.items():
                        path = ArtifactPath.join(wload.res_dir, f'{name}.log')
                        logger.debug(f'Saving {name} to {path}...')

                        with open(path, 'wb') as f:
                            f.write(content)

            # For convenience and to avoid depending too much on devlib's
            # BackgroundCommand in simple cases
            results['returncode'] = returncode

            if returncode:
                action = lambda: self.gen.throw(
                    CalledProcessError(
                        returncode=returncode,
                        cmd=f'<Workload {self.name}>',
                        output=results['stdout'],
                        stderr=results['stderr'],
                    )
                )
            else:
                action = lambda: self.gen.send(results)

            try:
                action()
            except StopIteration as e:
                output = e.value
            except Exception as e:
                output = _NotSet(e)

            self._output = output


class _WorkloadBase:
    """
    :meta public:

    Dummy base class so that :class:`Workload` is processed by
    ``__init_subclass__`` as well.
    """
    def __init_subclass__(cls, *args, **kwargs):
        """
        Automatically decorate ``_run()`` so that it returns a context
        manager.
        """
        try:
            _run = cls.__dict__['_run']
        except KeyError:
            pass
        else:
            cls._run = _WorkloadRunCMDecorator(_run)

        tools = {
            tool
            for cls in inspect.getmro(cls)
            for tool in getattr(cls, 'REQUIRED_TOOLS', [])
        }
        cls.REQUIRED_TOOLS = sorted(tools)

        super().__init_subclass__(*args, **kwargs)


class Workload(_WorkloadBase, PartialInit, Loggable):
    """
    Handle the execution of a command on a target, with custom output
    processing.

    :param target: The Target on which to execute this workload
    :type target: Target

    :param name: Name of the workload. Useful for naming related artefacts.
    :type name: str or None

    :param res_dir: Host directory into which artifacts will be stored
    :type res_dir: str or None

    :param run_dir: Target directory into which artifacts will be created.
    :type run_dir: str or None

    :param cpus: CPUs on which to restrict the workload execution (taskset)
    :type cpus: list(int) or None

    :param cgroup: cgroup in which to run the workload
    :type cgroup: str or None

    :param as_root: Whether to run the workload as root or not
    :type as_root: bool

    :param timeout: Timeout in seconds for the workload execution.
    :type timeout: int

    .. note:: A :class:`Workload` instance can be used as a context manager,
      which ensures :meth:`cleanup` is eventually invoked.

    **Usage example**::

        >>> printer = Printer(target, "test")
        >>> output = printer.run()
        INFO    : Printer      : Execution start: echo 42
        INFO    : Printer      : Execution complete
        >>> print(output)
        42\r\n
    """

    REQUIRED_TOOLS = []
    """
    The tools required to execute the workload. See
    :meth:`lisa.target.Target.install_tools`.
    """
    def __init__(self,
        *,
        target,
        name=None,
        res_dir=None,
        run_dir=None,
        cpus=None,
        cgroup=None,
        as_root=False,
        timeout=None,
        command=None,
        log_std_streams=True,
        wipe_run_dir=True,
        wipe_res_dir=False,
    ):
        res_dir = res_dir if res_dir else target.get_res_dir(
            name='{}{}'.format(
                self.__class__.__qualname__,
                f'-{name}' if name else '')
        )
        name = name or self.__class__.__qualname__

        self.target = target
        self.name = name
        self.res_dir = res_dir
        self._wipe_run_dir = wipe_run_dir
        self._wipe_res_dir = wipe_res_dir
        self._setup_cm = None

        # Generic settings that will be used by _basic_run()
        self._settings = dict(
            cpus=cpus,
            cgroup=cgroup,
            as_root=as_root,
            timeout=timeout,
            command=command,
            log_std_streams=log_std_streams,
        )

        if run_dir is None:
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            uuid_ = uuid.uuid4().hex
            run_dir = f"{now}_{uuid_}"

        wlgen_dir = self.target.path.join(
            target.working_directory,
            'lisa', 'wlgen',
        )
        self.run_dir = os.path.join(wlgen_dir, run_dir)

        # Deprecated
        self._output = ''

    @property
    def _deployed(self):
        return self._setup_cm is not None

    @contextlib.contextmanager
    def _setup(self):
        """
        Context manager function called to setup the target before the
        execution, and cleanup anything that was setup.

        .. note:: Ensure you call ``super()._setup()`` as a context manager in
            custom implementation::

                @contextlib.contextmanager
                def _setup(self):
                    with super()._setup():
                        ...
                        yield
                        ...
        """
        os.makedirs(self.res_dir, exist_ok=True)
        self.target.execute(f'mkdir -p {quote(self.run_dir)}')
        self.get_logger().info(f"Created workload's run target directory: {self.run_dir}")
        self.target.install_tools(self.REQUIRED_TOOLS)
        try:
            yield
        finally:
            if self._wipe_run_dir:
                self.wipe_run_dir()

            if self._wipe_res_dir:
                shutil.rmtree(self.res_dir)

    @property
    @deprecate('Processed output is returned by run() or by the ".output" attribute of the value returned by the run_background() context manager', deprecated_in='2.0', removed_in='2.1')
    def output(self):
        return self._output

    _ZOMBIE_CM_SET = set()

    def deploy(self):
        """
        Deploy the workload on the target.

        If not called manually, it will be called:

            * If the workload is used as a context manager, in
              ``__enter__``.
            * If not, in :meth:`run` or :meth:`run_background`.

        Calling it manually ahead of time makes can allow less garbage while
        tracing during the execution of the workload.

        .. note:: This method can be called any number of times, but will only
            have an effect the first time.

        .. note:: This method should not be overridden, see ``_setup()``.
        """
        if self._deployed:
            return
        else:
            cm = self._setup()
            # We have to hold onto the context manager at the class level, so
            # that it will survive even if the workload is garbage collected.
            # This is made necessary by the fact that generator iterators
            # receive a GeneratorExit exception when they are garbage
            # collected, meaning the workload would be "automatically cleaned
            # up". Unfortunately, this can lead to deadlocks in devlib if it
            # happens during the execution of another command.
            #
            # TODO: re-evalute this in light of the (future) discussion and
            # conclusion of:
            # https://github.com/ARM-software/devlib/issues/528
            self._ZOMBIE_CM_SET.add(cm)

            self._setup_cm = cm
            cm.__enter__()

    def cleanup(self):
        """
        Remove all the artifacts installed on the target with :meth:`deploy`.

        If not called manually, it will be called in :meth:`__exit__` or
        ``__del__`` if the workload is used as a context manager.

        .. note:: This method can be called any number of times, but will only
            have an effect the first time and if the target was deployed.

        .. note:: This method should not be overridden, see ``_setup()``.
        """
        if getattr(self, '_deployed', False):
            cm = self._setup_cm
            self._setup_cm = None
            self._ZOMBIE_CM_SET.remove(cm)
            cm.__exit__(None, None, None)

    def run_background(self):
        """
        Run the command asynchronously and give control back to the caller.

        Returns a transparent wrapper for
        :class:`devlib.connection.BackgroundCommand`.

        **Example**::

            wload = Workload(...)

            with wload.run_background() as bg:
                # do something. "bg" can be monitored, except for stdout/stderr
                # which are captured for later post-processing by the class.
                ...
                time.sleep(42)

            # Post-processed output, which would be equal to:
            # print(wload.run())
            print(bg.output)

        .. note:: Subclasses should implement ``_run()``.
        """
        self.deploy()
        return self._run()

    def run(self, cpus=None, cgroup=None, as_root=False, timeout=None):
        """
        Run the workload and returns the post-processed output.

        Calls :meth:`deploy` if it has not already been done. stdout and stderr
        will be saved into files in ``self.res_dir``

        .. note:: Subclasses should implement ``_run()``.
        """

        # For backward compatibility.
        # Note: if the values are the same as the current state, the same
        # instance will be returned so it's cheap
        compat = dict(
            cpus=cpus,
            cgroup=cgroup,
            timeout=timeout,
        )
        if any(compat.values()):
            params = ', '.join(
                f'{param}={val}'
                for param, val in compat.items()
                if val
            )
            warnings.warn(f'Workload.run({params}) parameters are deprecated, please pass them to the constructor instead', DeprecationWarning)

        self = self(**compat)

        self.deploy()
        with self._run() as x:
            pass

        # Only there to satisfy a deprecated API, do not rely on that in any
        # new code
        self._output = x.output

        return x.output

    def _run(self):
        """
        Run the workload.

        This method must be implemented by all subclasses and is used by
        :meth:`run` and :meth:`run_background`.

        The following constraints apply:

            * It must yield an instance of :class:`devlib.connection.BackgroundCommand`.
              This can be obtained using :meth:`_basic_run` or using
              :meth:`lisa.target.Target.background` directly.

            * It must only yield once.

            * It must return the post-processed output of the command.

        .. note:: This API mirrors what is expected from functions decorated
            with :func:`contextlib.contextmanager`, except that they have to
            return a post-processed value, rather than having their return
            value ignored.

        **Example**::

            def _run(self):
                cmd = 'echo hello world'
                try:
                    out = yield self._basic_run(cmd)
                except subprocess.CalledProcessError as e:
                    # The command returned with non-zero exit code.
                    ...

                # "out" is a dict with the following keys:
                #   * "stdout": content of captured stdout of the command, as a bytestring
                #   * "stderr": content of captured stderr of the command, as a bytestring
                #   * "returncode": return code of the command


                # Arbitrary post-processing is allowed. If there is no
                # interesting output to return, the method should return None,
                # so that it can be repurposed in the future to something
                # useful.
                # DO NOT RETURN stdout unless it's something truly interesting
                # and of use to the caller. The content of stdout is already
                # dumped to a file for debugging purposes.
                return out['stdout'].split()

        """
        yield self._basic_run()
        return None

    def wipe_run_dir(self):
        """
        Wipe all content from the ``run_dir`` target directory and all its
        empty parents.

        .. note :: This function should only be called directly in interactive
            sessions. For other purposes, use :class:`Workload` instances as a
            context manager.
        """
        logger = self.get_logger()
        logger.info(f"Wiping target run directory: {self.run_dir}")
        self.target.remove(self.run_dir)

        # Also get rid of empty parent folders
        path = Path(self.run_dir)
        self.target.execute('rmdir -p quote({path.parent})', check_exit_code=False)

    def __enter__(self):
        self.deploy()
        return self

    # def __del__(self):
        # This cannot be relied upon, but might improve things
        # TODO: reenable that once this issue is cleared out:
        # https://github.com/ARM-software/devlib/issues/528
        # self.cleanup()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleanup the artifacts of the workload on the target.
        """
        self.cleanup()

    def _basic_run(self, **kwargs):
        """
        Basic run function to be used in subclasses' implementation of
        ``_run()``.

        By default, it will use the settings saved in ``_settings`` attribute
        by :class:`Workload`'s ``__init__``, but they can all be overridden
        manually with keyword arguments if that is really necessary.
        """
        logger = self.get_logger()
        target = self.target

        settings = {
            **self._settings,
            **kwargs
        }
        as_root = settings.get('as_root')
        timeout = settings.get('timeout')
        command = settings.get('command')
        cpus = settings.get('cpus')
        cgroup = settings.get('cgroup')

        if command is None:
            raise ValueError('command must not be None')

        # Log the "clean" unmodified command. If the user wants the details,
        # debug log from devlib will provide it
        logger.info(f"Execution start: {command}")

        if cpus:
            target.install_tools(['taskset'])
            cpumask = list_to_mask(cpus)
            taskset_cmd = f"taskset {quote(f'0x{cpumask:x}')}"
            command = f'{taskset_cmd} {command}'

        if cgroup:
            command = target.cgroups.run_into_cmd(cgroup, command)

        command = f'cd {quote(self.run_dir)} && {command}'

        bg = target.background(command, as_root=as_root, timeout=timeout)
        return bg

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
