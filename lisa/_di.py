# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

import sys
import importlib
import contextlib
import functools
import inspect
import abc
import threading
from operator import itemgetter, attrgetter


class MissingDependencyError(Exception):
    """
    Exception raised when calling :meth:`~lisa._di.DependencyInjector.inject`
    if a mandatory dependency has not been provided.
    """
    def __init__(self, missing):
        self.missing = missing

    def _get_nested(self):
        def get(x):
            if isinstance(x, self.__class__):
                return [x] + get(x.__cause__)
            else:
                return []
        return get(self)

    @staticmethod
    def _expand_nested(xs):
        def expand(xs):
            if xs:
                x, *xs = xs
                if isinstance(x, str):
                    x = [x]

                yield from (
                    [x_] + xs_
                    for xs_ in expand(xs)
                    for x_ in x
                )
            else:
                yield []

        return list(expand(xs))

    def __str__(self):
        nested = map(attrgetter('missing'), self._get_nested())
        path = ', '.join(map('.'.join, self._expand_nested(nested)))
        return f'Missing mandatory dependency {path}'


class _ModuleProxy:
    """
    Proxy for a module that filters the attributes access with a list of names.
    """
    def __init__(self, mod, allowed):
        self.__mod = mod
        self.__allowed = allowed

    def __getattr__(self, name):
        # Trigger a regular AttributeError first if the attribute does not
        # exist
        x = getattr(self.__mod, name)

        if name not in self.__allowed:
            raise AttributeError(f'"{name}" is not exposed publicly')
        return x

    def __dir__(self):
        return sorted(self.__allowed)


class _ImportState:
    LOCK = threading.RLock()
    STATE = {}

    def __init__(self, inject, mod, depth):
        self.mod = mod
        self.inject = inject
        self.depth = depth

    @classmethod
    @contextlib.contextmanager
    def with_state(cls, mod_name, **kwargs):
        with cls.LOCK:
            prev_state = cls.STATE.get(mod_name)
            depth = prev_state.depth if prev_state else 0
            state = cls(depth=depth + 1, **kwargs)
            try:
                cls.STATE[mod_name] = state
                yield
            finally:
                if prev_state is None:
                    del cls.STATE[mod_name]
                else:
                    cls.STATE[mod_name] = prev_state

    @classmethod
    def get_state(cls, mod_name):
        with cls.LOCK:
            return cls.STATE[mod_name]


class DependencyInjector:
    """
    Allows injecting dependencies in a module, inspired by SML module system.

    It allows a module to be instantiated multiple times, each time with a
    different set of dependencies to provide implementations of the abstract
    base classes.

    :param dependencies: Mapping of names outside the of the ``with`` block to
        names inside the block. :meth:`~lisa._di.DependencyInjector.inject`
        keyword will be keys of that mapping, and the code inside the ``with``
        block will refer to them with the values of that mapping. ``None``
        value can be used to make a dependency private, i.e. non overridable
        with :meth:`~lisa._di.DependencyInjector.inject`. It will be available
        in the block under the same name.
    :type dependencies: dict(str, str or None)

    :param requires_abstract: If ``True``, dependencies that are abstract
        classes (or directly inheriting from :class:`abc.ABC`) will become
        mandatory. If the user does not inject them, it will result in an
        exception.
    :type requires_abstract: bool

    :param mod_name: Name of the current module. It should be detected
        automatically, but if it fails, it can be given the value of ``__name__``.
    :type mod_name: str or None

    **Example**::

        ##############
        # Module "foo"
        ##############

        import abc
        from lisa._di import DependencyInjector


        # Abstract Base Class that will be replaced by a concrete
        # implementation when the module is instantiated.
        class AbstractBase(abc.ABC):
            @abc.abstractmethod
            def a_method(self):
                pass

        def a_function():
            pass

        # Everything that is defined inside the "with" block will:
        #  * be part of the public API
        #  * be "cloned" when the module is instantiated with a set of
        #    dependencies. Everything that is defined outside of the block will
        #    be shared globally as singletons, just like names in a regular
        #    module.
        with DependencyInjector(
            # Mapping of names outside the of the "with" block to names inside the
            # block. DependencyInjector.inject() keyword will be keys of that mapping,
            # and the code inside the "with" will refer to them with the values of that
            # mapping. ``None`` value can be used to make a dependency private, i.e.
            # non overridable with DependenciesInjector.inject(). It will be available
            # in the block under the same name.
            #
            # The code in the "with" block cannot refer to any other name, as
            # the namespace is cleared and only the explicitly-defined
            # dependencies are left. It is however possible to import modules
            # inside the with block to access unrelated modules.
            {
                'AbstractBase': 'Base',
                # It is possible to depend on an name external to the block
                # without allowing the user to override it by using None.
                'a_function': None,
            }
        ) as DI:
            class Foo(Base):
                def another_method(self):
                    return self.a_method()

            f = a_function

        ###########
        # User code
        ###########

        import foo

        class CustomBase(foo.AbstractBase):
            def a_method(self):
                print('custom hello')

        # In order to make use of that module, we need to inject dependencies
        # that satisfy the abstract base class API
        custom_foo = foo.DI.inject(AbstractBase=CustomBase)

        # This will print "custom hello"
        custom_foo.Foo().another_method()

        # We can reinstanciate the module at will, and use both "custom_foo"
        # and "custom_foo2" at the same time.
        #
        # class CustomBase2(...): ...
        # custom_foo2 = foo.DI.inject(AbstractBase=CustomBase2)
    """
    def __init__(self, dependencies, requires_abstract=True, mod_name=None):
        self._requires_abstract = requires_abstract

        if mod_name is None:
            mod_name = sys._getframe(1).f_globals.get('__name__', '__main__')

        self._mod_name = mod_name
        # Since we don't update sys.modules when re-importing modules, we get
        # the reference over it that was setup by _copy_mod()
        try:
            mod = _ImportState.get_state(mod_name).mod
        except KeyError:
            mod = sys.modules[mod_name]

        self._spec = importlib.util.spec_from_loader(mod.__name__, mod.__loader__)
        self._mod = mod

        # _ImportState.STATE has been filled by inject() so that when we
        # re-execute the code of the module, we have the list of injects
        # available from the beginning.
        try:
            self._import_inject = _ImportState.get_state(mod_name).inject
        except KeyError:
            self._import_inject = {}

        self._state = None
        self._mod_attrs_at_enter = dict()
        self._defined_names = set()
        self._dependencies = dict()
        self._user_mapping = dependencies

    def _get_defined_names(self):
        return {
            k
            for k, v in self._mod.__dict__.items()
            # Do not display the dependency injector
            if v is not self
        }

    def _resolve_name(self, name):
        mod = self._mod
        # Trigger an AttributeError if the module does not have the name
        # already. We want that even if we are injecting a name, since that
        # injection is not supposed to be accessible yet
        try:
            from_mod = getattr(mod, name)
        except AttributeError as e:
            # It could be one of the names that have been removed in __enter__
            try:
                from_mod = self._mod_attrs_at_enter[name]
            except KeyError:
                raise e

        try:
            x = self._import_inject[name]
        except KeyError:
            x = from_mod
            # If the dependency is a DependencyInjector, we just return the
            # underlying module, so that names can be resolved normally when
            # the module is first imported. Once we are re-importing it with
            # injection, the result of x.inject() will be used instead.
            if isinstance(x, DependencyInjector):
                x = x._mod

        return (from_mod, x)

    def __enter__(self):
        if self._state == 'exited':
            raise RuntimeError('A dependency injector can only be used once')
        self._state = 'entered'

        mod = self._mod

        names_at_enter = self._get_defined_names()

        # Remove the names defined before the "with" statement, to force the
        # user to use declare their dependencies.
        #
        # Failing to do so would lead to using an object that is going to be
        # discarded and replaced by the one from the original module, leading
        # to nasty issues (especially if it was inherited from)
        self._mod_attrs_at_enter = {
            name: getattr(mod, name)
            for name in names_at_enter
        }
        for name in names_at_enter:
            if not (name.startswith('__') and name.endswith('__')):
                delattr(self._mod, name)

        for name, new_name in self._user_mapping.items():
            # Only allow names defined before __enter__ to be overridden
            if name not in names_at_enter:
                raise ValueError(f'Cannot use {name} as it was not defined before entering the "with" block')
            from_mod, x = self._resolve_name(name)
            # Private dependency, so we don't expose it in self._dependencies,
            # but we do allow its access under its current name.
            if new_name is None:
                new_name = name
            else:
                self._dependencies[name] = from_mod
            setattr(mod, new_name, x)

        return self

    def __exit__(self, *args):
        self._state = 'exited'

        names_at_exit = self._get_defined_names()
        names_at_enter = self._mod_attrs_at_enter.keys()
        self._defined_names = (names_at_exit - names_at_enter)

        # Restore the attributes that were removed in __enter__ unless they
        # have been overridden.
        mod = self._mod
        for name, x in self._mod_attrs_at_enter.items():
            if not hasattr(mod, name):
                setattr(mod, name, x)

    def _copy_mod(self, inject):
        spec = self._spec
        mod = importlib.util.module_from_spec(spec)
        assert mod is not self._mod

        # Equivalent to spec.loader.exec_module(mod), except that we control
        # what we inject in the module namespace.
        #
        # Unfortunately, customzing globals passed to exec() is not really
        # supported at the moment, so we cannot rely on that:
        # https://bugs.python.org/issue44749
        # If that bug ever get solved, we will not need _DependenciesProxy anymore as
        # we will be able to intercept access to self._mod globals directly, in
        # order to inject injectments
        #
        # code = mod.__loader__.get_code(mod.__name__)
        # if code is None:
        #     raise RuntimeError('module code is None')
        # dct = _ModuleDict(
        #     init=mod.__dict__,
        #     inject=inject,
        # )
        # exec(code, dct)
        # mod.__dict__.update(dct)


        with _ImportState.with_state(self._mod_name, mod=mod, inject=inject):
            spec.loader.exec_module(mod)

        # Reset all the things that are not going to be impacted by the
        # dependency injection. This avoids proliferation of identical classes,
        # and allows defining e.g. a base class outside of the "with" statement
        # that can be used for isinstance(). If we did not do that, each
        # instance of the module would get a different base class (all sharing
        # the same name and code), which would render it useless.
        #
        # Note that classes defined outside of the "with" statement will be the
        # same before and after a inject(), since _DependenciesProxy defaults to
        # looking up in the original module. This preserves a sane isinstance()
        # behavior.
        preserve_change = self._defined_names | inject.keys()
        for name, x in inspect.getmembers(self._mod):
            if name not in preserve_change:
                setattr(mod, name, x)

        return mod

    def _check_state(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if self._state == 'entered':
                raise RuntimeError(f'{f.__qualname__} cannot be called until the end of the "with" statement')
            elif self._state != 'exited':
                raise RuntimeError(f'{self.__class__.__name__} must be used as a context manager')

            return f(self, *args, **kwargs)
        return wrapper

    @_check_state
    def inject(self, **kwargs):
        """
        Inject the given dependencies into the module, creating a new
        module-like instance.

        :Variable keyword arguments: The keys of the ``dependencies`` mapping
            passed to :class:`lisa._di.DependencyInjector`
        """
        inject = kwargs

        # Handle calling inject() from a module that is also being injected to,
        # in order to avoid infinite recursion
        try:
            depth = _ImportState.get_state(self._mod_name).depth
        except KeyError:
            depth = 0

        if depth:
            mod = self._mod
        else:
            deps = self._dependencies

            if self._requires_abstract:
                def isabstract(cls):
                    return isinstance(cls, type) and (
                        # When there are some non-overridden abstract methods
                        inspect.isabstract(cls) or
                        # The class inherits directly from abc.ABC. The class may
                        # not be actually abstract given that not everything can be
                        # tracked by abc,abstractmethod() & friends, such as
                        # instance attributes.
                        abc.ABC in cls.__bases__
                    )

                missing = sorted(
                    name
                    for name, cls in deps.items()
                    if isabstract(cls) and name not in inject
                )
                if missing:
                    raise MissingDependencyError(missing)

            for name, x in inject.items():
                qualname = f'{self._mod_name}.{name}'
                try:
                    base = deps[name]
                except KeyError:
                    raise ValueError(f'Cannot override {qualname}, can only override: {", ".join(deps.keys())}')

                # When the injected object is a class, make sure it is a subclass
                # of the dependency placeholder
                if isinstance(base, type) and not issubclass(x, base):
                    base_qualname = f'{base.__module__}.{base.__qualname__}'
                    inject_qualname = f'{x.__module__}.{x.__qualname__}'
                    raise TypeError(f'{inject_qualname} for dependency {name} must be a subclass of {base_qualname}')


            # Handle dependencies on other DependencyInjector, so that the user
            # can provide a mapping to inject in the nested DependencyInjector
            for name, di in deps.items():
                if isinstance(di, DependencyInjector):
                    try:
                        inject[name] = di.inject(**inject.get(name, {}))
                    except MissingDependencyError as e:
                        # Ensure __cause__ is set, so that the exception will
                        # print the whole path
                        raise MissingDependencyError([name]) from e

            mod = self._copy_mod(inject)

        public_names = self._dependencies.keys() | self._defined_names
        return _ModuleProxy(mod, allowed=public_names)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
