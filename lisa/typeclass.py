# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020, Arm Limited and contributors.
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
This module provides a trait system known as typeclasses in Haskell and Scala,
and known as trait in Rust.

The fundamental idea is to decouple the followings:

    1. definition of an interface as a set of methods to implement.
    2. implementation of the aforementioned methods for a given class.
    3. the class definitions themselves.

Decoupling *2.* and *3.* allows providing implementation of the interface on
any type, including foreign types coming from other libraries, or even builtin
types. This is the core benefit from typeclasses as opposed to regular classes
in Object Oriented Programming. They allow extending existing types without
having to modify their inheritance hierarchy.

.. note:: The names of the concepts are drawn from Haskell typeclasses:

    * *typeclass*: This is the description of an interface, as a set of mandatory
      methods to implement, and optionally helper functions with default
      implementations. It's pretty close in concept to abstract base classes.

    * *superclass*: The mother typeclass of a given typeclass.

    * *instance*: This is the implementation of a given typeclass for a given
      (set of) type.

    * *values*: Values as opposed to types. Since *instance* is already used to
      refer to the implementation of a typeclass, we use the word *value*.

    * *type*: That is just a type, also known as *class* in Python.


Here is an example on how to work with typeclasses as provided by this module::

    from lisa.typeclass import TypeClass

    class FooBar(TypeClass):
        "Foobar interface"

        # TypeClass.required is an equivalent of abc.abstractmethod: It forces
        # implementations of a given set of method
        @TypeClass.required
        def my_method(self):
            pass

        # This helper can be used in the implementation of the typeclass, and
        # can be overriden by any instance.
        def extra_helper(self):
            return 42

    class ARandomClass:
        "Just a random class, it could be anything"
        pass

    # ``types`` can be either a tuple of types or a single type
    class ARandomClassFooBarInstance(FooBar, types=(ARandomClass, int)):
        "Implement the FooBar typeclass for both ARandomClass type and int at once."
        def my_method(self):
            return 'ARandomClass or int value'


    value = ARandomClass()

    # Both are equivalent
    # The @ version is more useful when combining multiple typeclasses on the fly
    value_as_foobar = FooBar(value)
    value_as_foobar = value @ FooBar

    # Inplace variant allows to "cast" the value directly.
    # These are all equivalent:
    # value @= FooBar
    # value = value @ FooBar
    # value = FooBar(value)

    # The typeclass machinery will dispatch the call to my_method() to the
    # right implementation
    value_as_foobar.my_method()

    # We also implemented FooBar for int type
    FooBar(3).my_method()

    # Raises a TypeError, since there is no instance for float
    FooBar(3.0).my_method()

    # Add an instance of FooBar for float type
    class FloatFooBarInstance(FooBar, types=float):
        def my_method(self):
            return 'float'

    # Now works once we added the instance
    FooBar(3.0).my_method()


Classmethod also work, so typeclasses can be used to define factory interfaces::

    from lisa.typeclass import TypeClass

    class FromString(TypeClass):
        "Build a value by parsing a string"

        @TypeClass.required
        @classmethod
        def from_str(cls, string):
            pass

    class IntFromStringInstance(FromString, types=int):
        @classmethod
        def from_str(cls, string):
            # Although cls is a value of type TypeProxy, it can be called just
            # like a regular class
            return cls(string)


    # Types can be cast just like values, so we can use the classmethods and
    # the staticmethods on them as well
    assert 33 == FromString(int).from_str('33')

A more advanced usage can involve a hierarchy of typeclasses that gets combined together::

    from lisa.typeclass import TypeClass

    class MyTP1(TypeClass):
        @TypeClass.required
        def meth1(self):
            pass

        @TypeClass.required
        def another_meth(self):
            pass


    class MyTP2(TypeClass):
        @TypeClass.required
        def meth2(self):
            pass

    class IntTP1Instance(MyTP1, types=int):
        def meth1(self):
            return 'int'

        def another_meth(self):
            return 42

    class IntTP2Instance(MyTP2, types=int):
        def meth2(self):
            return 'int'

        # Reuse an existing function implementation
        another_meth = IntTP1Instance.another_meth

    # Both are equivalent and allow creating a typeclass that provides
    # interfaces of both MyTP1 and MyTP2. If some methods are required by both
    # MyTP1 and MyTP2, the conflict is detected and a TypeError is raised:
    MyTP1AndMyTP2 = MyTP1 & MyTP2

    # This combined typeclass will automatically get the instances from its
    # superclasses
    class MyTP1AndMyTP2(MyTP1, MyTP2):
       pass

    # All are equivalent
    value = 2 @ (MyTP1 & MyTP2)
    value = 2 @ MyTP1AndMyTP2
    value = MyTP1AndMyTP2(2)
    value = (MyTP1 & MyTP2)(2)

    # We can now use the API of both MyTP1 and MyTP2
    value.meth1()
    value.meth2()


Note that it's possible to implement a typeclass for a type that has no values,
but for which ``isinstance(value, thetype)`` will return true. This can be
achieved using ``__instancecheck__`` or ``__subclasscheck__`` and is used in
particular by the abstract base classes provided by :mod:`collections.abc`.
:class:`lisa.conf.TypedList` is another example. Casting values "registered" as
instances of these types is expensive though, as validity of the cast depends
on the value itself. That means it's not possible to memoize the result of the
cast associated it with the type of the value.


One might wonder what casting a value to a typeclass gives. When possible, a
new value with a synthetic type is returned. That is implemented using a
shallow copy of the value, and then updating its ``__class__`` attribute. This
will provide native attribute lookup speed, and casting will be efficient. If
that is not possible (non-heap types, types using ``__slots__`` etc), an
instance of :class:`lisa.typeclass.ValueProxy` will be returned for values, and
a synthetic type will be created for types.

"""

import copy
import inspect
import itertools
import contextlib
from operator import attrgetter
from collections.abc import Mapping, MutableMapping, Sequence, Iterable

from lisa.utils import deduplicate

class TypeClassMeta(type):
    """
    Metaclass of all typeclasses.

    This implements most of the typeclass magic.

    :param name: Name of the typeclass or instance being created.
    :type name: str

    :param bases: tuple of superclasses of the typeclass being defined.
        When an instance is created, bases must have exactly one element, which
        is the typeclass being implemented.
    :type bases: tuple(type)

    :param dct: Dictionary of attributes defined in the body of the ``class``
        statement.
    :type dct: dict(str, object)

    :param types: Type or tuple of types for which the typeclass instance is
        provided.
    :type types: type or tuple(type) or None
    """
    def __new__(cls, name, bases, dct, *args, types=None, **kwargs):
        try:
            typeclass = bases[0]
        # That's TypeClass itself
        except IndexError:
            return super().__new__(cls, name, bases, dct, *args, **kwargs)

        # That's a typeclass being defined
        if types is None:
            dct.update(
                INSTANCES={},
                DEFAULTS={},
                REQUIRED=dict(),
            )

            superclasses = deduplicate(bases, keep_last=False)
            with contextlib.suppress(ValueError):
                superclasses.remove(TypeClass)
            dct['SUPERCLASSES'] = superclasses

            for typeclass in superclasses:
                conflicting = {
                    name
                    for name in dct['REQUIRED'].keys() & typeclass.REQUIRED.keys()
                    # If required method was specified in a base typeclass that
                    # happens to be shared, there is no problem
                    if dct['REQUIRED'][name] is not typeclass.REQUIRED[name]
                }
                if conflicting:
                    def flatten(l):
                        return list(itertools.chain.from_iterable(l))

                    # DFS traversal of superclass hierarchy, removing
                    # intermediate node that are just there to merge parent
                    # nodes without adding anything else. This avoids having
                    # intermediate classes created by __and__ for example, for
                    # better error reporting.
                    def expand(superclass):
                        # If that typeclass is an empty shim that just combines other typeclasses
                        if not (superclass.__dict__.keys() - _EmptyTypeClass.__dict__.keys()):
                            return flatten(map(expand, superclass.SUPERCLASSES))
                        else:
                            return [superclass]

                    superclasses = flatten(map(expand, superclasses))
                    superclasses = deduplicate(superclasses, keep_last=False)

                    def format_method(name):
                        return '{} (defined in: {} and {})'.format(
                            name,
                            dct['REQUIRED'][name].__qualname__,
                            typeclass.REQUIRED[name].__qualname__,
                        )
                    raise TypeError('Cannot merge typeclasses {} since the following methods conflict: {}'.format(
                        ', '.join(sorted(tp.__qualname__ for tp in superclasses)),
                        ', '.join(map(format_method, sorted(conflicting))),
                    ))
                else:
                    dct['DEFAULTS'].update(typeclass.DEFAULTS)
                    dct['REQUIRED'].update(typeclass.REQUIRED)

            typeclass = super().__new__(cls, name, bases, dct, *args, **kwargs)

            typeclass.REQUIRED.update({
                name: typeclass
                for name, attr in dct.items()
                if getattr(attr, '__required__', False)
            })

            not_copied = set(dict(inspect.getmembers(_EmptyTypeClass)).keys())
            not_copied |= dct['REQUIRED'].keys() | {'__qualname__', '__name__'}
            typeclass.DEFAULTS.update({
                attr: val
                for attr, val in dct.items()
                if attr not in not_copied
            })
            return typeclass
        # Someone tries to inherit from the typeclass to make an instance
        else:
            if len(bases) != 1:
                raise TypeError('A typeclass instance can only implement the methods of one typeclass, but multiple typeclasses were provided: {}'.format(
                    ', '.join(sorted(base.__qualname__ for base in bases))
                ))

            missing = typeclass.REQUIRED.keys() - dct.keys()
            if missing:
                raise NotImplementedError('Following methods are missing in {} instance and must be defined for instances of the {} typeclass: {}'.format(
                    name,
                    typeclass.__name__,
                    ', '.join(sorted(missing)),
                ))

            # Merge-in the typeclass default implementations before using it,
            # so each instance contains all the methods of the typeclass
            dct = {**typeclass.DEFAULTS, **dct}

            types = types if isinstance(types, Iterable) else [types]
            for type_ in types:
                # Create an instance for each type, with the type as base class.
                bases = (type_,)
                try:
                    instance = type(name, bases, dct, *args, **kwargs)
                # Some classes like bool cannot be subclassed. Work around by
                # listing all their attributes and making a new class that has
                # all of them.
                except TypeError:
                    total_dct = {**dict(inspect.getmembers(type_)), **dct}
                    instance = type(name, tuple(), total_dct, *args, **kwargs)

                typeclass.INSTANCES[type_] = (instance, dct)

                # Monkey patch the types so that the typeclass methods can be
                # called "natively" on them if wanted
                get_top_package = lambda mod: mod.split('.')[0]

                # Only add the attribute if it does not exist already on the
                # target class
                def update_attr(attr, val):
                    if not hasattr(type_, attr):
                        setattr(type_, attr, val)

                # If the instance is declared in the same top-level package,
                # update the type itself. This prevents foreign packages from
                # monkey patching types but allows instances anywhere in a
                # given package
                if get_top_package(type_.__module__) == dct['__module__']:
                    # Then the attributes defined in the instance
                    for attr, val in dct.items():
                        update_attr(attr, val)

            # We scavanged all what we needed, the class has just been used to
            # as a vehicle to create a scope but will not be used directly. It
            # will still live a secrete life internally for casting though.
            #
            # Instead, return a class that is equivalent to the typeclass but
            # with the docstring of the instance. This allows Sphinx to pick up
            # the instance's docstring.
            dct = {**dct, **{'__doc__': dct.get('__doc__')}}
            return type(name, (typeclass,), dct)

    @staticmethod
    def required(f):
        """
        Decorator used in a typeclass to flag a method to be required to be
        implemented by all instances.

        This is very similar to :func:`abc.abstractmethod`.
        """
        f.__required__ = True
        return f

    def __matmul__(self, obj):
        """
        Use the matrix multiplication operator (``@``) as a "cast" operator, to
        cast a value or a type to a typeclass.
        """
        return self(obj)

    # Also makes it work when operands are swapped.
    __rmatmul__ = __matmul__

    def __and__(self, other):
        """
        Allow quick combination of multiple typeclasses with bitwise ``&``
        operator.
        """
        class Combined(self, other):
            pass

        return Combined

class TypeClass(metaclass=TypeClassMeta):
    """
    Base class to inherit from to define a new typeclass.
    """
    def __new__(cls, obj):
        safe_to_memoize, instance, dct = cls._find_instance_dct(obj)
        # Shallow copy to allow "casting" to the right type. Using a made-up
        # class allows piggy backing on regular attribute lookup, which is much
        # faster than any pure-python __getattribute__ implementation
        try:
            new_obj = obj.__class__.__new__(obj.__class__)
            # Objects using __slots__ are not really handled anyway since
            # changing __class__ on them can lead to segfault in the
            # interpreter
            new_obj.__dict__ = copy.copy(obj.__dict__)
            new_obj.__class__ = instance
        # If obj.__class__ is not a heap type, it's not possible to "cast" the
        # value by modifying __class__ parameter (TypeError). Instead, we make
        # a proxy object, that has the typeclass attribute lookup implemented
        # with __getattribute__
        #
        # AttributeError can be raised if there is no __dict__ (e.g. if using
        # __slots__).
        except (TypeError, AttributeError):
            # Wrap the object in a proxy value that will implement the
            # typeclass-aware attribute lookup
            if isinstance(obj, type):
                new_obj = cls._make_type_proxy(obj, dct)
            else:
                new_obj = ValueProxy(obj, dct)

        return new_obj


    @staticmethod
    def _make_type_proxy(obj, dct):
        """
        Make a proxy object for given type.

        The proxy is itself a type inheriting from the original type, along
        with all the methods in ``dct``. ``__call__`` is overrident in the
        metaclass to make sure that invoking the type will yield instances of
        the original type.
        """
        class TypeProxyMeta(type):
            def __instancecheck__(self, x):
                return isinstance(x, obj)

            def __subclasscheck__(self, x):
                return issubclass(x, obj)

            # Allow calling the class as usual, which is necessary to
            # use factory classmethod that return new instances
            # (alternative constructors).
            __call__ = obj.__call__

        class TypeProxyBase(metaclass=TypeProxyMeta):
            pass

        try:
            class TypeProxy(obj, TypeProxyBase):
                pass
        # If we cannot inherit from the class (like bool), pick the first base
        # class that is suitable. That is a tad ugly but better than nothing
        except TypeError:
            # Make sure we get all the methods as on the original type we
            # wanted to subclass
            dct = {**dict(inspect.getmembers(obj)), **dct}
            for obj_ in inspect.getmro(obj):
                try:
                    class TypeProxy(obj_, TypeProxyBase):
                        pass
                except TypeError:
                    continue
                else:
                    break

        for attr, val in dct.items():
            with contextlib.suppress(TypeError, AttributeError):
                setattr(TypeProxy, attr, val)

        TypeProxy.__name__ = obj.__name__
        TypeProxy.__qualname__ = obj.__qualname__
        return TypeProxy

    @classmethod
    def _find_instance_dct(cls, obj):
        """
        Find the relevant instance and attribute dictionary for the given object.
        """
        from_type = isinstance(obj, type)
        if from_type:
            type_ = obj
        else:
            type_ = obj.__class__

        safe_to_memoize = True
        leaf_instance = None
        # Find the most derived class (according to MRO) with an instance
        # implemented for that typeclass
        for i, base in enumerate(type_.__mro__):
            try:
                instance, dct = cls.INSTANCES[base]
            except KeyError:
                pass
            else:
                # We got a "perfect" match on the first item of the MRO (a leaf
                # in class hierarchy), so we wont need to create any wrapper
                # class
                if i == 0:
                    leaf_instance = instance

                break
        # No instance was registered already
        else:

            # If we do have superclasses, we find their instance for the type
            # at hand and merge their dict
            dct = {}
            # Traverse the superclasses in reverse order, so that the leftmost
            # superclass has priority. This matches usual inheritance
            # precedence rule (i.e. MRO computed according to the C3 class
            # graph linearization algo).
            for typeclass in reversed(cls.SUPERCLASSES):
                safe_to_memoize_, instance_, dct_ = typeclass._find_instance_dct(obj)
                dct.update(dct_)
                # As soon as part of the methods are not safe to memoize, the
                # whole instance becomes unsafe
                safe_to_memoize &= safe_to_memoize_

            # Attempt with isinstance. It may succeed since some
            # classes register themselves as base classes without appearing
            # in the MRO of the "subclass". This can happen when
            # implementing __subclasscheck__ or __instancecheck__, such as
            # in abc.ABCMeta .
            instances = {
                instance: dct
                for cls, (instance, dct) in cls.INSTANCES.items()
                if isinstance(obj, cls)
            }

            if instances:
                # Do not register a new instance, since it's value-dependent.
                # Therefore, it has to be re-evaluated for every new value
                safe_to_memoize = False

                # Check that all dct are the same. If not, there is no way of
                # choosing one over the others, so we bail out
                dct_list = list(instances.values())
                if all(dct1 is dct2 for dct1, dct2 in zip(dct_list, dct_list[1:])):
                    dct.update(dct_list[0])
                else:
                    # TODO: attempt to find the most derived class among
                    #instances.keys(). If there is no most derived class,
                    #then raise the exception.
                    raise TypeError('Ambiguous instance for {} typeclass: {} could all be used'.format(
                        cls.__name__,
                        ', '.join(sorted(cls.__name__ for cls in instances.keys()))
                    ))
            else:
                # Check if all the required
                # methods are actually implemented. If so, it's enough to proceed.
                dct.update({
                    attr: getattr(type_, attr)
                    for attr in cls.REQUIRED.keys()
                    if hasattr(type_, attr)
                })

                # If there are some missing methods, then we cannot infer any
                # instance
                if cls.REQUIRED.keys() > dct.keys():
                    raise NotImplementedError('No instance of {} typeclass for {} type'.format(cls.__name__, type_.__name__))
                # If all required methods are there, carry on with that
                else:
                    dct = {**cls.DEFAULTS, **dct}

        if leaf_instance:
            instance = leaf_instance
        else:
            # Since no existing instance was registered for the specific class
            # of the object, we create a synthetic one for it, so attribute
            # resolution works as expected
            instance_name = '{}InstanceOf{}'.format(cls.__qualname__, obj.__class__.__name__)
            instance = type(instance_name, (obj.__class__,), dct)

            # Register that instance for faster future lookup
            if safe_to_memoize:
                cls.INSTANCES[type_] = (instance, dct)

        return (safe_to_memoize, instance, dct)


class ValueProxy:
    """
    Values of this class are returned when casting a value to a typeclass, if
    the value does not support shallow copy or ``__class__`` attribute
    assignment.

    It implements the modified attribute lookup, so we can use the typeclass
    methods. All other attribute lookups will go through untouched, except
    magic methods lookup (also known as dunder names).
    """
    def __init__(self, obj, dct):
        self._obj = obj
        self._instance_dct = dct

    def __getattribute__(self, attr):
        get = super().__getattribute__
        dct = get('_instance_dct')
        obj = get('_obj')

        try:
            val = dct[attr]
        # If that is not an method of the typeclass instance, fallback to
        # regular attribute lookup
        except KeyError:
            return obj.__class__.__getattribute__(obj, attr)
        # Otherwise, give priority to instance definition over inheritance
        else:
            # Bind descriptors
            if hasattr(val, '__get__'):
                if isinstance(obj, type):
                    # Bind to "self", so the method can use any other method of
                    # the typeclass
                    owner = self
                    value = None
                else:
                    owner = obj.__class__
                    # Bind to "self", so the method can use any other method of
                    # the typeclass
                    value = self

                return val.__get__(value, owner)
            else:
                return val


# Just to have something available to define the final _EmptyTypeClass
class _EmptyTypeClass:
    pass


# Serves to know the base set of attributes to not copy over when instantiating
# the typeclass
class _EmptyTypeClass(TypeClass):
    pass


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
