# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
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
Sphinc documentation building helpers.
"""

import enum
import io
import contextlib
import subprocess
import inspect
import itertools
import functools
import re
import types
import abc
import warnings
from collections.abc import Mapping
from collections import ChainMap
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from operator import attrgetter, itemgetter
import collections
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
import importlib
from pathlib import Path
import builtins
import copy
import typing

from sphinx.util.docutils import SphinxDirective
from docutils.parsers.rst import directives
from docutils import nodes
from docutils.statemachine import ViewList

from sphinx.util.nodes import nested_parse_with_titles
from sphinx.ext.autodoc import exclude_members_option
from sphinx.pycode import ModuleAnalyzer
from sphinx.errors import PycodeError

import lisa
import lisa.analysis
from lisa.analysis.base import AnalysisHelpers, TraceAnalysisBase, measure_time
from lisa.utils import get_subclasses, import_all_submodules, _DEPRECATED_MAP, get_obj_name, groupby, get_short_doc, order_as, is_link_dead, resolve_dotted_name, get_sphinx_role, _get_parent_namespace, _get_parent_namespaces, get_parent_namespace, memoized, _DelegatedBase, ffill, fixedpoint, deduplicate, fold
from lisa.trace import TraceEventCheckerBase
from lisa.conf import KeyDesc, SimpleMultiSrcConf, TopLevelKeyDesc
from lisa.version import format_version
import lisa._git


def _sphinx_recursive_parse(directive, viewlist):
    node = nodes.Element()
    nested_parse_with_titles(directive.state, viewlist, node)
    return node.children

class RecursiveDirective(SphinxDirective):
    """
    Base class helping nested parsing.

    Options:

    * ``literal``: If set, a literal block will be used, otherwise the text
      will be interpreted as reStructuredText.
    """
    option_spec = {
        'literal': directives.flag,
    }

    def parse_nested(self, txt, source=None):
        """
        Parse text as reStructuredText if the ``literal`` option is not set.

        Otherwise, interpret the text as a line block.
        """
        if 'literal' in self.options:
            node = nodes.literal_block(txt, txt, classes=[])
            # Avoid syntax highlight
            node['language'] = 'text'
            return [node]
        else:
            viewlist = ViewList(txt.splitlines(), source)
            return _sphinx_recursive_parse(self, viewlist)


class WithRefCtxDirective(SphinxDirective):
    """
    Allow temporarily switching to a different current class and module for
    reference resolution purpose.
    """
    has_content = True

    option_spec = {
        'module': directives.unchanged,
        'class': directives.unchanged,
    }

    def run(self):
        ctx = {
            'py:module': self.options.get('module'),
            'py:class': self.options.get('class'),
        }
        ctx = {
            k: v
            for k, v in ctx.items()
            if v
        }
        notset = object()
        old = {
            k: self.env.ref_context.get(k, notset)
            for k in ctx.keys()
        }

        try:
            self.env.ref_context.update(ctx)
            nodes = _sphinx_recursive_parse(self, self.content)
        finally:
            for k, v in old.items():
                if v is notset:
                    del self.env.ref_context[k]
                else:
                    self.env.ref_context[k] = v
        return nodes


class ExecDirective(RecursiveDirective):
    """
    reStructuredText directive to execute the specified python code and insert
    the output into the document::

        .. exec::

            import sys
            print(sys.version)

    This directive will also register a ``lisa-exec-state`` hook that will be
    called with no extra parameters. The return value will be made available as
    the ``state`` global variable injected in the block of code.

    Options:

    * ``literal``: If set, a literal block will be used, otherwise the text
      will be interpreted as reStructuredText.
    """
    has_content = True

    def run(self):
        stdout = io.StringIO()
        code = '\n'.join(self.content)
        code = dedent(code)

        state = self.env.app.emit_firstresult('lisa-exec-state')
        with contextlib.redirect_stdout(stdout):
            exec(code, {'state': state})
        out = stdout.getvalue()
        return self.parse_nested(out)


class RunCommandDirective(RecursiveDirective):
    """
    reStructuredText directive to execute the specified command and insert
    the output into the document::

        .. run-command::
           :capture-stderr:
           :ignore-error:
           :literal:

           exekall --help

    Options:

    * ``literal``: If set, a literal block will be used, otherwise the text
      will be interpreted as reStructuredText.
    * ``capture-stderr``: If set, stderr will be captured in addition to stdout.
    * ``ignore-error``: The return status of the command will be ignored.
      Otherwise, it will raise an exception and building the
      documentation will fail.
    """
    has_content = True
    option_spec = {
        'ignore-error': directives.flag,
        'capture-stderr': directives.flag,
        'literal': directives.flag,
    }

    def run(self):
        options = self.options
        if 'capture-stderr' in options:
            stderr = subprocess.STDOUT
        else:
            stderr = None

        check = False if 'ignore-error' in options else True

        cmd = '\n'.join(self.content)

        out = subprocess.run(
            cmd, shell=True, check=check,
            stdout=subprocess.PIPE, stderr=stderr,
        ).stdout.decode('utf-8')

        return self.parse_nested(out, cmd)


@functools.lru_cache(maxsize=128)
def sphinx_module_attrs_doc(mod):
    modname = mod.__name__
    if modname in ('builtins', '__main__'):
        return {}
    else:
        try:
            analyzer = ModuleAnalyzer.for_module(modname)
        # Some extension modules don't have source code and therefore cannot be
        # found by ModuleAnalyzer
        except PycodeError:
            return {}
        else:
            attrs = {
                '.'.join(x for x in name if x): '\n'.join(doc)
                for name, doc in analyzer.find_attr_docs().items()
            }
            return attrs


def relname(parent, child):
    return _relname(
        parent=get_obj_name(parent),
        child=get_obj_name(child),
    )

def _relname(parent, child):
    return '.'.join(
        a
        for a, b in itertools.zip_longest(
            child.split('.'),
            parent.split('.')
        )
        if a != b and a is not None
    )


_GETATTR_NOTSET = object()
def silent_getattr(obj, attr, default=_GETATTR_NOTSET):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return getattr(obj, attr)
        except AttributeError:
            if default is _GETATTR_NOTSET:
                raise
            else:
                return default


def silent_hasattr(obj, attr):
    try:
        silent_getattr(obj, attr)
    except AttributeError:
        return False
    else:
        return True


def _resolve_dotted_name(name):
    return resolve_dotted_name(
        name,
        getattr=silent_getattr,
    )


def _get_delegated_members(cls, avoid_delegated=None):
    if avoid_delegated:
        return _do_get_delegated_members(cls, avoid_delegated)
    else:
        return _get_delegated_members_memoized(cls)


@functools.lru_cache(maxsize=2048)
def _get_delegated_members_memoized(cls):
    return _do_get_delegated_members(cls, None)


def _do_get_delegated_members(cls, avoid_delegated):
    if issubclass(cls, _DelegatedBase) and cls._ATTRS_DELEGATED_TO_CLASSES:
        # Make sure we do not include anything we inherited from.
        avoid_delegated = set(avoid_delegated or [])

        # Including the entire MRO prevents O(N^2) complexity in case a
        # subclass delegates to one of its bases. This is ok to do since
        # whatever these bases provides, we already have via inheritance.
        avoid_delegated.update(inspect.getmro(cls))

        classes = deduplicate(
            [
                __cls
                for _cls in cls._ATTRS_DELEGATED_TO_CLASSES
                for __cls in (_cls, *get_subclasses(_cls, mro_order=True, only_leaves=True))
                if __cls not in avoid_delegated
            ],
            keep_last=False,
        )

        # Any class we already visited is to be avoided, so that we avoid
        # infinite recursion in case of cycles in the delegation graph. This
        # can happen if sibling classes A and B both define a member (or
        # inherit from it). In that case, the member would appear to be
        # inherited from A from the point of view of B and vice versa.
        avoid_delegated.update(classes)

        return _merge_members_stack(
            SphinxDocObject.from_namespace(_cls)._get_members(
                inherited=True,
                allow_reexport=False,
                avoid_delegated=avoid_delegated,
            )
            for _cls in classes
        )
    else:
        return {}


def _with_refctx(docobj, rst):
    obj = docobj.obj
    if isinstance(obj, type) or inspect.ismodule(obj):
        ns = docobj
    else:
        try:
            ns = docobj.parent
        except ValueError:
            ns = None

    if ns:
        if isinstance(ns.obj, type):
            _class = f'   :class: {ns.fullname}\n'
            _mod = f'   :module: {ns.__module__}'
        elif inspect.ismodule(ns.obj):
            _class = ''
            _mod = f'   :module: {ns.fullname}'
        else:
            raise TypeError(f'Namespace type not handled: {ns}')
    else:
        _class = ''
        _mod = ''

    return f'''
.. withrefctx::
{_mod}
{_class}

{indent(rst, level=3)}
'''


def _fixup_inherited_doc(docobj, inherited):
    # If a member overrides another member in the stack but the most
    # derived one has no doc while the other one has one, we keep the
    # original one. This implements "docstring inheritance".
    if inherited.doc and not docobj.doc:
        # Note that we don't just take the inherited docobj, as it might
        # not have the same kind (e.g. it is a property in the base
        # class and becomes a class attribute in a derived class). If
        # we don't preserve that kind properly, autodoc will not be
        # able to work on those items.
        return docobj.replace(
            doc=inherited.doc,
            _doc_refctx=inherited,
            # Docstring inheritance in autodoc only works if the item kind is
            # the same. But in our case, it is a string class attribute that
            # inherits from a abstract property, so autodoc does not propagate
            # the docstring that we have inherited.
            _broken_autodoc_inheritance=(
                inherited._broken_autodoc_inheritance or
                docobj.autodoc_kind != inherited.autodoc_kind
            ),
        )
    else:
        return docobj


def _merge_members_stack(stack):
    """
    Merge a stack of members (most derived first) with priority to the first
    entry in the stack for each member (the most derived one).
    """
    merged = {}
    stack = list(stack)
    for ns in reversed(stack):
        merged.update({
            membername: (
                docobj
                if (inherited := merged.get(membername)) is None else
                _fixup_inherited_doc(docobj, inherited)
            )
            for membername, docobj in ns.items()
        })

    return merged


def _sort_members(app, members):
    def key(x):
        membername, docobj = x
        inherited, place, visibility = docobj.resolve_inheritance_style(app)
        return (
            # Make deprecated members appear last in listings
            docobj.is_deprecated,
            # Make inherited members come last.
            inherited and visibility == 'public',
            docobj.fullname,
        )
    return dict(sorted(members.items(), key=key))


_NOTSET = object()


class SphinxDocObject:
    """
    Represent a variable that has a "docstring" attached to it.

    Those docstrings are not really docstrings as they are merely a
    string literal floating below a variable assignment, but sphinx
    recognizes those by parsing the source.
    """
    def __init__(self, modname, qualname, doc=_NOTSET, obj=_NOTSET, doc_refctx=None, parent=None, _can_be_resolved=True):
        def _getdoc(obj):
            if obj is _NOTSET:
                return None
            else:
                doc = inspect.getdoc(obj)
                # Instances of classes that have a docstring will have have a
                # __doc__ attribute available on them. This makes
                # inspect.getdoc() return it even for things like a dict
                # instance. In those cases, we do not want to pick it up.
                if obj is not type(obj) and doc == type(obj).__doc__:
                    return None
                else:
                    return doc

        self.__qualname__ = qualname
        self.__module__ = modname if modname else None
        fullname = self.fullname

        if obj is _NOTSET:
            try:
                obj = _resolve_dotted_name(fullname)
            # It is possible for an attribute to be listed by __dir__ and be
            # implemented by a descriptor and yet it raises an AttributeError:
            # https://stackoverflow.com/questions/24914584/abstractmethods-and-attributeerror
            except Exception:
                obj = _NOTSET

        if doc is _NOTSET:
            doc = _getdoc(obj)

        self.doc = doc
        self.obj = obj

        self._doc_refctx = doc_refctx
        self._broken_autodoc_inheritance = False
        self._parent = parent

        fullname = self.fullname
        self._can_be_resolved = _can_be_resolved and not any(x in fullname for x in ('<locals>', '<lambda>'))

    def __eq__(self, other):
        def eq(obj1, obj2):
            try:
                # Trying == first is important since some names will provide a
                # new object every time they are resolved, such as a
                # classmethod that will get instantiated every time.
                return obj1 == obj2
            except Exception:
                return obj1 is obj2

        return self is other or (
            isinstance(other, self.__class__) and
            eq(self.obj, other.obj) and
            self.fullname == other.fullname and
            self.doc == other.doc and
            self._can_be_resolved == other._can_be_resolved and
            self._broken_autodoc_inheritance == other._broken_autodoc_inheritance
        )

    def __hash__(self):
        return hash((self.fullname, self.doc))

    @property
    def is_deprecated(self):
        return any(
            docobj.fullname in get_deprecated_map()
            for docobj in (
                self,
                self.inherited_from(real=True, app=None),
            )
        )

    @property
    def doc_refctx(self):
        """
        The original SphinxDocObject in which the documentation was written.

        This allows correctly resolving relative class and module references in
        docstrings:
        * The class name is the derived class, so that we get the most
          local and public reference target possible. This will allow
          relative references to method work even if we inherit from a
          private base class, since those references will be resolved on
          the public derived class instead.

        * The module is the original one. If it is private, so be it and
          references will be broken. Method docstrings can only reference
          public items or they will not work.
        """
        if self._doc_refctx:
            return self._doc_refctx.doc_refctx
        else:
            docobj = self.inherited_from(app=None, real=True, topmost=False)
            if docobj == self and docobj._doc_refctx == self._doc_refctx:
                return docobj
            else:
                return docobj.doc_refctx

    @property
    def __doc__(self):
        if self.doc is None:
            return None
        else:
            return _with_refctx(self.doc_refctx, self.doc)

    def replace(self, **kwargs):
        new = copy.copy(self)
        for attr, val in kwargs.items():
            setattr(new, attr, val)
        return new

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.__module__!r}, {self.__qualname__!r})'

    def __str__(self):
        return repr(self)

    @property
    @memoized
    def mro(self):
        mro = inspect.getmro(self.obj)
        return [
            self.from_namespace(base)
            for base in mro
        ]

    @property
    def is_dynamic(self):
        if self.exists:
            fullname = self.fullname
            try:
                _resolve_dotted_name(fullname)
            except AttributeError:
                return True
            else:
                return False
        else:
            return False

    @property
    def exists(self):
        # Lookup the member we inherited from as DelegateToAttr might
        # introduce dynamic members via SphinxDocObject.mro that only
        # exist on live instances of the class, and in the class of the
        # attribute we delegate to.
        inherited_from = self.inherited_from(app=None, real=True)

        if inherited_from._can_be_resolved:
            try:
                _resolve_dotted_name(inherited_from.fullname)
            except AttributeError:
                return False
            else:
                return True
        else:
            # We cannot resolve attribute names since they might only be
            # available at runtime on class instances, so we just assume they
            # do exist.
            return True

    @memoized
    def inherited_from(self, app, real=False, topmost=True):
        assert app is not None or real

        def get(docobj):
            parent = docobj.parent
            if parent is not None and isinstance(parent.obj, type):
                membername = docobj.membername

                def get_inherited(base, membername):
                    cls = base.obj
                    base_members = base.get_members(inherited=False)
                    return base_members.get(membername)

                # Make sure the member we inherit from has its docstring fixed
                # up the same way as when merging an inheritance stack.
                def fixup_docstrings(xs):
                    return reversed(list(itertools.accumulate(
                        reversed(list(xs)),
                        lambda x_super, x: (
                            x
                            if x_super is None else
                            (
                                x_super
                                if x is None else
                                _fixup_inherited_doc(x, x_super)
                            )
                        )
                    )))

                mro = parent.mro
                inherited_members = (
                    get_inherited(base, membername)
                    for base in mro
                )
                inherited_members1, inherited_members2 = itertools.tee(inherited_members)
                iterator = (
                    (realinherited, publicbase)
                    for (select, realinherited, publicbase) in zip(
                        (x is not None for x in inherited_members1),
                        fixup_docstrings(inherited_members2),
                        ffill(
                            mro,
                            select=lambda _docobj: (not _docobj.autodoc_is_skipped(app)) if app else True,
                        ),
                    )
                    if select
                )

                try:
                    (realinherited, publicbase), *_ = iterator
                # It is not inherited from any base class
                except ValueError:
                    delegated_members = _get_delegated_members(parent.obj)
                    return delegated_members.get(membername, docobj)
                else:
                    if real:
                        # Return the attribute we actually inherit from
                        return realinherited
                    else:
                        if publicbase is None:
                            return docobj
                        else:
                            # Return the attribute on the top-most public class
                            # we appear to be inheriting from. This will hide
                            # the real location we inherit from if that
                            # location is in a private base class.
                            members = publicbase.get_members(inherited=True)
                            return members.get(membername, docobj)
            else:
                return docobj

        return fixedpoint(get, init=self, limit=None if topmost else 1, raise_=False)

    def resolve_inheritance_style(self, app):
        def get_toplevel_package(modname):
            package, *_ = modname.split('.')
            return package

        if self.is_class_member:
            # Use the "public parent" rather than the "real parent", so we
            # correctly infer "public" visibility rather than private.
            inherited_from = self.inherited_from(app, real=False)
            inherited = inherited_from != self

            if inherited:
                # If both the parent of the attribute (where it is inherited) and the real
                # parent of the attribute (where it is defined) are located in the same
                # package, we will just forward the docstring in the inherited class.
                #
                # This ensures we will have the docstring available for the user even if we
                # inherit the method from a private class.
                if get_toplevel_package(self.__module__) == get_toplevel_package(inherited_from.__module__):
                    # If we inherit from a skipped parent, we silently forward
                    # the docstring, so that it stays an implementation detail
                    # hidden to the user.
                    if any(
                        parent.autodoc_is_skipped(app)
                        for parent in inherited_from.parents
                    ):
                        return (True, 'local', 'private')
                    else:
                        return (True, 'local', 'public')
                # Otherwise if we are inheriting from a class defined in another package,
                # we replace the docstring with a stub reference. This ensures we can build
                # the documentation cleanly as we can fix any inherited docstring defined
                # in our package.
                else:
                    return (True, 'foreign', 'public')

        return (False, 'local', 'public')

    @property
    def is_class_member(self):
        try:
            parent = self.parent
        except ValueError:
            # If we cannot resolve the parent, this means we are inside a class
            # since modules cannot be defined as local variables, but classes
            # can.
            return True
        else:
            if parent:
                return isinstance(parent.obj, type)
            else:
                return False

    def get_short_doc(self, *args, **kwargs):
        proxy = types.SimpleNamespace(__doc__=self.doc)
        doc = get_short_doc(proxy, *args, **kwargs)
        return _with_refctx(self.doc_refctx, doc)

    @memoized
    def get_name(self, *, style=None, abbrev=False):
        # For some reason, autodoc does not display DynamicClassAttribute
        if style == 'rst' and isinstance(self.obj, types.DynamicClassAttribute):
            name = self.membername if abbrev else self.fullname
            return f':code:`{name}`'
        else:
            return get_obj_name(self, abbrev=abbrev, style=style, name=self.fullname)

    @memoized
    def autodoc_is_skipped(self, app):
        kind = self.autodoc_kind
        if kind in ('property', 'data'):
            kind = 'attribute'

        return app.emit_firstresult(
            'autodoc-skip-member', kind, self.fullname, self.obj, False, {}
        )

    @property
    @memoized
    def autodoc_kind(self):
        def get(obj, parent, is_classmember):
            if is_classmember:
                if (
                    inspect.isfunction(obj) or
                    inspect.ismethod(obj) or
                    isinstance(obj, (classmethod, staticmethod))
                ):
                    return 'method'
                else:
                    standalone = get(obj, parent=parent, is_classmember=False)
                    if standalone == 'data':
                        # autodoc does not include the docstring of an Enum
                        # member if ".. autoattribute::" directive is used, but
                        # it does if ".. autodata::" is used.
                        if parent and isinstance(parent.obj, type) and issubclass(parent.obj, enum.Enum):
                            return 'data'
                        else:
                            return 'attribute'
                    elif standalone == 'function':
                        return 'method'
                    else:
                        return standalone
            elif isinstance(obj, type):
                if issubclass(obj, BaseException):
                    return 'exception'
                else:
                    return 'class'
            elif inspect.ismodule(obj):
                return 'module'
            elif isinstance(obj, property):
                return 'property'
            elif callable(obj):
                return 'function'
            else:
                # Deal with decorators
                unwrapped = inspect.unwrap(obj)
                if unwrapped is obj:
                    return 'data'
                else:
                    return get(obj=obj, parent=parent, is_classmember=is_classmember)

        try:
            parent = self.parent
        except ValueError:
            parent = None

        return get(obj=self.obj, parent=parent, is_classmember=self.is_class_member)

    @property
    @memoized
    def sphinx_role(self):
        return get_sphinx_role(self, name=self.fullname)

    @property
    def fullname(self):
        modname = self.__module__
        qualname = self.__qualname__
        if modname:
            return f'{modname}.{qualname}'
        else:
            return qualname

    @property
    def membername(self):
        return self.__qualname__.rsplit('.', 1)[-1]

    @property
    def parent(self):
        return self._parent or self._resolve_parent()

    @memoized
    def _resolve_parent(self):
        namespace = _get_parent_namespace(self.fullname)
        return self.from_namespace(namespace) if namespace else None

    @property
    @memoized
    def parents(self):
        def gen():
            parent = self.parent
            while parent:
                yield parent
                parent = parent.parent
        return list(gen())

    @property
    def __wrapped__(self):
        return self.obj

    @classmethod
    def from_namespace(cls, namespace):
        if isinstance(namespace, type):
            modname = namespace.__module__
            qualname = namespace.__qualname__
        elif inspect.ismodule(namespace):
            *modname, qualname = namespace.__name__.split('.')
            modname = '.'.join(modname)
        else:
            raise ValueError(f'Namespace not handled: {namespace}')

        return cls._from_namespace(
            modname=modname,
            qualname=qualname,
            obj=namespace,
        )

    @classmethod
    @functools.lru_cache(maxsize=2048)
    def _from_namespace(cls, modname, qualname, obj):
        doc = inspect.getdoc(obj)
        return cls(
            modname=modname,
            qualname=qualname,
            doc=doc,
            obj=obj,
        )

    @classmethod
    @functools.lru_cache(maxsize=2048)
    def _from_name(cls, name):
        def get_member(parent, membername):
            if parent is None:
                try:
                    ns = _resolve_dotted_name(membername)
                except (AttributeError, ImportError) as e:
                    raise ValueError(str(e))
                else:
                    return cls.from_namespace(ns)
            else:
                try:
                    members = parent.get_members(allow_reexport=True)
                except TypeError as e:
                    raise ValueError(str(e))
                else:
                    try:
                        return members[membername]
                    except KeyError as e:
                        raise ValueError(str(e))

        assert name
        return fold(
            get_member,
            name.split('.'),
            init=None,
        )

    @classmethod
    def from_name(cls, name, obj=_NOTSET):
        """
        Resolve ``name`` to a :class:`SphinxDocObject`.

        :param name: Fully qualified name of the entity to document.
        :type name: str

        :param obj: If passed, it will be used as the ``obj`` parameter for
            :class:`SphinxDocObject`. If omitted, it will be resolved from the
            name.
        :type obj: object
        """
        docobj = cls._from_name(name)
        if obj is not _NOTSET:
            docobj = docobj.replace(obj=obj)

        return docobj

    @memoized
    def get_members(self, inherited=True, allow_reexport=False):
        return self._get_members(
            inherited=inherited,
            allow_reexport=allow_reexport,
            avoid_delegated=None,
        )

    def _get_members(self, inherited, allow_reexport, avoid_delegated):
        """
        ``inspect.getmembers`` plus the attributes that are documented with a
        "docstring" that is actually just a string literal floating after a
        variable assignment.
        """
        cls = self.__class__
        namespace = self.obj
        members_stack = []

        if isinstance(namespace, type):
            def make_docobj(membername, obj, _can_be_resolved=True, doc=_NOTSET):
                return cls(
                    modname=namespace.__module__,
                    qualname=f'{namespace.__qualname__}.{membername}',
                    obj=obj,
                    doc=doc,
                    _can_be_resolved=_can_be_resolved,
                    parent=self,
                )

            def cls_members(namespace, _cls):
                """
                Provide the members of a class, but not anything inherited
                """
                def get_instance_dir(cls):
                    try:
                        instance_dir = cls.__instance_dir__
                    except AttributeError:
                        return {}
                    else:
                        return dict(instance_dir())

                def get_sphinx_attrs(namespace, base):
                    def split_name(name):
                        try:
                            basename, membername = name.rsplit('.', 1)
                        except ValueError:
                            basename = None
                            membername = name

                        return (basename, membername)

                    mod = inspect.getmodule(base)
                    attrs = sphinx_module_attrs_doc(mod)
                    return {
                        membername: cls(
                            modname=namespace.__module__,
                            qualname=f'{namespace.__qualname__}.{membername}',
                            doc=doc,
                            _can_be_resolved=False,
                            parent=self,
                        )
                        for (basename, membername), doc in (
                            (split_name(name), doc)
                            for name, doc in attrs.items()
                        )
                        if basename == base.__qualname__
                    }

                def getmembers(_cls):
                    """
                    Provide all members of a class, including inherited ones.
                    """
                    def get_dynamic_attr_doc(_cls):
                        docobj = cls.from_namespace(_cls)
                        ref = docobj.get_name(style='rst')
                        return f'See {ref}'

                    return {
                        **{
                            membername: make_docobj(
                                membername=membername,
                                obj=None,
                                # This is going to be a dynamic attribute, we
                                # cannot expect lookup to succeed on the class,
                                # only on instances.
                                _can_be_resolved=False,
                                doc=get_dynamic_attr_doc(obj),
                            )
                            for membername, obj in get_instance_dir(_cls).items()
                        },
                        **{
                            membername: make_docobj(membername, obj)
                            for membername, obj in inspect.getmembers(_cls)
                        },
                        **get_sphinx_attrs(namespace, _cls),
                    }

                members = getmembers(_cls)
                base_members = _merge_members_stack(
                    getmembers(base)
                    for base in _cls.__bases__
                )
                # This is more correct than _cls.__dict__ as it will include
                # dynamic attributes that are reported by dir() but not in
                # __dict__.
                members = {
                    membername: docobj
                    for membername, docobj in members.items()
                    if (
                        # This was defined in the class itself, so it's always
                        # taken
                        membername in silent_getattr(_cls, '__dict__', {})
                        or
                        # The membername might not be in __dict__ if it is a
                        # dynamic attribute, so we still take it as long as the
                        # base classes do not provide it.
                        membername not in base_members
                        or
                        # The membername exists in the base but has been
                        # overridden with a new docstring.
                        (
                            # members we gather with inspect.getmembers() will
                            # have doc set to None. If the doc is set in a base
                            # class and then it is None in the derived class,
                            # it means we are inherited.
                            docobj.doc is not None and
                            base_members[membername].doc != docobj.doc
                        )
                    )
                }
                return members

            def _filter(membername, docobj):
                return True

            mro = self.mro
            assert mro[0] == self
            bases = mro if inherited else [mro[0]]
            bases = [base.obj for base in bases]

            members_stack.extend(
                cls_members(namespace, _cls)
                for _cls in bases
            )

            if inherited:
                # Lowest priority to delegated members, since they are
                # implemented with __getattr__ and can be overridden by any
                # actual member.
                members_stack.append({
                    membername: make_docobj(
                        membername=membername,
                        obj=docobj.obj,
                        doc=docobj.doc,
                        _can_be_resolved=False,
                    )
                    for membername, docobj in _get_delegated_members(namespace, avoid_delegated).items()
                })

        elif inspect.ismodule(namespace):
            attrs = sphinx_module_attrs_doc(namespace)
            members_stack.append({
                membername: cls(
                    modname=namespace.__name__,
                    qualname=membername,
                    doc=doc,
                    _can_be_resolved=False,
                    parent=self,
                )
                for membername, doc in attrs.items()
                # Names with dot in them mean the variable is actually a class
                # attribute rather than a top-level variable, so we don't want
                # them at module-level listing.
                if '.' not in membername
            })

            members_stack.append({
                membername: cls(
                    modname=namespace.__name__,
                    qualname=membername,
                    obj=obj,
                    parent=self,
                )
                for membername, obj in inspect.getmembers(namespace)
                if (
                    # Member actually defined in the module rather than just
                    # imported.
                    (
                        allow_reexport or (inspect.getmodule(obj) is namespace)
                    ) or
                    # Submodule.
                    (
                        inspect.ismodule(obj) and obj.__name__.startswith(f'{namespace.__name__}.')
                    )
                )
            })

            def _filter(membername, docobj):
                # Modules might have undocumented globals that are actually
                # defined somewhere else and simply imported, so remove them.
                return docobj.parent == self
        else:
            raise TypeError(f'Namespace not handled: {namespace}')

        members = _merge_members_stack(members_stack)

        def check(membername, docobj):
            parent = docobj.parent
            if parent == self:
                return True
            else:
                raise ValueError(f'Member {docobj} is not a child of expected parent: expected={self} actual={docobj.parent}')

        return {
            membername: docobj
            for membername, docobj in sorted(members.items())
            if _filter(membername, docobj) and check(membername, docobj)
        }


class ModuleListingDirective(RecursiveDirective):
    """
    reStructuredText directive similar to autosummary but with correct handling
    of inheritance::

        .. module-listing:: mymodule

    Options:
    """
    STUBS_FOLDER = 'generated'

    required_arguments = 1

    @classmethod
    def _run(cls, app, curr_loc, stubs_loc, modname, make_stub):

        def listing_entry(app, docobj):
            inherited, place, visibility = docobj.resolve_inheritance_style(app)

            tags = []
            if inherited and visibility == 'public':
                tags.append('inherited')
                ref_target = docobj.inherited_from(app)
            else:
                ref_target = docobj

            if ref_target.is_deprecated:
                tags.append('deprecated')

            ref = ref_target.get_name(style='rst', abbrev=True)

            if place == 'local':
                doc = docobj.get_short_doc(style='rst')
            else:
                long_ref = ref_target.get_name(style='rst', abbrev=False)
                # The doc could be invalid reStructuredText, so we just do not
                # include it and use a link instead.
                doc = f'See {long_ref}'

            tags = ', '.join(sorted(tags))
            tags = f' :sup:`{tags}`' if tags else ''
            return f'''
* - {ref}{tags}
  -
{indent(doc or '', level=3)}
'''.strip()

        def toc_entry(docobj, name, curr_loc, stubs_loc, make_stub):
            name = docobj.get_name()
            # Remove the common prefix coming from the parent module name
            # to avoid O(N^2) ToC entry size
            membername = docobj.membername

            path = stubs_loc / f'{name}.rst'

            if make_stub:
                stub_content = process_member(
                    app=app,
                    docobj=docobj,
                    curr_loc=path,
                    stubs_loc=stubs_loc,
                    make_stub=make_stub,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                stub_content = stub_content.encode('utf-8')
                with open(path, 'w+b') as f:
                    existing = f.read()
                    if stub_content != existing:
                        f.write(stub_content)

            toc_ref = str(
                (stubs_loc / name).relative_to(
                    curr_loc.parent
                )
            )

            entry = f'{membername} <{toc_ref}>'
            return entry

        def process_member(app, docobj, curr_loc, stubs_loc, make_stub):
            if inspect.ismodule(docobj.obj):
                return process_mod(
                    app=app,
                    docobj=docobj,
                    curr_loc=curr_loc,
                    stubs_loc=stubs_loc,
                    make_stub=make_stub,
                )
            else:
                return process_leaf(app=app, docobj=docobj)

        def group_members(app, members):
            grouped = order_as(
                groupby(
                    _sort_members(app, members).items(),
                    key=lambda x: x[1].autodoc_kind
                ),
                order_as=[
                    'module',
                    'data',
                    'class',
                    'attribute',
                    'property',
                    'method',
                    'function',
                    'exception',
                ],
                key=itemgetter(0),
            )
            return {
                key: dict(_members)
                for key, _members in grouped
            }

        def make_grouped(app, members, make_group):
            def make_pretty(name):
                return {
                    'method': 'methods',
                    'attribute': 'attributes',
                    'exception': 'exceptions',
                    'class': 'classes',
                    'module': 'modules',
                    'function': 'functions',
                    'property': 'properties',
                    'data': 'globals',
                }[name].title()

            grouped = group_members(app, members)
            return '\n'.join(
                group
                for _group, _members in grouped.items()
                if (group := make_group(
                    title=make_pretty(_group),
                    members=_members,
                )) is not None
            )

        def make_listing(app, members):
            def make_group(title, members):
                listing = '\n'.join(
                    listing_entry(app=app, docobj=docobj)
                    for docobj in members.values()
                    if not docobj.is_deprecated
                )
                if listing:
                    return f'''
.. rubric:: {title}

.. list-table::
   :align: left

{indent(listing, level=3)}
'''
                else:
                    return None

            return make_grouped(
                app=app,
                members=members,
                make_group=make_group,
            )

        def make_body_listing(app, docobj, members, extra=''):
            def make_item_doc(app, docobj):
                inherited, place, visibility = docobj.resolve_inheritance_style(app)
                if inherited and visibility == 'public' and docobj.is_deprecated:
                    return None
                else:
                    kind = docobj.autodoc_kind

                    # DelegateToAttr introduces dynamic members that can only be
                    # lookedup on live instances. This makes autodoc fail to
                    # generate an API doc for that member, so instead we just
                    # document the member without relying on autodoc.
                    if docobj.is_dynamic or docobj._broken_autodoc_inheritance:
                        ref = docobj.inherited_from(app).get_name(style='rst')
                        doc, _ = _inherited_docstring(docobj=docobj, app=app)
                        doc = doc or ''
                        obj = docobj.obj

                        fields = {
                            ':classmethod:': isinstance(obj, classmethod),
                            ':staticmethod:': isinstance(obj, staticmethod),
                            ':async:': inspect.iscoroutinefunction(obj),
                            ':abstractmethod:': getattr(obj, '__isabstractmethod__', False),
                            f':canonical: {docobj.__module__}.{docobj.__qualname__}': not inspect.ismodule(obj),
                            f':value: {docobj.obj!r}': kind in ('attribute', 'data') and docobj.obj is not _NOTSET
                        }

                        info_field_list = '\n'.join(sorted(
                            field
                            for field, cond in fields.items()
                            if cond
                        ))

                        # Use :canonical: to actually provide the reference target.
                        # This allows using just the __qualname__ for the title, so
                        # it is consistent with autodoc output.
                        return f'''
.. {kind}:: {docobj.__qualname__}
{indent(info_field_list, level=3)}

{indent(doc, level=3)}
'''
                    else:
                        return f'''
.. auto{kind}:: {docobj.__module__}::{docobj.__qualname__}
'''
            if isinstance(docobj.obj, type):
                def make_group(title, members):
                    body = '\n\n'.join(
                        doc
                        for docobj in members.values()
                        if (doc := make_item_doc(app, docobj))
                    )
                    return f'''
{title}
{'-' * len(title)}

{body}
'''
                body = make_grouped(
                    app=app,
                    make_group=make_group,
                    members={
                        name: docobj
                        for name, docobj in members.items()
                        if docobj.exists
                    }
                )
                return f'''
.. autoclass:: {docobj.__module__}::{docobj.__qualname__}
   :no-members:
   :no-inherited-members:
   :no-undoc-members:
   :no-private-members:
   :no-special-members:

{indent(extra, level=3)}

{body}
'''
            elif inspect.ismodule(docobj.obj):
                return extra
            else:
                doc = make_item_doc(app, docobj)
                return f'''
{doc}
{extra}
''' if doc else ''

        def document_title(docobj):
            parent = docobj.parent
            if parent:
                # Set a shorter doc title if possible, so the breadcrumbs UI
                # element does not grow in O(N^2) with the depth of the
                # nesting.
                return f'.. title:: {relname(parent.obj, docobj.obj)}\n\n'
            else:
                return ''

        def sort_mod_members(app, members):
            grouped = group_members(app, members)
            merged = list(grouped.values())
            return dict(ChainMap(*reversed(merged)))

        def process_mod(app, docobj, curr_loc, stubs_loc, make_stub, with_title=True):
            members = docobj.get_members()
            members = {
                name: docobj
                for name, docobj in members.items()
                if not docobj.autodoc_is_skipped(app)
            }
            members = sort_mod_members(app, members)

            listing = make_listing(app=app, members=members)

            toc_entries = '\n'.join(
                entry
                for name, docobj in members.items()
                if (entry:= toc_entry(
                    docobj=docobj,
                    name=name,
                    curr_loc=curr_loc,
                    stubs_loc=stubs_loc,
                    make_stub=make_stub,
                )) is not None
            )

            name = docobj.fullname
            toctree = f'''
.. toctree::
   :hidden:

{indent(toc_entries, level=3)}
'''

            automodule = f'''
.. automodule:: {name}
    :no-index:
    :no-members:
    :no-inherited-members:
    :no-undoc-members:
    :no-private-members:
    :no-special-members:
'''
            # autodoc seems broken and does not make the module specified by
            # ".. automodule:: <the module>" the current module in the
            # reference context, so we do it manually instead.
            automodule = _with_refctx(docobj, automodule)
            automodule = f'''
.. module:: {name}

{automodule}
'''

            title = f'{name}\n{"=" * len(name)}\n\n' if with_title else ''
            doc_title = document_title(docobj)
            content = f'{doc_title}{title}{toctree}{automodule}{listing}'
            return content


        def process_leaf(app, docobj):
            obj = docobj.obj
            if isinstance(obj, type) or inspect.ismodule(obj):
                members = docobj.get_members()
            else:
                members = {}

            members = {
                name: docobj
                for name, docobj in members.items()
                if not docobj.autodoc_is_skipped(app)
            }

            listing = make_listing(app=app, members=members)

            body = make_body_listing(
                app=app,
                docobj=docobj,
                members=members,
                extra=listing,
            )

            fullname = docobj.fullname
            content = f'''
.. title:: {docobj.membername}

{fullname}
{'=' * len(fullname)}

{body}
'''
            return content

        mod = importlib.import_module(modname)
        import_all_submodules(mod, best_effort=True)

        docobj = SphinxDocObject.from_namespace(mod)
        out = process_mod(
            app=app,
            docobj=docobj,
            curr_loc=curr_loc,
            stubs_loc=stubs_loc,
            make_stub=make_stub,
            with_title=False,
        )
        return out

    def run(self):
        curr_loc = Path(self.env.doc2path(self.env.docname)).resolve()
        stubs_loc = Path(
            curr_loc.parent,
            self.STUBS_FOLDER,
        ).resolve()

        modname, = self.arguments

        out = self._run(
            app=self.env.app,
            curr_loc=curr_loc,
            stubs_loc=stubs_loc,
            modname=modname,
            make_stub=False,
        )

        return self.parse_nested(out)

    @classmethod
    def make_stubs(cls, app):
        env = app.builder.env
        sources = [
            path
            for path in map(
                lambda x: Path(env.doc2path(x)).resolve(),
                env.found_docs
            )
            if path.is_file() and path.suffix == '.rst'
        ]

        def process_directive(path, modname):
            stubs_loc = path.parent / cls.STUBS_FOLDER
            out = cls._run(
                app=app,
                curr_loc=path,
                stubs_loc=stubs_loc,
                modname=modname,
                make_stub=True,
            )

        pattern = re.compile(r'\.\. module-listing\s*::\s*([a-zA-Z0-9_.]+)')
        for path in sources:
            modnames = re.findall(pattern, path.read_text())
            for modname in modnames:
                process_directive(path, modname)


# Sphinx extension setup
def setup(app):
    directives.register_directive('withrefctx', WithRefCtxDirective)
    directives.register_directive('run-command', RunCommandDirective)


    directives.register_directive('exec', ExecDirective)
    app.add_event('lisa-exec-state')


    directives.register_directive('module-listing', ModuleListingDirective)
    # We cannot add new sources in SphinxDirective.run(), so it needs to be
    # done earlier in the build process.
    app.connect('builder-inited', ModuleListingDirective.make_stubs)

    return {
        'version': '0.1',
        'env_version': 1,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }


def indent(content, level=1, idt=' '):
    idt = level * idt
    return idt + content.replace('\n', f'\n{idt}')


def is_test(method):
    """
    Check if a method is a test method.
    """
    if not callable(method):
        return False

    with contextlib.suppress(AttributeError):
        if method.__name__.startswith('test_'):
            return True

    # Tests are methods with an annotated return type, with at least
    # one base class with a name containing 'result'
    try:
        ret_type = inspect.signature(method).return_annotation
        base_cls_list = inspect.getmro(ret_type)
    except (ValueError, AttributeError, KeyError):
        return False
    else:
        return any(
            'result' in cls.__qualname__.casefold()
            for cls in base_cls_list
        )


# Unfortunately, this will not currently run for items that do not have a
# docstring at all because of:
# https://github.com/sphinx-doc/sphinx/issues/12678
def autodoc_process_inherited_members(app, what, name, obj, options, lines):
    """
    Replace docstrings of inherited members by a stub that points at the place
    where the member is actually defined.

    This prevents having issues when inheriting from members with docstrings
    that are not valid reStructuredText, which would make the build fail.
    """
    new, _ = _autodoc_process_inherited_members(app, what, name, obj, lines)
    lines[:] = new


def autodoc_process_inherited_signature(app, what, name, obj, options, signature, return_annotation):
    """
    Removes the signature when :func:`autodoc_process_inherited_members` would
    remove the docstring.
    """
    _, foreign = _autodoc_process_inherited_members(app, what, name, obj, [''])
    return (None, None) if foreign else (signature, return_annotation)


def _autodoc_process_inherited_members(app, what, name, obj, lines):
    try:
        docobj = SphinxDocObject.from_name(name, obj=obj)
    except ValueError:
        return (lines, False)
    else:
        doc, foreign = _inherited_docstring(
            app=app,
            docobj=docobj,
            doc=_with_refctx(
                docobj.doc_refctx,
                '\n'.join(lines),
            ),
        )
        lines = (doc or '').splitlines()
        return (lines, foreign)


def _inherited_docstring(app, docobj, doc=None):
    doc = doc or docobj.__doc__ or ''

    if docobj.is_class_member:
        inherited, place, visibility = docobj.resolve_inheritance_style(app)
        if inherited:
            ref = docobj.inherited_from(app).get_name(style='rst')
            kind = docobj.autodoc_kind

            if place == 'local':
                if visibility == 'private':
                    pass
                elif visibility == 'public':
                    # Set the current module to be the one in which the member was
                    # really defined. That will make any relative reference e.g. to
                    # another class in the same module work rather than requiring
                    # absolute references.
                    shortdoc = docobj.get_short_doc(style='rst')
                    snippet = f'''
*Inherited {kind}, see* {ref}

{shortdoc}
'''
                    doc = snippet
                else:
                    raise ValueError(f'Non handled inheritance visibility: {visibility}')
            # Otherwise if we are inheriting from a class defined in another package,
            # we replace the docstring with a stub reference. This ensures we can build
            # the documentation cleanly as we can fix any inherited docstring defined
            # in our package.
            elif place == 'foreign':
                doc = f'''
*Inherited {kind}, see* {ref}

'''
            else:
                raise ValueError(f'Non handled inheritance place: {place}')

        foreign = place == 'foreign'
        return (doc, foreign)

    elif isinstance(docobj.obj, type):
        # We do not want to inherit any class docstring, as it is usually
        # misleading.
        doc = docobj.obj.__doc__
        doc = inspect.cleandoc(doc) if doc else doc
        return (doc, False)
    else:
        return (doc, False)


def autodoc_process_test_method(app, what, name, obj, options, lines):
    # Append the list of available test methods for all classes that appear to
    # have some.
    if what == 'class':
        test_list = [
            member
            for member_name, member in inspect.getmembers(obj, is_test)
        ]
        if test_list:
            test_list_doc = '\n:Test methods:\n\n{}\n\n'.format('\n'.join(
                '    * :meth:`~{}`'.format(
                    method.__module__ + '.' + method.__qualname__
                )
                for method in test_list
            ))

            lines.extend(test_list_doc.splitlines())


def autodoc_process_analysis_events(app, what, name, obj, options, lines):
    """
    Append the list of required trace events
    """

    # We look for events in the getter method of properties
    if what == 'property':
        obj = obj.fget

    try:
        used_events = obj.used_events
    except AttributeError:
        return
    else:
        if not isinstance(used_events, TraceEventCheckerBase):
            return

        events_doc = f"\n:Required trace events:\n\n{used_events.doc_str()}\n\n"
        lines.extend(events_doc.splitlines())


def intersphinx_warn_missing_reference_handler(app, domain, node, non_ignored_refs):
    if domain and domain.name == 'py':
        reftarget = node['reftarget']
        class_ctx = node.get('py:class')
        mod_ctx = node.get('py:module')

        possible_names = [
            name
            for name in (
                f'{class_ctx}.{reftarget}' if class_ctx else None,
                f'{mod_ctx}.{reftarget}' if mod_ctx else None,
                reftarget,
            )
            if name
        ]
        for name in possible_names:
            try:
                docobj = SphinxDocObject.from_name(name)
            except ValueError:
                pass
            else:
                if docobj.autodoc_is_skipped(app):
                    return True
                elif any(
                    regex.match(docobj.fullname)
                    for regex in non_ignored_refs
                ):
                    return None
                else:
                    return True

        package = reftarget.split('.')[0]
        try:
            importlib.import_module(package)
        except ImportError:
            # If the top level package cannot even be imported, this probably
            # means we are referring to an optional dependency that is not
            # installed so we assume the name is valid.
            return True
        else:
            return None


def autodoc_process_bases_handler(app, name, obj, options, bases):
    """
    Apply the skipping logic to base classes, so we hide private base classes.
    """

    def rewrite_bases(bases):
        return itertools.chain.from_iterable(
            rewrite(base)
            for base in bases
        )

    def rewrite(cls):
        try:
            docobj = SphinxDocObject.from_namespace(cls)
        # Some bases might not be classes. They might be functions
        # monkey-patched with an __mro_entries__ attribute, like
        # typing.NamedTuple (in Python 3.12 at least)
        except ValueError:
            return [cls]
        else:
            skipped = docobj.autodoc_is_skipped(app)
            if skipped:
                return rewrite_bases(cls.__bases__)
            else:
                return [cls]

    new_bases = list(rewrite_bases(list(bases)))
    new_bases = [
        base
        for base in new_bases
        if base is not object
    ]
    new_bases = new_bases or [object]
    bases[:] = new_bases


def autodoc_skip_member_handler(app, what, name, obj, skip, options, default_exclude_members=None):
    """
    Enforce the "exclude-members" option, even in cases where it seems to be
    ignored by Sphinx.
    """
    UNINTERESTING_BASES = (
        object,
        type,
        abc.ABC,
        abc.ABCMeta,
        typing.NamedTuple,
    )

    def make_sub(cls):
        class Sub(cls):
            pass
        return Sub

    # Plain subclasses so that we filter out any default dunder attribute they
    # might get, such as non-default __base__.
    UNINTERESTING_BASES = list(UNINTERESTING_BASES) + list(map(make_sub, UNINTERESTING_BASES))

    def filter_name(fullname, excluded, doc):
        def _filter(membername):
            if membername in excluded:
                return True
            # Dunder names are a bit more tricky to handle since we cannot decide
            # whether it is skipped or not just based on the name. Unfortunately,
            # we also cannot just interpret the absence of doc as to be skipped,
            # since some default implementations have docstrings (e.g.
            # object.__init_subclass__). Those docstrings will be "inherited" by
            # any custom implementation that does not specify any docstring as per
            # inspect.getdoc() behavior.
            # As a result, we skip implementation that either:
            # * have no doc
            # * or have the same doc as one of the uninteresting implementations.
            elif membername.startswith('__') and membername.endswith('__'):
                def same_doc(cls):
                    try:
                        member = silent_getattr(cls, membername)
                    except AttributeError:
                        return False
                    else:
                        return inspect.getdoc(member) == doc

                return any(map(same_doc, UNINTERESTING_BASES))

            elif membername.startswith('_'):
                return True
            else:
                return False

        return any(map(_filter, fullname.split('.')))

    excluded = options.get('exclude-members', set())
    if excluded:
        # Either it's a one-item set with the string passed in conf.py
        try:
            excluded, = excluded
        # Or it's an already-processed set
        except ValueError:
            pass
        else:
            excluded = exclude_members_option(excluded)

    default_excluded = exclude_members_option(default_exclude_members or '')
    excluded = excluded | default_excluded

    # Workaround issue:
    # https://github.com/sphinx-doc/sphinx/issues/12674
    #
    # Note that if it was an inherited member, it will resolve the name to be
    # its real name where it was defined, rather than as a member of the
    # subclass.
    if '.' not in name:
        try:
            _name = get_obj_name(obj)
        except ValueError:
            pass
        else:
            # We only want to get the fully qualified version of "name". We
            # don't want to rename cases where a class attribute is assigned
            # something random that happens to have a name.
            if _name.split('.')[-1] == name:
                name = _name

    try:
        docobj = SphinxDocObject.from_name(name, obj=obj)
    except ValueError:
        membername = name.split('.')[-1]
        # Best effort attempt, in case the workaround for
        # https://github.com/sphinx-doc/sphinx/issues/12674
        # did not work.
        return filter_name(name, excluded, doc=None)
    else:
        obj = docobj.obj
        fullname = docobj.fullname
        membername = docobj.membername
        doc = docobj.doc or ''
        unwrapped = inspect.unwrap(obj)
        # Get rid of the default implementation of dunder names, since it adds no
        # value in the documentation
        if any(
            silent_getattr(cls, membername, object()) in (obj, unwrapped)
            # providers of "uninteresting" methods that are useless in our
            # documentation
            for cls in UNINTERESTING_BASES
        ):
            return True
        # Some classes like ABCMeta are more sneaky so also ban things that are
        # just builtin functions
        elif any(
            type_ in map(type, (obj, unwrapped))
            for type_ in (
                # Work with multiple Python versions
                silent_getattr(types, type_name)
                for type_name in (
                    'BuiltinFunctionType',
                    'BuiltinMethodType',
                    'WrapperDescriptorType',
                    'MethodWrapperType',
                    'MethodDescriptorType',
                    'ClassMethodDescriptorType',
                    'GetSetDescriptorType',
                    'MemberDescriptorType',
                )
                if hasattr(types, type_name)
            )
        ):
            return True
        elif re.search(r'^\s*:\s*meta\s*public\s*:', doc, re.MULTILINE):
            return False
        elif re.search(r'^\s*:\s*meta\s*private\s*:', doc, re.MULTILINE):
            return True
        else:
            return filter_name(fullname, excluded, doc=doc)


class DocPlotConf(SimpleMultiSrcConf):
    """
    Analysis plot method arguments configuration for the documentation.

    {generated_help}
    {yaml_example}
    """
    STRUCTURE = TopLevelKeyDesc('doc-plot-conf', 'Plot methods configuration', (
        # Avoid deepcopy of the value, since it contains a Trace object that we
        # don't want to duplicate for speed reasons
        KeyDesc('plots', 'Mapping of function qualnames to their settings', [Mapping], deepcopy_val=False),
    ))

def autodoc_pre_make_plots(conf, plot_methods):
    def spec_of_meth(conf, meth_name):
        plot_conf = conf['plots']
        default_spec = plot_conf.get('default', {})
        spec = plot_conf.get(meth_name, {})
        spec = {**default_spec, **spec}
        return spec

    def preload_events(conf, methods):
        """
        Preload the events in the traces that will be used so that they can be
        preloaded in parallel rather than invoking the parser several times.
        """
        methods = {
            meth.__qualname__: meth
            for meth in methods
        }

        def events_of(name):
            meth = methods[name]
            spec = spec_of_meth(conf, name)
            trace = spec['trace']
            try:
                events = meth.used_events
            except AttributeError:
                events = set()
            else:
                events = events.get_all_events()

            return (trace, events)

        traces = collections.defaultdict(set)

        for name in conf['plots'].keys():
            try:
                trace, events = events_of(name)
            except KeyError:
                pass
            else:
                traces[trace].update(events)

        for trace, events in traces.items():
            trace.get_view(events=events)

    def _make_plot(meth):
        spec = spec_of_meth(conf, meth.__qualname__)
        kwargs = spec.get('kwargs', {})
        trace = spec['trace']

        if spec.get('hide'):
            return None
        else:
            print(f'Generating plot for {meth.__qualname__} ...')

            # Suppress deprecation warnings so we can still have them in the doc
            with warnings.catch_warnings(), measure_time() as m:
                warnings.simplefilter("ignore", category=DeprecationWarning)

                rst_figure = TraceAnalysisBase.call_on_trace(meth, trace, {
                    'backend': 'bokeh',
                    'output': 'sphinx-rst',
                    **kwargs
                })

            print(f'Plot for {meth.__qualname__} generated in {m.delta}s')
            return rst_figure

    preload_events(conf, plot_methods)
    plots = {
        meth: _make_plot(meth)
        for meth in plot_methods
    }

    return plots


def autodoc_process_analysis_plots(app, what, name, obj, options, lines, plots):
    if what != 'method':
        return

    try:
        rst_figure = plots[name]
    except KeyError:
        return
    else:
        if rst_figure:
            rst_figure = f'{rst_figure}\n'
            lines[:0] = rst_figure.splitlines()


def ana_invocation(obj, name=None):
    if callable(obj):
        if name:
            try:
                cls = _get_parent_namespace(name)
            except ModuleNotFoundError:
                raise ValueError(f'Cannot compute the parent namespace of: {obj}')
        else:
            cls = get_parent_namespace(obj)

        if cls and (not inspect.ismodule(cls)) and issubclass(cls, AnalysisHelpers):
            on_trace_name = f'trace.ana.{cls.name}.{obj.__name__}'
            return f"*Called on* :class:`~lisa.trace.Trace` *instances as* ``{on_trace_name}()``"
        else:
            raise ValueError(f'{obj} is not a method of an analysis class')
    else:
        raise ValueError(f'{obj} is not a method')


def autodoc_process_analysis_methods(app, what, name, obj, options, lines):
    """
    Append the list of required trace events
    """
    try:
        extra_doc = ana_invocation(obj, name)
    except ValueError:
        pass
    else:
        extra_doc = f"\n{extra_doc}\n\n"
        # prepend
        lines[:0] = extra_doc.splitlines()


def get_analysis_list(meth_type):
    rst_list = []

    deprecated = {
        entry['obj']
        for entry in get_deprecated_map().values()
    }

    # Ensure all the submodules have been imported
    TraceAnalysisBase.get_analysis_classes()

    assert issubclass(TraceAnalysisBase, AnalysisHelpers)
    for subclass in get_subclasses(AnalysisHelpers):
        class_path = f"{subclass.__module__}.{subclass.__qualname__}"
        if meth_type == 'plot':
            meth_list = subclass.get_plot_methods()
        elif meth_type == 'df':
            meth_list = (
                subclass.get_df_methods()
                if issubclass(subclass, TraceAnalysisBase) else
                []
            )
        else:
            raise ValueError()

        meth_list = [
            f.__name__
            for f in meth_list
            if f not in deprecated
        ]

        rst_list += [
            f":class:`{subclass.name}<{class_path}>`::meth:`~{class_path}.{meth}`"
            for meth in meth_list
        ]

    joiner = '\n* '
    return joiner + joiner.join(sorted(rst_list))


def find_dead_links(content):
    """
    Look for HTTP URLs in ``content`` and return a dict of URL to errors when
    trying to open them.
    """
    regex = r"https?://[^\s]+"
    links = re.findall(regex, content)

    @functools.lru_cache(maxsize=None)
    def check(url):
        return is_link_dead(url)

    errors = {
        link: check(link)
        for link in links
        if check(link) is not None
    }
    return errors


def check_dead_links(filename):
    """
    Check ``filename`` for broken links, and raise an exception if there is any.
    """
    with open(filename) as f:
        dead_links = find_dead_links(f.read())

    if dead_links:
        raise RuntimeError('Found dead links in {}:\n  {}'.format(
            filename,
            '\n  '.join(
                f'{url}: {error}'
                for url, error in dead_links.items()
            )))


@functools.lru_cache()
def get_deprecated_map():
    """
    Get the mapping of deprecated names with some metadata.
    """

    # Import everything there is to import, so the map is fully populated
    import_all_submodules(lisa, best_effort=True)
    return _DEPRECATED_MAP

def get_deprecated_table():
    """
    Get a reStructuredText tables with titles for all the deprecated names in
    :mod:`lisa`.
    """

    def indent(string):
        idt = ' ' * 4
        return string.replace('\n', '\n' + idt)

    def make_entry(entry):
        msg = entry.get('msg') or ''
        removed_in = entry.get('removed_in')
        if removed_in is None:
            removed_in = ''
        else:
            removed_in = f'*Removed in: {format_version(removed_in)}*\n\n'

        name = get_obj_name(entry['obj'], style='rst')
        replaced_by = entry.get('replaced_by')

        if replaced_by is None:
            replaced_by = ''
        else:
            if isinstance(replaced_by, str):
                replaced_by = str(replaced_by)
            else:
                replaced_by = get_obj_name(replaced_by, style='rst')

            replaced_by = f"*Replaced by:* {replaced_by}\n\n"

        return "* - {name}{msg}{replaced_by}{removed_in}".format(
            name=indent(name + '\n\n'),
            msg=indent(msg + '\n\n' if msg else ''),
            replaced_by=indent(replaced_by),
            removed_in=indent(removed_in),
        )

    def make_table(entries, removed_in):
        if entries:
            entries = '\n'.join(
                make_entry(entry)
                for entry in sorted(entries, key=itemgetter('name'))
            )
            if removed_in:
                if removed_in > lisa.version.version_tuple:
                    remove = 'to be removed'
                else:
                    remove = 'removed'
                removed_in = f' {remove} in {format_version(removed_in)}'
            else:
                removed_in = ''

            table = ".. list-table:: Deprecated names{removed_in}\n    :align: left{entries}".format(
                entries=indent('\n\n' + entries),
                removed_in=removed_in,
            )
            header = f'Deprecated names{removed_in}'
            header += '\n' + '+' * len(header)

            return header + '\n\n' + table
        else:
            return ''

    entries = [
        {'name': name, **info}
        for name, info in get_deprecated_map().items()
    ]

    unspecified_removal = [
        entry
        for entry in entries
        if not entry['removed_in']
    ]

    other_entries = [
        entry
        for entry in entries
        if entry not in unspecified_removal
    ]

    tables = []
    tables.append(make_table(unspecified_removal, removed_in=None))
    tables.extend(
        make_table(entries, removed_in=removed_in)
        for removed_in, entries in groupby(other_entries, itemgetter('removed_in'), reverse=True)
    )

    return '\n\n'.join(tables)


def get_subclasses_bullets(cls, abbrev=True, style=None, only_leaves=False):
    """
    Return a formatted bullet list of the subclasses of the given class,
    including a short description for each.
    """
    return '\n'.join(
        f'* {subcls}: {doc}'
        for subcls, doc in sorted(
            (
                get_obj_name(subcls, style=style, abbrev=abbrev),
                get_short_doc(subcls, style=style)
            )
            for subcls in get_subclasses(cls, only_leaves=only_leaves)
        )
    )


def make_changelog(repo, since=None, head_release_name='Next release', fmt='rst'):
    """
    Generate a reStructuredText changelog to be included in the documentation.

    .. note:: The git repository cannot be a shallow clone, as the changelog is
        extracted from the git history.

    .. note:: The ``refs/notes/changelog`` notes is concatenated at the end of
        commit messages, and the resulting text is parsed. This allows fixing
        up changelog entries if markers were forgotten without rewriting the
        history.
    """

    if fmt == 'rst':
        escape_fmt = escape_rst
    else:
        escape_fmt = lambda x: x


    notes_ref = 'refs/notes/changelog'
    release_refs = (
        ['HEAD'] + (
            [since]
            if since else
            lisa._git.find_tags(repo, 'v*')
        )
    )

    def update_release_name(name):
        if name == 'HEAD':
            return head_release_name
        else:
            return name

    MARKERS = ['FEATURE', 'FIX', 'BREAKING CHANGE']

    # Filtering on the patterns we look for provides a considerable speedup
    commit_pattern = '(' + '|'.join(map(re.escape, MARKERS)) + ')'
    release_sha1s = {
        update_release_name(y): lisa._git.find_commits(
            repo=repo,
            ref=f'{x}..{y}',
            grep=commit_pattern,
            regex=True,
            notes_ref=notes_ref,
        )
        for x, y in zip(release_refs[1:], release_refs)
    }

    release_msgs = {
        release: [
            lisa._git.get_commit_message(
                repo=repo,
                ref=ref,
                notes_ref=notes_ref,
                format='%B%N',
            ).strip()
            for ref in refs
        ]
        for release, refs in release_sha1s.items()
    }

    def parse_msg(msg):
        selected = tuple(sorted({
            marker
            for marker in MARKERS
            if marker in msg
        }))
        for marker in selected:
            pattern = rf'^\s*{re.escape(marker)}\s*$'
            msg = re.sub(pattern, '', msg, flags=re.MULTILINE)

        return (msg, selected)

    def expand(msg, markers):
        for marker in markers:
            yield (marker, msg)

    release_msgs = {
        release: dict(
            map(
                lambda x: (x[0], list(map(itemgetter(1), x[1]))),
                groupby(
                    (
                        entry
                        for msg in msgs
                        for entry in expand(*parse_msg(msg))
                    ),
                    key=itemgetter(0)
                )
            )
        )
        for release, msgs in release_msgs.items()
    }

    def format_release(name, sections):
        title = f'{name}\n{len(name) * "="}\n'
        body = '\n\n'.join(
            format_section(marker, _msgs)
            for marker, _msgs in order_as(
                sections.items(),
                order_as=MARKERS,
                key=itemgetter(0),
            )
        )

        return f'{title}\n{body}'

    def format_section(name, msgs):
        title = f'{name.capitalize()}\n{len(name) * "+"}\n'
        body = '\n\n'.join(map(format_msg, sorted(msgs)))
        return f'{title}\n{body}'

    def format_msg(msg):
        subject = escape_fmt(msg.splitlines()[0].strip())
        return f'- {subject}'

    rst = '\n\n'.join(
        format_release(name, sections)
        for name, sections in release_msgs.items()
    )

    return rst


def escape_rst(s):
    """
    Escape the string so that it's considered plain reStructuredText input,
    without any markup even if it contains some. This avoids having to use a
    literal block that is displayed differently.
    """
    # https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#escaping-mechanism
    return re.sub(r'(?=[^\s])([^\\])', r'\\\1', s)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
