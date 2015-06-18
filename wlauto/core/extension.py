#    Copyright 2013-2015 ARM Limited
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


# pylint: disable=E1101
import os
import logging
import inspect
from copy import copy
from collections import OrderedDict

from wlauto.core.bootstrap import settings
from wlauto.exceptions import ValidationError, ConfigError
from wlauto.utils.misc import isiterable, ensure_directory_exists as _d, get_article
from wlauto.utils.types import identifier, integer, boolean


class AttributeCollection(object):
    """
    Accumulator for extension attribute objects (such as Parameters or Artifacts). This will
    replace any class member list accumulating such attributes through the magic of
    metaprogramming\ [*]_.

    .. [*] which is totally safe and not going backfire in any way...

    """

    @property
    def values(self):
        return self._attrs.values()

    def __init__(self, attrcls):
        self._attrcls = attrcls
        self._attrs = OrderedDict()

    def add(self, p):
        p = self._to_attrcls(p)
        if p.name in self._attrs:
            if p.override:
                newp = copy(self._attrs[p.name])
                for a, v in p.__dict__.iteritems():
                    if v is not None:
                        setattr(newp, a, v)
                self._attrs[p.name] = newp
            else:
                # Duplicate attribute condition is check elsewhere.
                pass
        else:
            self._attrs[p.name] = p

    append = add

    def __str__(self):
        return 'AC({})'.format(map(str, self._attrs.values()))

    __repr__ = __str__

    def _to_attrcls(self, p):
        if isinstance(p, basestring):
            p = self._attrcls(p)
        elif isinstance(p, tuple) or isinstance(p, list):
            p = self._attrcls(*p)
        elif isinstance(p, dict):
            p = self._attrcls(**p)
        elif not isinstance(p, self._attrcls):
            raise ValueError('Invalid parameter value: {}'.format(p))
        if (p.name in self._attrs and not p.override and
                p.name != 'modules'):  # TODO: HACK due to "diamond dependecy" in workloads...
            raise ValueError('Attribute {} has already been defined.'.format(p.name))
        return p

    def __iadd__(self, other):
        for p in other:
            self.add(p)
        return self

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, p):
        return p in self._attrs

    def __getitem__(self, i):
        return self._attrs[i]

    def __len__(self):
        return len(self._attrs)


class AliasCollection(AttributeCollection):

    def __init__(self):
        super(AliasCollection, self).__init__(Alias)

    def _to_attrcls(self, p):
        if isinstance(p, tuple) or isinstance(p, list):
            # must be in the form (name, {param: value, ...})
            p = self._attrcls(p[1], **p[1])
        elif not isinstance(p, self._attrcls):
            raise ValueError('Invalid parameter value: {}'.format(p))
        if p.name in self._attrs:
            raise ValueError('Attribute {} has already been defined.'.format(p.name))
        return p


class ListCollection(list):

    def __init__(self, attrcls):  # pylint: disable=unused-argument
        super(ListCollection, self).__init__()


class Param(object):
    """
    This is a generic parameter for an extension. Extensions instantiate this to declare which parameters
    are supported.

    """

    # Mapping for kind conversion; see docs for convert_types below
    kind_map = {
        int: integer,
        bool: boolean,
    }

    def __init__(self, name, kind=None, mandatory=None, default=None, override=False,
                 allowed_values=None, description=None, constraint=None, global_alias=None, convert_types=True):
        """
        Create a new Parameter object.

        :param name: The name of the parameter. This will become an instance member of the
                     extension object to which the parameter is applied, so it must be a valid
                     python  identifier. This is the only mandatory parameter.
        :param kind: The type of parameter this is. This must be a callable that takes an arbitrary
                     object and converts it to the expected type, or raised ``ValueError`` if such
                     conversion is not possible. Most Python standard types -- ``str``, ``int``, ``bool``, etc. --
                     can be used here. This defaults to ``str`` if not specified.
        :param mandatory: If set to ``True``, then a non-``None`` value for this parameter *must* be
                          provided on extension object construction, otherwise ``ConfigError`` will be
                          raised.
        :param default: The default value for this parameter. If no value is specified on extension
                        construction, this value will be used instead. (Note: if this is specified and
                        is not ``None``, then ``mandatory`` parameter will be ignored).
        :param override: A ``bool`` that specifies whether a parameter of the same name further up the
                         hierarchy should be overridden. If this is ``False`` (the default), an exception
                         will be raised by the ``AttributeCollection`` instead.
        :param allowed_values: This should be the complete list of allowed values for this parameter.
                               Note: ``None`` value will always be allowed, even if it is not in this list.
                               If you want to disallow ``None``, set ``mandatory`` to ``True``.
        :param constraint: If specified, this must be a callable that takes the parameter value
                           as an argument and return a boolean indicating whether the constraint
                           has been satisfied. Alternatively, can be a two-tuple with said callable as
                           the first element and a string describing the constraint as the second.
        :param global_alias: This is an alternative alias for this parameter, unlike the name, this
                             alias will not be namespaced under the owning extension's name (hence the
                             global part). This is introduced primarily for backward compatibility -- so
                             that old extension settings names still work. This should not be used for
                             new parameters.

        :param convert_types: If ``True`` (the default), will automatically convert ``kind`` values from
                              native Python types to WA equivalents. This allows more ituitive interprestation
                              of parameter values, e.g. the string ``"false"`` being interpreted as ``False``
                              when specifed as the value for a boolean Parameter.

        """
        self.name = identifier(name)
        if kind is not None and not callable(kind):
            raise ValueError('Kind must be callable.')
        if convert_types and kind in self.kind_map:
            kind = self.kind_map[kind]
        self.kind = kind
        self.mandatory = mandatory
        self.default = default
        self.override = override
        self.allowed_values = allowed_values
        self.description = description
        if self.kind is None and not self.override:
            self.kind = str
        if constraint is not None and not callable(constraint) and not isinstance(constraint, tuple):
            raise ValueError('Constraint must be callable or a (callable, str) tuple.')
        self.constraint = constraint
        self.global_alias = global_alias

    def set_value(self, obj, value=None):
        if value is None:
            if self.default is not None:
                value = self.default
            elif self.mandatory:
                msg = 'No values specified for mandatory parameter {} in {}'
                raise ConfigError(msg.format(self.name, obj.name))
        else:
            try:
                value = self.kind(value)
            except (ValueError, TypeError):
                typename = self.get_type_name()
                msg = 'Bad value "{}" for {}; must be {} {}'
                article = get_article(typename)
                raise ConfigError(msg.format(value, self.name, article, typename))
        current_value = getattr(obj, self.name, None)
        if current_value is None:
            setattr(obj, self.name, value)
        elif not isiterable(current_value):
            setattr(obj, self.name, value)
        else:
            new_value = current_value + [value]
            setattr(obj, self.name, new_value)

    def validate(self, obj):
        value = getattr(obj, self.name, None)
        if value is not None:
            if self.allowed_values:
                self._validate_allowed_values(obj, value)
            if self.constraint:
                self._validate_constraint(obj, value)
        else:
            if self.mandatory:
                msg = 'No value specified for mandatory parameter {} in {}.'
                raise ConfigError(msg.format(self.name, obj.name))

    def get_type_name(self):
        typename = str(self.kind)
        if '\'' in typename:
            typename = typename.split('\'')[1]
        elif typename.startswith('<function'):
            typename = typename.split()[1]
        return typename

    def _validate_allowed_values(self, obj, value):
        if 'list' in str(self.kind):
            for v in value:
                if v not in self.allowed_values:
                    msg = 'Invalid value {} for {} in {}; must be in {}'
                    raise ConfigError(msg.format(v, self.name, obj.name, self.allowed_values))
        else:
            if value not in self.allowed_values:
                msg = 'Invalid value {} for {} in {}; must be in {}'
                raise ConfigError(msg.format(value, self.name, obj.name, self.allowed_values))

    def _validate_constraint(self, obj, value):
        msg_vals = {'value': value, 'param': self.name, 'extension': obj.name}
        if isinstance(self.constraint, tuple) and len(self.constraint) == 2:
            constraint, msg = self.constraint  # pylint: disable=unpacking-non-sequence
        elif callable(self.constraint):
            constraint = self.constraint
            msg = '"{value}" failed constraint validation for {param} in {extension}.'
        else:
            raise ValueError('Invalid constraint for {}: must be callable or a 2-tuple'.format(self.name))
        if not constraint(value):
            raise ConfigError(value, msg.format(**msg_vals))

    def __repr__(self):
        d = copy(self.__dict__)
        del d['description']
        return 'Param({})'.format(d)

    __str__ = __repr__


Parameter = Param


class Artifact(object):
    """
    This is an artifact generated during execution/post-processing of a workload.
    Unlike metrics, this represents an actual artifact, such as a file, generated.
    This may be "result", such as trace, or it could be "meta data" such as logs.
    These are distinguished using the ``kind`` attribute, which also helps WA decide
    how it should be handled. Currently supported kinds are:

        :log: A log file. Not part of "results" as such but contains information about the
              run/workload execution that be useful for diagnostics/meta analysis.
        :meta: A file containing metadata. This is not part of "results", but contains
               information that may be necessary to reproduce the results (contrast with
               ``log`` artifacts which are *not* necessary).
        :data: This file contains new data, not available otherwise and should be considered
               part of the "results" generated by WA. Most traces would fall into this category.
        :export: Exported version of results or some other artifact. This signifies that
                 this artifact does not contain any new data that is not available
                 elsewhere and that it may be safely discarded without losing information.
        :raw: Signifies that this is a raw dump/log that is normally processed to extract
              useful information and is then discarded. In a sense, it is the opposite of
              ``export``, but in general may also be discarded.

              .. note:: whether a file is marked as ``log``/``data`` or ``raw`` depends on
                        how important it is to preserve this file, e.g. when archiving, vs
                        how much space it takes up. Unlike ``export`` artifacts which are
                        (almost) always ignored by other exporters as that would never result
                        in data loss, ``raw`` files *may* be processed by exporters if they
                        decided that the risk of losing potentially (though unlikely) useful
                        data is greater than the time/space cost of handling the artifact (e.g.
                        a database uploader may choose to ignore ``raw`` artifacts, where as a
                        network filer archiver may choose to archive them).

        .. note: The kind parameter is intended to represent the logical function of a particular
                 artifact, not it's intended means of processing -- this is left entirely up to the
                 result processors.

    """

    RUN = 'run'
    ITERATION = 'iteration'

    valid_kinds = ['log', 'meta', 'data', 'export', 'raw']

    def __init__(self, name, path, kind, level=RUN, mandatory=False, description=None):
        """"
        :param name: Name that uniquely identifies this artifact.
        :param path: The *relative* path of the artifact. Depending on the ``level``
                     must be either relative to the run or iteration output directory.
                     Note: this path *must* be delimited using ``/`` irrespective of the
                     operating system.
        :param kind: The type of the artifact this is (e.g. log file, result, etc.) this
                     will be used a hit to result processors. This must be one of ``'log'``,
                     ``'meta'``, ``'data'``, ``'export'``, ``'raw'``.
        :param level: The level at which the artifact will be generated. Must be either
                      ``'iteration'`` or ``'run'``.
        :param mandatory: Boolean value indicating whether this artifact must be present
                          at the end of result processing for its level.
        :param description: A free-form description of what this artifact is.

        """
        if kind not in self.valid_kinds:
            raise ValueError('Invalid Artifact kind: {}; must be in {}'.format(kind, self.valid_kinds))
        self.name = name
        self.path = path.replace('/', os.sep) if path is not None else path
        self.kind = kind
        self.level = level
        self.mandatory = mandatory
        self.description = description

    def exists(self, context):
        """Returns ``True`` if artifact exists within the specified context, and
        ``False`` otherwise."""
        fullpath = os.path.join(context.output_directory, self.path)
        return os.path.exists(fullpath)

    def to_dict(self):
        return copy(self.__dict__)


class Alias(object):
    """
    This represents a configuration alias for an extension, mapping an alternative name to
    a set of parameter values, effectively providing an alternative set of default values.

    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.params = kwargs
        self.extension_name = None  # gets set by the MetaClass

    def validate(self, ext):
        ext_params = set(p.name for p in ext.parameters)
        for param in self.params:
            if param not in ext_params:
                # Raising config error because aliases might have come through
                # the config.
                msg = 'Parameter {} (defined in alias {}) is invalid for {}'
                raise ConfigError(msg.format(param, self.name, ext.name))


class ExtensionMeta(type):
    """
    This basically adds some magic to extensions to make implementing new extensions, such as
    workloads less complicated.

    It ensures that certain class attributes (specified by the ``to_propagate``
    attribute of the metaclass) get propagated down the inheritance hierarchy. The assumption
    is that the values of the attributes specified in the class are iterable; if that is not met,
    Bad Things (tm) will happen.

    This also provides virtual method implementation, similar to those in C-derived OO languages,
    and alias specifications.

    """

    to_propagate = [
        ('parameters', Parameter, AttributeCollection),
        ('artifacts', Artifact, AttributeCollection),
        ('core_modules', str, ListCollection),
    ]

    virtual_methods = ['validate', 'initialize', 'finalize']
    global_virtuals = ['initialize', 'finalize']

    def __new__(mcs, clsname, bases, attrs):
        mcs._propagate_attributes(bases, attrs)
        cls = type.__new__(mcs, clsname, bases, attrs)
        mcs._setup_aliases(cls)
        mcs._implement_virtual(cls, bases)
        return cls

    @classmethod
    def _propagate_attributes(mcs, bases, attrs):
        """
        For attributes specified by to_propagate, their values will be a union of
        that specified for cls and it's bases (cls values overriding those of bases
        in case of conflicts).

        """
        for prop_attr, attr_cls, attr_collector_cls in mcs.to_propagate:
            should_propagate = False
            propagated = attr_collector_cls(attr_cls)
            for base in bases:
                if hasattr(base, prop_attr):
                    propagated += getattr(base, prop_attr) or []
                    should_propagate = True
            if prop_attr in attrs:
                propagated += attrs[prop_attr] or []
                should_propagate = True
            if should_propagate:
                attrs[prop_attr] = propagated

    @classmethod
    def _setup_aliases(mcs, cls):
        if hasattr(cls, 'aliases'):
            aliases, cls.aliases = cls.aliases, AliasCollection()
            for alias in aliases:
                if isinstance(alias, basestring):
                    alias = Alias(alias)
                alias.validate(cls)
                alias.extension_name = cls.name
                cls.aliases.add(alias)

    @classmethod
    def _implement_virtual(mcs, cls, bases):
        """
        This implements automatic method propagation to the bases, so
        that you don't have to do something like

            super(cls, self).vmname()

        This also ensures that the methods that have beend identified as
        "globally virtual" are executed exactly once per WA execution, even if
        invoked through instances of different subclasses

        """
        methods = {}
        called_globals = set()
        for vmname in mcs.virtual_methods:
            clsmethod = getattr(cls, vmname, None)
            if clsmethod:
                basemethods = [getattr(b, vmname) for b in bases if hasattr(b, vmname)]
                methods[vmname] = [bm for bm in basemethods if bm != clsmethod]
                methods[vmname].append(clsmethod)

                def generate_method_wrapper(vname):  # pylint: disable=unused-argument
                    # this creates a closure with the method name so that it
                    # does not need to be passed to the wrapper as an argument,
                    # leaving the wrapper to accept exactly the same set of
                    # arguments as the method it is wrapping.
                    name__ = vmname  # pylint: disable=cell-var-from-loop

                    def wrapper(self, *args, **kwargs):
                        for dm in methods[name__]:
                            if name__ in mcs.global_virtuals:
                                if dm not in called_globals:
                                    dm(self, *args, **kwargs)
                                    called_globals.add(dm)
                            else:
                                dm(self, *args, **kwargs)
                    return wrapper

                setattr(cls, vmname, generate_method_wrapper(vmname))


class Extension(object):
    """
    Base class for all WA extensions. An extension is basically a plug-in.
    It extends the functionality of WA in some way. Extensions are discovered
    and loaded dynamically by the extension loader upon invocation of WA scripts.
    Adding an extension is a matter of placing a class that implements an appropriate
    interface somewhere it would be discovered by the loader. That "somewhere" is
    typically one of the extension subdirectories under ``~/.workload_automation/``.

    """
    __metaclass__ = ExtensionMeta

    kind = None
    name = None
    parameters = [
        Parameter('modules', kind=list,
                  description="""
                  Lists the modules to be loaded by this extension. A module is a plug-in that
                  further extends functionality of an extension.
                  """),
    ]
    artifacts = []
    aliases = []
    core_modules = []

    @classmethod
    def get_default_config(cls):
        return {p.name: p.default for p in cls.parameters}

    @property
    def dependencies_directory(self):
        return _d(os.path.join(settings.dependencies_directory, self.name))

    @property
    def _classname(self):
        return self.__class__.__name__

    def __init__(self, **kwargs):
        self.__check_from_loader()
        self.logger = logging.getLogger(self._classname)
        self._modules = []
        self.capabilities = getattr(self.__class__, 'capabilities', [])
        for param in self.parameters:
            param.set_value(self, kwargs.get(param.name))
        for key in kwargs:
            if key not in self.parameters:
                message = 'Unexpected parameter "{}" for {}'
                raise ConfigError(message.format(key, self.name))

    def get_config(self):
        """
        Returns current configuration (i.e. parameter values) of this extension.

        """
        config = {}
        for param in self.parameters:
            config[param.name] = getattr(self, param.name, None)
        return config

    def validate(self):
        """
        Perform basic validation to ensure that this extension is capable of running.
        This is intended as an early check to ensure the extension has not been mis-configured,
        rather than a comprehensive check (that may, e.g., require access to the execution
        context).

        This method may also be used to enforce (i.e. set as well as check) inter-parameter
        constraints for the extension (e.g. if valid values for parameter A depend on the value
        of parameter B -- something that is not possible to enfroce using ``Parameter``\ 's
        ``constraint`` attribute.

        """
        if self.name is None:
            raise ValidationError('Name not set for {}'.format(self._classname))
        for param in self.parameters:
            param.validate(self)

    def initialize(self, context):
        pass

    def finalize(self, context):
        pass

    def check_artifacts(self, context, level):
        """
        Make sure that all mandatory artifacts have been generated.

        """
        for artifact in self.artifacts:
            if artifact.level != level or not artifact.mandatory:
                continue
            fullpath = os.path.join(context.output_directory, artifact.path)
            if not os.path.exists(fullpath):
                message = 'Mandatory "{}" has not been generated for {}.'
                raise ValidationError(message.format(artifact.path, self.name))

    def __getattr__(self, name):
        if name == '_modules':
            raise ValueError('_modules accessed too early!')
        for module in self._modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)

    def load_modules(self, loader):
        """
        Load the modules specified by the "modules" Parameter using the provided loader. A loader
        can be any object that has an atribute called "get_module" that implements the following
        signature::

            get_module(name, owner, **kwargs)

        and returns an instance of :class:`wlauto.core.extension.Module`. If the module with the
        specified name is not found, the loader must raise an appropriate exception.

        """
        modules = list(reversed(self.core_modules)) + list(reversed(self.modules or []))
        if not modules:
            return
        for module_spec in modules:
            if not module_spec:
                continue
            module = self._load_module(loader, module_spec)
            self._install_module(module)

    def has(self, capability):
        """Check if this extension has the specified capability. The alternative method ``can`` is
        identical to this. Which to use is up to the caller depending on what makes semantic sense
        in the context of the capability, e.g. ``can('hard_reset')`` vs  ``has('active_cooling')``."""
        return capability in self.capabilities

    can = has

    def _load_module(self, loader, module_spec):
        if isinstance(module_spec, basestring):
            name = module_spec
            params = {}
        elif isinstance(module_spec, dict):
            if len(module_spec) != 1:
                message = 'Invalid module spec: {}; dict must have exctly one key -- the module name.'
                raise ValueError(message.format(module_spec))
            name, params = module_spec.items()[0]
        else:
            message = 'Invalid module spec: {}; must be a string or a one-key dict.'
            raise ValueError(message.format(module_spec))

        if not isinstance(params, dict):
            message = 'Invalid module spec: {}; dict value must also be a dict.'
            raise ValueError(message.format(module_spec))

        module = loader.get_module(name, owner=self, **params)
        module.initialize(None)
        return module

    def _install_module(self, module):
        for capability in module.capabilities:
            if capability not in self.capabilities:
                self.capabilities.append(capability)
        self._modules.append(module)

    def __check_from_loader(self):
        """
        There are a few things that need to happen in order to get a valide extension instance.
        Not all of them are currently done through standard Python initialisation mechanisms
        (specifically, the loading of modules and alias resolution). In order to avoid potential
        problems with not fully loaded extensions, make sure that an extension is *only* instantiated
        by the loader.

        """
        stack = inspect.stack()
        stack.pop(0)  # current frame
        frame = stack.pop(0)
        # skip throuth the init call chain
        while stack and frame[3] == '__init__':
            frame = stack.pop(0)
        if frame[3] != '_instantiate':
            message = 'Attempting to instantiate {} directly (must be done through an ExtensionLoader)'
            raise RuntimeError(message.format(self.__class__.__name__))


class Module(Extension):
    """
    This is a "plugin" for an extension this is intended to capture functionality that may be optional
    for an extension, and so may or may not be present in a particular setup; or, conversely, functionality
    that may be reusable between multiple devices, even if they are not with the same inheritance hierarchy.

    In other words, a Module is roughly equivalent to a kernel module and its primary purpose is to
    implement WA "drivers" for various peripherals that may or may not be present in a particular setup.

    .. note:: A mudule is itself an Extension and can therefore have it's own modules.

    """

    capabilities = []

    @property
    def root_owner(self):
        owner = self.owner
        while isinstance(owner, Module) and owner is not self:
            owner = owner.owner
        return owner

    def __init__(self, owner, **kwargs):
        super(Module, self).__init__(**kwargs)
        self.owner = owner
        while isinstance(owner, Module):
            if owner.name == self.name:
                raise ValueError('Circular module import for {}'.format(self.name))

    def initialize(self, context):
        pass

