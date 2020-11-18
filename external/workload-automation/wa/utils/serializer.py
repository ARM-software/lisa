#    Copyright 2018 ARM Limited
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
This module contains wrappers for Python serialization modules for
common formats that make it easier to serialize/deserialize WA
Plain Old Data structures (serilizable WA classes implement
``to_pod()``/``from_pod()`` methods for converting between POD
structures and Python class instances).

The modifications to standard serilization procedures are:

    - mappings are deserialized as ``OrderedDict``\ 's rather than standard
      Python ``dict``\ 's. This allows for cleaner syntax in certain parts
      of WA configuration (e.g. values to be written to files can be specified
      as a dict, and they will be written in the order specified in the config).
    - regular expressions are automatically encoded/decoded. This allows for
      configuration values to be transparently specified as strings or regexes
      in the POD config.

This module exports the "wrapped" versions of serialization libraries,
and this should be imported and used instead of importing the libraries
directly. i.e. ::

    from wa.utils.serializer import yaml
    pod = yaml.load(fh)

instead of ::

    import yaml
    pod = yaml.load(fh)

It's also possible to use the serializer directly::

    from wa.utils import serializer
    pod = serializer.load(fh)

This can also be used to ``dump()`` POD structures. By default,
``dump()`` will produce JSON, but ``fmt`` parameter may be used to
specify an alternative format (``yaml`` or ``python``). ``load()`` will
use the file plugin to guess the format, but ``fmt`` may also be used
to specify it explicitly.

"""
# pylint: disable=unused-argument

import os
import re
import json as _json
from collections import OrderedDict
from collections.abc import Hashable
from datetime import datetime
import dateutil.parser
import yaml as _yaml  # pylint: disable=wrong-import-order
from yaml import MappingNode
try:
    from yaml import FullLoader as _yaml_loader
except ImportError:
    from yaml import Loader as _yaml_loader
from yaml.constructor import ConstructorError


# pylint: disable=redefined-builtin
from past.builtins import basestring  # pylint: disable=wrong-import-order

from wa.framework.exception import SerializerSyntaxError
from wa.utils.misc import isiterable
from wa.utils.types import regex_type, none_type, level, cpu_mask


__all__ = [
    'json',
    'yaml',
    'read_pod',
    'dump',
    'load',
    'is_pod',
    'POD_TYPES',
]

POD_TYPES = [
    list,
    tuple,
    dict,
    set,
    basestring,
    str,
    int,
    float,
    bool,
    OrderedDict,
    datetime,
    regex_type,
    none_type,
    level,
    cpu_mask,
]


class WAJSONEncoder(_json.JSONEncoder):

    def default(self, obj):  # pylint: disable=method-hidden,arguments-differ
        if isinstance(obj, regex_type):
            return 'REGEX:{}:{}'.format(obj.flags, obj.pattern)
        elif isinstance(obj, datetime):
            return 'DATET:{}'.format(obj.isoformat())
        elif isinstance(obj, level):
            return 'LEVEL:{}:{}'.format(obj.name, obj.value)
        elif isinstance(obj, cpu_mask):
            return 'CPUMASK:{}'.format(obj.mask())
        else:
            return _json.JSONEncoder.default(self, obj)


class WAJSONDecoder(_json.JSONDecoder):

    def decode(self, s, **kwargs):  # pylint: disable=arguments-differ
        d = _json.JSONDecoder.decode(self, s, **kwargs)

        def try_parse_object(v):
            if isinstance(v, basestring):
                if v.startswith('REGEX:'):
                    _, flags, pattern = v.split(':', 2)
                    return re.compile(pattern, int(flags or 0))
                elif v.startswith('DATET:'):
                    _, pattern = v.split(':', 1)
                    return dateutil.parser.parse(pattern)
                elif v.startswith('LEVEL:'):
                    _, name, value = v.split(':', 2)
                    return level(name, value)
                elif v.startswith('CPUMASK:'):
                    _, value = v.split(':', 1)
                    return cpu_mask(value)

            return v

        def load_objects(d):
            if not hasattr(d, 'items'):
                return d
            pairs = []
            for k, v in d.items():
                if hasattr(v, 'items'):
                    pairs.append((k, load_objects(v)))
                elif isiterable(v):
                    pairs.append((k, [try_parse_object(i) for i in v]))
                else:
                    pairs.append((k, try_parse_object(v)))
            return OrderedDict(pairs)

        return load_objects(d)


class json(object):

    @staticmethod
    def dump(o, wfh, indent=4, *args, **kwargs):
        return _json.dump(o, wfh, cls=WAJSONEncoder, indent=indent, *args, **kwargs)

    @staticmethod
    def dumps(o, indent=4, *args, **kwargs):
        return _json.dumps(o, cls=WAJSONEncoder, indent=indent, *args, **kwargs)

    @staticmethod
    def load(fh, *args, **kwargs):
        try:
            return _json.load(fh, cls=WAJSONDecoder, object_pairs_hook=OrderedDict, *args, **kwargs)
        except ValueError as e:
            raise SerializerSyntaxError(e.args[0])

    @staticmethod
    def loads(s, *args, **kwargs):
        try:
            return _json.loads(s, cls=WAJSONDecoder, object_pairs_hook=OrderedDict, *args, **kwargs)
        except ValueError as e:
            raise SerializerSyntaxError(e.args[0])


_mapping_tag = _yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
_regex_tag = 'tag:wa:regex'
_level_tag = 'tag:wa:level'
_cpu_mask_tag = 'tag:wa:cpu_mask'


def _wa_dict_representer(dumper, data):
    return dumper.represent_mapping(_mapping_tag, iter(data.items()))


def _wa_regex_representer(dumper, data):
    text = '{}:{}'.format(data.flags, data.pattern)
    return dumper.represent_scalar(_regex_tag, text)


def _wa_level_representer(dumper, data):
    text = '{}:{}'.format(data.name, data.level)
    return dumper.represent_scalar(_level_tag, text)


def _wa_cpu_mask_representer(dumper, data):
    return dumper.represent_scalar(_cpu_mask_tag, data.mask())


def _wa_regex_constructor(loader, node):
    value = loader.construct_scalar(node)
    flags, pattern = value.split(':', 1)
    return re.compile(pattern, int(flags or 0))


def _wa_level_constructor(loader, node):
    value = loader.construct_scalar(node)
    name, value = value.split(':', 1)
    return level(name, value)


def _wa_cpu_mask_constructor(loader, node):
    value = loader.construct_scalar(node)
    return cpu_mask(value)


class _WaYamlLoader(_yaml_loader):  # pylint: disable=too-many-ancestors

    def construct_mapping(self, node, deep=False):
        if isinstance(node, MappingNode):
            self.flatten_mapping(node)
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None,
                                   "expected a mapping node, but found %s" % node.id,
                                   node.start_mark)
        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError("while constructing a mapping", node.start_mark,
                                       "found unhashable key", key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


_yaml.add_representer(OrderedDict, _wa_dict_representer)
_yaml.add_representer(regex_type, _wa_regex_representer)
_yaml.add_representer(level, _wa_level_representer)
_yaml.add_representer(cpu_mask, _wa_cpu_mask_representer)
_yaml.add_constructor(_regex_tag, _wa_regex_constructor, Loader=_WaYamlLoader)
_yaml.add_constructor(_level_tag, _wa_level_constructor, Loader=_WaYamlLoader)
_yaml.add_constructor(_cpu_mask_tag, _wa_cpu_mask_constructor, Loader=_WaYamlLoader)
_yaml.add_constructor(_mapping_tag, _WaYamlLoader.construct_yaml_map, Loader=_WaYamlLoader)


class yaml(object):

    @staticmethod
    def dump(o, wfh, *args, **kwargs):
        return _yaml.dump(o, wfh, *args, **kwargs)

    @staticmethod
    def load(fh, *args, **kwargs):
        try:
            return _yaml.load(fh, *args, Loader=_WaYamlLoader, **kwargs)
        except _yaml.YAMLError as e:
            lineno = None
            if hasattr(e, 'problem_mark'):
                lineno = e.problem_mark.line  # pylint: disable=no-member
            message = e.args[0] if (e.args and e.args[0]) else str(e)
            raise SerializerSyntaxError(message, lineno)

    loads = load


class python(object):

    @staticmethod
    def dump(o, wfh, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def load(cls, fh, *args, **kwargs):
        return cls.loads(fh.read())

    @staticmethod
    def loads(s, *args, **kwargs):
        pod = {}
        try:
            exec(s, pod)  # pylint: disable=exec-used
        except SyntaxError as e:
            raise SerializerSyntaxError(e.message, e.lineno)
        for k in list(pod.keys()):  # pylint: disable=consider-iterating-dictionary
            if k.startswith('__'):
                del pod[k]
        return pod


def read_pod(source, fmt=None):
    if isinstance(source, str):
        with open(source) as fh:
            return _read_pod(fh, fmt)
    elif hasattr(source, 'read') and (hasattr(source, 'name') or fmt):
        return _read_pod(source, fmt)
    else:
        message = 'source must be a path or an open file handle; got {}'
        raise ValueError(message.format(type(source)))


def write_pod(pod, dest, fmt=None):
    if isinstance(dest, str):
        with open(dest, 'w') as wfh:
            return _write_pod(pod, wfh, fmt)
    elif hasattr(dest, 'write') and (hasattr(dest, 'name') or fmt):
        return _write_pod(pod, dest, fmt)
    else:
        message = 'dest must be a path or an open file handle; got {}'
        raise ValueError(message.format(type(dest)))


def dump(o, wfh, fmt='json', *args, **kwargs):
    serializer = {'yaml': yaml,
                  'json': json,
                  'python': python,
                  'py': python,
                  }.get(fmt)
    if serializer is None:
        raise ValueError('Unknown serialization format: "{}"'.format(fmt))
    serializer.dump(o, wfh, *args, **kwargs)


def load(s, fmt='json', *args, **kwargs):
    return read_pod(s, fmt=fmt)


def _read_pod(fh, fmt=None):
    if fmt is None:
        fmt = os.path.splitext(fh.name)[1].lower().strip('.')
        if fmt == '':
            # Special case of no given file extension
            message = ("Could not determine format "
                       "from file extension for \"{}\". "
                       "Please specify it or modify the fmt parameter.")
            raise ValueError(message.format(getattr(fh, 'name', '<none>')))
    if fmt == 'yaml':
        return yaml.load(fh)
    elif fmt == 'json':
        return json.load(fh)
    elif fmt == 'py':
        return python.load(fh)
    else:
        raise ValueError('Unknown format "{}": {}'.format(fmt, getattr(fh, 'name', '<none>')))


def _write_pod(pod, wfh, fmt=None):
    if fmt is None:
        fmt = os.path.splitext(wfh.name)[1].lower().strip('.')
    if fmt == 'yaml':
        return yaml.dump(pod, wfh)
    elif fmt == 'json':
        return json.dump(pod, wfh)
    elif fmt == 'py':
        raise ValueError('Serializing to Python is not supported')
    else:
        raise ValueError('Unknown format "{}": {}'.format(fmt, getattr(wfh, 'name', '<none>')))


def is_pod(obj):
    if type(obj) not in POD_TYPES:  # pylint: disable=unidiomatic-typecheck
        return False
    if hasattr(obj, 'items'):
        for k, v in obj.items():
            if not (is_pod(k) and is_pod(v)):
                return False
    elif isiterable(obj):
        for v in obj:
            if not is_pod(v):
                return False
    return True


class Podable(object):

    _pod_serialization_version = 0

    @classmethod
    def from_pod(cls, pod):
        pod = cls._upgrade_pod(pod)
        instance = cls()
        instance._pod_version = pod.pop('_pod_version')  # pylint: disable=protected-access
        return instance

    @classmethod
    def _upgrade_pod(cls, pod):
        _pod_serialization_version = pod.pop('_pod_serialization_version', None) or 0
        while _pod_serialization_version < cls._pod_serialization_version:
            _pod_serialization_version += 1
            upgrade = getattr(cls, '_pod_upgrade_v{}'.format(_pod_serialization_version))
            pod = upgrade(pod)
        return pod

    def __init__(self):
        self._pod_version = self._pod_serialization_version

    def to_pod(self):
        pod = {}
        pod['_pod_version'] = self._pod_version
        pod['_pod_serialization_version'] = self._pod_serialization_version
        return pod
