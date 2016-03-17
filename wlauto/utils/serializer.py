"""
This module contains wrappers for Python serialization modules for
common formats that make it easier to serialize/deserialize WA
Plain Old Data structures (serilizable WA classes implement
``to_pod()``/``from_pod()`` methods for converting between POD
structures and Python class instances).

The modifications to standard serilization procedures are:

    - mappings are deserialized as ``OrderedDict``\ 's are than standard
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
from datetime import datetime

import yaml as _yaml
import dateutil.parser

from wlauto.exceptions import SerializerSyntaxError
from wlauto.utils.types import regex_type
from wlauto.utils.misc import isiterable


__all__ = [
    'json',
    'yaml',
    'read_pod',
    'dump',
    'load',
]


class WAJSONEncoder(_json.JSONEncoder):

    def default(self, obj):  # pylint: disable=method-hidden
        if hasattr(obj, 'to_pod'):
            return obj.to_pod()
        elif isinstance(obj, regex_type):
            return 'REGEX:{}:{}'.format(obj.flags, obj.pattern)
        elif isinstance(obj, datetime):
            return 'DATET:{}'.format(obj.isoformat())
        else:
            return _json.JSONEncoder.default(self, obj)


class WAJSONDecoder(_json.JSONDecoder):

    def decode(self, s, **kwargs):
        d = _json.JSONDecoder.decode(self, s, **kwargs)

        def try_parse_object(v):
            if isinstance(v, basestring) and v.startswith('REGEX:'):
                _, flags, pattern = v.split(':', 2)
                return re.compile(pattern, int(flags or 0))
            elif isinstance(v, basestring) and v.startswith('DATET:'):
                _, pattern = v.split(':', 1)
                return dateutil.parser.parse(pattern)
            else:
                return v

        def load_objects(d):
            pairs = []
            for k, v in d.iteritems():
                if hasattr(v, 'iteritems'):
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
    def load(fh, *args, **kwargs):
        try:
            return _json.load(fh, cls=WAJSONDecoder, object_pairs_hook=OrderedDict, *args, **kwargs)
        except ValueError as e:
            raise SerializerSyntaxError(e.message)

    @staticmethod
    def loads(s, *args, **kwargs):
        try:
            return _json.loads(s, cls=WAJSONDecoder, object_pairs_hook=OrderedDict, *args, **kwargs)
        except ValueError as e:
            raise SerializerSyntaxError(e.message)


_mapping_tag = _yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
_regex_tag = u'tag:wa:regex'


def _wa_dict_representer(dumper, data):
    return dumper.represent_mapping(_mapping_tag, data.iteritems())


def _wa_regex_representer(dumper, data):
    text = '{}:{}'.format(data.flags, data.pattern)
    return dumper.represent_scalar(_regex_tag, text)


def _wa_dict_constructor(loader, node):
    pairs = loader.construct_pairs(node)
    seen_keys = set()
    for k, _ in pairs:
        if k in seen_keys:
            raise ValueError('Duplicate entry: {}'.format(k))
        seen_keys.add(k)
    return OrderedDict(pairs)


def _wa_regex_constructor(loader, node):
    value = loader.construct_scalar(node)
    flags, pattern = value.split(':', 1)
    return re.compile(pattern, int(flags or 0))


_yaml.add_representer(OrderedDict, _wa_dict_representer)
_yaml.add_representer(regex_type, _wa_regex_representer)
_yaml.add_constructor(_mapping_tag, _wa_dict_constructor)
_yaml.add_constructor(_regex_tag, _wa_regex_constructor)


class yaml(object):

    @staticmethod
    def dump(o, wfh, *args, **kwargs):
        return _yaml.dump(o, wfh, *args, **kwargs)

    @staticmethod
    def load(fh, *args, **kwargs):
        try:
            return _yaml.load(fh, *args, **kwargs)
        except _yaml.YAMLError as e:
            lineno = None
            if hasattr(e, 'problem_mark'):
                lineno = e.problem_mark.line  # pylint: disable=no-member
            raise SerializerSyntaxError(e.message, lineno)

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
            exec s in pod  # pylint: disable=exec-used
        except SyntaxError as e:
            raise SerializerSyntaxError(e.message, e.lineno)
        for k in pod.keys():
            if k.startswith('__'):
                del pod[k]
        return pod


def read_pod(source, fmt=None):
    if isinstance(source, basestring):
        with open(source) as fh:
            return _read_pod(fh, fmt)
    elif hasattr(source, 'read') and (hasattr(source, 'name') or fmt):
        return _read_pod(source, fmt)
    else:
        message = 'source must be a path or an open file handle; got {}'
        raise ValueError(message.format(type(source)))


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
    if fmt == 'yaml':
        return yaml.load(fh)
    elif fmt == 'json':
        return json.load(fh)
    elif fmt == 'py':
        return python.load(fh)
    else:
        raise ValueError('Unknown format "{}": {}'.format(fmt, getattr(fh, 'name', '<none>')))
