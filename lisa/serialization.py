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
import copy
import pickle

from ruamel.yaml import YAML

class Serializable:
    """
    A helper class for YAML serialization/deserialization

    Not to be used on its own - instead, your class should inherit from this
    class to gain serialization superpowers.
    """
    serialized_whitelist = []
    serialized_blacklist = []
    serialized_placeholders = dict()

    DEFAULT_SERIALIZATION_FMT = 'yaml'
    "Default format used when serializing objects"

    _yaml = YAML(typ='unsafe')
    _yaml.allow_unicode = True
    _yaml.default_flow_style = False
    _yaml.indent = 4

    def to_path(self, filepath, fmt=None):
        """
        Serialize the object to a file

        :param filepath: The path of the file in which the object will be dumped
        :type filepath: str

        :param fmt: Serialization format.
        :type fmt: str
        """
        if fmt is None:
            fmt = self.DEFAULT_SERIALIZATION_FMT

        if fmt == 'yaml':
            kwargs = dict(mode='w', encoding='utf-8')
            dumper = self._yaml.dump
        elif fmt == 'pickle':
            kwargs = dict(mode='wb')
            dumper = pickle.dump
        else:
            raise ValueError('Unknown format "{}"'.format(fmt))

        with open(str(filepath), **kwargs) as fh:
            dumper(self, fh)

    @classmethod
    def from_path(cls, filepath, fmt=None):
        """
        Deserialize an object from a file

        :param filepath: The path of file in which the object has been dumped
        :type filepath: str

        :param fmt: Serialization format.
        :type fmt: str
        """
        if fmt is None:
            fmt = cls.DEFAULT_SERIALIZATION_FMT

        if fmt == 'yaml':
            kwargs = dict(mode='r', encoding='utf-8')
            loader = cls._yaml.load
        elif fmt == 'pickle':
            kwargs = dict(mode='rb')
            loader = pickle.load
        else:
            raise ValueError('Unknown format "{}"'.format(fmt))

        with open(str(filepath), **kwargs) as fh:
            return loader(fh)

    # def __getstate__(self):
    #     """
    #     Filter the instance's attributes upon serialization.

    #     The following class attributes can be used to customize the serialized
    #     content:
    #         * :attr:`serialized_whitelist`: list of attribute names to
    #           serialize. All other attributes will be ignored and will not be
    #           saved/restored.

    #         * :attr:`serialized_blacklist`: list of attribute names to not
    #           serialize.  All other attributes will be saved/restored.

    #         * serialized_placeholders: Map of attribute names to placeholder
    #           values. These attributes will not be serialized, and the
    #           placeholder value will be used upon restoration.

    #         If both :attr:`serialized_whitelist` and
    #         :attr:`serialized_blacklist` are specified,
    #         :attr:`serialized_blacklist` is ignored.
    #     """

    #     dct = copy.copy(self.__dict__)
    #     if self.serialized_whitelist:
    #         dct = {attr: dct[attr] for attr in self.serialized_whitelist}

    #     elif self.serialized_blacklist:
    #         for attr in self.serialized_blacklist:
    #             dct.pop(attr, None)

    #     for attr, placeholder in self.serialized_placeholders.items():
    #         dct.pop(attr, None)

    #     return dct

    # def __setstate__(self, dct):
    #     if self.serialized_placeholders:
    #         dct.update(copy.deepcopy(self.serialized_placeholders))
    #     self.__dict__ = dct

    # def __copy__(self):
    #     """Make sure that copying the class still works as usual, without
    #     dropping some attributes.
    #     """
    #     return super().__copy__(self)

    # def __deepcopy__(self):
    #     return super().__deepcopy__()

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
