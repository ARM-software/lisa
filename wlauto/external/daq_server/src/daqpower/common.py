#    Copyright 2014-2015 ARM Limited
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
import json


class Serializer(json.JSONEncoder):

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, Serializable):
            return o.serialize()
        if isinstance(o, EnumEntry):
            return o.name
        return json.JSONEncoder.default(self, o)


class Serializable(object):

    @classmethod
    def deserialize(cls, text):
        return cls(**json.loads(text))

    def serialize(self, d=None):
        if d is None:
            d = self.__dict__
        return json.dumps(d, cls=Serializer)


class DaqServerRequest(Serializable):

    def __init__(self, command, params=None):  # pylint: disable=W0231
        self.command = command
        self.params = params or {}


class DaqServerResponse(Serializable):

    def __init__(self, status, message=None, data=None):  # pylint: disable=W0231
        self.status = status
        self.message = message.strip().replace('\r\n', ' ') if message else ''
        self.data = data or {}

    def __str__(self):
        return '{} {}'.format(self.status, self.message or '')


class EnumEntry(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __cmp__(self, other):
        return cmp(self.name, str(other))


class Enum(object):
    """
    Assuming MyEnum = Enum('A', 'B'),

    MyEnum.A and MyEnum.B are valid values.

    a = MyEnum.A
    (a == MyEnum.A) == True
    (a in MyEnum) == True

    MyEnum('A') == MyEnum.A

    str(MyEnum.A) == 'A'

    """

    def __init__(self, *args):
        for a in args:
            setattr(self, a, EnumEntry(a))

    def __call__(self, value):
        if value not in self.__dict__:
            raise ValueError('Not enum value: {}'.format(value))
        return self.__dict__[value]

    def __iter__(self):
        for e in self.__dict__:
            yield self.__dict__[e]


Status = Enum('OK', 'OKISH', 'ERROR')
