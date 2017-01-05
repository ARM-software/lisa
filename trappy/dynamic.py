#    Copyright 2015-2017 ARM Limited
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


"""The idea is to create a wrapper class that
returns a Type of a Class dynamically created based
on the input parameters. Similar to a factory design
pattern
"""
from trappy.base import Base
import re
from trappy.ftrace import GenericFTrace


def default_init(self):
    """Default Constructor for the
    Dynamic MetaClass. This is used for
    the dynamic object creation in
    :mod:`trappy.dynamic.DynamicTypeFactory`
    """

    kwords = {}

    try:
        kwords["parse_raw"] = self.parse_raw
    except AttributeError:
        pass

    super(type(self), self).__init__(**kwords)


class DynamicTypeFactory(type):

    """Override the type class to create
    a dynamic type on the fly. This Factory
    class is used internally by
    :mod:`trappy.dynamic.register_dynamic_ftrace`
    """

    def __new__(mcs, name, bases, dct):
        """Override the new method"""
        return type.__new__(mcs, name, bases, dct)

    def __init__(cls, name, bases, dct):
        """Override the constructor"""
        super(DynamicTypeFactory, cls).__init__(name, bases, dct)


def _get_name(name):
    """Internal Method to Change camelcase to
    underscores. CamelCase -> camel_case
    """
    return re.sub('(?!^)([A-Z]+)', r'_\1', name).lower()


def register_dynamic_ftrace(class_name, unique_word, scope="all",
                            parse_raw=False, pivot=None):
    """Create a Dynamic FTrace parser and register it with any FTrace parsing classes

    :param class_name: The name of the class to be registered
        (Should be in CamelCase)
    :type class_name: str

    :param unique_word: The unique_word to be matched in the
        trace
    :type unique_word: str

    :param scope: Registry Scope (Can be used to constrain
        the parsing of events and group them together)
    :type scope: str

    :param parse_raw: If, true, raw trace output (-R flag)
        will be used
    :type parse_raw: bool

    :param pivot: The data column about which the data can be grouped
    :type pivot: str

    For example if a new unique word :code:`my_unique_word` has
    to be registered with TRAPpy:
    ::

        import trappy
        custom_class = trappy.register_dynamic_ftrace("MyEvent", "my_unique_word")
        trace = trappy.FTrace("/path/to/trace_file")

        # New data member created in the ftrace object
        trace.my_event

    .. note:: The name of the member is :code:`my_event` from **MyEvent**


    :return: A class object of type :mod:`trappy.base.Base`
    """

    kwords = {
            "__init__": default_init,
            "unique_word": unique_word,
            "name": _get_name(class_name),
            "parse_raw" : parse_raw,
        }

    if pivot:
        kwords["pivot"] = pivot

    dyn_class = DynamicTypeFactory(class_name, (Base,), kwords)
    GenericFTrace.register_parser(dyn_class, scope)
    return dyn_class


def register_ftrace_parser(cls, scope="all"):
    """Register a new FTrace parser class implementation

    Should be used when the class has complex helper methods and does
    not expect to use the default constructor.

    :param cls: The class to be registered for
        enabling the parsing of an event in trace
    :type cls: :mod:`trappy.base.Base`

    :param scope: scope of this parser class.  The scope can be used
        to restrict the parsing done on an individual file.  Currently
        the only scopes available are "sched", "thermal" or "all"
    :type scope: string

    """

    # Check the argspec of the class
    GenericFTrace.register_parser(cls, scope)

def unregister_ftrace_parser(ftrace_parser):
    """Unregister an ftrace parser

    :param ftrace_parser: An ftrace parser class that was registered
        with register_ftrace_parser() or register_dynamic_ftrace().
        If done with the latter, the cls parameter is the return value
        of register_dynamic_ftrace()
    :type ftrace_parser: class derived from :mod:`trappy.base.Base`

    """
    GenericFTrace.unregister_parser(ftrace_parser)
