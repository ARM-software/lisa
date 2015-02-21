# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        dynamic.py
# ----------------------------------------------------------------
# $
#

"""The idea is to create a wrapper class that
returns a Type of a Class dynamically created based
on the input parameters. Similar to a factory design
pattern
"""
from cr2.base import Base
import re
from cr2.run import Run


def default_init(self):
    """Default Constructor for the
       Dynamic MetaClass
    """

    super(type(self), self).__init__(
        unique_word=self.unique_word,
    )


class DynamicTypeFactory(type):

    """Override the type class to create
       a dynamic type on the fly
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


def register_dynamic(class_name, unique_word, scope="all"):
    """Create a Dynamic Type and register
       it with the cr2 Framework"""

    dyn_class = DynamicTypeFactory(
        class_name, (Base,), {
            "__init__": default_init,
            "unique_word": unique_word,
            "name": _get_name(class_name)
        }
    )
    Run.register_class(dyn_class, scope)
    return dyn_class


def register_class(cls):
    """Register a new class implementation
       Should be used when the class has
       complex helper methods and does not
       expect to use the default constructor
    """

    # Check the argspec of the class
    Run.register_class(cls)
