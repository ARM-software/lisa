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
This module contains additional casting and adaptation functions for several
different datatypes and metadata types for use with the psycopg2 module. The
casting functions will transform Postgresql data types into Python objects, and
the adapters the reverse. They are named this way according to the psycopg2
conventions.

For more information about the available adapters and casters in the standard
psycopg2 module, please see:

http://initd.org/psycopg/docs/extensions.html#sql-adaptation-protocol-objects

"""

import re
import os

try:
    from psycopg2 import InterfaceError
    from psycopg2.extensions import AsIs
except ImportError:
    InterfaceError = None
    AsIs = None

from wa.utils.types import level


POSTGRES_SCHEMA_DIR = os.path.join(os.path.dirname(__file__),
                                   '..',
                                   'commands',
                                   'postgres_schemas')


def cast_level(value, cur):  # pylint: disable=unused-argument
    """Generic Level caster for psycopg2"""
    if not InterfaceError:
        raise ImportError('There was a problem importing psycopg2.')
    if value is None:
        return None

    m = re.match(r"([^\()]*)\((\d*)\)", value)
    name = str(m.group(1))
    number = int(m.group(2))

    if m:
        return level(name, number)
    else:
        raise InterfaceError("Bad level representation: {}".format(value))


def cast_vanilla(value, cur):  # pylint: disable=unused-argument
    """Vanilla Type caster for psycopg2

    Simply returns the string representation.
    """
    if value is None:
        return None
    else:
        return str(value)


# List functions and classes for adapting

def adapt_level(a_level):
    """Generic Level Adapter for psycopg2"""
    return "{}({})".format(a_level.name, a_level.value)


class ListOfLevel(object):
    value = None

    def __init__(self, a_level):
        self.value = a_level

    def return_original(self):
        return self.value


def adapt_ListOfX(adapt_X):
    """This will create a multi-column adapter for a particular type.

    Note that the type must itself need to be in array form. Therefore
    this function serves to seaprate out individual lists into multiple
    big lists.
    E.g. if the X adapter produces array (a,b,c)
    then this adapter will take an list of Xs and produce a master array:
    ((a1,a2,a3),(b1,b2,b3),(c1,c2,c3))

    Takes as its argument the adapter for the type which must produce an
    SQL array string.
    Note that you should NOT put the AsIs in the adapt_X function.

    The need for this function arises from the fact that we may want to
    actually handle list-creating types differently if they themselves
    are in a list, as in the example above, we cannot simply adopt a
    recursive strategy.

    Note that master_list is the list representing the array. Each element
    in the list will represent a subarray (column). If there is only one
    subarray following processing then the outer {} are stripped to give a
    1 dimensional array.
    """
    def adapter_function(param):
        if not AsIs:
            raise ImportError('There was a problem importing psycopg2.')
        param = param.value
        result_list = []
        for element in param:  # Where param will be a list of X's
            result_list.append(adapt_X(element))
        test_element = result_list[0]
        num_items = len(test_element.split(","))
        master_list = []
        for x in range(num_items):
            master_list.append("")
        for element in result_list:
            element = element.strip("{").strip("}")
            element = element.split(",")
            for x in range(num_items):
                master_list[x] = master_list[x] + element[x] + ","
        if num_items > 1:
            master_sql_string = "{"
        else:
            master_sql_string = ""
        for x in range(num_items):
            # Remove trailing comma
            master_list[x] = master_list[x].strip(",")
            master_list[x] = "{" + master_list[x] + "}"
            master_sql_string = master_sql_string + master_list[x] + ","
        master_sql_string = master_sql_string.strip(",")
        if num_items > 1:
            master_sql_string = master_sql_string + "}"
        return AsIs("'{}'".format(master_sql_string))
    return adapter_function


def return_as_is(adapt_X):
    """Returns the AsIs appended function of the function passed

    This is useful for adapter functions intended to be used with the
    adapt_ListOfX function, which must return strings, as it allows them
    to be standalone adapters.
    """
    if not AsIs:
        raise ImportError('There was a problem importing psycopg2.')

    def adapter_function(param):
        return AsIs("'{}'".format(adapt_X(param)))
    return adapter_function


def adapt_vanilla(param):
    """Vanilla adapter: simply returns the string representation"""
    if not AsIs:
        raise ImportError('There was a problem importing psycopg2.')
    return AsIs("'{}'".format(param))


def create_iterable_adapter(array_columns, explicit_iterate=False):
    """Create an iterable adapter of a specified dimension

    If explicit_iterate is True, then it will be assumed that the param needs
    to be iterated upon via param.iteritems(). Otherwise it will simply be
    iterated vanilla.
    The value of array_columns will be equal to the number of indexed elements
    per item in the param iterable. E.g. a list of 3-element-long lists has
    3 elements per item in the iterable (the master list) and therefore
    array_columns should be equal to 3.
    If array_columns is 0, then this indicates that the iterable contains
    single items.
    """
    if not AsIs:
        raise ImportError('There was a problem importing psycopg2.')

    def adapt_iterable(param):
        """Adapts an iterable object into an SQL array"""
        final_string = ""  # String stores a string representation of the array
        if param:
            if array_columns > 1:
                for index in range(array_columns):
                    array_string = ""
                    for item in param.iteritems():
                        array_string = array_string + str(item[index]) + ","
                    array_string = array_string.strip(",")
                    array_string = "{" + array_string + "}"
                    final_string = final_string + array_string + ","
                final_string = final_string.strip(",")
            else:
                # Simply return each item in the array
                if explicit_iterate:
                    for item in param.iteritems():
                        final_string = final_string + str(item) + ","
                else:
                    for item in param:
                        final_string = final_string + str(item) + ","
        return AsIs("'{{{}}}'".format(final_string))
    return adapt_iterable


# For reference only and future use
def adapt_list(param):
    """Adapts a list into an array"""
    if not AsIs:
        raise ImportError('There was a problem importing psycopg2.')
    final_string = ""
    if param:
        for item in param:
            final_string = final_string + str(item) + ","
        final_string = "{" + final_string + "}"
    return AsIs("'{}'".format(final_string))


def get_schema(schemafilepath):
    with open(schemafilepath, 'r') as sqlfile:
        sql_commands = sqlfile.read()

    schema_major = None
    schema_minor = None
    # Extract schema version if present
    if sql_commands.startswith('--!VERSION'):
        splitcommands = sql_commands.split('!ENDVERSION!\n')
        schema_major, schema_minor = splitcommands[0].strip('--!VERSION!').split('.')
        schema_major = int(schema_major)
        schema_minor = int(schema_minor)
        sql_commands = splitcommands[1]
    return schema_major, schema_minor, sql_commands


def get_database_schema_version(conn):
    with conn.cursor() as cursor:
        cursor.execute('''SELECT
                              DatabaseMeta.schema_major,
                              DatabaseMeta.schema_minor
                          FROM
                              DatabaseMeta;''')
        schema_major, schema_minor = cursor.fetchone()
    return (schema_major, schema_minor)


def get_schema_versions(conn):
    schemafilepath = os.path.join(POSTGRES_SCHEMA_DIR, 'postgres_schema.sql')
    cur_major_version, cur_minor_version, _ = get_schema(schemafilepath)
    db_schema_version = get_database_schema_version(conn)
    return (cur_major_version, cur_minor_version), db_schema_version
