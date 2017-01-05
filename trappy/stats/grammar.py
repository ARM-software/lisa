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

"""Grammar module allows the user to easily define relations
between data events and perform basic logical and arithmetic
operations on the data. The parser also handles super-indexing
and variable forwarding.
"""
from pyparsing import Literal, delimitedList, Optional, oneOf, nums,\
    alphas, alphanums, Forward, Word, opAssoc, operatorPrecedence, Combine, Group
import importlib
import pandas as pd
import types
import numpy as np
from trappy.stats.Topology import Topology
from trappy.stats import StatConf
from trappy.utils import handle_duplicate_index, listify


def parse_num(tokens):
    """Parser function for numerical data

    :param tokens: The grammar tokens
    :type tokens: list
    """
    return float(tokens[0])

# Suppressed Literals
LPAREN = Literal("(").suppress()
RPAREN = Literal(")").suppress()
COLON = Literal(":").suppress()
EXP_START = Literal("[").suppress()
EXP_END = Literal("]").suppress()

# Grammar Tokens

# DataFrame Accessor
INTEGER = Combine(Optional(oneOf("+ -")) + Word(nums))\
    .setParseAction(parse_num)
REAL = Combine(Optional(oneOf("+ -")) + Word(nums) + "." +
               Optional(Word(nums)) +
               Optional(oneOf("e E") + Optional(oneOf("+ -")) + Word(nums)))\
    .setParseAction(parse_num)

# Generic Identifier
IDENTIFIER = Word(alphas + '_', alphanums + '_')
# Python Like Function Name
FUNC_NAME = delimitedList(IDENTIFIER, delim=".", combine=True)
# Exponentiation operators
EXPONENTIATION_OPS = "**"
# Unary Operators
UNARY_OPS = oneOf("+ -")
# Multiplication/Division Operators
MULT_OPS = oneOf("* / // %")
# Addition/Subtraction Operators
SUM_OPS = oneOf("+ -")
# Relational Operators
REL_OPS = oneOf("> < >= <= == !=")
# Logical Operators
LOGICAL_OPS = oneOf("&& || & |")

# Operator to function mapping
OPERATOR_MAP = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "**": lambda a, b: a ** b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "||": lambda a, b: a or b,
    "&&": lambda a, b: a and b,
    "|": lambda a, b: a | b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "&": lambda a, b: a & b
}


def eval_unary_op(tokens):
    """Unary Op Evaluation

    :param tokens: The grammar tokens
    :type tokens: list
    """

    params = tokens[0]
    if params[0] == "-":
        return -1 * params[1]
    else:
        return params[1]


def iterate_binary_ops(tokens):
    """An iterator for Binary Operation tokens

    :param tokens: The grammar tokens
    :type tokens: list
    """

    itr = iter(tokens)
    while True:
        try:
            yield(itr.next(), itr.next())
        except StopIteration:
            break


def eval_binary_op(tokens):
    """Evaluate Binary operators

    :param tokens: The grammar tokens
    :type tokens: list
    """

    params = tokens[0]
    result = params[0]

    for opr, val in iterate_binary_ops(params[1:]):
        result = OPERATOR_MAP[opr](result, val)

    return result


def str_to_attr(cls_str):
    """Bring the attr specified into current scope
       and return a handler

    :param cls_str: A string representing the class
    :type cls_str: str

    :return: A class object
    """
    attr_name = cls_str.rsplit(".", 1)
    if len(attr_name) == 2:
        module_name, attr_name = attr_name
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    else:
        attr_name = attr_name[0]
        return globals()[attr_name]


def get_parse_expression(parse_func, parse_var_id):
    """return a parse expression with for the
    input parseActions
    """

    var_id = Group(
        FUNC_NAME + COLON + IDENTIFIER) | REAL | INTEGER | IDENTIFIER
    var_id.setParseAction(parse_var_id)

    # Forward declaration for an Arithmetic Expression
    arith_expr = Forward()
    func_call = Group(
        FUNC_NAME +
        LPAREN +
        Optional(
            Group(
                delimitedList(arith_expr))) +
        RPAREN)
    # An Arithmetic expression can have a var_id or
    # a function call as an operand
    # pylint: disable=expression-not-assigned
    arith_expr << operatorPrecedence(func_call | var_id,
                                     [
                                         (EXPONENTIATION_OPS, 2, opAssoc.LEFT,
                                          eval_binary_op),
                                         (UNARY_OPS, 1,
                                          opAssoc.RIGHT, eval_unary_op),
                                         (MULT_OPS, 2, opAssoc.LEFT,
                                          eval_binary_op),
                                         (SUM_OPS, 2, opAssoc.LEFT,
                                          eval_binary_op),
                                         (REL_OPS, 2, opAssoc.LEFT,
                                          eval_binary_op),
                                         (LOGICAL_OPS, 2,
                                          opAssoc.LEFT, eval_binary_op)
                                     ])

    # pylint: enable=expression-not-assigned
    # Argument expression for a function call
    # An argument to a function can be an
    # IDENTIFIER, Arithmetic expression, REAL number, INTEGER or a
    # Function call itself
    func_call.setParseAction(parse_func)
    return arith_expr


class Parser(object):

    """A parser class for solving simple
    data accesses and super-indexing data

    :param data: Trace Object
    :type data: instance of :mod:`trappy.ftrace.BareTrace` or a child
        class (like :mod:`trappy.ftrace.FTrace`)

    :param pvars: A dictionary of variables that need to be
        accessed from within the grammar
    :type pvars: dict

    :param method: The method to be used for reindexing data
        This can be one of the standas :mod:`pandas.DataFrame`
        methods (eg. pad, bfill, nearest). The default is pad
        or use the last valid observation.
    :type method: str

    :param limit: The number of indices a value will be propagated
        when reindexing. The default is None
    :type limit: int

    :param fill: Whether to fill the NaNs in the data.
        The default value is True.
    :type fill: bool

    :param window: A window of time in which to apply the data
        accesses.  By default the data accesses happen accross the
        whole trace.  With the window parameter you can limit it to a
        window of time inside the trace.  The first element of the
        tuple is the starting time and the second the ending time (set
        to None for end of trace).

    :type window: tuple

    :param filters: Restrict the parsing to the rows that match the
        specified criteria. For Example:
        ::

            filters =
                    {
                        "pid": 3338,
                        "cpu": [0, 2, 4],
                    }

        will only consider rows whose pid column is 3338 and cpu is
        either 0, 2 or 4.
    :type filters: dict

    - **Operators**

        +----------------+----------------------+---------------+
        | Operation      |      operator        | Associativity |
        +================+======================+===============+
        | Exponentiation | \*\*                 |    Left       |
        +----------------+----------------------+---------------+
        |Unary           | \-                   |    Right      |
        +----------------+----------------------+---------------+
        | Multiply/Divide| \*, /, //, %         |    Left       |
        +----------------+----------------------+---------------+
        | Add/Subtract   | +, \-,               |    Left       |
        +----------------+----------------------+---------------+
        | Comparison     | >, <, >=, <=, ==, != |    Left       |
        +----------------+----------------------+---------------+
        | Logical        | &&, ||, \|, &        |    Left       |
        +----------------+----------------------+---------------+

    - **Data Accessors**

        Since the goal of the grammar is to provide an
        easy language to access and compare data
        from a :mod:`trappy.trace.FTrace` object. The parser provides
        a simple notation to access this data.

        *Statically Defined Events*
        ::

            import trappy
            from trappy.stats.grammar import Parser

            trace = trappy.FTrace("path/to/trace/file")
            parser = Parser(trace)
            parser.solve("trappy.thermal.Thermal:temp * 2")

        *Aliasing*
        ::

            import trappy
            from trappy.stats.grammar import Parser

            pvars = {"THERMAL": trappy.thermal.Thermal}
            trace = trappy.FTrace("path/to/trace/file")
            parser = Parser(trace, pvars=pvars)
            parser.solve("THERMAL:temp * 2")

        *Using Event Name*
        ::

            import trappy
            from trappy.stats.grammar import Parser
            trace = trappy.FTrace("path/to/trace/file")
            parser = Parser(trace)
            parser.solve("thermal:temp * 2")

        The event :mod:`trappy.thermal.Thermal` is aliased
        as **thermal** in the grammar

        *Dynamic Events*
        ::

            import trappy
            from trappy.stats.grammar import Parser

            # Register Dynamic Event
            cls = trappy.register_dynamic_ftrace("my_unique_word", "event_name")

            pvars = {"CUSTOM": cls}
            trace = trappy.FTrace("path/to/trace/file")
            parser = Parser(trace, pvars=pvars)
            parser.solve("CUSTOM:col * 2")

        .. seealso:: :mod:`trappy.dynamic.register_dynamic_ftrace`

    """

    def __init__(self, data, pvars=None, window=(0, None), filters=None, **kwargs):
        if pvars is None:
            pvars = {}

        self.data = data
        self._pvars = pvars
        self._accessor = Group(
            FUNC_NAME + COLON + IDENTIFIER).setParseAction(self._pre_process)
        self._inspect = Group(
            FUNC_NAME + COLON + IDENTIFIER).setParseAction(self._parse_for_info)
        self._parse_expr = get_parse_expression(
            self._parse_func, self._parse_var_id)
        self._agg_df = pd.DataFrame()
        self._pivot_set = set()
        self._limit = kwargs.get("limit", StatConf.REINDEX_LIMIT_DEFAULT)
        self._method = kwargs.get("method", StatConf.REINDEX_METHOD_DEFAULT)
        self._fill = kwargs.get("fill", StatConf.NAN_FILL_DEFAULT)
        self._window = window
        self._filters = filters

    def solve(self, expr):
        """Parses and solves the input expression

        :param expr: The input expression
        :type expr: str

        :return: The return type may vary depending on
            the expression. For example:

            **Vector**
            ::

                import trappy
                from trappy.stats.grammar import Parser

                trace = trappy.FTrace("path/to/trace/file")
                parser = Parser(trace)
                parser.solve("trappy.thermal.Thermal:temp * 2")

            **Scalar**
            ::

                import trappy
                from trappy.stats.grammar import Parser

                trace = trappy.FTrace("path/to/trace/file")
                parser = Parser(trace)
                parser.solve("numpy.mean(trappy.thermal.Thermal:temp)")

            **Vector Mask**
            ::

                import trappy
                from trappy.stats.grammar import Parser

                trace = trappy.FTrace("path/to/trace/file")
                parser = Parser(trace)
                parser.solve("trappy.thermal.Thermal:temp > 65000")
        """

        # Pre-process accessors for indexing
        self._accessor.searchString(expr)
        return self._parse_expr.parseString(expr)[0]


        """

        # Pre-process accessors for indexing
        self._accessor.searchString(expr)
        return self._parse_expr.parseString(expr)[0]


        """

        # Pre-process accessors for indexing
        self._accessor.searchString(expr)
        return self._parse_expr.parseString(expr)[0]

    def _pivot(self, cls, column):
        """Pivot Data for concatenation"""

        data_frame = self._get_data_frame(cls)
        data_frame = handle_duplicate_index(data_frame)
        new_index = self._agg_df.index.union(data_frame.index)

        if hasattr(cls, "pivot") and cls.pivot:
            pivot = cls.pivot
            pivot_vals = list(np.unique(data_frame[pivot].values))
            data = {}


            for val in pivot_vals:
                data[val] = data_frame[data_frame[pivot] == val][[column]]
                if len(self._agg_df):
                    data[val] = data[val].reindex(
                        index=new_index,
                        method=self._method,
                        limit=self._limit)

            return pd.concat(data, axis=1).swaplevel(0, 1, axis=1)

        if len(self._agg_df):
            data_frame = data_frame.reindex(
                index=new_index,
                method=self._method,
                limit=self._limit)

        return pd.concat({StatConf.GRAMMAR_DEFAULT_PIVOT: data_frame[
                         [column]]}, axis=1).swaplevel(0, 1, axis=1)

    def _pre_process(self, tokens):
        """Pre-process accessors for super-indexing"""

        params = tokens[0]
        if params[1] in self._agg_df.columns:
            return self._agg_df[params[1]]

        cls = params[0]
        column = params[1]

        if cls in self._pvars:
            cls = self._pvars[cls]
        elif cls in self.data.class_definitions:
            cls = self.data.class_definitions[cls]
        else:
            cls = str_to_attr(cls)

        data_frame = self._pivot(cls, column)
        self._agg_df = pd.concat(
            [self._agg_df, data_frame], axis=1)

        if self._fill:
            self._agg_df = self._agg_df.fillna(method="pad")

        return self._agg_df[params[1]]

    def _parse_for_info(self, tokens):
        """Parse Action for inspecting data accessors"""

        params = tokens[0]
        cls = params[0]
        column = params[1]
        info = {}
        info["pivot"] = None
        info["pivot_values"] = None

        if cls in self._pvars:
            cls = self._pvars[cls]
        elif cls in self.data.class_definitions:
            cls = self.data.class_definitions[cls]
        else:
            cls = str_to_attr(cls)

        data_frame = self._get_data_frame(cls)

        info["class"] = cls
        info["length"] = len(data_frame)
        if hasattr(cls, "pivot") and cls.pivot:
            info["pivot"] = cls.pivot
            info["pivot_values"] = list(np.unique(data_frame[cls.pivot]))
        info["column"] = column
        info["column_present"] = column in data_frame.columns
        return info

    def _parse_var_id(self, tokens):
        """A function to parse a variable identifier
        """

        params = tokens[0]
        try:
            return float(params)
        except (ValueError, TypeError):
            try:
                return self._pvars[params]
            except KeyError:
                return self._agg_df[params[1]]

    def _parse_func(self, tokens):
        """A function to parse a function string"""

        params = tokens[0]
        func_name = params[0]
        if func_name in self._pvars and isinstance(
                self._pvars[func_name],
                types.FunctionType):
            func = self._pvars[func_name]
        else:
            func = str_to_attr(params[0])
        return func(*params[1])

    def _get_data_frame(self, cls):
        """Get the data frame from the BareTrace object, applying the window
        and the filters"""

        data_frame = getattr(self.data, cls.name).data_frame

        if self._window[1] is None:
            data_frame = data_frame.loc[self._window[0]:]
        else:
            data_frame = data_frame.loc[self._window[0]:self._window[1]]

        if self._filters:
            criterion = pd.Series([True] * len(data_frame),
                                  index=data_frame.index)

            for filter_col, wanted_vals in self._filters.iteritems():
                try:
                    dfr_col = data_frame[filter_col]
                except KeyError:
                    continue

                criterion &= dfr_col.isin(listify(wanted_vals))

            data_frame = data_frame[criterion]

        return data_frame

    def ref(self, mask):
        """Reference super indexed data with a boolean mask

        :param mask: A boolean :mod:`pandas.Series` that
            can be used to reference the aggregated data in
            the parser
        :type mask: :mod:`pandas.Series`

        :return: aggregated_data[mask]
        """

        return self._agg_df[mask]

    def inspect(self, accessor):
        """A function to inspect the accessor for information

        :param accessor: A data accessor of the format
            <event>:<column>
        :type accessor: str

        :return: A dictionary of information
        """
        return self._inspect.parseString(accessor)[0]
