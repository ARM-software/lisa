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
# File:        grammar.py
# ----------------------------------------------------------------
# $
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


def parse_num(tokens):
    """parse a number"""
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
# Unary Operators
UNARY_OPS = oneOf("+ -")
# Multiplication/Division Operators
MULT_OPS = oneOf("* /")
# Addition/Subtraction Operators
SUM_OPS = oneOf("+ -")
# Relational Operators
REL_OPS = oneOf("> < >= <=")
# Logical Operators
LOGICAL_OPS = oneOf("&& || & |")

# Operator to function mapping
OPERATOR_MAP = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "||": lambda a, b: a or b,
    "&&": lambda a, b: a and b,
    "|": lambda a, b: a | b,
    "&": lambda a, b: a & b
}


def eval_unary_op(tokens):
    """Unary Op Evaluation"""

    params = tokens[0]
    if params[0] == "-":
        return -1 * params[1]
    else:
        return params[1]


def iterate_binary_ops(tokens):
    """An iterator for Binary Operation tokens"""

    itr = iter(tokens)
    while True:
        try:
            yield(itr.next(), itr.next())
        except StopIteration:
            break


def eval_binary_op(tokens):
    """Evaluate Binary operators"""

    params = tokens[0]
    result = params[0]

    for opr, val in iterate_binary_ops(params[1:]):
        result = OPERATOR_MAP[opr](result, val)

    return result


def str_to_attr(cls_str):
    """Bring the attr specified into current scope
       and return a handler
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
    data accesses and super-indexing data"""

    def __init__(self, data, pvars=None):
        if pvars is None:
            pvars = {}

        self.data = data
        self._pvars = pvars
        self._accessor = Group(
            FUNC_NAME + COLON + IDENTIFIER).setParseAction(self._pre_process)
        self._parse_expr = get_parse_expression(
            self._parse_func, self._parse_var_id)
        self._agg_df = pd.DataFrame()

    def solve(self, expr):
        """Solve expression

        Args:
            expr(string)
        """

        # Pre-process accessors for indexing
        self._accessor.searchString(expr)
        return self._parse_expr.parseString(expr)[0]

    def _pre_process(self, tokens):
        """Pre-process accessors for super-indexing"""

        params = tokens[0]
        if params[1] in self._agg_df.columns:
            return self._agg_df[params[1]]

        cls = str_to_attr(params[0])
        data_frame = getattr(self.data, cls.name).data_frame
        col_keys = [column for column in self._agg_df].append(params[1])
        self._agg_df = pd.concat(
            [self._agg_df, data_frame[params[1]]], keys=col_keys, axis=1)
        self._agg_df.fillna(method="pad", inplace=True)
        return self._agg_df[params[1]]

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

    def ref(self, mask):
        """Reference super indexed data with a boolean mask"""

        return self._agg_df[mask]
