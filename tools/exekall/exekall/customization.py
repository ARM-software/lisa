#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
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

import numbers

from exekall.engine import ValueDB
from exekall.utils import out, get_name, NoValue, get_subclasses

class AdaptorBase:
    """
    Base class of all adaptors.

    :param args: Command line argument namespace as returned by
        :meth:`argparse.ArgumentParser.parse_args`. Depending on the subcommand
        used, it could be the arguments of ``exekall run`` or ``exekall
        compare`` (or any other subcommand).
    :type args: argparse.Namespace

    An adaptor is a class providing a number of hooks to customize the
    behaviour of exekall on a given codebase. It should be implemented in a
    module called ``exekall_customize`` in order to be found by ``exekall
    run``.
    """
    name = 'default'

    def __init__(self, args):
        self.args = args

    def get_non_reusable_type_set(self):
        """
        Return a set of non-reusable types.

        Defaults to an empty set.
        """
        return set()

    @staticmethod
    def get_tags(value):
        """
        Returns a dictionnary of tag names to their value.

        Default to empty set of tags, unless value is a :class:`numbers.Number`
        in which case the value of the number is used.
        """
        if isinstance(value, numbers.Number):
            tags = {'': value}
        else:
            tags = {}
        return tags

    def filter_op_set(self, op_set):
        """
        Returns a potentially filtered set of :class:`exekall.engine.Operator`.

        This allows removing some operators from the ones that will be used to
        build expressions. Defaults to filtering out operators without any
        parameter, since the "spark" values are usually introduced by the
        adaptor directly, instead of calling functions from the code base.
        """
        return {
            op for op in op_set
            # Only select operators with non-empty parameter list. This
            # rules out all classes __init__ that do not take parameter, as
            # they are typically not interesting to us.
            if op.get_prototype()[0]
        }

    def format_expr_list(self, expr_list, verbose=0):
        """
        Return a string that is printed right after the list of executed
        expressions.

        :param expr_list: List of :class:`exekall.engine.ExpressionBase` that
            will be executed. Note that this list has not yet undergone CSE,
            cloning for multiple iterations or other transformations.
        :type expr_list: list(exekall.engine.ExpressionBase)

        This can be used to add some information about the expressions that are
        about to be executed.
        """
        return ''

    def get_prebuilt_op_set(self):
        """
        Returns a set of :class:`exekall.engine.PrebuiltOperator`.

        This allows injecting any "spark" value that is needed to build the
        expressions, like configuration objects. These values are usually built
        out of the custom CLI parameters added by the adaptor.
        """
        return set()

    def get_hidden_op_set(self, op_set):
        """
        Returns the set of hidden :class:`exekall.engine.Operator`.

        This allows hiding parts of the IDs that would not add much information
        but clutter them.
        """
        self.hidden_op_set = set()
        return self.hidden_op_set

    @staticmethod
    def register_run_param(parser):
        """
        Register CLI parameters for the ``run`` subcommand.

        :param parser: Parser of the ``run`` subcommand to add arguments onto.
        :type parser: argparse.ArgumentParser
        """
        pass

    @staticmethod
    def register_compare_param(parser):
        """
        Register CLI parameters for the ``compare`` subcommand.

        :param parser: Parser of the ``compare`` subcommand to add arguments onto.
        :type parser: argparse.ArgumentParser
        """
        pass

    def compare_db_list(self, db_list):
        """
        Compare databases listed in ``db_list``.

        :param db_list: List of :class:`exekall.engine.ValueDB` to compare.
        :type db_list: list(exekall.engine.ValueDB)

        This is called by ``exekall compare`` to actually do something useful.
        """
        pass

    @staticmethod
    def get_default_type_goal_pattern_set():
        """
        Returns a set of patterns that will be used as the default value for
        ``exekall run --goal``.
        """
        return {'*Result'}

    @classmethod
    def reload_db(cls, db, path=None):
        """
        Hook called when reloading a serialized
        :class:`exekall.engine.ValueDB`. The returned database will be used.

        :param db: :class:`exekall.engine.ValueDB` that has just been
            deserialized.
        :type db: exekall.engine.ValueDB

        :param path: Path of the file of the serialized database if available.
        :type path: str or None
        """
        return db

    def finalize_expr(self, expr):
        """
        Finalize an :class:`exekall.engine.ComputableExpression` right after
        all its values have been computed.
        """
        pass

    def format_result(self, expr_val):
        """
        Format an :class:`exekall.engine.ExprVal` that is the result of an
        expression. It should return a (short) string that will be displayed at
        the end of the computation.
        """
        val = expr_val.value
        if val is NoValue or val is None:
            for failed_parent in expr_val.get_excep():
                excep = failed_parent.excep
                return 'EXCEPTION ({type}): {msg}'.format(
                    type = get_name(type(excep), full_qual=False),
                    msg = excep
                )
            return 'No value computed'
        else:
            return str(val)

    def get_summary(self, result_map):
        """
        Return the summary of an ``exekall run`` session as a string.

        :param result_map: Dictionary of expressions to the list of their
            values.
        :type result_map: dict(exekall.engine.ComputableExpression,
            exekall.engine.ExprVal)
        """
        hidden_callable_set = {
            op.callable_
            for op in self.hidden_op_set
        }

        # Get all IDs and compute the maximum length to align the output
        result_id_map = {
            result: result.get_id(
                hidden_callable_set=hidden_callable_set,
                full_qual=False,
                qual=False,
            )
            for expr, result_list in result_map.items()
            for result in result_list
        }

        max_id_len = len(max(result_id_map.values(), key=len))

        summary = []
        for expr, result_list in result_map.items():
            for result in result_list:
                msg = self.format_result(result)
                msg = msg + '\n' if '\n' in msg else msg
                summary.append('{id:<{max_id_len}} {result}'.format(
                    id=result_id_map[result],
                    result=msg,
                    max_id_len=max_id_len,
                ))
        return '\n'.join(summary)

    @classmethod
    def get_adaptor_cls(cls, name=None):
        """
        Return the adaptor class that has the name ``name``.

        .. note:: This is not intended to be overriden by subclasses.
        """
        subcls_list = list(get_subclasses(cls) - {cls})
        if not name:
            if len(subcls_list) > 1:
                raise ValueError('An adaptor name must be specified if there is more than one adaptor to choose from')
            else:
                if len(subcls_list) > 0:
                    return subcls_list[0]
                else:
                    return cls

        for subcls in subcls_list:
            if subcls.name == name:
                return subcls
        return None

