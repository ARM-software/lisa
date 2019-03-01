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
    name = 'default'

    def __init__(self, args=None):
        if args is None:
            args = dict()
        self.args = args

    def get_non_reusable_type_set(self):
        return set()

    @staticmethod
    def get_tags(value):
        if isinstance(value, numbers.Number):
            tags = {'': value}
        else:
            tags = {}
        return tags

    def update_expr_data(self, expr_data):
        return

    def filter_op_set(self, op_set):
        return {
            op for op in op_set
            # Only select operators with non-empty parameter list. This
            # rules out all classes __init__ that do not take parameter, as
            # they are typically not interesting to us.
            if op.get_prototype()[0]
        }

    def format_expr_list(self, expr_list, verbose=0):
        return ''

    def get_prebuilt_set(self):
        return set()

    def get_hidden_op_set(self, op_set):
        self.hidden_op_set = set()
        return self.hidden_op_set

    @staticmethod
    def register_run_param(parser):
        pass

    @staticmethod
    def register_compare_param(parser):
        pass

    def compare_db_list(self, db_list):
        pass

    @staticmethod
    def get_default_type_goal_pattern_set():
        return {'*Result'}

    def resolve_cls_name(self, goal):
        return utils.get_class_from_name(goal)

    @classmethod
    def reload_db(cls, db, path=None):
        return db

    def finalize_expr(self, expr):
        pass

    def result_str(self, result):
        val = result.value
        if val is NoValue or val is None:
            for failed_parent in result.get_excep():
                excep = failed_parent.excep
                return 'EXCEPTION ({type}): {msg}'.format(
                    type = get_name(type(excep), full_qual=False),
                    msg = excep
                )
            return 'No value computed'
        else:
            return str(val)

    def get_summary(self, result_map):
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
                msg = self.result_str(result)
                msg = msg + '\n' if '\n' in msg else msg
                summary.append('{id:<{max_id_len}} {result}'.format(
                    id=result_id_map[result],
                    result=msg,
                    max_id_len=max_id_len,
                ))
        return '\n'.join(summary)

    @classmethod
    def get_adaptor_cls(cls, name=None):
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

