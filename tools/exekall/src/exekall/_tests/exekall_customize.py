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


from exekall.utils import flatten_seq
from exekall.customization import AdaptorBase
from exekall._tests.suite import TestCaseABC, TestResult, TestResultStatus


class SelfTestAdaptor(AdaptorBase):
    name = 'exekall-self-tests'

    def filter_op_set(self, op_set):
        filtered_op_set = super().filter_op_set(op_set)

        # Make sure we keep the ones for TestCase
        filtered_op_set.update(
            op for op in op_set
            if issubclass(op.value_type, TestCaseABC)
        )
        return filtered_op_set

    @staticmethod
    def get_default_type_goal_pattern_set():
        """
        Returns a set of patterns that will be used as the default value for
        ``exekall run --goal``.
        """
        return {'*.TestResult'}

    def get_run_exit_code(self, result_map):
        expr_val_list = flatten_seq(
            expr_val_list
            for expr, expr_val_list in result_map.items()
        )

        for expr_val in expr_val_list:
            # An exception happened
            if expr_val.get_excep():
                return 20

            val = expr_val.value
            if isinstance(val, TestResult):
                if val.status is TestResultStatus.FAILED:
                    return 10
        return 0
