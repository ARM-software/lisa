#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, ARM Limited and contributors.
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

import enum
import abc
import functools
import operator
import contextlib
import shutil

import exekall.utils as utils
import exekall.engine as engine
from exekall.tests.utils import indent


class TestResultStatus(enum.Enum):
    PASSED = 1
    "Used when the test is passing"

    FAILED = 2
    "Used when the test failed, i.e. the feature is not behaving as expected."

    SKIPPED = 3
    "Used when a test is irrelevant."


class TestResult(Exception):
    """
    Result of a test with some context.
    """

    def __init__(self, status, msg=None, expr_list=None):
        self.status = status
        self.msg = msg
        self.expr_list = expr_list

    def __str__(self):
        return '{self.status.name}:{msg} ({ids})'.format(
            self=self,
            msg=' ' + self.msg if self.msg else '',
            ids=', '.join(expr.get_id(qual=False) for expr in self.expr_list)
        )

    @classmethod
    def fail_if(cls, cond, *args, **kwargs):
        if cond:
            raise cls(TestResultStatus.FAILED, *args, **kwargs)

    @classmethod
    def skip_if(cls, cond, *args, **kwargs):
        if cond:
            raise cls(TestResultStatus.SKIPPED, *args, **kwargs)


class Final:
    """
    Expressions built inside the test cases return instances of this class
    """
    pass


class TestCaseABC(abc.ABC):
    """
    Abstract Base Class for test cases.
    """
    @abc.abstractmethod
    def CALLABLES():
        """
        Set of callables to build expressions from
        """
        pass

    GOAL_TYPE = Final
    "Goal used by default when building expressions"

    NON_REUSABLE_TYPES = None
    "Set of types that are not considered reusable"

    @staticmethod
    def get_tags(obj):
        """
        Returns the dict of tags for a given value
        """
        return dict()

    @classmethod
    def make_expressions(cls, callable_set, goal_type, tags_getter=None, non_reusable_type_set=None):
        """
        Create a list of :class:`exekall.engine.Expression` out of
        the given ``callable_set``.
        """

        op_set = {
            engine.Operator(
                callable_,
                tags_getter=tags_getter,
                non_reusable_type_set=non_reusable_type_set,
            )
            for callable_ in callable_set
        }

        root_op_set = {
            op
            for op in op_set
            if issubclass(op.value_type, goal_type)
        }

        class_ctx = engine.ClassContext.from_op_set(op_set)

        expr_list = class_ctx.build_expr_list(
            root_op_set,
            non_produced_handler='raise',
            cycle_handler='raise',
        )

        expr_list.sort(key=lambda expr: expr.get_id(full_qual=True, with_tags=True))

        return expr_list

    def __init__(self, expr_data: engine.ExprData):
        self.expr_list = self.make_expressions(
            set(self.CALLABLES),
            goal_type=self.GOAL_TYPE,
            tags_getter=self.get_tags,
            non_reusable_type_set=self.NON_REUSABLE_TYPES,
        )

        self.artifact_dir = expr_data['artifact_dir']

        print('Tested expressions:')
        for expr in self.expr_list:
            print(indent(expr.format_structure()))

        self.dump_expr_layout()

    def dump_expr_layout(self):
        folder = self.artifact_dir / 'tested_expr'
        # Wipe if already exists
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(str(folder))
        folder.mkdir()

        for expr in self.expr_list:
            id_ = expr.get_id(qual=False)
            with open(str(folder / '{}.dot'.format(id_)), 'w') as f:
                f.write(expr.format_structure(graphviz=True))

    def get_computable_expr_list(self):
        return engine.ComputableExpression.from_expr_list(self.expr_list)

    @staticmethod
    def test(f):
        """
        Decorator to be used to mark a test method.

        This adds a return annotation to make it available to exekall, and will
        return any raised :class:`TestResult`. If the test does not raise
        :class:`TestResult`, it is assume the test is passing.

        It will catch any :class:`AssertionError` raised, and turn return
        a failed :class:`TestResult`.

        .. note:: The decorated method is expected to raise a
            :class:`TestResult`.
        """
        # Annotate with TestResult so it is picked up by exekall
        f.__annotations__['return'] = TestResult

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                f(self, *args, **kwargs)
            except TestResult as e:
                result = e
            except AssertionError as e:
                result = TestResult(
                    TestResultStatus.FAILED,
                    'failed assertion: {}\n{}'.format(
                        e,
                        indent(utils.format_exception(e))
                    ),
                    []
                )
            else:
                result = TestResult(TestResultStatus.PASSED)

            if result.expr_list is None:
                result.expr_list = self.expr_list

            return result

        return wrapper

    @staticmethod
    def check_excep(expr_val_list):
        """
        Check if any exception happened when computing
        :class:`exekall.engine.ExprVal` in the given list.

        If an exception is detected, a :class:`TestResult` is raised with
        :attr:`TestResultStatus.FAILED` status.
        """
        failed_expr_val_list = [
            failed_expr_val
            for expr_val in expr_val_list
            for failed_expr_val in expr_val.get_excep()
        ]

        if failed_expr_val_list:
            def format_error(expr_val):
                return '{}:\n{}'.format(
                    expr_val.get_id(qual=False),
                    indent(utils.format_exception(expr_val.excep))
                )
            raise TestResult(
                TestResultStatus.FAILED,
                'Exceptions during expression execution:\n{}\n'.format(
                    '\n\n'.join(
                        indent(format_error(failed_expr_val))
                        for failed_expr_val in failed_expr_val_list
                    )
                ),
                [expr_val.expr for expr_val in expr_val_list]
            )

    def execute(self, check_excep=True):
        """
        Execute the expressions of that test case.

        :return: An iterator yielding tuples with shape:
            (exekall.engine.ComputableExpression, list(exekall.engine.ExprVal))

        :param check_excep: If true, :meth:`check_excep` will be called on
            computed :class:`exekall.engine.ExprVal`.
        :type check_excep: bool
        """
        for computable_expr in self.get_computable_expr_list():
            expr_val_list = list(computable_expr.execute())
            if check_excep:
                self.check_excep(expr_val_list)
            yield computable_expr, expr_val_list


class TestCaseBase(TestCaseABC):
    """
    Base class for test cases.
    """

    # key: get_id() kwargs dict as a tuple to be passed to dict()
    # val: ID
    EXPR_ID = {}
    EXPR_VAL_ID = {}

    @TestCaseABC.test
    def test_expr_id(self):
        """
        Test that expressions have expected IDs.
        """
        def check_id(expr, id_map):
            for get_id_args, expected_id in id_map.items():
                id_ = expr.get_id(**dict(get_id_args))
                TestResult.fail_if(
                    id_ != expected_id,
                    'Wrong ID: expected {} but got {}'.format(
                        expected_id, id_,
                    ),
                    [expr]
                )

        TestResult.skip_if(not (self.EXPR_VAL_ID or self.EXPR_ID), 'no reference ID specified')

        for computable_expr, expr_val_list in self.execute(check_excep=False):
            for expr_val in expr_val_list:
                check_id(expr_val, self.EXPR_VAL_ID)

        if self.EXPR_ID:
            expr_list = self.get_computable_expr_list() + self.expr_list
            for expr in expr_list:
                check_id(expr, self.EXPR_ID)


class NoExcepTestCase(TestCaseBase):
    """
    Base class for tests with expressions that are expected to not raise any
    exception.
    """

    @TestCaseABC.test
    def test_excep(self):
        """
        Make sure no exception is raised when executing the expression.
        """
        # Consume all the values and check for exceptions while doing that
        list(self.execute(check_excep=True))

    @TestCaseABC.test
    def test_result_type(self):
        """
        Test that the expressions return the right value type.
        """
        for computable_expr, expr_val_list in self.execute():
            for expr_val in expr_val_list:
                TestResult.fail_if(
                    not isinstance(expr_val.value, Final),
                    'Wrong value type: expected {} but got {}'.format(
                        utils.get_name(Final),
                        utils.get_name(type(expr_val.value))
                    ),
                    [computable_expr]
                )

    @TestCaseABC.test
    def test_reexecute(self):
        """
        Test that the expressions gives the same number of values when executed
        multiple times.
        """
        ref_list = [
            expr_val_list
            for computable_expr, expr_val_list in self.execute()
        ]

        new_list = [
            expr_val_list
            for computable_expr, expr_val_list in self.execute()
        ]

        TestResult.fail_if(
            len(new_list) != len(ref_list),
            'Different number of expressions when re-executing'
        )

        def compare_expr_val(expr_val1, expr_val2):
            def is_comparable(expr_val):
                # object.__eq__ is useless in that context, as it behaves like
                # "is"
                return type(expr_val.value).__eq__ is not object.__eq__

            if is_comparable(expr_val1) and is_comparable(expr_val2):
                TestResult.fail_if(
                    expr_val1.value != expr_val2.value,
                    'Non-equal values for {} ({}) and {} ({})'.format(
                        expr_val1.get_id(qual=False),
                        expr_val1.value,
                        expr_val2.get_id(qual=False),
                        expr_val2.value,
                    )
                )

            params1 = sorted(expr_val1.keys())
            params2 = sorted(expr_val2.keys())

            TestResult.fail_if(
                params1 != params2,
                'Different parameters for {} ({}) and {} ({})'.format(
                    expr_val1.get_id(qual=False),
                    params1,
                    expr_val2.get_id(qual=False),
                    params2,
                )
            )

            for param in params1:
                compare_expr_val(expr_val1[param], expr_val2[param])

        for ref_list, new_list in zip(ref_list, new_list):
            TestResult.fail_if(
                len(new_list) != len(ref_list),
                'Different number of values when re-executing'
            )
            for ref, new in zip(ref_list, new_list):
                compare_expr_val(ref, new)

    VALUES_RELATIONS = []
    """
    Relations to be satisfied between values inside an expressions.

    In the shape of ``list(tuple(description, path1, relation, path2))``, where
    relation is a function taking 2 objects as parameters and returning a
    boolean result.  Paths are the parameter path to locate the values of
    subexpressions.
    """

    @TestCaseABC.test
    def test_relations(self):
        """
        Test that some relations are satisfied between values of subexpressions.

        For example, the same instance of reusable types is used when executing
        the expression, and that non-reusable types are instantiated multiple
        times.
        """
        TestResult.skip_if(not self.VALUES_RELATIONS, 'no relations specified')

        def get_val(expr_val, path):
            if path:
                return get_val(expr_val[path[0]], path[1:])
            else:
                return expr_val.value

        for computable_expr, expr_val_list in self.execute():
            for expr_val in expr_val_list:
                for description, path1, relation, path2 in self.VALUES_RELATIONS:
                    TestResult.fail_if(
                        not relation(
                            get_val(expr_val, path1),
                            get_val(expr_val, path2),
                        ),
                        'relations "{}" between {} and {} not satisfied'.format(
                            description,
                            '->'.join(path1),
                            '->'.join(path2),
                        )
                    )


class A:
    pass


class B:
    pass


class B2:
    pass


class B3:
    pass


def init() -> A:
    return A()


def middle(a: A) -> B:
    assert type(a) is A
    return B()


def middle2(a: A) -> B2:
    assert type(a) is A
    return B2()


def middle3(b2: B2) -> B3:
    assert type(b2) is B2
    return B3()


def final(b: B, b2: B2, b3: B3) -> Final:
    assert type(b) is B
    assert type(b2) is B2
    assert type(b3) is B3
    return Final()


class SingleExprTestCase(NoExcepTestCase):
    CALLABLES = {init, middle, middle2, middle3, final}
    NON_REUSABLE_TYPES = {B2}

    # key: get_id() kwargs dict as a tuple to be passed to dict()
    # val: ID
    EXPR_VAL_ID = {
        (('qual', True),): 'exekall.tests.suite.init:exekall.tests.suite.middle[tag1=val1][tag2=val2]:exekall.tests.suite.final(b2=exekall.tests.suite.init:exekall.tests.suite.middle2,b3=exekall.tests.suite.init:exekall.tests.suite.middle2:exekall.tests.suite.middle3)',
        (('qual', False),): 'init:middle[tag1=val1][tag2=val2]:final(b2=init:middle2,b3=init:middle2:middle3)',
    }
    EXPR_ID = {
        (('qual', True),): 'exekall.tests.suite.init:exekall.tests.suite.middle:exekall.tests.suite.final(b2=exekall.tests.suite.init:exekall.tests.suite.middle2,b3=exekall.tests.suite.init:exekall.tests.suite.middle2:exekall.tests.suite.middle3)',
        (('qual', False),): 'init:middle:final(b2=init:middle2,b3=init:middle2:middle3)',
    }

    VALUES_RELATIONS = [
        ('reusable type', ['b', 'a'], operator.is_, ['b2', 'a']),
        ('non reusable type', ['b2'], operator.is_not, ['b3', 'b2']),
    ]

    @staticmethod
    def get_tags(obj):
        if isinstance(obj, B):
            return {
                # non-sorted order to make sure tags are sorted in the ID
                'tag2': 'val2',
                'tag1': 'val1',
            }
        else:
            return {}

    @TestCaseABC.test
    def test_single_expr_val(self):
        """
        Test that only one :class:`exekall.engine.ExprVal` is computed for each
        expression.
        """
        for computable_expr, expr_val_list in self.execute():
            TestResult.fail_if(len(expr_val_list) != 1, "too many values", [computable_expr])


class Bderived(B):
    pass


def middle_derived(a: A) -> Bderived:
    assert type(a) is A
    return Bderived()


def final_derived(b: B) -> Final:
    assert type(b) is Bderived
    assert isinstance(b, B)
    return Final()


class InheritanceTestCase(NoExcepTestCase):
    CALLABLES = {init, middle_derived, final_derived}
    EXPR_ID = {
        (('qual', True),): 'exekall.tests.suite.init:exekall.tests.suite.middle_derived:exekall.tests.suite.final_derived',
        (('qual', False),): 'init:middle_derived:final_derived',
    }
    # no tags used
    EXPR_VAL_ID = EXPR_ID


def init_consumer(consumer: engine.Consumer) -> A:
    assert callable(consumer)
    return A()


class ConsumerTestCase(NoExcepTestCase):
    CALLABLES = {init_consumer, middle, middle2, middle3, final}

    VALUES_RELATIONS = [
        # Consumer acts like any other parameter for a reusable type.
        ('consumer ref allows reuse 1', ['b', 'a'], operator.is_not, ['b2', 'a']),
        ('consumer ref allows reuse 2', ['b3', 'b2', 'a'], operator.is_, ['b2', 'a']),
    ]

    EXPR_ID = {
        (('qual', True),): 'exekall.tests.suite.init_consumer:exekall.tests.suite.middle:exekall.tests.suite.final(b2=exekall.tests.suite.init_consumer:exekall.tests.suite.middle2,b3=exekall.tests.suite.init_consumer:exekall.tests.suite.middle2:exekall.tests.suite.middle3)',
        (('qual', False),): 'init_consumer:middle:final(b2=init_consumer:middle2,b3=init_consumer:middle2:middle3)',
    }
    # no tags used
    EXPR_VAL_ID = EXPR_ID


def middle_excep(a: A) -> B:
    raise ValueError('value error')


def final_excep(b: B, b2: B2) -> Final:
    raise RuntimeError('This should never trigger, since middle_excep is supposed to fail')


class ExceptionTestCase(TestCaseBase):
    CALLABLES = {init, middle_excep, middle2, final_excep}

    EXPR_ID = {
        (('qual', True),): 'exekall.tests.suite.init:exekall.tests.suite.middle_excep:exekall.tests.suite.final_excep(b2=exekall.tests.suite.init:exekall.tests.suite.middle2)',
        (('qual', False),): 'init:middle_excep:final_excep(b2=init:middle2)',
    }
    # no tags used
    EXPR_VAL_ID = EXPR_ID

    @TestCaseABC.test
    def test_excep(self):
        for computable_expr, expr_val_list in self.execute(check_excep=False):
            for expr_val in expr_val_list:
                failed_parent_list = expr_val.get_excep()
                TestResult.fail_if(
                    not failed_parent_list,
                    'No exception detected',
                    [computable_expr]
                )

                for failed_froz_val in failed_parent_list:
                    excep = failed_froz_val.excep
                    TestResult.fail_if(
                        not isinstance(excep, ValueError),
                        'Wrong exception type: expected {} but got {}'.format(
                            utils.get_name(ValueError),
                            repr(excep),
                        ),
                        [computable_expr]
                    )


class CloneTestCase(NoExcepTestCase):
    CALLABLES = {init, middle, middle2, middle3, final}

    VALUES_RELATIONS = [
        ('independent clones', ['b2'], operator.is_not, ['b3', 'b2']),
        ('non cloned 1', ['b2', 'a'], operator.is_, ['b', 'a']),
        ('non cloned 2', ['b3', 'b2', 'a'], operator.is_, ['b', 'a']),
    ]

    EXPR_ID = {
        (('qual', True),): 'exekall.tests.suite.init:exekall.tests.suite.middle:exekall.tests.suite.final(b2=exekall.tests.suite.init:exekall.tests.suite.middle2,b3=exekall.tests.suite.init:exekall.tests.suite.middle2:exekall.tests.suite.middle3)',
        (('qual', False),): 'init:middle:final(b2=init:middle2,b3=init:middle2:middle3)',
    }
    # no tags used
    EXPR_VAL_ID = EXPR_ID

    def get_computable_expr_list(self):
        # Make sure we clone after the last CSE is applied, so cloning is not
        # de-cloned by CSE
        def predicate(expr): return expr.op.callable_ in {final, middle2, middle3}
        return [
            expr.clone_by_predicate(predicate)
            for expr in super().get_computable_expr_list()
        ]
