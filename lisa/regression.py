# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

import argparse
import re
from collections import OrderedDict, namedtuple

from lisa.utils import groupby, memoized
from lisa.tests.base import Result, ResultBundle

import scipy.stats

ResultCount = namedtuple('ResultCount', ('passed', 'failed'))

class RegressionResult:
    """
    Compute failure-rate regression between old and new series.

    The regression is checked using Fisher's exact test.

    :param testcase_id: ID of the testcase, used for pretty-printing
    :type testcase_id: str

    :param old_count: number of times the test passed and failed in the old series
    :type old_count: ResultCount

    :param new_count: number of times the test passed and failed in the new series
    :type new_count: ResultCount

    :param alpha: Alpha risk when carrying the statistical test
    :type alpha: float


    """
    def __init__(self, testcase_id,
            old_count, new_count,
            alpha=None,
        ):
        self.old_count = old_count
        self.new_count = new_count

        self.testcase_id = testcase_id
        self.alpha = alpha if alpha is not None else 0.05

    @classmethod
    def from_result_list(cls, testcase_id, old_list, new_list, alpha=None):
        """
        Build a :class:`RegressionResult` from two list of
        :class:`lisa.tests.base.Result`, or objects that can be
        converted to `bool`.

        .. note:: Only ``FAILED`` and ``PASSED`` results are taken into account,
            other results are ignored.

        :param testcase_id: ID of the testcase
        :type testcase_id: str

        :param old_list: old series
        :type old_list: list(lisa.tests.base.Result)

        :param new_list: new series
        :type new_list: list(lisa.tests.base.Result)

        :param alpha: Alpha risk of the statistical test
        :type alpha: float
        """
        def coerce_to_bool(x, res):
            if isinstance(x, ResultBundle):
                return x.result is res
            # handle other types as well, as long as they can be
            # converted to bool
            else:
                if res is Result.FAILED:
                    return not bool(x)
                elif res is Result.PASSED:
                    return bool(x)

        def count(seq, res):
            return sum(
                coerce_to_bool(x, res)
                for x in seq
            )

        # Ignore errors and skipped tests
        old_count = ResultCount(
            failed=count(old_list, Result.FAILED),
            passed=count(old_list, Result.PASSED),
        )

        new_count = ResultCount(
            failed=count(new_list, Result.FAILED),
            passed=count(new_list, Result.PASSED),
        )

        return cls(
            testcase_id=testcase_id,
            old_count=old_count,
            new_count=new_count,
            alpha=alpha,
        )

    @property
    def sample_size(self):
        """
        Tuple of sample sizes for old and new series.

        """
        return (
            (self.old_count.passed + self.old_count.failed),
            (self.new_count.passed + self.new_count.failed),
        )

    @property
    def failure_pc(self):
        """
        Tuple of failure rate in percent for old an new series.
        """
        def div(x, y):
            try:
                return x/y
            except ZeroDivisionError:
                return float('Inf')
        failure_new_pc = 100 * div(self.new_count.failed, (self.new_count.failed + self.new_count.passed))
        _pc = 100 * div(self.old_count.failed, (self.old_count.failed + self.old_count.passed))

        return (_pc, failure_new_pc)

    @property
    def failure_delta_pc(self):
        """
        Delta between old and new failure rate in percent.
        """
        _pc, failure_new_pc = self.failure_pc
        return failure_new_pc - _pc

    @property
    def significant(self):
        """
        True if there is a significant difference in failure rate, False
        otherwise.
        """
        return self.p_val <= self.alpha

    @property
    def p_val(self):
        """
        P-value of the statistical test.
        """
        return self.get_p_val()

    @memoized
    def get_p_val(self, alternative='two-sided'):
        """
        Compute the p-value of the statistical test, with the given alternative
        hypothesis.
        """
        # Apply the Fisher exact test to all tests failures.
        odds_ratio, p_val = scipy.stats.fisher_exact(
            [
                # Ignore errors and skipped tests
                [self.old_count.failed, self.old_count.passed],
                [self.new_count.failed, self.new_count.passed],
            ],
            alternative = alternative,
        )
        return p_val

def compute_regressions(old_list, new_list, remove_tags=[], **kwargs):
    """
    Compute a list of :class:`RegressionResult` out of two lists of
    :class:`exekall.engine.FrozenExprVal`.

    The tests are first grouped by their ID, and then a
    :class:`RegressionResult` is computed for each of these ID.

    :param old_list: old series of :class:`exekall.engine.FrozenExprVal`.  Values
        with a UUID that is also present in `new_list` will be removed from
        that list before the regressions are computed.
    :type old_list: list(exekall.engine.FrozenExprVal)

    :param new_list: new series of :class:`exekall.engine.FrozenExprVal`. Values
        with a UUID that is also present in `old_list` will be removed from
        that list before the regressions are computed.
    :type new_list: list(exekall.engine.FrozenExprVal)

    :param remove_tags: remove the given list of tags from the IDs before
        computing the regression. That allows computing regressions with a
        different "board" tag for example.
    :type remove_tags: list(str)

    :param kwargs: extra :meth:`RegressionResult.from_result_list` parameters
    """

    def dedup_list(froz_val_list, excluded_froz_val_list):
        excluded_uuids = {
            froz_val.uuid
            for froz_val in excluded_froz_val_list
        }
        return [
            froz_val
            for froz_val in froz_val_list
            if froz_val.uuid not in excluded_uuids
        ]

    # Remove from the new_list all the FrozenExprVal that were carried from the
    # old_list sequence. That is important since running "exekall run --load-db"
    # will contain both new and old data, so old data needs to be filtered out
    # before we can actually compare the two sets.
    _new_list = dedup_list(new_list, old_list)
    _old_list = dedup_list(old_list, new_list)
    old_list = _old_list
    new_list = _new_list

    def get_id(froz_val):
        id_ = froz_val.get_id(qual=False, with_tags=True)

        # Remove tags, so that more test will share the same ID. This allows
        # cross-board comparison for example.
        for tag in remove_tags:
            id_ = re.sub(r'\[{}=.*?\]'.format(tag), '', id_)

        return id_

    def group_by_testcase(froz_val_list):
        return OrderedDict(
            (testcase_id, [froz_val.value for froz_val in froz_val_group])
            for testcase_id, froz_val_group in groupby(froz_val_list, key=get_id)
        )

    old_testcases = group_by_testcase(old_list)
    new_testcases = group_by_testcase(new_list)

    return [
        RegressionResult.from_result_list(
            testcase_id=testcase_id,
            old_list=old_testcases[testcase_id],
            new_list=new_testcases[testcase_id],
            **kwargs,
        )
        for testcase_id in sorted(old_testcases.keys() & new_testcases.keys())
    ]
