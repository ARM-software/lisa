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

import contextlib
import itertools
import inspect
import functools
import logging
import sys
import os.path
from pathlib import Path
import xml.etree.ElementTree as ET
import traceback

from lisa.env import TestEnv, TargetConf
from lisa.platforms.platinfo import PlatformInfo
from lisa.utils import HideExekallID, Loggable, ArtifactPath
from lisa.tests.kernel.test_bundle import TestBundle, Result, ResultBundle, CannotCreateError
from lisa.tests.kernel.scheduler.load_tracking import FreqInvarianceItem

from exekall.utils import info, get_name, get_mro
from exekall.engine import ExprData, Consumer, PrebuiltOperator, NoValue, StorageDB
from exekall.customization import AdaptorBase

class ExekallArtifactPath(ArtifactPath):
    @classmethod
    def from_expr_data(cls, data:ExprData, consumer:Consumer) -> 'ExekallArtifactPath':
        """
        Factory used when running under `exekall`
        """
        artifact_dir = Path(data['testcase_artifact_dir']).resolve()
        consumer_name = get_name(consumer)

        # Find a non-used directory
        for i in itertools.count(1):
            artifact_dir_ = Path(artifact_dir, consumer_name, str(i))
            if not artifact_dir_.exists():
                artifact_dir = artifact_dir_
                break

        cls.get_logger().info('Creating {consumer} artifact storage: {path}'.format(
            consumer = consumer_name,
            path = artifact_dir
        ))
        artifact_dir.mkdir(parents=True)
        # Get canonical absolute paths
        artifact_dir = artifact_dir.resolve()
        root = data['artifact_dir']
        relative = artifact_dir.relative_to(root)
        return cls(root, relative)

class LISAAdaptor(AdaptorBase):
    name = 'LISA'

    def get_non_reusable_type_set(self):
        return {
            ExekallArtifactPath,
        }

    def get_prebuilt_list(self):
        non_reusable_type_set = self.get_non_reusable_type_set()
        op_list = []
        if self.args.target_conf:
            op_list.append(
                PrebuiltOperator(TargetConf, [
                    TargetConf.from_yaml_map(self.args.target_conf)
                ],
                non_reusable_type_set=non_reusable_type_set
            ))

        if self.args.platform_info:
            op_list.append(
                PrebuiltOperator(PlatformInfo, [
                    PlatformInfo.from_yaml_map(self.args.platform_info)
                ],
                non_reusable_type_set=non_reusable_type_set
            ))

        return op_list

    def get_hidden_callable_set(self, op_map):
        hidden_callable_set = set()
        for produced, op_set in op_map.items():
            if issubclass(produced, HideExekallID):
                hidden_callable_set.update(op.callable_ for op in op_set)

        self.hidden_callable_set = hidden_callable_set
        return hidden_callable_set

    @staticmethod
    def register_cli_param(parser):
        parser.add_argument('--target-conf',
            help="Target config file")

        parser.add_argument('--platform-info',
            help="Platform info file")

    @staticmethod
    def get_default_type_goal_pattern_set():
        return {'*.ResultBundle'}

    @classmethod
    def load_db(cls, db_path, *args, **kwargs):
        # This will relocate ArtifactPath instances to the new absolute path of
        # the results folder, in case it has been moved to another place
        artifact_dir = Path(db_path).parent.resolve()
        db = StorageDB.from_path(db_path, *args, **kwargs)

        # Relocate ArtifactPath embeded in objects so they will always
        # contain an absolute path that adapts to the local filesystem
        for serial in db.obj_store.get_all():
            val = serial.value
            try:
                dct = val.__dict__
            except AttributeError:
                continue
            for attr, attr_val in dct.items():
                if isinstance(attr_val, ArtifactPath):
                    setattr(val, attr,
                        attr_val.with_root(artifact_dir)
                    )

        return db

    def finalize_expr(self, expr):
        testcase_artifact_dir = expr.data['testcase_artifact_dir']
        artifact_dir = expr.data['artifact_dir']
        for expr_val in expr.get_all_values():
            self._finalize_expr_val(expr_val, artifact_dir, testcase_artifact_dir)

    def _finalize_expr_val(self, expr_val, artifact_dir, testcase_artifact_dir):
        val = expr_val.value

        # Add symlinks to artifact folders for ExprValue that were used in the
        # ExprValue graph, but were initially computed for another Expression
        if isinstance(val, ArtifactPath):
            val = Path(val)
            is_subfolder = (testcase_artifact_dir.resolve() in val.resolve().parents)
            # The folder is reachable from our ExprValue, but is not a
            # subfolder of the testcase_artifact_dir, so we want to get a
            # symlink to it
            if not is_subfolder:
                # We get the name of the callable
                callable_folder = val.parts[-2]
                folder = testcase_artifact_dir/callable_folder

                # We build a relative path back in the hierarchy to the root of
                # all artifacts
                relative_artifact_dir = Path(os.path.relpath(str(artifact_dir), start=str(folder)))

                # The target needs to be a relative symlink, so we replace the
                # absolute artifact_dir by a relative version of it
                target = relative_artifact_dir/val.relative_to(artifact_dir)

                with contextlib.suppress(FileExistsError):
                    folder.mkdir(parents=True)

                for i in itertools.count(1):
                    symlink = Path(folder, str(i))
                    if not symlink.exists():
                        break

                symlink.symlink_to(target, target_is_directory=True)

        for param, param_expr_val in expr_val.param_value_map.items():
            self._finalize_expr_val(param_expr_val, artifact_dir, testcase_artifact_dir)

    @classmethod
    def get_tags(cls, value):
        tags = {}
        if isinstance(value, TestEnv):
            board_name = value.target_conf.get('name')
            if board_name:
                tags.append(board_name)
        elif isinstance(value, PlatformInfo):
            tags['board'] = value.get('name')
        elif isinstance(value, TestBundle):
            tags['board'] = value.plat_info.get('name')
            if isinstance(value, FreqInvarianceItem):
                if value.cpu is not None:
                    tags['cpu'] = '{}@{}'.format(value.cpu, value.freq)
        else:
            tags = super().get_tags(value)

        tags = {k: v for k, v in tags.items() if v is not None}

        return tags

    def process_results(self, result_map):
        super().process_results(result_map)

        # The goal is to implement something that is roughly compatible with:
        #  https://github.com/jenkinsci/xunit-plugin/blob/master/src/main/resources/org/jenkinsci/plugins/xunit/types/model/xsd/junit-10.xsd
        # This way, Jenkins should be able to read it, and other tools as well

        xunit_path = self.args.artifact_dir.joinpath('xunit.xml')
        et_root = self.create_xunit(result_map, self.hidden_callable_set)
        et_tree = ET.ElementTree(et_root)
        info('Writing xUnit file at: ' + str(xunit_path))
        et_tree.write(str(xunit_path))

    def create_xunit(self, result_map, hidden_callable_set):
        et_testsuites = ET.Element('testsuites')

        testcase_list = list(result_map.keys())
        # We group by module in which the root operators are defined. There will
        # be one testsuite for each such module.
        def key(expr):
            return expr.op.mod_name

        # One testsuite per module where a root operator is defined
        for mod_name, group in itertools.groupby(testcase_list, key=key):
            testcase_list = list(group)
            et_testsuite = ET.SubElement(et_testsuites, 'testsuite', attrib=dict(
                name = mod_name
            ))
            testsuite_counters = dict(failures=0, errors=0, tests=0, skipped=0)

            for testcase in testcase_list:
                artifact_path = testcase.data.get('artifact_dir', None)

                # If there is more than one value for a given expression, we
                # assume that they testcase will have unique names using tags
                expr_val_list = result_map[testcase]
                for expr_val in expr_val_list:
                    et_testcase = ET.SubElement(et_testsuite, 'testcase', dict(
                        name = expr_val.get_id(
                            full_qual=False,
                            qual=False,
                            with_tags=True,
                            hidden_callable_set=hidden_callable_set,
                        ),
                        # This may help locating the artifacts, even though it
                        # will only be valid on the machine it was produced on
                        artifact_path = str(artifact_path)
                    ))
                    testsuite_counters['tests'] += 1

                    for failed_expr_val in expr_val.get_failed_values():
                        excep = failed_expr_val.excep
                        # When one critical object cannot be created, we assume
                        # the test was skipped.
                        if isinstance(excep, CannotCreateError):
                            result = 'skipped'
                            testsuite_counters['skipped'] += 1
                        else:
                            result = 'error'
                            testsuite_counters['errors'] += 1

                        short_msg = str(excep)
                        msg = ''.join(traceback.format_exception(type(excep), excep, excep.__traceback__))
                        type_ = type(excep)

                        append_result_tag(et_testcase, result, type_, short_msg, msg)

                    value = expr_val.value
                    if isinstance(value, ResultBundle):
                        result = RESULT_TAG_MAP[value.result]
                        short_msg = value.result.lower_name
                        msg = str(value)
                        type_ = type(value)

                        append_result_tag(et_testcase, result, type_, short_msg, msg)
                        if value.result is Result.FAILED:
                            testsuite_counters['failures'] += 1

            et_testsuite.attrib.update(
                (k, str(v)) for k, v in testsuite_counters.items()
            )

        return et_testsuites


def append_result_tag(et_testcase, result, type_, short_msg, msg):
    et_result = ET.SubElement(et_testcase, result, dict(
        type=get_name(type_, full_qual=True),
        type_bases=','.join(
            get_name(type_, full_qual=True)
            for type_ in get_mro(type_)
        ),
        message=str(short_msg),
    ))
    et_result.text = str(msg)
    return et_result

RESULT_TAG_MAP = {
    # "passed" is an extension to xUnit format that we add for parsing
    # convenience
    Result.PASSED: 'passed',
    Result.FAILED: 'failure',
    # This tag is not part of xUnit format but necessary for our reporting
    Result.UNDECIDED: 'undecided'
}
# Make sure we cover all cases
assert set(RESULT_TAG_MAP.keys()) == set(Result)

