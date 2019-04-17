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

import argparse
import copy
import contextlib
import itertools
import re
import os.path
from pathlib import Path
from collections import OrderedDict, namedtuple

from lisa.target import Target, TargetConf
from lisa.trace import FtraceCollector, FtraceConf
from lisa.platforms.platinfo import PlatformInfo
from lisa.utils import HideExekallID, Loggable, ArtifactPath, get_subclasses, groupby, Serializable
from lisa.conf import MultiSrcConf
from lisa.tests.base import TestBundle, ResultBundle
from lisa.tests.scheduler.load_tracking import InvarianceItem
from lisa.regression import compute_regressions

from exekall.utils import get_name, get_method_class, add_argument
from exekall.engine import ExprData, Consumer, PrebuiltOperator
from exekall.customization import AdaptorBase

class NonReusable:
    pass

class ExekallArtifactPath(ArtifactPath, NonReusable):
    @classmethod
    def from_expr_data(cls, data:ExprData, consumer:Consumer) -> 'ExekallArtifactPath':
        """
        Factory used when running under `exekall`
        """
        artifact_dir = Path(data['expr_artifact_dir']).resolve()
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

class ExekallFtraceCollector(FtraceCollector, HideExekallID):
    @staticmethod
    def _get_consumer_conf(consumer):
        consumer_cls = get_method_class(consumer)
        conf = getattr(consumer_cls, 'ftrace_conf', FtraceConf())
        return conf

    @classmethod
    def from_user_conf(cls, target:Target, consumer:Consumer, user_conf:FtraceConf=None) -> 'ExekallFtraceCollector':
        base_conf = cls._get_consumer_conf(consumer)
        consumer_cls = get_method_class(consumer)
        merged_src = 'user+{}'.format(consumer_cls.__qualname__)

        return super().from_user_conf(
            target,
            base_conf, user_conf,
            merged_src=merged_src
        )

class LISAAdaptor(AdaptorBase):
    name = 'LISA'

    def get_non_reusable_type_set(self):
        return {NonReusable}

    def get_prebuilt_op_set(self):
        non_reusable_type_set = self.get_non_reusable_type_set()
        op_set = set()

        # Try to build as many configurations instances from all the files we
        # are given
        conf_cls_set = set(get_subclasses(MultiSrcConf, only_leaves=True))
        conf_list = []
        for conf_path in self.args.conf:
            for conf_cls in conf_cls_set:
                try:
                    # Do not add the default source, to avoid overriding user
                    # configuration with the default one.
                    conf = conf_cls.from_yaml_map(conf_path, add_default_src=False)
                except ValueError:
                    continue
                else:
                    conf_list.append((conf, conf_path))

        def keyfunc(conf_and_path):
            cls = type(conf_and_path[0])
            # We use the ID since classes are not comparable
            return id(cls), cls

        # Then aggregate all the conf from each type, so they just act as
        # alternative sources.
        for (_, conf_cls), conf_and_path_seq in groupby(conf_list, key=keyfunc):
            conf_and_path_list = list(conf_and_path_seq)

            # Get the default configuration, and stack all user-defined keys
            conf = conf_cls()
            for conf_src, conf_path in conf_and_path_list:
                src = os.path.basename(conf_path)
                conf.add_src(src, conf_src)

            op_set.add(PrebuiltOperator(
                conf_cls, [conf],
                non_reusable_type_set=non_reusable_type_set
            ))

        # Inject serialized objects as root operators
        for path in self.args.inject:
            obj = Serializable.from_path(path)
            op_set.add(PrebuiltOperator(type(obj), [obj],
                non_reusable_type_set=non_reusable_type_set
            ))

        # Inject a dummy empty TargetConf
        if self.args.inject_empty_target_conf:
            op_set.add(PrebuiltOperator(TargetConf, [TargetConf(conf={})],
                non_reusable_type_set=non_reusable_type_set
            ))

        return op_set

    def get_hidden_op_set(self, op_set):
        hidden_op_set = {
            op for op in op_set
            if issubclass(op.value_type, HideExekallID)
        }
        self.hidden_op_set = hidden_op_set
        return hidden_op_set

    def format_expr_list(self, expr_list, verbose=0):
        def get_callable_events(callable_):
            """
            Recursively unwraps all layers of wrappers, collecting the events
            at each stage. That is needed in order to cope with things like
            :class:`exekall.engine.UnboundMethod`.
            """
            try:
                used_events = callable_.used_events
            except AttributeError:
                events = set()
            else:
                events = set(used_events.get_all_events())

            with contextlib.suppress(AttributeError):
                events.update(get_callable_events(callable_.__wrapped__))

            return events

        def get_trace_events(expr):
            events = get_callable_events(expr.op.callable_)
            for param_expr in expr.param_map.values():
                events.update(get_trace_events(param_expr))
            return events

        events = set()
        for expr in expr_list:
            events.update(get_trace_events(expr))

        if events:
            joiner = '\n  - '
            events_str = joiner + joiner.join(sorted(events))
        else:
            events_str = ' <no events>'
        return 'Used trace events:{}'.format(events_str)

    @staticmethod
    def register_run_param(parser):
        add_argument(parser, '--conf', action='append',
            default=[],
            help="LISA configuration file. If multiple configurations of a given type are found, they are merged (last one can override keys in previous ones)")

        add_argument(parser, '--inject', action='append',
            metavar='SERIALIZED_OBJECT_PATH',
            default=[],
            help="Serialized object to inject when building expressions")

        # Create an empty TargetConf, so we are able to get the list of tests
        # as if we were going to execute them using a target.
        # note: that is only used for generating the documentation.
        add_argument(parser, '--inject-empty-target-conf', action='store_true',
            help=argparse.SUPPRESS)

    @staticmethod
    def register_compare_param(parser):
        add_argument(parser, '--alpha', type=float,
            default=5,
            help="""Alpha risk for Fisher exact test in percents.""")

        add_argument(parser, '--non-significant', action='store_true',
            help="""Also show non-significant changes of failure rate.""")

        add_argument(parser, '--remove-tag', action='append',
            default=[],
            help="""Remove the given tags in the testcase IDs before
comparison. Can be repeated.""")

    def compare_db_list(self, db_list):
        alpha = self.args.alpha / 100
        show_non_significant = self.args.non_significant

        result_list_old, result_list_new = [
            db.get_roots()
            for db in db_list
        ]

        regr_list = compute_regressions(
            result_list_old,
            result_list_new,
            remove_tags=self.args.remove_tag,
            alpha=alpha,
        )

        if not regr_list:
            print('No matching test IDs have been found, use "--remove-tag board" to match across "board" tags')
            return

        print('testcase failure rate changes with alpha={}\n'.format(alpha))

        id_len = max(len(regr.testcase_id) for regr in regr_list)

        header = '{id:<{id_len}}   old%   new% delta%      pvalue{regr_column}'.format(
            id='testcase'.format(alpha),
            id_len=id_len,
            regr_column=' changed' if show_non_significant else ''
        )
        print(header + '\n' + '-' * len(header))
        for regr in regr_list:
            if regr.significant or show_non_significant:
                old_pc, new_pc = regr.failure_pc
                print('{id:<{id_len}} {old_pc:>5.1f}% {new_pc:>5.1f}% {delta_pc:>5.1f}%    {pval:.2e} {has_regr}'.format(
                    id=regr.testcase_id,
                    old_pc=old_pc,
                    new_pc=new_pc,
                    delta_pc=regr.failure_delta_pc,
                    pval=regr.p_val,
                    id_len=id_len,
                    has_regr='*' if regr.significant and show_non_significant else '',
                ))

    @staticmethod
    def get_default_type_goal_pattern_set():
        return {'*.ResultBundle'}

    @classmethod
    def reload_db(cls, db, path=None):
        # If path is not known, we cannot do anything here
        if not path:
            return db

        # This will relocate ArtifactPath instances to the new absolute path of
        # the results folder, in case it has been moved to another place
        artifact_dir = Path(path).parent.resolve()

        # Relocate ArtifactPath embeded in objects so they will always
        # contain an absolute path that adapts to the local filesystem
        for serial in db.get_all():
            val = serial.value
            try:
                dct = val.__dict__
            except AttributeError:
                continue
            for attr, attr_val in dct.items():
                if isinstance(attr_val, ArtifactPath):
                    new_path = attr_val.with_root(artifact_dir)
                    # Only update paths to existing files, otherwise assume it
                    # was pointing outside the artifact_dir and therefore
                    # should not be fixed up
                    if os.path.exists(new_path):
                        setattr(val, attr, new_path)

        return db

    def finalize_expr(self, expr):
        expr_artifact_dir = expr.data['expr_artifact_dir']
        artifact_dir = expr.data['artifact_dir']
        for expr_val in expr.get_all_vals():
            self._finalize_expr_val(expr_val, artifact_dir, expr_artifact_dir)

    def _finalize_expr_val(self, expr_val, artifact_dir, expr_artifact_dir):
        val = expr_val.value

        def needs_rewriting(val):
            # Only rewrite ArtifactPath path values
            if not isinstance(val, ArtifactPath):
                return False
            # And only if they are a subfolder of artifact_dir. Otherwise, they
            # are something pointing outside of the artifact area, which we
            # cannot handle.
            return artifact_dir.resolve() in Path(val).resolve().parents

        # Add symlinks to artifact folders for ExprValue that were used in the
        # ExprValue graph, but were initially computed for another Expression
        if needs_rewriting(val):
            val = Path(val)
            is_subfolder = (expr_artifact_dir.resolve() in val.resolve().parents)
            # The folder is reachable from our ExprValue, but is not a
            # subfolder of the expr_artifact_dir, so we want to get a
            # symlink to it
            if not is_subfolder:
                # We get the name of the callable
                callable_folder = val.parts[-2]
                folder = expr_artifact_dir/callable_folder

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

        for param, param_expr_val in expr_val.param_map.items():
            self._finalize_expr_val(param_expr_val, artifact_dir, expr_artifact_dir)

    @classmethod
    def get_tags(cls, value):
        tags = {}
        if isinstance(value, Target):
            tags['board'] = value.name
        elif isinstance(value, InvarianceItem):
            if value.cpu is not None:
                tags['cpu'] = '{}@{}'.format(value.cpu, value.freq)
        elif isinstance(value, TestBundle):
            tags['board'] = value.plat_info.get('name')
        else:
            tags = super().get_tags(value)

        tags = {k: v for k, v in tags.items() if v is not None}

        return tags
