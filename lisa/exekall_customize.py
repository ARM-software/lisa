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
import contextlib
import itertools
import re
import os.path
from operator import attrgetter
from pathlib import Path

from exekall.utils import get_name, add_argument, NoValue, flatten_seq
from exekall.engine import ExprData, Consumer, PrebuiltOperator
from exekall.customization import AdaptorBase

from lisa.target import Target, TargetConf
from lisa.utils import HideExekallID, ArtifactPath, Serializable, get_nested_key, ExekallTaggable
from lisa.conf import MultiSrcConf
from lisa.tests.base import TestBundle, ResultBundleBase, Result, CannotCreateError
from lisa.regression import compute_regressions


class NonReusable:
    pass


class ExekallArtifactPath(ArtifactPath, NonReusable):
    @classmethod
    def from_expr_data(cls, data: ExprData, consumer: Consumer) -> 'ExekallArtifactPath':
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

        cls.get_logger().info(f'Creating {consumer_name} artifact storage: {artifact_dir}')
        artifact_dir.mkdir(parents=True)
        # Get canonical absolute paths
        artifact_dir = artifact_dir.resolve()
        root = data['artifact_dir']
        relative = artifact_dir.relative_to(root)
        return cls(root, relative)


class LISAAdaptor(AdaptorBase):
    name = 'LISA'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_op_set = None

    def get_non_reusable_type_set(self):
        return {NonReusable}

    def get_prebuilt_op_set(self):
        non_reusable_type_set = self.get_non_reusable_type_set()
        op_set = set()

        # Try to build as many configurations instances from all the files we
        # are given
        conf_map = MultiSrcConf.from_yaml_map_list(self.args.conf)
        for conf_cls, conf in conf_map.items():
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
        return f'Used trace events:{events_str}'

    @staticmethod
    def register_run_param(parser):
        add_argument(parser, '--conf', action='append',
            default=[],
            help="LISA configuration file. If multiple configurations of a given type are found, they are merged (last one can override keys in previous ones). Only load trusted files as it can lead to arbitrary code execution.")

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

        def get_roots(db):
            return {
                froz_val
                for froz_val in db.get_roots()
                # Filter-out NoValue so it does not get counted as a failure,
                # since bool(NoValue) is False
                if froz_val.value is not NoValue
            }

        result_list_old, result_list_new = [
            get_roots(db)
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

        print(f'testcase failure rate changes with alpha={alpha}\n')

        id_len = max(len(regr.testcase_id) for regr in regr_list)

        header = '{id:<{id_len}}   old%   new%  delta%       pvalue fix_iter# {regr_column}'.format(
            id='testcase',
            id_len=id_len,
            regr_column=' significant' if show_non_significant else ''
        )
        print(header + '\n' + '-' * len(header))
        for regr in regr_list:
            if regr.significant or show_non_significant:
                old_pc, new_pc = regr.failure_pc
                # Only show the number of iterations required to validate a fix
                # when there was a regression.
                if regr.failure_delta_pc > 0:
                    validation_nr = regr.fix_validation_min_iter_nr
                else:
                    validation_nr = ''

                print('{id:<{id_len}} {old_pc:>5.1f}% {new_pc:>5.1f}% {delta_pc:>6.1f}%    {pval:>9.2e} {validation_nr:>9} {significant}'.format(
                    id=regr.testcase_id,
                    old_pc=old_pc,
                    new_pc=new_pc,
                    delta_pc=regr.failure_delta_pc,
                    pval=regr.p_val,
                    id_len=id_len,
                    validation_nr=validation_nr,
                    significant='*' if regr.significant and show_non_significant else '',
                ))

    @staticmethod
    def _parse_uuid_attr(s):
        uuid_attr = s.split('.', 1)
        try:
            uuid, attr = uuid_attr
        except ValueError:
            uuid = s
            attr = None

        return uuid, attr

    @classmethod
    def register_show_param(cls, parser):
        uuid_attr_metavar = 'UUID[.ATTRIBUTE]'

        add_argument(parser, '--show', action='append', default=[],
            type=cls._parse_uuid_attr,
            metavar=uuid_attr_metavar,
            help="""Show the attribute value with given UUID, or one of its attribute.""")

        add_argument(parser, '--show-yaml', action='append', default=[],
            type=cls._parse_uuid_attr,
            metavar=uuid_attr_metavar,
            help="""Show the YAML dump of value with given UUID, or one of its attributes.""")

        add_argument(parser, '--serialize', nargs=2, action='append', default=[],
            metavar=(uuid_attr_metavar, 'PATH'),
            help="""Serialize the value of given UUID to PATH.""")

    def show_db(self, db):
        parse_uuid_attr = self._parse_uuid_attr

        def indent(s):
            idt = ' ' * 4
            return idt + s.replace('\n', '\n' + idt)

        def get_uuid(uuid):
            try:
                froz_val = db.get_by_uuid(uuid)
            except KeyError as e:
                raise KeyError(f'UUID={uuid} not found in the database') from e
            else:
                return froz_val

        def get_obj(froz_val):
            val = froz_val.value
            excep = froz_val.excep
            if val is NoValue and excep is not NoValue:
                return excep
            else:
                return val

        def get_attr_key(obj, attr_key):
            # parse "attr[key1][key2][...]"
            attr = attr_key.split('[', 1)[0]
            keys = re.findall(r'\[(.*?)\]', attr_key)
            if attr:
                obj = getattr(obj, attr)
            return get_nested_key(obj, keys)

        def resolve_attr(obj, attr_key):
            if attr_key is None:
                return obj

            try:
                attr_key, remainder = attr_key.split('.', 1)
            except ValueError:
                return get_attr_key(obj, attr_key)
            else:
                obj = get_attr_key(obj, attr_key)
                return resolve_attr(obj, remainder)

        args = self.args
        if not (args.show or args.show_yaml):
            super().show_db(db)

        attr_map = {}
        for uuid, attr in args.show:
            attr_map.setdefault(uuid, set()).add(attr)

        if len(args.show) == 1:
            show_format = '{val}'
        else:
            show_format = 'UUID={uuid} {type}{attr}{eq}{val}'

        serialize_spec_list = args.serialize
        yaml_show_spec_list = args.show_yaml

        for uuid, attr_set in attr_map.items():
            attr_list = sorted(attr_set)
            froz_val = get_uuid(uuid)
            obj = get_obj(froz_val)
            for attr in attr_list:
                attr_value = resolve_attr(obj, attr)

                attr_str = str(attr_value)
                if '\n' in attr_str:
                    attr_str = '\n' + indent(attr_str)
                    eq = ':'
                else:
                    eq = '='

                print(show_format.format(
                    uuid=froz_val.uuid,
                    type=get_name(type(obj)),
                    attr='.' + attr if attr else '',
                    val=attr_str,
                    eq=eq,
                ))

        if len(yaml_show_spec_list) == 1:
            yaml_show_format = '{yaml}'
            def yaml_indent(x):
                return x
        else:
            yaml_show_format = 'UUID={uuid} {type}:\n\n{yaml}'
            yaml_indent = indent

        for uuid, attr in yaml_show_spec_list:
            froz_val = get_uuid(uuid)
            obj = get_obj(froz_val)
            value = resolve_attr(obj, attr)

            if isinstance(value, Serializable):
                yaml_str = value.to_yaml()
            else:
                yaml_str = Serializable._to_yaml(value)

            print(yaml_show_format.format(
                uuid=uuid,
                type=get_name(type(value)),
                yaml=yaml_indent(yaml_str),
            ))

        for uuid_attr, path in serialize_spec_list:
            uuid, attr = parse_uuid_attr(uuid_attr)
            froz_val = get_uuid(uuid)
            obj = get_obj(froz_val)
            value = resolve_attr(obj, attr)

            if isinstance(value, Serializable):
                value.to_path(path)
            else:
                Serializable._to_path(value, path, fmt='yaml')

        return 0

    @staticmethod
    def get_default_type_goal_pattern_set():
        return {'*.ResultBundleBase'}

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
                folder = expr_artifact_dir / callable_folder

                # We build a relative path back in the hierarchy to the root of
                # all artifacts
                relative_artifact_dir = Path(os.path.relpath(str(artifact_dir), start=str(folder)))

                # The target needs to be a relative symlink, so we replace the
                # absolute artifact_dir by a relative version of it
                target = relative_artifact_dir / val.relative_to(artifact_dir)

                with contextlib.suppress(FileExistsError):
                    folder.mkdir(parents=True)

                for i in itertools.count(1):
                    symlink = Path(folder, str(i))
                    if not symlink.exists():
                        break

                symlink.symlink_to(target, target_is_directory=True)

        for param_expr_val in expr_val.param_map.values():
            self._finalize_expr_val(param_expr_val, artifact_dir, expr_artifact_dir)

    @classmethod
    def get_tags(cls, value):
        if isinstance(value, ExekallTaggable):
            tags = value.get_tags()
        else:
            tags = super().get_tags(value)

        tags = {k: v for k, v in tags.items() if v is not None}

        return tags

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
            if isinstance(val, ResultBundleBase):
                if val.result is Result.FAILED:
                    return 10
        return 0

    def format_result(self, expr_val):
        val = expr_val.value
        if val is NoValue or val is None:
            skip_exceps = [
                excep
                for excep in map(
                    attrgetter('excep'),
                    expr_val.get_excep()
                )
                if isinstance(excep, CannotCreateError)
            ]
            if skip_exceps:
                sep = '\n    * ' if len(skip_exceps) > 1 else ' '
                msg = sep.join(map(str, skip_exceps))
                return f'SKIPPED:{sep}{msg}'
            else:
                return super().format_result(expr_val)
        else:
            if isinstance(val, ResultBundleBase):
                return val.pretty_format()
            else:
                return str(val)
