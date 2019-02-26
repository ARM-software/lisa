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
import collections
import contextlib
import copy
import datetime
import hashlib
import inspect
import io
import itertools
import os
import pathlib
import random
import shutil
import sys

from exekall.customization import AdaptorBase
import exekall.utils as utils
from exekall.utils import NoValue, error, warn, debug, info, out
import exekall.engine as engine

DB_FILENAME = 'VALUE_DB.pickle.xz'

# Create an operator for all callables that have been detected in a given
# set of modules
def build_op_set(callable_pool, non_reusable_type_set, allowed_pattern_set, adaptor):
    op_set = {
        engine.Operator(
            callable_,
            non_reusable_type_set=non_reusable_type_set,
            tags_getter=adaptor.get_tags
        )
        for callable_ in callable_pool
    }

    filtered_op_set = adaptor.filter_op_set(op_set)
    # Make sure we have all the explicitely allowed operators
    filtered_op_set.update(
        op for op in op_set
        if utils.match_name(op.get_name(full_qual=True), allowed_pattern_set)
    )
    return filtered_op_set

def build_patch_map(sweep_spec_list, op_set):
    patch_map = dict()
    for sweep_spec in sweep_spec_list:
        number_type = float
        callable_pattern, param, start, stop, step = sweep_spec
        for op in op_set:
            callable_ = op.callable_
            callable_name = utils.get_name(callable_, full_qual=True)
            if not utils.match_name(callable_name, [callable_pattern]):
                continue
            patch_map.setdefault(op, dict())[param] = [
                i for i in utils.sweep_number(
                    callable_, param,
                    number_type(start), number_type(stop), number_type(step)
                )
            ]
    return patch_map

def apply_patch_map(patch_map, adaptor):
    prebuilt_op_set = set()
    for op, param_patch_map in patch_map.items():
        try:
            new_op_set = op.force_param(
                param_patch_map,
                tags_getter=adaptor.get_tags
            )
            prebuilt_op_set.update(new_op_set)
        except KeyError as e:
            error('Callable "{callable_}" has no parameter "{param}"'.format(
                callable_=op.name,
                param=e.args[0]
            ))
            continue

    return prebuilt_op_set

def load_from_db(db, adaptor, non_reusable_type_set, pattern_list, uuid_list, uuid_args):
    # We do not filter on UUID if we only got a type pattern list
    load_all_uuid = (
        pattern_list and not (
            uuid_list
            or uuid_args
        )
    )

    froz_val_set_set = set()
    if load_all_uuid:
        froz_val_set_set.update(
            utils.get_froz_val_set_set(db, None, pattern_list)
        )
    elif uuid_list:
        froz_val_set_set.update(
            utils.get_froz_val_set_set(db, uuid_list,
            pattern_list
        ))
    elif uuid_args:
        # Get the froz_val value we are interested in
        froz_val_list = utils.flatten_seq(
            utils.get_froz_val_set_set(db, [uuid_args],
            pattern_list
        ))
        for froz_val in froz_val_list:
            # Reload the whole context, except froz_val itself since we
            # only want its arguments. We load the "indirect" arguments as
            # well to ensure references to their types will be fulfilled by
            # them instead of computing new values.
            froz_val_set_set.add(frozenset(froz_val.get_by_predicate(
                lambda v: v is not froz_val and v.value is not NoValue
            )))

    # Otherwise, reload all the root froz_val values
    else:
        froz_val_set_set.update(
            frozenset(froz_val_seq)
            for froz_val_seq in db.froz_val_seq_list
        )

    prebuilt_op_set = set()

    # Build the set of PrebuiltOperator that will inject the loaded values
    # into the tests
    for froz_val_set in froz_val_set_set:
        froz_val_list = [
            froz_val for froz_val in froz_val_set
            if froz_val.value not in (NoValue, None)
        ]
        if not froz_val_list:
            continue

        def key(froz_val):
            # Since no two sub-expression is allowed to compute values of a
            # given type, it is safe to assume that grouping by the
            # non-tagged ID will group together all values of compatible
            # types into one PrebuiltOperator per root Expression.
            return froz_val.get_id(full_qual=True, with_tags=False)

        for full_id, group in itertools.groupby(froz_val_list, key=key):
            froz_val_list = list(group)

            type_ = utils.get_common_base(
                type(froz_val.value)
                for froz_val in froz_val_list
            )

            # Do not reload non-reusable objects, since that would lead to an
            # unexpected mix-up when multiple of them were used in the same
            # expresion.
            # Also, it would break the guarantee that they won't be used twice.
            if type_ in non_reusable_type_set:
                continue

            id_ = froz_val_list[0].get_id(
                full_qual=False,
                qual=False,
                # Do not include the tags to avoid having them displayed
                # twice, and to avoid wrongfully using the tag of the first
                # item in the list for all items.
                with_tags=False,
            )

            prebuilt_op_set.add(
                engine.PrebuiltOperator(
                    type_, froz_val_list, id_=id_,
                    non_reusable_type_set=non_reusable_type_set,
                    tags_getter=adaptor.get_tags,
                ))

    return prebuilt_op_set

def _main(argv):
    parser = argparse.ArgumentParser(description="""
Test runner

PATTERNS
    All patterns are fnmatch pattern, following basic shell globbing syntax.
    A pattern starting with "!" is used as a negative pattern.
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--debug', action='store_true',
        help="""Show complete Python backtrace when exekall crashes.""")

    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    run_parser = subparsers.add_parser('run',
    description="""
    Run the tests
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    # It is not possible to give a default value to positional options,
    # otherwise adaptor-specific options' values will be picked up as Python
    # sources, and importing the modules will therefore fail with unknown files
    # error.
    run_parser.add_argument('python_files', nargs='+',
        metavar='PYTHON_SRC',
        help="""Python modules files. If passed a folder, all contained files recursively are selected. By default, the current directory is selected.""")


    run_parser.add_argument('-s', '--select', action='append',
        metavar='SELECT_PATTERN',
        default=[],
        help="""Only run the expressions with an ID matching any of the supplied filters.""")

    # Same as --select, but allows multiple patterns without needing to
    # repeat the option. This is mostly available to support wrapper
    # scripts, and is not recommended for direct use since it can lead to
    # some parsing ambiguities.
    run_parser.add_argument('--select-multiple', nargs='*',
        default=[],
        help=argparse.SUPPRESS,
    )

    run_parser.add_argument('--list', action='store_true',
        help="""List the expressions that will be run without running them.""")

    # Show the list of expressions in reStructuredText format, suitable for
    # inclusion in Sphinx documentation
    run_parser.add_argument('--rst-list', action='store_true',
        help=argparse.SUPPRESS)

    run_parser.add_argument('--log-level', default='info',
        choices=('debug', 'info', 'warn', 'error', 'critical'),
        help="""Change the default log level of the standard logging module.""")

    run_parser.add_argument('--verbose', '-v', action='count', default=0,
        help="""More verbose output. Can be repeated for even more verbosity. This only impacts exekall output, --log-level for more global settings.""")

    artifact_dir_group = run_parser.add_mutually_exclusive_group()
    artifact_dir_group.add_argument('--artifact-root',
        default=os.getenv('EXEKALL_ARTIFACT_ROOT', 'artifacts'),
        help="Root folder under which the artifact folders will be created. Defaults to EXEKALL_ARTIFACT_ROOT env var.")

    artifact_dir_group.add_argument('--artifact-dir',
        default=os.getenv('EXEKALL_ARTIFACT_DIR'),
        help="""Folder in which the artifacts will be stored. Defaults to EXEKALL_ARTIFACT_DIR env var.""")

    run_parser.add_argument('--load-db', action='append',
        default=[],
        help="""Reload a database to use some of its objects. The DB and its artifact directory will be merged in the produced DB at the end of the execution, to form a self-contained artifact directory.""")

    run_parser.add_argument('--load-type', action='append',
        metavar='LOAD_TYPE_PATTERN',
        default=[],
        help="""Load the (indirect) instances of the given class from the database instead of the root objects.""")

    uuid_group = run_parser.add_mutually_exclusive_group()

    uuid_group.add_argument('--load-uuid', action='append',
        default=[],
        help="""Load the given UUID from the database.""")

    uuid_group.add_argument('--replay',
        help="""Replay the execution of the given UUID, loading as much prerequisite from the DB as possible.""")

    # Load the parameters that were used to compute the value with the given
    # UUID from the database. This can be used as a more flexible form of
    # --replay that does not imply restricting the selection
    uuid_group.add_argument('--load-uuid-args',
        help=argparse.SUPPRESS)

    run_parser.add_argument('--restrict', action='append',
        metavar='RESTRICT_PATTERN',
        default=[],
        help="""Callable names patterns. Types produced by these callables will only be produced by these (other callables will be excluded).""")

    run_parser.add_argument('--forbid', action='append',
        metavar='FORBID_PATTERN',
        default=[],
        help="""Fully qualified type names patterns. Callable returning these types or any subclass will not be called.""")

    run_parser.add_argument('--allow', action='append',
        metavar='ALLOW_PATTERN',
        default=[],
        help="""Allow using callable with a fully qualified name matching these patterns, even if they have been not selected for various reasons.""")

    goal_group = run_parser.add_mutually_exclusive_group()
    goal_group.add_argument('--goal', action='append',
        metavar='GOAL_PATTERN',
        default=[],
        help="""Compute expressions leading to an instance of a class with name matching this pattern (or a subclass of it).""")

    goal_group.add_argument('--callable-goal', action='append',
        metavar='CALLABLE_GOAL_PATTERN',
        default=[],
        help="""Compute expressions ending with a callable which name is matching this pattern.""")

    run_parser.add_argument('--sweep', nargs=5, action='append', default=[],
        metavar=('CALLABLE_PATTERN', 'PARAM', 'START', 'STOP', 'STEP'),
        help="""Parametric sweep on a function parameter. It needs five fields:
    * pattern matching qualified name of the callable
    * name of the parameter
    * start value
    * stop value
    * step size.""")

    run_parser.add_argument('--template-scripts', metavar='SCRIPT_FOLDER',
        help="""Only create the template scripts of the expressions without running them.""")

    run_parser.add_argument('--adaptor',
        help="""Adaptor to use from the customization module, if there is more than one to choose from.""")

    run_parser.add_argument('-n', type=int,
        default=1,
        help="""Run the tests for a number of iterations.""")

    run_parser.add_argument('--share', action='append',
        default=[],
        help="""Class name pattern to share between multiple iterations.""")

    run_parser.add_argument('--random-order', action='store_true',
        help="""Run the expressions in a random order, instead of sorting by name.""")


    merge_parser = subparsers.add_parser('merge',
    description="""
Merge artifact directories of "exekall run" executions.

By default, it will use hardlinks instead of copies to improve speed and avoid
eating up large amount of space, but that means that artifact directories
should be treated as read-only.
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    merge_parser.add_argument('artifact_dirs', nargs='+',
        help="""Artifact directories created using "exekall run", or value databases to merge.""")

    merge_parser.add_argument('-o', '--output', required=True,
        help="""Output merged artifacts directory or value database.""")

    merge_parser.add_argument('--copy', action='store_true',
        help="""Force copying files, instead of using hardlinks.""")


    compare_parser = subparsers.add_parser('compare',
    description="""
Compare two DBs produced by exekall run.
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    compare_parser.add_argument('db', nargs=2,
        help="""DBs created using exekall run to compare.""")

    # Avoid showing help message on the incomplete parser. Instead, we carry on
    # and the help will be displayed after the parser customization of run
    # subcommand has a chance to take place.
    help_options =  ('-h', '--help')
    no_help_argv = [
        arg for arg in argv
        if arg not in help_options
    ]
    try:
        # Silence argparse until we know what is going on
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            args, _ = parser.parse_known_args(no_help_argv)
    # If it fails, that may be because of an incomplete command line with just
    # --help for example. If it was for another reason, it will fail again and
    # show the message.
    except SystemExit:
        parser.parse_known_args(argv)
        # That should never be reached
        assert False

    if not args.subcommand:
        parser.print_help()
        return 2

    global show_traceback
    show_traceback = args.debug

    # Some subcommands need not parser customization, in which case we more
    # strictly parse the command line
    if args.subcommand not in ('run', 'compare'):
        parser.parse_args(argv)

    if args.subcommand == 'run':
        # do_run needs to reparse the CLI, so it needs the parser and argv
        return do_run(args, parser, run_parser, argv)

    elif args.subcommand == 'merge':
        return do_merge(
            artifact_dirs=args.artifact_dirs,
            output_dir=args.output,
            use_hardlink=(not args.copy),
        )

    elif args.subcommand == 'compare':
        return do_compare(
            parser=parser,
            compare_parser=compare_parser,
            argv=argv,
            db_path_list=args.db,
        )

def do_compare(parser, compare_parser, argv, db_path_list):
    assert len(db_path_list) == 2
    db_list = [
        engine.ValueDB.from_path(path)
        for path in db_path_list
    ]

    adaptor_cls_set = {
        db.adaptor_cls
        for db in db_list
    }
    if len(adaptor_cls_set) != 1:
        raise ValueError('Cannot compare DBs that were built using a different adaptor: {}'.format(adaptor_cls_set))
    adaptor_cls = utils.take_first(adaptor_cls_set)

    # Add all the CLI arguments of the adaptor before reparsing the
    # command line.
    adaptor_cls.register_compare_param(compare_parser)

    # Reparse the command line after the adaptor had a chance to add its own
    # arguments.
    args = parser.parse_args(argv)

    # Create the adaptor with the args, so it can use it to implement
    # comparison
    adaptor = adaptor_cls(args)

    return adaptor.compare_db_list(db_list)

def do_merge(artifact_dirs, output_dir, use_hardlink=True, output_exist=False):
    output_dir = pathlib.Path(output_dir)

    artifact_dirs = [pathlib.Path(path) for path in artifact_dirs]
    # Dispatch folders and databases
    db_path_list = [path for path in artifact_dirs if path.is_file()]
    artifact_dirs = [path for path in artifact_dirs if path.is_dir()]

    # Only DB paths
    if not artifact_dirs:
        merged_db_path = output_dir
    else:
        # This will fail loudly if the folder already exists
        os.makedirs(str(output_dir), exist_ok=output_exist)
        merged_db_path = output_dir/DB_FILENAME

    (output_dir/'BY_UUID').mkdir(exist_ok=True)
    testsession_uuid_list = []
    for artifact_dir in artifact_dirs:
        with (artifact_dir/'UUID').open(encoding='utf-8') as f:
            testsession_uuid = f.read().strip()
            testsession_uuid_list.append(testsession_uuid)

        src_by_uuid = artifact_dir/'BY_UUID'
        for uuid_symlink in src_by_uuid.iterdir():
            target = uuid_symlink.resolve()
            target = pathlib.Path('..', target.relative_to(artifact_dir.resolve()))
            (output_dir/'BY_UUID'/uuid_symlink.name).symlink_to(target)


        link_base_path = pathlib.Path('ORIGIN', testsession_uuid)
        shutil.copytree(
            str(src_by_uuid),
            str(output_dir/link_base_path/'BY_UUID'),
            symlinks=True,
        )

        # Copy all the files recursively
        for dirpath, dirnames, filenames in os.walk(str(artifact_dir)):
            dirpath = pathlib.Path(dirpath)
            for name in filenames:
                path = dirpath/name
                rel_path = pathlib.Path(os.path.relpath(str(path), str(artifact_dir)))
                link_path = output_dir/link_base_path/rel_path

                levels = pathlib.Path(*(['..'] * (
                      len(rel_path.parents)
                    + len(link_base_path.parents)
                    - 1
                )))
                src_link_path = levels/rel_path

                # top-level files are relocated under a ORIGIN instead of having
                # a symlink, otherwise they would clash
                if dirpath == artifact_dir:
                    dst_path = link_path
                    create_link = False
                # Otherwise, UUIDs will ensure that there is no clash
                else:
                    dst_path = output_dir/rel_path
                    create_link = True

                # Create the folder and make sure that all its parents get the
                # same stats as the original one, in order to preserve creation
                # date.
                os.makedirs(str(dst_path.parent), exist_ok=True)
                # We do not do copystat on the topmost parent, as it is shared
                # by all original artifact_dir
                for parent in list(rel_path.parents)[:-2]:
                    stat_src = artifact_dir/parent
                    stat_dst = output_dir/parent
                    shutil.copystat(str(stat_src), str(stat_dst))

                # Create a mirror of the original hierarchy
                if create_link:
                    os.makedirs(str(link_path.parent), exist_ok=True)
                    link_path.symlink_to(src_link_path)

                if use_hardlink:
                    os.link(str(path), str(dst_path))
                    # Preserve the original creation date
                    shutil.copystat(str(path), str(dst_path), follow_symlinks=False)
                else:
                    shutil.copy2(str(path), str(dst_path))

                if dirpath == artifact_dir and name == DB_FILENAME:
                    db_path_list.append(path)

    if artifact_dirs:
        # Combine the origin UUIDs to have a stable UUID for the merged
        # artifacts
        combined_uuid = hashlib.sha256(
            b'\n'.join(
                uuid_.encode('ascii')
                for uuid_ in sorted(testsession_uuid_list)
            )
        ).hexdigest()[:32]
        with (output_dir/'UUID').open('wt') as f:
            f.write(combined_uuid+'\n')

    if output_exist:
        db_path_list.append(merged_db_path)

    merged_db = engine.ValueDB.merge(
        engine.ValueDB.from_path(path)
        for path in db_path_list
    )
    merged_db.to_path(merged_db_path)

def do_run(args, parser, run_parser, argv):
    # Import all modules, before selecting the adaptor
    module_set = utils.import_paths(args.python_files)

    # Look for a customization submodule in one of the parent packages of the
    # modules we specified on the command line.
    module_set.update(utils.find_customization_module_set(module_set))

    adaptor_name = args.adaptor
    adaptor_cls = AdaptorBase.get_adaptor_cls(adaptor_name)
    if not adaptor_cls:
        raise RuntimeError('Adaptor "{}" cannot be found'.format(adaptor_name))
    # Add all the CLI arguments of the adaptor before reparsing the
    # command line.
    adaptor_cls.register_run_param(run_parser)

    # Reparse the command line after the adaptor had a chance to add its own
    # arguments.
    args = parser.parse_args(argv)

    verbose = args.verbose

    iteration_nr = args.n
    shared_pattern_set = set(args.share)
    random_order = args.random_order

    adaptor = adaptor_cls(args)

    only_list = args.list
    only_template_scripts = args.template_scripts

    rst_expr_list = args.rst_list
    if rst_expr_list:
        only_list = True

    type_goal_pattern_set = set(args.goal)
    callable_goal_pattern_set = set(args.callable_goal)

    if not (type_goal_pattern_set or callable_goal_pattern_set):
        type_goal_pattern_set = set(adaptor_cls.get_default_type_goal_pattern_set())

    load_db_path_list = args.load_db
    load_db_pattern_list = args.load_type
    load_db_uuid_list = args.load_uuid
    load_db_replay_uuid = args.replay
    load_db_uuid_args = load_db_replay_uuid or args.load_uuid_args

    user_filter_set = set(args.select)
    user_filter_set.update(args.select_multiple)

    if load_db_replay_uuid and user_filter_set:
        run_parser.error('--replay and --select cannot be used at the same time')

    if load_db_replay_uuid and not load_db_path_list:
        run_parser.error('--load-db must be specified to use --replay')

    restricted_pattern_set = set(args.restrict)
    forbidden_pattern_set = set(args.forbid)
    allowed_pattern_set = set(args.allow)
    allowed_pattern_set.update(restricted_pattern_set)
    allowed_pattern_set.update(callable_goal_pattern_set)

    # Setup the artifact_dir so we can create a verbose log in there
    date = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
    testsession_uuid = utils.create_uuid()
    if only_template_scripts:
        artifact_dir = pathlib.Path(only_template_scripts)
    elif args.artifact_dir:
        artifact_dir = pathlib.Path(args.artifact_dir)
    # If we are not given a specific folder, we create one under the root we
    # were given
    else:
        artifact_dir = pathlib.Path(args.artifact_root, date + '_' + testsession_uuid)

    if only_list:
        debug_log = None
        info_log = None
    else:
        artifact_dir.mkdir(parents=True)
        artifact_dir = artifact_dir.resolve()
        # Update the CLI arguments so the customization module has access to the
        # correct value
        args.artifact_dir = artifact_dir
        debug_log = artifact_dir/'DEBUG.log'
        info_log = artifact_dir/'INFO.log'

    utils.setup_logging(args.log_level, debug_log, info_log, verbose=verbose)

    # Get the set of all callables in the given set of modules
    callable_pool = utils.get_callable_set(module_set, verbose=verbose)

    # Build the pool of operators from the callables
    non_reusable_type_set = set(utils.flatten_seq(
        utils.get_subclasses(cls)
        for cls in adaptor.get_non_reusable_type_set()
    ))

    op_set = build_op_set(
        callable_pool, non_reusable_type_set, allowed_pattern_set, adaptor,
    )

    # Load objects from an existing database
    if load_db_path_list:
        db_list = []
        for db_path in load_db_path_list:
            db = engine.ValueDB.from_path(db_path)
            op_set.update(
                load_from_db(db, adaptor, non_reusable_type_set,
                    load_db_pattern_list, load_db_uuid_list, load_db_uuid_args
                )
            )
            db_list.append(db)
    # Get the prebuilt operators from the adaptor
    else:
        db_list = []
        op_set.update(adaptor.get_prebuilt_set())

    # Force some parameter values to be provided with a specific callable
    patch_map = build_patch_map(args.sweep, op_set)
    op_set.update(apply_patch_map(patch_map, adaptor))

    # Some operators are hidden in IDs since they don't add useful information
    # (internal classes)
    hidden_callable_set = {
        op.callable_
        for op in adaptor.get_hidden_op_set(op_set)
    }

    # These get_id() options are used for all user-exposed listing that is supposed to be
    # filterable with user_filter_set (like only_list)
    filterable_id_kwargs = dict(
        full_qual=False,
        qual=False,
        with_tags=False,
        hidden_callable_set=hidden_callable_set
    )

    # Restrict the Expressions that will be executed to just the one we
    # care about
    if db_list and load_db_replay_uuid:
        id_kwargs = copy.copy(filterable_id_kwargs)
        del id_kwargs['hidden_callable_set']
        # Let the merge logic handle duplicated UUIDs
        db = engine.ValueDB.merge(db_list)
        user_filter_set = {
            db.get_by_uuid(load_db_replay_uuid).get_id(**id_kwargs)
        }

    # Only print once per parameters' tuple
    if verbose:
        @utils.once
        def handle_non_produced(cls_name, consumer_name, param_name, callable_path):
            info('Nothing can produce instances of {cls} needed for {consumer} (parameter "{param}", along path {path})'.format(
                cls = cls_name,
                consumer = consumer_name,
                param = param_name,
                path = ' -> '.join(utils.get_name(callable_) for callable_ in callable_path)
            ))

        @utils.once
        def handle_cycle(path):
            error('Cyclic dependency detected: {path}'.format(
                path = ' -> '.join(
                    utils.get_name(callable_)
                    for callable_ in path
                )
            ))
    else:
        handle_non_produced = 'ignore'
        handle_cycle = 'ignore'

    # Get the callable goals, either by the callable name or the value type
    root_op_set = set(
        op for op in op_set
        if (
            utils.match_name(op.get_name(full_qual=True), callable_goal_pattern_set)
            or
            # All producers of the goal types can be a root operator in the
            # expressions we are going to build, i.e. the outermost function call
            utils.match_base_cls(op.value_type, type_goal_pattern_set)
        # Only keep the Expression where the outermost (root) operator is
        # defined in one of the files that were explicitely specified on the
        # command line.
        ) and inspect.getmodule(op.callable_) in module_set
    )

    # Build the class context from the set of Operator's that we collected
    class_ctx = engine.ClassContext.from_op_set(
        op_set=op_set,
        forbidden_pattern_set=forbidden_pattern_set,
        restricted_pattern_set=restricted_pattern_set
    )

    # Build the list of Expression that can be constructed from the set of
    # callables
    expr_list = class_ctx.build_expr_list(
        root_op_set,
        non_produced_handler=handle_non_produced,
        cycle_handler=handle_cycle,
    )
    # First, sort with the fully qualified ID so we have the strongest stability
    # possible from one run to another
    expr_list.sort(key=lambda expr: expr.get_id(full_qual=True, with_tags=True))
    # Then sort again according to what will be displayed. Since it is a stable
    # sort, it will keep a stable order for IDs that look the same but actually
    # differ in their hidden part
    expr_list.sort(key=lambda expr: expr.get_id(qual=False, with_tags=True))

    if random_order:
        random.shuffle(expr_list)

    if user_filter_set:
        expr_list = [
            expr for expr in expr_list
            if utils.match_name(expr.get_id(**filterable_id_kwargs), user_filter_set)
        ]

    if not expr_list:
        info('Nothing to do, check --help while passing some python sources to get the full help.')
        return 1

    id_kwargs = {
        **filterable_id_kwargs,
        'full_qual': bool(verbose),
    }

    if rst_expr_list:
        id_kwargs['style'] = 'rst'
        for expr in expr_list:
            out(expr.get_id(**id_kwargs))
    else:
        out('The following expressions will be executed:\n')
        for expr in expr_list:
            out(expr.get_id(**id_kwargs))

            if verbose >= 2:
                out(expr.get_structure() + '\n')

    if only_list:
        return 0

    # Get a list of ComputableExpression in order to execute them
    expr_list = engine.ComputableExpression.from_expr_list(expr_list)

    if iteration_nr > 1:
        shared_op_set = {
            # We don't allow matching on root operators, since that would be
            # pointless. Sharing root operators basically means doing the work
            # once, and then reusing everything at every iteration.
            op for op in (op_set - root_op_set)
            if utils.match_base_cls(op.value_type, shared_pattern_set)
        }
        predicate = lambda expr: expr.op not in shared_op_set

        iteration_expr_list = [
            # Apply CSE within each iteration
            engine.ComputableExpression.cse(
                expr.clone_by_predicate(predicate)
                for expr in expr_list
            )
            for i in range(iteration_nr)
        ]
    else:
        iteration_expr_list = [expr_list]




    exec_ret_code = exec_expr_list(
        iteration_expr_list=iteration_expr_list,
        adaptor=adaptor,
        artifact_dir=artifact_dir,
        testsession_uuid=testsession_uuid,
        hidden_callable_set=hidden_callable_set,
        only_template_scripts=only_template_scripts,
        adaptor_cls=adaptor_cls,
        verbose=verbose,
    )

    # If we reloaded a DB, merge it with the current DB so the outcome is a
    # self-contained artifact dir
    if load_db_path_list:
        orig_list = [
            path if path.is_dir() else path.parent
            for path in map(pathlib.Path, load_db_path_list)
        ]
        do_merge(orig_list, artifact_dir, output_exist=True)

    return exec_ret_code

def exec_expr_list(iteration_expr_list, adaptor, artifact_dir, testsession_uuid,
        hidden_callable_set, only_template_scripts, adaptor_cls, verbose):

    if not only_template_scripts:
        with (artifact_dir/'UUID').open('wt') as f:
            f.write(testsession_uuid+'\n')

        (artifact_dir/'BY_UUID').mkdir()

    out('\nArtifacts dir: {}\n'.format(artifact_dir))

    for expr in utils.flatten_seq(iteration_expr_list):
        expr_short_id = expr.get_id(
            hidden_callable_set=hidden_callable_set,
            with_tags=False,
            full_qual=False,
            qual=False,
        )

        data = expr.data
        data['id'] = expr_short_id
        data['uuid'] = expr.uuid

        expr_artifact_dir = pathlib.Path(
            artifact_dir,
            expr_short_id,
            expr.uuid
        )
        expr_artifact_dir.mkdir(parents=True)
        expr_artifact_dir = expr_artifact_dir.resolve()
        data['artifact_dir'] = artifact_dir
        data['expr_artifact_dir'] = expr_artifact_dir

        adaptor.update_expr_data(data)

        with (expr_artifact_dir/'UUID').open('wt') as f:
            f.write(expr.uuid + '\n')

        with (expr_artifact_dir/'ID').open('wt') as f:
            f.write(expr_short_id+'\n')

        with (expr_artifact_dir/'STRUCTURE').open('wt') as f:
            f.write(expr.get_id(
                hidden_callable_set=hidden_callable_set,
                with_tags=False,
                full_qual=True,
            ) + '\n\n')
            f.write(expr.get_structure() + '\n')

        is_svg, dot_output = utils.render_graphviz(expr)
        graphviz_path = expr_artifact_dir/'STRUCTURE.{}'.format(
            'svg' if is_svg else 'dot'
        )
        with graphviz_path.open('wt', encoding='utf-8') as f:
            f.write(dot_output)

        with (expr_artifact_dir/'TESTCASE_TEMPLATE.py').open(
            'wt', encoding='utf-8'
        ) as f:
            f.write(
                expr.get_script(
                    prefix = 'expr',
                    db_path = os.path.join('..', DB_FILENAME),
                    db_relative_to = '__file__',
                )[1]+'\n',
            )

    if only_template_scripts:
        return 0

    # Preserve the execution order, so the summary is displayed in the same
    # order
    result_map = collections.OrderedDict()
    for i, expr_list in enumerate(iteration_expr_list):
        i += 1
        info('Iteration #{}\n'.format(i))

        for expr in expr_list:
            exec_start_msg = 'Executing: {short_id}\n\nID: {full_id}\nArtifacts: {folder}\nUUID: {uuid_}'.format(
                    short_id=expr.get_id(
                        hidden_callable_set=hidden_callable_set,
                        full_qual=False,
                        qual=False,
                    ),

                    full_id=expr.get_id(
                        hidden_callable_set=hidden_callable_set if not verbose else None,
                        full_qual=True,
                    ),
                    folder=expr.data['expr_artifact_dir'],
                    uuid_=expr.uuid
            ).replace('\n', '\n# ')

            delim = '#' * (len(exec_start_msg.splitlines()[0]) + 2)
            out(delim + '\n# ' + exec_start_msg + '\n' + delim)

            result_list = list()
            result_map[expr] = result_list

            def pre_line():
                out('-' * 40)
            # Make sure that all the output of the expression is flushed to ensure
            # there won't be any buffered stderr output being displayed after the
            # "official" end of the Expression's execution.
            def flush_std_streams():
                sys.stdout.flush()
                sys.stderr.flush()

            def get_uuid_str(expr_val):
                return 'UUID={}'.format(expr_val.uuid)

            computed_expr_val_set = set()
            reused_expr_val_set = set()
            def log_expr_val(expr_val, reused):
                # Consider that PrebuiltOperator reuse values instead of actually
                # computing them.
                if isinstance(expr_val.expr.op, engine.PrebuiltOperator):
                    reused = True

                if reused:
                    msg = 'Reusing already computed {id} {uuid}'
                    reused_expr_val_set.add(expr_val)
                else:
                    msg = 'Computed {id} {uuid}'
                    computed_expr_val_set.add(expr_val)

                op = expr_val.expr.op
                if (
                    op.callable_ not in hidden_callable_set
                    and not issubclass(op.value_type, engine.ForcedParamType)
                ):
                    log_f = info
                else:
                    log_f = debug

                log_f(msg.format(
                    id=expr_val.get_id(
                        full_qual=False,
                        with_tags=True,
                        hidden_callable_set=hidden_callable_set,
                    ),
                    uuid=get_uuid_str(expr_val),
                ))

            # This returns an iterator
            executor = expr.execute(log_expr_val)

            out('')
            for result in utils.iterate_cb(executor, pre_line, flush_std_streams):
                for excep_val in result.get_excep():
                    excep = excep_val.excep
                    tb = utils.format_exception(excep)
                    error('{e_name}: {e}\nID: {id}\n{tb}'.format(
                            id=excep_val.get_id(),
                            e_name=utils.get_name(type(excep)),
                            e=excep,
                            tb=tb,
                        ),
                    )

                prefix = 'Finished {uuid} '.format(uuid=get_uuid_str(result))
                out('{prefix}{id}'.format(
                    id=result.get_id(
                        full_qual=False,
                        qual=False,
                        mark_excep=True,
                        with_tags=True,
                        hidden_callable_set=hidden_callable_set,
                    ).strip().replace('\n', '\n'+len(prefix)*' '),
                    prefix=prefix,
                ))

                out(adaptor.result_str(result))
                result_list.append(result)


            out('')
            expr_artifact_dir = expr.data['expr_artifact_dir']

            # Finalize the computation
            adaptor.finalize_expr(expr)

            # Dump the reproducer script
            with (expr_artifact_dir/'TESTCASE.py').open('wt', encoding='utf-8') as f:
                f.write(
                    expr.get_script(
                        prefix = 'testcase',
                        db_path = os.path.join('..', '..', DB_FILENAME),
                        db_relative_to = '__file__',
                    )[1]+'\n',
                )

            def format_uuid(expr_val_list):
                uuid_list = sorted({
                    expr_val.uuid
                    for expr_val in expr_val_list
                })
                return '\n'.join(uuid_list)

            def write_uuid(path, *args):
                with path.open('wt') as f:
                    f.write(format_uuid(*args) + '\n')

            write_uuid(expr_artifact_dir/'VALUES_UUID', result_list)
            write_uuid(expr_artifact_dir/'REUSED_VALUES_UUID', reused_expr_val_set)
            write_uuid(expr_artifact_dir/'COMPUTED_VALUES_UUID', computed_expr_val_set)

            # From there, use a relative path for symlinks
            expr_artifact_dir = pathlib.Path('..', expr_artifact_dir.relative_to(artifact_dir))
            computed_uuid_set = {
                expr_val.uuid
                for expr_val in computed_expr_val_set
            }
            computed_uuid_set.add(expr.uuid)
            for uuid_ in computed_uuid_set:
                (artifact_dir/'BY_UUID'/uuid_).symlink_to(expr_artifact_dir)

    db = engine.ValueDB(
        engine.FrozenExprValSeq.from_expr_list(
            utils.flatten_seq(iteration_expr_list),
            hidden_callable_set=hidden_callable_set,
        ),
        adaptor_cls=adaptor_cls,
    )

    db_path = artifact_dir/DB_FILENAME
    db.to_path(db_path)

    out('#'*80)
    info('Artifacts dir: {}'.format(artifact_dir))
    info('Result summary:')

    # Display the results summary
    summary = adaptor.get_summary(result_map)
    out(summary)
    with (artifact_dir/'SUMMARY').open('wt', encoding='utf-8') as f:
        f.write(summary + '\n')

    # Output the merged script with all subscripts
    script_path = artifact_dir/'ALL_SCRIPTS.py'
    result_name_map, all_scripts = engine.Expression.get_all_script(
        utils.flatten_seq(iteration_expr_list),
        prefix='expr',
        db_path=db_path.relative_to(artifact_dir),
        db_relative_to='__file__',
        db=db,
        adaptor_cls=adaptor_cls,
    )

    with script_path.open('wt', encoding='utf-8') as f:
        f.write(all_scripts + '\n')

    return 0

SILENT_EXCEPTIONS = (KeyboardInterrupt, BrokenPipeError)
GENERIC_ERROR_CODE = 1

# Global variable reset once we parsed the command line
show_traceback = True
def main(argv=sys.argv[1:]):
    return_code = 0

    try:
        return_code = _main(argv)
    # Quietly exit for these exceptions
    except SILENT_EXCEPTIONS:
        pass
    except SystemExit as e:
        return_code = e.code
    # Catch-all
    except Exception as e:
        if show_traceback:
            error(
                'Exception traceback:\n' +
                utils.format_exception(e)
            )
        # Always show the concise message
        error(e)
        return_code = GENERIC_ERROR_CODE

    sys.exit(return_code)

if __name__ == '__main__':
    main()
