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
import datetime
import hashlib
import importlib
import inspect
import io
import itertools
import os
import pathlib
import shutil
import sys
import glob

from exekall.customization import AdaptorBase
import exekall.engine as engine
from exekall.engine import NoValue
import exekall.utils as utils
from exekall.utils import take_first, error, warn, debug, info, out

DB_FILENAME = 'VALUE_DB.pickle.xz'

def _main(argv):
    parser = argparse.ArgumentParser(description="""
    LISA test runner
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

    # It is not possible to give a default value to that option, otherwise
    # adaptor-specific options' values will be picked up as Python sources, and
    # import the modules will therefore fail with unknown files.
    run_parser.add_argument('python_files', nargs='+',
        metavar='PYTHON_SRC',
        help="""Python modules files. If passed a folder, all contained files
are selected, recursively. By default, the current directory is selected.""")

    run_parser.add_argument('--adaptor',
        help="""Adaptor to use from the customization module, if there is more
than one to choose from.""")

    run_parser.add_argument('--filter',
        help="""Only run the testcases with an ID matching the filter.""")

    run_parser.add_argument('--restrict', action='append',
        default=[],
        help="""Callable names patterns. Types produced by these callables will
only be produced by these (other callables will be excluded).""")

    run_parser.add_argument('--forbid', action='append',
        default=[],
        help="""Type names patterns. Callable returning these types or any subclass will not be called.""")

    run_parser.add_argument('--allow', action='append',
        default=[],
        help="""Allow using callable with a fully qualified name matching these patterns, even if they have been not selected for various reasons.""")

    run_parser.add_argument('--modules-root', action='append', default=[],
        help="Equivalent to setting PYTHONPATH")

    artifact_dir_group = run_parser.add_mutually_exclusive_group()
    artifact_dir_group.add_argument('--artifact-root',
        default=os.getenv('EXEKALL_ARTIFACT_ROOT', 'artifacts'),
        help="Root folder under which the artifact folders will be created")

    artifact_dir_group.add_argument('--artifact-dir',
        default=os.getenv('EXEKALL_ARTIFACT_DIR'),
        help="""Folder in which the artifacts will be stored.""")

    run_parser.add_argument('--load-db',
        help="""Reload a database and use its results as prebuilt objects.""")

    run_parser.add_argument('--load-type', action='append', default=[],
        help="""Load the (indirect) instances of the given class from the
database instead of the root objects.""")

    uuid_group = run_parser.add_mutually_exclusive_group()

    uuid_group.add_argument('--load-uuid', action='append', default=[],
        help="""Load the given UUID from the database. What is reloaded can be
refined with --load-type.""")

    uuid_group.add_argument('--load-uuid-args',
        help="""Load the parameters of the values that were used to compute the
given UUID from the database.""")

    goal_group = run_parser.add_mutually_exclusive_group()
    goal_group.add_argument('--goal', action='append',
        help="""Compute expressions leading to an instance of the specified
class or a subclass of it.""")

    goal_group.add_argument('--callable-goal', action='append',
        default=[],
        help="""Compute expressions ending with a callable which name is
matching this pattern.""")

    run_parser.add_argument('--sweep', nargs=5, action='append', default=[],
        metavar=('CALLABLE', 'PARAM', 'START', 'STOP', 'STEP'),
        help="""Parametric sweep on a function parameter.
It needs five fields: the qualified name of the callable (pattern can be used),
the name of the parameter, the start value, stop value and step size.""")

    run_parser.add_argument('--verbose', '-v', action='count', default=0,
        help="""More verbose output.""")

    run_parser.add_argument('--dry-run', action='store_true',
        help="""Only show the tests that will be run without running them.""")

    run_parser.add_argument('--template-scripts', metavar='SCRIPT_FOLDER',
        help="""Only create the template scripts of the tests without running them.""")

    run_parser.add_argument('--log-level', default='info',
        choices=('debug', 'info', 'warn', 'error', 'critical'),
        help="""Change the default log level of the standard logging module.""")


    merge_parser = subparsers.add_parser('merge',
    description="""
    Merge artifact directories of "exekall run" executions
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    merge_parser.add_argument('artifact_dirs', nargs='+',
        help="""Artifact directories created using "exekall run", or value databases to merge.""")

    merge_parser.add_argument('--output', '-o', required=True,
        help="""Output merged artifacts directory or value database.""")

    merge_parser.add_argument('--copy', action='store_true',
        help="""Force copying files, instead of using hardlinks.""")

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
    if args.subcommand not in ['run']:
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

def do_merge(artifact_dirs, output_dir, use_hardlink=True):
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
        os.makedirs(str(output_dir))
        merged_db_path = output_dir/DB_FILENAME

    testsession_uuid_list = []
    for artifact_dir in artifact_dirs:
        with (artifact_dir/'UUID').open(encoding='utf-8') as f:
            testsession_uuid = f.read().strip()
            testsession_uuid_list.append(testsession_uuid)

        link_base_path = pathlib.Path('ORIGIN', testsession_uuid)

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

                os.makedirs(str(dst_path.parent), exist_ok=True)

                # Create a mirror of the original hierarchy
                if create_link:
                    os.makedirs(str(link_path.parent), exist_ok=True)
                    link_path.symlink_to(src_link_path)

                if use_hardlink:
                    os.link(path, dst_path)
                else:
                    shutil.copy2(path, dst_path)

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

    merged_db = engine.ValueDB.merge(
        engine.ValueDB.from_path(path)
        for path in db_path_list
    )
    merged_db.to_path(merged_db_path)

def do_run(args, parser, run_parser, argv):
    # Import all modules, before selecting the adaptor
    module_set = set()
    for path in args.python_files:
        path = pathlib.Path(path)
        # Recursively import all modules when passed folders
        if path.is_dir():
            for python_src in glob.iglob(str(path/'**'/'*.py'), recursive=True):
                module_set.add(utils.import_file(python_src))
        # If passed a file, just import it directly
        else:
            module_set.add(utils.import_file(path))

    # Look for a customization submodule in one of the parent packages of the
    # modules we specified on the command line.
    module_set.update(utils.find_customization_module_set(module_set))

    adaptor_name = args.adaptor
    adaptor_cls = AdaptorBase.get_adaptor_cls(adaptor_name)
    if not adaptor_cls:
        raise RuntimeError('Adaptor "{}" cannot be found'.format(adaptor_name))
    # Add all the CLI arguments of the adaptor before reparsing the
    # command line.
    adaptor_cls.register_cli_param(run_parser)

    # Reparse the command line after the adaptor had a chance to add its own
    # arguments.
    args = parser.parse_args(argv)

    verbose = args.verbose

    adaptor = adaptor_cls(args)

    dry_run = args.dry_run
    only_template_scripts = args.template_scripts

    type_goal_pattern = args.goal
    callable_goal_pattern_set = set(args.callable_goal)

    if not (type_goal_pattern or callable_goal_pattern_set):
        type_goal_pattern = set(adaptor_cls.get_default_type_goal_pattern_set())

    load_db_path = args.load_db
    load_db_pattern_list = args.load_type
    load_db_uuid_list = args.load_uuid
    load_db_uuid_args = args.load_uuid_args

    user_filter = args.filter
    restricted_pattern_set = set(args.restrict)
    forbidden_pattern_set = set(args.forbid)
    allowed_pattern_set = set(args.allow)
    allowed_pattern_set.update(restricted_pattern_set)
    allowed_pattern_set.update(callable_goal_pattern_set)

    sys.path.extend(args.modules_root)

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

    if dry_run:
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

    non_reusable_type_set = adaptor.get_non_reusable_type_set()

    # Get the prebuilt operators from the adaptor
    if not load_db_path:
        prebuilt_op_pool_list = adaptor.get_prebuilt_list()

    # Load objects from an existing database
    else:
        db = adaptor.load_db(load_db_path)

        # We do not filter on UUID if we only got a type pattern list
        load_all_uuid = (
            load_db_pattern_list and not (
                load_db_uuid_list
                or load_db_uuid_args
            )
        )

        serial_res_set = set()
        if load_all_uuid:
            serial_res_set.update(
                utils.load_serial_from_db(db, None, load_db_pattern_list)
            )
        elif load_db_uuid_list:
            serial_res_set.update(
                utils.load_serial_from_db(db, load_db_uuid_list,
                load_db_pattern_list
            ))
        elif load_db_uuid_args:
            # Get the serial value we are interested in
            serial_list = utils.flatten_nested_seq(
                utils.load_serial_from_db(db, [load_db_uuid_args],
                load_db_pattern_list
            ))
            for serial in serial_list:
                # Get all the UUIDs of its parameters
                param_uuid_list = [
                    param_serial.value_uuid
                    for param_serial in serial.param_expr_val_map.values()
                ]

                serial_res_set.update(
                        utils.load_serial_from_db(db, param_uuid_list,
                        load_db_pattern_list
                ))

        # Otherwise, reload all the root serial values
        else:
            serial_res_set.update(
                frozenset(l)
                for l in db.serial_seq_list
            )

        # Remove duplicates accross sets
        loaded_serial = set()
        serial_res_set_ = set()
        for serial_res in serial_res_set:
            serial_res = frozenset(serial_res - loaded_serial)
            loaded_serial.update(serial_res)
            if serial_res:
                serial_res_set_.add(serial_res)
        serial_res_set = serial_res_set_

        # Build the list of PrebuiltOperator that will inject the loaded values
        # into the tests
        prebuilt_op_pool_list = list()
        for serial_res in serial_res_set:
            serial_list = [
                serial for serial in serial_res
                if serial.value is not NoValue
            ]
            if not serial_list:
                continue

            def key(serial):
                # Since no two sub-expression is allowed to compute values of a
                # given type, it is safe to assume that grouping by the
                # non-tagged ID will group together all values of compatible
                # types into one PrebuiltOperator per root Expression.
                return serial.get_id(full_qual=True, with_tags=False)

            for full_id, group in itertools.groupby(serial_list, key=key):
                serial_list = list(group)

                type_ = type(serial_list[0].value)
                id_ = serial_list[0].get_id(
                    full_qual=False,
                    qual=False,
                    # Do not include the tags to avoid having them displayed
                    # twice, and to avoid wrongfully using the tag of the first
                    # item in the list for all items.
                    with_tags=False,
                )
                prebuilt_op_pool_list.append(
                    engine.PrebuiltOperator(
                        type_, serial_list, id_=id_,
                        non_reusable_type_set=non_reusable_type_set,
                        tags_getter=adaptor.get_tags,
                    ))

    # Pool of all callable considered
    callable_pool = utils.get_callable_set(module_set, verbose=verbose)
    op_pool = {
        engine.Operator(
            callable_,
            non_reusable_type_set=non_reusable_type_set,
            tags_getter=adaptor.get_tags
        )
        for callable_ in callable_pool
    }
    filtered_op_pool = adaptor.filter_op_pool(op_pool)
    # Make sure we have all the explicitely allowed operators
    filtered_op_pool.update(
        op for op in op_pool
        if utils.match_name(op.get_name(full_qual=True), allowed_pattern_set)
    )
    op_pool = filtered_op_pool

    # Force some parameter values to be provided with a specific callable
    patch_map = dict()
    for sweep_spec in args.sweep:
        number_type = float
        callable_pattern, param, start, stop, step = sweep_spec
        for callable_ in callable_pool:
            callable_name = utils.get_name(callable_, full_qual=True)
            if not utils.match_name(callable_name, [callable_pattern]):
                continue
            patch_map.setdefault(callable_name, dict())[param] = [
                i for i in utils.sweep_number(
                    callable_, param,
                    number_type(start), number_type(stop), number_type(step)
                )
            ]

    for op_name, param_patch_map in patch_map.items():
        for op in op_pool:
            if op.name == op_name:
                try:
                    new_op_pool = op.force_param(
                        param_patch_map,
                        tags_getter=adaptor.get_tags
                    )
                    prebuilt_op_pool_list.extend(new_op_pool)
                except KeyError as e:
                    error('Callable "{callable_}" has no parameter "{param}"'.format(
                        callable_=op_name,
                        param=e.args[0]
                    ))
                    continue

    # Register stub PrebuiltOperator for the provided prebuilt instances
    op_pool.update(prebuilt_op_pool_list)

    # Sort to have stable output
    op_pool = sorted(op_pool, key=lambda x: str(x.name))

    # Pool of classes that can be produced by the ops
    produced_pool = set(op.value_type for op in op_pool)

    # Set of all types that can be depended upon. All base class of types that
    # are actually produced are also part of this set, since they can be
    # dependended upon as well.
    cls_set = set()
    for produced in produced_pool:
        cls_set.update(utils.get_mro(produced))
    cls_set.discard(object)
    cls_set.discard(type(None))

    # Map all types to the subclasses that can be used when the type is
    # requested.
    cls_map = {
        # Make sure the list is deduplicated by building a set first
        cls: sorted({
            subcls for subcls in produced_pool
            if issubclass(subcls, cls)
        }, key=lambda cls: cls.__qualname__)
        for cls in cls_set
    }

    # Make sure that the provided PrebuiltOperator will be the only ones used
    # to provide their types
    only_prebuilt_cls = set(itertools.chain.from_iterable(
        # Augment the list of classes that can only be provided by a prebuilt
        # Operator with all the compatible classes
        cls_map[op.obj_type]
        for op in prebuilt_op_pool_list
    ))

    only_prebuilt_cls.discard(type(NoValue))

    # Map of all produced types to a set of what operator can create them
    def build_op_map(op_pool, only_prebuilt_cls, forbidden_pattern_set):
        op_map = dict()
        for op in op_pool:
            param_map, produced = op.get_prototype()
            is_prebuilt_op = isinstance(op, engine.PrebuiltOperator)
            if (
                (is_prebuilt_op or produced not in only_prebuilt_cls)
                and not utils.match_base_cls(produced, forbidden_pattern_set)
            ):
                op_map.setdefault(produced, set()).add(op)
        return op_map

    op_map = build_op_map(op_pool, only_prebuilt_cls, forbidden_pattern_set)

    # Restrict the production of some types to a set of operators.
    restricted_op_set = {
        # Make sure that we only use what is available
        op for op in itertools.chain.from_iterable(op_map.values())
        if utils.match_name(op.get_name(full_qual=True), restricted_pattern_set)
    }
    def apply_restrict(produced, op_set, restricted_op_set, cls_map):
        restricted_op_set = {
            op for op in restricted_op_set
            if op.value_type is produced
        }
        if restricted_op_set:
            # Make sure there is no other compatible type, so the only operators
            # that will be used to satisfy that dependency will be one of the
            # restricted_op_set item.
            cls_map[produced] = [produced]
            return restricted_op_set
        else:
            return op_set
    op_map = {
        produced: apply_restrict(produced, op_set, restricted_op_set, cls_map)
        for produced, op_set in op_map.items()
    }

    # Get the callable goals
    root_op_set = set()
    if callable_goal_pattern_set:
        root_op_set.update(
            op for op in op_pool
            if utils.match_name(op.get_name(full_qual=True), callable_goal_pattern_set)
        )

    # Get the list of root operators by produced type
    if type_goal_pattern:
        for produced, op_set in op_map.items():
            # All producers of the goal types can be a root operator in the
            # expressions we are going to build, i.e. the outermost function call
            if utils.match_base_cls(produced, type_goal_pattern):
                root_op_set.update(op_set)

    # Sort for stable output
    root_op_list = sorted(root_op_set, key=lambda op: str(op.name))

    # Some operators are hidden in IDs since they don't add useful information
    # (internal classes)
    hidden_callable_set = adaptor.get_hidden_callable_set(op_map)

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

    # Build the list of Expression that can be constructed from the set of
    # callables
    testcase_list = list(engine.ExpressionWrapper.build_expr_list(
        root_op_list, op_map, cls_map,
        non_produced_handler = handle_non_produced,
        cycle_handler = handle_cycle,
    ))
    # First, sort with the fully qualified ID so we have the strongest stability
    # possible from one run to another
    testcase_list.sort(key=lambda expr: take_first(expr.get_id(full_qual=True, with_tags=True)))
    # Then sort again according to what will be displayed. Since it is a stable
    # sort, it will keep a stable order for IDs that look the same but actually
    # differ in their hidden part
    testcase_list.sort(key=lambda expr: take_first(expr.get_id(qual=False, with_tags=True)))

    # Only keep the Expression where the outermost (root) operator is defined
    # in one of the files that were explicitely specified on the command line.
    testcase_list = [
        testcase
        for testcase in testcase_list
        if inspect.getmodule(testcase.op.callable_) in module_set
    ]

    if user_filter:
        testcase_list = [
            testcase for testcase in testcase_list
            if utils.match_name(take_first(testcase.get_id(
                # These options need to match what --dry-run gives (unless
                # verbose is used)
                full_qual=False,
                qual=False,
                hidden_callable_set=hidden_callable_set)), [user_filter])
        ]

    if not testcase_list:
        info('Nothing to do, exiting ...')
        return 0

    out('The following expressions will be executed:\n')
    for testcase in testcase_list:
        out(take_first(testcase.get_id(
            full_qual=bool(verbose),
            qual=bool(verbose),
            hidden_callable_set=hidden_callable_set
        )))
        if verbose >= 2:
            out(testcase.pretty_structure() + '\n')

    if dry_run:
        return 0

    if not only_template_scripts:
        with (artifact_dir/'UUID').open('wt') as f:
            f.write(testsession_uuid+'\n')

    db_loader = adaptor.load_db

    out('\nArtifacts dir: {}\n'.format(artifact_dir))

    # Apply the common subexpression elimination before trying to create the
    # template scripts
    executor_map = engine.Expression.get_executor_map(testcase_list)

    for testcase in executor_map.keys():
        testcase_short_id = take_first(testcase.get_id(
            hidden_callable_set=hidden_callable_set,
            with_tags=False,
            full_qual=False,
            qual=False,
        ))

        data = testcase.data
        data['id'] = testcase_short_id
        data['uuid'] = testcase.uuid

        testcase_artifact_dir = pathlib.Path(
            artifact_dir,
            testcase_short_id,
            testcase.uuid
        )
        testcase_artifact_dir.mkdir(parents=True)
        testcase_artifact_dir = testcase_artifact_dir.resolve()
        data['artifact_dir'] = artifact_dir
        data['testcase_artifact_dir'] = testcase_artifact_dir

        adaptor.update_expr_data(data)

        with (testcase_artifact_dir/'UUID').open('wt') as f:
            f.write(testcase.uuid + '\n')

        with (testcase_artifact_dir/'ID').open('wt') as f:
            f.write(testcase_short_id+'\n')

        with (testcase_artifact_dir/'STRUCTURE').open('wt') as f:
            f.write(take_first(testcase.get_id(
                hidden_callable_set=hidden_callable_set,
                with_tags=False,
                full_qual=True,
            )) + '\n\n')
            f.write(testcase.pretty_structure())

        with (testcase_artifact_dir/'TESTCASE_TEMPLATE.py').open(
                'wt', encoding='utf-8') as f:
            f.write(
                testcase.get_script(
                    prefix = 'testcase',
                    db_path = os.path.join('..', DB_FILENAME),
                    db_relative_to = '__file__',
                    db_loader=db_loader
                )[1]+'\n',
            )

    if only_template_scripts:
        return 0

    result_map = collections.defaultdict(list)
    for testcase, executor in executor_map.items():
        exec_start_msg = 'Executing: {short_id}\n\nID: {full_id}\nArtifacts: {folder}\nUUID: {uuid_}'.format(
                short_id=take_first(testcase.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual=False,
                    qual=False,
                )),

                full_id=take_first(testcase.get_id(
                    hidden_callable_set=hidden_callable_set if not verbose else None,
                    full_qual=True,
                )),
                folder=testcase.data['testcase_artifact_dir'],
                uuid_=testcase.uuid
        ).replace('\n', '\n# ')

        delim = '#' * (len(exec_start_msg.splitlines()[0]) + 2)
        out(delim + '\n# ' + exec_start_msg + '\n' + delim)

        result_list = list()
        result_map[testcase] = result_list

        def pre_line():
            out('-' * 40)
        # Make sure that all the output of the expression is flushed to ensure
        # there won't be any buffered stderr output being displayed after the
        # "official" end of the Expression's execution.
        def flush_std_streams():
            sys.stdout.flush()
            sys.stderr.flush()

        def get_uuid_str(expr_val):
            uuid_val = (expr_val.value_uuid or expr_val.excep_uuid)
            if uuid_val:
                return ' UUID={}'.format(uuid_val)
            else:
                return ''

        def log_expr_val(expr_val, reused):
            if expr_val.expr.op.callable_ in hidden_callable_set:
                return

            if reused:
                msg='Reusing already computed {id}{uuid}'
            else:
                msg='Computed {id}{uuid}'

            info(msg.format(
                id=expr_val.get_id(
                    full_qual=False,
                    with_tags=True,
                    hidden_callable_set=hidden_callable_set,
                ),
                uuid = get_uuid_str(expr_val),
            ))

        executor = executor(log_expr_val)

        out('')
        for result in utils.iterate_cb(executor, pre_line, flush_std_streams):
            for failed_val in result.get_failed_expr_vals():
                excep = failed_val.excep
                tb = utils.format_exception(excep)
                error('Error ({e_name}): {e}\nID: {id}\n{tb}'.format(
                        id=failed_val.get_id(),
                        e_name = utils.get_name(type(excep)),
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
        testcase_artifact_dir = testcase.data['testcase_artifact_dir']

        # Finalize the computation
        adaptor.finalize_expr(testcase)

        # Dump the reproducer script
        with (testcase_artifact_dir/'TESTCASE.py').open('wt', encoding='utf-8') as f:
            f.write(
                testcase.get_script(
                    prefix = 'testcase',
                    db_path = os.path.join('..', '..', DB_FILENAME),
                    db_relative_to = '__file__',
                    db_loader=db_loader
                )[1]+'\n',
            )

        with (testcase_artifact_dir/'VALUES_UUID').open('wt') as f:
            for expr_val in result_list:
                if expr_val.value is not NoValue:
                    f.write(expr_val.value_uuid + '\n')

                if expr_val.excep is not NoValue:
                    f.write(expr_val.excep_uuid + '\n')

    db = engine.ValueDB(
        engine.Expression.get_all_serializable_vals(
            testcase_list, hidden_callable_set,
        )
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
        testcase_list, prefix='testcase',
        db_path=db_path.relative_to(artifact_dir),
        db_relative_to='__file__',
        db=db,
        db_loader=db_loader,
    )

    with script_path.open('wt', encoding='utf-8') as f:
        f.write(all_scripts + '\n')

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
