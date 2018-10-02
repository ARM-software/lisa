#! /usr/bin/env python3

from pprint import pprint
import argparse
import collections
import copy
import datetime
import inspect
import io
import os
import pathlib
import sys
import traceback
import uuid
import logging
import traceback
import gzip
import fnmatch
import functools
import itertools
import importlib
import contextlib

from exekall.customization import AdaptorBase
import exekall.engine as engine
from exekall.engine import take_first, NoValue
import exekall.utils as utils
from exekall.utils import error, warn, debug, info, out

def _main(argv):
    parser = argparse.ArgumentParser(description="""
    LISA test runner
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    run_parser = subparsers.add_parser('run',
    description="""
    Run the tests
    """,
    formatter_class=argparse.RawTextHelpFormatter)

    run_parser.add_argument('python_files', nargs='+',
        metavar='PYTHON_SRC',
        help="Python modules files")

    run_parser.add_argument('--adaptor',
        help="""Adaptor to use from the customization module, if there is more
than one to choose from.""")

    run_parser.add_argument('--filter',
        help="""Only run the testcases with an ID matching the filter.""")

    run_parser.add_argument('--restrict', action='append',
        default=[],
        help="""Callable names patterns. Types produced by these callables will
only be produced by these (other callables will be excluded).""")

    run_parser.add_argument('--modules-root', action='append', default=[],
        help="Equivalent to setting PYTHONPATH")

    artifact_dir_group = run_parser.add_mutually_exclusive_group()
    artifact_dir_group.add_argument('--artifact-root',
        default=os.getenv('EXEKALL_ARTIFACT_ROOT', 'artifacts'),
        help="Root folder under which the artifact folders will be created")

    artifact_dir_group.add_argument('--artifact-dir',
        default=os.getenv('EXEKALL_ARTIFACT_DIR'),
        help="""Folder in which the artifacts will be stored. This take
precedence over --artifact-root""")

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

    run_parser.add_argument('--goal', default='*ResultBundle',
        help="""Compute expressions leading to an instance of the specified
class or a subclass of it.""")

    run_parser.add_argument('--sweep', nargs=5, action='append', default=[],
        metavar=('CALLABLE', 'PARAM', 'START', 'STOP', 'STEP'),
        help="""Parametric sweep on a function parameter.
It needs five fields: the qualified name of the callable (pattern can be used),
the name of the parameter, the start value, stop value and step size.""")

    run_parser.add_argument('--verbose', action='store_true',
        help="""More verbose output.""")

    run_parser.add_argument('--dry-run', action='store_true',
        help="""Only show the tests that will be run without running them.""")

    run_parser.add_argument('--template-scripts', metavar='SCRIPT_FOLDER',
        help="""Only create the template scripts of the tests without running them.""")

    run_parser.add_argument('--log-level', default='info',
        choices=('debug', 'info', 'warn', 'error', 'critical'),
        help="""Change the default log level of the standard logging module.""")

    run_parser.add_argument('--debug', action='store_true',
        help="""Show complete Python backtrace when exekall crashes.""")


    args = argparse.Namespace()
    # Avoid showing help message on the incomplete parser. Instead, we carry on
    # and the help will be displayed after the parser customization has a
    # chance to take place.
    help_options =  ('-h', '--help')
    no_help_argv = [
        arg for arg in argv
        if arg not in help_options
    ]
    try:
        # Silence argparse until we know what is going on
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            args, _ = parser.parse_known_args(no_help_argv, args)
    # If it fails, that may be because of an incomplete command line with just
    # --help for example. If it was for another reason, it will fail again and
    # show the message.
    except SystemExit:
        args, _ = parser.parse_known_args(argv, args)

    if not args.subcommand == 'run':
        # Show the help messages
        parser.parse_args(argv)
        return 0

    global show_traceback
    show_traceback = args.debug

    # Look for a customization submodule in one of the toplevel packages
    # of the modules we specified on the command line.
    module_list = [utils.import_file(path) for path in args.python_files]
    package_names_list = [
        module.__name__.split('.')
        for module in reversed(module_list)
    ]

    def build_full_names(l_l):
        """Explode list of lists, and build full package names."""
        for l in l_l:
            for i, _ in enumerate(l):
                i += 1
                yield '.'.join(l[:i])

    package_name_list = list(build_full_names(package_names_list))

    adaptor_cls = AdaptorBase
    module_set = set()
    for name in reversed(package_name_list):
        customize_name = name + '.exekall_customize'
        # Only hide ModuleNotFoundError exceptions when looking up that
        # specific module, we don't want to hide issues inside the module
        # itself.
        module_exists = False
        with contextlib.suppress(ModuleNotFoundError):
            module_exists = importlib.util.find_spec(customize_name)

        if module_exists:
            # Importing that module is enough to make the adaptor visible
            # to the Adaptor base class
            customize_module = importlib.import_module(customize_name)
            module_set.add(customize_module)
            break

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
    utils.setup_logging(args.log_level, verbose)

    adaptor = adaptor_cls(args)

    dry_run = args.dry_run
    only_template_scripts = args.template_scripts

    goal_pattern = args.goal

    load_db_path = args.load_db
    load_db_pattern_list = args.load_type
    load_db_uuid_list = args.load_uuid
    load_db_uuid_args = args.load_uuid_args

    user_filter = args.filter
    restrict_list = args.restrict

    sys.path.extend(args.modules_root)

    module_set.update(utils.import_file(path) for path in args.python_files)

    # Pool of all callable considered
    callable_pool = utils.get_callable_set(module_set)
    callable_pool = adaptor.filter_callable_pool(callable_pool)

    op_pool = {engine.Operator(callable_) for callable_ in callable_pool}

    # Force some parameter values to be provided with a specific callable
    patch_map = dict()
    for sweep_spec in args.sweep:
        number_type = float
        callable_pattern, param, start, stop, step = sweep_spec
        for callable_ in callable_pool:
            callable_name = engine.get_name(callable_)
            if not fnmatch.fnmatch(callable_name, callable_pattern):
                continue
            patch_map.setdefault(callable_name, dict())[param] = [
                i for i in utils.sweep_number(
                    callable_, param,
                    number_type(start), number_type(stop), number_type(step)
                )
            ]

    only_prebuilt_cls = set()

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
                    for param_serial in serial.param_value_map.values()
                ]

                serial_res_set.update(
                        utils.load_serial_from_db(db, param_uuid_list,
                        load_db_pattern_list
                ))

        # Otherwise, reload all the root serial values
        else:
            serial_res_set.update(
                frozenset(l)
                for l in db.obj_store.serial_seq_list
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
                id_ = serial_list[0].get_id(full_qual=False, with_tags=True)
                prebuilt_op_pool_list.append(
                    engine.PrebuiltOperator(
                        type_, serial_list, id_=id_
                    ))
                only_prebuilt_cls.add(type_)

    only_prebuilt_cls.discard(type(NoValue))

    for op_name, param_patch_map in patch_map.items():
        for op in op_pool:
            if op.name == op_name:
                try:
                    new_op_pool = op.force_param(param_patch_map)
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
    op_pool = sorted(op_pool, key=lambda x: x.name)

    # Pool of classes that can be produced by the ops
    produced_pool = set(op.value_type for op in op_pool)

    # Set of all types that can be depended upon. All base class of types that
    # are actually produced are also part of this set, since they can be
    # dependended upon as well.
    cls_set = set()
    for produced in produced_pool:
        cls_set.update(inspect.getmro(produced))
    cls_set.discard(object)

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

    cls_map = adaptor.filter_cls_map(cls_map)

    # Augment the list of classes that can only be provided by a prebuilt
    # Operator with all the compatible classes
    only_prebuilt_cls_ = set()
    for cls in only_prebuilt_cls:
        only_prebuilt_cls_.update(cls_map[cls])
    only_prebuilt_cls = only_prebuilt_cls_

    # Map of all produced types to a set of what operator can create them
    op_map = dict()
    for op in op_pool:
        param_map, produced = op.get_prototype()
        if not (
            # Some types may only be produced by prebuilt operators
            produced in only_prebuilt_cls and
            not isinstance(op, engine.PrebuiltOperator)
        ):
            op_map.setdefault(produced, set()).add(op)
    op_map = adaptor.filter_op_map(op_map)

    # Restrict the production of some types to a set of operators.
    restricted_op_set = {
        op for op in op_pool
        if any(
            fnmatch.fnmatch(op.get_name(full_qual=True), pattern)
            for pattern in restrict_list
        )
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

    # Get the list of root operators
    root_op_set = set()
    for produced, op_set in op_map.items():
        # All producers of Result can be a root operator in the expressions
        # we are going to build, i.e. the outermost function call
        if utils.match_base_cls(produced, goal_pattern):
            root_op_set.update(op_set)

    # Sort for stable output
    root_op_list = sorted(root_op_set, key=lambda op: op.name)

    # Some operators are hidden in IDs since they don't add useful information
    # (internal classes)
    hidden_callable_set = adaptor.get_hidden_callable_set(op_map)

    # Only print once per parameters' tuple
    @utils.once
    def handle_non_produced(cls_name, consumer_name, param_name, callable_path):
        info('Nothing can produce instances of {cls} needed for {consumer} (parameter "{param}", along path {path})'.format(
            cls = cls_name,
            consumer = consumer_name,
            param = param_name,
            path = ' -> '.join(engine.get_name(callable_) for callable_ in callable_path)
        ))

    @utils.once
    def handle_cycle(path):
        error('Cyclic dependency detected: {path}'.format(
            path = ' -> '.join(
                engine.get_name(callable_)
                for callable_ in path
            )
        ))

    # Build the list of Expression that can be constructed from the set of
    # callables
    testcase_list = list(engine.ExpressionWrapper.build_expr_list(
        root_op_list, op_map, cls_map,
        non_produced_handler = handle_non_produced,
        cycle_handler = handle_cycle
    ))

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
            if fnmatch.fnmatch(take_first(testcase.get_id(
                # These options need to match what --dry-run gives
                full_qual=False,
                hidden_callable_set=hidden_callable_set)), user_filter)
        ]

    if not testcase_list:
        info('Nothing to do, exiting ...')
        return 0

    date = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
    testsession_uuid = engine.create_uuid()

    if only_template_scripts:
        artifact_dir = pathlib.Path(only_template_scripts)
    elif args.artifact_dir:
        artifact_dir = pathlib.Path(args.artifact_dir)
    # If we are not given a specific folder, we create one under the root we
    # were given
    else:
        artifact_dir = pathlib.Path(args.artifact_root, date + '_' + testsession_uuid)

    artifact_dir = artifact_dir.resolve()
    # Update the CLI arguments so the customization module has access to the
    # correct value
    args.artifact_dir = artifact_dir

    out('The following expressions will be executed:\n')
    for testcase in testcase_list:
        out(take_first(testcase.get_id(
            full_qual = verbose,
            hidden_callable_set=hidden_callable_set
        )))
        if verbose:
            out(testcase.pretty_structure() + '\n')

    if dry_run:
        return 0

    artifact_dir.mkdir(parents=True)
    if not only_template_scripts:
        with open(str(artifact_dir.joinpath('UUID')), 'wt') as f:
            f.write(testsession_uuid+'\n')

    db_loader = adaptor.get_db_loader()

    for testcase in testcase_list:
        testcase_short_id = take_first(testcase.get_id(
            hidden_callable_set=hidden_callable_set,
            with_tags=False,
            full_qual = False
        ))
        testcase_id = take_first(testcase.get_id(
            hidden_callable_set=hidden_callable_set,
            with_tags=False,
            full_qual=True,
        ))

        testcase_short_id = take_first(testcase.get_id(
            hidden_callable_set=hidden_callable_set,
            with_tags=False,
            full_qual=False,
        ))
        data = testcase.data
        data['id'] = testcase_id
        data['uuid'] = testcase.uuid

        testcase_artifact_dir = pathlib.Path(
            artifact_dir,
            testcase.op.get_name(full_qual=False),
            testcase_short_id,
            testcase.uuid
        ).resolve()
        testcase_artifact_dir.mkdir(parents=True)
        data['artifact_dir'] = artifact_dir
        data['testcase_artifact_dir'] = testcase_artifact_dir

        with open(str(testcase_artifact_dir.joinpath('ID')), 'wt') as f:
            f.write(testcase_id+'\n')

        with open(
            str(testcase_artifact_dir.joinpath('testcase_template.py')),
            'wt', encoding='utf-8'
        ) as f:
            f.write(
                testcase.get_script(
                    prefix = 'testcase',
                    db_path = '../../storage.yml.gz',
                    db_relative_to = '__file__',
                    db_loader = db_loader
                )[1]+'\n',
            )

    if only_template_scripts:
        return 0

    out('\n')
    result_map = collections.defaultdict(list)
    for testcase, executor in engine.Expression.get_executor_map(testcase_list).items():
        exec_start_msg = 'Executing: {short_id}\n\nFull ID: {full_id}\nArtifacts: {folder}'.format(
                short_id=take_first(testcase.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual = False
                )),

                full_id=take_first(testcase.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual = True
                )),
                folder=testcase.data['testcase_artifact_dir']
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

        def log_expr_val(expr_val, reused):
            if expr_val.expr.op.callable_ in hidden_callable_set:
                return

            if reused:
                msg='Reusing already computed {id} UUID={uuid}'
            else:
                msg='Computed {id} UUID={uuid}'
            info(msg.format(
                id=expr_val.get_id(
                    full_qual=False,
                    with_tags=True,
                    hidden_callable_set=hidden_callable_set,
                ),
                uuid = expr_val.value_uuid
            ))

        executor = executor(log_expr_val)

        out('')
        for result in utils.iterate_cb(executor, pre_line, flush_std_streams):
            for failed_val in result.get_failed_values():
                excep = failed_val.excep
                tb = utils.format_exception(excep)
                error('Error ({e_name}): {e}\nID: {id}\n{tb}'.format(
                        id=failed_val.get_id(),
                        e_name = engine.get_name(type(excep)),
                        e=excep,
                        tb=tb,
                    ),
                )

            uuid_ = None
            if result.value is NoValue:
                uuid_ = result.excep_uuid
                uuid_name = 'Error UUID'
            else:
                uuid_ = result.value_uuid
                uuid_name = 'Result UUID'

            uuid_str = '' if uuid_ is None else '\n' + uuid_name + ': ' + uuid_
            info('Finished: {short_id}'.format(
                short_id = result.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual=False,
                ),
                full_id = result.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual=True,
                )
            ))
            prefix = 'Full ID: '
            #TODO: turn these prints into logs output, so they end up on the
            # same stderr or file as the logging output
            out('{prefix}{id}{uuid_str}'.format(
                id=result.get_id(
                    mark_excep=True,
                    hidden_callable_set=hidden_callable_set
                ).strip().replace('\n', '\n'+len(prefix)*' '),
                prefix=prefix,
                uuid_str = uuid_str
            ))
            out(adaptor.result_str(result))
            result_list.append(result)


        out('')
        testcase_artifact_dir = testcase.data['testcase_artifact_dir']

        # Finalize the computation
        adaptor.finalize_expr(testcase)

        # Dump the reproducer script
        with open(
            str(testcase_artifact_dir.joinpath('testcase.py')),
            'wt', encoding='utf-8'
        ) as f:
            f.write(
                testcase.get_script(
                    prefix = 'testcase',
                    db_path = '../../../storage.yml.gz',
                    db_relative_to = '__file__',
                    db_loader = db_loader
                )[1]+'\n',
            )


        with open(str(testcase_artifact_dir.joinpath('UUID')), 'wt') as f:
            for expr_val in result_list:
                if expr_val.value is not NoValue:
                    f.write(expr_val.value_uuid + '\n')

                if expr_val.excep is not NoValue:
                    f.write(expr_val.excep_uuid + '\n')

    obj_store = engine.ObjectStore(
        engine.Expression.get_all_serializable_values(
            testcase_list, hidden_callable_set,
        )
    )
    db = engine.StorageDB(obj_store)

    db_path = artifact_dir.joinpath('storage.yml.gz')
    db.to_path(db_path)

    out('#'*80)
    info('Result summary:\n')

    # Display the results
    adaptor.process_results(result_map)

    # Output the merged script with all subscripts
    script_path = artifact_dir.joinpath('all_scripts.py')
    result_name_map, all_scripts = engine.Expression.get_all_script(
        testcase_list, prefix='testcase',
        db_path=db_path.relative_to(artifact_dir),
        db_relative_to='__file__',
        obj_store=obj_store,
        db_loader=adaptor.get_db_loader()
    )

    with open(str(script_path), 'wt', encoding='utf-8') as f:
        f.write(all_scripts+'\n')

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
