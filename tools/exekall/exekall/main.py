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
from exekall.utils import error, warn, debug, info

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

    run_parser.add_argument('--filter',
        help="""Only run the testcases with an ID matching the filter.""")

    run_parser.add_argument('--modules-root', action='append', default=[],
        help="Equivalent to setting PYTHONPATH")

    run_parser.add_argument('--result-root', default='results',
        help="Folder in which the test artifacts will be stored")

    run_parser.add_argument('--load-db',
        help="""Reload a database and use its results as prebuilt objects.""")

    run_parser.add_argument('--load-type', action='append', default=[],
        help="""Load the (indirect) instances of the given class from the
database instead of the root objects.""")

    run_parser.add_argument('--load-uuid',
        help="""Load the given UUID from the database. If it is the UUID of an
error, the parameters of the callable that errored will be reloaded.
What is reloaded can be refined with --load-type.""")

    run_parser.add_argument('--goal', default='*ResultBundle',
        help="""Compute expressions leading to an instance of the specified
class or a subclass of it.""")

    run_parser.add_argument('--sweep', nargs=5, action='append', default=[],
        metavar=('CALLABLE', 'PARAM', 'START', 'STOP', 'STEP'),
        help="""Parametric sweep on a function parameter.
It needs five fields: the qualified name of the callable, the name of
the parameter, the start value, stop value and step size.""")

    run_parser.add_argument('--verbose', action='store_true',
        help="""More verbose output.""")

    run_parser.add_argument('--dry-run', action='store_true',
        help="""Only show the tests that will be run without running them.""")

    run_parser.add_argument('--template-scripts', metavar='SCRIPT_FOLDER',
        help="""Only create the template scripts of the tests without running them.""")

    run_parser.add_argument('--log-level', default='info',
        choices=('debug', 'info', 'warn', 'error', 'critical'),
        help="""Change the default log level of the standard logging module.""")


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

    # Look for a customization submodule in one of the toplevel packages
    # of the modules we specified on the command line.
    module_list = [utils.import_file(path) for path in args.python_files]
    toplevel_package_name_list = [
        module.__name__.split('.', 1)[0]
        for module in module_list
    ]

    adaptor_cls = AdaptorBase
    module_set = set()
    for name in toplevel_package_name_list:
        customize_name = name + '.exekall_customize'
        # If the module exists, we try to import it
        if importlib.util.find_spec(customize_name):
            # Importing that module is enough to make the adaptor visible
            # to the Adaptor base class
            customize_module = importlib.import_module(customize_name)
            module_set.add(customize_module)

        # TODO: Allow listing adapators and choosing the one we want
        adaptor_cls = AdaptorBase.get_adaptor_cls()
        # Add all the CLI arguments of the adaptor before reparsing the
        # command line.
        adaptor_cls.register_cli_param(run_parser)
        break

    # Reparse the command line after the adaptor had a chance to add its own
    # arguments.
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='[%(name)s][%(asctime)s] %(levelname)s  %(message)s',
        stream=sys.stdout,
    )

    adaptor = adaptor_cls(args)

    verbose = args.verbose
    dry_run = args.dry_run
    only_template_scripts = args.template_scripts

    result_root = pathlib.Path(args.result_root)

    goal_pattern = args.goal

    load_db_path = args.load_db
    load_db_pattern_list = args.load_type
    load_db_uuid_list = [args.load_uuid] if args.load_uuid is not None else []

    user_filter = args.filter

    sys.path.extend(args.modules_root)

    module_set.update(utils.import_file(path) for path in args.python_files)

    # Pool of all callable considered
    callable_pool = set()
    for module in module_set:
        callable_pool.update(utils.get_callable_set(module))
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

    # Load objects from an existing database
    if load_db_path:
        db = adaptor.load_db(load_db_path)
        if load_db_pattern_list or load_db_uuid_list:
            serial_res_set = set()
            if not load_db_uuid_list:
                for load_db_pattern in load_db_pattern_list:
                    # When we reload instances of a class from the DB, we don't
                    # want anything else to be able to produce it, since we want to
                    # run on that existing data set
                    def predicate(serial):
                        return utils.match_base_cls(
                            type(serial.value), load_db_pattern
                        )
                    serial_res_set_ = db.obj_store.get_by_predicate(predicate)
                    serial_res_set.update(serial_res_set_)
                    if not serial_res_set_:
                        raise ValueError('No result of type matching "{pattern}" could be found in the database'.format(
                            pattern=load_db_pattern
                        ))

            for load_db_uuid in load_db_uuid_list:
                assert load_db_uuid is not None
                def predicate(serial):
                    return (
                        serial.value_uuid == load_db_uuid or
                        serial.excep_uuid == load_db_uuid
                    )
                serial_res_set_ = db.obj_store.get_by_predicate(predicate)
                serial_res_set.update(serial_res_set_)
                if not serial_res_set_:
                    raise ValueError('No value with UUID "{uuid}" could be found in the database'.format(
                        uuid=load_db_uuid
                    ))
        else:
            serial_res_set = set(
                frozenset(l)
                for l in db.obj_store.serial_seq_list
            )

        # Remove duplicates accross sets
        loaded_serial = set()
        serial_res_set_ = set()
        for serial_res in serial_res_set:
            serial_res = frozenset(serial_res - loaded_serial)
            if serial_res:
                loaded_serial.update(serial_res)
                serial_res_set_.add(serial_res)
        serial_res_set = serial_res_set_

        # Build the list of PrebuiltOperator that will inject the loaded values
        # into the tests
        prebuilt_op_pool_list = list()
        for serial_res in serial_res_set:

            for serial in serial_res:
                # No value and no excep, this UUID was never calculated
                # because of a failed parent.
                if serial.value is NoValue and serial.excep is NoValue:
                    continue

                # If we end up with a NoValue here, that means we matched the
                # UUID of a failed value, so we load the parameters that lead
                # to it instead.
                #TODO: we should only allow one of these, otherwise we will
                # mixup the parameter values.
                if serial.value is NoValue:
                    for param_serial in serial.param_value_map.values():
                        if param_serial.value is NoValue:
                            continue

                        # Make sure we only get the types we care about
                        if load_db_pattern_list and not any(
                            utils.match_base_cls(
                                type(param_serial.value), load_db_pattern
                            ) for load_db_pattern in load_db_pattern_list
                        ):
                            continue

                        type_ = type(param_serial.value)
                        id_ = param_serial.simplified_qual_id
                        prebuilt_op_pool_list.append(engine.PrebuiltOperator(
                            type_, [param_serial], id_=id_
                        ))
                        only_prebuilt_cls.add(type_)

            serial_list = [
                serial for serial in serial_res
                if serial.value is not NoValue
            ]
            serial = take_first(serial_list)
            if serial is not NoValue:
                type_ = type(serial.value)
                # This provides a middle-ground: fully qualified names but some
                # callables are hidden
                id_ = serial.simplified_qual_id
                prebuilt_op_pool_list.append(engine.PrebuiltOperator(
                    type_, serial_list, id_=id_
                ))
                only_prebuilt_cls.add(type_)
    else:
        prebuilt_op_pool_list = adaptor.get_prebuilt_list()

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

    testcase_list = list(engine.ExpressionWrapper.build_expr_list(
        root_op_list, op_map, cls_map,
        non_produced_handler = handle_non_produced,
        cycle_handler = handle_cycle
    ))

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
        artifact_root = pathlib.Path(only_template_scripts)
    else:
        artifact_root = pathlib.Path(result_root, date + '_' + testsession_uuid)
    artifact_root = artifact_root.resolve()

    for testcase in testcase_list:
        print(take_first(testcase.get_id(
            full_qual = verbose,
            hidden_callable_set=hidden_callable_set
        )))
        if verbose:
            print(testcase.pretty_structure() + '\n')

    if dry_run:
        return 0

    artifact_root.mkdir(parents=True)
    if not only_template_scripts:
        with open(str(artifact_root.joinpath('UUID')), 'wt') as f:
            f.write(testsession_uuid+'\n')


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

        testcase_artifact_root = pathlib.Path(
            artifact_root,
            testcase.op.get_name(full_qual=False),
            testcase_short_id,
            testcase.uuid
        ).resolve()
        testcase_artifact_root.mkdir(parents=True)
        data['artifact_root'] = artifact_root
        data['testcase_artifact_root'] = testcase_artifact_root

        with open(str(testcase_artifact_root.joinpath('ID')), 'wt') as f:
            f.write(testcase_id+'\n')

        with open(
            str(testcase_artifact_root.joinpath('testcase_template.py')),
            'wt', encoding='utf-8'
        ) as f:
            f.write(
                testcase.get_script(
                    prefix = 'testcase',
                    db_path = '../../storage.yml.gz',
                    db_relative_to = '__file__'
                )[1]+'\n',
            )

    if only_template_scripts:
        return 0

    print('#'*80)
    result_map = collections.defaultdict(list)
    for testcase, executor in engine.Expression.get_executor_map(testcase_list).items():
        exec_start_msg = 'Executing: {short_id}\n\nID: {full_id}\nArtifacts: {folder}'.format(
                short_id=take_first(testcase.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual = False
                )),

                full_id=take_first(testcase.get_id(
                    hidden_callable_set=hidden_callable_set,
                    full_qual = True
                )),
                folder=testcase.data['testcase_artifact_root']
        ).replace('\n', '\n# ')

        delim = '#' * (len(exec_start_msg.splitlines()[0]) + 2)
        print(delim + '\n# ' + exec_start_msg + '\n' + delim)

        result_list = list()
        result_map[testcase] = result_list

        pre_line = lambda: print('-' * 40)
        post_line = lambda: print()
        for result in utils.iterate_cb(executor(), pre_line, post_line):
            for failed_val in result.get_failed_values():
                excep = failed_val.excep
                tb_list = traceback.format_exception(type(excep), excep, excep.__traceback__)
                tb =  ''.join(tb_list)
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
            prefix = 'ID: '
            print('{prefix}{id}{uuid_str}'.format(
                id=result.get_id(
                    mark_excep=True,
                    hidden_callable_set=hidden_callable_set
                ).strip().replace('\n', '\n'+len(prefix)*' '),
                prefix=prefix,
                uuid_str = uuid_str
            ))
            print(adaptor.result_str(result))
            result_list.append(result)


        print('\n')
        testcase_artifact_root = testcase.data['testcase_artifact_root']

        # Finalize the computation
        adaptor.finalize_expr(testcase)

        # Dump the reproducer script
        with open(
            str(testcase_artifact_root.joinpath('testcase.py')),
            'wt', encoding='utf-8'
        ) as f:
            f.write(
                testcase.get_script(
                    prefix = 'testcase',
                    db_path = '../../../storage.yml.gz',
                    db_relative_to = '__file__'
                )[1]+'\n',
            )


        with open(str(testcase_artifact_root.joinpath('UUID')), 'wt') as f:
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

    db_path = artifact_root.joinpath('storage.yml.gz')
    db.to_path(db_path)

    print('#'*80)
    info('Result summary:\n')

    # Display the results
    adaptor.process_results(result_map)

    # Output the merged script with all subscripts
    script_path = artifact_root.joinpath('all_scripts.py')
    result_name_map, all_scripts = engine.Expression.get_all_script(
        testcase_list, prefix='testcase',
        db_path=db_path.relative_to(artifact_root),
        db_relative_to='__file__',
    )

    with open(str(script_path), 'wt', encoding='utf-8') as f:
        f.write(all_scripts+'\n')

SILENT_EXCEPTIONS = (KeyboardInterrupt, BrokenPipeError)
GENERIC_ERROR_CODE = 1

def main():
    show_traceback = True
    return_code = 0

    try:
        return_code = _main(argv=sys.argv[1:])
    # Quietly exit for these exceptions
    except SILENT_EXCEPTIONS:
        pass
    except SystemExit as e:
        return_code = e.code
    # Catch-all
    except Exception as e:
        #TODO: remove that
        raise
        if show_traceback:
            error(
                'Exception traceback:\n' +
                ''.join(
                traceback.format_exception(type(e), e, e.__traceback__)
            ))
        # Always show the concise message
        error(e)
        return_code = GENERIC_ERROR_CODE

    sys.exit(return_code)

if __name__ == '__main__':
    main()
