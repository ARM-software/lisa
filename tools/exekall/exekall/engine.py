#! /usr/bin/env python3

import abc
import inspect
import collections
from collections import OrderedDict
import copy
import itertools
import numbers
import functools
import traceback
import uuid
import io
import os
import pickle
import datetime
import io
import gzip
import pathlib
import contextlib
import types
import pprint
import sys

import ruamel.yaml

yaml = ruamel.yaml.YAML(typ='unsafe')
yaml.allow_unicode = True
yaml.default_flow_style = False
yaml.indent = 4

# Basic reimplementation of typing.get_type_hints for Python versions that
# do not have a typing module available, and also avoids creating Optional[]
# when the parameter has a None default value.
def get_type_hints(f, module_vars=None):
    if module_vars is None:
        try:
            module_vars = f.__globals__
        except AttributeError:
            module_vars = dict()

    return resolve_annotations(f.__annotations__, module_vars)

def resolve_annotations(annotations, module_vars):
    return {
        # If we get a string, evaluate it in the global namespace of the
        # module in which the callable was defined
        param: cls if not isinstance(cls, str) else eval(cls, module_vars)
        for param, cls in annotations.items()
    }

def remove_indices(iterable, ignored_indices):
    return [v for i, v in enumerate(iterable) if i not in ignored_indices]

def take_first(iterable):
    for i in iterable:
        return i
    return NoValue

def create_uuid():
    return uuid.uuid4().hex


class NoOperatorError(Exception):
    pass

class _NoValueType:
    # Use a singleton pattern to make sure that even deserialized instances
    # will be the same object
    def __new__(cls):
        try:
            return cls._instance
        except AttributeError:
            obj = super().__new__(cls)
            cls._instance = obj
            return obj

    def __bool__(self):
        return False

    def __repr__(self):
        return 'NoValue'

    def __eq__(self, other):
        return type(self) is type(other)

NoValue = _NoValueType()

class IndentationManager:
    def __init__(self, style):
        self.style = style
        self.level = 0

    def indent(self):
        self.level += 1

    def dedent(self):
        self.level -= 1

    def __str__(self):
        return str(self.style) * self.level

class StorageDB:

    def __init__(self, obj_store):
        self.obj_store = obj_store

    @classmethod
    def from_path(cls, path, relative_to=None):
        if relative_to is not None:
            relative_to = pathlib.Path(relative_to).resolve()
            if not relative_to.is_dir():
                relative_to = pathlib.Path(relative_to).parent
            path = pathlib.Path(relative_to, path)

        with gzip.open(str(path), 'rt', encoding='utf-8') as f:
            db = yaml.load(f)
        assert isinstance(db, cls)

        return db

    def to_path(self, path):
        with gzip.open(str(path), 'wt', encoding='utf-8') as f:
            yaml.dump(self, f)

    # Having it there shortens the output of the generated scripts and makes
    # them more readable while avoiding to expose to much of the StorageDB
    # internals
    def by_uuid(self, *args, **kwargs):
        return self.obj_store.by_uuid(*args, **kwargs)

class ObjectStore:
    def __init__(self, serial_seq_list, db_var_name='db'):
        self.db_var_name = db_var_name
        self.serial_seq_list = serial_seq_list

    def get_value_snippet(self, value):
        _, id_uuid_map = self.get_indexes()
        return '{db}.by_uuid({key})'.format(
            db = self.db_var_name,
            key = repr(id_uuid_map[id(value)])
        )

    def by_uuid(self, uuid):
        uuid_value_map, _ = self.get_indexes()
        return uuid_value_map[uuid]

    @functools.lru_cache(maxsize=None, typed=True)
    def get_indexes(self):
        uuid_value_map = dict()
        id_uuid_map = dict()

        def update_map(serial_val):
            for uuid_, val in (
                (serial_val.value_uuid, serial_val.value),
                (serial_val.excep_uuid, serial_val.excep),
            ):
                uuid_value_map[uuid_] = val
                id_uuid_map[id(val)] = uuid_

        self._serial_val_dfs(update_map)

        return (uuid_value_map, id_uuid_map)

    def _serial_val_dfs(self, callback):
        for serial_seq in self.serial_seq_list:
            for serial_val in serial_seq:
                self._do_serial_val_dfs(serial_val, callback)

    def _do_serial_val_dfs(cls, serial_val, callback):
        callback(serial_val)
        for serial_val in serial_val.param_value_map.values():
            cls._do_serial_val_dfs(serial_val, callback)

    def get_all(self):
        serial_seq_set = self.get_by_predicate(lambda serial: True)
        all_set = set()
        for serial_seq in serial_seq_set:
            all_set.update(serial_seq)
        return all_set

    def get_by_predicate(self, predicate):
        """
        Return a set of sets, containing objects matching the predicate.
        There is a set for each computed expression in the store, but the same
        object will not be included twice (in case it is refered by different
        expressions).
        """
        serial_seq_set = set()

        # When we reload instances of a class from the DB, we don't
        # want anything else to be able to produce it, since we want to
        # run on that existing data set

        for serial_seq in self.serial_seq_list:
            serial_set = set()
            for serial in serial_seq:
                serial_set.update(serial.get_parent_set(predicate))

            serial_seq_set.add(frozenset(serial_set))

        return serial_seq_set

class CycleError(Exception):
    pass

class IgnoredCycleError(CycleError):
    pass

class ExpressionWrapper:
    def __init__(self, expr):
        self.expr = expr

    def __getattr__(self, attr):
        return getattr(self.expr, attr)

    @classmethod
    def build_expr_list(cls, result_op_seq, op_map, cls_map,
            non_produced_handler='raise', cycle_handler='raise'):
        op_map = copy.copy(op_map)
        cls_map = copy.copy(cls_map)
        for internal_cls in (Consumer, ExprData):
            op_map[internal_cls] = {Operator(internal_cls)}
            cls_map[internal_cls] = [internal_cls]

        expr_list = list()
        for result_op in result_op_seq:
            # We just skip over Expression where a CycleError happened
            with contextlib.suppress(IgnoredCycleError):
                expr_gen = cls._build_expr(result_op, op_map, cls_map,
                    op_stack = [],
                    non_produced_handler=non_produced_handler,
                    cycle_handler=cycle_handler
                )
                for expr in expr_gen:
                    if expr.validate_expr(op_map):
                        expr_list.append(expr)

        return expr_list

    @classmethod
    def _build_expr(cls, op, op_map, cls_map, op_stack, non_produced_handler, cycle_handler):
        new_op_stack = [op] + op_stack
        # We detected a cyclic dependency
        if op in op_stack:
            if cycle_handler == 'ignore':
                return
            elif cycle_handler == 'raise':
                raise CycleError('Cyclic dependency found: {path}'.format(
                    path = ' -> '.join(
                        op.name for op in new_op_stack
                    )
                ))
            else:
                cycle_handler(tuple(op.callable_ for op in new_op_stack))
                raise IgnoredCycleError


        op_stack = new_op_stack

        param_map, produced = op.get_prototype()
        if param_map:
            param_list, cls_list = zip(*param_map.items())
        # When no parameter is needed
        else:
            yield ExpressionWrapper(Expression(op, OrderedDict()))
            return

        # Build all the possible combinations of types suitable as parameters
        cls_combis = [cls_map.get(cls, list()) for cls in cls_list]

        # Only keep the classes for "self" on which the method can be applied
        if op.is_method:
            cls_combis[0] = [
                cls for cls in cls_combis[0]
                # If the method with the same name would resolve to "op", then
                # we keep this class as a candidate for "self", otherwise we
                # discard it
                if getattr(cls, op.callable_.__name__, None) is op.callable_
            ]

        # Check that some produced classes are available for every parameter
        ignored_indices = set()
        for param, wanted_cls, available_cls in zip(param_list, cls_list, cls_combis):
            if not available_cls:
                # If that was an optional parameter, just ignore it without
                # throwing an exception since it has a default value
                if param in op.optional_param:
                    ignored_indices.add(param_list.index(param))
                else:
                    if non_produced_handler == 'ignore':
                        return
                    elif non_produced_handler == 'raise':
                        raise NoOperatorError('No operator can produce instances of {cls} needed for {op} (parameter "{param}" along path {path})'.format(
                            cls = wanted_cls.__qualname__,
                            op = op.name,
                            param = param,
                            path = ' -> '.join(
                                op.name for op in op_stack
                            )
                        ))
                    else:
                        non_produced_handler(wanted_cls.__qualname__, op.name, param,
                            tuple(op.resolved_callable for op in op_stack)
                        )
                        return

        param_list = remove_indices(param_list, ignored_indices)
        cls_combis = remove_indices(cls_combis, ignored_indices)

        param_list_len = len(param_list)

        # For all possible combinations of types
        for cls_combi in itertools.product(*cls_combis):
            cls_combi = list(cls_combi)

            # Some classes may not be produced, but another combination
            # with containing a subclass of it may actually be produced so we can
            # just ignore that one.
            op_combis = [
                op_map[cls] for cls in cls_combi
                if cls in op_map
            ]

            # Build all the possible combinations of operators returning these
            # types
            for op_combi in itertools.product(*op_combis):
                op_combi = list(op_combi)

                # Get all the possible ways of calling these operators
                param_combis = itertools.product(*(
                    cls._build_expr(param_op, op_map, cls_map,
                        op_stack, non_produced_handler, cycle_handler
                    )
                    for param_op in op_combi
                ))

                for param_combi in param_combis:
                    param_map = OrderedDict(zip(param_list, param_combi))

                    # If all parameters can be built, carry on
                    if len(param_map) == param_list_len:
                        yield ExpressionWrapper(
                            Expression(op, param_map)
                        )

class Expression:
    def __init__(self, op, param_map, data=None):
        self.op = op
        # Map of parameters to other Expression
        self.param_map = param_map
        self.data = data if data is not None else dict()
        self.data_uuid = create_uuid()
        self.uuid = create_uuid()

        self.discard_result()

    def validate_expr(self, op_map):
        return True
        expr_map, valid = self._dfs_visit()
        if not valid:
            return False

        # Check that the Expression does not involve 2 classes that are compatible
        cls_bags = [set(cls_list) for cls_list in op_map.values()]
        cls_used = set(expr_map.keys())
        for cls1, cls2 in itertools.product(cls_used, repeat=2):
            for cls_bag in cls_bags:
                if cls1 in cls_bag and cls2 in cls_bag:
                    return False

        return True

    def _dfs_visit(self):
        expr_map = dict()
        return (expr_map, expr._dfs_visit(expr_map))

    def _do_dfs_visit(self, expr_map):
        value_type = self.op.value_type
        # If there was already an Expression producing that type, the Expression
        # is not valid
        found_callable = expr_map.get(value_type)
        if found_callable is not None and found_callable is not self.op.callable_:
            return False
        expr_map[value_type] = self.op.callable_

        for param_expr in self.param_map.values():
            if not param_expr._do_dfs_visit(expr_map):
                return False
        return True


    def get_param_map(self, reusable):
        reusable = bool(reusable)
        return OrderedDict(
            (param, param_expr)
            for param, param_expr
            in self.param_map.items()
            if bool(param_expr.op.reusable) == reusable
        )

    def get_all_values(self):
        value_list = list()
        for result in self.result_list:
            value_list.extend(result.value_list)

        return value_list

    def find_result_list(self, param_expr_val_map):
        def value_map(expr_value_map):
            return OrderedDict(
                # Extract the actual value from ExprValue
                (param, expr_val.value)
                for param, expr_val in expr_value_map.items()
            )
        param_value_map = value_map(param_expr_val_map)

        # Find the results that are matching the param_expr_val_map
        return [
            result
            for result in self.result_list
            # Check if param_expr_val_map is a subset of the param_value_map
            # of the ExprValue. That allows checking for reusable parameters
            # only.
            if param_value_map.items() <= value_map(result.param_expr_val_map).items()
        ]

    def discard_result(self):
        self.result_list = list()

    def __repr__(self):
        return '<Expression of {self.op.name} at {id}>'.format(
            self=self,
            id = hex(id(self))
        )

    def pretty_structure(self, indent=1):
        indent_str = 4*" " * indent

        if isinstance(self.op, PrebuiltOperator):
            op_name = '<provided>'
            value_type_name = (
                get_name(self.op.value_type, full_qual=True)
                # We just call the operator. It is cheap since it is only
                # returing a pre-built object
                + self._get_value_tag_str(self.op.callable_())
            )
        else:
            op_name = self.op.name
            value_type_name = get_name(self.op.value_type, full_qual=True)

        out = '{op_name} ({value_type_name})'.format(
            op_name = op_name,
            value_type_name = value_type_name,
        )
        if self.param_map:
            out += ':\n'+ indent_str + ('\n'+indent_str).join(
                '{param}: {desc}'.format(param=param, desc=desc.pretty_structure(indent+1))
                for param, desc in self.param_map.items()
            )
        return out

    def get_failed_values(self):
        for expr_val in self.get_all_values():
            yield from expr_val.get_failed_values()

    @staticmethod
    def _get_value_tag_str(value):
        tag_str = ''
        try:
            if value.tags:
                tag_str = '[' + '+'.join(str(v) for v in value.tags) + ']'
        except AttributeError:
            pass
        return tag_str


    def get_id(self, *args, marked_value_set=None, mark_excep=False, hidden_callable_set=None, **kwargs):
        if hidden_callable_set is None:
            hidden_callable_set = set()

        # We always hide the Consumer operator since it does not add anything
        # to the ID. It is mostly an implementation detail.
        hidden_callable_set.update((Consumer, ExprData))

        # Mark all the values that failed to be computed because of an
        # exception
        if mark_excep:
            marked_value_set = set(self.get_failed_values())

        for id_, marker in self._get_id(
                marked_value_set=marked_value_set, hidden_callable_set=hidden_callable_set,
                *args, **kwargs
            ):
            if marked_value_set:
                yield '\n'.join((id_, marker))
            else:
                yield id_

    def _get_id(self, with_tags=True, full_qual=True, expr_val=None, marked_value_set=None, hidden_callable_set=None):
        # When asked about NoValue, it means the caller did not have any value
        # computed for that parameter, but still wants an ID. Obviously, it
        # cannot have any tag since there is no ExprValue available to begin
        # with.
        if expr_val is NoValue:
            with_tags = False

        # No specific value was asked for, so we will cover the IDs of all
        # values
        if expr_val is None or expr_val is NoValue:
            def grouped_value_list():
                # Make sure we yield at least once even if no computed value
                # is available, so _get_id() is called at least once
                if (not self.result_list) or (not with_tags):
                    yield (OrderedDict(), [])
                else:
                    for result in self.result_list:
                        yield (result.param_expr_val_map, result.value_list)

        # If we were asked about the ID of a specific value, make sure we
        # don't explore other paths that lead to different values
        else:
            def grouped_value_list():
                # Only yield the ExprValue we are interested in
                yield (expr_val.param_value_map, [expr_val])

        for param_value_map, value_list in grouped_value_list():
            yield from self._get_id_internal(
                    param_value_map, value_list, with_tags, marked_value_set,
                    hidden_callable_set, full_qual)

    def _get_id_internal(self, param_value_map, value_list, with_tags, marked_value_set, hidden_callable_set, full_qual):
        separator = ':'
        marker_char = '^'

        if marked_value_set is None:
            marked_value_set = set()

        # We only get the ID's of the parameter ExprValue that lead to the
        # ExprValue we are interested in
        param_id_map = OrderedDict(
            (param, take_first(param_expr._get_id(
                with_tags = with_tags,
                full_qual = full_qual,
                # Pass a NoValue when there is no value available, since
                # None means all possible IDs (we just want one here).
                expr_val = param_value_map.get(param, NoValue),
                marked_value_set = marked_value_set,
                hidden_callable_set = hidden_callable_set,
            )))
            for param, param_expr in self.param_map.items()
            if not param_expr.op.callable_ in hidden_callable_set
        )

        get_tag = self._get_value_tag_str if with_tags else lambda v: ''

        def tags_iter(value_list):
            if value_list:
                for expr_val in value_list:
                    tag = get_tag(expr_val.value)
                    yield (expr_val, tag)
            # Yield at least once without any tag even if there is no computed
            # value available
            else:
                yield None, ''

        def get_marker_char(expr_val):
            return marker_char if expr_val in marked_value_set else ' '

        # No parameter to worry about
        if not param_id_map:
            for expr_val, tag_str in tags_iter(value_list):
                id_ = self.op.get_name(full_qual=full_qual) + tag_str
                marker_str = get_marker_char(expr_val) * len(id_)
                yield (id_, marker_str)

        # For all ExprValue we were asked about, we will yield an ID
        else:
            for expr_val, tag_str in tags_iter(value_list):
                # Make a copy to be able to pop items from it
                param_id_map = copy.copy(param_id_map)

                # Extract the first parameter to always use the prefix
                # notation, i.e. its value preceding the ID of the current
                # Expression
                if param_id_map:
                    param, (param_id, param_marker) = param_id_map.popitem(last=False)
                else:
                    param_id = ''
                    param_marker = ''

                if param_id:
                    separator_spacing = ' ' * len(separator)
                    param_str = param_id + separator
                else:
                    separator_spacing = ''
                    param_str = ''

                op_str = '{op}{tags}'.format(
                    op = self.op.get_name(full_qual=full_qual),
                    tags = tag_str,
                )
                id_ = '{param_str}{op_str}'.format(
                    param_str = param_str,
                    op_str = op_str,
                )
                marker_str = '{param_marker}{separator}{op_marker}'.format(
                    param_marker = param_marker,
                    separator = separator_spacing,
                    op_marker = len(op_str) * get_marker_char(expr_val)
                )

                # If there are some remaining parameters, show them in
                # parenthesis at the end of the ID
                if param_id_map:
                    param_str = '(' + ','.join(
                        param + '=' + param_id
                        for param, (param_id, param_marker)
                        # Sort by parameter name to have a stable ID
                        in param_id_map.items()
                        if param_id
                    ) + ')'
                    id_ += param_str
                    param_marker = ' '.join(
                        ' ' * (len(param) + 1) + param_marker
                        for param, (param_id, param_marker)
                        # Sort by parameter name to have a stable ID
                        in param_id_map.items()
                        if param_id
                    ) + ' '

                    marker_str += ' ' + param_marker

                yield (id_, marker_str)

    @classmethod
    def get_all_serializable_values(cls, expr_seq, *args, **kwargs):
        serialized_map = dict()
        result_list = list()
        for expr in expr_seq:
            for result in expr.result_list:
                result_list.append([
                    expr_val._get_serializable(serialized_map, *args, **kwargs)
                    for expr_val in result.value_list
                ])

        return result_list

    def get_script(self, *args, **kwargs):
        return self.get_all_script([self], *args, **kwargs)

    @classmethod
    def get_all_script(cls, expr_list, prefix='value', db_path='storage.yml.gz', db_relative_to=None, db_loader=None, obj_store=None):
        assert expr_list

        if obj_store is None:
            serial_list = Expression.get_all_serializable_values(expr_list)
            obj_store = ObjectStore(serial_list)

        db_var_name = obj_store.db_var_name

        def make_comment(txt):
            joiner = '\n# '
            return joiner + joiner.join(
                line for line in txt.splitlines()
                if line.strip()
            )

        module_name_set = set()
        plain_name_cls_set = set()
        script = ''
        result_name_map = dict()
        for i, expr in enumerate(expr_list):
            script += (
                '#'*80 + '\n# Computed expressions:' +
                ''.join(
                    make_comment(id_)
                    for id_ in expr.get_id(mark_excep=True, full_qual=False)
                ) + '\n' +
                make_comment(expr.pretty_structure()) + '\n\n'
            )
            idt = IndentationManager(' '*4)

            expr_val_set = set(expr.get_all_values())
            result_name, snippet = expr._get_script(
                prefix = prefix + str(i),
                obj_store = obj_store,
                module_name_set = module_name_set,
                idt = idt,
                expr_val_set = expr_val_set,
                consumer_expr_stack = [],
            )

            # If we can expect eval() to work on the representation, we
            # use that
            if pprint.isreadable(expr.data):
                expr_data = pprint.pformat(expr.data)
            else:
                # Otherwise, we try to get it from the DB
                try:
                    expr_data = obj_store.get_value_snippet(expr.data)
                # If the expr_data was not used when computing subexpressions
                # (that may happen if some subrexpressions were already
                # computed for an other expression), we just bail out, hoping
                # that nothing will need EXPR_DATA to be defined. That should
                # not happen often as EXPR_DATA is supposed to stay
                # pretty-printable
                except KeyError:
                    expr_data = '{} # cannot be pretty-printed'

            expr_data_snippet = cls.EXPR_DATA_VAR_NAME + ' = ' + expr_data + '\n'

            script += (
                expr_data_snippet +
                snippet +
                '\n'
            )
            plain_name_cls_set.update(type(x) for x in expr.data.values())

            result_name_map[expr] = result_name


        # Get the name of the customized db_loader
        if db_loader is None:
            db_loader_name = '{cls_name}.from_path'.format(
                cls_name=get_name(StorageDB, full_qual=True),
            )
        else:
            module_name_set.add(inspect.getmodule(db_loader).__name__)
            db_loader_name = get_name(db_loader, full_qual=True)

        # Add all the imports
        header = (
            '#! /usr/bin/env python3\n\n' +
            '\n'.join(
                'import {name}'.format(name=name)
                for name in sorted(module_name_set)
                if name != '__main__'
            ) +
            '\n'
        )

        # Since the __repr__ output of ExprData will usually return snippets
        # assuming the class is directly available by its name, we need to make
        # sure it is imported properly
        for cls_ in plain_name_cls_set:
            mod_name = cls_.__module__
            if mod_name == 'builtins':
                continue
            header += 'from {mod} import {cls}\n'.format(
                cls = cls_.__qualname__,
                mod = mod_name
            )

        header += '\n\n'

        # If there is no ExprValue referenced by that script, we don't need
        # to access any StorageDB
        if expr_val_set:
            if db_relative_to is not None:
                db_relative_to = ', relative_to='+db_relative_to
            else:
                db_relative_to = ''

            header += '{db} = {db_loader_name}({path}{db_relative_to})\n'.format(
                db = db_var_name,
                db_loader_name = db_loader_name,
                path = repr(str(db_path)),
                db_relative_to = db_relative_to
            )

        script = header + '\n' + script
        return (result_name_map, script)

    EXPR_DATA_VAR_NAME = 'EXPR_DATA'

    def _get_script(self, prefix, obj_store, module_name_set, idt, expr_val_set, consumer_expr_stack):
        def make_method_self_name(expr):
            return expr.op.value_type.__name__.replace('.', '')

        def make_var(name):
            # Make sure we don't have clashes between the variable names
            name = name.replace('_', '__')
            name = '_' + name if name else ''
            return prefix + name

        def make_comment(code, idt):
            prefix = idt + '# '
            return prefix + prefix.join(code.splitlines(True)) + '\n'

        def make_serialized(expr_val, attr):
            obj = getattr(expr_val, attr)
            # Try to Pickle the object to see if that raises any exception
            is_serializable(obj, raise_excep=True)

            # When the ExprValue is from an Expression of the Consumer
            # operator, we directly print out the name of the function that was
            # selected since it is not serializable
            callable_ = expr_val.expr.op.callable_
            if attr == 'value' and callable_ is Consumer:
                return Operator(obj).name
            elif attr == 'value' and callable_ is ExprData:
                return self.EXPR_DATA_VAR_NAME
            else:
                return obj_store.get_value_snippet(obj)

        def format_build_param(param_expr_val_map):
            out = list()
            for param, expr_val in param_expr_val_map.items():
                try:
                    value = format_expr_value(expr_val)
                # Cannot be serialized, so we skip it
                except NotSerializableError:
                    continue
                out.append('{param} = {value}'.format(
                    param=param, value=value
                ))
            return '\n' + ',\n'.join(out)


        def format_expr_value(expr_val, com=lambda x: ' # ' + x):
            excep = expr_val.excep
            value = expr_val.value

            if excep is NoValue:
                comment =  expr_val.get_id(full_qual=False) + ' (' + type(value).__name__ + ')'
                obj = make_serialized(expr_val, 'value')
            else:
                comment = type(excep).__name__ + ' raised when executing ' + expr_val.get_id()
                # Add extra comment marker for exception so the whole block can
                # be safely uncommented, without risking getting an exception
                # instead of the actual object.
                obj = '#' + make_serialized(expr_val, 'excep')

            comment = com(comment) if comment else ''
            return obj + comment

        # The parameter we are trying to compute cannot be computed and we will
        # just output a skeleton with a placeholder for the user to fill it
        is_user_defined = isinstance(self.op, PrebuiltOperator) and not self.result_list

        # Consumer operator is special since we don't compute anything to
        # get its value, it is just the name of a function
        if self.op.callable_ is Consumer:
            if not len(consumer_expr_stack) >= 2:
                return ('None', '')
            else:
                return (consumer_expr_stack[-2].op.name, '')
        elif self.op.callable_ is ExprData:
            # When we actually have an ExprValue, use it so we have the right
            # UUID.
            if expr_val_set:
                # They should all have be computed using the same ExprData,
                # so we check that all values are the same
                expr_val_list = [expr_val.value for expr_val in expr_val_set]
                assert expr_val_list[1:] == expr_val_list[:-1]

                expr_data = take_first(expr_val_set)
                return (format_expr_value(expr_data, lambda x:''), '')
            # Prior to execution, we don't have an ExprValue yet
            else:
                is_user_defined = True

        if not prefix:
            prefix = self.op.name
            # That is not completely safe, but very unlikely to break in
            # practice
            prefix = prefix.replace('.', '_')

        script = ''

        # Create the code to build all the parameters and get their variable
        # name
        snippet_list = list()
        param_var_map = OrderedDict()

        # Reusable parameter values are output first, so that non-reusable
        # parameters will be inside the for loops if any to be recomputed
        # for every combination of reusable parameters.
        param_map_chain = itertools.chain(
            self.get_param_map(reusable=True).items(),
            self.get_param_map(reusable=False).items(),
        )

        first_param = take_first(self.param_map.keys())

        for param, param_expr in param_map_chain:
            # Rename "self" parameter for more natural-looking output
            if param == first_param and self.op.is_method:
                is_meth_first_param = True
                pretty_param = make_method_self_name(param_expr)
            else:
                is_meth_first_param = False
                pretty_param = param

            param_prefix = make_var(pretty_param)

            # Get the set of ExprValue that were used to compute the
            # ExprValue given in expr_val_set
            param_expr_val_set = set()
            for expr_val in expr_val_set:
                # When there is no value for that parameter, that means it
                # could not be computed and therefore we skip that result
                with contextlib.suppress(KeyError):
                    param_expr_val = expr_val.param_value_map[param]
                    param_expr_val_set.add(param_expr_val)

            # Do a deep first search traversal of the expression.
            param_outvar, param_out = param_expr._get_script(
                param_prefix, obj_store, module_name_set, idt,
                param_expr_val_set,
                consumer_expr_stack = consumer_expr_stack + [self],
            )

            snippet_list.append(param_out)
            if is_meth_first_param:
                # Save a reference for future manipulation
                obj = param_outvar
            else:
                param_var_map[pretty_param] = param_outvar

        script += ''.join(snippet_list)

        # We now know our current indentation. The parameters will have indented
        # us if they are generator functions.
        idt_str = str(idt)

        if param_var_map:
            param_spec = ', '.join(
                '{param}={value}'.format(param=param, value=varname)
                for param, varname in param_var_map.items()
            )
        else:
            param_spec = ''

        do_not_call_callable = is_user_defined or isinstance(self.op, PrebuiltOperator)

        op_callable = self.op.name
        is_genfunc = self.op.is_genfunc
        # If it is a prebuilt operator and only one value is available, we just
        # replace the operator by it. That is valid since we will never end up
        # needing to call that operator in a different way.
        if (
            isinstance(self.op, PrebuiltOperator) and
                (
                    not self.result_list or
                    (
                        len(self.result_list) == 1 and
                        len(self.result_list[0].value_list) == 1
                    )
                )
        ):
            is_genfunc = False

        # The call expression is <obj>.<method>(...) instead of
        # <method>(self=<obj>, ...)
        elif self.op.is_method:
            op_callable = obj + '.' + self.op.callable_.__name__

        module_name_set.add(self.op.mod_name)

        # If the operator is a true generator function, we need to indent all
        # the code that depdends on us
        if is_genfunc:
            idt.indent()

        # Name of the variable holding the result of this expression
        outname = make_var('')

        # Dump the source file and line information
        src_file, src_line = self.op.src_loc
        if src_file and src_line:
            src_loc = '({src_file}:{src_line})'.format(
                src_line = src_line,
                src_file = src_file,
            )
        else:
            src_loc = ''

        script += '\n'
        script += make_comment('{id}{src_loc}'.format(
            id = list(self.get_id(with_tags=False, full_qual=False))[0],
            src_loc = '\n' + src_loc if src_loc else ''
        ), idt_str)

        # If no serialized value is available
        if is_user_defined:
            script += make_comment('User-defined:', idt_str)
            script += '{idt}{outname} = \n'.format(
                outname = outname,
                idt = idt_str,
            )

        # Dump the serialized value
        for result in self.result_list:
            # Make a copy to allow modifying the parameter names
            param_expr_val_map = copy.copy(result.param_expr_val_map)
            value_list = result.value_list

            # Restrict the list of ExprValue we are considering to the ones
            # we were asked about
            value_list = [
                expr_val for expr_val in value_list
                if expr_val in expr_val_set
            ]

            # Filter out values where nothing was computed and there was
            # no exception at this step either
            value_list = [
                expr_val for expr_val in value_list
                if (
                    (expr_val.value is not NoValue) or
                    (expr_val.excep is not NoValue)
                )
            ]
            if not value_list:
                continue

            # Rename "self" parameter to the name of the variable we are
            # going to apply the method on
            if self.op.is_method:
                first_param = take_first(param_expr_val_map)
                param_expr_val = param_expr_val_map.pop(first_param)
                self_param = make_var(make_method_self_name(param_expr_val.expr))
                param_expr_val_map[self_param] = param_expr_val

            # Multiple values to loop over
            try:
                if is_genfunc:
                    serialized_list = '\n' + idt.style + ('\n' + idt.style).join(
                        format_expr_value(expr_val, lambda x: ', # ' + x)
                        for expr_val in value_list
                    ) + '\n'
                    serialized_instance = 'for {outname} in ({values}):'.format(
                        outname = outname,
                        values = serialized_list
                    )
                # Just one value
                elif value_list:
                    serialized_instance = '{outname} = {value}'.format(
                        outname = outname,
                        value = format_expr_value(value_list[0])
                    )
            # The values cannot be serialized so we hide them
            except NotSerializableError:
                pass
            else:
                # Prebuilt operators use that code to restore the serialized
                # value, since they don't come from the execution of anything.
                if do_not_call_callable:
                    script += (
                        idt_str +
                        serialized_instance.replace('\n', '\n' + idt_str) +
                        '\n'
                    )
                else:
                    script += make_comment(serialized_instance, idt_str)

                # Show the origin of the values we have shown
                if param_expr_val_map:
                    origin = 'Built using:' + format_build_param(
                        param_expr_val_map
                    ) + '\n'
                    script += make_comment(origin, idt_str)

        # Dump the code to compute the values, unless it is a prebuilt op since it
        # has already been done
        if not do_not_call_callable:
            if is_genfunc:
                script += '{idt}for {output} in {op}({param}):\n'.format(
                    output = outname,
                    op = op_callable,
                    param = param_spec,
                    idt = idt_str
                )
            else:
                script += '{idt}{output} = {op}({param})\n'.format(
                    output = outname,
                    op = op_callable,
                    param = param_spec,
                    idt = idt_str,
                )

        return outname, script

    @classmethod
    def get_executor_map(cls, expr_wrapper_list):
        # Pool of deduplicated Expression
        expr_set = set()

        # Prepare all the wrapped Expression for execution, so they can be
        # deduplicated before being run
        for expr_wrapper in expr_wrapper_list:
            # The wrapped Expression could be deduplicated so we update it
            expr_wrapper.expr = expr_wrapper.expr._prepare_exec(expr_set)

        return {
            expr_wrapper: expr_wrapper.expr.execute
            for expr_wrapper in expr_wrapper_list
        }

    @classmethod
    def execute_all(cls, *args, **kwargs):
        executor_map = cls.get_executor_map(*args, **kwargs)

        for expr_wrapper, executor in executor_map.items():
            for expr_val in executor():
                yield (expr_wrapper, expr_val)

    def _prepare_exec(self, expr_set):
        self.discard_result()

        for param, param_expr in list(self.param_map.items()):
            # Update the param map in case param_expr was deduplicated
            self.param_map[param] = param_expr._prepare_exec(expr_set)

        # Look for an existing Expression that has the same parameters so we
        # don't add duplicates.
        for replacement_expr in expr_set - {self}:
            if (
                self.op.callable_ is replacement_expr.op.callable_ and
                self.param_map == replacement_expr.param_map
            ):
                return replacement_expr

        # Otherwise register this Expression so no other duplicate will be used
        else:
            expr_set.add(self)
            return self

    def execute(self, consumer_expr_stack=None):
        if consumer_expr_stack is None:
            consumer_expr_stack = []

        # Lazily compute the values of the Expression, trying to use
        # already computed values when possible

        # Check if we are allowed to reuse an instance that has already
        # been produced
        reusable = self.op.reusable

        # Get all the generators for reusable parameters
        reusable_param_exec_map = OrderedDict(
            (param, param_expr.execute(consumer_expr_stack=consumer_expr_stack + [self]))
            for param, param_expr in self.param_map.items()
            if param_expr.op.reusable
        )
        param_map_len = len(self.param_map)
        reusable_param_map_len = len(reusable_param_exec_map)

        # Consume all the reusable parameters, since they are generators
        for param_expr_val_map in consume_gen_map(reusable_param_exec_map, product=expr_value_product):
            # If some parameters could not be computed, we will not get all
            # values
            reusable_param_computed = (
                len(param_expr_val_map) == reusable_param_map_len
            )

            # Check if some ExprValue are already available for the current
            # set of reusable parameters. Non-reusable parameters are not
            # considered since they would be different every time in any case.
            if reusable and reusable_param_computed:
                # Check if we have already computed something for that
                # Expression and that set of parameter values
                result_list = self.find_result_list(param_expr_val_map)
                if result_list:
                    # Reusable objects should have only one ExprValueSeq
                    # that was computed with a given param_expr_val_map
                    assert len(result_list) == 1
                    expr_val_seq = result_list[0]
                    #TODO: call a callback for logging what we reused
                    yield from expr_val_seq.get_expr_value_iter()
                    continue

            # Only compute the non-reusable parameters if all the reusable one
            # are available, otherwise that is pointless
            if (
                reusable_param_computed and
                not any_value_is_NoValue(param_expr_val_map.values())
            ):
                # Non-reusable parameters must be computed every time, and we
                # don't take their cartesian product since we have fresh values
                # for all operator calls.
                nonreusable_param_exec_map = OrderedDict(
                    (param, param_expr.execute(consumer_expr_stack=consumer_expr_stack + [self]))
                    for param, param_expr in self.param_map.items()
                    if not param_expr.op.reusable
                )
                param_expr_val_map.update(next(
                    consume_gen_map(nonreusable_param_exec_map, product=no_product)
                ))

            # Propagate exceptions if some parameters did not execute
            # successfully.
            if (
                # Some arguments are missing: there was no attempt to compute
                # them because another argument failed to be computed
                len(param_expr_val_map) != param_map_len or
                # Or one of the arguments could not be computed
                any_value_is_NoValue(param_expr_val_map.values())
            ):
                expr_val = ExprValue(self, param_expr_val_map)
                expr_val_seq = ExprValueSeq(self, None, param_expr_val_map)
                expr_val_seq.value_list.append(expr_val)
                expr_val_seq.completed = True
                self.result_list.append(expr_val_seq)
                yield expr_val
                continue

            # If no value has been found, compute it and save the results in
            # a list.
            param_value_map = OrderedDict(
                # Extract the actual computed values wrapped in ExprValue
                (param, param_expr_val.value)
                for param, param_expr_val in param_expr_val_map.items()
            )

            # Consumer operator is special and we provide the value for it,
            # instead of letting it computing its own value
            if self.op.callable_ is Consumer:
                try:
                    consumer = consumer_expr_stack[-2].op.callable_
                except IndexError:
                    consumer = None
                iterated = [ ((consumer, None), (NoValue, None)) ]

            elif self.op.callable_ is ExprData:
                root_expr = consumer_expr_stack[0]
                iterated = [ ((root_expr.data, root_expr.data_uuid), (NoValue, None)) ]

            # Otherwise, we just call the operators with its parameters
            else:
                iterated = self.op.generator_wrapper(**param_value_map)

            iterator = iter(iterated)
            expr_val_seq = ExprValueSeq(self, iterator, param_expr_val_map)
            self.result_list.append(expr_val_seq)
            yield from expr_val_seq.get_expr_value_iter()

def infinite_iter(generator, value_list, from_gen):
    """Exhaust the `generator` when `from_gen=True`, yield from `value_list`
    otherwise.
    """
    if from_gen:
        for value in generator:
            value_list.append(value)
            yield value
    else:
        yield from value_list

def expr_value_product(*gen_list):
    """Similar to the cartesian product provided by itertools.product, with
    special handling of NoValue. It will only yield the combinations of values
    that are validated by :meth:`ExprValue.validate_expr_value_list`.
    """

    generator = gen_list[0]
    sub_generator_list = gen_list[1:]
    sub_generator_list_iterator = expr_value_product(*sub_generator_list)
    if sub_generator_list:
        from_gen = True
        value_list = list()
        for expr_val in generator:
            # The value is not useful, we can return early without calling the
            # other generators. That avoids spending time computing parameters
            # if they won't be used anyway.
            if expr_val.value is NoValue:
                # Returning an incomplete list will make the calling code aware
                # that some values were not computed at all
                yield [expr_val]
                continue

            for sub_value_list in infinite_iter(sub_generator_list_iterator, value_list, from_gen):
                expr_value_list = [expr_val] + sub_value_list
                if ExprValue.validate_expr_value_list(expr_value_list):
                    yield expr_value_list

            # After the first traversal of sub_generator_list_iterator, we
            # want to yield from the saved value_list
            from_gen = False
    else:
        for expr_val in generator:
            expr_value_list = [expr_val]
            if ExprValue.validate_expr_value_list(expr_value_list):
                yield expr_value_list

def no_product(*gen_list):
    # Take only one value from each generator, since non-reusable
    # operators are not supposed to produce more than one value.
    yield [next(generator) for generator in gen_list]

def consume_gen_map(param_map, product=itertools.product):
    if not param_map:
        yield OrderedDict()
    else:
        # sort to make sure we always compute the parameters in the same order
        gen_map = [(param, gen) for param, gen in param_map.items()]
        param_list, gen_list = zip(*gen_map)
        for values in product(*gen_list):
            yield OrderedDict(zip(param_list, values))

class AnnotationError(Exception):
    pass

class NotSerializableError(Exception):
    pass

def is_serializable(obj, raise_excep=False):
    stream = io.StringIO()
    try:
        # This may be slow for big objects but it is the only way to be sure
        # it can actually be serialized
        pickle.dumps(obj)
    except (TypeError, pickle.PickleError):
        if raise_excep:
            raise NotSerializableError(obj)
        return False
    else:
        return True

def get_class_from_name(cls_name, module_map):
    possible_mod_set = {
        mod_name
        for mod_name in module_map.keys()
        if cls_name.startswith(mod_name)
    }

    # Longest match in term of number of components
    possible_mod_list = sorted(possible_mod_set, key=lambda name: len(name.split('.')))
    if possible_mod_list:
        mod_name = possible_mod_list[-1]
    else:
        return None

    mod = module_map[mod_name]
    cls_name = cls_name[len(mod_name)+1:]
    return _get_class_from_name(cls_name, mod)

def _get_class_from_name(cls_name, namespace):
    if isinstance(namespace, collections.abc.Mapping):
        namespace = types.SimpleNamespace(**namespace)

    split = cls_name.split('.', 1)
    try:
        obj = getattr(namespace, split[0])
    except AttributeError as e:
        raise ValueError('Object not found') from e

    if len(split) > 1:
        return _get_class_from_name('.'.join(split[1:]), obj)
    else:
        return obj


def get_src_loc(obj):
    try:
        src_line = inspect.getsourcelines(obj)[1]
        src_file = inspect.getsourcefile(obj)
        src_file = str(pathlib.Path(src_file).resolve())
    except (OSError, TypeError):
        src_line, src_file = None, None

    return (src_file, src_line)

class Operator:
    # True to make all objects reusable by default, False otherwise
    REUSABLE_DEFAULT = True

    def __init__(self, callable_, name=None):
        assert callable(callable_)
        self._name = name
        self.callable_ = callable_

        self.signature = inspect.signature(self.resolved_callable)
        self.callable_globals = self.resolved_callable.__globals__
        self.annotations = copy.copy(self.resolved_callable.__annotations__)

        self.ignored_param = {
            param
            for param, param_spec in self.signature.parameters.items()
            # Ignore the parameters that have a default value without any
            # annotation
            if (
                param_spec.default is not inspect.Parameter.empty and
                param_spec.annotation is inspect.Parameter.empty
            )
        }

        self.optional_param = {
            param
            for param, param_spec in self.signature.parameters.items()
            # Parameters with a default value and and an annotation are
            # optional.
            if (
                param_spec.default is not inspect.Parameter.empty and
                param_spec.annotation is not inspect.Parameter.empty
            )
        }

        if hasattr(self.resolved_callable, 'reusable'):
            self.reusable = self.resolved_callable.reusable
        elif hasattr(self.value_type, 'reusable'):
            self.reusable = self.value_type.reusable
        else:
            self.reusable = self.REUSABLE_DEFAULT

        # At that point, we can get the prototype safely as the object is
        # mostly initialized.

        # Special support of return type annotation for classmethod
        if (
            inspect.ismethod(self.resolved_callable) and
            inspect.isclass(self.resolved_callable.__self__)
        ):
            return_type = self.value_type
            try:
                # If the return annotation type is an (indirect) base class of
                # the original annotation, we replace the annotation by the
                # subclass That allows implementing factory classmethods
                # easily.
                if issubclass(self.resolved_callable.__self__, return_type):
                    self.annotations['return'] = self.resolved_callable.__self__
            except TypeError:
                pass

    def __repr__(self):
        return '<Operator of ' + str(self.callable_) + '>'

    def force_param(self, param_callable_map):
        def define_type(param_type):
            class ForcedType(param_type):
                # Make ourselves transparent for better reporting
                __qualname__ = param_type.__qualname__
                __module__ = param_type.__module__
            return ForcedType

        prebuilt_op_set = set()
        for param, value_list in param_callable_map.items():
            # We just get the type of the first item in the list, which should
            # work in most cases
            param_type = type(take_first(value_list))

            # Create an artificial new type that will only be produced by
            # the PrebuiltOperator
            ForcedType = define_type(param_type)
            self.annotations[param] = ForcedType
            prebuilt_op_set.add(
                PrebuiltOperator(ForcedType, value_list)
            )

            # Make sure the parameter is not optional anymore
            self.optional_param.discard(param)
            self.ignored_param.discard(param)

        return prebuilt_op_set

    @property
    def resolved_callable(self):
        unwrapped = self.unwrapped_callable
        # We use __init__ when confronted to a class
        if inspect.isclass(unwrapped):
            return unwrapped.__init__
        return unwrapped

    @property
    def unwrapped_callable(self):
        return inspect.unwrap(self.callable_)

    @property
    def name(self):
        return self.get_name()

    def get_name(self, full_qual=True):
        if self._name is not None:
            if isinstance(self._name, str):
                return self._name
            # We allow passing in types for example, that will be used as the
            # source for the name
            else:
                return get_name(self._name, full_qual)
        try:
            name = get_name(self.callable_, full_qual)
        except AttributeError:
            name = self._name

        return name

    @property
    def mod_name(self):
        try:
            name = inspect.getmodule(self.unwrapped_callable).__name__
        except Exception:
            name = self.callable_globals['__name__']
        return name

    @property
    def src_loc(self):
        return get_src_loc(self.unwrapped_callable)

    @property
    def value_type(self):
        return self.get_prototype()[1]

    @property
    def is_genfunc(self):
        return inspect.isgeneratorfunction(self.unwrapped_callable)

    @property
    def is_class(self):
        return inspect.isclass(self.unwrapped_callable)

    @property
    def is_static_method(self):
        callable_ = self.unwrapped_callable
        try:
            cls = _get_class_from_name(
                callable_.__qualname__.rsplit('.', 1)[0],
                namespace=callable_.__globals__
            )
        # __globals__ is only defined for functions
        except (AttributeError, ValueError):
            return False

        if not inspect.isclass(cls):
            return False

        # We retrieve the function as it was defined in the class body, not as
        # it appears when accessed as a class attribute. That means we bypass
        # the descriptor protocol by reading the class' __dict__ directly, and
        # the staticmethod will not have a chance to "turn itself" into a
        # function.
        orig_callable = inspect.getattr_static(cls, callable_.__name__)
        return isinstance(orig_callable, staticmethod)

    @property
    def is_method(self):
        if self.is_cls_method or self.is_static_method:
            return False
        qualname = self.unwrapped_callable.__qualname__
        # Get the rightmost group, in case the callable has been defined
        # in a function
        qualname = qualname.rsplit('<locals>.', 1)[-1]

        # Dots in the qualified name means this function has been defined in a
        # class. This could also happen for closures, and they would get
        # "<locals>." somewhere in their name, but we handled that already.
        return '.' in qualname

    @property
    def is_cls_method(self):
        # Class methods appear as a bound method object when referenced through
        # their class. The method is bound to a class, which is not the case
        # if this is not a class method.
        return (
            inspect.ismethod(self.unwrapped_callable) and
            inspect.isclass(self.unwrapped_callable.__self__)
        )

    @property
    def generator_wrapper(self):
        if self.is_genfunc:
            @functools.wraps(self.callable_)
            def genf(*args, **kwargs):
                try:
                    has_yielded = False
                    for res in self.callable_(*args, **kwargs):
                        has_yielded = True
                        yield (res, create_uuid()), (NoValue, None)

                    # If no value at all were produced, we still need to yield
                    # something
                    if not has_yielded:
                        yield (NoValue, None), (NoValue, None)

                except Exception as e:
                    yield (NoValue, None), (e, create_uuid())
        else:
            @functools.wraps(self.callable_)
            def genf(*args, **kwargs):
                # yield one value and then return
                try:
                    val = self.callable_(*args, **kwargs)
                    yield (val, create_uuid()), (NoValue, None)
                except Exception as e:
                    yield (NoValue, None), (e, create_uuid())

        return genf

    def get_prototype(self):
        sig = self.signature
        first_param = take_first(sig.parameters)
        annotation_map = resolve_annotations(self.annotations, self.callable_globals)

        extra_ignored_param = set()
        # If it is a class
        if self.is_class:
            produced = self.unwrapped_callable
            # Get rid of "self", since annotating it with the class would lead to
            # infinite recursion when computing it signature. It will be handled
            # by execute() directly.
            annotation_map.pop(first_param, None)
            extra_ignored_param.add(first_param)

        # If it is just a function
        else:
            # When we have a method, we fill the annotations of the 1st
            # parameter with the name of the class it is defined in
            if self.is_method:
                cls_name = self.resolved_callable.__qualname__.split('.')[0]
                self.annotations[first_param] = cls_name

            produced = annotation_map['return']

        # Recompute after potentially modifying the annotations
        annotation_map = resolve_annotations(self.annotations, self.callable_globals)

        # Remove the return annotation, since we are handling that separately
        annotation_map.pop('return', None)

        # Check that we have annotations for all parameters that are not ignored
        for param, param_spec in sig.parameters.items():
            if (
                param not in annotation_map and
                param not in extra_ignored_param and
                param not in self.ignored_param
            ):
                raise AnnotationError('Missing annotation for "{param}" parameters of operator "{op}"'.format(
                    param = param,
                    op = self.callable_,
                ))

        def iter_by_keys(mapping, keys):
            for key in keys:
                try:
                    yield key, mapping[key]
                except KeyError:
                    pass

        param_map = OrderedDict(
            (name, annotation)
            for name, annotation in iter_by_keys(annotation_map, sig.parameters.keys())
            if not (
                name in self.ignored_param or
                name in extra_ignored_param
            )
        )

        return (param_map, produced)

class PrebuiltOperator(Operator):
    def __init__(self, obj_type, obj_list, id_=None, **kwargs):
        obj_list_ = list()
        uuid_list = list()
        for obj in obj_list:
            # Transparently copy the UUID to avoid having multiple UUIDs
            # refering to the same actual value.
            if isinstance(obj, SerializableExprValue):
                uuid_ = obj.value_uuid
                obj = obj.value
            else:
                uuid_ = create_uuid()

            uuid_list.append(uuid_)
            obj_list_.append(obj)

        self.obj_list = obj_list_
        self.uuid_list = uuid_list
        self.obj_type = obj_type

        if id_ is None:
            name = self.obj_type
        else:
            name = id_
            # Get rid of the existing tags, since the name of the operator
            # already carries that information.
            for obj in self.obj_list:
                try:
                    del obj.tags
                except AttributeError:
                    pass

        # Placeholder for the signature
        def callable_() -> self.obj_type:
            pass
        super().__init__(callable_, name, **kwargs)

    @property
    def src_loc(self):
        return get_src_loc(self.value_type)

    @property
    def is_genfunc(self):
        return len(self.obj_list) > 1

    @property
    def is_method(self):
        return False

    @property
    def generator_wrapper(self):
        def genf():
            for obj, uuid_ in zip(self.obj_list, self.uuid_list):
                yield (obj, uuid_), (NoValue, None)
        return genf

def reusable(reusable=Operator.REUSABLE_DEFAULT):
    def decorator(wrapped):
        wrapped.reusable = reusable
        return wrapped
    return decorator

class ExprValueSeq:
    def __init__(self, expr, iterator, param_expr_val_map):
        self.expr = expr
        self.iterator = iterator
        self.value_list = list()
        self.param_expr_val_map = param_expr_val_map
        self.completed = False

    def get_expr_value_iter(self):
        # Yield existing values
        yield from self.value_list

        # Then compute the remaining ones
        if not self.completed:
            for (value, value_uuid), (excep, excep_uuid) in self.iterator:
                expr_val = ExprValue(self.expr, self.param_expr_val_map,
                    value, value_uuid,
                    excep, excep_uuid
                )
                self.value_list.append(expr_val)
                value_list_idx = len(self.value_list) - 1
                yield expr_val

                # If value_list length has changed, catch up with the values
                # that were computed behind our back, so that this generator is
                # reentrant.
                if value_list_idx != len(self.value_list) - 1:
                    # This will yield all values, even if the list grows while
                    # we are yielding the control back to another piece of code.
                    yield from self.value_list[value_list_idx + 1:]


            self.completed = True

def any_value_is_NoValue(value_list):
    return any(
        expr_val.value is NoValue
        for expr_val in value_list
    )

class SerializableExprValue:
    def __init__(self, expr_val, serialized_map, hidden_callable_set=None):
        # YAML serializer uses __name__ instead of __qualname__ to store
        # references to objects, which breaks when storing a reference to a
        # method
        #  https://bitbucket.org/ruamel/yaml/issues/214/incorrect-tag-generation-for-python-types
        if expr_val.expr.op.callable_ is Consumer:
            self.value = None
            self.excep = NoValue
        else:
            self.value = expr_val.value if is_serializable(expr_val.value) else NoValue
            self.excep = expr_val.excep if is_serializable(expr_val.excep) else NoValue


        self.value_uuid = expr_val.value_uuid
        self.excep_uuid = expr_val.excep_uuid

        self.callable_qual_name = expr_val.expr.op.get_name(full_qual=True)
        self.callable_name = expr_val.expr.op.get_name(full_qual=False)

        # Pre-compute all the IDs so they are readily available once the value
        # is deserialized
        self.recorded_id_map = dict()
        for full_qual, with_tags in itertools.product((True, False), repeat=2):
            self.recorded_id_map[(full_qual, with_tags)] = expr_val.get_id(
                full_qual = full_qual,
                with_tags = with_tags
            )

        self.type_names = [
            get_name(type_, full_qual=True)
            for type_ in inspect.getmro(expr_val.expr.op.value_type)
            if type_ is not object
        ]

        self.param_value_map = collections.OrderedDict()
        for param, param_expr_val in expr_val.param_value_map.items():
            param_serialzable = param_expr_val._get_serializable(
                serialized_map,
                hidden_callable_set
            )
            self.param_value_map[param] = param_serialzable

    def get_id(self, full_qual=True, with_tags=True):
        args = (full_qual, with_tags)
        return self.recorded_id_map[args]

    def get_parent_set(self, predicate):
        parent_set = set()
        if predicate(self):
            parent_set.add(self)
        self._get_parent_set(parent_set, predicate)
        return parent_set

    def _get_parent_set(self, parent_set, predicate):
        for parent in self.param_value_map.values():
            if predicate(parent):
                parent_set.add(parent)
            parent._get_parent_set(parent_set, predicate)

def get_name(obj, full_qual=True):
    # Add the module's name in front of the name to get a fully
    # qualified name
    if full_qual:
        module_name = obj.__module__
        module_name = (
            module_name + '.'
            if module_name != '__main__' and module_name != 'builtins'
            else ''
        )
    else:
        module_name = ''
    # Classmethods appear as bound method of classes. Since each subclass will
    # get a different bound method object, we want to reflect that in the
    # qualified name we use, instead of always using the same qualname
    if inspect.ismethod(obj):
        qualname = (
            obj.__self__.__qualname__ + '.' +
            obj.__qualname__.rsplit('.', 1)[-1]
        )
    else:
        qualname = obj.__qualname__

    return module_name + qualname

class ExprValue:
    def __init__(self, expr, param_value_map,
            value=NoValue, value_uuid=None,
            excep=NoValue, excep_uuid=None
    ):
        self.value = value
        self.value_uuid = value_uuid
        self.excep = excep
        self.excep_uuid = excep_uuid
        self.expr = expr
        self.param_value_map = param_value_map

    def _get_serializable(self, serialized_map, *args, **kwargs):
        if serialized_map is None:
            serialized_map = dict()

        try:
            return serialized_map[self]
        except KeyError:
            serializable = SerializableExprValue(self, serialized_map, *args, **kwargs)
            serialized_map[self] = serializable
            return serializable

    def _dfs_visit(self):
        expr_map = dict()
        self._do_dfs_visit(expr_map)
        return expr_map

    def _do_dfs_visit(self, expr_map):
        expr_map[self.expr] = self

        for param_expr_val in self.param_value_map.values():
            param_expr_val._do_dfs_visit(expr_map)

    @classmethod
    def validate_expr_value_list(cls, expr_value_list):
        if not expr_value_list:
            return True

        expr_value_ref = expr_value_list[0]
        expr_map_ref = expr_value_ref._dfs_visit()

        for expr_val in expr_value_list[1:]:
            expr_map = expr_val._dfs_visit()
            # For all Expression's that directly or indirectly lead to both the
            # reference ExprValue and the ExprValue, check that it had the same
            # value. That ensures that we are not making incompatible combinations.

            if not all(
                expr_map_ref[expr] is expr_map[expr]
                for expr
                in expr_map.keys() & expr_map_ref.keys()
                # We don't consider the non-reusable parameters since it is
                # expected that they will differ
                if expr.op.reusable
            ):
                return False

            if not cls.validate_expr_value_list(expr_value_list[2:]):
                return False

        return True

    def get_id(self, *args, with_tags=True, **kwargs):
        # There exists only one ID for a given ExprValue so we just return it
        # instead of an iterator.
        return take_first(self.expr.get_id(with_tags=with_tags,
            expr_val=self, *args, **kwargs))

    def get_failed_values(self):
        if self.excep is not NoValue:
            yield self

        for param, expr_val in self.param_value_map.items():
            yield from expr_val.get_failed_values()

@reusable(False)
class Consumer:
    def __init__(self):
        pass

@reusable(False)
class ExprData(dict):
    def __init__(self):
        pass

