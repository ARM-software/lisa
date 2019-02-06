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

import inspect
import collections.abc
from collections import OrderedDict
import copy
import itertools
import functools
import lzma
import pathlib
import contextlib
import pickle
import pprint
import pickletools

import exekall._utils as utils
from exekall._utils import NoValue

class NoOperatorError(Exception):
    pass

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

class ValueDB:
    # Version 4 is available since Python 3.4 and improves a bit loading and
    # dumping speed.
    PICKLE_PROTOCOL = 4

    def __init__(self, froz_val_seq_list, adaptor_cls=None):
        # Avoid storing duplicate FrozenExprVal sharing the same value/excep
        # UUID
        self.froz_val_seq_list = self._dedup_froz_val_seq_list(froz_val_seq_list)
        self.adaptor_cls = adaptor_cls

    @classmethod
    def _dedup_froz_val_seq_list(cls, froz_val_seq_list):
        """
        Avoid keeping :class:`FrozenExprVal` that share the same value or
        excep UUID, since they are duplicates of each-other.
        """

        # First pass: find all frozen values corresponding to a given UUID
        uuid_map = {}
        def update_uuid_map(froz_val):
            uuid_map.setdefault(froz_val.uuid, set()).add(froz_val)
            return froz_val
        cls._froz_val_dfs(froz_val_seq_list, update_uuid_map)

        # Make sure no deduplication will occur on None, as it is used as a
        # marker when no exception was raised or when no value was available.
        uuid_map[(None, None)] = set()

        # Select one FrozenExprVal for each UUID pair
        def select_froz_val(froz_val_set):
            candidates = [
                froz_val
                for froz_val in froz_val_set
                # We discard candidates that have no parameters, as they
                # contain less information than the ones that do. This is
                # typically the case for PrebuiltOperator values
                if froz_val.param_map
            ]

            # At this point, there should be no more than one "original" value,
            # the other candidates were just values of PrebuiltOperator, or are
            # completely equivalent to the original value

            if candidates:
                return candidates[0]
            # If there was no better candidate, just return the first one
            else:
                return utils.take_first(froz_val_set)

        uuid_map = {
            uuid_pair: select_froz_val(froz_val_set)
            for uuid_pair, froz_val_set in uuid_map.items()
        }

        # Second pass: only keep one frozen value for each UUID
        def rewrite_graph(froz_val):
            return uuid_map[froz_val.uuid]

        return cls._froz_val_dfs(froz_val_seq_list, rewrite_graph)

    @classmethod
    def merge(cls, db_list):
        db_list = list(db_list)
        adaptor_cls_set = {
            db.adaptor_cls
            for db in db_list
        }
        if len(adaptor_cls_set) != 1:
            raise ValueError('Cannot merge ValueDB with different adaptor classes: {}'.format(adaptor_cls_set))
        adaptor_cls = utils.take_first(adaptor_cls_set)

        froz_val_seq_list = list(itertools.chain.from_iterable(
            db.froz_val_seq_list
            for db in db_list
        ))
        return cls(froz_val_seq_list, adaptor_cls=adaptor_cls)

    @classmethod
    def from_path(cls, path, relative_to=None):
        if relative_to is not None:
            relative_to = pathlib.Path(relative_to).resolve()
            if not relative_to.is_dir():
                relative_to = pathlib.Path(relative_to).parent
            path = pathlib.Path(relative_to, path)

        with lzma.open(str(path), 'rb') as f:
            # Disabling garbage collection while loading result in significant
            # speed improvement, since it creates a lot of new objects in a
            # very short amount of time.
            with utils.disable_gc():
                db = pickle.load(f)
        assert isinstance(db, cls)

        # Apply some post-processing on the DB with a known path
        cls._call_adaptor_reload(db, path=path)

        return db

    @classmethod
    def _reload_serialized(cls, dct):
        db = cls.__new__(cls)
        db.__dict__ = dct

        # Apply some post-processing on the DB that was just reloaded, with no
        # path since we don't even know if that method was invoked on something
        # serialized in a file.
        cls._call_adaptor_reload(db, path=None)

        return db

    def __reduce_ex__(self, protocol):
        return (self._reload_serialized, (self.__dict__,))

    @staticmethod
    def _call_adaptor_reload(db, path):
        adaptor_cls = db.adaptor_cls
        if adaptor_cls:
            db = adaptor_cls.reload_db(db, path=path)
        return db

    def to_path(self, path, optimize=True):
        """
        Write the DB to the given file.

        :param path: path to file to write the DB into
        :type path: pathlib.Path or str

        :param optimize: Optimize the representation of the DB. This may
            increase the dump time, but should speed-up loading/file size.
        :type optimize: bool
        """
        if optimize:
            bytes_ = pickle.dumps(self, protocol=self.PICKLE_PROTOCOL)
            bytes_ = pickletools.optimize(bytes_)
            dumper = lambda f: f.write(bytes_)
        else:
            dumper = lambda f: pickle.dump(self, f, protocol=self.PICKLE_PROTOCOL)

        with lzma.open(str(path), 'wb') as f:
            dumper(f)

    @property
    @utils.once
    def _uuid_map(self):
        uuid_map = dict()

        def update_map(froz_val):
            uuid_map[froz_val.uuid] = froz_val
            return froz_val

        self._froz_val_dfs(self.froz_val_seq_list, update_map)

        return uuid_map

    @classmethod
    def _froz_val_dfs(cls, froz_val_seq_list, callback):
        return [
            FrozenExprValSeq(
                froz_val_list=[
                    cls._do_froz_val_dfs(froz_val, callback)
                    for froz_val in froz_val_seq
                ],
                param_map={
                    param: cls._do_froz_val_dfs(froz_val, callback)
                    for param, froz_val in froz_val_seq.param_map.items()
                }
            )
            for froz_val_seq in froz_val_seq_list
        ]

    @classmethod
    def _do_froz_val_dfs(cls, froz_val, callback):
        updated_froz_val = callback(froz_val)
        updated_froz_val.param_map = {
            param: cls._do_froz_val_dfs(param_froz_val, callback)
            for param, param_froz_val in updated_froz_val.param_map.items()
        }
        return updated_froz_val

    def get_by_uuid(self, uuid):
        return self._uuid_map[uuid]

    def get_by_predicate(self, predicate, flatten=True, deduplicate=False):
        """
        Get objects matching the predicate.

        :param flatten: If False, return a set of frozenset of objects.
            There is a frozenset set for each expression result that shared
            their parameters.  If False, the top-level set is flattened into a
            set of objects matching the predicate.
        :type flatten: bool

        :param deduplicate: If True, there won't be duplicates across nested
            sets.
        :type deduplicate: bool
        """
        froz_val_set_set = set()

        # When we reload instances of a class from the DB, we don't
        # want anything else to be able to produce it, since we want to
        # run on that existing data set

        # Make sure we don't select the same froz_val twice
        if deduplicate:
            visited = set()
            def wrapped_predicate(froz_val):
                if froz_val in visited:
                    return False
                else:
                    visited.add(froz_val)
                    return predicate(froz_val)
        else:
            wrapped_predicate = predicate

        for froz_val_seq in self.froz_val_seq_list:
            froz_val_set = set()
            for froz_val in itertools.chain(
                    # traverse all values, including the ones from the
                    # parameters, even when there was no value computed
                    # (because of a failed parent for example)
                    froz_val_seq, froz_val_seq.param_map.values()
                ):
                froz_val_set.update(froz_val.get_by_predicate(wrapped_predicate))

            froz_val_set_set.add(frozenset(froz_val_set))

        if flatten:
            return set(utils.flatten_seq(froz_val_set_set))
        else:
            return froz_val_set_set

    def get_roots(self, flatten=True):
        froz_val_set_set = {
            frozenset(froz_val_seq)
            for froz_val_seq in self.froz_val_seq_list
        }
        if flatten:
            return set(utils.flatten_seq(froz_val_set_set))
        else:
            return froz_val_set_set

    def prune_by_predicate(self, predicate):
        def prune(froz_val):
            if isinstance(froz_val, PrunedFrozVal):
                return froz_val
            elif predicate(froz_val):
                return PrunedFrozVal(froz_val)
            else:
                # Edit the param_map in-place, so we keep it potentially shared
                # if possible.
                for param, param_froz_val in list(froz_val.param_map.items()):
                    froz_val.param_map[param] = prune(param_froz_val)

                return froz_val

        def make_froz_val_seq(froz_val_seq):
            froz_val_list = [
                prune(froz_val)
                for froz_val in froz_val_seq
                # Just remove the root PrunedFrozVal, since they are useless at
                # this level (i.e. nothing depends on them)
                if not predicate(froz_val)
            ]

            # All param_map will be the same in the list by construction
            try:
                param_map = froz_val_list[0].param_map
            except IndexError:
                param_map = {}

            return FrozenExprValSeq(
                froz_val_list=froz_val_list,
                param_map=param_map,
            )

        return self.__class__(
            froz_val_seq_list=[
                make_froz_val_seq(froz_val_seq)
                # That will keep proper inter-object references as in the
                # original graph of objects
                for froz_val_seq in copy.deepcopy(self.froz_val_seq_list)
            ]
        )

    def get_all(self, **kwargs):
        return self.get_by_predicate(lambda froz_val: True, **kwargs)

    def get_by_type(self, cls, include_subclasses=True, **kwargs):
        if include_subclasses:
            predicate = lambda froz_val: isinstance(froz_val.value, cls)
        else:
            predicate = lambda froz_val: type(froz_val.value) is cls
        return self.get_by_predicate(predicate, **kwargs)

    def get_by_id(self, id_, qual=False, full_qual=False, **kwargs):
        def predicate(froz_val):
            return utils.match_name(
                froz_val.get_id(qual=qual, full_qual=full_qual),
                [id_]
            )

        return self.get_by_predicate(predicate, **kwargs)


class ScriptValueDB:
    def __init__(self, db, var_name='db'):
        self.db = db
        self.var_name = var_name

    def get_snippet(self, expr_val, attr):
        return '{db}.get_by_uuid({uuid}).{attr}'.format(
            db=self.var_name,
            uuid=repr(expr_val.uuid),
            attr=attr,
        )

class CycleError(Exception):
    pass

class ExpressionBase:
    def __init__(self, op, param_map):
        self.op = op
        # Map of parameters to other Expression
        self.param_map = param_map

    @classmethod
    def cse(cls, expr_list):
        """
        Apply a flavor of common subexpressions elimination to the
        Expression.
        """

        expr_map = {}
        return [
            expr._cse(expr_map)
            for expr in expr_list
        ]

    def _cse(self, expr_map):
        # Deep first
        self.param_map = {
            param: param_expr._cse(expr_map=expr_map)
            for param, param_expr in self.param_map.items()
        }

        key = (
            self.op.callable_,
            # get a nested tuple sorted by param name with the shape:
            # ((param, val), ...)
            tuple(sorted(self.param_map.items(), key=lambda k_v: k_v[0]))
        )

        return expr_map.setdefault(key, self)

    def __repr__(self):
        return '<Expression of {name} at {id}>'.format(
            name=self.op.get_name(full_qual=True),
            id = hex(id(self))
        )

    def get_structure(self, full_qual=True, graphviz=False):
        if graphviz:
            return self._get_graphviz_structure(full_qual, level=0, visited=set())
        else:
            return self._get_structure(full_qual=full_qual)

    def _get_structure(self, full_qual=True, indent=1):
        indent_str = 4 * ' ' * indent

        if isinstance(self.op, PrebuiltOperator):
            op_name = '<provided>'
        else:
            op_name = self.op.get_name(full_qual=True)

        out = '{op_name} ({value_type_name})'.format(
            op_name = op_name,
            value_type_name = utils.get_name(self.op.value_type, full_qual=full_qual),
        )
        if self.param_map:
            out += ':\n'+ indent_str + ('\n'+indent_str).join(
                '{param}: {desc}'.format(param=param, desc=desc._get_structure(
                    full_qual=full_qual,
                    indent=indent+1
                ))
                for param, desc in self.param_map.items()
            )
        return out

    def _get_graphviz_structure(self, full_qual, level, visited):
        if self in visited:
            return ''
        else:
            visited.add(self)

        if isinstance(self.op, PrebuiltOperator):
            op_name = '<provided>'
        else:
            op_name = self.op.get_name(full_qual=True)

        # Use the Python id as it is guaranteed to be unique during the lifetime of
        # the object, so it is a good candidate to refer to a node
        uid = id(self)

        src_file, src_line = self.op.src_loc
        if src_file and src_line:
            src_loc = '({}:{})'.format(src_file, src_line)
        else:
            src_loc = ''

        out = ['{uid} [label="{op_name} {reusable}\\ntype: {value_type_name}\\n{loc}"]'.format(
            uid=uid,
            op_name=op_name,
            reusable='(reusable)' if self.op.reusable else '(non-reusable)',
            value_type_name=utils.get_name(self.op.value_type, full_qual=full_qual),
            loc=src_loc,
        )]
        if self.param_map:
            for param, param_expr in self.param_map.items():
                out.append(
                    '{param_uid} -> {uid} [label="{param}"]'.format(
                        param_uid=id(param_expr),
                        uid=uid,
                        param=param,
                    )
                )

                out.append(
                    param_expr._get_graphviz_structure(
                        full_qual=full_qual,
                        level=level+1,
                        visited=visited,
                    )
                )

        if level == 0:
            title = 'Structure of ' + self.get_id(qual=False)
            node_out = 'digraph structure {{\n{}\nlabel="' + title + '"\n}}'
        else:
            node_out = '{}'
        # dot seems to dislike empty line with just ";"
        return node_out.format(';\n'.join(line for line in out if line.strip()))

    def get_id(self, *args, marked_expr_val_set=set(), **kwargs):
        id_, marker = self._get_id(*args,
            marked_expr_val_set=marked_expr_val_set,
            **kwargs
        )
        if marked_expr_val_set:
            return '\n'.join((id_, marker))
        else:
            return id_

    def _get_id(self, with_tags=True, full_qual=True, qual=True, style=None, expr_val=None, marked_expr_val_set=None, hidden_callable_set=None):
        if hidden_callable_set is None:
            hidden_callable_set = set()

        # We always hide the Consumer operator since it does not add anything
        # to the ID. It is mostly an implementation detail.
        hidden_callable_set.update((Consumer, ExprData))

        if expr_val is None:
            param_map = dict()
        # If we were asked about the ID of a specific value, make sure we
        # don't explore other paths that lead to different values
        else:
            param_map = expr_val.param_map

        return self._get_id_internal(
            param_map=param_map,
            expr_val=expr_val,
            with_tags=with_tags,
            marked_expr_val_set=marked_expr_val_set,
            hidden_callable_set=hidden_callable_set,
            full_qual=full_qual,
            qual=qual,
            style=style,
        )

    def _get_id_internal(self, param_map, expr_val, with_tags, marked_expr_val_set, hidden_callable_set, full_qual, qual, style):
        separator = ':'
        marker_char = '^'
        get_id_kwargs = dict(
            full_qual=full_qual,
            qual=qual,
            style=style
        )

        if marked_expr_val_set is None:
            marked_expr_val_set = set()

        # We only get the ID's of the parameter ExprVal that lead to the
        # ExprVal we are interested in
        param_id_map = OrderedDict(
            (param, param_expr._get_id(
                **get_id_kwargs,
                with_tags = with_tags,
                # Pass None when there is no value available, so we will get
                # a non-tagged ID when there is no value computed
                expr_val = param_map.get(param),
                marked_expr_val_set = marked_expr_val_set,
                hidden_callable_set = hidden_callable_set,
            ))
            for param, param_expr in self.param_map.items()
            if (
                param_expr.op.callable_ not in hidden_callable_set
                # If the value is marked, the ID will not be hidden
                or param_map.get(param) in marked_expr_val_set
            )
        )

        def get_tags(expr_val):
            if expr_val is not None:
                if with_tags:
                    tag = expr_val.format_tags()
                else:
                    tag = ''
                return tag
            else:
                return ''

        def get_marker_char(expr_val):
            return marker_char if expr_val in marked_expr_val_set else ' '

        tag_str = get_tags(expr_val)

        # No parameter to worry about
        if not param_id_map:
            id_ = self.op.get_id(**get_id_kwargs) + tag_str
            marker_str = get_marker_char(expr_val) * len(id_)
            return (id_, marker_str)

        # Recursively build an ID
        else:
            # Make a copy to be able to pop items from it
            param_id_map = copy.copy(param_id_map)

            # Extract the first parameter to always use the prefix
            # notation, i.e. its value preceding the ID of the current
            # Expression
            param, (param_id, param_marker) = param_id_map.popitem(last=False)

            if param_id:
                separator_spacing = ' ' * len(separator)
                param_str = param_id + separator
            else:
                separator_spacing = ''
                param_str = ''

            op_str = '{op}{tags}'.format(
                op = self.op.get_id(**get_id_kwargs),
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
            return (id_, marker_str)

    def get_script(self, *args, **kwargs):
        return self.get_all_script([self], *args, **kwargs)

    @classmethod
    def get_all_script(cls, expr_list, prefix='value', db_path='VALUE_DB.pickle.xz', db_relative_to=None, db=None, adaptor_cls=None):
        assert expr_list

        if db is None:
            froz_val_seq_list = FrozenExprValSeq.from_expr_list(expr_list)
            script_db = ScriptValueDB(ValueDB(
                froz_val_seq_list,
                adaptor_cls=adaptor_cls,
            ))
        else:
            script_db = ScriptValueDB(db)


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
        reusable_outvar_map = dict()
        for i, expr in enumerate(expr_list):
            script += (
                '#'*80 + '\n# Computed expressions:' +
                make_comment(expr.get_id(mark_excep=True, full_qual=False))
                + '\n' +
                make_comment(expr.get_structure()) + '\n\n'
            )
            idt = IndentationManager(' '*4)

            expr_val_set = set(expr.get_all_vals())
            result_name, snippet = expr._get_script(
                reusable_outvar_map = reusable_outvar_map,
                prefix = prefix + str(i),
                script_db = script_db,
                module_name_set = module_name_set,
                idt = idt,
                expr_val_set = expr_val_set,
                consumer_expr_stack = [],
            )

            # ExprData must be printable to a string representation that can be
            # fed back to eval()
            expr_data = pprint.pformat(expr.data)

            expr_data_snippet = cls.EXPR_DATA_VAR_NAME + ' = ' + expr_data + '\n'

            script += (
                expr_data_snippet +
                snippet +
                '\n'
            )
            plain_name_cls_set.update(type(x) for x in expr.data.values())

            result_name_map[expr] = result_name


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

        # If there is no ExprVal referenced by that script, we don't need
        # to access any ValueDB
        if expr_val_set:
            if db_relative_to is not None:
                db_relative_to = ', relative_to='+db_relative_to
            else:
                db_relative_to = ''

            header += '{db} = {db_loader}({path}{db_relative_to})\n'.format(
                db = script_db.var_name,
                db_loader = utils.get_name(ValueDB.from_path, full_qual=True),
                path = repr(str(db_path)),
                db_relative_to = db_relative_to
            )

        script = header + '\n' + script
        return (result_name_map, script)

    EXPR_DATA_VAR_NAME = 'EXPR_DATA'

    def _get_script(self, reusable_outvar_map, *args, **kwargs):
        with contextlib.suppress(KeyError):
            outvar = reusable_outvar_map[self]
            return (outvar, '')
        outvar, script = self._get_script_internal(
            reusable_outvar_map, *args, **kwargs
        )
        if self.op.reusable:
            reusable_outvar_map[self] = outvar
        return (outvar, script)

    def _get_script_internal(self, reusable_outvar_map, prefix, script_db, module_name_set, idt, expr_val_set, consumer_expr_stack, expr_val_seq_list=[]):
        def make_method_self_name(expr):
            return expr.op.value_type.__name__.replace('.', '')

        def make_var(name):
            # If the variable name already contains a double underscore, we use
            # 3 of them for the separator between the prefix and the name, so
            # it will avoid ambiguity between these cases:
            # prefix="prefix", name="my__name":
            #   prefix___my__name
            # prefix="prefix__my", name="name":
            #   prefix__my__name

            # Find the longest run of underscores
            nr_underscore = 0
            current_counter = 0
            for letter in name:
                if letter == '_':
                    current_counter += 1
                else:
                    nr_underscore = max(current_counter, nr_underscore)
                    current_counter = 0

            sep = (nr_underscore + 1) * '_'
            name = sep + name if name else ''
            return prefix + name

        def make_comment(code, idt):
            prefix = idt + '# '
            return prefix + prefix.join(code.splitlines(True)) + '\n'

        def make_serialized(expr_val, attr):
            obj = getattr(expr_val, attr)
            utils.is_serializable(obj, raise_excep=True)

            # When the ExprVal is from an Expression of the Consumer
            # operator, we directly print out the name of the function that was
            # selected since it is not serializable
            callable_ = expr_val.expr.op.callable_
            if attr == 'value' and callable_ is Consumer:
                return Operator(obj).get_name(full_qual=True)
            elif attr == 'value' and callable_ is ExprData:
                return self.EXPR_DATA_VAR_NAME
            else:
                return script_db.get_snippet(expr_val, attr)

        def format_build_param(param_map):
            out = list()
            for param, expr_val in param_map.items():
                try:
                    value = format_expr_val(expr_val)
                # Cannot be serialized, so we skip it
                except utils.NotSerializableError:
                    continue
                out.append('{param} = {value}'.format(
                    param=param, value=value
                ))
            return '\n' + ',\n'.join(out)


        def format_expr_val(expr_val, com=lambda x: ' # ' + x):
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
        is_user_defined = isinstance(self.op, PrebuiltOperator) and not expr_val_seq_list

        # Consumer operator is special since we don't compute anything to
        # get its value, it is just the name of a function
        if self.op.callable_ is Consumer:
            if not len(consumer_expr_stack) >= 2:
                return ('None', '')
            else:
                return (consumer_expr_stack[-2].op.get_name(full_qual=True), '')
        elif self.op.callable_ is ExprData:
            # When we actually have an ExprVal, use it so we have the right
            # UUID.
            if expr_val_set:
                # They should all have be computed using the same ExprData,
                # so we check that all values are the same
                expr_val_list = [expr_val.value for expr_val in expr_val_set]
                assert expr_val_list[1:] == expr_val_list[:-1]

                expr_data = utils.take_first(expr_val_set)
                return (format_expr_val(expr_data, lambda x:''), '')
            # Prior to execution, we don't have an ExprVal yet
            else:
                is_user_defined = True

        if not prefix:
            prefix = self.op.get_name(full_qual=True)
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

        def get_param_map(reusable):
            return OrderedDict(
                (param, param_expr)
                for param, param_expr
                in self.param_map.items()
                if bool(param_expr.op.reusable) == reusable
            )
        param_map_chain = itertools.chain(
            get_param_map(reusable=True).items(),
            get_param_map(reusable=False).items(),
        )

        first_param = utils.take_first(self.param_map.keys())

        for param, param_expr in param_map_chain:
            # Rename "self" parameter for more natural-looking output
            if param == first_param and self.op.is_method:
                is_meth_first_param = True
                pretty_param = make_method_self_name(param_expr)
            else:
                is_meth_first_param = False
                pretty_param = param

            param_prefix = make_var(pretty_param)

            # Get the set of ExprVal that were used to compute the
            # ExprVal given in expr_val_set
            param_expr_val_set = set()
            for expr_val in expr_val_set:
                # When there is no value for that parameter, that means it
                # could not be computed and therefore we skip that result
                with contextlib.suppress(KeyError):
                    param_expr_val = expr_val.param_map[param]
                    param_expr_val_set.add(param_expr_val)

            # Do a deep first traversal of the expression.
            param_outvar, param_out = param_expr._get_script(
                reusable_outvar_map, param_prefix, script_db, module_name_set, idt,
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

        op_callable = self.op.get_name(full_qual=True)
        is_genfunc = self.op.is_genfunc
        # If it is a prebuilt operator and only one value is available, we just
        # replace the operator by it. That is valid since we will never end up
        # needing to call that operator in a different way.
        if (
            isinstance(self.op, PrebuiltOperator) and
                (
                    not expr_val_seq_list or
                    (
                        len(expr_val_seq_list) == 1 and
                        len(expr_val_seq_list[0].expr_val_list) == 1
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
            id = self.get_id(with_tags=False, full_qual=False),
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
        for expr_val_seq in expr_val_seq_list:
            # Make a copy to allow modifying the parameter names
            param_map = copy.copy(expr_val_seq.param_map)
            expr_val_list = expr_val_seq.expr_val_list

            # Restrict the list of ExprVal we are considering to the ones
            # we were asked about
            expr_val_list = [
                expr_val for expr_val in expr_val_list
                if expr_val in expr_val_set
            ]

            # Filter out values where nothing was computed and there was
            # no exception at this step either
            expr_val_list = [
                expr_val for expr_val in expr_val_list
                if (
                    (expr_val.value is not NoValue) or
                    (expr_val.excep is not NoValue)
                )
            ]
            if not expr_val_list:
                continue

            # Rename "self" parameter to the name of the variable we are
            # going to apply the method on
            if self.op.is_method:
                first_param = utils.take_first(param_map)
                param_expr_val = param_map.pop(first_param)
                self_param = make_var(make_method_self_name(param_expr_val.expr))
                param_map[self_param] = param_expr_val

            # Multiple values to loop over
            try:
                if is_genfunc:
                    serialized_list = '\n' + idt.style + ('\n' + idt.style).join(
                        format_expr_val(expr_val, lambda x: ', # ' + x)
                        for expr_val in expr_val_list
                    ) + '\n'
                    serialized_instance = 'for {outname} in ({values}):'.format(
                        outname = outname,
                        values = serialized_list
                    )
                # Just one value
                elif expr_val_list:
                    serialized_instance = '{outname} = {value}'.format(
                        outname = outname,
                        value = format_expr_val(expr_val_list[0])
                    )
            # The values cannot be serialized so we hide them
            except utils.NotSerializableError:
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
                if param_map:
                    origin = 'Built using:' + format_build_param(
                        param_map
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


class ComputableExpression(ExpressionBase):
    def __init__(self, op, param_map, data=None):
        self.uuid = utils.create_uuid()
        self.expr_val_seq_list = list()
        self.data = data if data is not None else ExprData()
        super().__init__(op=op, param_map=param_map)

    @classmethod
    def from_expr(cls, expr, **kwargs):
        param_map = {
            param: cls.from_expr(param_expr)
            for param, param_expr in expr.param_map.items()
        }
        return cls(
            op=expr.op,
            param_map=param_map,
            **kwargs,
        )

    @classmethod
    def from_expr_list(cls, expr_list):
        # Apply Common Subexpression Elimination to ExpressionBase before they
        # are run, and then get a bound reference of "execute" that can be
        # readily iterated over to get the results.
        return cls.cse(
            cls.from_expr(expr)
            for expr in expr_list
        )

    def _get_script(self, *args, **kwargs):
        return super()._get_script(*args, **kwargs,
            expr_val_seq_list=self.expr_val_seq_list
        )

    def get_id(self, mark_excep=False, marked_expr_val_set=set(), **kwargs):
        # Mark all the values that failed to be computed because of an
        # exception
        marked_expr_val_set = self.get_excep() if mark_excep else marked_expr_val_set

        return super().get_id(
            marked_expr_val_set=marked_expr_val_set,
            **kwargs
        )

    def find_expr_val_seq_list(self, param_map):
        def value_map(param_map):
            return ExprValParamMap(
                # Extract the actual value from ExprVal
                (param, expr_val.value)
                for param, expr_val in param_map.items()
            )
        param_map = value_map(param_map)

        # Find the results that are matching the param_map
        return [
            expr_val_seq
            for expr_val_seq in self.expr_val_seq_list
            # Check if param_map is a subset of the param_map
            # of the ExprVal. That allows checking for reusable parameters
            # only.
            if param_map.items() <= value_map(expr_val_seq.param_map).items()
        ]

    @classmethod
    def execute_all(cls, expr_list, *args, **kwargs):
        for comp_expr in cls.from_expr_list(expr_list):
            for expr_val in comp_expr.execute(*args, **kwargs):
                yield (comp_expr, expr_val)

    def execute(self, post_compute_cb=None):
        return self._execute([], post_compute_cb)

    def _execute(self, consumer_expr_stack, post_compute_cb):
        # Lazily compute the values of the Expression, trying to use
        # already computed values when possible

        # Check if we are allowed to reuse an instance that has already
        # been produced
        reusable = self.op.reusable

        def filter_param_exec_map(param_map, reusable):
            return OrderedDict(
                (param, param_expr._execute(
                    consumer_expr_stack=consumer_expr_stack + [self],
                    post_compute_cb=post_compute_cb,
                ))
                for param, param_expr in param_map.items()
                if param_expr.op.reusable == reusable
            )

        # Get all the generators for reusable parameters
        reusable_param_exec_map = filter_param_exec_map(self.param_map, True)

        # Consume all the reusable parameters, since they are generators
        for param_map in ExprValParamMap.from_gen_map_product(self, reusable_param_exec_map):
            # Check if some ExprVal are already available for the current
            # set of reusable parameters. Non-reusable parameters are not
            # considered since they would be different every time in any case.
            if reusable and not param_map.is_partial(ignore_error=True):
                # Check if we have already computed something for that
                # Expression and that set of parameter values
                expr_val_seq_list = self.find_expr_val_seq_list(param_map)
                if expr_val_seq_list:
                    # Reusable objects should have only one ExprValSeq
                    # that was computed with a given param_map
                    assert len(expr_val_seq_list) == 1
                    expr_val_seq = expr_val_seq_list[0]
                    yield from expr_val_seq.iter_expr_val()
                    continue

            # Only compute the non-reusable parameters if all the reusable one
            # are available, otherwise that is pointless
            if not param_map.is_partial():
                # Non-reusable parameters must be computed every time, and we
                # don't take their cartesian product since we have fresh values
                # for all operator calls.

                nonreusable_param_exec_map = filter_param_exec_map(self.param_map, False)
                param_map.update(ExprValParamMap.from_gen_map(self, nonreusable_param_exec_map))

            # Propagate exceptions if some parameters did not execute
            # successfully.
            if param_map.is_partial():
                expr_val = ExprVal(self, param_map)
                expr_val_seq = ExprValSeq.from_one_expr_val(
                    self, expr_val, param_map,
                )
                self.expr_val_seq_list.append(expr_val_seq)
                yield expr_val
                continue

            # If no value has been found, compute it and save the results in
            # a list.
            param_val_map = OrderedDict(
                # Extract the actual computed values wrapped in ExprVal
                (param, param_expr_val.value)
                for param, param_expr_val in param_map.items()
            )

            # Consumer operator is special and we provide the value for it,
            # instead of letting it computing its own value
            if self.op.callable_ is Consumer:
                try:
                    consumer = consumer_expr_stack[-2].op.callable_
                except IndexError:
                    consumer = None
                iterated = [ (None, consumer, NoValue) ]

            elif self.op.callable_ is ExprData:
                root_expr = consumer_expr_stack[0]
                expr_data = root_expr.data
                iterated = [ (expr_data.uuid, expr_data, NoValue) ]

            # Otherwise, we just call the operators with its parameters
            else:
                iterated = self.op.generator_wrapper(**param_val_map)

            iterator = iter(iterated)
            expr_val_seq = ExprValSeq(
                self, iterator, param_map,
                post_compute_cb
            )
            self.expr_val_seq_list.append(expr_val_seq)
            yield from expr_val_seq.iter_expr_val()

    def get_all_vals(self):
        return utils.flatten_seq(
            expr_val_seq.expr_val_list
            for expr_val_seq in self.expr_val_seq_list
        )

    def get_excep(self):
        return set(utils.flatten_seq(
            expr_val.get_excep()
            for expr_val in self.get_all_vals()
        ))

class ClassContext:
    def __init__(self, op_map, cls_map):
        self.op_map = op_map
        self.cls_map = cls_map

    @staticmethod
    def _build_cls_map(op_set, compat_cls):
        # Pool of classes that can be produced by the ops
        produced_pool = set(op.value_type for op in op_set)

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
        return {
            # Make sure the list is deduplicated by building a set first
            cls: sorted({
                subcls for subcls in produced_pool
                if compat_cls(subcls, cls)
            }, key=lambda cls: cls.__qualname__)
            for cls in cls_set
        }

    # Map of all produced types to a set of what operator can create them
    @staticmethod
    def _build_op_map(op_set, cls_map, forbidden_pattern_set):
        # Make sure that the provided PrebuiltOperator will be the only ones used
        # to provide their types
        only_prebuilt_cls = set(itertools.chain.from_iterable(
            # Augment the list of classes that can only be provided by a prebuilt
            # Operator with all the compatible classes
            cls_map[op.obj_type]
            for op in op_set
            if isinstance(op, PrebuiltOperator)
        ))

        op_map = dict()
        for op in op_set:
            param_map, produced = op.get_prototype()
            is_prebuilt_op = isinstance(op, PrebuiltOperator)
            if (
                (is_prebuilt_op or produced not in only_prebuilt_cls)
                and not utils.match_base_cls(produced, forbidden_pattern_set)
            ):
                op_map.setdefault(produced, set()).add(op)
        return op_map

    @staticmethod
    def _restrict_op_map(op_map, cls_map, restricted_pattern_set):
        cls_map = copy.copy(cls_map)

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
                # Make sure there is no other compatible type, so the only
                # operators that will be used to satisfy that dependency will
                # be one of the restricted_op_set item.
                cls_map[produced] = [produced]
                return restricted_op_set
            else:
                return op_set
        op_map = {
            produced: apply_restrict(produced, op_set, restricted_op_set, cls_map)
            for produced, op_set in op_map.items()
        }

        return (op_map, cls_map)

    @classmethod
    def from_op_set(cls, op_set, forbidden_pattern_set=set(), restricted_pattern_set=set(), compat_cls=issubclass):
        # Build the mapping of compatible classes
        cls_map = cls._build_cls_map(op_set, compat_cls)
        # Build the mapping of classes to producing operators
        op_map = cls._build_op_map(op_set, cls_map, forbidden_pattern_set)
        op_map, cls_map = cls._restrict_op_map(op_map, cls_map, restricted_pattern_set)

        return cls(
            op_map=op_map,
            cls_map=cls_map
        )

    def build_expr_list(self, result_op_seq,
            non_produced_handler='raise', cycle_handler='raise'):
        op_map = copy.copy(self.op_map)
        cls_map = {
            cls: compat_cls_set
            for cls, compat_cls_set in self.cls_map.items()
            # If there is at least one compatible subclass that is produced, we
            # keep it, otherwise it will mislead _build_expr into thinking the
            # class can be built where in fact it cannot
            if compat_cls_set & op_map.keys()
        }
        internal_cls_set = {Consumer, ExprData}
        for internal_cls in internal_cls_set:
            op_map[internal_cls] = {
                Operator(internal_cls, non_reusable_type_set=internal_cls_set)
            }
            cls_map[internal_cls] = [internal_cls]

        expr_list = list()
        for result_op in result_op_seq:
            expr_gen = self._build_expr(result_op, op_map, cls_map,
                op_stack = [],
                non_produced_handler=non_produced_handler,
                cycle_handler=cycle_handler,
            )
            for expr in expr_gen:
                if expr.validate(op_map):
                    expr_list.append(expr)

        # Apply CSE to get a cleaner result
        return Expression.cse(expr_list)

    @classmethod
    def _build_expr(cls, op, op_map, cls_map, op_stack, non_produced_handler, cycle_handler):
        new_op_stack = [op] + op_stack
        # We detected a cyclic dependency
        if op in op_stack:
            if cycle_handler == 'ignore':
                return
            elif callable(cycle_handler):
                cycle_handler(tuple(op.callable_ for op in new_op_stack))
                return
            elif cycle_handler == 'raise':
                raise CycleError('Cyclic dependency found: {path}'.format(
                    path = ' -> '.join(
                        op.name for op in new_op_stack
                    )
                ))
            else:
                raise ValueError('Invalid cycle_handler')

        op_stack = new_op_stack

        param_map, produced = op.get_prototype()
        if param_map:
            param_list, cls_list = zip(*param_map.items())
        # When no parameter is needed
        else:
            yield Expression(op, OrderedDict())
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
                    elif callable(non_produced_handler):
                        non_produced_handler(wanted_cls.__qualname__, op.name, param,
                            tuple(op.resolved_callable for op in op_stack)
                        )
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
                        raise ValueError('Invalid non_produced_handler')

        param_list = utils.remove_indices(param_list, ignored_indices)
        cls_combis = utils.remove_indices(cls_combis, ignored_indices)

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
                param_combis = itertools.product(*(cls._build_expr(
                        param_op, op_map, cls_map,
                        op_stack, non_produced_handler, cycle_handler,
                    ) for param_op in op_combi
                ))

                for param_combi in param_combis:
                    param_map = OrderedDict(zip(param_list, param_combi))

                    # If all parameters can be built, carry on
                    if len(param_map) == param_list_len:
                        yield Expression(op, param_map)

class Expression(ExpressionBase):
    def validate(self, op_map):
        type_map, valid = self._get_type_map()
        if not valid:
            return False

        # Check that the Expression does not involve 2 classes that are
        # compatible
        cls_bags = [set(cls_list) for cls_list in op_map.values()]
        cls_used = set(type_map.keys())
        for cls1, cls2 in itertools.product(cls_used, repeat=2):
            for cls_bag in cls_bags:
                if cls1 in cls_bag and cls2 in cls_bag:
                    return False

        return True

    def _get_type_map(self):
        type_map = dict()
        return (type_map, self._populate_type_map(type_map))

    def _populate_type_map(self, type_map):
        value_type = self.op.value_type
        # If there was already an Expression producing that type, the Expression
        # is not valid
        found_callable = type_map.get(value_type)
        if found_callable is not None and found_callable is not self.op.callable_:
            return False
        type_map[value_type] = self.op.callable_

        for param_expr in self.param_map.values():
            if not param_expr._populate_type_map(type_map):
                return False
        return True

class AnnotationError(Exception):
    pass

class PartialAnnotationError(AnnotationError):
    pass

class ForcedParamType:
    pass

class Operator:
    def __init__(self, callable_, non_reusable_type_set=None, tags_getter=None):
        if non_reusable_type_set is None:
            non_reusable_type_set = set()

        if not tags_getter:
            tags_getter = lambda v: {}
        self.tags_getter = tags_getter

        assert callable(callable_)
        self.callable_ = callable_

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

        self.reusable = self.value_type not in non_reusable_type_set

        # At that point, we can get the prototype safely as the object is
        # mostly initialized.

        # Special support of return type annotation for factory classmethod
        if self.is_factory_cls_method:
            # If the return annotation type is an (indirect) base class of
            # the original annotation, we replace the annotation by the
            # subclass That allows implementing factory classmethods
            # easily.
            self.annotations['return'] = self.resolved_callable.__self__

    @property
    def callable_globals(self):
        return self.resolved_callable.__globals__

    @property
    def signature(self):
        return inspect.signature(self.resolved_callable)

    def __repr__(self):
        return '<Operator of ' + str(self.callable_) + '>'

    def force_param(self, param_callable_map, tags_getter=None):
        prebuilt_op_set = set()
        for param, value_list in param_callable_map.items():
            # Get the most derived class that is in common between all
            # instances
            value_type = utils.get_common_base(type(v) for v in value_list)

            try:
                param_annot = self.annotations[param]
            except KeyError:
                pass
            else:
                # If there was an annotation, make sure the type we computed is
                # compatible with what the annotation specifies.
                assert issubclass(value_type, param_annot)

            # We do not inherit from value_type, since it may not always work,
            # e.g. subclassing bool is forbidden. Therefore, it is purely used
            # as a unique marker.
            class ParamType(ForcedParamType):
                pass

            # References to this type won't be serializable with pickle, but
            # instances will be. This is because pickle checks that only one
            # type exists with a given __module__ and __qualname__.
            ParamType.__name__ = value_type.__name__
            ParamType.__qualname__ = value_type.__qualname__
            ParamType.__module__ = value_type.__module__

            # Create an artificial new type that will only be produced by
            # the PrebuiltOperator
            self.annotations[param] = ParamType

            prebuilt_op_set.add(
                PrebuiltOperator(ParamType, value_list,
                    tags_getter=tags_getter
            ))

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

    def get_name(self, *args, **kwargs):
        try:
            return utils.get_name(self.callable_, *args, **kwargs)
        except AttributeError:
            return None

    def get_id(self, full_qual=True, qual=True, style=None):
        if style == 'rst':
            if self.is_factory_cls_method:
                qualname = utils.get_name(self.value_type, full_qual=True)
            else:
                qualname = self.get_name(full_qual=True)
            name = self.get_id(full_qual=full_qual, qual=qual, style=None)

            if self.is_class:
                role = 'class'
            elif self.is_method or self.is_static_method or self.is_cls_method:
                role = 'meth'
            else:
                role = 'func'

            return ':{role}:`{name}<{qualname}>`'.format(role=role, name=name, qualname=qualname)

        else:
            # Factory classmethods are replaced by the class name when not
            # asking for a qualified ID
            if not (qual or full_qual) and self.is_factory_cls_method:
                return utils.get_name(self.value_type, full_qual=full_qual, qual=qual)
            else:
                return self.get_name(full_qual=full_qual, qual=qual)

    @property
    def name(self):
        return self.get_name()

    @property
    def id_(self):
        return self.get_id()

    @property
    def mod_name(self):
        try:
            name = inspect.getmodule(self.unwrapped_callable).__name__
        except Exception:
            name = self.callable_globals['__name__']
        return name

    @property
    def src_loc(self):
        return utils.get_src_loc(self.unwrapped_callable)

    @property
    def value_type(self):
        return self.get_prototype()[1]

    @property
    def is_genfunc(self):
        return inspect.isgeneratorfunction(self.resolved_callable)

    @property
    def is_class(self):
        return inspect.isclass(self.unwrapped_callable)

    @property
    def is_static_method(self):
        callable_ = self.unwrapped_callable

        try:
            callable_globals = callable_.__globals__
        # __globals__ is only defined for functions
        except AttributeError:
            return False

        try:
            cls = utils._get_class_from_name(
                callable_.__qualname__.rsplit('.', 1)[0],
                namespace=callable_globals
            )
        except ValueError:
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
    def is_factory_cls_method(self):
        return self.is_cls_method and issubclass(self.unwrapped_callable.__self__, self.value_type)

    @property
    def generator_wrapper(self):
        if self.is_genfunc:
            @functools.wraps(self.callable_)
            def genf(*args, **kwargs):
                try:
                    has_yielded = False
                    for res in self.callable_(*args, **kwargs):
                        has_yielded = True
                        yield (utils.create_uuid(), res, NoValue)

                    # If no value at all were produced, we still need to yield
                    # something
                    if not has_yielded:
                        yield (utils.create_uuid(), NoValue, NoValue)

                except Exception as e:
                    yield (utils.create_uuid(), NoValue, e)
        else:
            @functools.wraps(self.callable_)
            def genf(*args, **kwargs):
                uuid_ = utils.create_uuid()
                # yield one value and then return
                try:
                    val = self.callable_(*args, **kwargs)
                    yield (uuid_, val, NoValue)
                except Exception as e:
                    yield (uuid_, NoValue, e)

        return genf

    def get_prototype(self):
        sig = self.signature
        first_param = utils.take_first(sig.parameters)
        annotation_map = utils.resolve_annotations(self.annotations, self.callable_globals)
        pristine_annotation_map = copy.copy(annotation_map)

        extra_ignored_param = set()
        # If it is a class
        if self.is_class:
            produced = self.unwrapped_callable
            # Get rid of "self", since annotating it with the class would lead to
            # infinite recursion when computing it signature. It will be handled
            # by execute() directly. Also, "self" is not a parameter of the
            # class when it is called, so it makes sense not to include it.
            annotation_map.pop(first_param, None)
            extra_ignored_param.add(first_param)

        # If it is any other callable
        else:
            # When we have a method, we fill the annotations of the 1st
            # parameter with the name of the class it is defined in
            if self.is_method and first_param is not NoValue:
                cls_name = self.resolved_callable.__qualname__.split('.')[0]
                self.annotations[first_param] = cls_name

            # No return annotation is accepted and is equivalent to None return
            # annotation
            produced = annotation_map.get('return')
            # "None" annotation is accepted, even though it is not a type
            # strictly speaking
            if produced is None:
                produced = type(None)

        # Recompute after potentially modifying the annotations
        annotation_map = utils.resolve_annotations(self.annotations, self.callable_globals)

        # Remove the return annotation, since we are handling that separately
        annotation_map.pop('return', None)

        # Check that we have annotations for all parameters that are not ignored
        for param, param_spec in sig.parameters.items():
            if (
                param not in annotation_map and
                param not in extra_ignored_param and
                param not in self.ignored_param
            ):
                # If some parameters are annotated but not all, we raise a
                # slightly different exception to allow better reporting
                if pristine_annotation_map:
                    excep_cls = PartialAnnotationError
                else:
                    excep_cls = AnnotationError

                raise excep_cls('Missing annotation for "{param}" parameters of operator "{op}"'.format(
                    param = param,
                    op = self.name,
                ))

        # Iterate over keys and values of "mapping" in the same order as "keys"
        def iter_by_keys(mapping, keys):
            for key in keys:
                try:
                    yield key, mapping[key]
                except KeyError:
                    pass

        # Use an OrderedDict to retain the declaration order of parameters
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
            if isinstance(obj, FrozenExprVal):
                uuid_ = obj.uuid
                obj = obj.value
            else:
                uuid_ = utils.create_uuid()

            uuid_list.append(uuid_)
            obj_list_.append(obj)

        self.obj_list = obj_list_
        self.uuid_list = uuid_list
        self.obj_type = obj_type
        self._id = id_

        # Placeholder for the signature
        def callable_() -> self.obj_type:
            pass
        super().__init__(callable_, **kwargs)

    def get_name(self, *args, **kwargs):
        return None

    def get_id(self, *args, style=None, **kwargs):
        return self._id or utils.get_name(self.obj_type, *args, **kwargs)

    @property
    def src_loc(self):
        return utils.get_src_loc(self.value_type)

    @property
    def is_genfunc(self):
        return len(self.obj_list) > 1

    @property
    def is_method(self):
        return False

    @property
    def generator_wrapper(self):
        def genf():
            yield from zip(self.uuid_list, self.obj_list, itertools.repeat(NoValue))
        return genf

class ExprValSeq:
    def __init__(self, expr, iterator, param_map, post_compute_cb=None):
        self.expr = expr
        assert isinstance(iterator, collections.abc.Iterator)
        self.iterator = iterator
        self.expr_val_list = []
        self.param_map = param_map
        self.post_compute_cb = post_compute_cb

    @classmethod
    def from_one_expr_val(cls, expr, expr_val, param_map):
        iterated = [
            (expr_val.uuid, expr_val.value, expr_val.excep)
        ]
        new = cls(
            expr=expr,
            iterator=iter(iterated),
            param_map=param_map,
            # no post_compute_cb, since we are not really going to compute
            # anything
            post_compute_cb=None,
        )
        # consume the iterator to make sure new.expr_val_list is updated
        for _ in new.iter_expr_val():
            pass
        return new

    def iter_expr_val(self):
        callback = self.post_compute_cb
        if not callback:
            callback = lambda x, reused: None

        def yielder(iteratable, reused):
            for x in iteratable:
                callback(x, reused=reused)
                yield x

        # Yield existing values
        yield from yielder(self.expr_val_list, True)

        # Then compute the remaining ones
        if self.iterator:
            for uuid_, value, excep in self.iterator:
                expr_val = ExprVal(
                    expr=self.expr,
                    param_map=self.param_map,
                    value=value,
                    excep=excep,
                    uuid=uuid_,
                )
                callback(expr_val, reused=False)

                self.expr_val_list.append(expr_val)
                expr_val_list_len = len(self.expr_val_list)
                yield expr_val

                # If expr_val_list length has changed, catch up with the values
                # that were computed behind our back, so that this generator is
                # reentrant.
                if expr_val_list_len != len(self.expr_val_list):
                    # This will yield all values, even if the list grows while
                    # we are yielding the control back to another piece of code.
                    yield from yielder(
                        self.expr_val_list[expr_val_list_len:],
                        True
                    )

            self.iterator = None


class ExprValParamMap(OrderedDict):
    def is_partial(self, ignore_error=False):
        def is_partial(expr_val):
            # Some arguments are missing: there was no attempt to compute
            # them because another argument failed to be computed
            if isinstance(expr_val, UnEvaluatedExprVal):
                return True

            # Or computation did take place but failed
            if expr_val.value is NoValue and not ignore_error:
                return True

            return False

        return any(
            is_partial(expr_val)
            for expr_val in self.values()
        )

    @classmethod
    def from_gen_map(cls, expr, param_gen_map):
        # Pre-fill UnEvaluatedExprVal with in case we exit the loop early
        param_map = cls(
            (param, UnEvaluatedExprVal(expr))
            for param in param_gen_map.keys()
        )

        for param, generator in param_gen_map.items():
            val = next(generator)
            # There is no point in computing values of the other generators if
            # one failed to produce a useful value
            if val.value is NoValue:
                break
            else:
                param_map[param] = val

        return param_map

    @classmethod
    def from_gen_map_product(cls, expr, param_gen_map):
        """
        Yield :class:`collections.OrderedDict` for each combination of parameter
        values.

        :param param_gen_map: Mapping of parameter names to an iterator that is ready
            to generate the possible values for the generator.
        :type param_gen_map: collections.OrderedDict

        """
        if not param_gen_map:
            yield cls()
        else:
            # Since param_gen_map is an OrderedDict, we will always consume
            # parameters in the same order
            param_list, gen_list = zip(*param_gen_map.items())
            for values in cls._product(expr, gen_list):
                yield cls(zip(param_list, values))

    @classmethod
    def _product(cls, expr, gen_list):
        """
        Similar to the cartesian product provided by itertools.product, with
        special handling of NoValue and some checks on the yielded sequences.

        It will only yield the combinations of values that are validated by
        :meth:`validate`.
        """
        def validated(generator):
            """
            Ensure we only yield valid lists of :class:`ExprVal`
            """
            for expr_val_list in generator:
                if ExprVal.validate(expr_val_list):
                    yield expr_val_list
                else:
                    continue

        def acc_product(product_generator, generator):
            """
            Combine a "cartesian-product-style" generator with a plain
            generator, giving a new "cartesian-product-style" generator.
            """
            # We will need to use it more than once in the inner loop, so it
            # has to be "restartable" (like a list, and unlike a plain
            # iterator)
            product_iter = utils.RestartableIter(product_generator)
            for expr_val in generator:
                # The value is not useful, we can return early without calling
                # the other generators. That avoids spending time computing
                # parameters if they won't be used anyway.
                if expr_val.value is NoValue:
                    # Returning an incomplete list will make the calling code
                    # aware that some values were not computed at all
                    yield [expr_val]
                else:
                    for expr_val_list in product_iter:
                        yield [expr_val] + expr_val_list

        def reducer(product_generator, generator):
            yield from validated(acc_product(product_generator, generator))

        def initializer():
            yield []

        # We need to pad since we may truncate the list of values we yield if
        # we detect an error in one of them.
        def pad(generator, length):
            for xs in generator:
                xs.extend(
                    UnEvaluatedExprVal(expr)
                    for i in range(length - len(xs))
                )
                yield xs

        # reverse the gen_list so we get the rightmost generator varying the
        # fastest. Typically, margins-like parameter on which we do sweeps are
        # on the right side of the parameter list (to have a default value)
        return pad(
            functools.reduce(reducer, reversed(gen_list), initializer()),
            len(gen_list)
        )

class ExprValBase(collections.abc.Mapping):
    def __init__(self, param_map, value, excep):
        self.param_map = param_map
        self.value = value
        self.excep = excep

    def get_by_predicate(self, predicate):
        return list(self._get_by_predicate(predicate))

    def _get_by_predicate(self, predicate):
        if predicate(self):
            yield self

        for val in self.param_map.values():
            yield from val._get_by_predicate(predicate)

    def get_excep(self):
        """
        Get all the failed parents.
        """
        def predicate(val):
            return val.excep is not NoValue

        return self.get_by_predicate(predicate)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        # consistent with definition of __eq__
        return id(self)

    def __getitem__(self, k):
        return self.param_map[k]

    def __len__(self):
        return len(self.param_map)

    def __iter__(self):
        return self.param_map.keys()

class FrozenExprVal(ExprValBase):
    def __init__(self,
            param_map, value, excep, uuid,
            callable_qualname, callable_name, recorded_id_map,
        ):
        self.uuid = uuid
        self.callable_qualname = callable_qualname
        self.callable_name = callable_name
        self.recorded_id_map = recorded_id_map
        super().__init__(param_map=param_map, value=value, excep=excep)

        if self.excep is not NoValue:
            self.excep_tb = utils.format_exception(self.excep)
        else:
            self.excep_tb = None

    @property
    def type_names(self):
        return [
            utils.get_name(type_, full_qual=True)
            for type_ in utils.get_mro(type(value))
            if type_ is not object
        ]

    @classmethod
    def from_expr_val(cls, expr_val, hidden_callable_set=None):
        value = expr_val.value if utils.is_serializable(expr_val.value) else NoValue
        excep = expr_val.excep if utils.is_serializable(expr_val.excep) else NoValue

        op = expr_val.expr.op

        # Reloading these values will lead to issues, and they are regenerated
        # for any new Expression that would be created anyway.
        if op.callable_ in (ExprData, Consumer):
            value = NoValue
            excep = NoValue

        callable_qualname = op.get_name(full_qual=True)
        callable_name = op.get_name(full_qual=False, qual=False)

        # Pre-compute all the IDs so they are readily available once the value
        # is deserialized
        recorded_id_map = dict()
        for full_qual, qual, with_tags in itertools.product((True, False), repeat=3):
            key = cls._make_id_key(
                full_qual=full_qual,
                qual=qual,
                with_tags=with_tags
            )
            recorded_id_map[key] = expr_val.get_id(
                **dict(key),
                hidden_callable_set=hidden_callable_set,
            )

        param_map = ExprValParamMap(
            (param, cls.from_expr_val(
                param_expr_val,
                hidden_callable_set=hidden_callable_set,
            ))
            for param, param_expr_val in expr_val.param_map.items()
        )

        froz_val = cls(
            uuid=expr_val.uuid,
            value=value,
            excep=excep,
            callable_qualname=callable_qualname,
            callable_name=callable_name,
            recorded_id_map=recorded_id_map,
            param_map=param_map,
        )

        return froz_val

    @staticmethod
    # Since tuples are immutable, reuse the same tuple by memoizing the
    # function. That allows more compact serialized representation in both YAML
    # and Pickle.
    @utils.once
    def _make_id_key(**kwargs):
        return tuple(sorted(kwargs.items()))

    def get_id(self, full_qual=True, qual=True, with_tags=True):
        full_qual = full_qual and qual
        key = self._make_id_key(
            full_qual=full_qual,
            qual=qual,
            with_tags=with_tags
        )
        return self.recorded_id_map[key]

class PrunedFrozVal(FrozenExprVal):
    def __init__(self, froz_val):
        super().__init__(
            param_map={},
            value=NoValue,
            excep=NoValue,
            uuid=froz_val.uuid,
            callable_qualname=froz_val.callable_qualname,
            callable_name=froz_val.callable_name,
            recorded_id_map=copy.copy(froz_val.recorded_id_map),
        )

class FrozenExprValSeq(collections.abc.Sequence):
    def __init__(self, froz_val_list, param_map):
        self.froz_val_list = froz_val_list
        self.param_map = param_map

    def __getitem__(self, k):
        return self.froz_val_list[k]

    def __len__(self):
        return len(self.froz_val_list)

    @classmethod
    def from_expr_val_seq(cls, expr_val_seq, **kwargs):
        return cls(
            froz_val_list=[
                FrozenExprVal.from_expr_val(expr_val, **kwargs)
                for expr_val in expr_val_seq.expr_val_list
            ],
            param_map={
                param: FrozenExprVal.from_expr_val(expr_val, **kwargs)
                for param, expr_val in expr_val_seq.param_map.items()
            }
        )

    @classmethod
    def from_expr_list(cls, expr_list, **kwargs):
        expr_val_seq_list = utils.flatten_seq(expr.expr_val_seq_list for expr in expr_list)
        return [
            cls.from_expr_val_seq(expr_val_seq, **kwargs)
            for expr_val_seq in expr_val_seq_list
        ]


class ExprVal(ExprValBase):
    def __init__(self, expr, param_map,
        value=NoValue, excep=NoValue, uuid=None,
    ):
        self.uuid = uuid if uuid is not None else utils.create_uuid()
        self.expr = expr
        super().__init__(param_map=param_map, value=value, excep=excep)

    def format_tags(self):
        tag_map = self.expr.op.tags_getter(self.value)
        if tag_map:
            return ''.join(
                '[{}={}]'.format(k, v) if k else '[{}]'.format(v)
                for k, v in sorted(tag_map.items())
            )
        else:
            return ''

    @classmethod
    def validate(cls, expr_val_list):
        expr_map = {}
        def update_map(expr_val1):
            # The check does not apply for non-reusable operators, since it is
            # expected that the same expression may reference multiple values
            # of the same Expression.
            if not expr_val1.expr.op.reusable:
                return

            expr_val2 = expr_map.setdefault(expr_val1.expr, expr_val1)
            # Check that there is only one ExprVal per Expression, for all
            # expressions that were (indirectly) involved into computation of
            # expr_val_list
            if expr_val2 is not expr_val1:
                raise ValueError

        try:
            for expr_val in expr_val_list:
                # DFS traversal
                expr_val.get_by_predicate(update_map)
        except ValueError:
            return False
        else:
            return True

    def get_id(self, *args, with_tags=True, **kwargs):
        return self.expr.get_id(
            with_tags=with_tags,
            expr_val=self,
            *args, **kwargs
        )

class UnEvaluatedExprVal(ExprVal):
    def __init__(self, expr):
        super().__init__(
            expr=expr,
            param_map=ExprValParamMap(),
            uuid=None,
            value=NoValue,
            excep=NoValue,
        )

class Consumer:
    def __init__(self):
        pass

class ExprData(dict):
    def __init__(self):
        super().__init__()
        self.uuid = utils.create_uuid()

