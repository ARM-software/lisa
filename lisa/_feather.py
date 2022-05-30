# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022, Arm Limited and contributors.
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
import itertools

import pyarrow as pa
from pyarrow.compute import struct_field
import pyarrow.feather

import pandas as pd


def _do_make_expand(field):
    typ = field.type

    if pa.types.is_union(typ):
        _names, _expands = zip(*(
            _do_make_expand(field)
            for field in typ
        ))

        tags = pa.array([
            typ.field(i).name
            for i in typ.type_codes
        ])

        def expand(arr):
            if arr:
                # We need to combine the chunks in order to get back a
                # pa.UnionArray, which is the only way I found to get access to the
                # type_codes (they are not exposed as a child on the ChunkedArray)
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()

                first = struct_field(arr, [0])
                unpacked = itertools.chain(
                    (first,),
                    (
                        struct_field(arr, [i])
                        for i in range(1, typ.num_fields)
                    )
                )

                # Build the equivalent of pandas Category to map the union
                # type_codes to their actual name.
                tag_array = pa.DictionaryArray.from_arrays(
                    # Use a numpy view in the array as "mask" parameters is
                    # currently unsupported for pyarrow arrays.
                    arr.type_codes.to_numpy(),
                    dictionary=tags,
                    # type_code == 0 encodes both for the first variant and
                    # also lack of data. The only way to distinguish both
                    # is to look at the the bool column associated with variant
                    # 0 and check if it is null or something else.
                    mask=pa.compute.and_(
                        first.is_null(),
                        pa.compute.equal(arr.type_codes, 0)
                    ).to_pandas()
                )

                return itertools.chain(
                    (tag_array,),
                    itertools.chain.from_iterable(
                        expand(arr)
                        for arr, expand in zip(unpacked, _expands)
                    )
                )
            else:
                return [[] * (len(_expands) + 1)]

        columns = [field.name] + [
            f'{field.name}.{subname}' if field.name else subname
            for subnames in _names
            for subname in subnames
        ]

    elif pa.types.is_struct(typ):
        _names, _expands = zip(*(
            _do_make_expand(field)
            for field in typ
        ))

        def expand(arr):
            if arr:

                # We could use arr.flatten() instead of this combine_chunks()
                # followed by arr.field(), but it unfortunately crashes with
                # some inputs:
                # https://github.com/apache/arrow/issues/14736
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()

                unpacked = (
                    arr.field(i)
                    for i in range(typ.num_fields)
                )

                return itertools.chain.from_iterable(
                    expand(arr)
                    for arr, expand in zip(unpacked, _expands)
                )

            else:
                return [[] * len(_expands)]

        columns = [
            f'{field.name}.{subname}' if field.name else subname
            for subnames in _names
            for subname in subnames
        ]

    else:
        def expand(arr):
            return (arr,)
        columns = [field.name]

    return columns, expand


def _make_expand(schema):
    fields = list(schema)
    if fields:
        _columns, _expands = zip(*(
            _do_make_expand(field)
            for field in fields
        ))

        def expand(table):
            return itertools.chain.from_iterable(
                _expand(arr)
                for arr, _expand in zip(table, _expands)
            )

        columns = list(itertools.chain.from_iterable(_columns))
    else:
        def expand(table):
            return table
        columns = []
    return columns, expand


def _expand_table(table):
    schema = table.schema
    columns, expand = _make_expand(schema)
    arrays = expand(table)
    return zip(columns, arrays)


def table_to_df(table, columns=None):
    cols = _expand_table(table)
    if cols:
        names, arrays = zip(*cols)
        if columns:
            columns = set(columns)
            cols = zip(names, arrays)
            cols = filter(lambda x: x[0] in columns, cols)
            names, arrays = zip(*cols)
    else:
        names = []
        arrays = []


    table = pa.Table.from_arrays(arrays, names=names)

    # Use nullable dtypes for all types
    dtype_mapping = {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.float32(): pd.Float32Dtype(),
        pa.float64(): pd.Float64Dtype(),
        pa.string(): pd.StringDtype(),
    }
    return table.to_pandas(
        types_mapper=dtype_mapping.get,
        # This serves 2 purposes:
        # 1. Rust enum tags are encoded as a DictionaryArray, and we want to
        #    map those to pandas Categorical. Since they are always strings, we
        #    can conveniently just set this flag.
        # 2. Strings we use in analysis are 99% categorical, like a task's
        #    comm. Using Categorical pandas dtype for them will save a lot of
        #    memory and will make equality-based selection faster.
        strings_to_categorical=True
    )


def load_feather(path, columns=None):
    table = pa.feather.read_table(path, memory_map=True)
    return table_to_df(table, columns=columns)
