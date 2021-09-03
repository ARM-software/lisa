# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020, Arm Limited and contributors.
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
import uuid
import functools
from operator import itemgetter
import contextlib
from math import nan
import itertools
from itertools import combinations
from collections import OrderedDict
import warnings

import scipy.stats
import pandas as pd
import numpy as np
import holoviews as hv
import holoviews.operation
from bokeh.models import HoverTool

from lisa.utils import Loggable, memoized, FrozenDict, deduplicate, fold
from lisa.datautils import df_split_signals, df_make_empty_clone, df_filter, df_find_redundant_cols

# Ensure hv.extension() is called
import lisa.notebook

# Expose bokeh option "level" to workaround:
# https://github.com/holoviz/holoviews/issues/1968
hv.Store.add_style_opts(
    hv.ErrorBars,
    ['level'],
    backend='bokeh'
)


def series_mean_stats(series, kind, confidence_level=0.95):
    """
    Compute the mean along with the a confidence interval based on the T-score.

    :returns: A tuple with:

        0. The mean
        1. The standard deviation, or its equivalent
        2. The standard error of the mean, or its equivalent
           (Harmonic Standard Error, Geometric Standard Error).
        3. The interval, as an 2-tuple of +/- values

    :param kind: Kind of mean to use:
        * ``arithmetic``
        * ``harmonic``
        * ``geometric``
    :type kind: str

    :param confidence_level: Confidence level of the confidence interval.
    :type confidence_level: float

    """
    if kind == 'geometric':
        pre = np.log
        post = np.exp
    elif kind == 'harmonic':
        pre = lambda x: 1 / x
        post = pre
    elif kind == 'arithmetic':
        pre = lambda x: x
        post = pre
    else:
        raise ValueError(f'Unrecognized kind of mean: {kind}')

    series = pre(series)

    mean = series.mean()
    sem = scipy.stats.sem(series)
    std = series.std()
    interval = scipy.stats.t.interval(
        confidence_level,
        len(series) - 1,
        loc=mean,
        scale=sem,
    )
    # Convert it into a +/- format
    interval = [
        abs(bound - mean)
        for bound in interval
    ]
    mean = post(mean)
    sem = post(sem)
    std = post(std)
    interval = tuple(sorted(map(post, interval)))
    return (mean, std, sem, interval)

def guess_mean_kind(unit, control_var):
    """
    Guess which kind of mean should be used to summarize results in the given
    unit.

    :returns: ``'arithmetic'`` if an arithmetic mean should be used, or
        ``'harmonic'``. Geometric mean uses cannot be inferred by this
        function.

    :param unit: Unit of the values, e.g. ``'km/h'``.
    :type unit: str

    :param control_var: Control variable, i.e. variable that is fixed during
        the experiment. For example, in a car speed experiment, the control
        variable could be the distance (fixed distance), or the time. In that case,
        we would have ``unit='km/h'`` and ``control_var='h'`` if the time was
        fixed, or ``control_var='km'`` if the distance was fixed.
    :type control_var: str
    """
    if unit is None or control_var is None:
        kind = 'arithmetic'
    else:
        if '(' in unit or ')' in unit:
            raise ValueError('Units containing parenthesis are not allowed')

        split_unit = unit.split('/')
        if len(split_unit) == 1:
            kind = 'arithmetic'
        else:
            try:
                pos = split_unit.index(control_var)
            except ValueError:
                # Default to arithmetic
                kind = 'arithmetic'
            else:
                is_divisor = bool(pos % 2)
                if is_divisor:
                    kind = 'arithmetic'
                else:
                    kind = 'harmonic'

    return kind


class _Unit:
    def __init__(self, name, normalizable=True):
        self.name = name
        self.normalizable = normalizable


class Stats(Loggable):
    """
    Compute the statistics on an input :class:`pandas.DataFrame` in "database"
    format.

    :param df: Dataframe in database format, i.e. meaningless index, and values
        in a given column with the other columns used as tags.

        .. note:: Redundant tag columns (aka that are equal) will be removed
            from the dataframe.
    :type df: pandas.DataFrame

    :param value_col: Name of the column containing the values.
    :type value_col: str

    :param ref_group: Reference group used to compare the other groups against.
        It's format is ``dict(tag_column_name, tag_value)``. The comparison
        will be made on subgroups built out of all the other tag columns, with
        the reference subgroups being the one matching that dictionary. If the
        tag value is ``None``, the key will only be used for grouping in
        graphs. Comparison will add the following statistics:

            * A 2-sample Komolgorov-Smirnov test ``'ks2samp_test'`` column.
              This test is non-parametric and checks for difference in
              distributions. The only assumption is that the distribution is
              continuous, which should suit almost all use cases
            * Most statistics will be normalized against the reference group as
              a difference percentage, except for a few non-normalizable
              values.

        .. note:: The group referenced must exist, otherwise unexpected
            behaviours might occur.

    :type ref_group: dict(str, object)

    :param filter_rows: Filter the given :class:`pandas.DataFrame` with a dict
        of `{"column": value)` that rows has to match to be selected.
    :type filter_rows: dict(object, object) or None

    :param compare: If ``True``, normalize most statistics as a percentage of
        change compared to ``ref_group``.
    :type compare: bool

    :param agg_cols: Columns to aggregate on. In a sense, the given columns will
        be treated like a compound iteration number. Defaults to:

            * ``iteration`` column if available, otherwise
            * All the tag columns that are neither the value nor part of the
              ``ref_group``.

    :type agg_cols: list(str)

    :param mean_ci_confidence: Confidence level used to establish the mean
        confidence interval, between ``0`` and ``1``.
    :type mean_ci_confidence: float

    :param stats: Dictionnary of statistical functions to summarize each value
        group formed by tag columns along the aggregation columns. If ``None``
        is given as value, the name will be passed to
        :meth:`pandas.core.groupby.GroupBy.agg`. Otherwise, the provided
        function will be run.

        .. note:: One set of keys is special: ``'mean'``, ``'std'`` and
            ``'sem'``. When value ``None`` is used, a custom function is used
            instead of the one from :mod:`pandas`, which will compute other
            related statistics and provide a confidence interval. An attempt
            will be made to guess the most appropriate kind of mean to use
            using the ``mean_kind_col``, ``unit_col`` and ``control_var_col``:

                * The mean itself, as:

                    * ``'mean'`` (arithmetic)
                    * ``'hmean'`` (harmonic)
                    * ``'gmean'`` (geometric)

                * The Standard Error of the Mean (SEM):

                    * ``'sem'`` (arithmetic)
                    * ``'hse'`` (harmonic)
                    * ``'gse'`` (geometric)

                * The standard deviation:

                    * ``'std'`` (arithmetic)
                    * ``'hsd'`` (harmonic)
                    * ``'gsd'`` (geometric)

    :type stats: dict(str, str or collections.abc.Callable)

    :param stat_col: Name of the column used to hold the name of the statistics
        that are computed.
    :type stat_col: str

    :param unit_col: Name of the column holding the unit of each value (as a string).
    :type unit_col: str

    :param ci_cols: Name of the two columns holding the confidence interval for each
        computed statistics.
    :type ci_cols: tuple(str, str)

    :param control_var_col: Name of the column holding the control variable
        name in the experiment leading to the given value.
        .. seealso:: :func:`guess_mean_kind`
    :param control_var_col: str

    :param mean_kind_col: Type of mean to be used to summarize this value.

        .. note:: Unless geometric mean is used, ``unit_col`` and
            ``control_var_col`` should be used to make things more obvious and
            reduce risks of confusion.

    :type mean_kind_col: str

    :param non_normalizable_units: List of units that cannot be normalized
        against the reference group.
    :type non_normalizable_units: list(str)

    **Examples**::

        import pandas as pd

        # The index is meaningless, all what matters is to uniquely identify
        # each row using a set of tag columns, such as 'board', 'kernel',
        # 'iteration', ...
        df = pd.DataFrame.from_records(
            [
                ('juno', 'kernel1', 'bench1', 'score1', 1, 42, 'frame/s', 's'),
                ('juno', 'kernel1', 'bench1', 'score1', 2, 43, 'frame/s', 's'),
                ('juno', 'kernel1', 'bench1', 'score2', 1, 420, 'frame/s', 's'),
                ('juno', 'kernel1', 'bench1', 'score2', 2, 421, 'frame/s', 's'),
                ('juno', 'kernel1', 'bench2', 'score',  1, 54, 'foobar', ''),
                ('juno', 'kernel2', 'bench1', 'score1', 1, 420, 'frame/s', 's'),
                ('juno', 'kernel2', 'bench1', 'score1', 2, 421, 'frame/s', 's'),
                ('juno', 'kernel2', 'bench1', 'score2', 1, 4200, 'frame/s', 's'),
                ('juno', 'kernel2', 'bench1', 'score2', 2, 4201, 'frame/s', 's'),
                ('juno', 'kernel2', 'bench2', 'score',  1, 540, 'foobar', ''),

                ('hikey','kernel1', 'bench1', 'score1', 1, 42, 'frame/s', 's'),
                ('hikey','kernel1', 'bench1', 'score2', 1, 420, 'frame/s', 's'),
                ('hikey','kernel1', 'bench2', 'score',  1, 54, 'foobar', ''),
                ('hikey','kernel2', 'bench1', 'score1', 1, 420, 'frame/s', 's'),
                ('hikey','kernel2', 'bench1', 'score2', 1, 4200, 'frame/s', 's'),
                ('hikey','kernel2', 'bench2', 'score',  1, 540, 'foobar', ''),
            ],
            columns=['board', 'kernel', 'benchmark', 'metric', 'iteration', 'value', 'unit', 'fixed'],
        )


        # Get a Dataframe will all the default statistics.
        Stats(df).df

        # Use a ref_group will also compare other groups against it
        Stats(df, ref_group={'board': 'juno', 'kernel': 'kernel1'}).df
    """


    _STATS_UNIT = {
        'ks2samp_test': _Unit('pval', normalizable=False),
        'count': _Unit('samples', normalizable=True),
    }

    def __init__(self,
        df,
        value_col='value',
        ref_group=None,
        filter_rows=None,
        compare=True,
        agg_cols=None,
        mean_ci_confidence=None,
        stats=None,
        stat_col='stat',
        unit_col='unit',
        ci_cols=('ci_minus', 'ci_plus'),
        control_var_col='fixed',
        mean_kind_col='mean_kind',
        non_normalizable_units={
            unit.name
            for unit in _STATS_UNIT.values()
            if not unit.normalizable
        },
    ):
        if df.empty:
            raise ValueError('Empty dataframes are not handled')

        if filter_rows:
            df = df_filter(df, filter_rows)

        ref_group = dict(ref_group or {}) or {}

        # Columns controlling the behavior of this class, but that are not tags
        # nor values
        tweak_cols = {mean_kind_col, control_var_col}

        tag_cols = sorted(
            (set(df.columns) - {value_col, *ci_cols} - tweak_cols) | {unit_col}
        )

        # Find tag columns that are 100% correlated to ref_group keys, and add
        # them to the ref_group. Otherwise, it will break the reference
        # subgroup computation, since the subgroup found in non-ref groups will
        # not have any equivalent in the reference subgroup.
        for col, ref in list(ref_group.items()):
            redundant = df_find_redundant_cols(
                df,
                col,
                cols=sorted(set(tag_cols) - set(agg_cols or []) - {unit_col} - tweak_cols),
            )
            for _col, mapping in redundant.items():
                _ref = ref_group.get(_col)
                # If ref is None, we want None as a corresponding value
                corresponding = mapping.get(ref)
                if _ref == corresponding:
                    pass
                elif _ref is None:
                    ref_group[_col] = corresponding
                else:
                    raise ValueError(f'The ref_group key {col}={ref} is incompatible with {_col}={_ref}, as both columns are equivalent')

        group_cols = list(ref_group.keys())

        # TODO: see if the grouping machinery can be changed to accomodate redundant tags
        # Having duplicate tags will break various grouping mechanisms, so we
        # need to get rid of them
        for col1, col2 in combinations(tag_cols.copy(), 2):
            try:
                if (df[col1] == df[col2]).all():
                    if col1 not in ref_group:
                        to_remove = col1
                    elif col2 not in ref_group:
                        to_remove = col2
                    elif ref_group[col1] == ref_group[col2]:
                        to_remove = col2
                        ref_group.pop(to_remove)
                    else:
                        raise ValueError(f'ref_group has different values for "{col1}" and "{col2}" but the columns are equal')

                    df = df.drop(columns=[to_remove])
                else:
                    to_remove = None
            except KeyError:
                pass
            else:
                if to_remove is not None:
                    try:
                        tag_cols.remove(to_remove)
                    except ValueError:
                        pass

        # Check that tags are sufficient to describe the data, so that we don't
        # end up with 2 different values for the same set of tags
        duplicated_tags_size = df.groupby(tag_cols, observed=True).size()
        duplicated_tags_size = duplicated_tags_size[duplicated_tags_size > 1]
        if not duplicated_tags_size.empty:
            raise ValueError(f'Same tags applied to more than one value, another tag column is needed to distinguish them:\n{duplicated_tags_size}')

        if agg_cols:
            pass
        # Default to "iteration" if there was no ref group nor columns to
        # aggregate over
        elif 'iteration' in df.columns:
            agg_cols = ['iteration']
        # Aggregate over all tags that are not part of the ref group, since the
        # ref group keys are the tags that will remain after aggregation
        elif group_cols:
            agg_cols = sorted(set(tag_cols) - set(group_cols))
        else:
            raise ValueError('No aggregation column can be inferred. Either pass a ref_group or agg_cols')

        agg_cols = sorted(set(agg_cols) - {value_col, unit_col})
        if not agg_cols:
            raise ValueError('No aggregation columns have been selected, ensure that each special column has only one use')

        # Ultimately, the tags we want to have in the stat dataframe will not
        # include the one we aggregated over
        stat_tag_cols = [
            tag
            for tag in tag_cols
            if tag not in agg_cols
        ]

        # Sub groups that allows treating tag columns that are not part of
        # the group not as an aggregation column
        sub_group_cols = set(stat_tag_cols) - set(group_cols)
        plot_group_cols = sub_group_cols - {unit_col}

        self._orig_df = df
        self._stats = stats or {
            'median': None,
            'count': None,
            # This one is custom and not from pandas
            'mean': None,
            'std': None,
        }
        self._ref_group = ref_group
        self._group_cols = group_cols
        self._compare = compare and bool(ref_group)
        self._val_col = value_col
        self._tag_cols = tag_cols
        self._stat_tag_cols = stat_tag_cols
        self._sub_group_cols = sub_group_cols
        self._plot_group_cols = plot_group_cols
        self._agg_cols = agg_cols
        self._stat_col = stat_col
        self._mean_kind_col = mean_kind_col
        self._mean_ci_confidence = 0.95 if mean_ci_confidence is None else mean_ci_confidence
        self._unit_col = unit_col
        self._control_var_col = control_var_col
        self._tweak_cols = tweak_cols
        self._ci_cols = ci_cols
        self._non_normalizable_units = non_normalizable_units

    @staticmethod
    def _restrict_cols(cols, df):
        """
        Restrict the given list of columns to columns actually available in df.
        """
        return sorted(set(cols) & set(df.columns))

    def _df_remove_tweak_cols(self, df):
        for col in self._tweak_cols:
            with contextlib.suppress(KeyError):
                df = df.drop(columns=col)
        return df

    def _df_format(self, df):
        tag_cols = self._restrict_cols(self._stat_tag_cols, df)
        # Group together lines for each given tag
        df = df.sort_values(by=tag_cols, ignore_index=True)

        # Reorder columns
        cols = deduplicate(
            deduplicate(
                tag_cols +
                [self._stat_col, self._val_col, self._unit_col, self._control_var_col, self._mean_kind_col],
                keep_last=True,
            ) +
            list(df.columns),
            keep_last=False,
        )
        return df[[col for col in cols if col in df.columns]]

    def _needs_ref(f):
        """
        Decorator to bypass a function if no reference group was provided by
        the user
        """
        # pylint: disable=no-self-argument
        @functools.wraps(f)
        def wrapper(self, df, *args, **kwargs):
            if self._ref_group:
                return f(self, df, *args, **kwargs) # pylint: disable=not-callable
            else:
                return df

        return wrapper

    def _melt(self, df, **kwargs):
        """
        Unpivot the dataframe, i.e. turn the all the columns that are not the
        tags into 2 columns:
            * One with values being the former column name identifying the value
            * One with values being the values of the former column
        """
        return pd.melt(df,
            id_vars=self._restrict_cols(self._stat_tag_cols, df),
            value_name=self._val_col,
            var_name=self._stat_col,
            **kwargs
        )

    def _df_group_apply(self, df, func, melt=False, index_cols=None):
        """
        Apply ``func`` on subsets of the dataframe and return the concatenated
        result.

        :param df: Dataframe in database format (meaningless index, tag and
            value columns).
        :type df: pandas.DataFrame

        :param func: Callable called with 3 parameters:

            * ``ref``: Reference subgroup dataframe for comparison purposes. In
              some cases, there is nothing to compare to (the user passed
              ``None`` for all keys in ``ref_group``) so ``ref`` will be
              ``None``.
            * ``df``: Dataframe of the subgroup, to compare to ``ref``.
            * ``group``: Dictionary ``dict(column_name, value)`` identifying
              the ``df`` subgroup.
        :type func: collections.abc.Callable

        :param melt: If ``True``, extra columns added by the callback in the
            return :class:`pandas.DataFrame` will be melted, i.e. they will be
            turned into row with the column name being copied to the stat column.
        :type melt: bool

        :param index_cols: Columns to aggregate on that will be used for
            indexing the sub-dataframes, instead of the default ``agg_cols``.
        :type index_cols: list(str) or None
        """
        ref_group = FrozenDict(self._ref_group)
        # All the columns that are not involved in the group itself except the
        # value will be used as index, so that the reference group and other
        # groups can be joined meaningfully on the index for comparison
        # purposes.
        index_cols = index_cols if index_cols is not None else self._agg_cols
        index_cols = self._restrict_cols(index_cols, df)
        sub_group_cols = self._restrict_cols(self._sub_group_cols, df)

        def process_subgroup(df, group, subgroup):
            subgroup = FrozenDict(subgroup)
            ref = subref.get(subgroup)
            group = {**group, **subgroup}

            # Make sure that the columns/index levels relative to the group are
            # removed, since they are useless because they have a constant value
            def remove_cols(df):
                to_remove = group.keys()
                df = df.drop(columns=self._restrict_cols(to_remove, df))
                try:
                    drop_level = df.index.droplevel
                except AttributeError:
                    pass
                else:
                    df.index = drop_level(sorted(set(df.index.names) & set(to_remove)))
                return df

            df = remove_cols(df)
            if ref is not None:
                ref = remove_cols(ref)

            df = func(ref, df, group)
            if df is None:
                return None

            # Only assign-back subgroup columns if they have not been set by the
            # callback directly.
            to_assign = group.keys() - set(
                col
                for col in df.columns
                if not df[col].isna().all()
            )
            df = df.assign(**{
                col: val
                for col, val in group.items()
                if col in to_assign
            })

            # Drop RangeIndex to avoid getting an "index" column that is
            # useless
            drop_index = isinstance(df.index, pd.RangeIndex)
            df.reset_index(drop=drop_index, inplace=True)
            return df

        # Groups as asked by the user
        comparison_groups = {
            FrozenDict(group): df.set_index(index_cols)
            for group, df in df_split_signals(df, ref_group.keys())
        }

        # We elect a comparison reference and split it in subgroups
        comp_ref_group = FrozenDict(dict(
            (k, v)
            for k, v in ref_group.items()
            if v is not None
        ))
        try:
            ref = comparison_groups[comp_ref_group]
        except KeyError:
            subref = {}
        else:
            subref = {
                FrozenDict(subgroup): subdf
                for subgroup, subdf in df_split_signals(ref, sub_group_cols)
            }

        # For each group, split it further in subgroups
        dfs = [
            process_subgroup(subdf, group, subgroup)
            for group, df in comparison_groups.items()
            for subgroup, subdf in df_split_signals(df, sub_group_cols)
        ]
        dfs = [df for df in dfs if df is not None]
        if dfs:
            df = pd.concat(dfs, ignore_index=True, copy=False)
            if melt:
                df = self._melt(df)
        else:
            df = pd.DataFrame()

        return df

    @property
    @memoized
    def df(self):
        """
        :class:`pandas.DataFrame` containing the statistics.

        .. seealso:: :meth:`get_df` for more controls.
        """
        return self.get_df()

    def get_df(self, remove_ref=None, compare=None):
        """
        Returns a :class:`pandas.DataFrame` containing the statistics.

        :param compare: See :class:`Stats` ``compare`` parameter. If ``None``,
            it will default to the value provided to :class:`Stats`.
        :type compare: bool or None

        :param remove_ref: If ``True``, the rows of the reference group
            described by ``ref_group`` for this object will be removed from the
            returned dataframe. If ``None``, it will default to ``compare``.
        :type remove_ref: bool or None
        """
        compare = compare if compare is not None else self._compare
        remove_ref = remove_ref if remove_ref is not None else compare

        df = self._df_stats()
        df = self._df_stats_test(df)

        if compare:
            df = self._df_compare_pct(df)

        if remove_ref:
            filter_on = {
                k: v
                for k, v in self._ref_group.items()
                if v is not None
            }
            df = df_filter(df, filter_on, exclude=True)

        df = self._df_format(df)
        return df

    def _df_mean(self, df, provide_stats):
        """
        Compute the mean and associated stats
        """
        def get_const_col(group, df, col):
            vals = df[col].unique()
            if len(vals) > 1:
                raise ValueError(f"Column \"{col}\" has more than one value ({', '.join(vals)}) for the group: {group}")
            return vals[0]

        def mean_func(ref, df, group): # pylint: disable=unused-argument
            try:
                mean_kind = get_const_col(group, df, self._mean_kind_col)
            except KeyError:
                try:
                    unit = get_const_col(group, df, self._unit_col)
                except KeyError:
                    unit = None
                try:
                    control_var = get_const_col(group, df, self._control_var_col)
                except KeyError:
                    control_var = None

                mean_kind = guess_mean_kind(unit, control_var)
            else:
                mean_kind = mean_kind or 'arithmetic'

            try:
                mean_name, sem_name, std_name = {
                    'arithmetic': ('mean', 'sem', 'std'),
                    'harmonic': ('hmean', 'hse', 'hsd'),
                    'geometric': ('gmean', 'gse', 'gsd'),
                }[mean_kind]
            except KeyError:
                # pylint: disable=raise-missing-from
                raise ValueError(f'Unrecognized mean kind: {mean_kind}')

            series = df[self._val_col]
            min_sample_size = 30
            series_len = len(series)
            if series_len < min_sample_size:
                group_str = ', '.join(sorted(f'{k}={v}' for k, v in group.items()))
                self.get_logger().warning(f'Sample size smaller than {min_sample_size} is being used, the mean confidence interval will only be accurate if the data is normally distributed: {series_len} samples for group {group_str}')

            mean, std, sem, ci = series_mean_stats(series, kind=mean_kind, confidence_level=self._mean_ci_confidence)

            # Only display the stats we were asked for
            rows = [
                values
                for stat, values in (
                    ('mean', (mean_name, mean, ci[0], ci[1])),
                    ('sem', (sem_name, sem, nan, nan)),
                    ('std', (std_name, std, nan, nan)),
                )
                if stat in provide_stats
            ]
            return pd.DataFrame.from_records(
                rows,
                columns=(
                    self._stat_col,
                    self._val_col,
                    self._ci_cols[0],
                    self._ci_cols[1]
                )
            )

        return self._df_group_apply(df, mean_func, index_cols=self._agg_cols)

    def _df_stats(self):
        """
        Compute the stats on aggregated values
        """
        df = self._orig_df
        stats = self._stats.copy()
        tag_cols = self._restrict_cols(self._stat_tag_cols, df)

        # Specific handling for the mean, as it has to be handled per group
        special_stats = {
            stat
            for stat in ('mean', 'sem', 'std')
            if stat in stats and stats[stat] is None
        }
        if special_stats:
            df_mean = self._df_mean(df, special_stats)
            for stat in special_stats:
                stats.pop(stat)
        else:
            df_mean = df_make_empty_clone(df)
            df_mean.drop(columns=self._agg_cols, inplace=True)

        # Create a DataFrame with stats for the groups
        funcs = {
            name: func or name
            for name, func in stats.items()
        }
        if funcs:
            grouped = df.groupby(tag_cols, observed=True, sort=False)
            df = grouped[self._val_col].agg(**funcs).reset_index()
            # Transform the newly created stats columns into rows
            df = self._melt(df)
        else:
            df = pd.DataFrame()

        df = pd.concat([df, df_mean])
        df = self._df_remove_tweak_cols(df)

        unit_col = self._unit_col
        default_unit = ''
        if unit_col in df:
            df[unit_col].fillna(default_unit, inplace=True)
        else:
            df[unit_col] = default_unit

        for stat, unit in self._STATS_UNIT.items():
            df.loc[df[self._stat_col] == stat, unit_col] = unit.name

        return df

    @_needs_ref
    def _df_stats_test(self, df):
        """
        Compare the groups with a stat test
        """
        value_col = self._val_col
        stat_name = 'ks2samp_test'

        def get_pval(ref, df):
            _, p_value = scipy.stats.ks_2samp(ref[value_col], df[value_col])
            return p_value

        def func(ref, df, group): # pylint: disable=unused-argument
            if ref is None:
                return None
            else:
                return pd.DataFrame({stat_name: [get_pval(ref, df)]})

        # Summarize each group by the p-value of the test against the reference group
        test_df = self._df_group_apply(self._orig_df, func, melt=True)
        test_df[self._unit_col] = 'pval'
        test_df = self._df_remove_tweak_cols(test_df)

        return df.append(test_df, ignore_index=True)

    @_needs_ref
    def _df_compare_pct(self, df):
        """
        Normalize the computed values against the reference.
        """
        val_col = self._val_col
        unit_col = self._unit_col
        ci_cols = self._ci_cols
        stat_col = self._stat_col
        tag_cols = self._tag_cols
        non_normalizable_units = self._non_normalizable_units

        def diff_pct(ref, df, group):
            if ref is None or group[unit_col] in non_normalizable_units:
                return df
            else:
                # (val - ref) / ref == (val / ref) - 1
                factor =  1 / ref[val_col]
                transform = lambda x: 100 * (x * factor - 1)
                df[val_col] = transform(df[val_col])

                # Remove the confidence interval as it is significantly more
                # complex to compute and would require access to other
                # statistics too. All in all it's not really worth the hassle,
                # since the comparison should be based on the stat test anyway.
                _ci_cols = self._restrict_cols(ci_cols, df)
                df = df.drop(columns=_ci_cols)

                df[unit_col] = '%'
                return df

        index_cols = sorted(
            (set(tag_cols) | {unit_col, stat_col}) -
            (self._ref_group.keys() | {val_col})
        )
        df = self._df_group_apply(df, diff_pct, index_cols=index_cols)
        # Divisions can end up yielding extremely small values like 1e-14,
        # which seems to create problems while plotting
        df[val_col] = df[val_col].round(10)
        return df

    def _plot(self, df, title, plot_func, facet_rows, facet_cols, collapse_cols, filename=None, backend=None):
        def fixup_tuple(x):
            """
            DataFrame.groupby() return type is "interesting":
            When grouping on one column only, the group is not a tuple, but the
            value itself, leading to equally "interesting" bugs.
            """
            return x if isinstance(x, tuple) else (x,)

        def plot_subdf(group, subdf):
            group = fixup_tuple(group)
            group_dict = OrderedDict(
                (k, v)
                for k, v in sorted(
                    zip(group_on, group),
                    key=itemgetter(0),
                )
                if k in group_keys
            )

            if subdf.empty:
                fig = hv.Empty()
            else:
                subdf = subdf.drop(columns=group_on)
                subdf = self._collapse_cols(subdf, collapse_group)
                fig = plot_func(subdf, collapsed_col, group_dict)

            return (fig, group_dict)

        unit_col = self._unit_col

        group_on = list(facet_rows) + list(facet_cols)
        # Only show the group keys that are not constant in the whole
        # sub dataframe, to remove a bit of clutter
        group_keys = self._trim_group(df, group_on)

        # Collapse together all the tag columns that are not already in use
        not_collapse = set(group_on) | {unit_col}
        collapse_cols = [
            col
            for col in self._restrict_cols(collapse_cols, df)
            if col not in not_collapse
        ]
        if len(collapse_cols) > 1:
            collapsed_col = 'group'
            collapse_group = {collapsed_col: collapse_cols}
        elif collapse_cols:
            collapsed_col = collapse_cols[0]
            collapse_group = {}
        else:
            collapsed_col = None
            collapse_group = {}

        subplots = dict(
            plot_subdf(group, subdf)
            for group, subdf in df.groupby(group_on, observed=True)
        )

        kdims = sorted(set(itertools.chain.from_iterable(
            idx.keys()
            for idx in subplots.values()
        )))

        if facet_cols:
            ncols = len(df.drop_duplicates(subset=facet_cols, ignore_index=True))
        else:
            ncols = 1

        fig = hv.NdLayout(
            [
                (
                    tuple(
                        idx.get(key, 'N/A')
                        for key in kdims
                    ),
                    fig
                )
                for fig, idx in subplots.items()
            ],
            kdims=kdims,
        ).cols(ncols).options(
            title=title,
            shared_axes=False,
        ).options(
            backend='bokeh',
            toolbar='left',
        ).options(
            backend='matplotlib',
            hspace=1.5,
            vspace=0.7,
        ).options(
            # All plots are wrapped in an Overlay, either because they are true
            # overlays or because NdLayout needs to deal with a single element
            # type.
            'Overlay',
            backend='bokeh',
            hooks=[lisa.notebook._hv_multi_line_title_hook],
        )

        if filename:
            hv.save(fig, filename, backend=backend)

        return fig

    def plot_stats(self, filename=None, remove_ref=None, backend=None, groups_as_row=False, kind=None, **kwargs):
        """
        Returns a :class:`matplotlib.figure.Figure` containing the statistics
        for the class input :class:`pandas.DataFrame`.

        :param filename: Path to the image file to write to.
        :type filename: str or None

        :param remove_ref: If ``True``, do not plot the reference group.
            See :meth:`get_df`.
        :type remove_ref: bool or None

        :param backend: Holoviews backend to use: ``bokeh`` or ``matplotlib``.
            If ``None``, the current holoviews backend selected with
            ``hv.extension()`` will be used.
        :type backend: str or None

        :param groups_as_row: By default, subgroups are used as rows in the
            subplot matrix so that the values shown on a given graph can be
            expected to be in the same order of magnitude. However, when there
            are many subgroups, this can lead to very large and somewhat hard
            to navigate plot matrix. In this case, using the group for the rows
            might help a great deal.
        :type groups_as_row: bool

        :param kind: Type of plot. Can be any of:

            * ``horizontal_bar``
            * ``vertical_bar``
            * ``None``
        :type kind: str or None

        :Variable keyword arguments: Forwarded to :meth:`get_df`.
        """
        # Resolve the backend so we can use backend-specific workarounds
        backend = backend or hv.Store.current_backend

        kind = kind if kind is not None else 'horizontal_bar'
        df = self.get_df(
            remove_ref=remove_ref,
            **kwargs
        )

        mean_suffix = ' (CL: {:.1f}%)'.format(
            self._mean_ci_confidence * 100
        )
        df = df.copy()
        df.loc[df[self._stat_col] == 'mean', self._stat_col] += mean_suffix

        pretty_ref_group = ' and '.join(
            f'{k}={v}'
            for k, v in self._ref_group.items()
            if v is not None
        )
        title = 'Statistics{}'.format(
            f' compared against: {pretty_ref_group}' if self._compare else ''
        )

        def make_unique_col(prefix):
            return prefix + '_' + uuid.uuid4().hex

        # Generate a random name so it does not clash with anything. Also add a
        # fixed prefix that does not confuse bokeh hovertool.
        value_str_col = make_unique_col('value_display')

        def plot(df, collapsed_col, group):
            def format_val(val):
                return f'{val:.2f}' if abs(val) > 1e-2 else f'{val:.2e}'

            def make_val_hover(show_unit, row):
                val = row[y_col]
                unit = row[unit_col] if show_unit else ''
                try:
                    ci = [
                        row[col]
                        for col in self._ci_cols
                    ]
                except AttributeError:
                    ci = ''
                else:
                    if not any(map(pd.isna, ci)):
                        ci = list(map(format_val, ci))
                        if ci[0] == ci[1]:
                            ci = f'\n(±{ci[0]})'
                        else:
                            ci = f'\n(+{ci[1]}/-{ci[0]})'
                    else:
                        ci = ''

                return f'{format_val(val)} {unit}{ci}'

            # There is only one bar to display, aka nothing to compare against,
            # so we add a placeholder column so we can still plot on bar per
            # subplot
            if collapsed_col is None:
                collapsed_col = make_unique_col('group')
                collapsed_col_hover = ''
                df = df.copy(deep=False)
                df[collapsed_col] = ''
            else:
                collapsed_col_hover = collapsed_col

            try:
                error = [
                    df[col]
                    for col in self._ci_cols
                ]
            except KeyError:
                ci_cols = None
            else:
                # Avoid warning from numpy inside matplotlib when there is no
                # confidence interval value at all
                if all(
                    series.isna().all()
                    for series in error
                ):
                    ci_cols = None
                else:
                    ci_cols = self._ci_cols

            y_col = self._val_col
            unit_col = self._unit_col

            if kind == 'horizontal_bar':
                invert_axes = True
            elif kind == 'vertical_bar':
                invert_axes = False
            else:
                raise ValueError(f'Unsupported plot kind: {kind}')

            show_unit = True
            tooltip_val_name = y_col
            try:
                unit, = df[unit_col].unique()
            except ValueError:
                pass
            else:
                unit = unit.strip()
                if unit:
                    show_unit = False
                    tooltip_val_name = unit

            df[value_str_col] = df.apply(
                functools.partial(make_val_hover, show_unit),
                axis=1
            )
            hover = HoverTool(
                tooltips=[
                    (collapsed_col_hover, f'@{collapsed_col}'),
                    (tooltip_val_name, f'@{value_str_col}'),
                ]
            )

            bar_df = df[[collapsed_col, y_col, value_str_col]].dropna(
                subset=[collapsed_col]
            )
            # Holoviews barfs on empty data for Bars
            if bar_df.empty:
                # TODO: should be replaced by hv.Empty() but this raises an
                # exception
                fig = hv.Curve([]).options(
                    xlabel='',
                    ylabel='',
                )
            else:
                fig = hv.Bars(
                    bar_df[[collapsed_col, y_col, value_str_col]].dropna(subset=[collapsed_col]),
                ).options(
                    ylabel='',
                    xlabel='',
                    invert_axes=invert_axes,
                    # The legend is useless since we only have a consistent set of
                    # bar on each plot, but it can still be displayed in some cases
                    # when an other element is overlaid, such as the ErrorBars
                    show_legend=False,
                ).options(
                    backend='bokeh',
                    tools=[hover],
                    # Color map on the subgroup
                    cmap='glasbey_hv',
                    color=collapsed_col,
                )
                if ci_cols is not None:
                    fig *= hv.ErrorBars(
                        df[[collapsed_col, y_col, *ci_cols]],
                        vdims=[y_col, *ci_cols],
                    ).options(
                        backend='bokeh',
                        # Workaround error bars being hidden by the bar plot:
                        # https://github.com/holoviz/holoviews/issues/1968
                        level='annotation',
                    )

                # Labels do not work with matplotlib unfortunately:
                # https://github.com/holoviz/holoviews/issues/4992
                if backend != 'matplotlib':
                    df_label = df.copy(deep=False)
                    # Center the label in the bar
                    df_label[y_col] = df_label[y_col] / 2
                    fig *= hv.Labels(
                        df_label[[collapsed_col, y_col, value_str_col]],
                        vdims=[value_str_col],
                        kdims=[collapsed_col, y_col],
                    ).options(
                        backend='bokeh',
                        text_font_size='8pt',
                    )

                # Label after applying the error bars, so that the whole
                # Overlay gets the label
                fig = fig.relabel(
                    # Provide a short label to allow the user to manipulate
                    # individual layout elements more easily
                    '_'.join(map(str, group.values())),
                )

            # Wrap in an Overlay so we can ensure that NdLayout only has to
            # deal with a single element type
            fig = hv.Overlay([fig])

            fig = fig.options(
                # Set the title on the Overlay, otherwise it will be ignored
                title='\n'.join(
                    f'{k}={v}'
                    for k, v in group.items()
                )
            )

            return fig

        # Subplot matrix:
        # * one line per sub-group (e.g. metric)
        # * one column per stat
        #
        # On each plot:
        # * one bar per value of the given stat for the given group
        facet_rows = self._restrict_cols(self._plot_group_cols, df)
        facet_cols = [self._stat_col]
        collapse_cols = set(self._stat_tag_cols) - {self._unit_col, *facet_rows, *facet_cols}

        # If we want each row to be a group (e.g. kernel), swap with the bargraph X axis.
        # Note that this can create scale issues as the result of multiple
        # subgroups will be on the same plot (e.g. different benchmarks)
        if groups_as_row:
            facet_rows, collapse_cols = collapse_cols, facet_rows

        return self._plot(
            df,
            title=title,
            plot_func=plot,
            facet_rows=facet_rows,
            facet_cols=facet_cols,
            collapse_cols=collapse_cols,
            filename=filename,
            backend=backend,
        )

    @staticmethod
    def _trim_group(df, group):
        trimmed = [
            col
            for col in group
            # If the column to collapse has a constant value, there is
            # usually no need to display it in titles and such as it is
            # just noise
            if (
                col in df.columns and
                df[col].nunique() > 1
            )
        ]
        # If we got rid of all columns, keep them all. Otherwise we will
        # end up with nothing to display which is problematic
        return trimmed if trimmed else group

    @classmethod
    def _collapse_cols(cls, df, groups, hide_constant=True):
        groups = {
            leader: (
                cls._trim_group(df, group)
                if hide_constant else
                group
            )
            for leader, group in groups.items()
            if group
        }
        if groups:
            # Collapse together columns that are part of a group
            def collapse_group(acc, col):
                if acc is None:
                    sep = ''
                    acc = ''
                else:
                    sep = '\n'

                def make_str(val):
                    # Some columns have empty string to flag there is nothing
                    # to display like for unit
                    if val == '':
                        return ''
                    else:
                        return f'{col}={val}{sep}'

                return df[col].apply(make_str) + acc

            df = df.copy()
            for leader, group in groups.items():
                if leader in df.columns:
                    combine = lambda leader, group: df[leader] + ' (' + group + ')'
                else:
                    combine = lambda leader, group: group

                # If there is only one member in the group, there is no need to
                # add the column name as there is no ambiguity so we avoid the
                # extra noise
                if len(group) == 1:
                    df[leader] = df[group[0]]
                elif group:
                    df[leader] = combine(leader, fold(collapse_group, group))
                # If len(group) == 0, there is nothing to be done
                else:
                    df[leader] = ''

                df.drop(columns=group, inplace=True)

        return df

    def plot_histogram(self, cumulative=False, bins=50, nbins=None, density=False, **kwargs):
        """
        Returns a :class:`matplotlib.figure.Figure` with histogram of the values in the
        input :class:`pandas.DataFrame`.

        :param cumulative: Cumulative plot (CDF).
        :type cumulative: bool

        :param bins: Number of bins for the distribution.
        :type bins: int or None

        :param filename: Path to the image file to write to.
        :type filename: str or None
        """
        if nbins:
            warnings.warn('"nbins" parameter is deprecated and will be removed, use "bins" instead', DeprecationWarning)
            bins = nbins

        def plot_func(df, group, x_col, y_col): # pylint: disable=unused-argument
            points = hv.Scatter(df[[x_col, y_col]])
            fig = hv.operation.histogram(
                points,
                cumulative=cumulative,
                num_bins=bins,
            )
            if cumulative:
                # holoviews defaults to a bar plot for CDF
                fig = hv.Curve(fig).options(
                    interpolation='steps-post',
                )

            if density:
                return hv.Distribution(fig)
            else:
                return fig

        return self._plot_values(
            title='Values histogram',
            plot_func=plot_func,
            **kwargs,
        )

    def plot_values(self, **kwargs):
        """
        Returns a holoviews element with the values in the input
        :class:`pandas.DataFrame`.

        :param filename: Path to the image file to write to.
        :type filename: str or None
        """

        def plot_func(df, group, x_col, y_col):
            try:
                unit = group[self._unit_col]
            except KeyError:
                unit = None

            data = df[[x_col, y_col]].sort_values(x_col)

            return (
                hv.Curve(
                    data,
                ).options(
                    ylabel=unit,
                ) *
                hv.Scatter(
                    data,
                ).options(
                    backend='bokeh',
                    marker='circle',
                    size=10,
                ).options(
                    backend='matplotlib',
                    marker='o',
                    s=100,
                )
            )

        return self._plot_values(
            title=f"Values over {', '.join(self._agg_cols)}",
            plot_func=plot_func,
            **kwargs,
        )

    def _plot_values(self, title, plot_func, **kwargs):
        agg_cols = self._agg_cols

        df = self._orig_df

        facet_cols = []
        facet_rows = [
            col
            for col in df.columns
            if (
                col not in self._agg_cols and
                col != self._val_col and
                col not in facet_cols
            )
        ]

        def plot(df, collapsed_col, group): # pylint: disable=unused-argument
            title = '\n'.join(
                f'{k}={v}'
                for k, v in group.items()
                if v != ''
            )

            if len(agg_cols) > 1:
                x_col = ''
                df = self._collapse_cols(df, {x_col: agg_cols})
            else:
                x_col, = agg_cols

            y_col = self._val_col
            return plot_func(
                df,
                group=group,
                x_col=x_col,
                y_col=y_col
            ).options(
                title=title,
            ).options(
                backend='bokeh',
                width=800,
            ).options(
                'Curve',
                backend='bokeh',
                tools=['hover'],
                hooks=[lisa.notebook._hv_multi_line_title_hook],
            ).options(
                'Histogram',
                backend='bokeh',
                tools=['hover'],
                hooks=[lisa.notebook._hv_multi_line_title_hook],
            ).options(
                'Distribution',
                backend='bokeh',
                tools=['hover'],
                hooks=[lisa.notebook._hv_multi_line_title_hook],
            ).options(
                'Overlay',
                backend='bokeh',
                hooks=[lisa.notebook._hv_multi_line_title_hook],
            )

        return self._plot(
            df,
            title=title,
            plot_func=plot,
            collapse_cols=facet_cols,
            facet_rows=facet_rows,
            facet_cols=[],
            **kwargs
        )
