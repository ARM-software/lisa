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
import functools
from operator import itemgetter
import contextlib
from math import nan
from itertools import combinations

import scipy.stats
import pandas as pd
import numpy as np

from lisa.utils import Loggable, memoized, FrozenDict, deduplicate, fold
from lisa.datautils import df_split_signals, df_make_empty_clone, df_filter
from lisa.notebook import make_figure, COLOR_CYCLE


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
        if filter_rows:
            df = df_filter(df, filter_rows)

        ref_group = ref_group or {}
        group_cols = list(ref_group.keys())
        ref_group = {
            k: v
            for k, v in ref_group.items()
            if v is not None
        }

        # Columns controlling the behavior of this class, but that are not tags
        # nor values
        tweak_cols = {mean_kind_col, control_var_col}

        tag_cols = sorted(
            (set(df.columns) - {value_col, *ci_cols} - tweak_cols) | {unit_col}
        )

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
        sub_group_cols = set(stat_tag_cols) - ref_group.keys()
        plot_group_cols = set(stat_tag_cols) - set(group_cols) - {unit_col}

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

            * ``ref``: Reference subgroup dataframe for comparison purposes.
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

            try:
                ref = subref[subgroup]
            except KeyError:
                return None

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
            ref = remove_cols(ref)

            df = func(ref, df, group)
            # Only assign-back subgroup columns if they have not been set by the
            # callback directly.
            to_assign = group.keys() - set(df.columns)
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
        ref = comparison_groups[ref_group]
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
        df = pd.concat((df for df in dfs if df is not None), ignore_index=True, copy=False)

        if melt:
            df = self._melt(df)

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
            df = df_filter(df, self._ref_group, exclude=True)

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

            def fixup_nan(x):
                return 0 if pd.isna(x) else x
            mean, std, sem, ci = series_mean_stats(series, kind=mean_kind, confidence_level=self._mean_ci_confidence)
            ci = tuple(map(fixup_nan, ci))

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
            if group[unit_col] in non_normalizable_units:
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

    def _plot(self, df, title, plot_func, facet_rows, facet_cols, collapse_cols, filename=None, interactive=None):
        unit_col = self._unit_col

        group_on = list(facet_rows) + list(facet_cols)
        facet_rows_len = len(facet_rows)
        def split_row_col(group):
            row = tuple(group[:facet_rows_len])
            col = tuple(group[facet_rows_len:])
            return (row, col)

        grouped = df.groupby(group_on, observed=True)
        # DataFrame.groupby() return type is "interesting":
        # When grouping on one column only, the group is not a tuple, but the
        # value itself, leading to equally "interesting" bugs.
        fixup_tuple = lambda x: x if isinstance(x, tuple) else (x,)

        unzip = lambda x: zip(*x)
        rows, cols = map(sorted, map(set, unzip(map(split_row_col, map(fixup_tuple, map(itemgetter(0), grouped))))))
        nrows, ncols = len(rows), len(cols)

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

        figure, axes = make_figure(
            width=16,
            height=16,
            nrows=nrows,
            ncols=ncols,
            interactive=interactive,
        )
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        figure.set_tight_layout(dict(
            h_pad=3.5,
        ))
        figure.suptitle(title, y=1.01, fontsize=30)

        for group, subdf in grouped:
            group = fixup_tuple(group)

            row, col = split_row_col(group)
            ax = axes[rows.index(row)][cols.index(col)]

            if subdf.empty:
                figure.delaxes(ax)
            else:
                subdf = subdf.drop(columns=group_on)
                subdf = self._collapse_cols(subdf, collapse_group)
                group_dict = dict(zip(group_on, group))
                plot_func(subdf, ax, collapsed_col, group_dict)

        if filename:
            # The suptitle is not taken into account by tight layout by default:
            # https://stackoverflow.com/questions/48917631/matplotlib-how-to-return-figure-suptitle
            suptitle = figure._suptitle
            figure.savefig(filename, bbox_extra_artists=[suptitle], bbox_inches='tight')

        return figure

    def plot_stats(self, filename=None, remove_ref=None, interactive=None, groups_as_row=True, kind=None, **kwargs):
        """
        Returns a :class:`matplotlib.figure.Figure` containing the statistics
        for the class input :class:`pandas.DataFrame`.

        :param filename: Path to the image file to write to.
        :type filename: str or None

        :param remove_ref: If ``True``, do not plot the reference group.
            See :meth:`get_df`.
        :type remove_ref: bool or None

        :param interactive: Forwarded to :func:`lisa.notebook.make_figure`
        :type interactive: bool or None

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

        kind = kind if kind is not None else 'horizontal_bar'
        df = self.get_df(
            remove_ref=remove_ref,
            **kwargs
        )

        mean_suffix = ' (confidence level: {:.1f}%)'.format(
            self._mean_ci_confidence * 100
        )
        df = df.copy()
        df.loc[df[self._stat_col] == 'mean', self._stat_col] += mean_suffix

        pretty_ref_group = ' and '.join(
            f'{k}={v}'
            for k, v in self._ref_group.items()
        )
        title = 'Statistics{}'.format(
            f' compared against: {pretty_ref_group}' if self._compare else ''
        )

        def plot(df, ax, collapsed_col, group):
            try:
                error = [
                    df[col]
                    for col in self._ci_cols
                ]
            except KeyError:
                error = None
            else:
                # Avoid warning from numpy inside matplotlib when there is no
                # confidence interval value at all
                if all(
                    series.isna().all()
                    for series in error
                ):
                    error = None

            if kind == 'horizontal_bar':
                error_bar = dict(xerr=error)
            else:
                error_bar = dict(yerr=error)

            y_col = self._val_col

            if kind == 'horizontal_bar':
                plot = df.plot.barh
            elif kind == 'vertical_bar':
                plot = df.plot.bar
            else:
                raise ValueError(f'Unsupported plot kind: {kind}')

            plot(
                ax=ax,
                x=collapsed_col,
                y=y_col,
                legend=None,
                color=COLOR_CYCLE,
                **error_bar,
                capsize=2,
            )
            title = ' '.join(
                f'{k}={v}'
                for k, v in group.items()
            )
            ax.set_title(title)

            def format_val(val):
                return f'{val:.2f}' if val > 1e-2 else f'{val:.2e}'

            # Display the value on the bar
            for row, patch in zip(df.itertuples(), ax.patches):
                val = getattr(row, y_col)

                try:
                    unit = getattr(row, self._unit_col)
                except AttributeError:
                    unit = ''

                # 3 chars allow things like 'f/s' or 'ms'
                if len(unit) > 3:
                    unit = f'\n{unit}'

                try:
                    ci = [
                        getattr(row, col)
                        for col in self._ci_cols
                    ]
                except AttributeError:
                    ci = ''
                else:
                    if not any(map(pd.isna, ci)):
                        if ci[0] == ci[1]:
                            ci = '\n(+/-{})'.format(format_val(ci[0]))
                        else:
                            ci = '\n(+{}/-{})'.format(*(map(format_val, ci)))
                    else:
                        ci = ''

                text = f'{format_val(val)} {unit}{ci}'

                if kind == 'horizontal_bar':
                    coord = (
                        patch.get_width(),
                        patch.get_y() + (
                            (patch.get_height() / 2)
                            if ci
                            else 0
                        ),
                    )
                else:
                    coord = (
                        patch.get_x() + (
                            # Make some room for the error bar
                            (patch.get_width() / 1.9)
                            if ci
                            else 0
                        ),
                        patch.get_height()
                    )

                ax.annotate(
                    text,
                    coord,
                    xytext=(0, 5),
                    textcoords='offset points',
                )

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
            interactive=interactive,
        )

    @staticmethod
    def _collapse_cols(df, groups, hide_constant=True):
        def trim_group(group):
            trimmed = [
                col
                for col in group
                # If the column to collapse has a constant value, there is
                # usually no need to display it in titles and such as it is
                # just noise
                if (not hide_constant) or df[col].nunique() > 1
            ]
            # If we got rid of all columns, keep them all. Otherwise we will
            # end up with nothing to display which is problematic
            return trimmed if trimmed else group

        groups = {
            leader: trim_group(group)
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
                    sep = ' '

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

    def plot_histogram(self, cumulative=False, nbins=50, density=True, **kwargs):
        """
        Returns a :class:`matplotlib.figure.Figure` with histogram of the values in the
        input :class:`pandas.DataFrame`.

        :param cumulative: Cumulative plot (CDF).
        :type cumulative: bool

        :param nbins: Number of bins for the distribution.
        :type nbins: int or None

        :param filename: Path to the image file to write to.
        :type filename: str or None
        """
        def plot_func(df, group, ax, x_col, y_col): # pylint: disable=unused-argument
            df.plot.hist(
                ax=ax,
                x=x_col,
                y=y_col,
                legend=None,
                bins=nbins,
                cumulative=cumulative,
                density=density,
            )

        return self._plot_values(
            title='Values histogram',
            plot_func=plot_func,
            **kwargs,
        )

    def plot_values(self, **kwargs):
        """
        Returns a :class:`matplotlib.figure.Figure` with the values in the input
        :class:`pandas.DataFrame`.

        :param filename: Path to the image file to write to.
        :type filename: str or None
        """

        def plot_func(df, group, ax, x_col, y_col):
            df.plot.line(
                ax=ax,
                x=x_col,
                y=y_col,
                legend=None,
                marker='o',
            )
            try:
                unit = group[self._unit_col]
            except KeyError:
                pass
            else:
                if unit:
                    ax.set_ylabel(unit)

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

        def plot(df, ax, collapsed_col, group): # pylint: disable=unused-argument
            title = ' '.join(
                f'{k}={v}'
                for k, v in group.items()
                if v != ''
            )
            ax.set_title(title)

            if len(agg_cols) > 1:
                x_col = ''
                df = self._collapse_cols(df, {x_col: agg_cols})
            else:
                x_col, = agg_cols

            # Increase the width of the figure to the size required by the largest plot
            needed_width = len(df) / 2
            fig = ax.get_figure()
            x, y = fig.get_size_inches()
            x = max(x, needed_width)
            fig.set_size_inches((x, y))

            y_col = self._val_col
            plot_func(
                df,
                group=group,
                ax=ax,
                x_col=x_col,
                y_col=y_col
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
