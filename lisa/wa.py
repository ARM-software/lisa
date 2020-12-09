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
from collections.abc import Mapping
from collections import defaultdict
from operator import attrgetter
import functools
import subprocess
import inspect
import os
import abc
import contextlib

import pandas as pd
from wa import RunOutput, discover_wa_outputs, Status, HostError

from lisa.version import VERSION_TOKEN
from lisa.stats import Stats
from lisa.utils import Loggable, memoized, get_subclasses
from lisa.git import find_shortest_symref, get_commit_message
from lisa.trace import Trace


def _df_concat(dfs):
    return pd.concat(dfs, ignore_index=True, copy=False, sort=False)


class WAOutputNotFoundError(Exception):
    def __init__(self, collectors):
        self.collectors = collectors

    def __str__(self):
        sep = '\n    '
        return 'Could not find output for collectors{}{}'.format(
            ':' + sep if len(self.collectors) > 1 else ' ',
            sep.join(
                '{} ({}): {}'.format(collector.NAME, excep.__class__.__qualname__, excep)
                for collector, excep in self.collectors.items()
            )
        )

    @classmethod
    def from_collector(cls, collector, excep):
        return cls({collector: excep})

    @classmethod
    def from_excep_list(cls, exceps):
        return cls(
            collectors={
                collector: excep
                for excep in exceps
                for collector, excep in excep.collectors.items()
            }
        )


class StatsProp:
    """
    Provides a ``stats`` property.
    """
    _STATS_GROUPS = ['board', 'kernel']
    """
    Tag columns commonly used to group plots of WA dataframes.
    """

    def get_stats(self, ensure_default_groups=True, ref_group=None, **kwargs):
        """
        Returns a :class:`lisa.stats.Stats` loaded with the result
        :class:`pandas.DataFrame`.

        :param ensure_default_groups: If ``True``, ensure `ref_group` will contain
            appropriate keys for usual Workload Automation result display.
        :type ensure_default_groups: bool

        :param ref_group: Forwarded to :class:`lisa.stats.Stats`

        :Variable keyword arguments: Forwarded to :class:`lisa.stats.Stats`

        """
        if ensure_default_groups:
            # Make sure the tags to group on are present in the ref_group, even
            # if the user did not specify them
            ref_group = {
                **{
                    k: None
                    for k in self._STATS_GROUPS
                },
                **(ref_group or {}),
            }

        return Stats(self.df, ref_group=ref_group, **kwargs)

    @property
    def stats(self):
        """
        Short-hand property equivalent to ``self.get_stats()``

        .. seealso:: :meth:`get_stats`
        """
        return self.get_stats()


class WAOutput(StatsProp, Mapping, Loggable):
    """
    Recursively parse a ``Workload Automation`` output, using registered
    collectors (leaf subclasses of :class:`WACollectorBase`). The data
    collected are accessible through a :class:`pandas.DataFrame` in "database"
    format:

        * meaningless index
        * all values are tagged using tag columns

    :param path: Path containing a Workload Automation output.
    :type path: str

    :param kernel_path: Kernel source path. Used to resolve the name of the
        kernel which ran the workload.
    :param kernel_path: str

    **Example**::

        wa_output = WAOutput('wa/output/path')
        # Pick a specific collector. See also WAOutput.get_collector()
        stats = wa_output['results'].stats
        stats.plot_stats(filename='stats.html')
    """

    def __init__(self, path, kernel_path=None):
        self.path = path
        self.kernel_path = kernel_path

        collector_classes = {
            cls.NAME: cls
            for cls in get_subclasses(WACollectorBase, only_leaves=True)
        }
        auto_collectors = {
            name: cls
            for name, cls in collector_classes.items()
            if not self._needs_params(cls)
        }
        self._auto_collectors = auto_collectors
        self._available_collectors = collector_classes

    def __hash__(self):
        """
        Each instance is different, like regular objects, and unlike dictionaries.
        """
        return id(self)

    def __eq__(self, other):
        return self is other

    @memoized
    def __getitem__(self, key):
        cls = self._available_collectors[key]
        if key not in self._auto_collectors:
            raise KeyError("Collector {name} needs mandatory parameter, use get_collector('{name}', ...) instead".format(
                name=key
            ))
        else:
            return cls(self)

    def __iter__(self):
        return iter(self._auto_collectors)

    def __contains__(self, key):
        return key in self._auto_collectors

    def __len__(self):
        return len(self._auto_collectors)

    @property
    @memoized
    def df(self):
        """
        DataFrame containing the data collected by all the registered
        :class:`WAOutput` collectors.
        """
        dfs = []
        exceps = {}
        for name, collector in self.items():
            try:
                df = collector.df
            except Exception as e:
                exceps[collector] = e
                self.get_logger().debug('Could not get dataframe of collector {}: {}'.format(
                    name, e,
                ))
            else:
                dfs.append(df)

        if not dfs:
            raise WAOutputNotFoundError.from_excep_list([
                e
                if isinstance(e, WAOutputNotFoundError)
                # Wrap other exceptions in a WAOutputNotFoundError
                else WAOutputNotFoundError.from_collector(collector, e)
                for collector, e in exceps.items()
            ])

        return _df_concat(dfs)

    def get_collector(self, name, **kwargs):
        """
        Returns a new collector with custom parameters passed to it.

        :param name: Name of the collector.
        :type name: str

        :Variable keyword arguments: Forwarded to the collector's constructor.

        **Example**::

            WAOutput('wa/output/path').get_collector('energy', postprocess=func)
        """
        return self._available_collectors[name](self, **kwargs)

    @staticmethod
    def _needs_params(col):
        """
        Whether a collector has mandatory parameters.
        """
        parameters = list(inspect.signature(col).parameters.items())
        return any(
            param.default == param.empty
            # The first parameter is provided so we can skip it
            for name, param in parameters[1:]
        )


class WACollectorBase(StatsProp, Loggable, abc.ABC):
    """
    Base class for all ``Workload Automation`` dataframe collectors.

    :param wa_output: :class:`WAOutput` parent object.
    :type wa_output: WAOutput

    :param df_postprocess: Function called to postprocess the collected
        :class:`pandas.DataFrame`.
    :type df_postprocess: collections.abc.Callable

    .. seealso:: Instances of this classes are typically built using
        :meth:`WAOutput.get_collector` rather than directly.
    """

    _EXPECTED_WORKLOAD_NAME = None

    def __init__(self, wa_output, df_postprocess=None):
        self.wa_output = wa_output
        self._df_postprocess = df_postprocess or (lambda x: x)

    @abc.abstractclassmethod
    def _get_job_df(cls, job):
        """
        Process a :class:`wa.framework.JobOutput` and return a
        :class:`pandas.DataFrame` with the results.

        :param job: WA job run output to process
        :type job: wa.framework.JobOutput

        It is a good idea to then feed the dataframe to :meth:`_add_job_info`
        to get all the tags from WA before returning it.

        .. note:: If one of these column provides the same information as some
            column from artifact dataframe, consider the following:

            * Which column have values looking better ?
              Values will be used in legends, titles etc

            * If the :meth:`_add_job_info` column looks worst, can there still
              be value in keeping it ?
              Maybe its values can be fed back to other WA APIs ?

            * Are you sure the column is *always* provided by
              :meth:`_add_job_info` ?
              User can inject arbitrary classifier from WA config, so you might
              have some columns that cannot be relied upon.
        """
        pass

    @property
    def logger(self):
        return self.get_logger()

    @property
    @memoized
    def df(self):
        """
        :class:`pandas.DataFrame` containing the data collected.
        """
        return self._get_df()

    def _get_df(self):
        self.logger.debug("Collecting dataframe for {}".format(self.NAME))

        wa_outputs = list(discover_wa_outputs(self.wa_output.path))

        def load_df(job):
            def loader(job):
                cache_path = os.path.join(
                    job.basepath,
                    '.{}-cache.{}.parquet'.format(self.NAME, VERSION_TOKEN)
                )

                # _get_job_df usually returns fairly large dataframes, so cache
                # the result for faster reloading
                try:
                    df = pd.read_parquet(cache_path)
                except OSError:
                    df = self._get_job_df(job)
                    df.to_parquet(cache_path)
                return df

            try:
                df = loader(job)
            except Exception as e:
                # Swallow the error if that job was not from the expected
                # workload
                expected_name = self._EXPECTED_WORKLOAD_NAME
                if expected_name is None or job.spec.workload_name == expected_name:
                    self.logger.error('Could not load {} dataframe for job {}: {}'.format(self.NAME, job, e))
                else:
                    return None
            else:
                return self._df_postprocess(df)

        dfs = [
            self._add_kernel_version(wa_output, df)
            for wa_output in wa_outputs
            for df in [
                load_df(job)
                for job in wa_output.jobs
                if job.status == Status.OK
            ]
            if df is not None
        ]

        if not dfs:
            raise WAOutputNotFoundError.from_collector(self, 'Could not find any valid job output')

        # It is unfortunately not safe to cache the output of load_df, as the
        # user postprocessing could change at any time
        df = _df_concat(dfs)
        return self._add_kernel_id(df)

    @staticmethod
    def _add_job_info(job, df):
        df['iteration'] = job.iteration
        df['workload'] = job.label
        df['id'] = job.id
        df = df.assign(**job.classifiers)
        return df

    @staticmethod
    def _add_kernel_version(wa_output, df):
        kver = wa_output.target_info.kernel_version
        df['kernel_name'] = kver.release
        df['kernel_sha1'] = kver.sha1
        return df

    def _add_kernel_id(self, df):
        kernel_path = self.wa_output.kernel_path
        resolvers = [
            find_shortest_symref,
            get_commit_message,
        ]

        def resolve_readable(sha1):
            if kernel_path:
                for resolver in resolvers:
                    with contextlib.suppress(ValueError):
                        return resolver(kernel_path, sha1)

            return sha1

        kernel_ids = {
            sha1: resolve_readable(sha1)
            for sha1 in df['kernel_sha1'].unique()
            if sha1 is not None
        }

        # Reduce the human readable id if possible
        common_prefix = os.path.commonprefix(list(kernel_ids.values()))

        kernel_ids = {
            sha1: ref[len(common_prefix):] or ref
            for sha1, ref in kernel_ids.items()
        }

        df['kernel'] = df['kernel_sha1'].map(kernel_ids).fillna(
            df['kernel_name']
        )
        df.drop(columns=['kernel_sha1', 'kernel_name'], inplace=True)
        return df


class WAResultsCollector(WACollectorBase):
    """
    Collector for the ``Workload Automation`` test results.
    """

    NAME = 'results'

    @classmethod
    def _get_job_df(cls, job):
        df = pd.DataFrame.from_records(
            {
                'metric': metric.name,
                'value': metric.value,
                'unit': metric.units or '',
            }
            for metric in job.metrics
        )
        return cls._add_job_info(job, df)


class WAArtifactCollectorBase(WACollectorBase):
    """
    ``Workload Automation`` artifact collector base class.
    """
    _ARTIFACT_NAME = None

    @abc.abstractmethod
    def _get_artifact_df(self, path):
        """
        Process the artifact at the given path and return a
        :class:`pandas.DataFrame`.
        """

    def _get_job_df(self, job):
        path = job.get_artifact_path(self._ARTIFACT_NAME)
        df = self._get_artifact_df(path)
        return self._add_job_info(job, df)


class WAEnergyCollector(WAArtifactCollectorBase):
    """
    WA collector for the energy_measurement augmentation.

    **Example**::

        def postprocess(df):
            df = df.pivot_table(values='value', columns='metric', index=['sample', 'iteration', 'workload'])

            df = pd.DataFrame({
                'CPU_power': (
                    df['A55_power'] +
                    df['A76_1_power'] +
                    df['A76_2_power']
                ),
            })
            df['unit'] = 'Watt'
            df = df.reset_index()
            df = df.melt(id_vars=['sample', 'iteration', 'workload', 'unit'], var_name='metric')
            return df

        WAOutput('wa/output/path').get_collector(
            'energy',
            df_postprocess=postprocess,
        ).df
    """

    NAME = 'energy'
    _ARTIFACT_NAME = "energy_instrument_output"
    _ARTIFACT_METRIC_SUFFIX_UNIT = defaultdict(
        # Default is an empty string
        str,
        {
            'power': 'Watt',
            'voltage': 'V',
        },
    )

    def _get_artifact_df(self, path):
        df = pd.read_csv(path)

        # Record the CSV line as sample nr so each measurement is uniquely
        # identified by a metric and a sample number
        df.index.name = 'sample'
        df.reset_index(inplace=True)

        df = df.melt(id_vars=['sample'], var_name='metric')

        suffix_unit = self._ARTIFACT_METRIC_SUFFIX_UNIT
        df['unit'] = df['metric'].apply(
            lambda x: suffix_unit[x.rsplit('_', 1)[-1]]
        )
        return df

    def get_stats(self, **kwargs):
        return super().get_stats(
            # Aggregate over pairs (sample, iteration)
            agg_cols=['sample', 'iteration'],
            **kwargs,
        )


class WATraceCollector(WAArtifactCollectorBase):
    """
    WA collector for the trace augmentation.

    :param trace_to_df: Function used by the collector to convert the
        :class:`lisa.trace.Trace` to a :class:`pandas.DataFrame`.
    :type trace_to_df: collections.abc.Callable

    :Variable keyword arguments: Forwarded to :class:`lisa.trace.Trace`.

    **Example**::

        def trace_idle_analysis(trace):
            cpu = 0
            df = trace.analysis.idle.df_cluster_idle_state_residency(cpu)
            df = df.reset_index()
            df['cpu'] = cpu

            # Melt the column 'time' into lines, so that the dataframe is in
            # "database" format: each value is uniquely identified by "tag"
            # columns
            return df.melt(
                var_name='metric',
                value_vars=['time'],
                id_vars=['idle_state'],
             )

        WAOutput('wa/output/path').get_collector(
            'trace',
            trace_to_df=trace_idle_analysis,
        ).df

    """
    NAME = 'trace'
    _ARTIFACT_NAME = 'trace-cmd-bin'

    def __init__(self, wa_output, trace_to_df, **kwargs):
        self._trace_to_df = trace_to_df
        self._trace_kwargs = kwargs
        super().__init__(wa_output, df_postprocess=None)

    def _get_artifact_df(self, path):
        trace = Trace(path, **self._trace_kwargs)
        return self._trace_to_df(trace)
