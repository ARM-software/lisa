#    Copyright 2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import namedtuple, defaultdict
import csv
import json
import numpy as np
import re
import os
import pandas as pd
import subprocess
import logging
import warnings
import sqlite3

from scipy.stats import ttest_ind
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from lisa.conf import LisaLogging

from bart.common.Utils import area_under_curve
from devlib.target import KernelVersion
from devlib.utils.misc import memoized
from trappy.utils import handle_duplicate_index

from IPython.display import display

from lisa.trace import Trace
from lisa.git import Git

class WaResultsCollector(object):
    """
    Collects, analyses and visualises results from multiple WA3 directories

    Takes a list of output directories from Workload Automation 3 and parses
    them. Finds metrics reported by WA itself, and extends those metrics with
    extra detail extracted from ftrace files, energy instrumentation output, and
    workload-specific artifacts that are found in the output.

    Results can be grouped according to the following terms:

    - 'metric' is a specific measurable quantity such as a single frame's
      rendering time or the average energy consumed during a workload run.

    - 'workload' is the general name of a workload such as 'jankbench' or
      'youtube'.

    - 'test' is a more specific identification for workload - for example this
      might identify one of Jankbench's sub-benchmarks, or specifically playing
      a certain video on Youtube for 30s.

      WaResultsCollector ultimately derives 'test' names from the
      'classifiers'::'test' field of the WA3 agenda file's 'workloads' entries.

    - 'tag' is an identifier for a set of run-time target configurations that
      the target was run under. For example there might exist one 'tag'
      identifying running under the schedutil governor and another for the
      performance governor.

      WaResultsCollector ultimately derives 'tag' names from the 'classifiers'
      field of the WA3 agenda file's 'sections' entries.

    - 'kernel' identifies the kernel that was running when the metric was
      collected. This may be a SHA1 or a symbolic ref (branch/tag) derived from
      a provided Git repository. To try to keep identifiers readable, common
      prefixes of refs are removed: if the raw refs are 'test/foo/bar' and
      'test/foo/baz', they will be referred to just as 'bar' and 'baz'.

    Aside from the provided helper attributes, all metrics are exposed in a
    DataFrame as the ``results_df`` attribute.

    :param wa_dirs: List of paths to WA3 output directories or a regexp of WA3
                    output directories names to consider starting from the
                    specified base_path
    :type wa_dirs: str

    :param base_dir: The path of a directory containing a collection of WA3
                     output directories
    :type base_dir: str

    :param platform: Optional LISA platform description. If provided, used to
                     enrich extra metrics gleaned from trace analysis.

    :param kernel_repo_path: Optional path to kernel repository. WA3 reports the
                     SHA1 of the kernel that workloads were run against. If this
                     param is provided, the repository is search for symbolic
                     references to replace SHA1s in data representation. This is
                     purely to make the output more manageable for humans.

    :param parse_traces: This class uses LISA to parse and analyse ftrace files
                         for extra metrics. With multiple/large traces this
                         can take some time. Set this param to False to disable
                         trace parsing.

    :param use_cached_trace_metrics: This class uses LISA to parse and analyse
                     ftrace files for extra metrics. With multiple/large traces
                     this can take some time, so the extracted metrics are
                     cached in the provided output directories. Set this param
                     to False to disable this caching.

    :param display_charts: This class uses IPython.display module to render some
                           charts of workloads' results. But we also want to use
                           this class without rendering any charts when we are
                           only interested in table of figures. Set this param
                           to False if you only want table of results but not
                           display them.
    """
    RE_WLTEST_DIR = re.compile(r"wa\.(?P<sha1>\w+)_(?P<name>.+)")

    def __init__(self, base_dir=None, wa_dirs=".*", platform=None,
                 kernel_repo_path=None, parse_traces=True,
                 use_cached_trace_metrics=True, display_charts=True):

        self._log = logging.getLogger('WaResultsCollector')

        if base_dir:
            base_dir = os.path.expanduser(base_dir)
            if not isinstance(wa_dirs, basestring):
                raise ValueError(
                    'If base_dir is provided, wa_dirs should be a regexp')
            regex = wa_dirs
            wa_dirs = self._list_wa_dirs(base_dir, regex)
            if not wa_dirs:
                raise ValueError("Couldn't find any WA results matching '{}' in {}"
                                 .format(regex, base_dir))
        else:
            if not hasattr(wa_dirs, '__iter__'):
                raise ValueError(
                    'if base_dir is not provided, wa_dirs should be a list of paths')


        wa_dirs = [os.path.expanduser(p) for p in wa_dirs]

        self.platform = platform
        self.parse_traces = parse_traces
        if not self.parse_traces:
            self._log.warning("Trace parsing disabled")
        self.use_cached_trace_metrics = use_cached_trace_metrics
        self.display_charts = display_charts

        df = pd.DataFrame()
        df_list = []
        for wa_dir in wa_dirs:
            self._log.info("Reading wa_dir %s", wa_dir)
            df_list.append(self._read_wa_dir(wa_dir))
        df = df.append(df_list)

        kernel_refs = {}
        if kernel_repo_path:
            for sha1 in df['kernel_sha1'].unique():
                ref = Git.find_shortest_symref(kernel_repo_path, sha1)
                if ref:
                    kernel_refs[sha1] = ref

        common_prefix = os.path.commonprefix(kernel_refs.values())
        for sha1, ref in kernel_refs.iteritems():
            kernel_refs[sha1] = ref[len(common_prefix):]

        df['kernel'] = df['kernel_sha1'].replace(kernel_refs)

        self.results_df = df

    def _list_wa_dirs(self, base_dir, wa_dirs_re):
        dirs = []
        self._log.info("Processing WA3 dirs matching [%s], rooted at %s",
                       wa_dirs_re, base_dir)
        wa_dirs_re = re.compile(wa_dirs_re)

        for subdir in os.listdir(base_dir):
            dir = os.path.join(base_dir, subdir)
            if not os.path.isdir(dir) or not wa_dirs_re.search(subdir):
                continue

            # WA3 results dirs contains a __meta directory at the top level.
            if '__meta' not in os.listdir(dir):
                self._log.warning('Ignoring {}, does not contain __meta directory')
                continue

            dirs.append(dir)

        return dirs

    def _read_wa_dir(self, wa_dir):
        """
        Get a DataFrame of metrics from a single WA3 output directory.

        Includes the extra metrics derived from workload-specific artifacts and
        ftrace files.

        Columns returned:

        kernel_sha1,kernel,id,workload,tag,test,iteration,metric,value,units
        """
        # A WA output directory looks something like:
        #
        # wa_output/
        # |- __meta/
        # |  | - jobs.json
        # |  |   (some other bits)
        # |- results.csv
        # |- pelt-wk1-jankbench-1/
        # |  | - result.json
        # |  |   (other results from iteration 1 of pelt-wk1, which is a
        # |  |    jankbench job)
        # |- pelt-wk1-jankbench-2/
        #      [etc]

        # results.csv contains all the metrics reported by WA for all jobs.
        df = pd.read_csv(os.path.join(wa_dir, 'results.csv'))
        # When using Monsoon, the device is a single channel which reports
        # two metrics. This means that devlib's DerivedEnergymeasurements class
        # cannot see the output. Due to the way that the monsoon.py script
        # works, it looks difficult to change Monsoon over to the Acme way of
        # operating. As a workaround, let's mangle the results here instead.
        unique_metrics = df['metric'].unique()
        if 'device_total_energy' not in unique_metrics:
            # potentially, we need to assemble a device_total_energy from
            # other energy values we can add together.
            if 'output_total_energy' in unique_metrics and 'USB_total_energy' in unique_metrics:
                new_rows = []
                output_df = df[df['metric'] == 'output_total_energy']
                usb_df = df[df['metric'] == 'USB_total_energy']
                # for each 'output_total_energy' metric, we will find
                # the matching 'USB_total_energy' metric and assemble
                # a 'device_total_energy' metric by adding them.
                for row in output_df.iterrows():
                    vals = row[1]
                    _id = vals['id']
                    _workload = vals['workload']
                    _iteration = vals['iteration']
                    _value = vals['value']
                    usb_row = usb_df[(usb_df['workload'] == _workload) & (usb_df['id'] == _id) & (usb_df['iteration'] == _iteration)]
                    new_val = float(_value) + float(usb_row['value'])
                    # instead of creating a new row, just change the name
                    # and value of this one
                    vals['metric'] = 'device_total_energy'
                    vals['value'] = new_val
                    new_rows.append(vals)
                # add all the new rows in one go at the end
                df = df.append(new_rows, ignore_index=True)

        # __meta/jobs.json describes the jobs that were run - we can use this to
        # find extra artifacts (like traces and detailed energy measurement
        # data) from the jobs, which we'll use to add additional metrics that WA
        # didn't report itself.
        with open(os.path.join(wa_dir, '__meta', 'jobs.json')) as f:
            jobs = json.load(f)['jobs']

        subdirs_done = []

        # Keep track of how many times we've seen each job id so we know which
        # iteration to look at (If we use the proper WA3 API this awkwardness
        # isn't necessary).
        next_iteration = defaultdict(lambda: 1)

        # Keep track of which jobs we skipped for each iteration
        skipped_jobs = defaultdict(lambda: [])

        # Dicts mapping job IDs to things determined about the job - this will
        # be used to add extra columns to the DataFrame (that aren't reported
        # directly in WA's results.csv)
        tag_map = {}
        test_map = {}
        job_dir_map = {}
        extra_dfs = []

        for job in jobs:
            workload = job['workload_name']

            job_id = job['id']

            # If there's a 'tag' in the 'classifiers' object, use that to
            # identify the runtime configuration. If not, use a representation
            # of the full key=value pairs.
            classifiers = job['classifiers'] or {}

            if 'test' in classifiers:
                # If the workload spec has a 'test' classifier, use that to
                # identify it.
                test = classifiers.pop('test')
            elif 'test' in job['workload_parameters']:
                # If not, some workloads have a 'test' workload_parameter, try
                # using that
                test = job['workload_parameters']['test']
            elif 'test_ids' in job['workload_parameters']:
                # If not, some workloads have a 'test_ids' workload_parameter, try
                # using that
                test = job['workload_parameters']['test_ids']
            else:
                # Otherwise just use the workload name.
                # This isn't ideal because it means the results from jobs with
                # different workload parameters will be amalgamated.
                test = workload

            rich_tag = ';'.join('{}={}'.format(k, v) for k, v in classifiers.iteritems())
            tag = classifiers.get('tag', rich_tag)

            if job_id in tag_map:
                # Double check I didn't do a stupid
                if tag_map[job_id] != tag:
                    raise RuntimeError('Multiple tags ({}, {}) found for job ID {}'
                                       .format(tag, tag_map[job_id], job_id))
            tag_map[job_id] = tag

            if job_id in test_map:
                # Double check I didn't do a stupid
                if test_map[job_id] != test:
                    raise RuntimeError('Multiple tests ({}, {}) found for job ID {}'
                                       .format(test, test_map[job_id], job_id))
            test_map[job_id] = test

            iteration = next_iteration[job_id]
            next_iteration[job_id] += 1

            job_dir = os.path.join(wa_dir,
                                   '-'.join([job_id, workload, str(iteration)]))

            job_dir_map[job_id] = job_dir

            # Jobs can fail due to target misconfiguration or other problems,
            # without preventing us from collecting the results for the jobs
            # that ran OK.
            my_file = os.path.join(job_dir, 'result.json')
            if not os.path.isfile(my_file):
                skipped_jobs[iteration].append(job_id)
                continue

            with open(my_file) as f:
                job_result = json.load(f)
                if job_result['status'] == 'FAILED':
                    skipped_jobs[iteration].append(job_id)
                    continue

            extra_df = self._get_extra_job_metrics(job_dir, workload)
            if extra_df.empty:
                continue

            extra_df.loc[:, 'workload'] = workload
            extra_df.loc[:, 'iteration'] = iteration
            extra_df.loc[:, 'id'] = job_id
            extra_df.loc[:, 'tag'] = tag
            extra_df.loc[:, 'test'] = test
            # Collect all these DFs to merge them in one go at the end.
            extra_dfs.append(extra_df)

        # Append all extra DFs to the results WA's results DF
        if extra_dfs:
            df = df.append(extra_dfs)

        for iteration, job_ids in skipped_jobs.iteritems():
            self._log.warning("Skipped failed iteration %d for jobs:", iteration)
            self._log.warning("   %s", ', '.join(job_ids))

        df['tag'] = df['id'].replace(tag_map)
        df['test'] = df['id'].replace(test_map)
        # TODO: This is a bit lazy: we're storing the directory that every
        # single metric came from in a DataFrame column. That's redundant really
        # - instead, to get from a row in results_df to a job output directory,
        # we should just store a mapping from kernel identifiers to wa_output
        # directories, then derive at the job dir from that mapping plus the
        # job_id+workload+iteration in the results_df row. This works fine for
        # now, though - that refactoring would probably belong alongside a
        # refactoring to use WA's own API for reading output directories.
        df['_job_dir'] = df['id'].replace(job_dir_map)
        df.loc[:, 'kernel_sha1'] = self._wa_get_kernel_sha1(wa_dir)

        return df

    def _get_trace_metrics(self, trace_path):
        """
        Parse a trace (or used cached results) and extract extra metrics from it

        Returns a DataFrame with columns:

        metric,value,units
        """
        cache_path = os.path.join(os.path.dirname(trace_path), 'lisa_trace_metrics.csv')
        if self.use_cached_trace_metrics and os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        # I wonder if this should go in LISA itself? Probably.

        metrics = []
        events = ['irq_handler_entry', 'cpu_frequency', 'nohz_kick', 'sched_switch',
                  'sched_load_cfs_rq', 'sched_load_avg_task', 'thermal_temperature']
        trace = Trace(trace_path, events, self.platform)

        metrics.append(('cpu_wakeup_count', len(trace.data_frame.cpu_wakeups()), None))

        # Helper to get area under curve of multiple CPU active signals
        def get_cpu_time(trace, cpus):
            df = pd.DataFrame([trace.getCPUActiveSignal(cpu) for cpu in cpus])
            return df.sum(axis=1).sum(axis=0)

        clusters = trace.platform.get('clusters')
        if clusters:
            for cluster in clusters.values():
                name = '-'.join(str(c) for c in cluster)

                df = trace.data_frame.cluster_frequency_residency(cluster)
                if df is None or df.empty:
                    self._log.warning("Can't get cluster freq residency from %s",
                                      trace.data_dir)
                else:
                    df = df.reset_index()
                    avg_freq = (df.frequency * df.time).sum() / df.time.sum()
                    metric = 'avg_freq_cluster_{}'.format(name)
                    metrics.append((metric, avg_freq, 'MHz'))

                df = trace.data_frame.trace_event('cpu_frequency')
                df = df[df.cpu == cluster[0]]
                metrics.append(('freq_transition_count_{}'.format(name), len(df), None))

                active_time = area_under_curve(trace.getClusterActiveSignal(cluster))
                metrics.append(('active_time_cluster_{}'.format(name),
                                active_time, 'seconds'))

                metrics.append(('cpu_time_cluster_{}'.format(name),
                                get_cpu_time(trace, cluster), 'cpu-seconds'))

        metrics.append(('cpu_time_total',
                        get_cpu_time(trace, range(trace.platform['cpus_count'])),
                        'cpu-seconds'))

        event = None
        if trace.hasEvents('sched_load_cfs_rq'):
            event = 'sched_load_cfs_rq'
            row_filter = lambda r: r.path == '/'
            column = 'util'
        elif trace.hasEvents('sched_load_avg_cpu'):
            event = 'sched_load_avg_cpu'
            row_filter = lambda r: True
            column = 'util_avg'
        if event:
            df = trace.data_frame.trace_event(event)
            util_sum = (handle_duplicate_index(df)[row_filter]
                        .pivot(columns='cpu')[column].ffill().sum(axis=1))
            avg_util_sum = area_under_curve(util_sum) / (util_sum.index[-1] - util_sum.index[0])
            metrics.append(('avg_util_sum', avg_util_sum, None))

        if trace.hasEvents('thermal_temperature'):
            df = trace.data_frame.trace_event('thermal_temperature')
            for zone, zone_df in df.groupby('thermal_zone'):
                metrics.append(('tz_{}_start_temp'.format(zone),
                                zone_df.iloc[0]['temp_prev'],
                                'milliCelcius'))

                if len(zone_df == 1): # Avoid division by 0
                    avg_tmp = zone_df['temp'].iloc[0]
                else:
                    avg_tmp = (area_under_curve(zone_df['temp'])
                               / (zone_df.index[-1] - zone_df.index[0]))

                metrics.append(('tz_{}_avg_temp'.format(zone),
                                avg_tmp,
                                'milliCelcius'))

        ret = pd.DataFrame(metrics, columns=['metric', 'value', 'units'])
        ret.to_csv(cache_path, index=False)

        return ret

    def _get_extra_job_metrics(self, job_dir, workload):
        """
        Get extra metrics (not reported directly by WA) from a WA job output dir

        Returns a DataFrame with columns:

        metric,value,units
        """
        # return
        # value,metric,units
        extra_metric_list = []

        artifacts = self._read_artifacts(job_dir)
        if self.parse_traces and 'trace-cmd-bin' in artifacts:
            extra_metric_list.append(
                self._get_trace_metrics(artifacts['trace-cmd-bin']))

        if 'jankbench_results_csv' in artifacts:
            df = pd.read_csv(artifacts['jankbench_results_csv'])
            df = pd.DataFrame({'value': df['total_duration']})
            df.loc[:, 'metric'] = 'frame_total_duration'
            df.loc[:, 'units'] = 'ms'

            extra_metric_list.append(df)
        elif 'jankbench-results' in artifacts:
            con = sqlite3.connect(artifacts['jankbench-results'])
            df = pd.read_sql_query("SELECT _id, name, run_id, iteration, total_duration, jank_frame from ui_results", con)
            df = pd.DataFrame({'value': df['total_duration']})
            df.loc[:, 'metric'] = 'frame_total_duration'
            df.loc[:, 'units'] = 'ms'

            extra_metric_list.append(df)

        # WA's metrics model just exports overall energy metrics, not individual
        # samples. We're going to extend that with individual samples so if you
        # want to you can see how much variation there was in energy usage.
        # So we'll look for the actual CSV files and parse that by hand.
        # The parsing necessary is specific to the energy measurement backend
        # that was used, which WA doesn't currently report directly.
        # TODO: once WA's reporting of this data has been cleaned up a bit I
        # think we can simplify this.
        for artifact_name, path in artifacts.iteritems():
            if os.stat(path).st_size == 0:
                self._log.info(" no data for %s",  path)
                continue

            if artifact_name.startswith('energy_instrument_output'):

                try:
                    df = pd.read_csv(path)
                except pandas.errors.ParserError as e:
                    self._log.info(" no data for %s",  path)
                    continue

                if 'device_power' in df.columns:
                    # Looks like this is from an ACME

                    df = pd.DataFrame({'value': df['device_power']})

                    # Figure out what to call the sample metrics. If the
                    # artifact name has something extra, that will be the
                    # channel (IIO device) name. Use that to differentiate where
                    # the samples came from. If not just call it
                    # 'device_power_sample'.
                    device_name = artifact_name[len('energy_instrument_output') + 1:]
                    name_extra = device_name or 'device'
                    df.loc[:, 'metric'] = '{}_power_sample'.format(name_extra)

                    df.loc[:, 'units'] = 'watts'

                    extra_metric_list.append(df)
                elif 'output_power' in df.columns and 'USB_power' in df.columns:
                    # Looks like this is from a Monsoon
                    # For monsoon the USB and device power are collected
                    # together with the same timestamps, so we can just add them
                    # up.
                    power_samples = df['output_power'] + df['USB_power']
                    df = pd.DataFrame({'value': power_samples})
                    df.loc[:, 'metric'] = 'device_power_sample'
                    df.loc[:, 'units'] = 'watts'

                    extra_metric_list.append(df)
        if len(extra_metric_list) > 0:
            return pd.DataFrame().append(extra_metric_list)
        else:
            return pd.DataFrame()

    @memoized
    def _wa_get_kernel_sha1(self, wa_dir):
        """
        Find the SHA1 of the kernel that a WA3 run was run against
        """
        with open(os.path.join(wa_dir, '__meta', 'target_info.json')) as f:
            target_info = json.load(f)

        # Read the kernel release reported by the target
        sha1 = KernelVersion(target_info['kernel_release']).sha1
        if sha1:
            return sha1

        # Couldn't get the release sha1, default to reading it from the
        # directory name built by test_series
        res_dir = os.path.basename(wa_dir)
        match = re.search(WaResultsCollector.RE_WLTEST_DIR, res_dir)
        if match:
            return match.group("sha1")

        raise RuntimeError("Couldn't find the sha1 of the kernel of the device "
                           "that produced {}".format(wa_dir))

    @memoized
    def _select(self, tag='.*', kernel='.*', test='.*'):
        _df = self.results_df
        _df = _df[_df.tag.str.contains(tag)]
        _df = _df[_df.kernel.str.contains(kernel)]
        _df = _df[_df.test.str.contains(test)]
        return _df

    @property
    def workloads(self):
        return self.results_df['kernel'].unique()

    @property
    def workloads(self):
        return self.results_df['workload'].unique()

    @property
    def tags(self):
        return self.results_df['tag'].unique()

    @memoized
    def tests(self, workload=None):
        df = self.results_df
        if workload:
            df = df[df['workload'] == workload]
        return df['test'].unique()

    def workload_available_metrics(self, workload):
        return (self.results_df
                .groupby('workload').get_group(workload)
                ['metric'].unique())

    @memoized
    def _get_metric_df(self, workload, metric, tag, kernel, test):
        """
        Common helper for getting results to plot for a given metric
        """

        df = self._select(tag, kernel, test)
        if df.empty:
            self._log.warn("No data to plot for (tag: %s, kernel: %s, test: %s)",
                           tag, kernel, test)
            return None

        valid_workloads = df.workload.unique()
        if workload not in valid_workloads:
            self._log.warning("No data for [%s] workload", workload)
            self._log.info("Workloads with data, for the specified filters, are:")
            self._log.info(" %s", ','.join(valid_workloads))
            return None
        df = df[df['workload'] == workload]

        valid_metrics = df.metric.unique()
        if metric not in valid_metrics:
            self._log.warning("No metric [%s] collected for workoad [%s]",
                              metric, workload)
            self._log.info("Metrics with data, for the specied filters, are:")
            self._log.info("   %s", ', '.join(valid_metrics))
            return None
        df = df[df['metric'] == metric]

        units = df['units'].unique()
        if len(units) > 1:
            raise RuntimError('Found different units for workload "{}" metric "{}": {}'
                              .format(workload, metric, units))

        return df


    SortBy = namedtuple('SortBy', ['key', 'params', 'column'])

    def _get_sort_params(self, sort_on):
        """
        Validate a sort criteria and return the parameters required by the
        boxplot and report methods.
        """
        valid_sort = ['count', 'mean', 'std', 'min', 'max']

        # Verify if valid percentile string has been required
        match = re.match('^(?P<quantile>\d{1,3})\%$', sort_on)
        if match:
            quantile = int(match.group('quantile'))
            if quantile < 1 or quantile > 100:
                raise ValueError("Error sorting data: Quantile value out of range [1..100]")
            return self.SortBy('quantile', {'q': quantile/100.}, sort_on)

        # Otherwise, verify if it's a valid Pandas::describe()'s column name
        if sort_on in valid_sort:
            return self.SortBy(sort_on, {}, sort_on)

        raise ValueError(
            "sort_on={} not supported, allowed values are percentile or {}"
            .format(sort_on, valid_sort))

    def boxplot(self, workload, metric,
                tag='.*', kernel='.*', test='.*',
                by=['test', 'tag', 'kernel'],
                sort_on='mean', ascending=False,
                xlim=None):
        """
        Display boxplots of a certain metric

        Creates horizontal boxplots of metrics in the results. Check
        ``workloads`` and ``workload_available_metrics`` to find the available
        workloads and metrics. Check ``tags``, ``tests`` and ``kernels``
        to find the names that results can be filtered against.

        By default, the box with the lowest mean value is plotted at the top of
        the graph, this can be customized with ``sort_on`` and ``ascending``.

        :param workload: Name of workload to display metrics for
        :param metric: Name of metric to display

        :param tag: regular expression to filter tags that should be plotted
        :param kernel: regular expression to filter kernels that should be plotted
        :param tag: regular expression to filter tags that should be plotted

        :param by: List of identifiers to group output as in DataFrame.groupby.

        :param sort_on: Name of the statistic to order data for.
                        Supported values are: count, mean, std, min, max.
                        You may alternatively specify a percentile to sort on,
                        this should be an integer in the range [1..100]
                        formatted as a percentage, e.g. 95% is the 95th
                        percentile.
        :param ascending: When True, boxplots are plotted by increasing values
                          (lowest-valued boxplot at the top of the graph) of the
                          specified `sort_on` statistic.
        """
        if not self.display_charts:
            return

        sp = self._get_sort_params(sort_on)
        df = self._get_metric_df(workload, metric, tag, kernel, test)
        if df is None:
            return
        gb = df.groupby(by)

        # Convert the groupby into a DataFrame with a column for each group
        max_group_size = max(len(group) for group in gb.groups.itervalues())
        _df = pd.DataFrame()
        for group_name, group in gb:
            # Need to pad the group's column so that they all have the same
            # length
            padding_length = max_group_size - len(group)
            padding = pd.Series(np.nan, index=np.arange(padding_length))
            col = group['value'].append(padding)
            col.index = np.arange(max_group_size)
            _df[group_name] = col

        # Sort the columns
        # With default params this puts the box with the lowest mean at the
        # bottom.
        # NOTE: the not(ascending) condition is required to keep these plots
        # aligned with the way describe() reports the stats corresponding to
        # each boxplot
        sorted_df = getattr(_df, sp.key)(**sp.params)
        sorted_df = sorted_df.sort_values(ascending=not(ascending))
        _df = _df[sorted_df.index]

        # Plot boxes sorted by mean
        fig, axes = plt.subplots(figsize=(16,8))
        _df.boxplot(ax=axes, vert=False, showmeans=True)
        fig.suptitle('')
        if xlim:
            axes.set_xlim(xlim)
        [units] = df['units'].unique()
        axes.set_xlabel('{} [{}]'.format(metric, units))
        axes.set_title('{}:{}'.format(workload, metric))
        plt.show()

        return axes

    def describe(self, workload, metric,
                 tag='.*', kernel='.*', test='.*',
                 by=['test', 'tag', 'kernel'],
                 sort_on='mean', ascending=False):
        """
        Return a DataFrame of statistics for a certain metric

        Compute mean, std, min, max and [50, 75, 95, 99] percentiles for
        the values collected on each iteration of the specified metric.

        Check ``workloads`` and ``workload_available_metrics`` to find the
        available workloads and metrics.
        Check ``tags``, ``tests`` and ``kernels`` to find the names that
        results can be filtered against.

        :param workload: Name of workload to display metrics for
        :param metric: Name of metric to display

        :param tag: regular expression to filter tags that should be plotted
        :param kernel: regular expression to filter kernels that should be plotted
        :param tag: regular expression to filter tags that should be plotted

        :param by: List of identifiers to group output as in DataFrame.groupby.

        :param sort_on: Name of the statistic to order data for.
                        Supported values are: count, mean, std, min, max.
                        It's also supported at the usage of a percentile value,
                        which has to be an integer in the range [1..100] and
                        formatted as a percentage,
                        e.g. 95% is the 95th percentile.
        :param ascending: When True, the statistics are reported by increasing values
                          of the specified `sort_on` column
        """
        sp = self._get_sort_params(sort_on)
        df = self._get_metric_df(workload, metric, tag, kernel, test)
        if df is None:
            return

        # Add the eventually required additional percentile
        percentiles = [0.75, 0.95, 0.99]
        if sp.params and 'q' in sp.params:
            percentiles.append(sp.params['q'])
            percentiles = sorted(list(set(percentiles)))

        grouped = df.groupby(by)['value']
        stats_df = pd.DataFrame(
            grouped.describe(percentiles=percentiles))

        # Use a consistent formatting independently from the PANDAs version
        if 'value' in stats_df.columns:
            # We must be running on a pre-0.20.0 version of pandas.
            # unstack will convert the old output format to the new.
            #    http://pandas.pydata.org/pandas-docs/version/0.20/whatsnew.html#groupby-describe-formatting
            # Main difference is that here we have a top-level column
            # named 'value'
            stats_df = stats_df.unstack()
        else:
            # Let's add a top-level column named 'value' which will be replaced
            # by the actual metric name by the following code
            stats_df.columns = pd.MultiIndex.from_product(
                [['value'], stats_df.columns])

        # Sort entries by the required metric and order value
        stats_df.sort_values(by=[('value', sp.column)],
                             ascending=ascending, inplace=True)
        stats_df.rename(columns={'value': metric}, inplace=True)

        return stats_df

    def report(self, workload, metric,
               tag='.*', kernel='.*', test='.*',
               by=['test', 'tag', 'kernel'],
               sort_on='mean', ascending=False,
               xlim=None):
        """
        Report a boxplot and a set of statistics for a certain metric

        This is a convenience method to call both ``boxplot`` and ``describe``
        at the same time to get a consistent graphical and numerical
        representation of the values for the specified metric.

        Check ``workloads`` and ``workload_available_metrics`` to find the
        available workloads and metrics.
        Check ``tags``, ``tests`` and ``kernels`` to find the names that
        results can be filtered against.

        :param workload: Name of workload to display metrics for
        :param metric: Name of metric to display

        :param tag: regular expression to filter tags that should be plotted
        :param kernel: regular expression to filter kernels that should be plotted
        :param tag: regular expression to filter tags that should be plotted

        :param by: List of identifiers to group output as in DataFrame.groupby.
        """
        axes = self.boxplot(workload, metric, tag, kernel, test,
                            by, sort_on, ascending, xlim)
        stats_df = self.describe(workload, metric, tag, kernel, test,
                                 by, sort_on, ascending)
        if self.display_charts:
            display(stats_df)

        return (axes, stats_df)


    CDF = namedtuple('CDF', ['df', 'threshold', 'above', 'below'])

    def _get_cdf(self, data, threshold):
        """
        Build the "Cumulative Distribution Function" (CDF) for the given data
        """
        # Build the series of sorted values
        ser = data.sort_values()
        if len(ser) < 1000:
            # Append again the last (and largest) value.
            # This step is important especially for small sample sizes
            # in order to get an unbiased CDF
            ser = ser.append(pd.Series(ser.iloc[-1]))
        df = pd.Series(np.linspace(0., 1., len(ser)), index=ser)

        # Compute percentage of samples above/below the specified threshold
        below = float(max(df[:threshold]))
        above = 1 - below
        return self.CDF(df, threshold, above, below)

    def plot_cdf(self, workload='jankbench', metric='frame_total_duration',
                 threshold=16, tag='.*', kernel='.*', test='.*'):
        """
        Display cumulative distribution functions of a certain metric

        Draws CDFs of metrics in the results. Check ``workloads`` and
        ``workload_available_metrics`` to find the available workloads and
        metrics. Check ``tags``, ``tests`` and ``kernels`` to find the
        names that results can be filtered against.

        The most likely use-case for this is plotting frame rendering times
        under Jankbench, so default parameters are provided to make this easy.

        :param workload: Name of workload to display metrics for
        :param metric: Name of metric to display

        :param threshold: Value to highlight in the plot - the likely use for
                          this is highlighting the maximum acceptable
                          frame-rendering time in order to see at a glance the
                          rough proportion of frames that were rendered in time.

        :param tag: regular expression to filter tags that should be plotted
        :param kernel: regular expression to filter kernels that should be plotted
        :param tag: regular expression to filter tags that should be plotted

        :param by: List of identifiers to group output as in DataFrame.groupby.
        """

        if not self.display_charts:
            return

        df = self._get_metric_df(workload, metric, tag, kernel, test)
        if df is None:
            return

        test_cnt = len(df.groupby(['test', 'tag', 'kernel']))
        colors = iter(cm.rainbow(np.linspace(0, 1, test_cnt+1)))

        fig, axes = plt.subplots()
        axes.axvspan(0, threshold, facecolor='g', alpha=0.1);

        labels = []
        lines = []
        for keys, df in df.groupby(['test', 'tag', 'kernel']):
            labels.append("{:16s}: {:32s}".format(keys[2], keys[1]))
            color = next(colors)
            cdf = self._get_cdf(df['value'], threshold)
            [units] = df['units'].unique()
            ax = cdf.df.plot(ax=axes, legend=False, xlim=(0,None), figsize=(16, 6),
                             title='Total duration CDF ({:.1f}% within {} [{}] threshold)'\
                             .format(100. * cdf.below, threshold, units),
                             label=test,
                             color=to_hex(color))
            lines.append(ax.lines[-1])
            axes.axhline(y=cdf.below, linewidth=1,
                         linestyle='--', color=to_hex(color))
            self._log.debug("%-32s: %-32s: %.1f", keys[2], keys[1], 100.*cdf.below)

        axes.grid(True)
        axes.legend(lines, labels)
        plt.show()

    def find_comparisons(self, base_id=None, by='kernel'):
        """
        Find metrics that changed between a baseline and variants

        The notion of 'variant' and 'baseline' is defined by the `by` param. If
        by='kernel', then `base_id` should be a kernel SHA (or whatever key the
        'kernel' column in the results_df uses). If by='tag' then `base_id`
        should be a WA 'tag id' (as named in the WA agenda).
        """
        comparisons = []

        # I dunno why I wrote this with a namedtuple instead of just a dict or
        # whatever, but it works fine
        Comparison = namedtuple('Comparison', ['metric', 'test', 'inv_id',
                                               'base_id', 'base_mean', 'base_std',
                                               'new_id', 'new_mean', 'new_std',
                                               'diff', 'diff_pct', 'pvalue'])

        # If comparing by kernel, only check comparisons where the 'tag' is the same
        # If comparing by tag, only check where kernel is same
        if by == 'kernel':
            invariant = 'tag'
        elif by == 'tag':
            invariant = 'kernel'
        else:
            raise ValueError('`by` must be "kernel" or "tag"')

        available_baselines = self.results_df[by].unique()
        if base_id is None:
            base_id = available_baselines[0]
        if base_id not in available_baselines:
            raise ValueError('base_id "{}" not a valid "{}" (available: {}). '
                            'Did you mean to set by="{}"?'.format(
                                base_id, by, available_baselines, invariant))

        for metric, metric_results in self.results_df.groupby('metric'):
            # inv_id will either be the id of the kernel or of the tag,
            # depending on the `by` param.
            # So wl_inv_results will be the results entries for that workload on
            # that kernel/tag
            for (test, inv_id), wl_inv_results in metric_results.groupby(['test', invariant]):
                gb = wl_inv_results.groupby(by)['value']

                if base_id not in gb.groups:
                    self._log.warning('Skipping - No baseline results for test '
                                      '[%s] %s [%s] metric [%s]',
                                      test, invariant, inv_id, metric)
                    continue

                base_results = gb.get_group(base_id)
                base_mean = base_results.mean()

                for group_id, group_results in gb:
                    if group_id == base_id:
                        continue

                    # group_id is now a kernel id or a tag (depending on
                    # `by`). group_results is a slice of all the rows of self.results_df
                    # for a given metric, test, tag/test tuple. We
                    # create comparison object to show how that metric changed
                    # wrt. to the base tag/test.

                    group_mean = group_results.mean()
                    mean_diff = group_mean - base_mean
                    # Calculate percentage difference in mean metric value
                    if base_mean != 0:
                        mean_diff_pct = mean_diff * 100. / base_mean
                    else:
                        # base mean is 0, can't divide by that.
                        if group_mean == 0:
                            # Both are 0 so diff_pct is 0
                            mean_diff_pct =0
                        else:
                            # Tricky one - base value was 0, new value isn't.
                            # Let's just call it a 100% difference.
                            mean_diff_pct = 100

                    if len(group_results) <= 1 or len(base_results) <= 1:
                        # Can't do ttest_ind if we only have one sample. There
                        # are proper t-tests for this, but let's just assume the
                        # worst.
                        pvalue = 1.0
                    elif mean_diff == 0:
                        # ttest_ind also gives a warning if the two data sets
                        # are the same and have no variance. I don't know why
                        # that is to be honest, but anyway if there's no
                        # difference in the mean, we don't care about the
                        # p-value.
                        pvalue = 1.0
                    else:
                        # Find a p-value which hopefully represents the
                        # (complement of the) certainty that any difference in
                        # the mean represents something real.
                        _, pvalue = ttest_ind(group_results, base_results, equal_var=False)

                    comparisons.append(Comparison(
                        metric, test, inv_id,
                        base_id, base_mean, base_results.std(),
                        group_id, group_mean, group_results.std(),
                        mean_diff, mean_diff_pct, pvalue))

        return pd.DataFrame(comparisons)

    def plot_comparisons(self, base_id=None, by='kernel'):
        """
        Visualise metrics that changed between a baseline and variants

        The notion of 'variant' and 'baseline' is defined by the `by` param. If
        by='kernel', then `base_id` should be a kernel SHA (or whatever key the
        'kernel' column in the results_df uses). If by='tag' then `base_id`
        should be a WA 'tag id' (as named in the WA agenda).
        """
        if not self.display_charts:
            return

        df = self.find_comparisons(base_id=base_id, by=by)

        if df.empty:
            self._log.error('No comparisons by %s found', by)
            if len(self.results_df[by].unique()) == 1:
                self._log.warning('There is only one %s in the results', by)
            return

        # Separate plot for each test (e.g. one plot for Jankbench list_view)
        for (test, inv_id), test_comparisons in df.groupby(('test', 'inv_id')):
            # Vertical size of plot depends on how many metrics we're comparing
            # and how many things (kernels/tags) we're comparing metrics for.
            # a.k.a the total length of the comparisons df.
            fig, ax = plt.subplots(figsize=(15, len(test_comparisons) / 2.))

            # pos is used as the Y-axis. The y-axis is a discrete axis with a
            # point for each of the metrics we're comparing. matplotlib needs
            # that in numerical form.
            # We also have one more tick on the Y-axis than we actually need -
            # this is a terrible hack which is necessary because when we set the
            # opacity of the first bar, it sets the opacity of the legend. So we
            # introduce a dummy bar with a value of 0 and an opacity of 1.
            all_metrics = test_comparisons['metric'].unique()
            pos = np.arange(-1, len(all_metrics))

            # At each point on the discrete y-axis we'll have one bar for each
            # comparison: one per kernel/tag (depending on the `by` param), minus
            # one for the baseline.
            # If there are more bars we'll need to make them thinner so they
            # fit. The sum of the bars' thicknesses should be 60% of a tick on
            # the 'y-axis'.
            thickness= 0.6 / len(test_comparisons.groupby('new_id'))

            # TODO: something is up with the calculations above, because there's
            # always a bit of empty space at the bottom of the axes.


            gb = test_comparisons.groupby('new_id')
            colors = cm.rainbow(np.linspace(0, 1, len(gb)))
            for i, (group, gdf) in enumerate(gb):
                def get_dummy_row(metric):
                    return pd.DataFrame({col: 0 for col in gdf.columns}, index=[metric])

                missing_metrics = set(all_metrics) - set(gdf['metric'].unique())
                gdf = gdf.set_index('metric')
                for missing_metric in missing_metrics:
                    self._log.warning(
                        "Data missing, can't compare metric [{}] for {} [{}]"
                        .format(missing_metric, by, group))
                    gdf = gdf.append(get_dummy_row(missing_metric))

                # Ensure the comparisons are in the same order for each group
                gdf = gdf.reindex(all_metrics)

                # Append the dummy row we're using to fix the legend opacity
                gdf = get_dummy_row('').append(gdf)

                # For each of the things we're comparing we'll plot a bar chart
                # but slightly shifted. That's how we get multiple bars on each
                # y-axis point.
                bars = ax.barh(bottom=pos + (i * thickness),
                               width=gdf['diff_pct'],
                               height=thickness, label=group,
                               color=colors[i % len(colors)], align='center')
                # Decrease the opacity for comparisons with a high p-value
                for bar, pvalue in zip(bars, gdf['pvalue']):
                    bar.set_alpha(1 - (min(pvalue * 10, 0.95)))

            # Add some text for labels, title and axes ticks
            ax.set_xlabel('Percent difference')
            [baseline] = test_comparisons['base_id'].unique()
            ax.set_title('{} ({}): Percent difference compared to {} \nopacity depicts p-value'
                         .format(test, inv_id, baseline))
            ax.set_yticklabels(gdf.index.tolist())
            ax.set_yticks(pos + thickness / 2)
            # ax.set_xlim((-50, 50))
            ax.legend(loc='best')

            ax.grid(True)

        plt.show()

    def _read_artifacts(self, job_dir):
        with open(os.path.join(job_dir, 'result.json')) as f:
            ret = {a['name']: os.path.join(job_dir, a['path'])
                   for a in json.load(f)['artifacts']}
        return ret

    def _find_job_dir(self, workload='.*', tag='.*', kernel='.*', test='.*',
                      iteration=1):
        df = self._select(tag, kernel, test)
        df = df[df['workload'].str.match(workload)]

        job_dirs = df['_job_dir'].unique()

        if len(job_dirs) > 1:
            raise ValueError("Params for get_artifacts don't uniquely identify a job. "
                "for workload='{}' tag='{}' kernel='{}' test='{}' iteration={}, "
                "found:\n{}" .format(
                    workload, tag, kernel, test, iteration, '\n'.join(job_dirs)))
        if not job_dirs:
            raise ValueError(
                "No job found for "
                "workload='{}' tag='{}' kernel='{}' test='{}' iteration={}"
                .format(workload, tag, kernel, test, iteration))

        [job_dir] = job_dirs
        return job_dir

    def get_artifacts(self, workload='.*', tag='.*', kernel='.*', test='.*',
                      iteration=1):
        """
        Get a dict mapping artifact names to file paths for a specific job.

        artifact_name specifies the name of an artifact, e.g. 'trace_bin' to
        find the ftrace file from the specific job run. The other parameters
        should be used to uniquely identify a run of a job.
        """
        job_dir = self._find_job_dir(workload, tag, kernel, test, iteration)
        return self._read_artifacts(job_dir)

    def get_artifact(self, artifact_name, workload='.*',
                     tag='.*', kernel='.*', test='.*',
                     iteration=1):
        """
        Get the path of an artifact attached to a job output.

        artifact_name specifies the name of an artifact, e.g. 'trace_bin' to
        find the ftrace file from the specific job run. The other parameters
        should be used to uniquely identify a run of a job.
        """
        job_dir = self._find_job_dir(workload, tag, kernel, test, iteration)
        artifacts = self._read_artifacts(job_dir)

        if not artifact_name in artifacts:
            raise ValueError("No '{}' artifact found in {} (have {})".format(
                artifact_name, job_dir, artifacts.keys()))

        return artifacts[artifact_name]
